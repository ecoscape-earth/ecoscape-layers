import subprocess
import os
import requests
import numpy as np
import csv
import fiona
from rasterio import features
from pyproj import Transformer
from ebird.api import get_taxonomy
from shapely.geometry import shape
from birdmaps.scgt import GeoTiff, Window
from birdmaps.config import EBIRD_KEY, REDLIST_KEY

"""
Module of functions that involve interfacing with the Red List API
"""
class RedList():

    def __init__(self):
        self.params = { "token": REDLIST_KEY }

    def get_from_redlist(self, url):
        """
        Convenience function for sending GET request to Red List API with the key
        """
        res = requests.get(url, params=self.params).json()
        return res["result"]

    def get_scientific_name(self, species):
        """
        Translates eBird codes to scientific names for use in Red List
        """
        return get_taxonomy(EBIRD_KEY, species=species)[0]["sciName"]

    def get_habitats(self, name, region=None):
        """
        Gets habitat assessments for suitability for a given species.
        Also adds the associated terrain map's code and resistance value, which are useful.

        str name: scientific name of the species
        str region: specific region to assess habitats in (https://apiv3.iucnredlist.org/api/v3/docs#regions)
        """
        url = "https://apiv3.iucnredlist.org/api/v3/habitats/species/name/{0}".format(name)
        if region is not None:
            url += "/region/{1}".format(region)

        habs = self.get_from_redlist(url)

        for hab in habs:
            code = hab["code"]
            sep = code.index(".")
            # only take up to level 2 (xx.xx), therefore truncating codes with more than 1 period separator
            if code.count(".") > 1:
                code = code[:code.index(".", sep+1)]
            hab["map_code"] = int(code[:sep] + code[sep+1:].zfill(2))
            hab["resistance"] = 0 if hab["majorimportance"] == "Yes" else 0.1

        return habs


"""
For things like reprojecting, building resistance tables, processing the terrain.
Keeps track of a common CRS, resolution, and resampling method for this purpose.
"""
class HabitatGenerator(object):

    def __init__(self, terrain_path, terrain_codes_path, crs, resolution=None, resampling="near"):
        self.terrain_path = terrain_path
        self.terrain_codes_path = terrain_codes_path
        self.crs = crs
        self.resolution = resolution
        self.resampling = resampling
        # rio_resampling accounts for rasterio's different resampling parameter names from gdal
        if self.resampling == "near":
            self.rio_resampling = "nearest"
        elif self.resampling == "cubicspline":
            self.rio_resampling = "cubic_spline"
        else:
            self.rio_resampling = self.resampling

    def reproject_terrain(self):
        """
        Reprojects terrain to the CRS and resolution desired using the chosen resampling method.
        The CRS, resolution, and resampling method are taken from the current class instance's settings.
        """
        with GeoTiff.from_file(self.terrain_path) as ter:
            # reproject terrain if resolution is set and not equal to current terrain resolution
            # or if the CRS is not the same
            if (self.resolution is not None and self.resolution != ter.dataset.transform[0]) or ter.dataset.crs != self.crs:
                reproj_terrain_path = self.append_settings_to_name(self.terrain_path)
                ter.reproject_from_crs(reproj_terrain_path, self.crs, (self.resolution, self.resolution), self.rio_resampling)
                self.terrain_path = reproj_terrain_path
            else:
                self.resolution = int(ter.dataset.transform[0])

    def crop_terrain(self, bounds, padding=0):
        """
        Crop terrain to a certain bounding rectangle with optional padding.
        This does not modify the existing file, but creates a new one.
        """
        with GeoTiff.from_file(self.terrain_path) as file:
            cropped_terrain_path = self.terrain_path[:-4] + "_cropped.tif"
            cropped_file = file.crop_to_new_file(cropped_terrain_path, bounds=bounds, padding=padding)
            cropped_file.dataset.close()
            self.terrain_path = cropped_terrain_path

    def get_map_codes(self):
        """
        Obtains the list of unique terrain map codes from terrain_codes_path.
        Used to determine the map codes for which resistance values need to be defined.
        """
        all_map_codes = []
        with open(self.terrain_codes_path, newline="") as ter_codes:
            for row in csv.reader(ter_codes):
                all_map_codes.append(int(row[0]))
        return all_map_codes

    def write_map_codes(self):
        """
        Finds the unique map code values from the terrain tiff and writes them to a CSV,
        to make creating resistance tables easier later.
        """
        all_map_codes = set()

        # find all map codes in the terrain
        with GeoTiff.from_file(self.terrain_path) as ter:
            reader = ter.get_reader(b=0, w=10000, h=10000)
            for tile in reader:
                tile.fit_to_bounds(width=ter.width, height=ter.height)
                window = Window(tile.x, tile.y, tile.w, tile.h)
                all_map_codes.update(np.unique(ter.dataset.read(window=window)))

        # write map codes to a csv file in a single column
        with open(self.terrain_codes_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            for map_code in all_map_codes:
                writer.writerow([map_code])

    def get_ranges_from_ebird(self, species_list_path, species_range_folder):
        """
        Downloads range map shapefiles from eBird by using the ebirdst R package.
        This utilizes the R script "ebird_range_download.R" to download the ranges.
        """
        result = subprocess.run(["Rscript", "./ebird_range_download.R", species_list_path, species_range_folder], capture_output=True, text=True)
        if result.returncode != 0:
            print(result)
            raise AssertionError("Problem occurred while downloading ranges")

    def generate_resistance_table(self, species, output_path, map_codes):
        """
        Generates the terrain-to-resistance table for a given species as a CSV file.
        - major importance terrain is assigned 0 resistance
        - suitable (but not major importance) terrain is assigned 0.1 resistance
        - all other terrains are assigned 1 resistance

        str output_path: path to CSV file to which resistance table should be saved
        list map_codes: list of map codes to assign resistances to, obtainable from get_map_codes().
        """
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(species["habitats"][0].keys())
            # map codes from the terrain map
            for map_code in map_codes:
                h = next((hab for hab in species["habitats"] if hab["map_code"] == map_code), None)
                if h is not None:
                    writer.writerow(h.values())
                else:
                    writer.writerow([''] * 5 + [map_code] + [1])

    def refine_habitat(self, ter, species, shapes, output_path, refine_method="majoronly"):
        """
        Creates the habitat for a given species based on terrain and range.

        ter: opened terrain Geotiff
        list shapes: list of shapes to use for range. They should be given in the same projection as the terrain.
        str refine_method: "forest", "suitable", or "major" to decide what terrain should be considered for habitat
        """

        shapes = [shape(shapes["geometry"])]

        with ter.clone_shape(output_path) as output:
            reader = output.get_reader(b=0, w=10000, h=10000)
            good_terrain_for_hab = self.get_good_terrain(species, refine_method)

            for tile in reader:
                # get window and fit to the tiff's bounds if necessary
                tile.fit_to_bounds(width=output.width, height=output.height)
                window = Window(tile.x, tile.y, tile.w, tile.h)

                # mask out pixels from terrain not within range of shapes
                window_data = ter.dataset.read(window=window, masked=True)
                shape_mask = features.geometry_mask(shapes, out_shape=(tile.h, tile.w), transform=ter.dataset.window_transform(window))
                window_data.mask = window_data.mask | shape_mask
                window_data = window_data.filled(0)

                # get pixels where terrain is good
                window_data = np.isin(window_data, good_terrain_for_hab)

                output.dataset.write(window_data, window=window)

            # remove old attribute table if it exists so that values can be updated
            if os.path.isfile(output_path + ".aux.xml"):
                os.remove(output_path + ".aux.xml")

    def get_good_terrain(self, species, refine_method="forest"):
        """
        Determine the terrain deemed suitable for habitat based on the refining method.
        This decides what terrain map codes should be used to filter the habitat.
        """
        if refine_method == "forest":
            return [x for x in range(100, 110)]
        elif refine_method == "forest_add308":
            return [x for x in range(100, 110)] + [308]
        elif refine_method == "allsuitable":
            return [hab["map_code"] for hab in species["habitats"] if hab["suitability"] == "Suitable"]
        elif refine_method == "majoronly":
            return [hab["map_code"] for hab in species["habitats"] if hab["majorimportance"] == "Yes"]

    def append_settings_to_name(self, file_path):
        """
        Adds the resolution and resampling settings to the file path name.
        If old file name is "filename.extension", new name is "filename_[resolution]_[resampling].extension".
        """
        sep = file_path.index(".")
        return file_path[:sep] + "_" + str(self.resolution) + "_" + self.resampling + file_path[sep:]

def reproject_shapefile(shapes_path, dest_crs, shapes_layer=None, file_path=None):
    """
    This takes a specified shapefile or geopackage and reprojects it to a different CRS.
    Returns a list of reprojected features.

    str shapes_path: file path to the shapefile or geopackage to reproject
    str dest_crs: CRS to reproject to, as an ESRI WKT string
    str shapes_layer: if file is a geopackage, use this to specify which layer should be reprojected
    str file_path: if specified, write the reprojected result to this file path as a shapefile
    """
    myfeatures = []

    with fiona.open(shapes_path, 'r', layer=shapes_layer) as shp:
        # create a Transformer for changing from the current CRS to the destination CRS
        transformer = Transformer.from_crs(crs_from=shp.crs_wkt, crs_to=dest_crs, always_xy=True)

        # loop through polygons in each features, transforming all point coordinates within those polygons
        for feature in shp:
            for i, polygon in enumerate(feature['geometry']['coordinates']):
                for j, ring in enumerate(polygon):
                    if isinstance(ring, list):
                        feature['geometry']['coordinates'][i][j] = [transformer.transform(*point) for point in ring]
                    else:
                        # "ring" is really just a single point
                        feature['geometry']['coordinates'][i][j] = [transformer.transform(*ring)]
            myfeatures.append(feature)

        # if file_path is specified, write the result to a new shapefile
        if file_path is not None:
            meta = shp.meta
            meta.update({
                'driver': 'ESRI Shapefile',
                'crs_wkt': dest_crs
            })
            with fiona.open(file_path, 'w', **meta) as output:
                output.writerecords(myfeatures)

    return myfeatures