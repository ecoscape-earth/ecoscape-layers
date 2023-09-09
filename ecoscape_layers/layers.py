import subprocess
import os
import requests
import numpy as np
import csv
import fiona
from rasterio import features
from rasterio.windows import Window
from pyproj import Transformer
from ebird.api import get_taxonomy
from shapely.geometry import shape
from scgt import GeoTiff

class RedList():
    """
    A module of functions that involve interfacing with the IUCN Red List API.
    """

    def __init__(self, redlist_key, ebird_key):
        """
        Initializes a RedList object.
        API keys are required to access the IUCN Red List API and eBird API respectively; see the documentation for more information.
        """
        self.redlist_params = { "token": redlist_key }
        self.ebird_key = ebird_key

    def get_from_redlist(self, url):
        """
        Convenience function for sending GET request to Red List API with the key.

        :param url: the URL for the request.
        :return: response for the request.
        """
        res = requests.get(url, params=self.redlist_params).json()
        return res["result"]

    def get_scientific_name(self, species):
        """
        Translates eBird codes to scientific names for use in Red List.

        :param species: a 6-letter eBird code for a bird species.
        :return: the scientific name of the bird species.
        """
        return get_taxonomy(self.ebird_key, species=species)[0]["sciName"]

    def get_habitats(self, name, region=None):
        """
        Gets habitat assessments for suitability for a given species.
        This also adds the associated terrain map's code and resistance value to the API response, which are useful for creating resistance mappings and/or habitat layers.

        :param name: scientific name of the species.
        :param region: a specific region to assess habitats in (see https://apiv3.iucnredlist.org/api/v3/docs#regions).
        :return: a list of habitats identified by the IUCN Red List as suitable for the species.
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


class LayerGenerator(object):
    """
    For things like reprojecting, building resistance tables, and creating habitat layers and matrix (terrain) layers.
    This class maintains a common CRS, resolution, and resampling method for this purpose.
    """

    def __init__(self, terrain_path, terrain_codes_path, crs=None, resolution=None, resampling="near", bounds=None, padding=0, force_new_terrain_codes=False):
        """
        Initializes a HabitatGenerator object.

        :param terrain_path: file path to the initial terrain raster.
        :param terrain_codes_path: file path to a CSV containing terrain resistance codes. If not generated, it may be generated from the terrain using the generate_resistance_table method.
        :param crs: desired common CRS of the layers as an ESRI WKT string.
        :param resolution: desired resolution of the layers in the units of the CRS as an integer.
        :param resampling: resampling method if resampling is necessary to produce layers with the desired CRS and/or resolution.
        :param force_new_terrain_codes: if set to True, forcefully generates a new CSV of the terrain map codes. This will overwrite any existing CSV.
        """
        self.terrain_path = terrain_path
        self.terrain_codes_path = terrain_codes_path
        self.crs = crs
        self.resolution = resolution
        self.resampling = resampling
        self.bounds = bounds
        self.padding = padding

        # rio_resampling accounts for rasterio's different resampling parameter names from gdal
        if self.resampling == "near":
            self.rio_resampling = "nearest"
        elif self.resampling == "cubicspline":
            self.rio_resampling = "cubic_spline"
        else:
            self.rio_resampling = self.resampling
        
        # reproject/crop terrain if needed
        self.reproject_terrain()
        if bounds is not None:
            self.crop_terrain(bounds, padding)

        # generate map codes CSV if terrain has been modified or force_new_terrain_codes is set to True
        if self.terrain_path != terrain_path or force_new_terrain_codes:
            self.write_map_codes()

    def reproject_terrain(self):
        """
        Reprojects the terrain to the CRS and resolution desired if needed, creating a new file in doing so.
        terrain_path is reassigned to the path of this new file.
        The CRS and resolution are taken from the current class instance's settings if specified.
        If reprojection occurs, the resampling method used is taken from the current class instance.
        """
        with GeoTiff.from_file(self.terrain_path) as ter:
            if self.crs is None:
                self.crs = ter.dataset.crs
            if self.resolution is None:
                self.resolution = int(ter.dataset.transform[0])
            
            # reproject terrain if resolution and/or CRS attributes differ from current resolution and CRS
            if self.resolution != int(ter.dataset.transform[0]) or self.crs != ter.dataset.crs:
                reproj_terrain_path = self.append_settings_to_name(self.terrain_path)
                ter.reproject_from_crs(reproj_terrain_path, self.crs, (self.resolution, self.resolution), self.rio_resampling)
                self.terrain_path = reproj_terrain_path

    def crop_terrain(self, bounds, padding=0):
        """
        Crops the terrain to a certain bounding rectangle with optional padding.
        This does not modify the existing file, but creates a new one that terrain_path is assigned to.

        :param bounds: bounds to crop the terrain layer to, specified as a bounding box (xmin, ymin, xmax, ymax).
        :param padding: padding to add around the bounds in the units of the current CRS.
        """
        if not isinstance(bounds, tuple):
            raise TypeError("Bounds should be given as a tuple")
        if len(bounds) != 4:
            raise ValueError("Invalid bounding box")

        with GeoTiff.from_file(self.terrain_path) as file:
            cropped_terrain_path = self.terrain_path[:-4] + "_cropped.tif"
            cropped_file = file.crop_to_new_file(cropped_terrain_path, bounds=bounds, padding=padding)
            cropped_file.dataset.close()
            self.terrain_path = cropped_terrain_path

    def get_map_codes(self):
        """
        Obtains the list of unique terrain map codes from terrain_codes_path, if the file path exists.
        This is used to determine the map codes for which resistance values need to be defined.
        """
        if not os.path.isfile(self.terrain_codes_path):
            raise FileNotFoundError("Terrain codes file not found, generate with write_map_codes")

        all_map_codes = []
        with open(self.terrain_codes_path, newline="") as ter_codes:
            for row in csv.reader(ter_codes):
                all_map_codes.append(int(row[0]))
        return all_map_codes

    def write_map_codes(self):
        """
        Finds the unique map code values from the terrain tiff and writes them to a CSV located at terrain_codes_path, for creating resistance tables later.
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
        An API key for eBird is required; see the documentation for more information.

        :param species_list_path: list of species to download range map shapefiles for, given as 6-letter eBird codes.
        :param species_range_folder: folder to which the downloaded range maps should be saved.
        """
        if not os.path.exists(species_range_folder):
            os.makedirs(species_range_folder)

        downloader_path = os.path.join(os.path.dirname(__file__), "ebird_range_download.R")
        result = subprocess.run(["Rscript", downloader_path, species_list_path, species_range_folder], capture_output=True, text=True)

        if result.returncode != 0:
            print(result)
            raise AssertionError("Problem occurred while downloading ranges")

    def generate_resistance_table(self, habitats, map_codes, output_path):
        """
        Generates the terrain-to-resistance table for a given species as a CSV file using habitat preference data from the IUCN Red List.
        - Major importance terrain is assigned a resistance of 0.
        - Suitable (but not major importance) terrain is assigned a resistance of 0.1.
        - All other terrains are assigned a resistance of 1.

        :param habitats: IUCN Red List habitat data for the species for which the table should be generated.
        :param map_codes: list of map codes to assign resistances to.
        :param output_path: path of CSV file to which the species' resistance table should be saved.
        """
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(habitats[0].keys())
            # map codes from the terrain map
            for map_code in map_codes:
                h = next((hab for hab in habitats if hab["map_code"] == map_code), None)
                if h is not None:
                    writer.writerow(h.values())
                else:
                    writer.writerow([''] * 5 + [map_code] + [1])

    def refine_habitat(self, ter, habitats, shapes, output_path, refine_method="forest_add308"):
        """
        Creates the habitat layer for a given species based on the terrain and range map.

        :param ter: open instance of the terrain GeoTiff.
        :param habitats: IUCN Red List habitat data for the species for which the habitat layer should be generated.
        :param shapes: list of shapes to use as the range map, given in the same projection as the terrain.
        :param output_path: file path to save the habitat layer to.
        :param refine_method: used to decide what terrain should be considered for habitat.
        """

        shapes = [shape(shapes["geometry"])]

        with ter.clone_shape(output_path) as output:
            reader = output.get_reader(b=0, w=10000, h=10000)
            good_terrain_for_hab = self.get_good_terrain(habitats, refine_method)

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

    def get_good_terrain(self, habitats, refine_method="forest_add308"):
        """
        Determine the terrain deemed suitable for habitat based on the refining method.
        This decides what terrain map codes should be used to filter the habitat.

        :param habitats: IUCN Red List habitat data for the species for which suitable terrain is computed.
        :param refine_method: method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option.
        :return: list of terrain map codes filtered by refine_method.
        """

        if refine_method == "forest":
            return [x for x in range(100, 110)]
        elif refine_method == "forest_add308":
            return [x for x in range(100, 110)] + [308]
        elif refine_method == "allsuitable":
            return [hab["map_code"] for hab in habitats if hab["suitability"] == "Suitable"]
        elif refine_method == "majoronly":
            return [hab["map_code"] for hab in habitats if hab["majorimportance"] == "Yes"]

    def append_settings_to_name(self, file_path):
        """
        Adds the resolution and resampling settings to the file path name.
        If the old file name is "filename.ext", the new name is "filename_[resolution]_[resampling].ext".

        :param file_path: file path to modify.
        :return: modified version of the file path.
        """

        sep = file_path.index(".")
        return file_path[:sep] + "_" + str(self.resolution) + "_" + self.resampling + file_path[sep:]

def reproject_shapefile(shapes_path, dest_crs, shapes_layer=None, file_path=None):
    """
    Takes a specified shapefile or geopackage and reprojects it to a different CRS.

    :param shapes_path: file path to the shapefile or geopackage to reproject.
    :param dest_crs: CRS to reproject to as an ESRI WKT string.
    :param shapes_layer: if file is a geopackage, the name of the layer that should be reprojected.
    :param file_path: if specified, the file path to write the reprojected result to as a shapefile.
    :return: list of reprojected features.
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