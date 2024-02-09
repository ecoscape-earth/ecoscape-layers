import csv
import fiona
import numpy as np
import os
import requests
from ebird.api import get_taxonomy
from pyproj import Transformer
from rasterio import features
from rasterio.windows import Window, from_bounds, bounds
from scgt import GeoTiff
from shapely.geometry import shape

from .constants import REFINE_METHODS


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

        :param species: 6-letter eBird code for a bird species.
        :return: the scientific name of the bird species.
        """
        res = get_taxonomy(self.ebird_key, species=species)
        return res[0]["sciName"] if len(res) > 0 else None

    def get_habitats(self, name, region=None):
        """
        Gets habitat assessments for suitability for a given species.
        This also adds the associated landcover map's code and resistance value to the API response, which are useful for creating resistance mappings and/or habitat layers.

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
    For things like reprojecting, building resistance tables, and creating habitat layers and landcover matrix layers.
    This class maintains a common CRS, resolution, and resampling method for this purpose.
    """

    def __init__(self, redlist_key, ebird_key, landcover_path, crs=None, resolution=None, resampling="near",
                 bounds=None, padding=0, out_landcover_path=None):
        """
        Initializes a LayerGenerator object.

        :param redlist_key: IUCN Red List API key.
        :param ebird_key: eBird API key.
        :param landcover_path: file path to the initial landcover raster.
        :param crs: desired common CRS of the layers as an ESRI WKT string.
        :param resolution: desired resolution of the layers in the units of the CRS as an integer.
        :param resampling: resampling method if resampling is necessary to produce layers with the desired CRS and/or resolution.
        :param bounds: bounds to crop generated layers to in the units of the chosen CRS, specified as a bounding box (xmin, ymin, xmax, ymax).
        :param padding: padding in units of chosen CRS to add around the bounds.
        :param out_landcover_path: file path to save new landcover to if a new one is produced by cropping/reprojection/rescaling. If not given, one can be generated based on the parameters applied.
        """
        self.landcover_path = os.path.abspath(landcover_path)
        self.redlist = RedList(redlist_key, ebird_key)
        self.ebird_key = ebird_key
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
        
        self.out_landcover_path = os.path.abspath(out_landcover_path) if out_landcover_path else None

    def process_landcover(self):
        """
        Processes the landcover layer based on the parameters specified at initialization.
        If no reprojection or cropping is needed, it may not be necessary to generate a new layer file.
        """
        self.orig_landcover_path = self.landcover_path

        # reproject/crop landcover if needed
        self.reproject_landcover()
        self.crop_landcover()
        
        if self.orig_landcover_path != self.landcover_path:
            # if a specific output landcover path was specified, rename the processed file to that
            if self.out_landcover_path:
                os.replace(self.landcover_path, self.out_landcover_path)
                if os.path.isfile(self.landcover_path + ".aux.xml"):
                    os.replace(self.landcover_path + ".aux.xml", self.out_landcover_path + ".aux.xml")
                self.landcover_path = self.out_landcover_path
            print("New landcover in", self.landcover_path)
        else:
            print("Landcover already meets the desired parameters, no reprojection or cropping needed")

    def reproject_landcover(self):
        """
        Reprojects the landcover to the CRS and resolution desired if needed, creating a new file in doing so.
        landcover_path is reassigned to the path of this new file.
        The CRS and resolution are taken from the current class instance's settings if specified.
        If reprojection occurs, the resampling method used is taken from the current class instance.
        """
        with GeoTiff.from_file(self.landcover_path) as ter:
            if self.crs is None:
                self.crs = ter.dataset.crs
            if self.resolution is None:
                self.resolution = int(ter.dataset.transform[0])
            
            # reproject landcover if resolution and/or CRS attributes differ from current resolution and CRS
            if self.resolution != int(ter.dataset.transform[0]) or self.crs != ter.dataset.crs:
                reproj_landcover_path = self.landcover_path[:-4] + "_" + str(self.resolution) + "_" + self.resampling + ".tif"
                ter.reproject_from_crs(reproj_landcover_path, self.crs, (self.resolution, self.resolution), self.rio_resampling)

                self.landcover_path = reproj_landcover_path

    def crop_landcover(self):
        """
        Crops the landcover to the desired bounding rectangle with optional padding.
        This does not modify the existing file, but creates a new one that landcover_path is assigned to.
        """
        if self.bounds is None:
            return
        if not isinstance(self.bounds, tuple):
            raise TypeError("Bounds should be given as a tuple of 4 coordinates")
        if len(self.bounds) != 4:
            raise ValueError("Invalid bounding box, bounds should have 4 coordinates")

        with GeoTiff.from_file(self.landcover_path) as file:
            # check that file is not already cropped to the bounds (rounded by window lengths/offsets)
            padded_bounds = (self.bounds[0] - self.padding, self.bounds[1] - self.padding, self.bounds[2] + self.padding, self.bounds[3] + self.padding)
            cropped_window = from_bounds(*padded_bounds, transform=file.dataset.transform).round_lengths().round_offsets(pixel_precision=0)
            crop_bounds = bounds(cropped_window, file.dataset.transform)

            if crop_bounds != file.dataset.bounds:
                # perform the cropping operation if needed
                cropped_landcover_path = self.landcover_path[:-4] + "_cropped.tif"
                cropped_file = file.crop_to_new_file(cropped_landcover_path, bounds=self.bounds, padding=self.padding)
                cropped_file.dataset.close()
                self.landcover_path = cropped_landcover_path

    def get_map_codes(self):
        """
        Obtains the list of unique landcover map codes present in the landcover map.
        This is used to determine the map codes for which resistance values need to be defined.
        """
        with GeoTiff.from_file(self.landcover_path) as landcover:
            tile = landcover.get_all_as_tile()
            map_codes = sorted(list(np.unique(tile.m)))
        return map_codes

    def get_range_from_ebird(self, species_code, output_path):
        """
        Gets range map in geopackage (.gpkg) format for a given bird species.

        :param species_code: 6-letter eBird code for a bird species.
        :param output_path: path to write the range map to.
        """
        req_url = f"https://st-download.ebird.org/v1/fetch?objKey=2022/{species_code}/ranges/{species_code}_range_smooth_9km_2022.gpkg&key={self.ebird_key}"
        res = requests.get(req_url)
        with open(output_path, "wb") as res_file:
            res_file.write(res.content)

    def generate_resistance_table(self, habitats, output_path):
        """
        Generates the resistance dictionary for a given species as a CSV file using habitat preference data from the IUCN Red List.
        - Major importance terrain is assigned a resistance of 0.
        - Suitable (but not major importance) terrain is assigned a resistance of 0.1.
        - All other terrain is assigned a resistance of 1.

        :param habitats: IUCN Red List habitat data for the species for which the table should be generated.
        :param output_path: path of CSV file to which the species' resistance table should be saved.
        """
        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(habitats[0].keys())
            # map codes from the landcover map
            for map_code in self.get_map_codes():
                h = next((hab for hab in habitats if hab["map_code"] == map_code), None)
                if h is not None:
                    writer.writerow(h.values())
                else:
                    writer.writerow([''] * 5 + [map_code] + [1])

    def get_good_terrain(self, habitats, refine_method="forest_add308"):
        """
        Determine the terrain deemed suitable for habitat based on the refining method.
        This decides what map codes from the landcover should be used to filter the habitat.

        :param habitats: IUCN Red List habitat data for the species for which suitable terrain is computed.
        :param refine_method: method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option.
        :return: list of map codes filtered by refine_method.
        """

        if refine_method == "forest":
            return [x for x in range(100, 110)]
        elif refine_method == "forest_add308":
            return [x for x in range(100, 110)] + [308]
        elif refine_method == "allsuitable":
            return [hab["map_code"] for hab in habitats if hab["suitability"] == "Suitable"]
        elif refine_method == "majoronly":
            return [hab["map_code"] for hab in habitats if hab["majorimportance"] == "Yes"]
    
    def generate_habitat(self, species_code, habitat_fn=None, resistance_dict_fn=None, range_fn=None,
                        refine_method="forest", refine_list=None):
        """
        Runner function for full process of habitat and matrix layer generation for one bird species.

        :param species_code: 6-letter eBird code of the bird speciess to generate layers for.
        :param habitat_fn: name of output habitat layer.
        :param resistance_dict_fn: name of output resistance dictionary CSV.
        :param range_fn: name of output range map for the species, which is downloaded as an intermediate step for producing the habitat layer.
        :param refine_method: method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option.
        :param refine_list: list of map codes for which the corresponding pixels should be considered habitat. Alternative to refine_method, which offers limited options. If both refine_method and refine_list are given, refine_list is prioritized.
        """

        if refine_list:
            refine_method = None
        elif refine_method not in REFINE_METHODS:
            refine_method = "forest"

        # If file names not specified, build default ones.
        if habitat_fn is None:
            habitat_fn = os.path.join(os.getcwd(), species_code, "habitat.tif")
        if resistance_dict_fn is None:
            habitat_fn = os.path.join(os.getcwd(), species_code, "resistance.csv")
        if range_fn is None:
            habitat_fn = os.path.join(os.getcwd(), species_code, "range_map.gpkg")

        # Ensure that directories to habitat layer, range map, and resistance dictionary exist.
        make_dirs_for_file(habitat_fn)
        make_dirs_for_file(resistance_dict_fn)
        make_dirs_for_file(range_fn)
        
        # Obtain species habitat information from the IUCN Red List.
        # Manual corrections made here for differences between eBird and IUCN Red List scientific names.
        if species_code == "whhwoo":
            sci_name = "Leuconotopicus albolarvatus"
        elif species_code == "yebmag":
            sci_name = "Pica nutalli"
        elif species_code == "pilwoo":
            sci_name = "Hylatomus pileatus"
        elif species_code == "recwoo":
            sci_name = "Leuconotopicus borealis"
        else:
            sci_name = self.redlist.get_scientific_name(species_code)
        
        habs = self.redlist.get_habitats(sci_name)
        if refine_method == "forest_add308" and len([hab for hab in habs if hab["code"] == "3.8"]) == 0:
            habs.append({
                "code": "3.8",
                "habitat": "Shrubland - Mediterranean-type shrubby vegetation",
                "suitability": "Suitable",
                "season": "Resident",
                "majorimportance": "Yes",
                "map_code": 308,
                "resistance": 0
            })

        if len(habs) == 0:
            print("Habitat preferences for", species_code, "could not be found on the IUCN Red List (perhaps due to a name mismatch with eBird?). Habitat layer and resistance dictionary were not generated.")
            return

        # Create the resistance table for each species.
        self.generate_resistance_table(habs, resistance_dict_fn)

        # Download species range as geopackage from eBird.
        self.get_range_from_ebird(species_code, range_fn)
        if not os.path.isfile(range_fn):
            print("A range map could not be downloaded for", species_code, "from eBird. Habitat layer was not generated.")
            return

        # Perform intersection between the range and habitable landcover.
        with GeoTiff.from_file(self.landcover_path) as landcover:
            range_shapes = reproject_shapefile(range_fn, self.crs, "range")
            shapes_for_mask = [shape(range_shapes[0]["geometry"])]
            good_terrain_for_hab = refine_list if refine_list is not None else self.get_good_terrain(habs, refine_method)

            with landcover.clone_shape(habitat_fn) as output:
                # Prevent 0 from being interpreted as a nodata value
                output.dataset.nodata = -1

                reader = output.get_reader(b=0, w=10000, h=10000)
                for tile in reader:
                    # get window and fit to the tiff's bounds if necessary
                    tile.fit_to_bounds(width=output.width, height=output.height)
                    window = Window(tile.x, tile.y, tile.w, tile.h)

                    # mask out pixels from landcover not within range of shapes
                    window_data = landcover.dataset.read(window=window, masked=True)
                    shape_mask = features.geometry_mask(shapes_for_mask, out_shape=(tile.h, tile.w), transform=landcover.dataset.window_transform(window))
                    window_data.mask = window_data.mask | shape_mask
                    window_data = window_data.filled(0)

                    # get pixels where terrain is good
                    window_data = np.isin(window_data, good_terrain_for_hab)

                    output.dataset.write(window_data, window=window)

                # remove old attribute table if it exists so that values can be updated
                if os.path.isfile(habitat_fn + ".aux.xml"):
                    os.remove(habitat_fn + ".aux.xml")
        
        print("Habitat layer successfully generated for", species_code)

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

def make_dirs_for_file(file_name):
    """
    Creates intermediate directories in the file path for a file if they don't exist yet.
    The file itself is not created; this just ensures that the directory of the file and all preceding ones
    exist first.

    :param file_name: file to make directories for.
    """
    dirs, _ = os.path.split(file_name)
    os.makedirs(dirs, exist_ok=True)
