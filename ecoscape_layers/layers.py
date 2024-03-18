import csv
import numpy as np
import os
import requests
from rasterio import features
from rasterio.windows import Window, from_bounds, bounds
from scgt import GeoTiff
from shapely import unary_union
from shapely.geometry import shape
from osgeo import gdal, ogr

from .constants import HAB_308, REFINE_METHODS
from .redlist import RedList
from .utils import reproject_shapefile, make_dirs_for_file


class LayerGenerator(object):
    """
    For things like reprojecting, building resistance tables, and creating habitat layers and landcover matrix layers.
    This class maintains a common CRS, resolution, and resampling method for this purpose.
    """

    def __init__(self, redlist_key, ebird_key, landcover_path, iucn_range_src=None):
        """
        Initializes a LayerGenerator object.

        :param redlist_key: IUCN Red List API key.
        :param ebird_key: eBird API key.
        :param landcover_path: file path to the initial landcover raster.
        :param iucn_range_src: file path to the IUCN range source if wanted.
        """
        self.landcover_path = os.path.abspath(landcover_path)
        self.redlist = RedList(redlist_key, ebird_key)
        self.ebird_key = ebird_key
        self.iucn_range_src = iucn_range_src

    def get_map_codes(self):
        """
        Obtains the list of unique landcover map codes present in the landcover map.
        This is used to determine the map codes for which resistance values need to be defined.
        """
        with GeoTiff.from_file(self.landcover_path) as landcover:
            tile = landcover.get_all_as_tile()
            map_codes = sorted(list(np.unique(tile.m)))
        return map_codes
    
    def get_range_from_iucn(self, species_name, input_ranges_gdb, output_path):
        '''
        Using IUCN gdb file, creates shapefiles usable for refining ranges for specific species with GDAL's ogr module.

        :param species_name: scientific name of species to obtain range for.
        :param input_ranges_gdb: path to input file (/BOTW.gdb)
        :param output_path: path for output .shp file (if exists already, the old file(s) will be deleted)
        '''
        # We want to set this option to avoid spending too much time organizing polygons.
        # See https://gdal.org/api/ogrgeometry_cpp.html#_CPPv4N18OGRGeometryFactory16organizePolygonsEPP11OGRGeometryiPiPPKc
        # and https://gdal.org/user/configoptions.html#general-options.
        gdal.SetConfigOption('OGR_ORGANIZE_POLYGONS', 'CCW_INNER_JUST_AFTER_CW_OUTER')

        # Open input file and layer, and apply attribute filter using scientific name
        input_src = ogr.Open(input_ranges_gdb, 0)
        input_layer = input_src.GetLayer()
        input_layer_defn = input_layer.GetLayerDefn()
        input_layer.SetAttributeFilter("sci_name = '" + species_name + "'")
        input_spatial_ref = input_layer.GetSpatialRef()
        input_spatial_ref.MorphToESRI()

        # Define output driver, delete old output file(s) if they exist
        output_driver = ogr.GetDriverByName('ESRI Shapefile')
        if os.path.exists(output_path):
            output_driver.DeleteDataSource(output_path)
        
        # Create the output shapefile
        output_src = output_driver.CreateDataSource(output_path)
        output_layer_name = os.path.splitext(os.path.split(output_path)[1])[0]
        output_layer = output_src.CreateLayer(output_layer_name, geom_type=ogr.wkbMultiPolygon)

        # Add fields to output
        for i in range(0, input_layer_defn.GetFieldCount()):
            output_layer.CreateField(input_layer_defn.GetFieldDefn(i))

        # Add filtered features to output
        for inFeature in input_layer:
            output_layer.CreateFeature(inFeature)

        # Create .prj file by taking the projection of the input file
        output_prj = open(os.path.splitext(output_path)[0] + '.prj', 'w')
        output_prj.write(input_spatial_ref.ExportToWkt())
        output_prj.close()

        # Save and close files
        input_src = None
        output_src = None

        # Reset the GDAL config option
        gdal.SetConfigOption('OGR_ORGANIZE_POLYGONS', 'DEFAULT')

    def get_range_from_ebird(self, species_code, output_path):
        """
        Gets range map in geopackage (.gpkg) format for a given bird species.

        :param species_code: 6-letter eBird code for a bird species.
        :param output_path: path to write the range map to.
        """
        req_url = f"https://st-download.ebird.org/v1/fetch?objKey=2022/{species_code}/ranges/{species_code}_range_smooth_9km_2022.gpkg&key={self.ebird_key}"
        res = requests.get(req_url)
        if res.status_code == 200:
            with open(output_path, "wb") as res_file:
                res_file.write(res.content)

    def generate_resistance_table(self, habitats, output_path, refine_method):
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
                    if refine_method == "forest" or refine_method == "forest_add308":
                        h['resistance'] = 0 if map_code >= 100 and map_code < 200 else h['resistance']
                    writer.writerow(h.values())
                else:
                    if refine_method == "forest" or refine_method == "forest_add308":
                        writer.writerow([''] * 5 + [map_code] + [0 if map_code >= 100 and map_code < 200 else 1])
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
    
    def generate_habitat(self, species_code, habitat_fn=None, resistance_dict_fn=None, elevation_fn=None,
                         range_fn=None, range_src="ebird", refine_method="forest", refine_list=None):
        """
        Runner function for full process of habitat and matrix layer generation for one bird species.

        :param species_code: 6-letter eBird code of the bird speciess to generate layers for.
        :param habitat_fn: name of output habitat layer.
        :param resistance_dict_fn: name of output resistance dictionary CSV.
        :param elevation_fn: name of input elevation raster for filtering habitat by elevation.
        :param range_fn: name of output range map for the species, which is downloaded as an intermediate step for producing the habitat layer.
        :param range_src: source from which to obtain range maps; "ebird" or "iucn".
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
            resistance_dict_fn = os.path.join(os.getcwd(), species_code, "resistance.csv")
        if range_fn is None:
            range_fn = os.path.join(os.getcwd(), species_code, "range_map.gpkg")

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
        
        habs = self.redlist.get_habitat_data(sci_name)

        if refine_method == "forest_add308" and len([hab for hab in habs if hab["code"] == "3.8"]) == 0:
            habs.append(HAB_308)

        if len(habs) == 0:
            raise AssertionError("Habitat preferences for " + str(species_code) + " could not be found on the IUCN Red List (perhaps due to a name mismatch with eBird?). Habitat layer and resistance dictionary were not generated.")

        # Create the resistance table for each species.
        self.generate_resistance_table(habs, resistance_dict_fn, refine_method)

        # Obtain species range as either shapefile from IUCN or geopackage from eBird.
        if range_src == "iucn":
            if self.iucn_range_src is None:
                raise ValueError("No IUCN range source was specified. Habitat layer was not generated.")
            self.get_range_from_iucn(sci_name, self.iucn_range_src, range_fn)
        else:
            self.get_range_from_ebird(species_code, range_fn)

        if not os.path.isfile(range_fn):
            raise FileNotFoundError("Range map could not be found for " + str(species_code) + " from " + ("IUCN" if range_src == "iucn" else "eBird") + ". Habitat layer was not generated.")

        # Perform intersection between the range and habitable landcover.
        with GeoTiff.from_file(self.landcover_path) as landcover:
            _, ext = os.path.splitext(range_fn)
            range_shapes = reproject_shapefile(range_fn, landcover.dataset.crs, "range" if ext == ".gpkg" else None)

            # Prepare range defined as shapes for masking
            if range_src == "iucn":
                # correction for polygon coordinates format
                for s in range_shapes:
                    if s['geometry']['type'] == 'Polygon':
                        s['geometry']['coordinates'] = [[el[0] for el in s['geometry']['coordinates'][0]]]
                shapes_for_mask = [unary_union([shape(s['geometry']) for s in range_shapes])]
            else:
                shapes_for_mask = [shape(range_shapes[0]["geometry"])]

            # Define map codes for which corresponding pixels should be considered habitat
            good_terrain_for_hab = refine_list if refine_list is not None else self.get_good_terrain(habs, refine_method)

            # Create the habitat layer
            with landcover.clone_shape(habitat_fn) as output:
                # If elevation raster is provided, obtain min/max elevation and read elevation raster
                if elevation_fn is not None:
                    min_elev, max_elev = self.redlist.get_elevation(sci_name)
                    elev = GeoTiff.from_file(elevation_fn)
                    cropped_window = from_bounds(*output.dataset.bounds, transform=elev.dataset.transform).round_lengths().round_offsets(pixel_precision=0)
                    x_offset, y_offset = cropped_window.col_off, cropped_window.row_off
                
                output.dataset.nodata = None
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

                    # mask out pixels not within elevation range (if elevation raster is provided)
                    if elevation_fn is not None:
                        elev_window = Window(tile.x + x_offset, tile.y + y_offset, tile.w, tile.h)
                        elev_window_data = elev.dataset.read(window=elev_window)
                        window_data = window_data & (elev_window_data >= min_elev) & (elev_window_data <= max_elev)

                    # write the window result
                    output.dataset.write(window_data, window=window)

                # remove old attribute table if it exists so that values can be updated
                if os.path.isfile(habitat_fn + ".aux.xml"):
                    os.remove(habitat_fn + ".aux.xml")
            
                if elevation_fn is not None:
                    elev.dataset.close()
        
        print("Habitat layer successfully generated for", species_code)

