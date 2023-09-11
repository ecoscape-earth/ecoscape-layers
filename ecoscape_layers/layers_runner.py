import importlib.util
import os
import sys
from scgt import GeoTiff
from layers import RedList, LayerGenerator, reproject_shapefile
from constants import EBIRD_INDIV_RANGE_PATH, EBIRD_INDIV_RANGE_LAYER


def generate_layers(config_path, species_list_path, terrain_path, terrain_codes_path, species_range_folder,
                    output_folder, crs=None, resolution=None, resampling="near", bounds=None, padding=0,
                    refine_method="forest"):
    """
    Runner function for habitat generation.

    :param config_path: file path to Python config file containing IUCN Red List and eBird API keys.
    :param species_list_path: file path to text file of the bird species for which habitat layers should be generated, formatted as 6-letter eBird species codes on individual lines.
    :param terrain_path: file path to initial terrain raster.
    :param terrain_codes_path: file path to a CSV containing terrain resistance codes. If not generated, it can be generated by setting reproject_inputs to True.
    :param species_range_folder: folder path for where downloaded eBird range maps should be saved.
    :param output_folder: folder path to place habitat layer output files and terrain-to-resistance CSV files into.
    :param crs: desired common CRS of the layers as an ESRI WKT string.
    :param resolution: desired resolution in the units of the chosen CRS, or None to use the resolution of the input terrain raster.
    :param resampling: resampling method if resampling is necessary to produce layers with the desired CRS and/or resolution; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid arguments.
    :param bounds: bounds to crop generated layers to in the units of the chosen CRS, specified as a bounding box (xmin, ymin, xmax, ymax).
    :param padding: padding in units of chosen CRS to add around the bounds.
    :param refine_method: method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option.
    """

    # Get API keys from config.py file.
    config_spec = importlib.util.spec_from_file_location("config", config_path)
    config = importlib.util.module_from_spec(config_spec)
    sys.modules["module.name"] = config
    config_spec.loader.exec_module(config)

    REDLIST_KEY = config.REDLIST_KEY
    EBIRD_KEY = config.EBIRD_KEY

    # Define eBird-specific range map path and gpkg layer.
    indiv_range_path = os.path.join(species_range_folder, EBIRD_INDIV_RANGE_PATH)
    indiv_range_layer = EBIRD_INDIV_RANGE_LAYER

    # Get the list of bird species from species_list_path.
    with open(species_list_path) as file:
        species_list = file.read().splitlines()

    # Generate output folder.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Generate species output folders.
    for species in species_list:
        species_output_folder = os.path.join(output_folder, species)
        if not os.path.exists(species_output_folder):
            os.makedirs(species_output_folder)
    
    redlist = RedList(REDLIST_KEY, EBIRD_KEY)
    layer_generator = LayerGenerator(terrain_path, terrain_codes_path, crs, resolution, resampling,
                                        bounds, padding)

    # Generate terrain layer.
    print("Generating terrain layer...")
    layer_generator.generate_terrain()
    
    # Obtain species habitat information from the IUCN Red List.
    print("Gathering species habitat preferences...")
    species_data = []

    for species in species_list:
        sci_name = redlist.get_scientific_name(species)
        habs = redlist.get_habitats(sci_name)

        # Manual corrections for differences between eBird and IUCN Red List scientific names.
        if species == "whhwoo":
            sci_name = "Leuconotopicus albolarvatus"
            habs = redlist.get_habitats(sci_name)
        if species == "yebmag":
            sci_name = "Pica nutalli"
            habs = redlist.get_habitats(sci_name)

        if len(habs) == 0:
            print("Skipping", species, "due to not finding info on IUCN Red List (perhaps a name mismatch with eBird)?")
            continue
        else:
            species_data.append({
                "name": species,
                "sci_name": sci_name,
                "habitats": habs
            })

    # Download species ranges as shapefiles from eBird.
    print("Downloading species range maps...")
    layer_generator.get_ranges_from_ebird(species_list_path, species_range_folder)

    # Create the resistance table for each species.
    print("Creating resistance CSVs...")
    all_map_codes = layer_generator.get_map_codes()
    for species in species_data:
        code = species["name"]
        resistance_output_path = os.path.join(output_folder, code, f"{code}_resistance.csv")
        layer_generator.generate_resistance_table(species["habitats"], all_map_codes, resistance_output_path)

    # Perform the intersection between the range and habitable terrain.
    print("Generating habitat layers...")
    with GeoTiff.from_file(layer_generator.terrain_path) as ter:
        resolution = int(ter.dataset.transform[0])
        
        for species in species_data:
            if species == "":
                break

            code = species["name"]
            habitats = species["habitats"]

            if not os.path.isfile(indiv_range_path.format(code=code)):
                print("Skipping {code}, no associated range map found".format(code=code))
                continue

            range_shapes = reproject_shapefile(
                shapes_path=indiv_range_path.format(code=code),
                dest_crs=crs,
                shapes_layer=indiv_range_layer
            )

            if len(range_shapes) == 1:
                # not a seasonal bird
                path = os.path.join(output_folder, code, f"habitat_2020_{resolution}_{resampling}_{refine_method}.tif")
                layer_generator.refine_habitat(ter, habitats=habitats, shapes=range_shapes[0], output_path=path, refine_method=refine_method)
            else:
                # seasonal bird, different output for each shape
                for s in range_shapes:
                    season = str(s["properties"]["season"])
                    path = os.path.join(output_folder, code, f"{season}_habitat_2020_{resolution}_{resampling}_{refine_method}.tif")
                    layer_generator.refine_habitat(ter, habitats=habitats, shapes=s, output_path=path, refine_method=refine_method)
    
    print("Layers successfully generated in " + output_folder)