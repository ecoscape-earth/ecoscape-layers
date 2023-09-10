import argparse
import os
from constants import RESAMPLING_METHODS, REFINE_METHODS
from layers_runner import generate_layers

def main(args):
    output_folder = os.getcwd()

    print(f"Generating layers with parameters:\n\t \
            config {args.config}\n\t \
            species_list {args.species_list}\n\t \
            terrain {args.terrain}\n\t \
            terrain_codes {args.terrain_codes}\n\t \
            species_range_folder {args.species_range_folder}\n\t \
            output_folder {output_folder}\n\t \
            crs {args.crs}\n\t \
            resolution {args.resolution}\n\t \
            resampling {args.resampling}\n\t \
            bounds {args.bounds}\n\t \
            padding {args.padding}\n\t \
            refine_method {args.refine_method}\n\t \
            force_new_terrain_codes {args.force_new_terrain_codes}\n\t \
                ")

    # validate inputs
    assert os.path.isfile(args.config), f"config {args.config} is not a valid file"

    assert os.path.isfile(args.species_list), f"species_list {args.species_list} is not a valid file"
    assert os.path.isfile(args.terrain), f"terrain {args.terrain} is not a valid file"
    assert os.path.isfile(args.terrain_codes) or os.access(os.path.dirname(args.terrain_codes), os.W_OK), \
        f"output_folder {args.terrain_codes} is not a valid directory"
    assert os.path.isdir(args.species_range_folder), \
        f"species_range_folder {args.species_range_folder} is not a valid directory"
    assert os.path.isdir(args.output_folder), f"output_folder {args.output_folder} is not a valid directory"

    assert args.resolution == None or isinstance(args.resolution, int), "invalid resolution"
    assert args.resampling in RESAMPLING_METHODS, \
        f"{args.resampling} is not a valid resampling value. See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options"
    assert isinstance(args.bounds, tuple) and len(args.bounds) == 4 and \
          all(isinstance(coord, (int, float)) and not isinstance(coord, bool) for coord in args.bounds), \
          "invalid bounds"
    assert isinstance(args.padding, int), "invalid padding"
    
    assert args.refine_method in REFINE_METHODS, \
        f"{args.resampling} is not a valid refine method. Value must be in {REFINE_METHODS}"
    assert isinstance(args.force_new_terrain_codes, bool), f"force_new_terrain_codes is not a boolean value"

    generate_layers(args.species_list, args.terrain, args.terrain_codes, args.species_range_folder,
                      args.output_folder, args.crs, args.resolution, args.resampling, args.bounds,
                      args.padding, args.refine_method, args.force_new_terrain_codes)

    print("ran generate_layers")


def cli():
    parser = argparse.ArgumentParser()
    
    current_dir = os.getcwd()
    default_terrain_codes = os.path.join(current_dir, "terrain_codes.csv")
    default_species_range_folder = os.path.join(current_dir, "ebird_ranges.csv")

    parser.add_argument('-k', '--config', type=os.path.abspath, default=None,
                        help='Path to config.py file containing IUCN Red List and eBird API keys')
    
    parser.add_argument('-s', '--species_list', type=os.path.abspath, default=None,
                        help='Path to txt file of the bird species for which habitat layers should be generated, formatted as 6-letter eBird species codes on individual lines')
    parser.add_argument('-t', '--terrain', type=os.path.abspath, default=None,
                        help='Path to terrain raster')
    parser.add_argument('-c', '--terrain_codes', type=os.path.abspath, default=default_terrain_codes,
                        help='Path to a CSV containing terrain map codes. If it does not yet exist, a CSV based on the final terrain matrix layer will be created at this path')
    parser.add_argument('-r', '--species_range_folder', type=os.path.abspath, default=default_species_range_folder,
                        help='Path to folder to which downloaded eBird range maps should be saved')
    parser.add_argument('-o', '--output_folder', type=os.path.abspath, default=current_dir,
                        help='Path to output folder')
    
    parser.add_argument('-C', '--crs', type=str, default=None,
                        help='Desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input terrain raster')
    parser.add_argument('-R', '--resolution', type=int, default=None,
                        help='Desired resolution in the units of the chosen CRS, or None to use the resolution of the input terrain raster')
    parser.add_argument('-e', '--resampling', type=str, default="near",
                        help='Resampling method to use if reprojection of the input terrain layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options')
    parser.add_argument('-b', '--bounds', type=tuple, default=None,
                        help='Tuple representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS')
    parser.add_argument('-p', '--padding', type=int, default=0,
                        help='Padding to add around the bounds in the units of the chosen CRS')
    
    parser.add_argument('-m', '--refine_method', type=str, default="forest",
                        help='Method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option')
    parser.add_argument('-f', '--force_new_terrain_codes', type=bool, default=False,
                        help='If set to True, forcefully generates a new CSV of the terrain map codes, potentially overwriting any previously existing CSV')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()
