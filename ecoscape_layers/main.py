import argparse
import os
from .layers import LayerGenerator
from .constants import RESAMPLING_METHODS, REFINE_METHODS


def main(args):
    # validate some inputs
    assert os.path.isfile(args.landcover_path), f"landcover {args.landcover_path} is not a valid file"

    assert args.resolution == None or isinstance(args.resolution, int), "invalid resolution"
    assert args.resampling in RESAMPLING_METHODS, \
        f"{args.resampling} is not a valid resampling value. See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options"
    assert len(args.bounds) == 4, "invalid bounds"
    assert isinstance(args.padding, int), "invalid padding"
    
    assert args.refine_method in REFINE_METHODS, \
        f"{args.resampling} is not a valid refine method. Value must be in {REFINE_METHODS}"

    print(len((args.redlist_key, args.ebird_key, args.landcover_path, args.crs.replace("'", '"'), args.resolution, args.resampling, tuple(args.bounds), args.padding)))

    layer_generator = LayerGenerator(args.redlist_key, args.ebird_key, args.landcover_path, args.crs.replace("'", '"'), args.resolution, args.resampling, tuple(args.bounds), args.padding)
    layer_generator.process_landcover()
    layer_generator.generate_habitat(args.species_code, args.habitat_fn, args.resistance_dict_fn, args.range_fn, args.refine_method)


def cli():
    parser = argparse.ArgumentParser(add_help=False)
    
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    required.add_argument('-k', '--redlist_key', type=str, default=None, required=True,
                        help='IUCN Red List API key')
    required.add_argument('-K', '--ebird_key', type=str, default=None, required=True,
                        help='eBird API key')
    required.add_argument('-s', '--species_code', type=str, default=None, required=True,
                        help='Bird species for which habitat layer and landcover matrix layer should be generated as a 6-letter eBird code')
    required.add_argument('-l', '--landcover_path', type=os.path.abspath, default=None, required=True,
                        help='Path to initial landcover matrix raster')
    
    optional.add_argument('-h', '--help', action='help', default=argparse.SUPPRESS,
                        help='show this help message and exit')

    optional.add_argument('-H', '--habitat_fn', type=os.path.abspath, default=None,
                        help='Path to outputted habitat layer')
    optional.add_argument('-d', '--resistance_dict_fn', type=os.path.abspath, default=None,
                        help='Path to outputted resistance dictionary CSV')
    optional.add_argument('-r', '--range_fn', type=os.path.abspath, default=None,
                        help='Path to outputted range map')
    
    optional.add_argument('-c', '--crs', type=str, default=None,
                        help='Desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input terrain raster')
    optional.add_argument('-R', '--resolution', type=int, default=None,
                        help='Desired resolution in the units of the chosen CRS, or None to use the resolution of the input terrain raster')
    optional.add_argument('-e', '--resampling', type=str, default="near",
                        help='Resampling method to use if reprojection of the input terrain layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options')
    optional.add_argument('-b', '--bounds', nargs=4, type=float, default=None,
                        help='Four coordinate numbers representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS')
    optional.add_argument('-p', '--padding', type=int, default=0,
                        help='Padding to add around the bounds in the units of the chosen CRS')
    
    optional.add_argument('-m', '--refine_method', type=str, default="forest",
                        help='Method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option')

    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()
