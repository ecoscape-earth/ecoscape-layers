import fiona
import os
from pyproj import Transformer
from osgeo import gdal

            
def warp(input, output, crs, bounds, padding, resolution, resampling='near'):
    '''
    :param input: input file path
    :param output: output file path
    :param crs: output CRS
    :param bounds: output bounds in output CRS
    :param padding: padding to add to the bounds
    :param res: x/y resolution
    :param resampling: resampling algorithm to use. See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r.
    '''

    # Obtain input CRS
    input_src = gdal.Open(input, 0)
    input_crs = input_src.GetProjection()
    input_src = None

    padded_bounds = (bounds[0] - padding, bounds[1] - padding, bounds[2] + padding, bounds[3] + padding)

    # Perform the warp using GDAL
    kwargs = {
        'format': 'GTiff',
        'srcSRS': input_crs,
        'dstSRS': crs,
        'creationOptions': { 'COMPRESS=LZW', },
        'outputBounds': padded_bounds,
        'xRes': resolution,
        'yRes': resolution,
        'resampleAlg': resampling
    }

    gdal.Warp(output, input, **kwargs)

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
