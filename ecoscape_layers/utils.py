import fiona
import os
from pyproj import Transformer


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

    with fiona.open(shapes_path, "r", layer=shapes_layer) as shp:
        # create a Transformer for changing from the current CRS to the destination CRS
        transformer = Transformer.from_crs(
            crs_from=shp.crs_wkt, crs_to=dest_crs, always_xy=True
        )

        # loop through polygons in each features, transforming all point coordinates within those polygons
        for feature in shp:
            for i, polygon in enumerate(feature["geometry"]["coordinates"]):
                for j, ring in enumerate(polygon):
                    if isinstance(ring, list):
                        feature["geometry"]["coordinates"][i][j] = [
                            transformer.transform(*point) for point in ring
                        ]
                    else:
                        # "ring" is really just a single point
                        feature["geometry"]["coordinates"][i][j] = [
                            transformer.transform(*ring)
                        ]
            myfeatures.append(feature)

        # if file_path is specified, write the result to a new shapefile
        if file_path is not None:
            meta = shp.meta
            meta.update({"driver": "ESRI Shapefile", "crs_wkt": dest_crs})
            with fiona.open(file_path, "w", **meta) as output:
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


def transform_box(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    crs_in: str,
    crs_out: str,
):
    """
    Transforms a bounding box from one coordinate reference system (CRS) to another.

    Args:
        min_lon (float): The minimum longitude of the bounding box.
        min_lat (float): The minimum latitude of the bounding box.
        max_lon (float): The maximum longitude of the bounding box.
        max_lat (float): The maximum latitude of the bounding box.
        crs_in (str): The input CRS in EPSG format. For example "EPSG:4326".
        crs_out (str): The output CRS in EPSG format. For example "EPSG:3395".

    Returns:
        tuple: A tuple containing the transformed coordinates of the bounding box in the output CRS.
        The tuple has the format (min_x, min_y, max_x, max_y).
    """

    # Define in and out projections
    in_proj = Transformer.from_crs(crs_in, crs_out)

    # Transform corner points individually
    min_x, min_y = in_proj.transform(min_lat, min_lon)
    max_x, max_y = in_proj.transform(max_lat, max_lon)

    return min_x, min_y, max_x, max_y
