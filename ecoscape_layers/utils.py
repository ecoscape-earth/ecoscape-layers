import fiona
import os
from pyproj import Transformer
import scgt
import numpy as np
from itertools import chain
from .constants import MAP_CODE_RANGES
import pandas as pd
from typing import Any, Callable
from textwrap import dedent
from osgeo import gdal
import tqdm
import psutil


def reproject_shapefile(
    shapes_path: str,
    dest_crs,
    shapes_layer: str | None = None,
    file_path: str | None = None,
) -> list[dict]:
    """
    Takes a specified shapefile or geopackage and reprojects it to a different CRS.

    :param shapes_path: file path to the shapefile or geopackage to reproject.
    :param dest_crs: CRS to reproject to as an ESRI WKT string.
    :param shapes_layer: if file is a geopackage, the name of the layer that should be reprojected.
    :param file_path: if specified, the file path to write the reprojected result to as a shapefile.
    :return: list of reprojected features.
    """

    features = []

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
            features.append(feature)

        # if file_path is specified, write the result to a new shapefile
        if file_path is not None:
            meta = shp.meta
            meta.update({"driver": "ESRI Shapefile", "crs_wkt": dest_crs})
            with fiona.open(file_path, "w", **meta) as output:
                output.writerecords(features)

    return features


def make_dirs_for_file(file_name: str):
    """Creates intermediate directories in the file path for a file if they don't exist yet.
    The file itself is not created; this just ensures that the directory of the file and all preceding ones
    exist first.

    Args:
        file_name (str): file to make directories for.
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
) -> tuple[float, float, float, float]:
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
        tuple: A tuple containing the transformed coordinates of the bounding box in the output CRS. The tuple has the format (min_x, min_y, max_x, max_y).
    """

    # Define in and out projections
    in_proj = Transformer.from_crs(crs_in, crs_out)

    # Transform corner points individually
    min_x, min_y = in_proj.transform(min_lat, min_lon)
    max_x, max_y = in_proj.transform(max_lat, max_lon)

    return min_x, min_y, max_x, max_y


def iucn_habs_to_codes(
    habitat_codes: list[str | int],
) -> list[str | int]:
    """Takes a list of strings and integers and replaces strings that are found in the IUCN habitat classification scheme with their respective map codes.
    This function does not guarantee removing all strings from the list, as there could be strings that are not found in the IUCN habitat classification scheme.

    Args:
        habitats (list[str  |  int]): A list of map codes which can contain strings matching the IUCN habitat classification scheme. Other strings can also be included but will not be replaced.

    Returns:
        list[str | int]: A list with all IUCN habitat classification scheme keywords replaced with map codes. Strings may still be present that are not found in IUCN habitat classification scheme.
    """

    for i in range(len(habitat_codes)):
        # check for map code range keyword from IUCN habitat classification scheme
        if habitat_codes[i] in MAP_CODE_RANGES:
            # get codes
            map_code_range = MAP_CODE_RANGES[str(habitat_codes[i])]
            new_codes = list(range(*map_code_range))

            # replace keyword with first code
            habitat_codes[i] = new_codes[0]

            # if new_codes have more than one code then put the rest at the end
            # NOTE: new_codes should have more than one code unless a keyword is made with one code
            if len(new_codes) > 1:
                habitat_codes.extend(new_codes[1:])

    return habitat_codes


def confirm_list_int(arr: list[Any]) -> list[int] | None:
    """Confirm that a list of unknown contents contains only integers.

    Args:
        arr (list[Any]): The list that is being checked.

    Returns:
        list[int] | None: A list of ints if successful and None if a non int type is found.
    """

    output: list[int] = []
    for elm in arr:
        if type(elm) is int:
            output.append(elm)
        else:
            # There was a non-int element
            return None

    return output


def in_habs(map_code: int, criteria: list[str | int]) -> bool:
    """Check if a map code is in a list of specified criteria. This criteria is made up of keywords that represent map code ranges and specific map codes.

    Examples:
        criteria_forest308: ["forest", 308]
        criteria_custom: ["forest", "shrubland", *list(range(600, 750))]

    Args:
        map_code (int): An integer map code.
        habitat_codes (list[str  |  int]): A list of strings and/or integers representing the criteria to check map_code against.

    Raises:
        KeyError: If the keyword used does not exist in the IUCN Habitat Classification Scheme.

    Returns:
        bool: True if a map code meets the criteria and False otherwise.
    """

    for criterion in criteria:
        if type(criterion) is str:
            if criterion in MAP_CODE_RANGES:
                rng = MAP_CODE_RANGES[criterion]
                if rng[0] <= map_code < rng[1]:
                    return True
            else:
                error = f"Keyword {criterion} not in IUCN Habitat Classification Scheme"
                raise KeyError(error)
        else:
            if map_code == criterion:
                return True

    return False


def get_map_codes(landcover: scgt.GeoTiff) -> list[int]:
    """Obtains the list of unique map codes present in the landcover/terrain map.

    Args:
        landcover (scgt.GeoTiff): The landcover/terrain map in GeoTiff format.

    Raises:
        ValueError: If the landcover/terrain map is empty.

    Returns:
        list[int]: The list of unique map codes.
    """

    # get landcover as a tile
    tile = landcover.get_all_as_tile()

    if tile is None:
        raise ValueError("Landcover file is empty.")

    # get map data from tile.m as a numpy matrix
    map_data: np.ndarray = tile.m

    # get unique map codes
    map_codes: list[int] = sorted(list(set(chain(map_data.flatten()))))

    return map_codes


def default_refinement_method(
    map_code: int, habitats: dict[int, dict[str, str | bool]]
) -> float:
    if map_code not in habitats:
        return 1.0

    if habitats[map_code]["majorimportance"]:
        return 0.0
    elif habitats[map_code]["suitability"]:
        return 0.1
    else:
        return 1.0


def generate_resistance_table(
    habitats: dict[int, dict[str, str | bool]],
    output_path: str,
    map_codes: list[int] = list(range(100, 1900)),
    refinement_method: Callable[
        [int, dict[int, dict[str, str | bool]]], float
    ] = default_refinement_method,
) -> pd.DataFrame:
    """Generates the resistance dictionary for a given species as a CSV file using habitat preference data from the IUCN Red List.

    - Link to map code definitions: https://www.iucnredlist.org/resources/habitat-classification-scheme
    - The default_refinement_method is a general use refinement method. If you want to use your own refinement method, the default_refinement_method is a great starting point.

    Args:
        habitats (dict[int, dict[str, str | bool]]): IUCN Red List habitat data for the species for which the table should be generated.
        output_path (str): The output path for the resistance csv.
        map_codes (list[int], optional): The map_codes that will be processed in the refinement method. The map codes defined in the habitat data are automatically added to this parameter. Default is all map codes from IUCN plus some extraneous map codes.
        refinement_method (function(int, dict[int, dict[str, str | bool]]) -> float, optional): Function that defines resistance when resistance is not defined in habitats. Defaults general pre-defined refinement method.

    Returns:
        DataFrame: The pandas data frame that is created into a csv.
    """

    # make sure the directory is created for resistance csv
    make_dirs_for_file(output_path)

    # make sure that habitats map codes are included in map_codes
    map_codes = list(set(map_codes + list(habitats.keys())))

    # create resistance data dict based on habitat data
    resistances: dict[int, dict[str, float | str | bool]] = {}
    for c, h in habitats.items():
        resistances[c] = {}
        for k, v in h.items():
            resistances[c][k] = v

    # Get resistance values for each map code
    for code in map_codes:
        # Create dict for code key in resistance dict
        if code not in resistances:
            resistances[code] = {}

        # use resistance from refinement method
        resistances[code]["resistance"] = refinement_method(code, habitats)

    # create a list of all dictionaries ordered on all IUCN codes
    dicts_ordered: list[dict[str, float | str | bool]] = []
    for code in range(100, 1900):
        if code in resistances:
            resistances[code]["map_code"] = code
            dicts_ordered.append(resistances[code])

    # Initialize a pandas data frame
    data = pd.DataFrame(dicts_ordered)

    # reorder the columns
    columns: list[str] = list(data.columns)

    # add the resistance to the end
    if "resistance" in columns:
        i = columns.index("resistance")
        if i != len(columns) - 1:
            columns.append(columns.pop(i))

    # add the map_code to the front
    if "map_code" in columns:
        i = columns.index("map_code")
        if i != 0:
            columns.insert(0, columns.pop(i))

    # set the data to new column order
    data = data[columns]

    # create the csv
    data.to_csv(output_path, index=False)

    return data


def get_current_habitat(
    habitats: dict[int, dict[str, str | bool]],
    overrides: list[str | int] | None = None,
) -> list[int]:
    """Determine the map codes that are considered habitat for a species.
    These map codes will then be used to create the habitat layer with habitat at these map codes.

    - Using the overrides parameter allows you to determine your custom definition of what should be considered habitat.
    Inputs for the overrides may consist of integers that represent specific map codes and strings that
    represent keywords, which will then be converted into map codes. These keywords can take the form of the
    IUCN Habitat Classification Scheme categories listed in the constants.py file. Additionally, you can
    specify "majorimportance" or "suitable" to only use habitats with these qualities for a species.

    Examples:
        overrides_forest308: ["forest", 308]
        overrides_custom: ["forest", "shrubland", *list(range(600, 750))]

    Args:
        habitats (dict[int, dict[str, int  |  float  |  str  |  bool]]): The habitat data received from the IUCN Redlist.
        overrides (list[str  |  int] | None, optional): List of strings/integers representing a custom definition of what constitutes habitat. Defaults to None.

    Raises:
        KeyError: This occurs when keyword is used in overrides that does not exist.
        AssertionError: This error should not occur, but if it does, there is an issue with the code implementation.

    Returns:
        list[int]: List of map codes that are considered habitat for a species.
    """

    # check if to use the override
    if overrides is not None:
        # remove all duplicates
        overrides = list(set(overrides))

        # replace all IUCN habitat classification scheme keywords with map codes
        overrides = iucn_habs_to_codes(overrides)

        # replace keywords (majorimportance, suitable) with map codes
        # search for the keywords and error on invalid keywords
        major_i = None
        suit_i = None
        for i in range(len(overrides)):
            if overrides[i] == "majorimportance":
                major_i = i
            elif overrides[i] == "suitable":
                suit_i = i
            elif type(overrides[i]) is str:
                error = f"""\
                    Keyword {overrides[i]} not found in IUCN habitat classification scheme keywords
                    and is not 'majorimportance' or 'suitable'."""
                raise KeyError(dedent(error))

        # define function to add new codes based on keyword
        def replace_keywords(keyword: str, keyword_i: int):
            if keyword_i is not None:
                new_codes: list[int] = []

                # search for codes that meet condition
                for code, hab in habitats.items():
                    if hab[keyword]:
                        new_codes.append(code)

                # check to suppress warnings
                if overrides is None:
                    return

                # replace keyword with first item
                overrides[keyword_i] = new_codes[0]

                # place remaining codes at the end
                if len(new_codes) > 1:
                    overrides.extend(new_codes[1:])

        # add new codes based on keywords
        if major_i is not None:
            replace_keywords("majorimportance", major_i)
        if suit_i is not None:
            replace_keywords("suitable", suit_i)

        # remove and duplicates due to overlap
        overrides = list(set(overrides))

        # check that all string have been removed
        output = confirm_list_int(overrides)
        if output is None:
            error = "There is an issue with the get_current_habitat implementation."
            raise AssertionError(error)

        return output

    # The default action is to return map_codes in which habitat is considered major importance by IUCN
    output = []
    for code, hab in habitats.items():
        if hab["majorimportance"]:
            output.append(code)

    return output


def warp(
    input: str,
    output: str,
    crs: str,
    resolution: float,
    bounds: tuple[float, float, float, float] | None = None,
    padding: int = 0,
    resampling: str = "near",
):
    """Transform a raster into another raster with various parameters.

    - Note that you cannot add padding to something that is not cropped with bounds, as otherwise there is no space for padding.

    Args:
        input (str): input file path for a raster
        output (str): output file path for a raster
        crs (str): output CRS
        resolution (float): x/y resolution
        bounds (tuple[float, float, float, float] | None, optional): output bounds in output CRS. Defaults to None.
        padding (int, optional): padding to add to the bounds. Defaults to 0.
        resampling (str, optional): resampling algorithm to use. See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r.
    """

    # Obtain input CRS
    input_src = gdal.Open(input, 0)
    input_crs = input_src.GetProjection()
    input_src = None

    if bounds is not None:
        padded_bounds = (
            bounds[0] - padding,
            bounds[1] - padding,
            bounds[2] + padding,
            bounds[3] + padding,
        )
    else:
        padded_bounds = None

    input_name = os.path.basename(input)
    progress = tqdm.tqdm(
        total=100, desc=f"Warping ({input_name})", unit="%", position=0
    )

    def _progress_callback(complete, message, data):
        progress.update(int(complete * 100 - progress.n))

    # get memory in bytes with minimum of 1GB or total memory
    # WARNING: gdal docs says that the memory max is in megabytes but it actual wants bytes from testing
    total_memory = psutil.virtual_memory().total
    min_memory = 1 * (1024**3)
    available_memory = psutil.virtual_memory().available
    if total_memory < min_memory:
        memory = None
    elif available_memory < min_memory:
        memory = min_memory
    else:
        memory = available_memory

    # Perform the warp using GDAL
    kwargs = {
        "format": "GTiff",
        "srcSRS": input_crs,
        "dstSRS": crs,
        "creationOptions": {
            "COMPRESS=LZW",
        },
        "outputBounds": padded_bounds,
        "xRes": resolution,
        "yRes": resolution,
        "resampleAlg": resampling,
        "multithread": True,
        "warpMemoryLimit": memory,
        "callback": _progress_callback,
    }

    gdal.Warp(output, input, **kwargs)

    progress.close()
