import csv
import numpy as np
import os
import requests
from rasterio import features
from rasterio.windows import Window, from_bounds
from scgt import GeoTiff
from shapely import unary_union
from shapely.geometry import shape
from osgeo import gdal, ogr

from .constants import HAB_308, REFINE_METHODS
from .redlist import RedList
from .utils import reproject_shapefile, make_dirs_for_file

import tqdm
import psutil


class LayerGenerator(object):
    """
    For things like reprojecting, building resistance tables, and creating habitat layers and landcover matrix layers.
    This class maintains a common CRS, resolution, and resampling method for this purpose.
    """

    def __init__(
        self,
        redlist_key: str,
        landcover_fn: str,
        elevation_fn: str | None = None,
        iucn_range_src: str | None = None,
        ebird_key: str | None = None,
    ):
        """
        Initializes a LayerGenerator object.
        :param redlist_key: IUCN Red List API key.
        :param landcover_fn: file path to the initial landcover raster.
        :param elevation_fn: file path to optional input elevation raster for filtering habitat by elevation; use None for no elevation consideration.
        :param iucn_range_src: file path to the IUCN range source if wanted.
        :param ebird_key: eBird API key.
        """
        self.redlist = RedList(redlist_key, ebird_key)
        self.ebird_key = ebird_key
        self.landcover_fn = os.path.abspath(landcover_fn)
        self.elevation_fn = None if elevation_fn is None else os.path.abspath(elevation_fn)
        self.iucn_range_src = iucn_range_src

    def get_map_codes(self, landcover: GeoTiff) -> list[int]:
        """
        Obtains the list of unique landcover map codes present in the landcover map.
        This is used to determine the map codes for which resistance values need to be defined.

        :param landcover: The landcover map in GeoTiff format.
        """

        # get the bounds of the landcover map
        bounds = landcover.dataset.bounds

        # convert bounds to id num
        bounds_id = hash(bounds)

        # get the file name of the landcover map
        landcover_fn_basename = os.path.basename(self.landcover_fn)
        landcover_fn_basename = os.path.splitext(landcover_fn_basename)[0]

        # add the bounds to the landcover file name
        map_codes_id = f"{landcover_fn_basename}_{bounds_id}"

        # create map codes file name
        map_codes_fn = f"map_codes_{map_codes_id}.csv"

        # create map codes directory
        map_codes_dir = os.path.join(os.path.dirname(self.landcover_fn), "map_codes")
        os.makedirs(map_codes_dir, exist_ok=True)

        # create the full path to the map codes file
        map_codes_output_path = os.path.join(map_codes_dir, map_codes_fn)

        # check if the map codes file exists
        if os.path.exists(map_codes_output_path):
            # read the map codes file
            with open(map_codes_output_path, "r") as csvfile:
                reader = csv.reader(csvfile)
                next(reader)
                map_codes = [int(row[0]) for row in reader]

            return map_codes

        # get all unique map codes from the landcover map
        tile = landcover.get_all_as_tile()

        if tile is None:
            raise ValueError("Landcover file is empty.")

        map_codes: list[int] = sorted(list(np.unique(tile.m)))

        # write the map codes to a file
        with open(map_codes_output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["map_code"])
            for map_code in map_codes:
                writer.writerow([map_code])

        return map_codes

    def get_range_from_iucn(self, species_name: str, input_ranges_gdb: str, output_path: str):
        """
        Using IUCN gdb file, creates shapefiles usable for refining ranges for specific species with GDAL's ogr module.

        :param species_name: scientific name of species to obtain range for.
        :param input_ranges_gdb: path to input file (/BOTW.gdb)
        :param output_path: path for output .shp file (if exists already, the old file(s) will be deleted)
        """
        # We choose to use this option to avoid spending too much time organizing polygons.
        # See https://gdal.org/api/ogrgeometry_cpp.html#_CPPv4N18OGRGeometryFactory16organizePolygonsEPP11OGRGeometryiPiPPKc, https://gdal.org/user/configoptions.html#general-options.
        gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS", "CCW_INNER_JUST_AFTER_CW_OUTER")

        # Open input file and layer, and apply attribute filter using scientific name
        input_src = ogr.Open(input_ranges_gdb, 0)
        input_layer = input_src.GetLayer()
        input_layer_defn = input_layer.GetLayerDefn()
        input_layer.SetAttributeFilter(f"sci_name = '{species_name}'")
        input_spatial_ref = input_layer.GetSpatialRef()
        input_spatial_ref.MorphToESRI()

        # Define output driver, delete old output file(s) if they exist
        output_driver = ogr.GetDriverByName("ESRI Shapefile")
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
        output_prj = open(os.path.splitext(output_path)[0] + ".prj", "w")
        output_prj.write(input_spatial_ref.ExportToWkt())
        output_prj.close()

        # Save and close files
        input_src = None
        output_src = None

        # Reset the GDAL config option
        gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS", "DEFAULT")

    def get_range_from_ebird(self, species_code: str, output_path: str):
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
        else:
            raise requests.HTTPError(
                f"Failed to download range map for {species_code} from eBird API. "
                + "Your eBird API key may be expired or invalid."
            )

    def generate_resistance_table(
        self,
        habitats: list[dict[str, str | int | float]],
        landcover: GeoTiff,
        output_path: str,
        refine_method: str,
    ):
        """
        Generates the resistance dictionary for a given species as a CSV file using habitat preference data from the IUCN Red List.
        - Major importance terrain is assigned a resistance of 0.
        - Suitable (but not major importance) terrain is assigned a resistance of 0.1.
        - All other terrain is assigned a resistance of 1.

        :param habitats: IUCN Red List habitat data for the species for which the table should be generated.
        :param landcover: GeoTiff object representing the landcover map.
        :param output_path: Path of the CSV file to which the species' resistance table should be saved.
        :param refine_method: Method used for refining the resistance values.
        """

        # get the map codes
        map_codes = self.get_map_codes(landcover)

        with open(output_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(habitats[0].keys())

            # map codes from the landcover map
            for map_code in map_codes:
                h = next((hab for hab in habitats if hab["map_code"] == map_code), None)
                if h is not None:
                    if refine_method == "forest" or refine_method == "forest_add308":
                        h["resistance"] = 0 if map_code >= 100 and map_code < 200 else h["resistance"]
                    writer.writerow(h.values())
                else:
                    if refine_method == "forest" or refine_method == "forest_add308":
                        writer.writerow([""] * 5 + [map_code] + [0 if map_code >= 100 and map_code < 200 else 1])
                    else:
                        writer.writerow([""] * 5 + [map_code] + [1])

    def get_good_terrain(self, habitats, refine_method="forest_add308") -> list[int]:
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
        else:
            return []

    def generate_habitat(
        self,
        species_code: str,
        habitat_fn: str | None = None,
        resistance_dict_fn: str | None = None,
        range_fn: str | None = None,
        range_src: str = "iucn",
        refine_method: str | None = "forest",
        refine_list: list | None = None,
    ):
        """
        Runner function for full process of habitat and matrix layer generation for one bird species.
        :param species_code: IUCN scientific name or 6-letter eBird code that is determined by range_src.
        :param habitat_fn: name of output habitat layer.
        :param resistance_dict_fn: name of output resistance dictionary CSV.
        :param range_fn: name of output range map for the species, which may be created as an intermediate step for producing the habitat layer.
        :param range_src: source from which to obtain range maps; "ebird" or "iucn".
        :param refine_method: method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option.
        :param refine_list: list of map codes for which the corresponding pixels should be considered habitat. Alternative to refine_method, which offers limited options. If both refine_method and refine_list are given, refine_list is prioritized.
        """

        if range_src == "ebird" and self.ebird_key is None:
            raise ValueError("eBird API key is required to get range maps from eBird.")

        if range_src == "iucn" and self.iucn_range_src is None:
            raise ValueError("iucn_range_src is required when range_src == 'iucn'.")

        if refine_list:
            refine_method = None
        elif refine_method not in REFINE_METHODS:
            refine_method = "forest"

        # If file names not specified, build default ones.
        cwd = os.getcwd()
        if habitat_fn is None:
            habitat_fn = os.path.join(cwd, species_code, "habitat.tif")
        if resistance_dict_fn is None:
            resistance_dict_fn = os.path.join(cwd, species_code, "resistance.csv")
        if range_fn is None:
            range_fn = os.path.join(cwd, species_code, "range_map.gpkg")

        # Ensure that directories to habitat layer, range map, and resistance dictionary exist.
        make_dirs_for_file(habitat_fn)
        make_dirs_for_file(resistance_dict_fn)
        make_dirs_for_file(range_fn)

        # Obtain species habitat information from the IUCN Red List.
        if range_src == "iucn":
            sci_name = species_code
        else:
            sci_name = self.redlist.get_scientific_name(species_code)
            if sci_name is None:
                error_msg = f"Scientific name for eBird code '{species_code}' could not be found."
                raise ValueError(error_msg)

        habs = self.redlist.get_habitat_data(sci_name)

        if refine_method == "forest_add308" and len([hab for hab in habs if hab["code"] == "3.8"]) == 0:
            habs.append(HAB_308)

        if len(habs) == 0:
            raise AssertionError(
                f"Habitat preferences for {species_code} could not be found on the IUCN Red List. "
                + "Habitat layer and resistance dictionary were not generated."
            )

        # Perform intersection between the range and habitable landcover.
        with GeoTiff.from_file(self.landcover_fn) as landcover:
            # Create the resistance table.
            self.generate_resistance_table(
                habitats=habs,
                landcover=landcover,
                output_path=resistance_dict_fn,
                refine_method=refine_method,
            )

            # Obtain species range as either shapefile from IUCN or geopackage from eBird.
            if range_src == "iucn":
                if self.iucn_range_src is None:
                    raise ValueError("No IUCN range source was specified. Habitat layer was not generated.")
                self.get_range_from_iucn(sci_name, self.iucn_range_src, range_fn)
            else:
                self.get_range_from_ebird(species_code, range_fn)

            if not os.path.isfile(range_fn):
                src = "IUCN" if range_src == "iucn" else "eBird"
                error_msg = f"Range map could not be found for {species_code} from {src}."
                raise FileNotFoundError(error_msg)

            _, ext = os.path.splitext(range_fn)
            range_shapes = reproject_shapefile(range_fn, landcover.dataset.crs, "range" if ext == ".gpkg" else None)

            # Prepare range defined as shapes for masking
            if range_src == "iucn":
                for s in range_shapes:
                    if s["geometry"]["type"] == "Polygon":
                        s["geometry"]["coordinates"] = [[el[0] for el in s["geometry"]["coordinates"][0]]]
                shapes_for_mask = [unary_union([shape(s["geometry"]) for s in range_shapes])]
            else:
                shapes_for_mask = [shape(range_shapes[0]["geometry"])]

            # Define map codes for which corresponding pixels should be considered habitat
            if refine_list is None:
                good_terrain_for_hab = self.get_good_terrain(habs, refine_method)
            else:
                good_terrain_for_hab = refine_list

            # Create the habitat layer
            with landcover.clone_shape(habitat_fn) as output:
                # If elevation raster is provided, obtain min/max elevation and read elevation raster
                if self.elevation_fn is not None:
                    min_elev, max_elev = self.redlist.get_elevation(sci_name)
                    elev = GeoTiff.from_file(self.elevation_fn)
                    cropped_window = (
                        from_bounds(*output.dataset.bounds, transform=elev.dataset.transform)
                        .round_lengths()
                        .round_offsets(pixel_precision=0)
                    )
                    x_offset, y_offset = cropped_window.col_off, cropped_window.row_off

                reader = output.get_reader(b=0, w=10000, h=10000)

                for tile in reader:
                    # get window and fit to the tiff's bounds if necessary
                    tile.fit_to_bounds(width=output.width, height=output.height)
                    window = Window(tile.x, tile.y, tile.w, tile.h)

                    # mask out pixels from landcover not within range of shapes
                    window_data = landcover.dataset.read(window=window, masked=True)
                    shape_mask = features.geometry_mask(
                        shapes_for_mask,
                        out_shape=(tile.h, tile.w),
                        transform=landcover.dataset.window_transform(window),
                    )
                    window_data.mask = window_data.mask | shape_mask
                    window_data = window_data.filled(0)

                    # get pixels where terrain is good
                    window_data = np.isin(window_data, good_terrain_for_hab)

                    # mask out pixels not within elevation range (if elevation raster is provided)
                    if self.elevation_fn is not None:
                        elev_window = Window(tile.x + x_offset, tile.y + y_offset, tile.w, tile.h)
                        elev_window_data = elev.dataset.read(window=elev_window)
                        window_data = window_data & (elev_window_data >= min_elev) & (elev_window_data <= max_elev)

                    # write the window result
                    output.dataset.write(window_data, window=window)

                if self.elevation_fn is not None:
                    elev.dataset.close()

            # This sets nodata to None for now, but should be changed later if scgt is modified to support that.
            with GeoTiff.from_file(habitat_fn) as output:
                output.dataset.nodata = None

        print("Habitat layer successfully generated for", species_code)


def warp(
    input: str,
    output: str,
    crs: str,
    resolution: float,
    bounds=None,
    padding: int = 0,
    resampling: str = "near",
):
    """
    :param input: input file path
    :param output: output file path
    :param crs: output CRS
    :param res: x/y resolution
    :param resampling: resampling algorithm to use. See https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r.
    :param bounds: output bounds in output CRS
    :param padding: padding to add to the bounds
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
    progress = tqdm.tqdm(total=100, desc=f"Warping ({input_name})", unit="%", position=0)

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
