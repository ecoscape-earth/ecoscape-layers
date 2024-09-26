import numpy as np
import os
import requests
from rasterio import features
from rasterio.windows import Window, from_bounds
from scgt import GeoTiff
from shapely import unary_union
from shapely.geometry import shape
from osgeo import gdal, ogr
from textwrap import dedent

from .redlist import RedList
from .utils import reproject_shapefile, make_dirs_for_file, get_current_habitat


class LayerGenerator:
    """Generate a Habitat Layer with various parameters to control the final output.
    The Habitat Layer is a raster that indicates what is considered habitat for a species.
    """

    def __init__(
        self,
        landcover_fn: str,
        redlist_key: str,
        ebird_key: str | None = None,
        elevation_fn: str | None = None,
        iucn_range_src: str | None = None,
    ):
        """Initializes a LayerGenerator object.

        Args:
            landcover_fn (str): file path to the initial landcover raster.
            redlist_key (str): IUCN Red List API key.
            ebird_key (str | None, optional): eBird API key. Defaults to None.
            elevation_fn (str | None, optional): file path to optional input elevation raster for filtering habitat by elevation; use None for no elevation consideration. Defaults to None.
            iucn_range_src (str | None, optional): file path to the IUCN range source if wanted. Defaults to None.
        """

        self.redlist = RedList(redlist_key, ebird_key)
        self.ebird_key = ebird_key
        self.landcover_fn = os.path.abspath(landcover_fn)
        self.elevation_fn = os.path.abspath(elevation_fn) if elevation_fn else None
        self.iucn_range_src = iucn_range_src

    def get_range_from_iucn(self, species_name: str, output_path: str):
        """Using IUCN gdb file, creates shapefiles usable for refining ranges for specific species with GDAL's ogr module.

        Args:
            species_name (str): scientific name of species to obtain range for.
            output_path (str): path for output .shp file (if it exists already, the old file(s) will be overwritten)
        """

        # We choose to use this option to avoid spending too much time organizing polygons.
        # See https://gdal.org/api/ogrgeometry_cpp.html#_CPPv4N18OGRGeometryFactory16organizePolygonsEPP11OGRGeometryiPiPPKc, https://gdal.org/user/configoptions.html#general-options.
        gdal.SetConfigOption("OGR_ORGANIZE_POLYGONS", "CCW_INNER_JUST_AFTER_CW_OUTER")

        # Open input file and layer, and apply attribute filter using scientific name
        input_src = ogr.Open(self.iucn_range_src, 0)
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
        output_layer = output_src.CreateLayer(
            output_layer_name, geom_type=ogr.wkbMultiPolygon
        )

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
        """Gets range map in geopackage (.gpkg) format for a given bird species.

        Args:
            species_code (str): eBird code for a bird species.
            output_path (str): path to write the range map to.

        Raises:
            requests.HTTPError: Error occurs when the eBird API is expired or invalid
        """

        req_url = f"https://st-download.ebird.org/v1/fetch?objKey=2022/{species_code}/ranges/{species_code}_range_smooth_9km_2022.gpkg&key={self.ebird_key}"
        res = requests.get(req_url)

        if res.status_code == 200:
            with open(output_path, "wb") as res_file:
                res_file.write(res.content)
        else:
            error = f"""\
                Failed to download range map for {species_code} from eBird API.
                Your eBird API key may be expired or invalid."""
            raise requests.HTTPError(dedent(error))

    def generate_habitat(
        self,
        species_code: str,
        iucn_habitat_data: dict[int, dict[str, str | bool]] | None = None,
        habitat_fn: str | None = None,
        range_fn: str | None = None,
        range_src: str = "iucn",
        current_hab_overrides: list[str | int] | None = None,
    ):
        """Runner function for full process of habitat layer generation for one species.

        - iucn_habitat_data is an optional parameter, but should ideally be used to prevent unnecessarily fetching IUCN habitat data multiple times.
        - current_hab_overrides should most likely not be used to preserve consistency across what is determined as habitat.
        However, it gives users more custom control.

        Args:
            species_code (str): IUCN scientific name or eBird code. Determined based on value of range_src.
            iucn_habitat_data (dict[int, dict[str, int  |  float  |  str  |  bool]] | None, optional): The habitat data for a species received from IUCN Redlist. Defaults to None.
            habitat_fn (str | None, optional): The output path for the habitat layer. Defaults to None.
            range_fn (str | None, optional): The output path for the range map. This is created before making the habitat layer. Defaults to None.
            range_src (str, optional): The source from which to obtain range maps from. Defaults to "iucn".
            current_hab_overrides (list[str  |  int] | None, optional): This parameter is passed to the get_current_habitat function and allows the user to redefine what is considered habitat. Defaults to None.

        Raises:
            ValueError: If the user said they wanted eBird range maps but did not provide a ebird_key.
            ValueError: If the user said they wanted IUCN range maps but did not provide an IUCN range map data src.
            AssertionError: If the IUCN Redlist does not contain any habitat data for species.
            FileNotFoundError: If the range map file that is created/fetched cannot be found later in the habitat layer creation pipeline.
        """

        # Check for errors related to optional data provided to class
        if range_src == "ebird" and self.ebird_key is None:
            raise ValueError("eBird API key is required to get range maps from eBird.")

        if range_src == "iucn" and self.iucn_range_src is None:
            raise ValueError("iucn_range_src is required when range_src == 'iucn'.")

        # If file names not specified, build default ones.
        cwd = os.getcwd()
        if habitat_fn is None:
            habitat_fn = os.path.join(cwd, species_code, "habitat.tif")
        if range_fn is None:
            range_fn = os.path.join(cwd, species_code, "range_map.gpkg")

        # Ensure that directories to habitat layer, range map, and resistance dictionary exist.
        make_dirs_for_file(habitat_fn)
        make_dirs_for_file(range_fn)

        # Obtain scientific name of species if needed.
        if range_src == "iucn":
            sci_name = species_code
        else:
            sci_name = self.redlist.get_scientific_name(species_code)

        # Obtain species habitat information from the IUCN Red List.
        if iucn_habitat_data is None:
            iucn_habitat_data = self.redlist.get_habitat_data(sci_name)

        # Check for errors before processing habitat layer.
        if len(iucn_habitat_data) == 0:
            error = f"""\
                Habitat preferences for {species_code} could not be found on the IUCN Red List.
                Habitat layer and resistance dictionary were not generated."""
            raise AssertionError(dedent(error))

        if range_src == "iucn":
            self.get_range_from_iucn(sci_name, range_fn)
        else:
            self.get_range_from_ebird(species_code, range_fn)

        if not os.path.isfile(range_fn):
            src = "IUCN" if range_src == "iucn" else "eBird"
            error = f"Range map could not be found for {species_code} from {src}."
            raise FileNotFoundError(error)

        # Perform intersection between the range and habitable landcover.
        with GeoTiff.from_file(self.landcover_fn) as landcover:
            # Obtain species range as either shapefile from IUCN or geopackage from eBird.
            _, ext = os.path.splitext(range_fn)
            range_shapes = reproject_shapefile(
                range_fn, landcover.dataset.crs, "range" if ext == ".gpkg" else None
            )

            # Prepare range defined as shapes for masking
            if range_src == "iucn":
                for s in range_shapes:
                    if s["geometry"]["type"] == "Polygon":
                        s["geometry"]["coordinates"] = [
                            [el[0] for el in s["geometry"]["coordinates"][0]]
                        ]
                shapes_for_mask = [
                    unary_union([shape(s["geometry"]) for s in range_shapes])
                ]
            else:
                shapes_for_mask = [shape(range_shapes[0]["geometry"])]

            # Define map codes for which corresponding pixels should be considered habitat
            current_habitat_map_codes = get_current_habitat(
                iucn_habitat_data, current_hab_overrides
            )

            # Create the habitat layer
            with landcover.clone_shape(habitat_fn) as output:
                # If elevation raster is provided, obtain min/max elevation and read elevation raster
                if self.elevation_fn is not None:
                    min_elev, max_elev = self.redlist.get_elevation(sci_name)
                    elev = GeoTiff.from_file(self.elevation_fn)
                    cropped_window = (
                        from_bounds(
                            *output.dataset.bounds, transform=elev.dataset.transform
                        )
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
                    window_data = np.isin(window_data, current_habitat_map_codes)

                    # mask out pixels not within elevation range (if elevation raster is provided)
                    if self.elevation_fn is not None:
                        elev_window = Window(
                            tile.x + x_offset, tile.y + y_offset, tile.w, tile.h
                        )
                        elev_window_data = elev.dataset.read(window=elev_window)
                        window_data = (
                            window_data
                            & (elev_window_data >= min_elev)
                            & (elev_window_data <= max_elev)
                        )

                    # write the window result
                    output.dataset.write(window_data, window=window)

                if self.elevation_fn is not None:
                    elev.dataset.close()

            # This sets nodata to None for now, but should be changed later if scgt is modified to support that.
            with GeoTiff.from_file(habitat_fn) as output:
                output.dataset.nodata = None

        print("Habitat layer successfully generated for", species_code)
