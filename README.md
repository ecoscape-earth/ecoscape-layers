# EcoScape Layers

This package implements the computation of the landscape matrix layer, habitat layers, and landcover-to-resistance mappings that are needed as inputs to the EcoScape algorithm.

**Authors:**

* Jasmine Tai (cjtai@ucsc.edu)
* Ian Holloway (imhollow@ucsc.edu)
* Aadity Sharma (ashar115@ucsc.edu)
* Luca de Alfaro (luca@ucsc.edu)
* Coen Adler (ctadler@ucsc.edu)
* Artie Nazarov (anazarov@ucsc.edu)
* Natalia Ocampo-Peñuela (nocampop@ucsc.edu)
* Natalie Valett (nvalett@ucsc.edu)

## Setup

To use the package, you will need:

- An API key for the IUCN Red List API, which is obtainable from http://apiv3.iucnredlist.org/.

- The initial landcover raster that we use to produce our layers originates from a global map produced by [Jung et al.](https://doi.org/10.1038/s41597-020-00599-8) and is available for download at https://zenodo.org/record/4058819 (iucn_habitatclassification_composite_lvl2_ver004.zip). It follows the [IUCN Red List Habitat Classification Scheme](https://www.iucnredlist.org/resources/habitat-classification-scheme).

- An API key for the eBird Status and Trends API, which is obtainable from https://science.ebird.org/en/status-and-trends/download-data, if you wish to use eBird range maps. We use the data for 2022 in this version of the package. The EcoScape paper uses data from 2020, which has been archived by eBird; see the paper for more details. Note that while eBird is the default source for range maps in layer generation, it mainly provides range map data for birds in the US. If range maps are not found for the species you are studying, consider using range maps from the IUCN Red List (described below).

- If you would like to use range maps from the IUCN Red List instead of eBird, you will need to obtain a copy of the dataset in geodatabase format from http://datazone.birdlife.org/species/requestdis. This can then be passed in as an input to the package. Range maps from the IUCN Red List are generally available for birds around the world, which gives more freedom to experiment with birds that are not in the US compared to using eBird range maps.

## Usage

This package is used as a module. Use the `warp` function in `utils.py` as needed to produce the landcover matrix layer and/or elevation raster with the desired parameters/bounds. The class `LayerGenerator` in `layers.py` can then be used to create corresponding habitat layers for various bird species.

Refer to [tests/test_layers.ipynb](./tests/test_layers.ipynb) for a simple example of how to use the package to produce landcover matrix layers and habitat layers. This example shows many of the various custom overrides that the package has for fine-tuned control of outputs.

### Preparing the landcover matrix layer

The `warp` function is used for reprojecting, rescaling, and/or cropping a raster; the primary use for this would be to process the landcover matrix layer before creating the habitat layers for various bird species afterwards. If an elevation raster is also given for creating habitat layers later, this function can also be used to process that with the same projection, resolution, and bounds/padding as the landcover matrix layer. `warp` accepts as parameters:

- `input`: input raster to process.

- `output`: name of the processed raster.

- `crs`: desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input landcover raster.

  - <b>Note</b>: if the ESRI WKT string contains double quotes that are ignored when the string is given as a command line argument, use single quotes in place of double quotes.

- `resolution`: desired resolution in the units of the chosen CRS, or None to use the resolution of the input landcover raster.

- `bounds`: four coordinate numbers representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS. Optional, but recommended to specify.

- `padding`: padding to add around the bounds in the units of the chosen CRS. Optional.

- `resampling`: resampling method to use if reprojection of the input landcover layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options. Optional.

### Creating Resistance CSV

In `utils.py`, the function `generate_resistance_table` generates resistance values corresponding to terrain type classified by the [IUCN Redlist Habitat Classification Scheme](https://www.iucnredlist.org/resources/habitat-classification-scheme). Resistance values should be based on a specific terrain/habitat type. To control how these resistance values are created, use these function parameters:

- `habitats` (dict[int, dict[str, str | bool]]): IUCN Red List habitat data for the species for which the table should be generated.
- `output_path` (str): The output path for the resistance csv.
- `map_codes` (list\[int\], optional): The map_codes that will be processed in the refinement method. The map codes defined in the habitat data are automatically added to this parameter. Default is all map codes from IUCN plus some extraneous map codes.
- `refinement_method` (function(int, dict[int, dict[str, str | bool]]) -> float, optional): Function that defines resistance when resistance is not defined in habitats. Defaults to general pre-defined refinement method.
  - The default_refinement_method is a general use refinement method. If you want to use your own refinement method, the default_refinement_method is a great starting point.

This function will create a resistance CSV in the desired location and it will also return the data in the form of a pandas Dataframe.

### Creating habitat layers

Once you have the landcover matrix layer prepared, a `LayerGenerator` instance may be initialized given:

- `landcover_fn` (str): file path to the initial landcover raster.
- `redlist_key` (str): IUCN Red List API key.
- `ebird_key` (str | None, optional): eBird API key. Defaults to None.
- `elevation_fn` (str | None, optional): file path to optional input elevation raster for filtering habitat by elevation. Use None for no elevation consideration. Defaults to None.
- `iucn_range_src` (str | None, optional): file path to the IUCN range source if wanted. Defaults to None.

You can then use the `generate_habitat` method to produce a habitat layer for a given bird species based on range map data, terrain preferences, and elevation if specified in the constructor. This method takes the following parameters:

- `species_code` (str): IUCN scientific name or eBird code. It is determined based on what that range_src is set to.
- `iucn_habitat_data` (dict[int, dict[str, int | float | str | bool]] | None, optional): The habitat data for a species received from IUCN Redlist. Defaults to None.
  - The `iucn_habitat_data` is optional parameter but will most likely be used to prevent the fetching of IUCN habitat data multiple times.
- `habitat_fn` (str | None, optional): The output path for the habitat layer. Defaults to None.
- `range_fn` (str | None, optional): The output path for the range map. This is created before making the habitat layer. Defaults to None.
- `range_src` (str, optional): The source from which to obtain range maps from. Defaults to "iucn".
- `current_hab_overrides` (list[str | int] | None, optional): This parameter is passed to the get_current_habitat function and allows the user to redefine what is considered habitat. Defaults to None.
  - `current_hab_overrides` should most likely not be used to preserve consistency across what is determined as habitat. However, it gives users more custom control.

## Known issues

- Migratory bird species and bird species with seasonal ranges are currently _not_ supported.
