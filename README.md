# EcoScape Layers

This package implements the computation of the landscape matrix layer, habitat layers, and landcover-to-resistance mappings that are needed as inputs to the EcoScape algorithm.

## Setup

To use the package, you will need:

- An API key for the IUCN Red List API, which is obtainable from http://apiv3.iucnredlist.org/.

- An API key for the eBird Status and Trends API, which is obtainable from https://science.ebird.org/en/status-and-trends/download-data. We use the data for 2022 in this version of the package. The EcoScape paper uses data from 2020, which has been archived by eBird; see the paper for more details. Note that while eBird is the default source for range maps in layer generation, it mainly provides range map data for birds in the US. If range maps are not found for the species you are studying, consider using range maps from the IUCN Red List (described below).

- If you would like to use range maps from the IUCN Red List, you will need to obtain a copy of the dataset in geodatabase format from http://datazone.birdlife.org/species/requestdis. This can then be passed in as an input to the package.

The initial ladncover raster that we use to produce our layers originates from a global map produced by [Jung et al.](https://doi.org/10.1038/s41597-020-00599-8) and is available for download at https://zenodo.org/record/4058819 (iucn_habitatclassification_composite_lvl2_ver004.zip). It follows the [IUCN Red List Habitat Classification Scheme](https://www.iucnredlist.org/resources/habitat-classification-scheme).

## Usage

This package is used as a module. Use the `warp` function in `layers.py` as needed to produce the landcover matrix layer and/or elevation raster with the desired parameters/bounds. The class `LayerGenerator` in `layers.py` can then be used to create corresponding habitat layers for various bird species.

Refer to `tests/test_layers.ipynb` for a simple example of how to use the package to produce landcover matrix layers and habitat layers.

### Preparing the landcover matrix layer

The `warp` function is used for reprojecting, rescaling, and/or cropping a raster; the primary use for this would be to process the landcover matrix layer before creating the habitat layers for various bird species afterwards. If an elevation raster is also given for creating habitat layers later, this function can also be used to process that with the same projection, resolution, and bounds/padding as the landcover matrix layer. `warp` accepts as required parameters:

- `input`: input raster to process.

- `output`: name of the processed raster.

- `crs`: desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input landcover raster.
    - <b>Note</b>: if the ESRI WKT string contains double quotes that are ignored when the string is given as a command line argument, use single quotes in place of double quotes.

- `resolution`: desired resolution in the units of the chosen CRS, or None to use the resolution of the input landcover raster.

- `bounds`: four coordinate numbers representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS. Optional, but recommended to specify.

- `padding`: padding to add around the bounds in the units of the chosen CRS. Optional.

- `resampling`: resampling method to use if reprojection of the input landcover layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options. Optional.

### Creating habitat layers

Once you have the landcover matrix layer prepared, the `LayerGenerator` instance may be initialized with parameters:

- `redlist_key`: IUCN Red List API key.

- `ebird_key`: eBird API key.

- `landcover_fn`: path to landcover matrix raster. Habitat layers produced under the instance will take on the projection, resolution, and bounds of the landcover matrix raster.

- `elevation_fn`: path to optional elevation raster for filtering habitats by species elevation. If not specified, elevation will not be considered in the creation of habitat layers.

- `iucn_range_src`: path to optional IUCN dataset of ranges for bird species. Refer to Setup for how to obtain this if needed.

You can then use the `generate_habitat` method to produce a habitat layer for a given bird species based on range map data, terrain preferences, and elevation if specified in the constructor. This method takes parameters:

- `species_code`: 6-letter eBird code of the species for which habitat layers should be generated. This can be found by looking up the species on eBird and taking the 6-letter code found at the end of the species page's URL.

- `habitat_fn`: name of output habitat layer.

- `resistance_dict_fn`: name of output resistance dictionary CSV.

- `range_fn`: name of output range map for the species, which is downloaded from eBird or extracted from `iucn_range_src` as an intermediate step for producing the habitat layer.

- `range_src`: source from which to obtain range maps; one of "ebird" or "iucn". If "iucn" is specified, then `iucn_range_src` from the constructor must be specified also.

- `refine_method`: method by which habitat pixels should be selected when creating a habitat layer.
    - `forest`: selects all forest pixels.
    - `forest_add308`: selects all forest pixels and pixels with code "308" (Shrubland – Mediterranean-type shrubby vegetation).
    - `allsuitable`: selects all pixels with landcover deemed suitable for the species, as determined by the IUCN Red List.
    - `majoronly`: selects all pixels with landcover deemed of major importance to the species, as determined by the IUCN Red List.

- `refine_list`: list of map codes for which the corresponding pixels should be considered habitat. This is provided as an alternative to refine_method, which offers limited options, and overrides refine_method if both refine_method and refine_list are specified.

## Known issues

- The eBird and IUCN Red List scientific names do not match for certain bird species, such as the white-headed woodpecker (eBird code: `whhwoo`). As the IUCN Red List API only accepts scientific names for its API queries, if this occurs for a bird species, the 6-letter eBird species code for the species must be manually matched to the corresponding scientific name from the IUCN Red List.

- Bird species with seasonal ranges are currently not supported.
