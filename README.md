# EcoScape Layers

This package implements the computation of the landscape matrix layer, habitat layers, and landcover-to-resistance mappings that are needed as inputs to the EcoScape algorithm.

## Setup

To use the package, you will need to have API keys for the IUCN Red List and eBird APIs, which are used to obtain various data on bird species:

- A key for the IUCN Red List API is obtainable from http://apiv3.iucnredlist.org/.

- A key for the eBird Status and Trends API is obtainable from https://science.ebird.org/en/status-and-trends/download-data. This access key must also be used to set up the `ebirdst` R package in order to download range maps from eBird. Please consult the Installation and Data Access sections in https://cornelllabofornithology.github.io/ebirdst/index.html for instructions on configuring the R package. EcoScape currently uses version 1.2020.1 of `ebirdst`.

The initial ladncover raster that we use to produce our layers originates from a global map produced by [Jung et al.](https://doi.org/10.1038/s41597-020-00599-8) and is available for download at https://zenodo.org/record/4058819 (iucn_habitatclassification_composite_lvl2_ver004.zip). It follows the [IUCN Red List Habitat Classification Scheme](https://www.iucnredlist.org/resources/habitat-classification-scheme). Since this raster is quite large, it is advisable to crop to the rough area of study rather than letting the package process the entire global landcover.

## Usage

This package can be used on the command line or as a Python module.

For the command line, view argument options with `ecoscape_layers --help`.

For use as a module, the class `LayerGenerator` in `layers.py` can be used to process landcover matrix layers and create habitat layers for various bird species.

### Arguments

Required:

- `redlist_key`: IUCN Red List API key.

- `ebird_key`: eBird API key.

- `species_code`: 6-letter eBird code of the species for which habitat layers should be generated. This can be found by looking up the species on eBird and taking the 6-letter code found at the end of the species page's URL.

- `landcover`: path to initial landcover raster.

Optional:

- `habitat_fn` name of output habitat layer.

- `resistance_dict_fn` name of output resistance dictionary CSV.

- `range_fn`: name of output range map for the species, which is downloaded as an intermediate step for producing the habitat layer.
    
- `crs`: desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input landcover raster.
    - <b>Note</b>: if the ESRI WKT string contains double quotes that are ignored when the string is given as a command line argument, use single quotes in place of double quotes.

- `resolution`: desired resolution in the units of the chosen CRS, or None to use the resolution of the input landcover raster.

- `resampling`: resampling method to use if reprojection of the input landcover layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options.

- `bounds`: four coordinate numbers representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS.

- `padding`: padding to add around the bounds in the units of the chosen CRS.

- `refine_method`: method by which habitat pixels should be selected when creating a habitat layer.
    - `forest`: selects all forest pixels.
    - `forest_add308`: selects all forest pixels and pixels with code "308" (Shrubland – Mediterranean-type shrubby vegetation).
    - `allsuitable`: selects all pixels with landcover deemed suitable for the species, as determined by the IUCN Red List.
    - `majoronly`: selects all pixels with landcover deemed of major importance to the species, as determined by the IUCN Red List.

## Known issues

- The eBird and IUCN Red List scientific names do not match for certain bird species, such as the white-headed woodpecker (eBird code: whhwoo). As the IUCN Red List API only accepts scientific names for its API queries, if this occurs for a bird species, the 6-letter eBird species code for the species must be manually matched to the corresponding scientific name from the IUCN Red List.

- Bird species with different seasonal ranges on eBird are currently not supported.
