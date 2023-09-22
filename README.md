# EcoScape Layers

This package implements the computation of the landscape matrix (terrain) layer, habitat layers, and landcover-to-resistance mappings that are needed as inputs to the EcoScape algorithm.

## Setup

Besides the dependencies outlined in `requirements.txt`, this package relies on an R script to download range maps from eBird. If you would like to download these range maps, ensure that you have R installed first.

In addition, to use the package to its fullest extent, you will need to have API keys for the IUCN Red List and eBird APIs, which are used to obtain various data on bird species:

- A key for the IUCN Red List API is obtainable from http://apiv3.iucnredlist.org/.

- A key for the eBird Status and Trends API is obtainable from https://science.ebird.org/en/status-and-trends/download-data. This access key must also be used to set up the `ebirdst` R package in order to download range maps from eBird. Please consult the Installation and Data Access sections in https://cornelllabofornithology.github.io/ebirdst/index.html for instructions on configuring the R package. EcoScape currently uses version 1.2020.1 of `ebirdst`.

The initial ladncover raster that we use to produce our layers originates from a global map produced by [Jung et al.](https://doi.org/10.1038/s41597-020-00599-8) and is available for download at https://zenodo.org/record/4058819 (iucn_habitatclassification_composite_lvl2_ver004.zip). It follows the [IUCN Red List Habitat Classification Scheme](https://www.iucnredlist.org/resources/habitat-classification-scheme).
<!-- Since this raster is quite large, it is advisable to crop to the rough area of study rather than letting the package process the entire global landcover. We begin with a raster cropped to the United States. -->

The user needs to create a text file with the six-letter eBird codes for the birds they want to to stufy. These must be supplied on a separate line each. To find the eBird six-letter code, visit \url{https://ebird.org/explore} and type the species common or scientific name under "Explore Species"; once in the species website take the last six letters of the web address and use that as the code (e.g. for the Steller's Jay the web address looks like this \url{https://ebird.org/species/stejay} so the six-letter code is "stejay").

## Usage

This package can be used on the command line or as a Python module.

For the command line, view argument options with `ecoscape_layers --help`.

For use as a module, there is a main function `generate_layers` in `layers.py` that can be used for generating layers. `layers.py` also includes the code for the various classes and functions used by `generate_layers`.

### Arguments

Required:

- `redlist`: IUCN Red List API key.

- `ebird`: eBird API key.

- `species_list`: path to txt file of the bird species for which habitat layers should be generated, formatted as 6-letter eBird species codes on individual lines.

- `terrain`: path to initial landcover raster.

Optional:

- `terrain_codes`: path to a CSV containing landcover type codes. If it does not yet exist, a CSV based on the final landscapeerrain matrix layer will be created at this path.

- `species_range_folder`: path to folder to which downloaded eBird range maps should be saved.

- `output_folder`: path to output folder.
    
- `crs`: desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input terrain raster.
    - <b>Note</b>: if the ESRI WKT string contains double quotes that are ignored when the string is given as a command line argument, use single quotes in place of double quotes.

- `resolution`: desired resolution in the units of the chosen CRS, or None to use the resolution of the input terrain raster.

- `resampling`: resampling method to use if reprojection of the input terrain layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options.

- `bounds`: four coordinate numbers representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS.

- `padding`: padding to add around the bounds in the units of the chosen CRS.

- `refine_method`: method by which habitat pixels should be selected when creating a habitat layer.
    - `forest`: selects all forest pixels.
    - `forest_add308`: selects all forest pixels and pixels with code "308" (Shrubland – Mediterranean-type shrubby vegetation).
    - `allsuitable`: selects all terrain deemed suitable for the species, as determined by the IUCN Red List.
    - `majoronly`: selects all terrain deemed of major importance to the species, as determined by the IUCN Red List.

## Known issues

- The eBird and IUCN Red List scientific names do not match for certain bird species, such as the white-headed woodpecker (eBird code: whhwoo). As the IUCN Red List API only accepts scientific names for its API queries, if this occurs for a bird species, the 6-letter eBird species code for the species must be manually matched to the corresponding scientific name from the IUCN Red List.
