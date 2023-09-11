# EcoScape Layers

This package implements the computation of the matrix (terrain) layer, habitat layers, and terrain-to-resistance mappings that are needed as inputs to the EcoScape algorithm.

## Setup

Besides the dependencies outlined in `requirements.txt`, this package relies on an R script to download range maps from eBird. If you would like to download these range maps, ensure that you have R installed first.

In addition, to use the package to its fullest extent, you will need to have API keys for the IUCN Red List and eBird APIs, which are used to obtain various data on bird species:

- A key for the IUCN Red List API is obtainable from http://apiv3.iucnredlist.org/.

- A key for the eBird Status and Trends API is obtainable from https://science.ebird.org/en/status-and-trends/download-data. This access key must also be used to set up the `ebirdst` R package in order to download range maps from eBird. Please consult the Installation and Data Access sections in https://cornelllabofornithology.github.io/ebirdst/index.html for instructions on configuring the R package. EcoScape currently uses version 1.2020.1 of `ebirdst`.

For command line usage, define these keys as variables `REDLIST_KEY` and `EBIRD_KEY` in a Python file which can then be given as an argument. An example configuration file with dummy keys, `sample_config.py`, is provided for reference. For usage as a Python module, simply provide the keys upon initialization of any `RedList` instance.

The initial terrain raster that we use to produce our layers is a global map produced by [Jung et al.](https://doi.org/10.1038/s41597-020-00599-8) and is available for download at https://zenodo.org/record/4058819 (iucn_habitatclassification_composite_lvl2_ver004.zip). It follows the [IUCN Red List Habitat Classification Scheme](https://www.iucnredlist.org/resources/habitat-classification-scheme).

## Usage

This package can be used on the command line or as a Python module.

For the command line, view argument options with `ecoscape_layers --help`.

For use as a module, there is a general runner function `generate_layers` in `layers_runner.py` that can be used. For more control over the generation process, `layers.py` provides code for the various classes and functions used by `generate_layers`. Illustrative notebook examples of using these classes and functions can be found in the `tests` directory.

### Arguments

Required:

- `config`: path to Python config file containing IUCN Red List and eBird API keys.

- `species_list`: path to txt file of the bird species for which habitat layers should be generated, formatted as 6-letter eBird species codes on individual lines.

- `terrain`: path to initial terrain raster.

Optional:

- `terrain_codes`: path to a CSV containing terrain map codes. If it does not yet exist, a CSV based on the final terrain matrix layer will be created at this path.

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

## Examples

See the `tests` directory for example Jupyter notebooks that use the package to create layers.

- `test_run.ipynb`: a simple example for two bird species performed in a small section of California.

- `ca_birds_habitats.ipynb`: code for reproducing the EcoScape input habitat and matrix layers.

## Known issues

- The eBird and IUCN Red List scientific names do not agree for certain bird species, such as the white-headed woodpecker (eBird code: whhwoo). As the IUCN Red List API only accepts scientific names for its API queries, if this occurs for a bird species, the 6-letter eBird species code for the species must be manually matched to the corresponding scientific name from the IUCN Red List.
