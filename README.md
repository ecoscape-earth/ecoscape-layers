# EcoScape Layers

This package implements the computation of the matrix layer, habitat layers, and terrain-to-resistance mappings that are needed as inputs to the EcoScape algorithm.

## Setup

Besides the dependencies outlined in `requirements.txt`, this package relies on an R script to download range maps from eBird. If you would like to download these range maps, ensure that you have R installed first.

In addition, to use the package to its fullest extent, you will need to have API keys for the IUCN Red List and eBird APIs, which are used to obtain various data on bird species:

- A key for the IUCN Red List API is obtainable from http://apiv3.iucnredlist.org/.

- A key for the eBird Status and Trends API is obtainable from https://science.ebird.org/en/status-and-trends/download-data. This access key must also be used to set up the `ebirdst` R package in order to download range maps from eBird. Please consult the Installation and Data Access sections in https://cornelllabofornithology.github.io/ebirdst/index.html for instructions on configuring the R package. EcoScape currently uses version 1.2020.1 of `ebirdst`.

For command line usage, define these keys as variables `REDLIST_KEY` and `EBIRD_KEY` in a Python file which can then be given as an argument. An example configuration file with dummy keys, `sample_config.py`, is provided for reference. For usage as a Python module, simply provide the keys upon initialization of any `RedList` instance.

## Usage

- config 'Path to Python config file containing IUCN Red List and eBird API keys'
    
- species_list 'Path to txt file of the bird species for which habitat layers should be generated, formatted as 6-letter eBird species codes on individual lines'

- terrain 'Path to terrain raster'

- terrain_codes 'Path to a CSV containing terrain map codes. If it does not yet exist, a CSV based on the final terrain matrix layer will be created at this path'

--species_range_folder', type=os.path.abspath, default=default_species_range_folder,
                        help='Path to folder to which downloaded eBird range maps should be saved'

- output_folder 'Path to output folder'
    
- crs 'Desired common CRS of the outputted layers as an ESRI WKT string, or None to use the CRS of the input terrain raster'

- resolution 'Desired resolution in the units of the chosen CRS, or None to use the resolution of the input terrain raster'

- resampling 'Resampling method to use if reprojection of the input terrain layer is required; see https://gdal.org/programs/gdalwarp.html#cmdoption-gdalwarp-r for valid options'

- bounds 'Four coordinate numbers representing a bounding box (xmin, ymin, xmax, ymax) for the output layers in terms of the chosen CRS'

- padding 'Padding to add around the bounds in the units of the chosen CRS'
    
- refine_method 'Method by which habitat pixels should be selected ("forest", "forest_add308", "allsuitable", or "majoronly"). See documentation for detailed descriptions of each option'

- force_new_terrain_codes 'If set to True, forcefully generates a new CSV of the terrain map codes, potentially overwriting any previously existing CSV'

## Examples

See the `tests` directory for example Jupyter notebooks that use the package to create habitat and matrix layers.

- test_run.ipynb: notebook for basic testing of module functionality on a small square of terrain

- ca_birds_habitats.ipynb: notebook for generating terrain and habitats for bird species in California

## Known issues

- The eBird and IUCN Redlist scientific names do not agree for certain bird species, like the white-headed woodpecker. As the IUCN Redlist API only accepts scientific names for its API queries, if this occurs for a bird species, the 6-letter eBird species code for the species must be manually matched to the corresponding scientific name from the IUCN Redlist.
