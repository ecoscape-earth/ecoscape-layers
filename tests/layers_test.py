import os
import sys
sys.path.append(".")

from ecoscape_layers import (
    LayerGenerator,
    warp,
    RedList,
    generate_resistance_table,
    in_habs,
    default_refinement_method,
)



species_list = ["cowpig1", "ibgshr1"]
# species_list = ["acowoo", "stejay"]

# Paths
BASE_DIR = "."
DATA_PATH = os.path.join(BASE_DIR, "tests")

# Load keys
REDLIST_KEY = open(os.path.join(BASE_DIR, "private/iucn_key.txt"), "r").read().strip(r' ')
EBIRD_KEY = open(os.path.join(BASE_DIR, "private/ebird_key.txt"), "r").read().strip(r' ')

# Initialze redlist object with keys
redlist = RedList(REDLIST_KEY, EBIRD_KEY)


landcover_fn = os.path.join(DATA_PATH, "inputs", "test_terrain_spain.tif")

layer_generator = LayerGenerator(landcover_fn, REDLIST_KEY, EBIRD_KEY)
redlist = RedList(REDLIST_KEY, EBIRD_KEY)


habitat_data = {}

for species_code in species_list:
    habitat_fn = os.path.join(DATA_PATH, "outputs", species_code, "habitat_test.tif")
    resistance_dict_fn = os.path.join(DATA_PATH, "outputs", species_code, "resistance_test.csv")
    range_fn = os.path.join(DATA_PATH, "outputs", species_code, "range_map_2022.gpkg")

    range_src = "ebird"

    # get IUCN Redlist Habitat data
    habitat_data = redlist.get_habitat_data(species_code, ebird_code=True)

    # create the resistance csv
    generate_resistance_table(habitat_data, resistance_dict_fn)

    # create the habitat layer
    layer_generator.generate_habitat(
        species_code,
        habitat_data,
        habitat_fn,
        range_fn,
        range_src,
        current_hab_overrides = ["suitable"]
    )
