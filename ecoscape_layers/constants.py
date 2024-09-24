RESAMPLING_METHODS = [
    "near",
    "bilinear",
    "cubic",
    "cubicspline",
    "lanczos",
    "average",
    "rms",
    "mode",
    "max",
    "min",
    "med",
    "q1",
    "q3",
    "sum",
]


# IUCN map code ranges
# NOTE: These codes can be found on https://www.iucnredlist.org/resources/habitat-classification-scheme
MAP_CODE_RANGES: dict[str, tuple[int, int]] = {
    "forest": (100, 200),
    "savanna": (200, 300),
    "shrubland": (300, 400),
    "grassland": (400, 500),
    "wetlands": (500, 600),
    "rocky": (600, 700),
    "caves": (700, 800),
    "desert": (800, 900),
    "marine": (900, 1400),
    "marine_neritic": (900, 1000),
    "marine_oceanic": (1000, 1100),
    "marine_deep": (1100, 1200),
    "marine_intertidal": (1200, 1300),
    "marine_coastal": (1300, 1400),
    "artificial": (1400, 1600),
    "artificial_terrestrial": (1400, 1500),
    "artificial_aquatic": (1500, 1600),
    "introduced_vegetation": (1600, 1700),
    "other": (1700, 1800),
    "unknown": (1800, 1900),
}
