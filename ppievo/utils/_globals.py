import os
import numpy as np
from esm.data import Alphabet
from ._time import get_quantization_points_from_geometric_grid

amino_acids = (
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
)

gap_character = "-"

# Paths
MAIN_DIR = "/home/bear/projects/protein-evolution/"
DATA_DIR = os.path.join(MAIN_DIR, "data/local_data/human_ppi")

# Define global quantization time bins
TIME_BINS = np.array(
    [
        float(t)
        for t in get_quantization_points_from_geometric_grid()
        if float(t) >= 5e-3
    ]
)
