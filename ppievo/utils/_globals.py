import os

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
