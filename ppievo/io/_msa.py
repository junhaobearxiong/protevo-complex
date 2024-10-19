import json
import os
from joblib import Parallel, delayed
from tqdm import tqdm
from ppievo.io import read_msa
from ppievo.utils import DATA_DIR


# Writing to file
def write_dict(dictionary, filename):
    with open(filename, "w") as f:
        json.dump(dictionary, f)


# Reading from file
def read_dict(filename):
    with open(filename, "r") as f:
        return json.load(f)


def read_fasta(filename):
    """
    Simple helper to read fasta file
    """
    with open(filename) as f:
        header, sequence = "", ""
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if header and sequence:
                    yield (header, sequence)
                header, sequence = line[1:], ""
            else:
                sequence += line
        if header and sequence:
            yield (header, sequence)


def construct_pair_msa(
    protein1,
    protein2,
    output_dir=os.path.join(DATA_DIR, "pair_msa"),
    msa_dir=os.path.join(DATA_DIR, "humanPPI_MSA/MSA"),
):
    """
    Function to construct pair MSA from the individual MSAs of two proteins
    This is modified from https://colab.research.google.com/drive/1suhoIB5q6xn0APFHJE8c1eMiCuv9gCk_

    Note:
    1. This function is simple because the individual MSAs already have the paralogs filtered out
        "All predicted protein sequences were aligned to their human orthologs, if available,
        and we used reciprocal best-hit criteria (36) to distinguish orthologs from paralogs."
    """
    # Read sequences for both proteins
    seqs1 = dict(read_fasta(os.path.join(msa_dir, f"{protein1}.fas")))
    seqs2 = dict(read_fasta(os.path.join(msa_dir, f"{protein2}.fas")))

    # Find common taxa
    common_taxa = set(seqs1.keys()) & set(seqs2.keys())

    # Write paired MSA
    output_file = os.path.join(output_dir, f"{protein1}_{protein2}.fas")
    with open(output_file, "w") as f:
        # Write header
        f.write(
            f"#{len(next(iter(seqs1.values())))},{len(next(iter(seqs2.values())))}\t1,1\n"
        )
        f.write(f">{protein1}\t{protein2}\n")
        f.write(f"{next(iter(seqs1.values()))}{next(iter(seqs2.values()))}\n")

        # Write paired sequences
        for taxon in common_taxa:
            if taxon != "query":  # Exclude query sequence
                # Header is taxon
                f.write(f">{taxon}\n")
                f.write(f"{seqs1[taxon]}{seqs2[taxon]}\n")


def get_protein_lengths(
    protein1, protein2, pair_msa_dir=os.path.join(DATA_DIR, "pair_msa")
):
    """
    Function to get the lengths of two proteins from their paired MSA file.

    Args:
    protein1 (str): Name of the first protein
    protein2 (str): Name of the second protein
    pair_msa_dir (str): Directory containing the paired MSA files

    Returns:
    tuple: Lengths of protein1 and protein2 as integers
    """

    # Construct the filename
    filename = os.path.join(pair_msa_dir, f"{protein1}_{protein2}.fas")

    # Check if the file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(
            f"Paired MSA file for {protein1} and {protein2} not found."
        )

    # Read the first line of the file
    with open(filename, "r") as f:
        first_line = f.readline().strip()

    # Extract the lengths
    if first_line.startswith("#"):
        lengths = first_line[1:].split("\t")[0].split(",")
        if len(lengths) == 2:
            return tuple(map(int, lengths))
        else:
            raise ValueError("Unexpected format in the header of the paired MSA file.")
    else:
        raise ValueError("The paired MSA file does not start with the expected header.")


def read_msa_index_with_genome_id(msa_path: str) -> dict[str, str]:
    """
    Read MSA and use only the genome assembly identifier

    For some reasons, the tree construction strips the intermediate white space in the sequence identifier
    when annotating each leaf (i.e. rows of a MSA)
    i.e. 'GCA_016906955.1 Phacochoerus:Suidae:Artiodactyla:Mammalia:Chordata'
    becomes 'GCA_016906955.1'
    So to index into the MSA, we need to change the key to only include the genome identifier
    """
    msa = read_msa(msa_path)
    msa_new = dict()
    for seq_name, seq in msa.items():
        if "\t" in seq_name:
            # If it is the query, the sequence id is '{protein1}\t{protein2}'
            new_seq_name = seq_name.split("\t")[0]
        else:
            new_seq_name = seq_name.split(" ")[0]
        msa_new[new_seq_name] = seq

    assert len(msa) == len(
        msa_new
    ), "Genome assembly identifiers are not unique to the rows"
    return msa_new
