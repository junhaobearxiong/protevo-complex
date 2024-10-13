"""
Helper functions to process the dataset from the paper
https://www.biorxiv.org/content/10.1101/2024.10.01.615885v1

Datasets are downloaded from: https://conglab.swmed.edu/humanPPI/humanPPI_download.html
on 10-07-2024. In particular, we downloaded:
1. The omicsMSA for each individual human protein ("humanPPI_MSA.zip")
2. The structures and contacts probability for final confident predictions 
    from Alphafold2 ("PDBs.zip")

To construct pair MSAs, we identify all interacting pairs by the name of the PDB files
then use the species information to pair the individual MSAs
"""

import hashlib
import os
import re
from typing import Optional
import cherryml.phylogeny_estimation
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import cherryml
from functools import partial

from ppievo.io import write_msa
from ppievo.utils import amino_acids, gap_character

MAIN_DIR = "/home/bear/projects/protein-evolution/"
DATA_DIR = os.path.join(MAIN_DIR, "data/local_data/human_ppi")


def get_all_interacting_pairs(pdb_dir=os.path.join(DATA_DIR, "PDBs")):
    """
    Retrieve all putative interacting pairs from the directory storing
    the final confident predictions from Alphafold2
    """
    interacting_pairs = set()
    pattern = re.compile(r"([A-Z0-9]+)_S\d+__([A-Z0-9]+)_S\d+\.(pdb|contacts)")

    for filename in os.listdir(pdb_dir):
        match = pattern.match(filename)
        if match:
            protein1, protein2 = match.group(1), match.group(2)
            interacting_pairs.add(tuple(sorted((protein1, protein2))))

    return list(interacting_pairs)


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
    output_dir=os.path.join(DATA_DIR, "pair_msas"),
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
    protein1, protein2, pair_msa_dir=os.path.join(DATA_DIR, "pair_msas")
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


def _subsample_and_split_pair_msa(
    pair_msa_path: str,
    num_sequences: Optional[int],
    output_pair_msa_dir: str,
    output_unpair_msa_dir: str,
    return_full_length_unaligned_sequences: bool = False,
):
    """
    Subsample sequences in an paired MSA, also split the subsampled pair MSA
    into the unpaired MSAs and store these separately

    If `return_full_length_unaligned_sequences=True`, then the original,
    full-length, unaligned sequences will be returned. (In particular,
    insertions wrt the original sequence will be retained and uppercased,
    and gaps will be removed from all sequences.)
    """
    if not os.path.exists(pair_msa_path):
        raise FileNotFoundError(f"MSA file {pair_msa_path} does not exist!")

    # Read MSA
    # type: list[tuple[str, str]]
    pair_msa, unpair_msa1, unpair_msa2 = [], [], []
    with open(pair_msa_path) as file:
        lines = list(file)
        n_lines = len(lines)
        # First line contain the length of the two alignments, indicates where the two MSAs were concatenated
        protein1_len = int(lines[0].split(",")[0].split("#")[1])
        # protein2_len = int(file_header.split(',')[1].split('\t')[0])
        # Second line contains the name of the two proteins in the pairs
        protein1 = lines[1][1:].split("\t")[0]
        protein2 = lines[1][1:].split("\t")[1][:-1]
        pair_name = f"{protein1}_{protein2}"
        for i in range(1, n_lines, 2):
            if not lines[i][0] == ">":
                raise Exception("Protein ID line should start with '>'")
            # Protein ID contains species identifier
            # e.g. 'GCA_016906955.1 Phacochoerus:Suidae:Artiodactyla:Mammalia:Chordata'
            protein_id = lines[i][1:].strip()
            protein_seq = lines[i + 1].strip()
            # TODO: we will revisit this
            # Current understanding is all insertions relative to humans are already removed
            # and unclear what lower case characters mean
            # For now, we are keeping all lower case characters
            if return_full_length_unaligned_sequences:
                # In this case, we make all lowercase letters uppercase and
                # remove gaps.
                def make_upper_if_lower_and_clear_gaps(c: str) -> str:
                    # Note that c may be a gap, which is why we
                    # don't just do c.upper(); technically it
                    # works to do "-".upper() but I think it's
                    # confusing so I'll just write more code.
                    if c.islower():
                        return c.upper()
                    elif c.isupper():
                        return c
                    else:
                        assert c == "-"
                        return ""

                pair_seq = "".join(
                    [make_upper_if_lower_and_clear_gaps(c) for c in protein_seq]
                )
                unpair_seq1 = "".join(
                    [
                        make_upper_if_lower_and_clear_gaps(c)
                        for i, c in enumerate(protein_seq)
                        if i < protein1_len
                    ]
                )
                unpair_seq2 = "".join(
                    [
                        make_upper_if_lower_and_clear_gaps(c)
                        for i, c in enumerate(protein_seq)
                        if i >= protein1_len
                    ]
                )
            else:
                # Keeping all lowercase characters for now
                pair_seq = "".join([c.upper() for c in protein_seq])
                unpair_seq1 = "".join(
                    [c.upper() for i, c in enumerate(protein_seq) if i < protein1_len]
                )
                unpair_seq2 = "".join(
                    [c.upper() for i, c in enumerate(protein_seq) if i >= protein1_len]
                )

            # Replace unknown characters with gap
            pair_seq = "".join(
                [c if c in amino_acids else gap_character for c in pair_seq]
            )
            unpair_seq1 = "".join(
                [c if c in amino_acids else gap_character for c in unpair_seq1]
            )
            unpair_seq2 = "".join(
                [c if c in amino_acids else gap_character for c in unpair_seq2]
            )

            # Add record to each MSA
            pair_msa.append((protein_id, pair_seq))
            unpair_msa1.append((protein_id, unpair_seq1))
            unpair_msa2.append((protein_id, unpair_seq2))

        # Check that all sequences in the MSA have the same length.
        if not return_full_length_unaligned_sequences:
            for i in range(len(pair_msa) - 1):
                if len(pair_msa[i][1]) != len(pair_msa[i + 1][1]):
                    raise Exception(
                        f"Sequence\n{pair_msa[i][1]}\nand\n{pair_msa[i + 1][1]}\nin the "
                        f"MSA do not have the same length! ({len(pair_msa[i][1])} vs"
                        f" {len(pair_msa[i + 1][1])})"
                    )

    # Subsample MSA
    # Sebastian's fancy way to get random seeds
    family_int_hash = (
        int(
            hashlib.sha512((pair_name + "-_subsample_msa").encode("utf-8")).hexdigest(),
            16,
        )
        % 10**8
    )
    rng = np.random.default_rng(family_int_hash)
    nseqs = len(pair_msa)
    if num_sequences is not None:
        max_seqs = min(nseqs, num_sequences)
        # Ensure query sequence is always included
        seqs_to_keep = [0] + list(
            rng.choice(range(1, nseqs, 1), size=max_seqs - 1, replace=False)
        )
        seqs_to_keep = sorted(seqs_to_keep)
        pair_msa = [pair_msa[i] for i in seqs_to_keep]
        unpair_msa1 = [unpair_msa1[i] for i in seqs_to_keep]
        unpair_msa2 = [unpair_msa2[i] for i in seqs_to_keep]

    # Note: this function sorts the rows by the sequence name
    # so the query sequence might not be the first row
    write_msa(
        msa=dict(pair_msa),
        msa_path=os.path.join(output_pair_msa_dir, pair_name + ".txt"),
    )
    write_msa(
        msa=dict(unpair_msa1),
        msa_path=os.path.join(output_unpair_msa_dir, f"{pair_name}-{protein1}.txt"),
    )
    write_msa(
        msa=dict(unpair_msa2),
        msa_path=os.path.join(output_unpair_msa_dir, f"{pair_name}-{protein2}.txt"),
    )


def _train_test_split_interaction_pairs(
    num_train_pairs: int = 1000, num_test_pairs: int = 1000, seed: int = 42
):
    """
    Split protein interaction pairs into train and test sets, respecting the interaction structure.

    This function splits protein interaction pairs into train and test sets in a way that
    maximizes the likelihood of both proteins in a pair being in the same set. It does this
    by treating the interactions as a graph and splitting based on connected components.

    Parameters:
    num_train_pairs (int): The desired number of pairs in the train set. Default is 1000.
    num_test_pairs (int): The desired number of pairs in the test set. Default is 1000.
    seed (int): Random seed for reproducibility. Default is 42.

    Returns:
    tuple: Two lists of tuples, (train_pairs, test_pairs), where each tuple is a protein pair.

    Raises:
    ValueError: If there aren't enough pairs to satisfy the requested split sizes.

    Note:
    - This function assumes the existence of a get_all_interacting_pairs() function that
      returns all protein interaction pairs.
    - The function guarantees that no protein will appear in both train and test sets.
    - The split respects the interaction structure, keeping connected proteins together.
    """

    all_pairs = get_all_interacting_pairs()
    rng = np.random.default_rng(seed)

    # Build the graph
    graph = defaultdict(set)
    for protein1, protein2 in all_pairs:
        graph[protein1].add(protein2)
        graph[protein2].add(protein1)

    def get_component(start):
        """
        Get the connected component containing the start protein.

        Parameters:
        start: The starting protein.

        Returns:
        set: A set of proteins in the same connected component as the start protein.
        """
        component = set()
        stack = [start]
        while stack:
            protein = stack.pop()
            if protein not in component:
                component.add(protein)
                stack.extend(graph[protein] - component)
        return component

    # Split graph into connected components
    components = []
    proteins = set(graph.keys())
    while proteins:
        protein = proteins.pop()
        component = get_component(protein)
        components.append(component)
        proteins -= component

    # Shuffle and split components into train and test
    rng.shuffle(components)
    train_proteins = set()
    test_proteins = set()
    for component in components:
        if len(train_proteins) < len(test_proteins):
            train_proteins.update(component)
        else:
            test_proteins.update(component)

    # Assign pairs
    train_pairs = []
    test_pairs = []
    for protein1, protein2 in all_pairs:
        if protein1 in train_proteins and protein2 in train_proteins:
            if len(train_pairs) < num_train_pairs:
                train_pairs.append((protein1, protein2))
        elif protein1 in test_proteins and protein2 in test_proteins:
            if len(test_pairs) < num_test_pairs:
                test_pairs.append((protein1, protein2))

        if len(train_pairs) == num_train_pairs and len(test_pairs) == num_test_pairs:
            break

    if len(train_pairs) < num_train_pairs or len(test_pairs) < num_test_pairs:
        raise ValueError(
            "Not enough interaction pairs to satisfy the requested split sizes."
        )

    return train_pairs, test_pairs


def _test_train_test_split_interaction_pairs(train_pairs, test_pairs):
    """
    Check that there is no protein in train pairs that is also in test pairs, and vice versa.

    Args:
    train_pairs: List of tuples, where each tuple contains two protein identifiers
    test_pairs: List of tuples, where each tuple contains two protein identifiers

    Returns:
    bool: True if the split is valid (no overlapping proteins), False otherwise
    """
    # Create sets of all proteins in train and test pairs
    train_proteins = set()
    for pair in train_pairs:
        train_proteins.update(pair)

    test_proteins = set()
    for pair in test_pairs:
        test_proteins.update(pair)

    # Check for intersection between train and test proteins
    overlap = train_proteins.intersection(test_proteins)
    assert not overlap


def train_test_split_subsample_all_msas(
    input_msa_dir: str,
    output_msa_dir: str,
    num_sequences: int = 1024,
    num_train_pairs: int = 1000,
    num_test_pairs: int = 1000,
    return_full_length_sequences: bool = False,
    seed: int = 42,
    num_processes: int = 1,
):
    """
    Split data into train and testing pairs
    Then subsample the pair msas for each, store the subsampled
    paired and unpaired msas separatedly
    """
    # Split all pairs into train and test
    train_pairs, test_pairs = _train_test_split_interaction_pairs(
        num_train_pairs=num_train_pairs, num_test_pairs=num_test_pairs, seed=seed
    )
    if not os.path.exists(output_msa_dir):
        os.makedirs(output_msa_dir)
    output_dirnames = {
        "train_pair_msa": os.path.join(output_msa_dir, "train", "pair"),
        "train_unpair_msa": os.path.join(output_msa_dir, "train", "unpair"),
        "test_pair_msa": os.path.join(output_msa_dir, "test", "pair"),
        "test_unpair_msa": os.path.join(output_msa_dir, "test", "unpair"),
    }
    for dirname in output_dirnames.values():
        os.makedirs(dirname, exist_ok=True)

    Parallel(n_jobs=num_processes)(
        delayed(_subsample_and_split_pair_msa)(
            pair_msa_path=os.path.join(input_msa_dir, f"{protein1}_{protein2}.fas"),
            num_sequences=num_sequences,
            output_pair_msa_dir=output_dirnames["train_pair_msa"],
            output_unpair_msa_dir=output_dirnames["train_unpair_msa"],
            return_full_length_unaligned_sequences=return_full_length_sequences,
        )
        for protein1, protein2 in tqdm(train_pairs)
    )

    Parallel(n_jobs=num_processes)(
        delayed(_subsample_and_split_pair_msa)(
            pair_msa_path=os.path.join(input_msa_dir, f"{protein1}_{protein2}.fas"),
            num_sequences=num_sequences,
            output_pair_msa_dir=output_dirnames["test_pair_msa"],
            output_unpair_msa_dir=output_dirnames["test_unpair_msa"],
            return_full_length_unaligned_sequences=return_full_length_sequences,
        )
        for protein1, protein2 in tqdm(test_pairs)
    )
    return output_dirnames.values()


def estimate_trees(
    msa_dir: str,
    output_dir: str,
    num_rate_categories: int = 1,
    rate_matrix_path: str = "data/rate_matrices/wag.txt",
    num_processes: int = 1,
):
    """
    Estimate trees for all MSAs in a directory
    """
    # Get the file names of all MSAs
    msa_names = [
        file.split(".txt")[0] for file in os.listdir(msa_dir) if file.endswith(".txt")
    ]

    # Create output directories
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_tree_dir = os.path.join(output_dir, "output_tree_dir")
    output_site_rates_dir = os.path.join(output_dir, "output_site_rates_dir")
    output_likelihood_dir = os.path.join(output_dir, "output_likelihood__dir")
    if not os.path.exists(output_tree_dir):
        os.makedirs(output_tree_dir)
    if not os.path.exists(output_site_rates_dir):
        os.makedirs(output_site_rates_dir)
    if not os.path.exists(output_likelihood_dir):
        os.makedirs(output_likelihood_dir)

    # FUTURE NOTE: since we wish to examine the intermediate outputs
    # we don't used the cached function
    fast_tree_bin = (
        cherryml.phylogeny_estimation._fast_tree._install_fast_tree_and_return_bin_path()
    )
    Parallel(n_jobs=num_processes)(
        delayed(
            cherryml.phylogeny_estimation._fast_tree.run_fast_tree_with_custom_rate_matrix
        )(
            msa_path=os.path.join(msa_dir, f"{msa_name}.txt"),
            family=msa_name,
            rate_matrix_path=rate_matrix_path,
            num_rate_categories=num_rate_categories,
            output_tree_dir=output_tree_dir,
            output_site_rates_dir=output_site_rates_dir,
            output_likelihood_dir=output_likelihood_dir,
            extra_command_line_args="",
            fast_tree_bin=fast_tree_bin,
        )
        for msa_name in tqdm(msa_names)
    )


def main():
    num_processes = 16

    # # Train test split and subsample the MSAs
    # msa_dirs = train_test_split_subsample_all_msas(
    #     input_msa_dir=os.path.join(DATA_DIR, "pair_msas"),
    #     output_msa_dir=os.path.join(DATA_DIR, "cache", "subsampled_msas"),
    #     num_processes=num_processes,
    # )

    # Estimate trees on the unpair MSAs
    msa_parent_dir = os.path.join(DATA_DIR, "cache", "subsampled_msas")
    msa_dirs = {
        "train": os.path.join(msa_parent_dir, "train", "unpair"),
        "test": os.path.join(msa_parent_dir, "test", "unpair"),
    }
    tree_parent_dir = os.path.join(DATA_DIR, "cache", "fast_tree")
    tree_dirs = {
        "train": os.path.join(tree_parent_dir, "train"),
        "test": os.path.join(tree_parent_dir, "test"),
    }
    for name, msa_dir in msa_dirs.items():
        estimate_trees(
            msa_dir=msa_dir, output_dir=tree_dirs[name], num_processes=num_processes
        )


if __name__ == "__main__":
    main()
