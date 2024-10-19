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

import warnings
import hashlib
import os
import re
import pandas as pd
from typing import Optional
import cherryml.phylogeny_estimation
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import cherryml
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from ppievo.io import (
    write_msa,
    read_dict,
    Tree,
    get_leaf_distance,
    read_msa_index_with_genome_id,
    read_tree,
    write_transitions,
    construct_pair_msa,
)
from ppievo.utils import (
    DATA_DIR,
    amino_acids,
    gap_character,
    seq_identity,
    calculate_distance_matrix,
)


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


def construct_all_pair_msa(
    output_dir=os.path.join(DATA_DIR, "pair_msa"),
    msa_dir=os.path.join(DATA_DIR, "humanPPI_MSA/MSA"),
):
    """
    Construct pair MSAs for all interacting pairs
    """
    all_pairs = get_all_interacting_pairs()
    print(f"Writing all pair MSAs for {len(all_pairs)} pairs to {output_dir}")
    Parallel(n_jobs=-1)(
        delayed(construct_pair_msa)(
            protein1=protein1, protein2=protein2, output_dir=output_dir, msa_dir=msa_dir
        )
        for protein1, protein2 in tqdm(all_pairs)
    )


def _read_pair_msa(
    pair_msa_path: int,
    return_full_length_unaligned_sequences: bool = False,
):
    """
    Read in a pair MSA and return list of (id, sequence) tuples for
    1) concatenated sequence of protein 1 and 2
    2) sequence of protein 1
    3) sequence of protein 2
    """
    if not os.path.exists(pair_msa_path):
        raise FileNotFoundError(f"MSA file {pair_msa_path} does not exist!")

    # Read MSA
    # type: list[tuple[str, str]]
    pair_msa, unpair_msa1, unpair_msa2 = [], [], []
    # Keep track of unique sequences and only include unique sequences
    pair_seqs_unique = []
    with open(pair_msa_path) as file:
        lines = list(file)
        n_lines = len(lines)
        # First line contain the length of the two alignments, indicates where the two MSAs were concatenated
        protein1_len = int(lines[0].split(",")[0].split("#")[1])
        # protein2_len = int(file_header.split(',')[1].split('\t')[0])
        for i in range(1, n_lines, 2):
            if not lines[i][0] == ">":
                print(f"Protein ID line should start with '>'")
                print(f"File: {pair_msa_path}, skipping line {i}: {lines[i]}")
                continue
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

            # Add record to each MSA if the pair sequence doesn't already exist
            if pair_seq not in pair_seqs_unique:
                pair_seqs_unique.append(pair_seq)
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

    return pair_msa, unpair_msa1, unpair_msa2


def _cluster_and_sample(
    pair_msa: list[tuple[str, str]],
    unpair_msa1: list[tuple[str, str]],
    unpair_msa2: list[tuple[str, str]],
    max_num_sequences: int,
    sequence_id_filtering: float,
    sequence_id_filter_on_pair: bool,
    rng: np.random.Generator,
) -> list[int]:
    """
    Cluster sequences in a pair MSA then subsample the sequences
    If sequence_id_filter_on_pair is True, input should be the pair MSA
    and the clustering is based on the sequence identity of concatenated sequences
    Else, input should be two unpair MSAs and the clustering is done by
    separately computing the sequence identity in each MSA, then average the
    sequence identity
    """

    if sequence_id_filter_on_pair:
        seqs = [seq for _, seq in pair_msa]
        distance_matrix = calculate_distance_matrix(seqs)
    else:
        assert len(unpair_msa1) == len(
            unpair_msa2
        ), "msa1 and msa2 should have the same number of sequences"
        seqs1 = [seq for _, seq in unpair_msa1]
        seqs2 = [seq for _, seq in unpair_msa2]
        distance_matrix1 = calculate_distance_matrix(seqs1)
        distance_matrix2 = calculate_distance_matrix(seqs2)
        distance_matrix = (distance_matrix1 + distance_matrix2) / 2

    nseqs = len(pair_msa)
    # Two sequences are connected if their distance < 1 - seq id threshold
    adjacency_matrix = (distance_matrix < 1 - sequence_id_filtering).astype(int)
    # Convert to sparse matrix for efficiency
    sparse_graph = csr_matrix(adjacency_matrix)
    # Perform connected components analysis
    n_components, clusters = connected_components(
        csgraph=sparse_graph, directed=False, return_labels=True
    )

    # Always include the query sequence
    sampled_indices = [0]
    cluster_representatives = {clusters[0]: [0]}
    # Shuffle the remaining indices
    candidate_indices = list(range(1, nseqs))
    rng.shuffle(candidate_indices)

    # First pass: select one representative from each cluster
    for i in candidate_indices:
        if clusters[i] not in cluster_representatives:
            cluster_representatives[clusters[i]] = [i]
            sampled_indices.append(i)
        else:
            cluster_representatives[clusters[i]].append(i)

        if len(sampled_indices) >= max_num_sequences:
            break

    # If we still need more sequences, sample from the clusters
    if len(sampled_indices) < max_num_sequences:
        cluster_sizes = [
            len(representatives) for representatives in cluster_representatives.values()
        ]

        # Calculate sampling probabilities based on cluster sizes
        sampling_probs = np.array(cluster_sizes) / sum(cluster_sizes)

        while len(sampled_indices) < max_num_sequences:
            # Choose a cluster based on its size
            chosen_cluster = rng.choice(
                list(cluster_representatives.keys()), p=sampling_probs
            )

            # Choose a random sequence from the chosen cluster that hasn't been sampled yet
            available_indices = [
                idx
                for idx in cluster_representatives[chosen_cluster]
                if idx not in sampled_indices
            ]
            if available_indices:
                chosen_index = rng.choice(available_indices)
                sampled_indices.append(chosen_index)

            # If this cluster is exhausted, set its probability to 0 and renormalize
            if not available_indices:
                sampling_probs[
                    list(cluster_representatives.keys()).index(chosen_cluster)
                ] = 0
                sampling_probs = sampling_probs / np.sum(sampling_probs)

    return sampled_indices


def _greedy_sample(
    pair_msa: list[tuple[str, str]],
    unpair_msa1: list[tuple[str, str]],
    unpair_msa2: list[tuple[str, str]],
    max_num_sequences: int,
    sequence_id_filtering: float,
    sequence_id_filter_on_pair: bool,
    rng: np.random.Generator,
) -> list[int]:
    nseqs = len(pair_msa)
    # Subsample and ensure the concatenated sequence have sequence id < sequence_id_filtering
    filtered_indices = [0]  # Always include the query sequence
    # Shuffle indices to randomize selection
    candidate_indices = list(range(1, nseqs))
    rng.shuffle(candidate_indices)
    for idx in candidate_indices:
        if len(filtered_indices) >= max_num_sequences:
            break
        if sequence_id_filter_on_pair:
            # Perform sequence identity filtering on pair sequence
            candidate_seq = pair_msa[idx][1]
            # Check identity with query and all previously selected sequences
            pass_filter = all(
                [
                    seq_identity(candidate_seq, pair_msa[i][1]) <= sequence_id_filtering
                    for i in filtered_indices
                ]
            )
            if pass_filter:
                filtered_indices.append(idx)
        else:
            # Perform sequence identity filtering on unpair sequences
            candidate_seq1 = unpair_msa1[idx][1]
            candidate_seq2 = unpair_msa2[idx][1]
            pass_filter_seq1 = all(
                [
                    seq_identity(candidate_seq1, unpair_msa1[i][1])
                    <= sequence_id_filtering
                    for i in filtered_indices
                ]
            )
            pass_filter_seq2 = all(
                [
                    seq_identity(candidate_seq2, unpair_msa2[i][1])
                    <= sequence_id_filtering
                    for i in filtered_indices
                ]
            )
            if pass_filter_seq1 and pass_filter_seq2:
                filtered_indices.append(idx)

    return filtered_indices


def _subsample_and_split_pair_msa(
    pair_msa_path: str,
    max_num_sequences: Optional[int],
    output_pair_msa_dir: str,
    output_unpair_msa_dir: str,
    return_full_length_unaligned_sequences: bool = False,
    sequence_id_filtering: float = 1.0,
    sequence_id_filter_on_pair: bool = True,
):
    """
    Subsample sequences in an paired MSA, also split the subsampled pair MSA
    into the unpaired MSAs and store these separately

    If `return_full_length_unaligned_sequences=True`, then the original,
    full-length, unaligned sequences will be returned. (In particular,
    insertions wrt the original sequence will be retained and uppercased,
    and gaps will be removed from all sequences.)

    If `sequence_id_filter_on_pair=True`, then the sequence identity filtering
    if done on the pair sequenece, otherwise it is done separately for each protein
    and only pair that passes the filter for each protein is kept
    """
    pair_msa, unpair_msa1, unpair_msa2 = _read_pair_msa(
        pair_msa_path=pair_msa_path,
        return_full_length_unaligned_sequences=return_full_length_unaligned_sequences,
    )
    # The ID of the first entry of pair msa contains the name of the two proteins in the pairs
    protein1 = pair_msa[0][0].split("\t")[0]
    protein2 = pair_msa[0][0].split("\t")[1]
    pair_name = f"{protein1}_{protein2}"
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

    if max_num_sequences is not None and sequence_id_filtering < 1:
        # Cluster sequences based on sequence identity then subsample sequences
        sampled_indices = _greedy_sample(
            pair_msa=pair_msa,
            unpair_msa1=unpair_msa1,
            unpair_msa2=unpair_msa2,
            max_num_sequences=max_num_sequences,
            sequence_id_filtering=sequence_id_filtering,
            sequence_id_filter_on_pair=sequence_id_filter_on_pair,
            rng=rng,
        )
        seqs_to_keep = sorted(sampled_indices)
        if len(seqs_to_keep) != max_num_sequences:
            print(
                f"Only have {len(seqs_to_keep)} for pair MSA={pair_msa_path} with seq id filter={sequence_id_filtering}"
            )
    elif max_num_sequences is not None:
        max_seqs = min(nseqs, max_num_sequences)
        # Ensure query sequence is always included
        seqs_to_keep = [0] + list(
            rng.choice(range(1, nseqs, 1), size=max_seqs - 1, replace=False)
        )
        seqs_to_keep = sorted(seqs_to_keep)
    else:
        seqs_to_keep = range(nseqs)

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
    num_train_pairs: int = 1000,
    num_test_pairs: int = 1000,
    seed: int = 42,
    min_num_sequences: Optional[int] = 100,
    max_vertebrates_proportion: Optional[float] = None,
    pair_msa_num_sequences_path: Optional[str] = os.path.join(
        DATA_DIR, "cache/metadata/pair_msa_num_sequences.json"
    ),
    pair_msa_species_stats_path: Optional[str] = os.path.join(
        DATA_DIR, "cache/metadata/pair_msa_species_stats.csv"
    ),
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
    min_num_sequences (int): Minimum number of sequences in the pair MSAs. Default is 100
    max_vertebrates_proportion (float): Maximum proportion of sequences in the
        pair MSAs that are vertebrates. Default is None (no filtering)
        Smaller values means more diverse but potentially less sequences

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

    # Filter out any pair whose pair MSA is too shallow
    if min_num_sequences is not None:
        pair_msa_num_sequences = read_dict(pair_msa_num_sequences_path)
        all_pairs = [
            (protein1, protein2)
            for protein1, protein2 in all_pairs
            if pair_msa_num_sequences[f"{protein1}_{protein2}"] >= min_num_sequences
        ]
        print(
            f"{len(all_pairs)} protein pairs left after filtering out pair MSAs with < {min_num_sequences} sequences"
        )

    if max_vertebrates_proportion is not None:
        pair_msa_species_stats = pd.read_csv(pair_msa_species_stats_path)
        pair_msa_species_stats = pair_msa_species_stats[
            ["pair_name", "vertebrate_proportion"]
        ].set_index("pair_name")
        all_pairs = [
            (protein1, protein2)
            for protein1, protein2 in all_pairs
            if pair_msa_species_stats.loc[f"{protein1}_{protein2}"].iloc[0]
            <= max_vertebrates_proportion
        ]
        print(
            f"{len(all_pairs)} protein pairs left after filtering out pair MSAs with > \
            {max_vertebrates_proportion * 100}% vertebrate sequences"
        )

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

    _test_train_test_split_interaction_pairs(train_pairs, test_pairs)
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
    max_num_sequences: int = 1024,
    min_num_sequences: int = 500,
    num_train_pairs: int = 1000,
    num_test_pairs: int = 1000,
    return_full_length_sequences: bool = False,
    seed: int = 42,
    num_processes: int = 1,
    sequence_id_filtering: float = 1.0,
    sequence_id_filter_on_pair: bool = True,
    max_vertebrates_proportion: Optional[float] = None,
):
    """
    Split data into train and testing pairs
    Then subsample the pair msas for each, store the subsampled
    paired and unpaired msas separatedly
    """
    # Split all pairs into train and test
    train_pairs, test_pairs = _train_test_split_interaction_pairs(
        num_train_pairs=num_train_pairs,
        num_test_pairs=num_test_pairs,
        seed=seed,
        min_num_sequences=min_num_sequences,
        max_vertebrates_proportion=max_vertebrates_proportion,
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
            max_num_sequences=max_num_sequences,
            output_pair_msa_dir=output_dirnames["train_pair_msa"],
            output_unpair_msa_dir=output_dirnames["train_unpair_msa"],
            return_full_length_unaligned_sequences=return_full_length_sequences,
            sequence_id_filtering=sequence_id_filtering,
            sequence_id_filter_on_pair=sequence_id_filter_on_pair,
        )
        for protein1, protein2 in tqdm(train_pairs)
    )

    # Parallel(n_jobs=num_processes)(
    #     delayed(_subsample_and_split_pair_msa)(
    #         pair_msa_path=os.path.join(input_msa_dir, f"{protein1}_{protein2}.fas"),
    #         max_num_sequences=max_num_sequences,
    #         output_pair_msa_dir=output_dirnames["test_pair_msa"],
    #         output_unpair_msa_dir=output_dirnames["test_unpair_msa"],
    #         return_full_length_unaligned_sequences=return_full_length_sequences,
    #         sequence_id_filtering=sequence_id_filtering,
    #         sequence_id_filter_on_pair=sequence_id_filter_on_pair,
    #     )
    #     for protein1, protein2 in tqdm(test_pairs)
    # )
    return output_dirnames


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
    output_likelihood_dir = os.path.join(output_dir, "output_likelihood_dir")
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


def extract_pair_transitions_from_gene_trees(
    tree_x: Tree,
    tree_y: Tree,
    msa_x: dict[str, str],
    msa_y: dict[str, str],
    include_gaps: bool = True,
) -> list[tuple[str, str, float]]:
    """
    Extract transitions of a protein pairs from the gene trees.

    Suppose we have two proteins x and y, we pick cherries from the tree of y
    then extract the corresponding transitions in the tree of x
    The transitions come from all (recursively picked) cherries in the tree of y,
    and are bidirectional, so that if (y1, y2, t) is a transition, then also is
    (y2, y1, t).

    Whether gaps (assumed to be "-") are included or not is determined by
    `include_gaps`.
    """
    total_pairs = []
    transitions = []

    def dfs(node) -> Optional[tuple[int, float]]:
        """
        Pair up leaves under me.

        Return a single unpaired leaf and its distance, it such exists.
        """
        if tree_y.is_leaf(node):
            return (node, 0.0)
        unmatched_leaves_under = []
        distances_under = []
        for child, branch_length in tree_y.children(node):
            maybe_unmatched_leaf, maybe_distance = dfs(child)
            if maybe_unmatched_leaf is not None:
                assert maybe_distance is not None
                unmatched_leaves_under.append(maybe_unmatched_leaf)
                distances_under.append(maybe_distance + branch_length)
        assert len(unmatched_leaves_under) == len(distances_under)
        index = 0

        while index + 1 <= len(unmatched_leaves_under) - 1:
            total_pairs.append(1)
            (leaf_1, branch_length_1), (leaf_2, branch_length_2) = (
                (unmatched_leaves_under[index], distances_under[index]),
                (
                    unmatched_leaves_under[index + 1],
                    distances_under[index + 1],
                ),
            )

            # Extract y1, y2 from protein y
            leaf_seq_y1, leaf_seq_y2 = msa_y[leaf_1], msa_y[leaf_2]
            distance_y = branch_length_1 + branch_length_2
            # Extract x1, x2 from the protein x corresponding to the same leaves
            leaf_seq_x1, leaf_seq_x2 = msa_x[leaf_1], msa_x[leaf_2]
            # Estimate the distance between y1 and y2 in the tree of y
            distance_x = get_leaf_distance(tree=tree_x, leaf1=leaf_1, leaf2=leaf_2)
            transitions.append(
                (
                    leaf_seq_x1,
                    leaf_seq_y1,
                    leaf_seq_x2,
                    leaf_seq_y2,
                    distance_x,
                    distance_y,
                )
            )  # Note: gaps will be removed later
            transitions.append(
                (
                    leaf_seq_x2,
                    leaf_seq_y2,
                    leaf_seq_x1,
                    leaf_seq_y1,
                    distance_x,
                    distance_y,
                )
            )  # Note: gaps will be removed later
            index += 2
        if len(unmatched_leaves_under) % 2 == 0:
            return (None, None)
        else:
            return (unmatched_leaves_under[-1], distances_under[-1])

    dfs(tree_x.root())
    if not include_gaps:
        # Remove gaps from all sequences.
        def remove_gaps(seq: str) -> str:
            return "".join([char for char in seq if char != "-"])

        transitions = [
            (remove_gaps(x1), remove_gaps(y1), remove_gaps(x2), remove_gaps(y2), tx, ty)
            for (x1, y1, x2, y2, tx, ty) in transitions
        ]
    assert len(total_pairs) == int(len(tree_x.leaves()) / 2)
    assert 2 * len(total_pairs) == len(transitions)
    return transitions


def extract_transitions(
    msa_dir: str,
    tree_dir: str,
    pair_names: list[str],
    num_processes: int,
    include_gaps: bool = True,
    output_transitions_dir: Optional[str] = None,
):

    def _extract_transition_for_pair(pair_name: str):
        protein1, protein2 = pair_name.split("_")

        # Read in both unpair MSA for a protein pair
        msa1 = read_msa_index_with_genome_id(
            os.path.join(msa_dir, f"{pair_name}-{protein1}.txt")
        )
        msa2 = read_msa_index_with_genome_id(
            os.path.join(msa_dir, f"{pair_name}-{protein2}.txt")
        )
        # Read the tree built for one of them
        tree1 = read_tree(os.path.join(tree_dir, f"{pair_name}-{protein1}.txt"))
        tree2 = read_tree(os.path.join(tree_dir, f"{pair_name}-{protein2}.txt"))
        # Extract transitions
        transitions = extract_pair_transitions_from_gene_trees(
            tree_x=tree1,
            tree_y=tree2,
            msa_x=msa1,
            msa_y=msa2,
            include_gaps=include_gaps,
        )
        # list[(x1, y1, x2, y2, tx, ty)]
        transitions = sorted(
            transitions,
            key=lambda transition: (
                transition[5],
                transition[4],
                transition[3],
                transition[2],
                transition[1],
                transition[0],
            ),
        )
        transitions_path = os.path.join(output_transitions_dir, f"{pair_name}.txt")
        write_transitions(transitions=transitions, transitions_path=transitions_path)

    Parallel(n_jobs=num_processes)(
        delayed(_extract_transition_for_pair)(pair_name=pair_name)
        for pair_name in tqdm(pair_names)
    )


def main():
    # Specify script params
    num_processes = 16
    seq_id = 0.9
    seq_id_filter_on_pair = True
    max_vertebrates_proportion = 0.7
    suffix = (
        f"seq_id_{int(seq_id * 100)}_pair-vert_{int(max_vertebrates_proportion * 100)}"
    )
    warnings.simplefilter(action="ignore", category=FutureWarning)

    msa_parent_dir = os.path.join(DATA_DIR, "cache", f"subsampled_msas-{suffix}")
    tree_parent_dir = os.path.join(DATA_DIR, "cache", f"fast_tree-{suffix}")
    transition_parent_dir = os.path.join(DATA_DIR, "cache", f"transitions-{suffix}")
    msa_dirs = {
        "train": os.path.join(msa_parent_dir, "train", "unpair"),
        "test": os.path.join(msa_parent_dir, "test", "unpair"),
    }
    tree_dirs = {
        "train": os.path.join(tree_parent_dir, "train"),
        "test": os.path.join(tree_parent_dir, "test"),
    }
    transition_dirs = {
        "train": os.path.join(transition_parent_dir, "train"),
        "test": os.path.join(transition_parent_dir, "test"),
    }

    # Train test split and subsample the MSAs
    print("Train test split and subsampling MSAs...")
    train_test_split_subsample_all_msas(
        input_msa_dir=os.path.join(DATA_DIR, "pair_msa"),
        output_msa_dir=msa_parent_dir,
        max_num_sequences=1024,
        min_num_sequences=500,
        max_vertebrates_proportion=max_vertebrates_proportion,
        num_train_pairs=1000,
        num_test_pairs=1000,
        num_processes=num_processes,
        sequence_id_filtering=seq_id,
        sequence_id_filter_on_pair=seq_id_filter_on_pair,
    )

    # Estimate trees on the unpair MSAs
    print("Estimating trees...")
    estimate_trees(
        msa_dir=msa_dirs["train"],
        output_dir=tree_dirs["train"],
        num_processes=num_processes,
    )
    # estimate_trees(
    #     msa_dir=msa_dirs["test"],
    #     output_dir=tree_dirs["test"],
    #     num_processes=num_processes,
    # )

    # Extract paired transitions from trees
    print("Extracting transitions...")
    # Read pair names from the subsampled pair MSA dirs
    train_pair_names = [
        file.split(".txt")[0]
        for file in os.listdir(os.path.join(msa_parent_dir, "train", "pair"))
        if file.endswith(".txt")
    ]
    extract_transitions(
        msa_dir=msa_dirs["train"],
        tree_dir=os.path.join(tree_dirs["train"], "output_tree_dir"),
        pair_names=train_pair_names,
        num_processes=num_processes,
        include_gaps=True,
        output_transitions_dir=transition_dirs["train"],
    )
    # test_pair_names = [
    #     file.split(".txt")[0]
    #     for file in os.listdir(os.path.join(msa_parent_dir, "test", "pair"))
    #     if file.endswith(".txt")
    # ]
    # extract_transitions(
    #     msa_dir=msa_dirs["test"],
    #     tree_dir=os.path.join(tree_dirs["test"], "output_tree_dir"),
    #     pair_names=test_pair_names,
    #     num_processes=num_processes,
    #     include_gaps=True,
    #     output_transitions_dir=transition_dirs["test"],
    # )


if __name__ == "__main__":
    main()
