"""
Aggregate the per-site log likelihood to be mean over each time bucket
"""

from typing import Union, Callable, Optional
import numpy as np
import os

from ppievo.io import (
    read_transitions_log_likelihood_per_site,
    read_transitions,
    filter_transitions_by_protein_and_length,
)
from ppievo.utils import get_quantile_idx, TIME_BINS


def mean_per_site_log_likelihood_by_time(
    transitions_dir: str,
    log_likelihood_dir: str,
    pair_names: list[str],
    which_protein: str = "y",
    quantization_points: np.ndarray = TIME_BINS,
    filter_func=None,
    filter_gap=False,
    max_length: int = 1022,
):
    # Define gap filter function as a default is filter_func is not provided
    # but filter_gap is true
    if filter_func is None and filter_gap:
        filter_func = lambda s1, s2, t: [i for i in range(len(s2)) if s2[i] != "-"]

    quantization_points = np.array(sorted([float(x) for x in quantization_points]))
    total_sites = [0] * len(quantization_points)
    total_ll = [0.0] * len(quantization_points)
    for pair_name in pair_names:
        try:
            transitions = read_transitions(
                os.path.join(transitions_dir, f"{pair_name}.txt")
            )
            lls = read_transitions_log_likelihood_per_site(
                os.path.join(log_likelihood_dir, f"{pair_name}.txt")
            )
        except:
            print(f"Cannot read transitions or lls for {pair_name}, skip..")
            continue

        # lls have all transitions where one of the sequences are too long filtered out
        transitions = filter_transitions_by_protein_and_length(
            transitions=transitions, which_protein=which_protein, max_length=max_length
        )
        if len(transitions) != len(lls):
            # raise ValueError(
            #     "List of transitions and of lls "
            #     "for pair should have the same length. The lengths are: "
            #     f"{len(transitions)} and {len(lls)} "
            #     "respectively."
            # )
            print(f"Skip {pair_name}, transitions and ll shape do not match")
            continue

        for (seq1, seq2, t), ll in zip(transitions, lls):
            q_idx = get_quantile_idx(quantization_points, t)
            # Return a list of indices that are kept after filtering
            # typically this would return the indices of the nongap positions of seq2
            if filter_func is not None:
                filtered_indices = filter_func(seq1, seq2, t)
            else:
                filtered_indices = np.arange(len(ll))
            filtered_log_likelihoods = [ll[i] for i in filtered_indices]
            total_sites[q_idx] += len(filtered_log_likelihoods)
            total_ll[q_idx] += np.sum(filtered_log_likelihoods)

    nonzero_indices = [i for i in range(len(quantization_points)) if total_sites[i] > 0]
    quantization_points = [quantization_points[i] for i in nonzero_indices]
    total_sites = [total_sites[i] for i in nonzero_indices]
    total_ll = [total_ll[i] for i in nonzero_indices]
    mean_per_site_ll_per_t = np.array(
        [total_ll[i] / total_sites[i] for i in range(len(total_sites))]
    )
    return mean_per_site_ll_per_t, quantization_points
