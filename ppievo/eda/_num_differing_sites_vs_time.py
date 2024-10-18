"""
Plot time vs number of mutations.
"""

import os
from typing import List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from cherryml import markov_chain

from ppievo.utils import (
    get_quantization_points_from_geometric_grid,
    get_quantile_idx,
    gap_character,
    matrix_exponential_reversible,
)
from ppievo.io import read_transitions


PLT_SAVEFIG_KWARGS = {
    "bbox_inches": "tight",
    "dpi": 300,
}


def plot_num_differing_sites_vs_time(
    transitions_dir: str,
    pair_names: Optional[List[str]] = None,
    quantiles: List[Union[str, float]] = get_quantization_points_from_geometric_grid(),
    title: Optional[str] = "Probability of differing site vs Time bucket",
    output_dir: Optional[str] = None,  # Will plt.show() when equals ""
    rate_matrix: Optional[np.array] = None,
    exclude_mutations_involving_gaps: bool = True,
) -> None:
    """
    Plot the number of mutations as a function of time (bucket).
    """

    def _count_num_mutations_local(seq1: str, seq2: str, exclude_gaps: bool = True):
        if exclude_gaps:
            num_mutations_local = sum(
                [
                    s1_i != s2_i
                    for (s1_i, s2_i) in zip(seq1, seq2)
                    if s1_i != gap_character and s2_i != gap_character
                ]
            )
        else:
            num_mutations_local = sum(
                [s1_i != s2_i for (s1_i, s2_i) in zip(seq1, seq2)]
            )
        return num_mutations_local

    def _count_num_non_mutations_local(seq1: str, seq2: str, exclude_gaps: bool = True):
        if exclude_gaps:
            num_non_mutations_local = sum(
                [
                    s1_i == s2_i
                    for (s1_i, s2_i) in zip(seq1, seq2)
                    if s1_i != gap_character and s2_i != gap_character
                ]
            )
        else:
            num_non_mutations_local = len(seq1) - _count_num_mutations_local(
                seq1, seq2, exclude_gaps=False
            )
        return num_non_mutations_local

    if pair_names is None:
        pair_names = [
            file.split(".txt")[0]
            for file in os.listdir(transitions_dir)
            if file.endswith(".txt")
        ]
    quantiles = np.array(sorted([float(x) for x in quantiles]))
    mutations_x = np.zeros((len(quantiles), len(quantiles)))
    non_mutations_x = np.zeros((len(quantiles), len(quantiles)))
    mutations_y = np.zeros((len(quantiles), len(quantiles)))
    non_mutations_y = np.zeros((len(quantiles), len(quantiles)))
    for pair_name in tqdm(pair_names):
        transitions = read_transitions(
            os.path.join(transitions_dir, pair_name + ".txt")
        )
        for x1, y1, x2, y2, tx, ty in transitions:
            qidx_x = get_quantile_idx(quantiles=quantiles, t=tx)
            qidx_y = get_quantile_idx(quantiles=quantiles, t=ty)

            num_mutations_local_x = _count_num_mutations_local(
                x1, x2, exclude_gaps=exclude_mutations_involving_gaps
            )
            num_non_mutations_local_x = _count_num_non_mutations_local(
                x1, x2, exclude_gaps=exclude_mutations_involving_gaps
            )
            num_mutations_local_y = _count_num_mutations_local(
                y1, y2, exclude_gaps=exclude_mutations_involving_gaps
            )
            num_non_mutations_local_y = _count_num_non_mutations_local(
                y1, y2, exclude_gaps=exclude_mutations_involving_gaps
            )
            mutations_x[qidx_x, qidx_y] += num_mutations_local_x
            non_mutations_x[qidx_x, qidx_y] += num_non_mutations_local_x
            mutations_y[qidx_x, qidx_y] += num_mutations_local_y
            non_mutations_y[qidx_x, qidx_y] += num_non_mutations_local_y

    # Calculate mutation probabilities
    mutation_probability_x = np.where(
        mutations_x + non_mutations_x > 0,
        mutations_x / (mutations_x + non_mutations_x),
        np.nan,
    )
    mutation_probability_y = np.where(
        mutations_y + non_mutations_y > 0,
        mutations_y / (mutations_y + non_mutations_y),
        np.nan,
    )

    # Create the plot
    fig, axs = plt.subplots(2, 2, figsize=(20, 16))
    plt.suptitle(title, fontsize=16)
    quantiles_labels = [f"{q:.1e}" for q in quantiles]

    # Helper function to plot bivariate histogram
    def plot_heatmap(ax, data, x_label, y_label, title):
        df = pd.DataFrame(data, index=quantiles_labels, columns=quantiles_labels)
        sns.heatmap(
            data=df, cmap="YlGnBu", cbar=True, xticklabels=10, yticklabels=10, ax=ax
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

    # Plot empirical probabilities
    plot_heatmap(
        axs[0, 0],
        mutation_probability_x,
        "Time Y (bucket)",
        "Time X (bucket)",
        "Empirical Mutation Probability of X",
    )
    plot_heatmap(
        axs[0, 1],
        mutation_probability_y,
        "Time Y (bucket)",
        "Time X (bucket)",
        "Empirical Mutation Probability of Y",
    )

    # Calculate and plot WAG probabilities
    if rate_matrix is not None:
        mexp_array = matrix_exponential_reversible(
            rate_matrix=rate_matrix,
            exponents=list(quantiles),
        )
        pi = markov_chain.compute_stationary_distribution(rate_matrix=rate_matrix)

        theoretical_mut_probs = np.array(
            [
                sum(
                    [
                        pi[j] * (1.0 - mexp_array[i, j, j])
                        for j in range(rate_matrix.shape[0])
                    ]
                )
                for i in range(len(quantiles))
            ]
        )

        # Create 2D theoretical probability arrays
        theoretical_probs_2d = np.outer(theoretical_mut_probs, theoretical_mut_probs)

        # Plot WAG probabilities
        plot_heatmap(
            axs[1, 0],
            theoretical_probs_2d,
            "Time X (bucket)",
            "Time Y (bucket)",
            "WAG Mutation Probability - X dimension",
        )
        plot_heatmap(
            axs[1, 1],
            theoretical_probs_2d,
            "Time X (bucket)",
            "Time Y (bucket)",
            "WAG Mutation Probability - Y dimension",
        )

    plt.tight_layout()
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(
            os.path.join(output_dir, "num_diff_sites_vs_time.png"), **PLT_SAVEFIG_KWARGS
        )
        plt.close()
    else:
        plt.show()
    return {
        "mutations_x": mutations_x,
        "non_mutations_x": non_mutations_x,
        "mutations_y": mutations_y,
        "non_mutations_y": non_mutations_y,
    }
