import pandas as pd
import numpy as np
import os

from ppievo.io import (
    read_rate_matrix,
    read_transitions,
    write_transitions_log_likelihood_per_site,
    filter_transitions_by_protein_and_length,
)
from ppievo.utils import matrix_exponential_reversible, MAIN_DIR


def evaluate_wag_per_site_log_likelihood_for_pair(
    transitions_dir: list[tuple[str, str, float]],
    pair_name: str,
    which_protein: str = "y",
    rate_matrix: pd.DataFrame = read_rate_matrix(
        os.path.join(MAIN_DIR, "data/rate_matrices/wag_gap.txt")
    ),
    output_dir: str | None = None,
) -> list[list[float]]:
    """
    Compute the per-site log-likelihood of the given transitions under the WAG model.

    It is assumed that the rate_matrix represents a reversible model.

    The log-likelihood under the WAG model is given by:
    P(y_i | x_i, t) = log( exp(rate_matrix * t)[x[i], y[i]] )

    Args:
        transitions: The transitions for which to compute the log-likelihood.
        rate_matrix: The rate matrix parameter of the WAG model.
    Returns:
        lls: The per-site log-likelihood of each transition.
    """
    transitions = read_transitions(os.path.join(transitions_dir, f"{pair_name}.txt"))
    # Filter transitions to the relevant protein and
    # filter out the ones where one of the sequences are too long filtered out
    # to be consistent with other models
    transitions = filter_transitions_by_protein_and_length(
        transitions=transitions, which_protein=which_protein
    )

    matrix_exponentials = matrix_exponential_reversible(
        rate_matrix=rate_matrix.to_numpy(),
        exponents=[t for (seq1, seq2, t) in transitions],
    )
    lls = []
    for i, (seq1, seq2, t) in enumerate(transitions):
        if len(seq1) != len(seq2):
            raise ValueError(
                f"Transition has two sequences of different lengths: {seq1}, {seq2}."
            )
        mexp_df = pd.DataFrame(
            matrix_exponentials[i, :, :],
            index=rate_matrix.index,
            columns=rate_matrix.columns,
        )
        lls.append([np.log(mexp_df.at[s1_i, s2_i]) for s1_i, s2_i in zip(seq1, seq2)])

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_fpath = os.path.join(output_dir, f"{pair_name}.txt")
        write_transitions_log_likelihood_per_site(lls, output_fpath)
    return lls
