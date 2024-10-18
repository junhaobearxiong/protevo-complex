import numpy as np
from cherryml import markov_chain


def matrix_exponential_reversible(
    rate_matrix: np.array,
    exponents: list[float],
) -> np.array:
    """
    Compute matrix exponential (batched).

    Args:
        rate_matrix: Rate matrix for which to compute the matrix exponential
        exponents: List of exponents.
    Returns:
        3D tensor where res[:, i, i] contains exp(rate_matrix * exponents[i])
    """
    return markov_chain.matrix_exponential_reversible(
        exponents=exponents,
        fact=markov_chain.FactorizedReversibleModel(rate_matrix),
        device="cpu",
    )
