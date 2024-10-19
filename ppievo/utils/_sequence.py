import numpy as np


def seq_identity(seq1: str, seq2: str) -> float:
    """Compute sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
    return matches / len(seq1)


def calculate_distance_matrix(seqs):
    """
    Compute normalized pairwise hamming distance of all sequences
    Gaps are included
    Use array encoding for faster computational
    """
    arrays = np.array([np.array([ord(c) for c in seq], dtype=np.int8) for seq in seqs])
    seq_len = arrays.shape[1]
    # Shape (B, B)
    mismatches = (arrays[:, None, :] != arrays[None, :, :]).sum(axis=2)
    distance_matrix = mismatches / seq_len
    return distance_matrix


def _test_calculate_distance_matrix():
    # Test sequences
    seqs = [
        "ATCGA",
        "ATCGA",
        "TTTTT",
        "ATC-A",
        "AC-GT",
    ]

    # Calculate the distance matrix
    distance_matrix = calculate_distance_matrix(seqs)

    # Expected results
    expected = np.array(
        [
            [0.0, 0.0, 0.8, 0.2, 0.6],
            [0.0, 0.0, 0.8, 0.2, 0.6],
            [0.8, 0.8, 0.0, 0.8, 0.8],
            [0.2, 0.2, 0.8, 0.0, 0.8],
            [0.6, 0.6, 0.8, 0.8, 0.0],
        ]
    )
    assert np.allclose(
        distance_matrix, expected
    ), "Calculated distances do not match expected values"


_test_calculate_distance_matrix()
