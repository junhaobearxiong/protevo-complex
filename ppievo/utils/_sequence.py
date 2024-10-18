def seq_identity(seq1: str, seq2: str) -> float:
    """Quickly compute sequence identity between two sequences."""
    if len(seq1) != len(seq2):
        raise ValueError("Sequences must be of equal length")
    matches = sum(c1 == c2 for c1, c2 in zip(seq1, seq2))
    return matches / len(seq1)
