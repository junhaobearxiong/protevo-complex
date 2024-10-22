import torch
import esm


def _create_sequence_mask(
    attn_mask_in_length: torch.Tensor, sequence_idx: int
) -> torch.Tensor:
    """
    Generate a boolean mask for a specific sequence index across all batch elements.

    Args:
        attn_mask_in_length (torch.Tensor): Tensor of shape (B, L) containing nonzero elements
            that indicate the lengths of sequences in each batch element
        sequence_idx (int): The index of the sequence for which to generate the mask

    Returns:
        torch.Tensor: Boolean tensor of shape (B, L) where True indicates positions
            belonging to the sequence_idx'th sequence in each batch element

    Example:
        >>> attn_mask = torch.tensor([
        ...     [2, 4, 0, 0, 0, 0],
        ...     [3, 2, 0, 0, 0, 0]
        ... ])
        >>> get_sequence_mask(attn_mask, 0)
        tensor([[ True,  True, False, False, False, False],
                [ True,  True,  True, False, False, False]])
        >>> get_sequence_mask(attn_mask, 1)
        tensor([[False, False,  True,  True,  True,  True],
                [False, False, False,  True,  True, False]])
    """
    # Get batch size and sequence length
    batch_size, seq_len = attn_mask_in_length.shape
    device = attn_mask_in_length.device

    # Create cumulative sum of lengths, padding with zeros
    cumsum = torch.zeros((batch_size, seq_len + 1), device=device)
    cumsum[:, 1:] = torch.cumsum(attn_mask_in_length, dim=1)

    # Create position indices tensor
    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)

    # Get start and end positions for the requested sequence
    start_pos = cumsum[:, sequence_idx]
    end_pos = cumsum[:, sequence_idx + 1]

    # Create mask where True indicates positions within the sequence_idx'th sequence
    mask = (positions >= start_pos.unsqueeze(1)) & (positions < end_pos.unsqueeze(1))

    return mask


def _create_padding_mask(attention_mask_in_length):
    """
    Create an attention padding mask from the lengths of sequences in batch
    The padding starts at the sum of the lengths of sequences in the batch
    """
    batch_size, seq_len = attention_mask_in_length.shape
    mask = torch.arange(seq_len, device=attention_mask_in_length.device).expand(
        batch_size, seq_len
    )
    return mask < attention_mask_in_length.sum(dim=-1, keepdim=True)


def _expand_distances_to_seqlen(self, distances_in_length, attn_mask_in_length):
    """
    Expand the per sequence distance to a positional encoding tensor
    using the length of each sequence in the batch
    """
    pos = torch.zeros_like(distances_in_length)
    B, _ = distances_in_length.size()
    for i in range(B):
        lengths = attn_mask_in_length[
            i, torch.nonzero(attn_mask_in_length[i, :], as_tuple=False).flatten()
        ].long()
        dists = distances_in_length[
            i, torch.nonzero(attn_mask_in_length[i, :], as_tuple=False).flatten()
        ].float()
        # repeat dists for each length
        dists2 = dists.repeat_interleave(lengths)
        pos[i, : dists2.size(0)] = dists2

    return pos


def _generate_test_sequence_and_attn_mask_in_length(
    vocab=None, device=torch.device("cuda")
):
    """
    Generate some small sequence and attn_mask_in_length data to test some functions
    """
    if vocab is None:
        vocab = esm.data.Alphabet.from_architecture("ESM-1b")

    # Initialize parameters
    batch_size = 5
    max_seq_len = 20
    start_tok_idx, end_tok_idx = 4, 10

    # Create input data
    x = torch.full((batch_size, max_seq_len), vocab.padding_idx, device=device)
    attn_mask_in_length = torch.zeros(
        (batch_size, max_seq_len), dtype=torch.long, device=device
    )

    # Store ground truth sequence positions for verification
    true_seq_positions = []

    # Fill in sequences and store true positions
    for i in range(batch_size):
        seq1_len = torch.randint(6, 10, (1,)).item()
        seq2_len = torch.randint(6, 10, (1,)).item()

        # First sequence
        x[i, 0] = vocab.cls_idx
        x[i, 1 : seq1_len - 1] = torch.randint(
            start_tok_idx, end_tok_idx, (seq1_len - 2,)
        )
        x[i, seq1_len - 1] = vocab.eos_idx

        # Second sequence
        x[i, seq1_len] = vocab.cls_idx
        x[i, seq1_len + 1 : seq1_len + seq2_len - 1] = torch.randint(
            start_tok_idx, end_tok_idx, (seq2_len - 2,)
        )
        x[i, seq1_len + seq2_len - 1] = vocab.eos_idx

        attn_mask_in_length[i, 0] = seq1_len
        attn_mask_in_length[i, 1] = seq2_len

        # Store true positions for each sequence
        true_seq_positions.append(
            {0: (0, seq1_len), 1: (seq1_len, seq1_len + seq2_len)}
        )

    return x, attn_mask_in_length, true_seq_positions


def _test_create_sequence_mask():
    x, attn_mask_in_length, true_seq_positions = (
        _generate_test_sequence_and_attn_mask_in_length()
    )
    batch_size, max_seq_len = x.shape

    # Test sequence masks
    for seq_idx in [0, 1]:
        mask = _create_sequence_mask(attn_mask_in_length, seq_idx)

        # Verify mask values for each batch element
        for batch_idx in range(batch_size):
            start, end = true_seq_positions[batch_idx][seq_idx]

            # Check that mask is True only for the correct sequence positions
            expected_mask = torch.zeros(max_seq_len, dtype=torch.bool, device=x.device)
            expected_mask[start:end] = True

            assert torch.all(
                mask[batch_idx] == expected_mask
            ), f"Incorrect mask values for batch {batch_idx}, sequence {seq_idx}"

            # Verify number of True values matches sequence length
            seq_len = attn_mask_in_length[batch_idx, seq_idx]
            assert (
                torch.sum(mask[batch_idx]) == seq_len
            ), f"Number of True values doesn't match sequence length for batch {batch_idx}, sequence {seq_idx}"
