from typing import Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_varlen_qkvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import (
    pad_input,
    unpad_input,
    unpad_input_for_concatenated_sequences,
)
from flash_attn.layers.rotary import rotate_half

from ppievo.models._model_utils import (
    _create_padding_mask,
    _create_sequence_mask,
    _expand_distances_to_seqlen,
    _generate_test_sequence_and_attn_mask_in_length,
)


class SwiGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x


def gelu(x):
    """Implementation of the gelu activation function.

    For information: OpenAI GPT's gelu is slightly different
    (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


### Rotary Positional Encoding ###
class MultiSequenceRotaryPositionalEncoding(nn.Module):
    """
    Modified Rotary Positional Encoding (RoPE) module that applies positional encoding
    to input tensors based on sequence lengths within each example in a batch.

    Args:
        dim (int): The dimension of the input features (usually the head dimension).
        base (int, optional): The base for the positional encoding calculation.
                              Defaults to 10000.
        use_fp32_for_idx (bool, optional): Whether to use float32 for positional indices.
                                            This is important when working with larger
                                            sequence lengths in lower precision.
                                            Defaults to True.

    Input:
        x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_heads, head_dim).
        index_tensor (torch.Tensor): Tensor of shape (batch_size, seq_length) containing
                                     the lengths of sequences in each example within the batch.

    Output:
        torch.Tensor: Tensor of the same shape as input `x` with rotary positional
                      encoding applied.

    Example:
        >>> B, L, n, h = 2, 10, 20, 32
        >>> rope_module = ModifiedRotaryPositionalEncoding(h)
        >>> x = torch.randn(B, L, n, h)
        >>> index_tensor = torch.tensor([
        ...     [4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ...     [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]
        ... ])
        >>> output = rope_module(x, index_tensor)
    """

    def __init__(self, dim: int, base: int = 10000, use_fp32_for_idx: bool = True):
        super().__init__()

        self.dim = dim
        self.base = base
        self.use_fp32_for_idx = use_fp32_for_idx

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def _compute_rope(self, positions: torch.Tensor):
        freqs = torch.einsum("bl,d->bld", positions.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

    def _create_positional_indices(self, index_tensor: torch.Tensor):
        B, _ = index_tensor.shape
        device = index_tensor.device
        # Create position indices
        dtype = torch.float32 if self.use_fp32_for_idx else index_tensor.dtype
        positions = torch.zeros_like(index_tensor, dtype=dtype, device=device)
        for i in range(B):
            lengths = index_tensor[
                i, torch.nonzero(index_tensor[i, :], as_tuple=False).flatten()
            ]
            pos = torch.cat([torch.arange(l, device=device) for l in lengths], dim=0)
            positions[i, : pos.shape[0]] = pos
        return positions

    def forward(self, x: torch.Tensor, index_tensor: torch.Tensor):
        """
        Apply rotary positional encoding to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, num_heads, head_dim).
            index_tensor (torch.Tensor): Tensor of shape (batch_size, seq_length) containing
                                         the lengths of sequences in each batch.

        Returns:
            torch.Tensor: Tensor of the same shape as input `x` with rotary positional
                          encoding applied.
        """
        B, L, n, h = x.shape

        # Create positional indexes based on the index_tensor
        positions = self._create_positional_indices(index_tensor)

        # Compute RoPE components
        cos, sin = self._compute_rope(positions)

        # Reshape for broadcasting
        cos = cos.view(B, L, 1, h)
        sin = sin.view(B, L, 1, h)

        # Apply RoPE
        return (x * cos) + (rotate_half(x) * sin)


class TimestepEmbedder(nn.Module):

    def __init__(self, frequency_embedding_size=256, start=1e-5, stop=0.25):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.start = start
        self.stop = stop

    def timestep_embedding(self, timesteps, dim):
        freqs = torch.tensor(
            np.geomspace(start=self.start, stop=self.stop, num=dim // 2),
            dtype=timesteps.dtype,
        ).to(timesteps.device)
        args = timesteps[..., None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_emb = self.timestep_embedding(t, dim=self.frequency_embedding_size)
        return t_emb


class MultiSequenceMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for sets of sequences with restricted attention within each sequence.

    This module applies attention separately to each sequence within a set, using rotary positional
    encoding that resets for each sequence. It utilizes Flash Attention for efficient computation.
    This works for single sequences in the set as well, and should replicate a normal FlashAttentionMHA module.

    Args:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        max_position_embeddings (int): Maximum number of position embeddings.
        attention_dropout (float, optional): Dropout probability for attention weights. Defaults to 0.0.

    Example:
        >>> hidden_size = 768
        >>> num_attention_heads = 12
        >>> max_position_embeddings = 2048
        >>> attention = MultiSequenceMultiHeadAttention(hidden_size, num_attention_heads, max_position_embeddings)
        >>> hidden_states = torch.randn(2, 10, hidden_size)
        >>> attention_mask_in_length = torch.tensor([[4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
        ...                                          [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]])
        >>> output = attention(hidden_states, attention_mask_in_length)
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        dropout_p=0.0,
        causal=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(
            dim=self.head_dim,
            use_fp32_for_idx=True,
        )

        self.dropout_p = dropout_p
        self.causal = causal
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask_in_length):
        """

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
                Contains the input features for the attention mechanism.
            attention_mask_in_length (torch.Tensor): Tensor of shape (batch_size, seq_len) containing
                the lengths of sequences in each batch. Non-zero values indicate the start and
                length of each sequence, while zeros represent padding.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size) containing
                the attended features.

        Note:
            This method assumes that sequences within each batch are concatenated and padded to
            the maximum sequence length. The `attention_mask_in_length` tensor is used to identify
            the boundaries of individual sequences within each batch. It also keeps track of padding
        """
        bsz, q_len, _ = hidden_states.shape

        # Project input to query, key, and value
        qkv = self.qkv_proj(hidden_states)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_attention_heads
        )

        # Split qkv into separate tensors
        q, k, v = qkv.unbind(dim=2)
        q = q * self.head_dim**-0.5  # rescale

        # Apply rotary positional encoding to q and k
        q_rotary = self.rotary_emb(q, attention_mask_in_length)
        k_rotary = self.rotary_emb(k, attention_mask_in_length)

        # Recombine into qkv
        qkv_rotary = torch.stack([q_rotary, k_rotary, v], dim=2)

        # Prepare for Flash Attention
        qkv_flash = rearrange(qkv_rotary, "b s three h d -> b s (three h d)")
        qkv_unpad, indices, cu_q_lens, max_s = unpad_input_for_concatenated_sequences(
            qkv_flash, attention_mask_in_length
        )
        qkv_unpad = rearrange(
            qkv_unpad,
            "nnz (three h d) -> nnz three h d",
            three=3,
            h=self.num_attention_heads,
        )

        # Apply Flash Attention
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_q_lens,
            max_s,
            self.dropout_p,
            softmax_scale=None,
            causal=self.causal,
        )

        # Pad the output back to the original shape
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, q_len
            ),
            "b s (h d) -> b s h d",
            h=self.num_attention_heads,
        )

        # Final projection
        attn_output = rearrange(output, "b s h d -> b s (h d)")
        attn_output = self.out_proj(attn_output)

        return attn_output


class FullMultiSequenceMultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention module for pairs of sequences with full attention between sequences.

    This module applies attention to pairs of sequences (e.g., interacting proteins) within each item in a batch.
    It uses rotary positional encoding that resets for each sequence in the pair, while allowing full attention
    between the sequences. It utilizes Flash Attention for efficient computation.

    Args:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        dropout_p (float, optional): Dropout probability for attention weights. Defaults to 0.0.
        causal (bool, optional): Whether to apply causal masking. Defaults to False.
        layer_idx (Optional[int], optional): Layer index, used for logging or debugging. Defaults to None.

    Attributes:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        qkv_proj (nn.Linear): Linear projection for query, key, and value.
        out_proj (nn.Linear): Output projection.
        rotary_emb (MultiSequenceRotaryPositionalEncoding): Rotary positional encoding module.
        dropout_p (float): Dropout probability.
        causal (bool): Whether to apply causal masking.
        layer_idx (Optional[int]): Layer index.

    Example:
        >>> hidden_size = 768
        >>> num_attention_heads = 12
        >>> attention = MultiSequenceMultiHeadSelfAttention(hidden_size, num_attention_heads)
        >>> hidden_states = torch.randn(2, 1000, hidden_size)
        >>> attention_mask_in_length = torch.tensor([[446, 355, 0, ..., 0],
        ...                                          [335, 355, 0, ..., 0]])
        >>> output = attention(hidden_states, attention_mask_in_length)
    """

    def __init__(
        self,
        hidden_size,
        num_attention_heads,
        dropout_p=0.0,
        causal=False,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.qkv_proj = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(
            dim=self.head_dim,
            use_fp32_for_idx=True,
        )

        self.dropout_p = dropout_p
        self.causal = causal
        self.layer_idx = layer_idx

    def forward(self, hidden_states, attention_mask_in_length):
        """
        Forward pass of the multi-sequence multi-head self-attention module.

        Args:
            hidden_states (torch.Tensor): Input tensor of shape (batch_size, seq_len, hidden_size).
                Contains the input features for the attention mechanism.
            attention_mask_in_length (torch.Tensor): Tensor of shape (batch_size, num_sequences) containing
                the lengths of sequences in each batch. Non-zero values indicate the length of each sequence,
                while zeros represent padding. For example, [446, 355, 0, ..., 0] represents two sequences
                of lengths 446 and 355 in one item of the batch.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, hidden_size) containing
                the attended features.
        """
        bsz, seq_len, _ = hidden_states.shape

        # Project input to query, key, and value
        qkv = self.qkv_proj(hidden_states)
        qkv = rearrange(
            qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_attention_heads
        )

        # Split qkv into separate tensors
        q, k, v = qkv.unbind(dim=2)
        q = q * self.head_dim**-0.5  # rescale

        # Apply rotary positional encoding to q and k
        q_rotary = self.rotary_emb(q, attention_mask_in_length)
        k_rotary = self.rotary_emb(k, attention_mask_in_length)

        # Recombine into qkv
        qkv_rotary = torch.stack([q_rotary, k_rotary, v], dim=2)
        qkv_rotary = rearrange(qkv_rotary, "b s three h d -> b s (three h d)")

        # Create padding mask
        # Note: the main difference to Treeformer is the padding_mask here
        # is on the full sequence and we use `unpad_input`
        # rather than `unpad_input_for_concatenated_sequences` with attention_mask_in_length
        padding_mask = _create_padding_mask(attention_mask_in_length)
        # Unpad inputs
        qkv_unpad, indices, cu_seqlens, max_seqlen = unpad_input(
            qkv_rotary, padding_mask
        )
        qkv_unpad = rearrange(
            qkv_unpad,
            "nnz (three h d) -> nnz three h d",
            three=3,
            h=self.num_attention_heads,
        )

        # Apply Flash Attention
        output_unpad = flash_attn_varlen_qkvpacked_func(
            qkv_unpad,
            cu_seqlens,
            max_seqlen,
            self.dropout_p,
            softmax_scale=None,
            causal=self.causal,
        )

        # Pad the output back to the original shape
        output = rearrange(
            pad_input(
                rearrange(output_unpad, "nnz h d -> nnz (h d)"), indices, bsz, seq_len
            ),
            "b s (h d) -> b s h d",
            h=self.num_attention_heads,
        )

        # Final projection
        attn_output = rearrange(output, "b s h d -> b s (h d)")
        attn_output = self.out_proj(attn_output)

        return attn_output


class MultiSequenceCrossAttention(nn.Module):
    """
    This module performs cross-attention between a query sequence and a set of key-value sequences,
    applying both rotary positional encoding within sequences and distance-based sinusoidal encoding.
    It uses Flash Attention for efficient computation.

    Args:
        hidden_size (int): Size of the input and output features.
        num_attention_heads (int): Number of attention heads.
        max_position_embeddings (int): Maximum number of position embeddings for rotary encoding.
        max_distance (int): Maximum distance for sinusoidal encoding.
        dropout_p (float, optional): Dropout probability for attention weights. Defaults to 0.0.

    """

    def __init__(self, hidden_size, num_attention_heads, dropout_p=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        assert (
            self.head_dim * num_attention_heads == hidden_size
        ), "hidden_size must be divisible by num_attention_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.kv_proj = nn.Linear(hidden_size, 2 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)

        self.rotary_emb = MultiSequenceRotaryPositionalEncoding(self.head_dim)
        self.distance_emb = TimestepEmbedder(self.head_dim)

        self.dropout_p = dropout_p

    def forward(
        self,
        q_states,
        kv_states,
        attention_mask_in_length_q,
        attention_mask_in_length_kv,
        distances_in_length,
    ):
        # confusingly, y is the query sequence and x is the key-value sequence
        bsz, q_len, _ = q_states.shape
        _, kv_len, _ = kv_states.shape

        # Project inputs
        q = self.q_proj(q_states)
        kv = self.kv_proj(kv_states)

        # rescale q
        q *= self.head_dim**-0.5

        # reshape
        q = rearrange(q, "b s (h d) -> b s h d", h=self.num_attention_heads)
        kv = rearrange(
            kv, "b s (two h d) -> b s two h d", two=2, h=self.num_attention_heads
        )

        # Apply rotary positional encoding to q and k
        q_rotary = self.rotary_emb(q, attention_mask_in_length_q)
        k_rotary = self.rotary_emb(kv[:, :, 0], attention_mask_in_length_kv)

        # Apply sinusoidal distance encoding to k
        distance_positions = _expand_distances_to_seqlen(
            distances_in_length, attention_mask_in_length_kv
        )
        k_distance_embeddings = self.distance_emb(distance_positions)
        k_distance_embeddings = k_distance_embeddings.view(
            k_distance_embeddings.shape[0], k_distance_embeddings.shape[1], 1, -1
        )

        k_rotary += k_distance_embeddings

        # Recombine kv
        kv_encoded = torch.stack([k_rotary, kv[:, :, 1]], dim=2)

        # Create attention masks
        q_mask = _create_padding_mask(attention_mask_in_length_q)
        kv_mask = _create_padding_mask(attention_mask_in_length_kv)

        # Unpad inputs
        q, idx_q, cu_seqlens_q, max_s_q = unpad_input(q_rotary, q_mask)
        kv, _, cu_seqlens_k, max_s_k = unpad_input(kv_encoded, kv_mask)

        out = flash_attn_varlen_kvpacked_func(
            q=q,
            kv=kv,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_s_q,
            max_seqlen_k=max_s_k,
            dropout_p=self.dropout_p,
            softmax_scale=1.0,  # q has already been rescaled
            causal=False,
        )
        out = pad_input(out, idx_q, bsz, q_len)
        out = rearrange(out, "... h d -> ... (h d)")  # concat heads

        return self.out_proj(out)


class MultiSequenceESMEmbed(nn.Module):
    def __init__(self, esm_model: nn.Module, vocab: object):
        super().__init__()
        self.esm = esm_model
        self.vocab = vocab
        self.esm.eval()
        # Freeze ESM
        self.esm.requires_grad_(False)
        # The embedding size is the shape of the last layer's layer norm
        self.embed_dim = self.esm.layers[-1].final_layer_norm.normalized_shape[0]

    def forward(self, x, attn_mask_in_length):
        """
        Compute ESM embedding separately for each sequence in the batch

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len)
            attn_mask_in_length (torch.Tensor): Tensor of shape (batch_size, seq_len) containing
                the lengths of sequences in each batch.
                Non-zero values indicate the length of each sequence,
                while zeros represent padding.
        """
        bsz, seq_len = x.shape
        num_sequences = (attn_mask_in_length > 0).sum(dim=1)

        # Assert that all batch elements have the same number of sequences
        assert torch.all(
            num_sequences == num_sequences[0]
        ), "All batch elements must have the same number of sequences"

        num_sequences = num_sequences[0].item()

        combined_embedding = torch.zeros(
            (bsz, seq_len, self.embed_dim), dtype=torch.float, device=x.device
        )

        for seq_idx in range(num_sequences):
            # Shape (B, L), True for the residues of `seq_idx`th sequence of each batch element
            seq_mask = _create_sequence_mask(attn_mask_in_length, sequence_idx=seq_idx)
            # Create a copy of the input to avoid modifying the original
            x_seq = x.clone()
            # Set padding for all other positions
            x_seq.masked_fill_(~seq_mask, self.vocab.padding_idx)

            # Compute embeddings for the current sequence
            with torch.no_grad():
                output = self.esm(x_seq, repr_layers=[self.esm.num_layers])
                embedding = output["representations"][self.esm.num_layers]

            # Add the current sequence embedding to the combined embedding
            combined_embedding += embedding * seq_mask.unsqueeze(-1)

        return combined_embedding


def _test_multi_sequence_esm_embed(esm_model, vocab, device=torch.device("cuda")):
    """
    Test the multi-sequence embedding produced the same outputs as if the
    sequences are embedded separatedly
    """

    def esm_embed_seq(x):
        output = esm_model(x, repr_layers=[esm_model.num_layers])
        embedding = output["representations"][esm_model.num_layers]
        return embedding

    atol = 1e-5  # absolute tolerance
    # Generate some test sequences
    x, attn_mask_in_length, _ = _generate_test_sequence_and_attn_mask_in_length(
        vocab, device
    )
    batch_size = x.shape[0]

    # Embed the concat sequence
    esm_embed_multiseq = MultiSequenceESMEmbed(esm_model=esm_model, vocab=vocab)
    multiseq_embed = esm_embed_multiseq(x, attn_mask_in_length)

    # Embed the sequences separately
    max_seq1_len = attn_mask_in_length[:, 0].max().item()
    max_seq2_len = attn_mask_in_length[:, 1].max().item()

    # Prepare and embed first sequences
    x_seq1 = torch.full((batch_size, max_seq1_len), vocab.padding_idx, device=device)
    for i in range(batch_size):
        seq1_len = attn_mask_in_length[i, 0]
        x_seq1[i, :seq1_len] = x[i, :seq1_len]
    seq1_embedding = esm_embed_seq(x_seq1)

    # Prepare and embed second sequences
    x_seq2 = torch.full((batch_size, max_seq2_len), vocab.padding_idx, device=device)
    for i in range(batch_size):
        seq1_len = attn_mask_in_length[i, 0]
        seq2_len = attn_mask_in_length[i, 1]
        x_seq2[i, :seq2_len] = x[i, seq1_len : seq1_len + seq2_len]
    seq2_embedding = esm_embed_seq(x_seq2)

    # Check if embeddings match for each sequence in each batch element separately
    for i in range(batch_size):
        seq1_len = attn_mask_in_length[i, 0]
        seq2_len = attn_mask_in_length[i, 1]

        if not torch.allclose(
            multiseq_embed[i, :seq1_len], seq1_embedding[i, :seq1_len], atol=atol
        ):
            import pdb

            pdb.set_trace()

        assert torch.allclose(
            multiseq_embed[i, :seq1_len], seq1_embedding[i, :seq1_len], atol=atol
        ), f"Mismatch in sequence 1 for batch {i}"
        assert torch.allclose(
            multiseq_embed[i, seq1_len : seq1_len + seq2_len],
            seq2_embedding[i, :seq2_len],
            atol=atol,
        ), f"Mismatch in sequence 2 for batch {i}"


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        dropout_p: float,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.layer_idx = layer_idx
        self.dropout_p = dropout_p
        self.ffn_embed_dim = ffn_embed_dim

        self.self_attn = MultiSequenceMultiHeadSelfAttention(
            hidden_size=embed_dim,
            num_attention_heads=attention_heads,
            dropout_p=dropout_p,
            causal=False,
            layer_idx=layer_idx,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, attn_mask):
        """
        Self attention forward pass.
        Applies self attention to the input, the attention is over the full input sequence
        so x1 can attend to y1, and vice versa
        attn_mask is used to define the positions for the rotary positional encoding,
        so that it resets at each sequence boundary.

        e.g. if attn_mask =
        [[4, 6, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 5, 0, 0, 0, 0, 0, 0, 0, 0]]

        This means that for the first element in the batch, we have two sequences of length 4 and 6. For the second
        element, we have two sequences of length 3 and 5. So the corresponding rotary positional encoding for the input
        is:

        [[0, 1, 2, 3, 0, 1, 2, 3, 4, 5],
            [0, 1, 2, 0, 1, 2, 3, 4, 0, 0]]
        NB: the final 0s don't matter because they go beyond the sequence length and will be ignored by the unpad.
        """

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, attn_mask)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        attention_heads: int,
        embed_dim: int,
        ffn_embed_dim: int,
        dropout_p: float,
        layer_idx: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.attention_heads = attention_heads
        self.layer_idx = layer_idx
        self.dropout_p = dropout_p
        self.ffn_embed_dim = ffn_embed_dim

        self.self_attn = MultiSequenceMultiHeadSelfAttention(
            hidden_size=embed_dim,
            num_attention_heads=attention_heads,
            dropout_p=dropout_p,
            causal=True,
            layer_idx=layer_idx,
        )

        self.cross_attn = MultiSequenceCrossAttention(
            hidden_size=embed_dim,
            num_attention_heads=attention_heads,
            dropout_p=dropout_p,
        )

        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, enc_out, dec_attn_mask, enc_attn_mask, distance_mask):
        """
        Forward pass using the x (the decoder hiddens, abbreviated as "dec_out")
        and enc_out (the encoder output hiddens)
        as well as their attention length masks, and the distance mask of each sequence
        dec_out provides the queries, enc_out provides the keys and values.

        Unpadding and repadding occur in the various modules, and the rotary positional encoding is applied.
        Note that for both dec_out and enc_out, rotary restarts at each sequence boundary.
        The attn masks define the lengths of sequence in each batch for dec_out and enc_out.
        This helps define padding boundaries, as well as the rotary positional encoding positions.

        The distance mask is similar to the attn_masks, but defines for each sequence
        (x1 and x2 are separated by tx, y1 and y2 are separated by ty)
        Hence, when (x2, y2) goes into the cross attention module as q,
        the distance mask defines a positional encoding offset to k (which comes from x1, y1)
        """
        # Self attention of decoder input
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, dec_attn_mask)
        x = x + residual

        # Cross attention between decoder input and encoder output
        # query: decoder input
        # key, value: encoder output
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x, enc_out, dec_attn_mask, enc_attn_mask, distance_mask)
        x = x + residual

        # Layer norm + feed forward
        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


class LMHead(nn.Module):
    """
    Simple one layer linear head for language modeling to project
    hidden dimension back to vocab
    Weights are shared with the input embeddings
    """

    def __init__(self, embed_dim, vocab_size, weight):
        super().__init__()
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.proj = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, x):
        return F.linear(x, self.weight) + self.bias
