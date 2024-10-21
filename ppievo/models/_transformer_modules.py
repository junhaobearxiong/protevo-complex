"""
Some modules for standard transformer architecture
"""

import torch
import torch.nn as nn
import numpy as np
from esm.modules import gelu  # use esm gelu
from einops import rearrange

from flash_attn.bert_padding import (
    pad_input,
    unpad_input,
)
from flash_attn.modules.mha import (
    flash_attn_varlen_kvpacked_func,
    flash_attn_varlen_qkvpacked_func,
)
from flash_attn.layers.rotary import (
    RotaryEmbedding as FlashRotaryEmbedding,
)


class RopeFlashMHA(nn.Module):
    """Flash Multi-Head Attention module for transformer, with Rotary Embedding"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        add_bias_kv=False,
        dropout=0.0,
        self_attn=True,
        causal: bool = False,
        layer_idx=None,
    ):
        super().__init__()

        self.head_dim = embed_dim // num_heads
        self.num_heads = num_heads
        self.causal = causal
        self.self_attn = self_attn
        self.layer_idx = layer_idx
        self.dropout = dropout
        self.add_bias_kv = add_bias_kv

        hidden_size = embed_dim
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

        # NOTE: the FlashRotaryEmbedding module is used here, make sure that pos_idx_in_fp32 is set to True,
        # otherwise positional indexing will fail for large indices due to range limitations of bf16
        # i.e. 265 = 270 due to rounding
        self.rot_emb = FlashRotaryEmbedding(
            dim=self.head_dim,
            base=10000.0,
            interleaved=False,
            scale_base=None,
            pos_idx_in_fp32=True,  # very important for bf16
        )

    def forward(
        self,
        x,
        y=None,
        x_padding_mask=None,
        y_padding_mask=None,
    ):

        Bx, Lx, Dx = x.size()

        if self.self_attn:
            # project x to q, k, v
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
        else:
            # y comes from encoder, provides keys and values
            assert y is not None, "Cross attention requires y input"
            q = self.q_proj(x)
            k = self.k_proj(y)
            v = self.v_proj(y)

        # rescale q
        q *= self.head_dim**-0.5

        q = rearrange(q, "b l (n h) -> b l n h", n=self.num_heads)
        k = rearrange(k, "b l (n h) -> b l n h", n=self.num_heads)
        v = rearrange(v, "b l (n h) -> b l n h", n=self.num_heads)

        # NOTE: flash atten's rot emb performs this in-place
        q, k = self.rot_emb(
            q,
            torch.stack([k, v], dim=2),
            seqlen_offset=0,
            max_seqlen=max(
                q.shape[1], k.shape[1]
            ),  # this is important if q, k are not the same shape
        )

        # at this point, k contains k and v, and is of shape [B, L, 2, N, H]
        if x_padding_mask is None:
            x_padding_mask = torch.ones(Bx, Lx, device=x.device, dtype=torch.bool)

        q, idx_q, cu_seqlens_q, max_seqlen_q = unpad_input(q, x_padding_mask)

        if self.self_attn:
            k, idx_k, cu_seqlens_k, max_seqlen_k = unpad_input(
                k, x_padding_mask
            )  # k = kv

            qkv = torch.cat([q.unsqueeze(1), k], dim=1)  # (total_nonpad, 3, N, H)
            out = flash_attn_varlen_qkvpacked_func(
                qkv,
                cu_seqlens_q,
                max_seqlen_q,
                dropout_p=self.dropout,
                softmax_scale=1.0,  # q has been scaled already
                causal=self.causal,
            )
        else:
            # cross attention
            if y_padding_mask is None:
                By, Ly, Dy = y.size()
                y_padding_mask = torch.ones(By, Ly, device=y.device, dtype=torch.bool)
            k, idx_k, cu_seqlens_k, max_seqlen_k = unpad_input(k, y_padding_mask)

            out = flash_attn_varlen_kvpacked_func(
                q,
                k,
                cu_seqlens_q,
                cu_seqlens_k,
                max_seqlen_q,
                max_seqlen_k,
                dropout_p=self.dropout,
                softmax_scale=1.0,  # q has been scaled already
                causal=self.causal,
            )

        out = pad_input(out, idx_q, Bx, Lx)  # repad
        out = rearrange(out, "... h d -> ... (h d)")  # concatenate heads

        return self.out_proj(out)  # linear projection


class FlashMHAEncoderBlock(nn.Module):
    """Flash Multi-Head Attention Encoder Block for transformer, with Rotary Embedding.
    This implementation yields identical results to that of ESM2's MHA.
    """

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_bias=True,
        add_bias_kv=False,
        dropout_p=0.0,
        layer_idx=None,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads

        # initialized submodules
        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            bias=use_bias,
            add_bias_kv=add_bias_kv,
            dropout=dropout_p,
            self_attn=True,
            causal=False,
            layer_idx=layer_idx,
        )

        # layer norms
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        # ffn layers
        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, **kwargs):
        """
        Forward pass using the x sequence. kwargs should contain the x padding mask as x_padding_mask.
        """

        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x, **kwargs)

        x = residual + x

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x

        return x


class FlashMHADecoderBlock(nn.Module):
    """Flash Multi-Head Attention Decoder Block for transformer, with Rotary Embedding.
    Will perform both causal self-attention and cross-attention."""

    def __init__(
        self,
        embed_dim,
        ffn_embed_dim,
        attention_heads,
        use_bias=True,
        add_bias_kv=False,
        dropout_p=0.0,
        layer_idx=None,
        **kwargs
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.use_bias = use_bias
        self.ffn_embed_dim = ffn_embed_dim
        self.add_bias_kv = add_bias_kv
        self.dropout_p = dropout_p
        self.layer_idx = layer_idx
        self.attention_heads = attention_heads

        self.self_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            self_attn=True,
            causal=True,
            layer_idx=layer_idx,
            dropout=dropout_p,
        )
        self.cross_attn = RopeFlashMHA(
            embed_dim=embed_dim,
            num_heads=attention_heads,
            self_attn=False,
            causal=False,
            layer_idx=layer_idx,
            dropout=dropout_p,
        )
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)

        self.fc1 = nn.Linear(embed_dim, ffn_embed_dim)
        self.fc2 = nn.Linear(ffn_embed_dim, embed_dim)

    def forward(self, x, y, **kwargs):
        """
        Forward pass using the x sequence and the y sequence. kwargs should contain both the
        x and y padding masks as x_padding_mask and y_padding_mask.

        The unpadding and repadding is done in the FlashMHA module so that rotary embeddings can be used.

        My notation is not ideal - the x, y refer to the x,y sequences from an x,y,t trio, but since this is
        the decoder, x is y and y is x (i.e. y provides the q, while k,v come from x).
        This is handled in the overall transformer forward pass (reassigning x to y and the padding masks accordingly).
        """

        self_attn_kwargs = {"x_padding_mask": kwargs.get("x_padding_mask", None)}

        # causal self-attention
        residual = x
        x = self.self_attn_layer_norm(x)
        x = self.self_attn(x=x, **self_attn_kwargs)
        x = x + residual

        # cross attention
        residual = x
        x = self.cross_attn_layer_norm(x)
        x = self.cross_attn(x=x, y=y, **kwargs)
        x = x + residual

        residual = x
        x = self.final_layer_norm(x)
        x = gelu(self.fc1(x))
        x = self.fc2(x)
        x = x + residual

        return x


class GeometricTimeEmbedder(nn.Module):

    def __init__(self, frequency_embedding_size=256, start=1e-5, stop=0.25):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.start = start
        self.stop = stop  # bigger = flatter# modifies the scale of the distances eg the scale of the y-axis

    def timestep_embedding(self, timesteps, dim):
        freqs = torch.tensor(
            np.geomspace(start=self.start, stop=self.stop, num=dim // 2),
            dtype=timesteps.dtype,
        ).to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_emb = self.timestep_embedding(t, dim=self.frequency_embedding_size)
        return t_emb
