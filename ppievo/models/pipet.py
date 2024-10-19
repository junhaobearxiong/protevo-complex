from typing import List, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT

from ppievo.models.modules import EncoderLayer, DecoderLayer, LMHead


class Pipet(nn.Module):
    """
    Pipet: Protein-protein interaction evolution in time

    Transformer that models P(y2, x2 | y1, y2, tx, ty).
    The model is trained to predict y2 and x2 given y1 and x2,
    where y1 and y2 are separated in evolution by ty, and x1 and x2 by tx
    The logic is handled in the encoder and decoder modules, as well as the data preprocessing.
    """

    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embedding_dim: int,
        num_heads: int,
        vocab,
        **kwargs
    ):

        super().__init__()
        self.vocab = vocab
        self.embedding_dim = embedding_dim
        self.head_dim = embedding_dim // num_heads
        self.num_heads = num_heads
        assert (
            self.head_dim * num_heads == embedding_dim
        ), "embedding_dim must be divisible by num_heads"
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

        # Same weights are used to encode vocab into embedding
        # and decode embedding back to vocab
        self.embed_tokens = nn.Embedding(len(vocab), embedding_dim)
        self.lm_head = LMHead(embedding_dim, len(vocab), self.embed_tokens.weight)

        self.criterion = nn.CrossEntropyLoss(ignore_index=vocab.padding_idx)

        self.enc_layers = nn.ModuleList(
            [
                EncoderLayer(
                    attention_heads=num_heads,
                    embed_dim=embedding_dim,
                    ffn_embed_dim=4 * embedding_dim,
                    dropout_p=kwargs.get("dropout_p", 0.0),
                    layer_idx=i,
                )
                for i in range(num_encoder_layers)
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                DecoderLayer(
                    attention_heads=num_heads,
                    embed_dim=embedding_dim,
                    ffn_embed_dim=4 * embedding_dim,
                    dropout_p=kwargs.get("dropout_p", 0.0),
                    layer_idx=i,
                )
                for i in range(num_decoder_layers)
            ]
        )

    def forward(
        self, enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances, **kwargs
    ):
        # embed
        h_enc = self.embed_tokens(enc_in)
        h_dec = self.embed_tokens(dec_in)

        for i, enc_layer in enumerate(self.enc_layers):
            h_enc = enc_layer(h_enc, enc_attn_mask)

            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                decoder_kwargs = {
                    "dec_attn_mask": dec_attn_mask,
                    "enc_attn_mask": enc_attn_mask,
                    "distance_mask": distances,
                }
                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                h_dec = dec_layer(h_dec, h_enc, **decoder_kwargs)

        enc_logits = self.lm_head(h_enc)
        dec_logits = self.lm_head(h_dec)

        return enc_logits, dec_logits
