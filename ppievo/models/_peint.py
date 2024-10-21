"""
Peint model with flash attention and frozen ESM encoder
Mostly from: https://github.com/songlab-cal/protein-evolution/blob/rnn/protevo/models/_transformer.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from esm.modules import RobertaLMHead

from ppievo.models._transformer_modules import (
    FlashMHAEncoderBlock,
    FlashMHADecoderBlock,
    GeometricTimeEmbedder,
)
from ppievo.training._training_utils import get_polynomial_decay_schedule_with_warmup


class ESMPretrainedTransformer(nn.Module):
    """Special version of the encoder-decoder transformer that builds on top of an ESM Model.
    The ESM model will first encode a sequence, and the final hidden representation will go into an encoder/decoder stack.
    The encoder/decoder stack will then work as a standard transformer.

    The ESM model is frozen and the transformer is trained on top of it.
    This also freezes the embedding layer and the LM head to stay in the same amino acid representation space.

    Time is encoded using sinusoidal positional encodings by bin.
    Positions are encoded using Rotary Embeddings (handled within the MHA modules).

    The entire model uses Flash Attention, including a rewritten version of the ESM model.
    """

    def __init__(
        self,
        esm_model,
        esm_vocab,
        embed_dim,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        max_len=1022,
        **kwargs,
    ):

        super(ESMPretrainedTransformer, self).__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.max_len = max_len

        self.esm = esm_model
        self.vocab = esm_vocab
        self.esm.eval()
        self.esm.requires_grad_(False)  # freeze the ESM model
        self.dropout_p = kwargs.get("dropout_p", 0.0)
        self.use_bias = kwargs.get("use_attention_bias", True)

        self.criterion = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=self.vocab.padding_idx
        )
        # embedding layer from ESM
        self.embedding = nn.Embedding(len(self.vocab), embed_dim)
        self.embedding.load_state_dict(self.esm.embed_tokens.state_dict())
        self.embedding.requires_grad_(False)

        self.time_embedding = GeometricTimeEmbedder(frequency_embedding_size=embed_dim)

        self.enc_layers = nn.ModuleList(
            [
                FlashMHAEncoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                )
                for l in range(num_encoder_layers)
            ]
        )

        self.dec_layers = nn.ModuleList(
            [
                FlashMHADecoderBlock(
                    embed_dim=self.embed_dim,
                    ffn_embed_dim=4 * self.embed_dim,
                    attention_heads=self.num_heads,
                    add_bias_kv=False,
                    dropout_p=self.dropout_p,
                    use_bias=self.use_bias,
                    layer_idx=l,
                )
                for l in range(num_decoder_layers)
            ]
        )

        self.lm_head = RobertaLMHead(
            embed_dim=self.embed_dim,
            output_dim=len(self.vocab),
            weight=self.embedding.weight,
        )

        self.lm_head.load_state_dict(self.esm.lm_head.state_dict())
        self.lm_head.requires_grad_(False)

    def _compute_language_model_representations(self, x):
        # x already has CLS and EOS from dataloader
        # esm deals with the padding directly (doesn't need the mask even though we have it)

        res = self.esm(x, repr_layers=[self.esm.num_layers], need_head_weights=False)

        esm_s = res["representations"][
            self.esm.num_layers
        ]  # just the final hidden state

        return esm_s

    def forward(self, x, y, t, x_attn_mask, y_attn_mask):
        # x: b x l, y: b x l, t: b x 1, x_pad_mask: b x l, y_pad_mask: b x l
        # embed y with t

        h_y = self.embedding(y)
        ht = self.time_embedding(t)
        ht = ht.expand_as(h_y)
        h_y = h_y + ht

        # get lm representations
        h_x = self._compute_language_model_representations(x)

        x_attn_mask = ~x_attn_mask  # now 1 means attend, 0 means don't
        y_attn_mask = ~y_attn_mask

        for i, enc_layer in enumerate(self.enc_layers):
            encoder_kwargs = {"x_padding_mask": x_attn_mask}
            h_x = enc_layer(x=h_x, **encoder_kwargs)

            if self.num_decoder_layers - self.num_encoder_layers + i >= 0:
                decoder_kwargs = {
                    "x_padding_mask": y_attn_mask,
                    "y_padding_mask": x_attn_mask,
                }

                idx = self.num_decoder_layers - self.num_encoder_layers + i
                dec_layer = self.dec_layers[idx]
                h_y = dec_layer(x=h_y, y=h_x, **decoder_kwargs)

        x_logits = self.lm_head(h_x)
        y_logits = self.lm_head(h_y)

        return x_logits, y_logits

    def training_step(self, batch):
        [x, x_targets, y, y_targets, t, x_atten_mask, y_atten_mask] = batch
        logits = self(x, y, t, x_atten_mask, y_atten_mask)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            # cross entropy loss expects the logits to be in the second to last dimension
            logits = [l.transpose(-1, -2) for l in logits]
        x_logits, y_logits = logits

        mlm_loss = self.criterion(x_logits, x_targets)
        tlm_loss = self.criterion(y_logits, y_targets)

        mlm_ppl = torch.exp(mlm_loss.detach())
        tlm_ppl = torch.exp(tlm_loss.detach())

        loss = mlm_loss + tlm_loss

        return {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "tlm_loss": tlm_loss,
            "mlm_ppl": mlm_ppl,
            "tlm_ppl": tlm_ppl,
        }

    def validation_step(self, batch):
        [x, x_targets, y, y_targets, t, x_atten_mask, y_atten_mask] = batch

        yt_mask = y_targets != self.vocab.padding_idx  # actual values

        times = t.expand_as(y_targets)  # expand time to match y_targets
        tbins = times[yt_mask]  # get time bins for actual values

        with torch.no_grad():
            x_logits, y_logits = self(x, y, t, x_atten_mask, y_atten_mask)

        y_loss = F.cross_entropy(
            y_logits.transpose(-1, -2),
            y_targets,
            ignore_index=self.vocab.padding_idx,
            reduction="none",
        )  # keep unreduced to get per-site time likelihood
        mlm_loss = F.cross_entropy(
            x_logits.transpose(-1, -2),
            x_targets,
            ignore_index=self.vocab.padding_idx,
            reduction="mean",
        )  # not as important to retain time info
        mlm_ppl = torch.exp(mlm_loss.detach())

        mask_loss = -1 * y_loss[yt_mask]

        ppl_per_bin = {
            b.item(): mask_loss[tbins == b].cpu().numpy() for b in t
        }  # gets the actual bin idx

        acc = (
            (y_logits.argmax(-1)[yt_mask] == y_targets[yt_mask]).float().mean().detach()
        )

        return {
            "loss": y_loss[yt_mask].mean(),
            "ppl": torch.exp(y_loss[yt_mask].mean()),
            "acc": acc,
            "mlm_loss": mlm_loss,
            "mlm_ppl": mlm_ppl,
        }, ppl_per_bin


class ProtEvoPretrainedTransformerModule(pl.LightningModule):
    def __init__(
        self,
        esm_model,
        esm_vocab,
        max_seq_len: int,
        num_heads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embed_dim: int,
        lr: float = 1e-4,
        num_warmup_steps: int = 10000,
        num_training_steps: int = 100000,
        **kwargs,
    ):
        super().__init__()

        self.model = ESMPretrainedTransformer(
            esm_model=esm_model,
            esm_vocab=esm_vocab,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            max_len=max_seq_len,
            **kwargs,
        )

        self.lr = lr
        self.wd = kwargs.get("weight_decay", 0.0)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_hyperparameters(
            ignore=["esm_model", "esm_vocab"]
        )  # don't save the esm model and vocab

    def forward(self, x, y, t, x_attn_mask, y_attn_mask):
        return self.model(x, y, t, x_attn_mask, y_attn_mask)

    def _log(self, loss_metrics, train=True):
        phase = "train" if train else "val"
        for k, v in loss_metrics.items():
            self.log(
                f"{phase}/{k}",
                v,
                on_step=train,
                on_epoch=(not train),
                sync_dist=True,
            )
            if train:
                self.log(
                    f"{phase}/{k}_epoch",
                    v,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                )

    def training_step(self, batch, batch_idx):
        metrics = self.model.training_step(batch)
        self._log(metrics, train=True)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics, ppl_per_bin = self.model.validation_step(batch)
        self._log(metrics, train=False)
        return metrics, ppl_per_bin

    def configure_optimizers(self):
        # multiple param groups for the encoder and decoder
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": self.wd},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.lr)

        scheduler = {
            "scheduler": get_polynomial_decay_schedule_with_warmup(
                optimizer,
                self.num_warmup_steps,
                self.num_training_steps,
                power=2.0,
            ),
            "name": "inverse-sqrt-lr",
            "interval": "step",
        }
        return [optimizer], [scheduler]
