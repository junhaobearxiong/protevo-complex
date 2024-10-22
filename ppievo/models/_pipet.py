import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
import lightning as pl

from ppievo.models._modules import (
    EncoderLayer,
    DecoderLayer,
    LMHead,
    MultiSequenceESMEmbed,
)
from ppievo.models._model_utils import (
    _create_sequence_mask,
    _expand_distances_to_seqlen,
)
from ppievo.training._training_utils import get_polynomial_decay_schedule_with_warmup


class Pipet(nn.Module):
    """
    Pipet: Protein interaction evolution in time

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
        esm_model=None,
        use_esm_input_embed=False,
        use_esm_final_embed=False,
        **kwargs,
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
        self.use_esm_input_embed = use_esm_input_embed
        self.use_esm_final_embed = use_esm_final_embed

        self.embed_tokens = nn.Embedding(len(vocab), embedding_dim)
        if self.use_esm_input_embed:
            if embedding_dim != esm_model.embed_tokens.embedding_dim:
                raise ValueError(
                    f"embedding_dim needs to be same as esm embedding dim={esm_model.embed_tokens.embedding_dim}"
                )
            # Use ESM input embedding layer weights and freeze it
            self.embed_tokens.load_state_dict(esm_model.embed_tokens.state_dict())
            self.embed_tokens.requires_grad_(False)

        if self.use_esm_final_embed:
            # Use the entire ESM to compute its final embedding
            # as the input to the encoder
            self.esm_embed = MultiSequenceESMEmbed(esm_model=esm_model, vocab=vocab)

        # Same weights are used to encode vocab into embedding
        # and decode embedding back to vocab
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
        # Embed input tokens for decoder
        h_dec = self.embed_tokens(dec_in)

        if self.use_esm_final_embed:
            # Use the ESM final layer embedding for encoder inputs
            # Each sequence in encoder input is embedded separatedly
            # enc_attn_mask indicates the sequence boundary
            h_enc = self.esm_embed(enc_in, enc_attn_mask)
        else:
            # Use standard input embedding for encoder inputs
            h_enc = self.embed_tokens(enc_in)

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

    def training_step(self, batch):
        (
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_targets,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
            distances_tensor,
        ) = batch

        logits = self(
            enc_inputs,
            dec_inputs,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
            distances_tensor,
        )

        if isinstance(self.criterion, nn.CrossEntropyLoss):
            # Cross entropy loss expects the logits to be in the second to last dimension
            logits = [l.transpose(-1, -2) for l in logits]
        enc_logits, dec_logits = logits

        # Mask language model loss on the encoder
        mlm_loss = self.criterion(enc_logits, enc_targets)
        # Causal language model loss on the decoder
        clm_loss = self.criterion(dec_logits, dec_targets)

        mlm_ppl = torch.exp(mlm_loss.detach())
        clm_ppl = torch.exp(clm_loss.detach())

        loss = mlm_loss + clm_loss

        metrics = {
            "loss": loss,
            "mlm_loss": mlm_loss,
            "clm_loss": clm_loss,
            "mlm_ppl": mlm_ppl,
            "clm_ppl": clm_ppl,
        }
        return metrics

    def validation_step(self, batch):
        (
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_targets,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
            distances_tensor,
        ) = batch

        with torch.no_grad():
            logits = self(
                enc_inputs,
                dec_inputs,
                attn_mask_enc_lengths,
                attn_mask_dec_lengths,
                distances_tensor,
            )

        enc_logits, dec_logits = logits
        mlm_loss = F.cross_entropy(
            enc_logits.transpose(-1, -2),
            enc_targets,
            ignore_index=self.vocab.padding_idx,
            reduction="mean",
        )
        mlm_ppl = torch.exp(mlm_loss.detach())

        # Keep unreduced to get per-site time likelihood
        clm_loss = F.cross_entropy(
            dec_logits.transpose(-1, -2),
            dec_targets,
            ignore_index=self.vocab.padding_idx,
            reduction="none",
        ).detach()

        # Obtain the per-site likelihood for x2 and y2 separately
        # Generate mask for x2 and y2 in the decoder targets
        x_seq_mask = _create_sequence_mask(attn_mask_dec_lengths, sequence_idx=0)
        y_seq_mask = _create_sequence_mask(attn_mask_dec_lengths, sequence_idx=1)
        # Expand the distance to be per position, shape (B, L)
        distances_seqlen = _expand_distances_to_seqlen(
            distances_tensor, attn_mask_dec_lengths
        )

        # Shape (total num tokens of x in the batch,)
        x_dist = distances_seqlen[x_seq_mask]
        x_ll = -clm_loss[x_seq_mask]
        # Dict: distance --> per-site log likelihood of x
        x_ll_per_dist = {
            d.item(): x_ll[x_dist == d].cpu().numpy() for d in x_dist.unique()
        }
        # Shape (total num tokens of y in the batch,)
        y_dist = distances_seqlen[y_seq_mask]
        y_ll = -clm_loss[y_seq_mask]
        y_ll_per_dist = {
            d.item(): y_ll[y_dist == d].cpu().numpy() for d in y_dist.unique()
        }

        # Average decoder metrics
        padding_mask = dec_targets != self.vocab.padding_idx
        clm_loss_mean = clm_loss[padding_mask].mean()
        clm_ppl = torch.exp(clm_loss_mean)
        loss = mlm_loss + clm_loss_mean
        acc = (
            (dec_logits.argmax(-1)[padding_mask] == dec_targets[padding_mask])
            .float()
            .mean()
            .item()
        )

        metrics = {
            "loss": loss,
            "clm_loss": clm_loss_mean,
            "clm_ppl": clm_ppl,
            "acc": acc,
            "mlm_loss": mlm_loss,
            "mlm_ppl": mlm_ppl,
        }
        ll_per_dist = (x_ll_per_dist, y_ll_per_dist)
        return metrics, ll_per_dist


class PipetModule(pl.LightningModule):
    """
    Lightning Module Wrapper around the Pipet Model
    """

    def __init__(
        self,
        lr: float = 1e-4,
        num_warmup_steps: int = 10000,
        num_training_steps: int = 100000,
        **kwargs,
    ):
        super().__init__()

        self.model = Pipet(**kwargs)
        self.lr = lr
        self.wd = kwargs.get("weight_decay", 0.0)
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.save_hyperparameters(ignore=["vocab"])

    def forward(self, enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances):
        return self.model(enc_in, dec_in, enc_attn_mask, dec_attn_mask, distances)

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
        metrics, ll_per_dist = self.model.validation_step(batch)
        self._log(metrics, train=False)
        return metrics, ll_per_dist

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
