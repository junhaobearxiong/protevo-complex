"""
Per-site log likelihood evaluation of Peint
"""

import torch
import numpy as np
import esm
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import os

from ppievo.datasets._torch_dataset import PairMSADataset, PeintCollator
from ppievo.models._flash_esm import ESM2Flash
from ppievo.models._peint import (
    ProtEvoPretrainedTransformerModule,
    ESMPretrainedTransformer,
)
from ppievo.utils import Alphabet, amino_acids
from ppievo.io import write_transitions_log_likelihood_per_site


def load_model(
    model_checkpoint_path: str,
    device: torch.device,
):
    """
    Load Peint model in two steps:
    1. Load the pretrained ESM2, use it to initialize the flash ESM2 object
    2. Load the checkpoint for the trained portion of Peint
    """
    vocab = esm.data.Alphabet.from_architecture("ESM-1b")

    esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
    flash_esm = ESM2Flash(
        num_layers=esm_model.num_layers,
        embed_dim=esm_model.embed_dim,
        attention_heads=esm_model.attention_heads,
        alphabet="ESM-1b",
        token_dropout=True,
        dropout_p=0.0,
    )

    flash_esm.load_state_dict(esm_model.state_dict(), strict=False)
    del esm_model

    sd = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    module = ProtEvoPretrainedTransformerModule(
        esm_model=flash_esm, esm_vocab=vocab, **sd["hyper_parameters"]
    )

    module.load_state_dict(sd["state_dict"])

    return module.model.eval().to(device), vocab


def evaluate_peint_per_site_log_likelihood_for_pair(
    model: ESMPretrainedTransformer,
    vocab: Alphabet,
    pair_name: str,
    transitions_dir: str,
    which_protein: str = "y",
    renormalize_non_standard_states: bool = True,
    batch_size: int = 1,
    output_dir: str | None = None,
):
    """
    Get Peint per site likelihood on all transitions from a pair MSA
    Output size is the same as the alignment length, with gapped position zeroed out
    Optionally write to a file os.path.join(output_dir, '{pair_name}.txt')
    """
    # print("Start loading dataset..")
    dataset = PairMSADataset(
        pair_name=pair_name, transitions_dir=transitions_dir, vocab=vocab
    )
    collate_fn = PeintCollator(vocab=vocab, which_protein=which_protein, mask_prob=0)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    # print("Finished loading dataset")

    all_lls = []
    for i, batch in enumerate(loader):
        (
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_targets,
            distance_tensor,
            enc_padding_mask,
            dec_padding_mask,
            dec_aln_seqs,
        ) = batch

        dec_target_padding_mask = dec_targets.ne(vocab.padding_idx).cpu()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            with torch.no_grad():
                # ESMPretrainedTransformer gives two ouputs
                # ESMTransformer also outputs representation and attention
                _, dec_logits = model(
                    x=enc_inputs,
                    y=dec_inputs,
                    t=distance_tensor,
                    x_attn_mask=enc_padding_mask,
                    y_attn_mask=dec_padding_mask,
                )

                if renormalize_non_standard_states:
                    # Set logits for all non-standard states to -np.inf
                    # prior to the softmax in cross entropy
                    zero_idx = torch.tensor(
                        [
                            vocab.get_idx(t)
                            for t in vocab.all_toks
                            if t not in amino_acids
                        ]
                    )
                    dec_logits[:, :, zero_idx] = -np.inf

                # Cross entropy is negative log likelihood
                # Shape (B, L)
                dec_loss = -1 * F.cross_entropy(
                    dec_logits.transpose(-1, -2),
                    dec_targets,
                    ignore_index=vocab.padding_idx,
                    reduction="none",
                )
                dec_loss = dec_loss.detach().cpu()

                # Format the likelihood to be the same shape of the alignment
                # with nonzero likelihood at the non-gapped position
                # Suppose alignment is length D
                # batched output is length L
                # a particular sequence in the batch is length K (including <eos>)
                # then lls is shape (B, D), with K-1 nonzero elements
                # Note: we assume every position in the sequence is in the alignment
                for k in range(len(dec_aln_seqs)):
                    # Shape (D,), should have K-1 true elements
                    aln_nongap = dec_aln_seqs[k].ne(vocab.get_idx("-"))
                    # Shape (1, D)
                    lls = torch.zeros_like(dec_aln_seqs[k]).float().unsqueeze(0)
                    # Shape (K-1,), exclude both <eos> and all paddings
                    dec_loss_nonpad = dec_loss[k, dec_target_padding_mask[k]][:-1]
                    # Both true positions in `aln_nongap` and should corres
                    lls[:, aln_nongap] = dec_loss_nonpad
                    all_lls.append(lls.numpy())

    all_lls = np.concatenate(all_lls)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        output_fpath = os.path.join(output_dir, f"{pair_name}.txt")
        write_transitions_log_likelihood_per_site(all_lls.tolist(), output_fpath)
    return all_lls
