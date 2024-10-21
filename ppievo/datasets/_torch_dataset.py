import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

from esm.data import Alphabet
from ppievo.io import read_transitions


# TODO: allow loading multiple families
# https://github.com/songlab-cal/protein-evolution/blob/91e64da3a87fe8497694acaac22aefdb233d2210/protevo/datasets/_torch_datasets.py#L154
class PairMSADataset(Dataset):
    def __init__(
        self,
        pair_name: str,
        transitions_dir: str,
        vocab: Alphabet,
        max_length: int = 1022,
    ):
        """
        Torch dataset object for transition data from pair MSAs
        Note: the indication of which protein in the complex is 'x' and
        which one is 'y' is somewhat arbitrary.
        In the current implementation, 'y' is the protein whose tree determines the cherries.

        Args:
        pair_name (str): "{protein_1}_{protein_2}"
        transitions_dir(str): Directory storing all transitions, each file is
            "{protein_x}_{protein_y}.txt"
        vocab (Alphabet): Vocabulary for encoding sequences
        """
        super(PairMSADataset, self).__init__()
        transitions = read_transitions(
            os.path.join(transitions_dir, pair_name + ".txt")
        )
        self.vocab = vocab
        # Tokenized unaligned sequences (i.e. no gaps)
        self.x1_toks, self.x2_toks, self.y1_toks, self.y2_toks = [], [], [], []
        # Tokenized aligned sequences (i.e. have gaps)
        self.x1_aln_toks, self.x2_aln_toks, self.y1_aln_toks, self.y2_aln_toks = (
            [],
            [],
            [],
            [],
        )
        # Lengths of unaligned sequences
        self.x1_len, self.x2_len, self.y1_len, self.y2_len = [], [], [], []
        # Branch lengths
        self.tx, self.ty = [], []

        for x1_aln, y1_aln, x2_aln, y2_aln, tx, ty in transitions:
            x1_seq_len = len(x1_aln.replace("-", ""))
            x2_seq_len = len(x2_aln.replace("-", ""))
            y1_seq_len = len(y1_aln.replace("-", ""))
            y2_seq_len = len(y2_aln.replace("-", ""))

            # Filter out transitions where any of the protein is too long
            # This is because we use ESM to embed each protein separately
            # and ESM has a length limit
            if (
                x1_seq_len > max_length
                or x2_seq_len > max_length
                or y1_seq_len > max_length
                or y2_seq_len > max_length
            ):
                continue

            self.x1_len.append(x1_seq_len)
            self.x2_len.append(x2_seq_len)
            self.y1_len.append(y1_seq_len)
            self.y2_len.append(y2_seq_len)

            self.x1_aln_toks.append(torch.tensor(self.vocab.encode(x1_aln)))
            self.x1_toks.append(
                torch.tensor(self.vocab.encode(x1_aln.replace("-", "")))
            )
            self.x2_aln_toks.append(torch.tensor(self.vocab.encode(x2_aln)))
            self.x2_toks.append(
                torch.tensor(self.vocab.encode(x2_aln.replace("-", "")))
            )
            self.y1_aln_toks.append(torch.tensor(self.vocab.encode(y1_aln)))
            self.y1_toks.append(
                torch.tensor(self.vocab.encode(y1_aln.replace("-", "")))
            )
            self.y2_aln_toks.append(torch.tensor(self.vocab.encode(y2_aln)))
            self.y2_toks.append(
                torch.tensor(self.vocab.encode(y2_aln.replace("-", "")))
            )

            self.tx.append(tx)
            self.ty.append(ty)

    def __len__(self):
        return len(self.x1_toks)

    def __getitem__(self, index):
        return {
            "x1_toks": self.x1_toks[index],
            "x2_toks": self.x2_toks[index],
            "y1_toks": self.y1_toks[index],
            "y2_toks": self.y2_toks[index],
            "x1_aln_toks": self.x1_aln_toks[index],
            "x2_aln_toks": self.x2_aln_toks[index],
            "y1_aln_toks": self.y1_aln_toks[index],
            "y2_aln_toks": self.y2_aln_toks[index],
            "x1_len": self.x1_len[index],
            "x2_len": self.x2_len[index],
            "y1_len": self.y1_len[index],
            "y2_len": self.y2_len[index],
            "tx": self.tx[index],
            "ty": self.ty[index],
        }


def mask_for_mlm(
    seqs: list[torch.Tensor], mask_prob: float, mask_idx: int, padding_idx: int
):
    """
    Apply uniform masking on sequences
    Should be done prior to padding since in the targets we fill all the
    non-masked positions with paddings
    """
    x_mlm_masks = [torch.rand_like(seq, dtype=torch.float) < mask_prob for seq in seqs]

    x_mlm_inputs = [
        seq.masked_fill(mask, mask_idx) for seq, mask in zip(seqs, x_mlm_masks)
    ]

    x_mlm_targets = [
        seq.masked_fill(~mask, padding_idx) for seq, mask in zip(seqs, x_mlm_masks)
    ]

    return x_mlm_inputs, x_mlm_targets


def mask_set_for_mlm(
    seqset: list[list[torch.Tensor]], mask_prob: float, mask_idx: int, padding_idx: int
):
    inputs = []
    targets = []
    for seqs in seqset:
        x_mlm_inputs, x_mlm_targets = mask_for_mlm(
            seqs=seqs, mask_prob=mask_prob, mask_idx=mask_idx, padding_idx=padding_idx
        )
        inputs.append(x_mlm_inputs)
        targets.append(x_mlm_targets)
    return inputs, targets


class PipetCollator:
    def __init__(
        self,
        vocab: Alphabet,
        mask_prob: float = 0.15,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Batch collator for PairMSA Dataset used for the Pipet

        FUTURE NOTE:
        Current implementation ignores the tokens including gaps, which is not needed for training
        For likelihood evaluation, if the full length unaligned sequence is longer than the alignment
        (e.g. when we have insertions relative to the query)
        we would need a mask to denote which position in the unaligned is in the alignment
        and the aligned tokens to identify the non-gap positions.
        Similar to: https://github.com/songlab-cal/protein-evolution/blob/91e64da3a87fe8497694acaac22aefdb233d2210/protevo/evaluation/_per_site_log_likelihood_transformer.py#L218
        Since currently, the unaligned sequence is obtained from the alignment
        i.e. by removing all gaps, we can keep all likelihoods
        """
        self.vocab = vocab
        self.mask_prob = mask_prob
        # We use one of the superfluous token in the ESM-1b vocab
        # as the token to separate the chains in the decoder
        self.sep_token = "."
        self.sep_idx = self.vocab.get_idx(self.sep_token)
        self.device = device

    def __call__(self, batch):
        """
        Process batch
        """
        # Sequences pairs for the encoder and decoder
        enc_seqs, dec_seqs = [], []
        # Length of sequence pairs for the encoder and decoder
        enc_lengths, dec_lengths = [], []
        # Distance between x1 and x2; between y1 and y2
        distances = []
        for b in batch:
            enc_seqs.append([b["x1_toks"], b["y1_toks"]])
            dec_seqs.append([b["x2_toks"], b["y2_toks"]])
            # Each sequence in the encoder inputs have <cls> and <eos>
            enc_lengths.append([b["x1_len"] + 2, b["y1_len"] + 2])
            # Each sequence in the decoder input has <.> or <eos>
            dec_lengths.append([b["x2_len"] + 1, b["y2_len"] + 1])
            distances.append([b["tx"], b["ty"]])

        # Create masked x1 and y1 for MLM loss#
        enc_inputs, enc_targets = mask_set_for_mlm(
            seqset=enc_seqs,
            mask_prob=self.mask_prob,
            mask_idx=self.vocab.mask_idx,
            padding_idx=self.vocab.padding_idx,
        )

        # Add <cls> and <eos> to both encoder input sequences
        # Add <pad> to start and end of the encoder target sequences
        # Shape: list[list[torch.Tensor(x1_len), torch.Tensor(y1_len)]]
        # seqset = (x1, y1)
        enc_inputs = [
            [F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in seqset]
            for seqset in enc_inputs
        ]
        enc_inputs = [
            [F.pad(seq, (1, 0), value=self.vocab.cls_idx) for seq in seqset]
            for seqset in enc_inputs
        ]
        enc_targets = [
            [F.pad(seq, (0, 1), value=self.vocab.padding_idx) for seq in seqset]
            for seqset in enc_targets
        ]
        enc_targets = [
            [F.pad(seq, (1, 0), value=self.vocab.padding_idx) for seq in seqset]
            for seqset in enc_targets
        ]

        # Concatenate x1 and y1
        # Shape: (B, x1_len + y1_len)
        enc_inputs = [torch.cat(seqset, dim=0) for seqset in enc_inputs]
        enc_targets = [torch.cat(seqset, dim=0) for seqset in enc_targets]
        # Add padding to make all concat sequences in the batch have the same lengths
        enc_inputs = nn.utils.rnn.pad_sequence(
            enc_inputs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        enc_targets = nn.utils.rnn.pad_sequence(
            enc_targets, batch_first=True, padding_value=self.vocab.padding_idx
        )

        # Add '.' to the end of x2 and <eos> to end of y2
        dec_seqs = [
            [
                F.pad(seqset[0], (0, 1), value=self.sep_idx),
                F.pad(seqset[1], (0, 1), value=self.vocab.eos_idx),
            ]
            for seqset in dec_seqs
        ]
        # Concatenate x2 and y2
        # Shape: (B, x2_len + y2_len)
        dec_seqs = [torch.cat(seqset, dim=0) for seqset in dec_seqs]
        # Add paddings to make all concat sequences the same length
        dec_seqs = nn.utils.rnn.pad_sequence(
            dec_seqs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        # Input has the <cls> to start of concat sequences to form the
        # decoder input
        dec_inputs = F.pad(dec_seqs[:, :-1], (1, 0), value=self.vocab.cls_idx)

        # Create masks to record the lengths of sequences in each pair
        # Non-zero values indicate the start and
        # length of each sequence, while zeros represent padding.
        attn_mask_enc_lengths = torch.zeros_like(enc_targets)
        attn_mask_dec_lengths = torch.zeros_like(dec_seqs)
        for i, l in enumerate(enc_lengths):
            attn_mask_enc_lengths[i, : len(l)] = torch.tensor(l)
        for i, l in enumerate(dec_lengths):
            attn_mask_dec_lengths[i, : len(l)] = torch.tensor(l)

        # Store distance tx and ty
        distances_tensor = torch.zeros_like(enc_targets, dtype=torch.float32)
        for i, d in enumerate(distances):
            distances_tensor[i, : len(d)] = torch.tensor(d)

        batch = (
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_seqs,
            attn_mask_enc_lengths,
            attn_mask_dec_lengths,
            distances_tensor,
        )
        batch = [b.to(self.device) for b in batch]
        return batch


class PeintCollator:

    def __init__(
        self,
        vocab: Alphabet,
        which_protein: str = "y",
        mask_prob: float = 0.0,
        device: torch.device = torch.device("cuda"),
        return_aln_seq: bool = True,
    ):
        """
        Batch collator for Peint (which doesn't model complexes)
        Follows https://github.com/songlab-cal/protein-evolution/blob/91e64da3a87fe8497694acaac22aefdb233d2210/protevo/datasets/_torch_datasets.py#L442

        Args:
        which_protein: Indicate which protein in the complex we want Peint to model
        either x or y
        return_aln_seq: If true, also return the tokenized aligned target sequence
        (no special token added) as a list of tensors
        """
        self.vocab = vocab
        self.which_protein = which_protein
        if self.which_protein not in ["x", "y"]:
            raise ValueError("`which_protein` should be 'x' or 'y'")
        self.mask_prob = mask_prob
        self.device = device
        self.return_aln_seq = return_aln_seq

    def __call__(self, batch):
        if self.which_protein == "x":
            enc_seqs = [b["x1_toks"] for b in batch]
            dec_seqs = [b["x2_toks"] for b in batch]
            distances = [b["tx"] for b in batch]
            dec_aln_seqs = [b["x2_aln_toks"] for b in batch]
        else:
            enc_seqs = [b["y1_toks"] for b in batch]
            dec_seqs = [b["y2_toks"] for b in batch]
            distances = [b["ty"] for b in batch]
            dec_aln_seqs = [b["y2_aln_toks"] for b in batch]

        # Process encoder inputs
        enc_inputs, enc_targets = mask_for_mlm(
            seqs=enc_seqs,
            mask_prob=self.mask_prob,
            mask_idx=self.vocab.mask_idx,
            padding_idx=self.vocab.padding_idx,
        )
        # Add <cls>, <eos>, then pad --> shape (B, L)
        enc_inputs = [
            F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in enc_inputs
        ]
        enc_inputs = [
            F.pad(seq, (1, 0), value=self.vocab.cls_idx) for seq in enc_inputs
        ]
        enc_inputs = nn.utils.rnn.pad_sequence(
            enc_inputs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        enc_targets = [
            F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in enc_targets
        ]
        enc_targets = [
            F.pad(seq, (1, 0), value=self.vocab.cls_idx) for seq in enc_targets
        ]
        enc_targets = nn.utils.rnn.pad_sequence(
            enc_targets, batch_first=True, padding_value=self.vocab.padding_idx
        )

        # Process decoder inputs
        # Add <eos> then pad the decoder sequence
        dec_seqs = [F.pad(seq, (0, 1), value=self.vocab.eos_idx) for seq in dec_seqs]
        dec_seqs = nn.utils.rnn.pad_sequence(
            dec_seqs, batch_first=True, padding_value=self.vocab.padding_idx
        )
        # Remove <eos> then add <cls> to start to form decoder input
        dec_inputs = F.pad(dec_seqs[:, :-1], (1, 0), value=self.vocab.cls_idx)
        dec_targets = dec_seqs

        # Process distance, shape (B, 1)
        distance_tensor = torch.tensor(distances).unsqueeze(-1)

        enc_padding_mask = enc_inputs == self.vocab.padding_idx
        dec_padding_mask = dec_inputs == self.vocab.padding_idx

        batch = (
            enc_inputs,
            enc_targets,
            dec_inputs,
            dec_targets,
            distance_tensor,
            enc_padding_mask,
            dec_padding_mask,
        )
        batch = [b.to(self.device) for b in batch]
        if self.return_aln_seq:
            batch.append(dec_aln_seqs)
        return batch
