import os
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import wandb
import numpy as np
import argparse
import torch
import esm

from ppievo.datasets._torch_dataset import PairMSADataModule
from ppievo.models._pipet import PipetModule
from ppievo.training._training_utils import (
    ValidationLikelihoodCallback,
    GradNormCallback,
)


def main(args):
    if args.seed is not None:
        pl.seed_everything(args.seed)

    logger = pl.pytorch.loggers.wandb.WandbLogger(
        name=args.wandb_name, project="protevo-complex", entity="junhaobearxiong"
    )

    # Read pair names from the transitions dir
    train_pair_names = [
        file.split(".txt")[0]
        for file in os.listdir(args.transitions_dir)
        if file.endswith(".txt")
    ]
    assert len(train_pair_names) > 0, "No transition files found"
    if args.num_train_pairs == -1:
        # Use all training pairs (1k)
        train_pair_names = train_pair_names
    else:
        # Shuffle and randomly select num_train_pairs
        # Seed determined by pl
        assert args.num_train_pairs <= len(
            train_pair_names
        ), f"num_train_pairs must be <= {len(train_pair_names)}"
        np.random.shuffle(train_pair_names)
        train_pair_names = train_pair_names[: args.num_train_pairs]
    print(f"Train on {len(train_pair_names)} pairs")

    # Use ESM2 vocab
    vocab = esm.data.Alphabet.from_architecture("ESM-1b")

    # Dataloader removes sequences that are longer than max_length
    # Creates validation set by randomly splitting transitions from the training set
    dm = PairMSADataModule(
        pair_names=train_pair_names,
        transitions_dir=args.transitions_dir,
        vocab=vocab,
        mask_prob=0.15,
        train_frac=0.85,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_batch_sampler=True,
        max_num_tokens=args.max_num_tokens,
    )
    dm.setup(stage="fit")
    print("Finished setting up datamodule")

    # Set up model
    if args.use_esm_input_embed or args.use_esm_final_embed:
        esm_model, _ = esm.pretrained.esm2_t30_150M_UR50D()
        esm_model.to(torch.device("cuda"))
    else:
        esm_model = None
    model = PipetModule(
        esm_model=esm_model,
        vocab=vocab,
        use_esm_input_embed=args.use_esm_input_embed,
        use_esm_final_embed=args.use_esm_final_embed,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        embedding_dim=args.embedding_dim,
        num_heads=args.num_heads,
        lr=args.lr,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_steps,
        weight_decay=args.weight_decay,
        dropout_p=args.dropout_p,
    )
    print("Finished loading models")

    # Create output directory for saving model checkpoints
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=False)
    # Define some call backs
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        save_top_k=-1,
        every_n_train_steps=args.checkpoint_every,
        # every_n_epochs = 2
    )

    if args.accelerator == "gpu" and len(args.devices) > 1:
        strategy = "ddp"
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        strategy=strategy,
        logger=logger,
        callbacks=[
            lr_monitor,
            checkpoint_callback,
            ValidationLikelihoodCallback(),
            GradNormCallback(),
        ],
        accelerator=args.accelerator,
        devices=args.devices,
        max_steps=args.max_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        detect_anomaly=False,
        gradient_clip_val=10,
        precision="bf16",
    )

    if args.resume_path is not None:
        trainer.fit(model, dm, ckpt_path=args.resume_path)
    else:
        trainer.fit(model, dm)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--transitions_dir",
        type=str,
        default="data/local_data/human_ppi/cache/transitions-seq_id_90_pair-vert_70",
        help="Directory storing the transition files for all traininig family pairs",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/local_data/checkpoints/pipet/",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--num_train_pairs",
        type=int,
        default=200,
        help="Number of family pairs to train on",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=200000,
        help="Maximum number of tokens in batch",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--max_length", type=int, default=1022, help="Maximum sequence length"
    )
    parser.add_argument(
        "--use_esm_input_embed",
        action="store_true",
        help="Use only ESM input embedding layer",
    )
    parser.add_argument(
        "--use_esm_final_embed", action="store_true", help="Use full ESM model"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_encoder_layers",
        type=int,
        default=5,
        help="Number of encoder transformer layers",
    )
    parser.add_argument(
        "--num_decoder_layers",
        type=int,
        default=5,
        help="Number of decoder transformer layers",
    )
    parser.add_argument("--embedding_dim", type=int, default=512, help="Embedding size")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--wandb_name", type=str, default="pipet_v0", help="Name of wandb run"
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=10000, help="Number of warmup steps"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading",
    )
    parser.add_argument("--accelerator", type=str, default="gpu", help="Accelerator")
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=None,
        help="GPU devices to use. Leave as None to auto-detect from Slurm.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=-1, help="Maximum number of steps to train for"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Number of batches to accumulate gradients over",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5000,
        help="Save checkpoint every n steps",
    )
    parser.add_argument(
        "--check_val_every_n_epoch", type=int, default=1, help="Validate every n epochs"
    )
    parser.add_argument(
        "--resume_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to model checkpoint to resume training",
    )
    parser.add_argument(
        "--dropout_p",
        type=float,
        default=0.1,
        help="Dropout in multihead attention (default 0.1)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight Decay to pass to optimizer (default = 0)",
    )

    args = parser.parse_args()
    if args.seed is None and (args.accelerator == "gpu" and args.devices > 1):
        raise ValueError("Must set seed when using multiple GPUs")
    print(args)
    main(args)
