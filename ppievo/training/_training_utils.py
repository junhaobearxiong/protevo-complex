from typing import Any
import os
import numpy as np
import lightning as pl
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.utilities.types import STEP_OUTPUT
import wandb
from torch.optim.lr_scheduler import LambdaLR

from ppievo.utils import TIME_BINS, get_quantile_idx


# Taken from HuggingFace Transformers.
# https://github.com/huggingface/transformers/blob/641f418e102218c4bf16fcd3124bfebed6217ef6/src/transformers/optimization.py#L170
def get_polynomial_decay_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    lr_end=1e-7,
    power=1.0,
    last_epoch=-1,
):
    """
    Create a schedule with a learning rate that decreases as a polynomial decay from the initial lr set in the
    optimizer to end lr defined by `lr_end`, after a warmup period during which it increases linearly from 0 to the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        lr_end (:obj:`float`, `optional`, defaults to 1e-7):
            The end LR.
        power (:obj:`float`, `optional`, defaults to 1.0):
            Power factor.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.
    Note: `power` defaults to 1.0 as in the fairseq implementation, which in turn is based on the original BERT
    implementation at
    https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/optimization.py#L37
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    lr_init = optimizer.defaults["lr"]
    assert (
        lr_init > lr_end
    ), f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})"

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step > num_training_steps:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init
        else:
            lr_range = lr_init - lr_end
            decay_steps = num_training_steps - num_warmup_steps
            pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


class ValidationLikelihoodCallback(Callback):
    """
    This callback is used to calculate the time-bin likelihoods for the validation set.
    The actual likelihoods are calculated in the model's validation_step method.
    This aggregates them and stores the t, likelihood pairs.
    """

    def __init__(self):
        super().__init__()
        self.x_ll_per_bin, self.y_ll_per_bin = {}, {}

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        _, (x_ll_per_dist, y_ll_per_dist) = outputs
        for k, v in x_ll_per_dist.items():
            # Quantized time
            q_idx = get_quantile_idx(TIME_BINS, k)
            if q_idx not in self.x_ll_per_bin:
                self.x_ll_per_bin[q_idx] = []
            self.x_ll_per_bin[q_idx].extend(v)
        for k, v in y_ll_per_dist.items():
            # Quantized time
            q_idx = get_quantile_idx(TIME_BINS, k)
            if q_idx not in self.y_ll_per_bin:
                self.y_ll_per_bin[q_idx] = []
            self.y_ll_per_bin[q_idx].extend(v)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Mean likelihoods over sites for each time bin
        x_mean_likelihoods = np.zeros(len(TIME_BINS))
        y_mean_likelihoods = np.zeros(len(TIME_BINS))
        for k, v in self.x_ll_per_bin.items():
            x_mean_likelihoods[k] = np.exp(np.mean(v))
        for k, v in self.y_ll_per_bin.items():
            y_mean_likelihoods[k] = np.exp(np.mean(v))

        # Create tables for plotting
        nonzero_x = np.nonzero(x_mean_likelihoods)[0]
        data_x = [
            [d1, d2]
            for d1, d2 in zip(TIME_BINS[nonzero_x], x_mean_likelihoods[nonzero_x])
        ]
        table_x = wandb.Table(data=data_x, columns=["Time", "Mean per-site Likelihood"])

        nonzero_y = np.nonzero(y_mean_likelihoods)[0]
        data_y = [
            [d1, d2]
            for d1, d2 in zip(TIME_BINS[nonzero_y], y_mean_likelihoods[nonzero_y])
        ]
        table_y = wandb.Table(data=data_y, columns=["Time", "Mean per-site Likelihood"])

        # Important for multi-gpu runs, doesn't seem to affect single gpu runs
        if trainer.global_rank == 0:
            trainer.logger.experiment.log(
                {
                    "x_time_bin_likelihood": wandb.plot.line(
                        table_x,
                        "Time",
                        "Mean per-site Likelihood",
                        title="Likelihood per time bin, Sequence x",
                    )
                }
            )
            trainer.logger.experiment.log(
                {
                    "y_time_bin_likelihood": wandb.plot.line(
                        table_y,
                        "Time",
                        "Mean per-site Likelihood",
                        title="Likelihood per time bin, Sequence y",
                    )
                }
            )

        # Clear for next epoch
        self.x_ll_per_bin, self.y_ll_per_bin = {}, {}


def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm


class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        trainer.logger.experiment.log({"my_model/grad_norm": gradient_norm(pl_module)})
