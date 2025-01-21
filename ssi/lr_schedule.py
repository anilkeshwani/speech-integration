import logging
import math
import os
import sys
from functools import partial

from omegaconf import DictConfig
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def lr_lambda(
    num_training_steps: int,
    current_step: int,
    num_warmup_steps: int,
    num_cycles: float,
) -> float:
    # linear warmup phase
    if current_step < num_warmup_steps:
        return current_step / max(1, num_warmup_steps)
    # cosine
    progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
    cosine_lr_multiple = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return max(0.0, cosine_lr_multiple)


def setup_lr_scheduler(
    cfg: DictConfig,
    optimizer: Optimizer,
    last_epoch: int,
    current_step: int,
    num_training_steps: int,
    optimizer_in_bwd: bool,
    optim_ckpt_wrapper=None,
) -> LambdaLR | None:
    if cfg.lr_scheduler is None:
        LOGGER.info("No learning rate scheduler configured. Using constant learning rate.")
        return None

    if optimizer_in_bwd:
        raise NotImplementedError
        # Use the first optimizer from the wrapper to represent the learning rate
        optimizer = next(iter(optim_ckpt_wrapper.optim_map.values()))
    else:
        # Standard case: use the single optimizer
        optimizer = optimizer

    # Instantiate the learning rate scheduler
    lr_lambda_partial = partial(
        lr_lambda,
        num_training_steps=num_training_steps,
        current_step=current_step,
        num_warmup_steps=cfg.lr_scheduler.num_warmup_steps,
        num_cycles=cfg.lr_scheduler.num_cycles,
    )

    lr_scheduler = LambdaLR(optimizer, lr_lambda_partial, last_epoch)

    if optimizer_in_bwd:
        # Modify the scheduler for optimizer_in_bwd case
        optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

    return lr_scheduler
