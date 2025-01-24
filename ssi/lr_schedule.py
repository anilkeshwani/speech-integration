import logging
import math
import os
import sys
from functools import partial

from omegaconf import DictConfig
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup, get_lr


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def setup_lr_scheduler(
    cfg: DictConfig,
    optimizer: Optimizer,
    global_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    optimizer_in_bwd: bool = False,  # for future impl.
    optim_ckpt_wrapper=None,  # for future impl.
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

    # NOTE PyTorch LR schedulers have the extremely misleadingly name `last_epoch` parameter, which is in fact the
    # global step in the cosine annealing with warmup regime, where we require step-level granularity
    get_cosine_schedule_with_warmup(optimizer, 

    if optimizer_in_bwd:
        # Modify the scheduler for optimizer_in_bwd case
        optim_ckpt_wrapper.set_lr_scheduler(lr_scheduler)

    return lr_scheduler
