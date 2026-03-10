import logging

from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchtune.training.lr_schedulers import get_cosine_schedule_with_warmup


LOGGER = logging.getLogger(__name__)


def setup_lr_scheduler(
    cfg: DictConfig,
    optimizer: Optimizer,
    global_step: int,
    num_training_steps: int,
    # arguments for future implementation of optimizer_in_bwd
    optimizer_in_bwd: bool = False,
    optim_ckpt_wrapper=None,
) -> LambdaLR | None:
    if cfg.get("lr_scheduler") is None:
        LOGGER.info("No learning rate scheduler configured. Using constant learning rate.")
        return None


    # NOTE PyTorch LR schedulers have the extremely misleadingly name `last_epoch` parameter, which is in fact the
    # global step in the cosine annealing with warmup regime, where we require step-level granularity
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        last_epoch=global_step,
        **cfg.lr_scheduler,
    )

    return lr_scheduler
