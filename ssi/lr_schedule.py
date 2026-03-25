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
) -> LambdaLR | None:
    if cfg.get("lr_scheduler") is None:
        LOGGER.info("No learning rate scheduler configured. Using constant learning rate.")
        return None

    # PyTorch's LambdaLR calls step() once during __init__, which increments last_epoch by 1 before
    # any training begins. Passing global_step - 1 therefore applies lr_lambda(global_step) to the
    # first batch, keeping the schedule consistent between fresh starts and resumes.
    # (PyTorch misleadingly calls this parameter `last_epoch`; it is a step counter here.)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_training_steps=num_training_steps,
        last_epoch=global_step,
        **cfg.lr_scheduler,
    )

    return lr_scheduler
