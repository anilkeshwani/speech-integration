import logging
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchtune.modules import TransformerDecoder
from torchtune.utils import batch_to_device

from ssi.loss import compute_loss


LOGGER = logging.getLogger(__name__)


def compute_dataset_loss(
    model: TransformerDecoder,
    data_dev: DataLoader,
    loss_fn: Callable,
    epoch: int,
    global_step: int,
    steps_per_epoch: int,
    device: torch.device,
) -> float:
    dev_loss_running: float = 0
    num_tokens_dev: int = 0
    model.eval()
    with torch.inference_mode():
        for i_dev, dev_batch in enumerate(data_dev):
            batch_to_device(dev_batch, device)
            num_tokens_dev_batch = (dev_batch["labels"] != loss_fn.ignore_index).sum()
            num_tokens_dev += num_tokens_dev_batch
            dev_loss_batch = compute_loss(dev_batch, model, loss_fn) * num_tokens_dev_batch
            dev_loss_running += dev_loss_batch.item()
            LOGGER.info(
                f"Epoch {epoch + 1:03d} | "
                f"Global Step {global_step:0{len(str(steps_per_epoch))}d} | "  # TODO bad zero padding
                f"Dev Iter {i_dev:0{len(str(steps_per_epoch))}d} / {steps_per_epoch} | "
                f"Dev Batch {i_dev:0{len(str(len(data_dev)))}d} / {len(data_dev)} | "
                f"Dev Loss (batch): {dev_loss_batch.item():.4f}"
            )
    model.train()
    return dev_loss_running / num_tokens_dev  # loss per token
