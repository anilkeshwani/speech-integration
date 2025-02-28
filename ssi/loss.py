from typing import Callable

import torch
from torchtune.modules import TransformerDecoder


def compute_loss(batch: dict[str, torch.Tensor], model: TransformerDecoder, loss_fn: Callable) -> torch.Tensor:
    labels = batch.pop("labels")  # shape [b, s] needed for the loss not the model
    logits = model(**batch)  # NOTE add activation offloading context
    labels = torch.hstack((labels[..., 1:], torch.full_like(labels[..., -1:], loss_fn.ignore_index)))
    if not isinstance(logits, list):
        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))
    loss = loss_fn(logits, labels)
    del logits  # free logits otherwise peaks backward memory
    return loss
