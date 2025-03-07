from typing import Callable

import torch
from torchtune.modules import TransformerDecoder


def compute_loss(batch: dict[str, torch.Tensor], model: TransformerDecoder, loss_fn: Callable) -> torch.Tensor:
    logits = model(  # NOTE TODO add activation offloading context for model forward pass
        tokens=batch["tokens"],
        mask=batch.get("mask", None),
        encoder_input=batch.get("encoder_input", None),
        encoder_mask=batch.get("encoder_mask", None),
        input_pos=batch.get("input_pos", None),
    )
    labels = batch.pop("labels")  # shape [b, s] needed for the loss not the model
    labels = torch.hstack((labels[..., 1:], torch.full_like(labels[..., -1:], loss_fn.ignore_index)))
    if not isinstance(logits, list):
        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))
    loss = loss_fn(logits, labels)
    del logits  # free logits otherwise peaks backward memory # TODO should be freed when the function scope ends ???
    return loss
