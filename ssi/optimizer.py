from typing import Any

from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torchtune.modules import TransformerDecoder


def setup_optimizer(
    cfg: DictConfig,
    model: TransformerDecoder,
    optimizer_state_dict: dict[str, Any] | None = None,
) -> AdamW:
    if cfg.optimizer_in_bwd:
        raise NotImplementedError
    optimizer = AdamW(model.parameters(), **cfg.optimizer)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    return optimizer
