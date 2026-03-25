from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch.optim import AdamW
from torchtune.modules import TransformerDecoder


def setup_optimizer(
    cfg: DictConfig,
    model: TransformerDecoder,
    optimizer_state_dict: dict[str, Any] | None = None,
) -> AdamW:
    optimizer_kwargs = OmegaConf.to_container(cfg.optimizer, resolve=True)
    optimizer = AdamW(model.parameters(), **optimizer_kwargs)
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    return optimizer
