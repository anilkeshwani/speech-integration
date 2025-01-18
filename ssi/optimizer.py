from typing import Any

from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torchtune.modules import TransformerDecoder


def setup_optimizer(
    cfg: DictConfig,
    model: TransformerDecoder,
    optimzer_state_dict: dict[str, Any] | None = None,
) -> Optimizer | None:
    """
    Set up the optimizer. This method also handles loading the optimizer state_dict, if specified.
    """
    if cfg.optimizer_in_bwd:
        raise NotImplementedError
    else:
        optimizer = AdamW(model.parameters(), **cfg.optimizer)
        if optimzer_state_dict:
            optimizer.load_state_dict(optimzer_state_dict)
        return optimizer
