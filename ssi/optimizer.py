from typing import Any

from omegaconf import DictConfig
from torch.optim import AdamW, Optimizer
from torchtune.modules import TransformerDecoder


def setup_optimizer(
    cfg: DictConfig,
    model: TransformerDecoder,
    optimzer_state_dict: dict[str, Any] | None = None,
) -> AdamW:
    if cfg.optimizer_in_bwd:
        raise NotImplementedError
    else: # if...else... retained for skeleton when adding optimizer_in_bwd support
        optimizer = AdamW(model.parameters(), **cfg.optimizer)
        if optimzer_state_dict is not None:
            optimizer.load_state_dict(optimzer_state_dict)
        return optimizer
