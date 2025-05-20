import logging
from typing import Any

import torch
from omegaconf import DictConfig
from torchtune import training
from torchtune.models.llama3_2 import llama3_2
from torchtune.modules import TransformerDecoder
from torchtune.training import get_dtype
from torchtune.utils import get_device

from ssi.llama_configs import ConfigLlama3_2


LOGGER = logging.getLogger(__name__)


def setup_llama3_2_1b(
    cfg: DictConfig,
    llama_config: ConfigLlama3_2,
    model_state_dict: dict[str, Any],
    dtype_default: torch.dtype | str = torch.get_default_dtype(),  # type: ignore
    device_default: torch.device | str = torch.get_default_device(),  # type: ignore
) -> TransformerDecoder:
    if isinstance(dtype_default, str):
        dtype_default: torch.dtype = get_dtype(cfg.dtype)
    if isinstance(device_default, str):
        device_default: torch.device = get_device(cfg.device)
    with training.set_default_dtype(dtype_default), device_default:
        model = llama3_2(**llama_config.parameters)
    if cfg.compile:
        training.compile_model(model)
    if cfg.enable_activation_checkpointing:
        raise NotImplementedError
        training.set_activation_checkpointing(model, auto_wrap_policy={modules.TransformerSelfAttentionLayer})
    model.load_state_dict(model_state_dict)  # load model weights
    training.validate_expected_param_dtype(model.named_parameters(), dtype=dtype_default)
    if cfg.enable_activation_offloading:
        raise NotImplementedError
        activations_handling_ctx = training.get_act_offloading_ctx_manager(model, cfg.enable_activation_offloading)
    if cfg.enable_activation_checkpointing and (not cfg.enable_activation_offloading):
        LOGGER.warning("Activation checkpointing is enabled but activation offloading is not.")
    return model
