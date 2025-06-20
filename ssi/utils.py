import os
import pdb
import sys
import traceback
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION


if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor


def parse_model_path(model_dir: Path, experiments_root_dir: Path) -> dict[str, Any]:
    if not model_dir.is_relative_to(experiments_root_dir):
        raise ValueError(
            f"Model directory must be in the experiments root directory. "
            f"Got model_dir: {model_dir}. experiments_root_dir: {experiments_root_dir}"
        )
    model_training, wandb_dir, _, epoch_dir, global_step_dir = model_dir.relative_to(experiments_root_dir).parts
    *wandb_run_name_parts, wandb_run_id_prefixed = wandb_dir.split("-")
    wandb_run_name = "-".join(wandb_run_name_parts)
    wandb_run_id = wandb_run_id_prefixed.removeprefix("id_")
    *base_model_parts, training_type = model_training.split("-")
    base_model_name = "-".join(base_model_parts)
    epoch = int(epoch_dir.removeprefix("epoch_"))
    global_step = int(global_step_dir.removeprefix("global_step_"))
    return {
        "base_model_name": base_model_name,
        "training_type": training_type,
        "wandb_run_id": wandb_run_id,
        "wandb_run_name": wandb_run_name,
        "epoch": epoch,
        "global_step": global_step,
    }


def hash_cfg(cfg: DictConfig, length: int = 7) -> str:
    """Compute truncated SHA-256 hex hash of resolved and sorted DictConfig"""
    return sha256(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True).encode()).hexdigest()[:length]


def info_excepthook(type, value, tb):
    if hasattr(sys, "ps1") or not sys.stderr.isatty():
        # interactive mode or we don't have a tty-like device: call the default hook
        sys.__excepthook__(type, value, tb)
    else:
        # NOT in interactive mode: print the exception then start the debugger in post-mortem mode
        traceback.print_exception(type, value, tb)
        pdb.post_mortem(tb)


def get_terminal_width(default_width: int = 120) -> int:
    try:
        TERMINAL_WIDTH = os.get_terminal_size().columns
    except OSError:
        TERMINAL_WIDTH = default_width
    return TERMINAL_WIDTH


def batch_to_device(batch: dict, device: torch.device, exclude_keys: list[str] = []) -> None:
    """Function that takes a dictionary (or nested dictionary) of tensors and sets them
    all to the same device. This utility is intended to be used for batches of data to be
    moved to device, the update is inplace.

    Args:
        batch (dict): dict of Tensors or more nested dicts of tensors.
        device (torch.device): torch device to move the tensor's too
        exclude_keys (list[str]): keys to exclude from moving to device (top-level only)

    Raises:
        AttributeError: if batch dict contains anything other than tensors
    """
    for k, v in batch.items():
        if k in exclude_keys:
            continue  # skip this key

        if isinstance(v, dict):
            batch_to_device(v, device, [])  # NOTE explicit - we only exclude keys at the *top* level
        elif isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif _SUPPORTS_FLEX_ATTENTION and isinstance(v, BlockMask):
            batch[k] = v.to(device)
        else:
            raise ValueError(
                f"""To use batch_to_device, all elements in the batch must be a dict or Tensor.
Got key "{k}" with value of type {type(v)}"""
            )
