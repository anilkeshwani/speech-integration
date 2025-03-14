import os
import pdb
import sys
import traceback

import torch
from torchtune.utils._import_guard import _SUPPORTS_FLEX_ATTENTION


if _SUPPORTS_FLEX_ATTENTION:
    from torch.nn.attention.flex_attention import BlockMask
else:
    BlockMask = torch.Tensor


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
