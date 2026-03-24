"""Pure utility functions for training: config validation, resume state
parsing, token type accounting.

These are stateless helpers used by :class:`ssi.trainer.Trainer` and by
other modules (checkpoint tests, plotting scripts, etc.).  They were
originally defined inline in the monolithic ``train()`` function and
extracted during the stateful Trainer refactor.
"""

import logging
from typing import Any

from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torchtune.training.precision import PRECISION_STR_TO_DTYPE

from ssi.constants import (
    CHECKPOINT_VERSION,
    CHECKPOINT_VERSION_KEY,
    CONSUMED_SAMPLES_KEY,
    CUMULATIVE_METRICS_KEY,
    GLOBAL_STEP_KEY,
    LR_SCHEDULER_KEY,
    OPTIMIZER_KEY,
    RNG_KEY,
    SEED,
    SEED_KEY,
    SUPPORTED_DTYPES,
    TRAINING_HPARAMS_KEY,
)
from ssi.llama_configs import ConfigLlama3_2


LOGGER = logging.getLogger(__name__)


def validate_train_cfg(cfg: DictConfig) -> None:
    if PRECISION_STR_TO_DTYPE.get(cfg.dtype) not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Supported dtypes: {SUPPORTED_DTYPES}")

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Missing keys in config: {missing_keys}")

    positive_int_fields = ("gradient_accumulation_steps", "max_steps", "log_interval", "eval_steps", "save_steps")
    for field in positive_int_fields:
        if cfg.get(field, 0) <= 0:
            raise ValueError(f"Config field '{field}' must be a positive integer, got: {cfg.get(field)}")

    if cfg.save_steps % cfg.eval_steps != 0:
        raise ValueError(f"save_steps ({cfg.save_steps}) must be a multiple of eval_steps ({cfg.eval_steps})")


def resume_training_state(ckpt_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract and validate resume state from a versioned checkpoint dict."""
    if CHECKPOINT_VERSION_KEY not in ckpt_dict:
        raise ValueError(
            "Checkpoint predates the versioned schema (no 'checkpoint_version' key). "
            "Legacy checkpoints are not supported. Start a fresh training run."
        )
    ckpt_version = ckpt_dict[CHECKPOINT_VERSION_KEY]
    if ckpt_version != CHECKPOINT_VERSION:
        raise ValueError(
            f"Checkpoint version mismatch: checkpoint has version {ckpt_version}, "
            f"but this code expects version {CHECKPOINT_VERSION}."
        )
    if ckpt_dict[SEED_KEY] != SEED:
        raise ValueError(f"Seed mismatch: config={SEED}, checkpoint={ckpt_dict[SEED_KEY]}")
    return {
        "global_step": ckpt_dict[GLOBAL_STEP_KEY],
        "optimizer_state": ckpt_dict[OPTIMIZER_KEY],
        "lr_scheduler_state": ckpt_dict[LR_SCHEDULER_KEY],
        "rng_state": ckpt_dict[RNG_KEY],
        "training_hparams": ckpt_dict[TRAINING_HPARAMS_KEY],
        "consumed_samples": ckpt_dict[CONSUMED_SAMPLES_KEY],
        "cumulative_metrics": ckpt_dict[CUMULATIVE_METRICS_KEY],
    }


def validate_resume_hparams(
    ckpt_hparams: dict[str, Any],
    current_hparams: dict[str, Any],
    force_resume: bool = False,
) -> None:
    """Validate that training hyperparameters match between checkpoint and current config."""
    for key in ("batch_size", "gradient_accumulation_steps", "world_size", "steps_per_epoch"):
        if key in ckpt_hparams and ckpt_hparams[key] != current_hparams[key]:
            msg = (
                f"Training hparam mismatch on resume for '{key}': "
                f"checkpoint={ckpt_hparams[key]}, current={current_hparams[key]}. "
                f"This breaks the step-to-data-position mapping."
            )
            if force_resume:
                LOGGER.warning(msg)
            else:
                raise ValueError(msg)


def get_token_type_ranges(llama_config: ConfigLlama3_2) -> dict[str, tuple[int, int]]:
    """Produce inclusive ranges for each token type in the vocabulary."""
    ranges: dict[str, tuple[int, int]] = {
        "text": (0, llama_config._base_vocab_size_txt - 1),
        "dsu": (llama_config._base_vocab_size_txt, llama_config._base_vocab_size_txt + llama_config.n_dsus - 1),
    }
    offset = llama_config._base_vocab_size_txt + llama_config.n_dsus
    if llama_config.modality_tokens:
        ranges["modality"] = (offset, offset + 1)
        offset += 2
    # NOTE special_text category includes padding token; usually "<|finetune_right_pad_id|>": 128004 for Llama 3.2 tknzr
    ranges["special_text"] = (offset, offset + llama_config._n_special_txt - 1)

    offset += llama_config._n_special_txt  # resolve offset to final vocab size -> check ranges cover whole vocabulary
    if offset != llama_config.vocab_size:
        raise ValueError(f"Vocab vs token ranges mismatch: {offset} != {llama_config.vocab_size}")
    if "total" in ranges:
        raise AssertionError('"total" key reserved')  # NOTE this avoids hot loop if placed in token_type_counts; TODO
    return ranges


def count_token_types(tokens: Tensor, ranges: dict[str, tuple[int, int]], pad_idx: int) -> dict[str, int]:
    """Count the number of tokens of each type in the given tensor.

    Args:
        tokens: The tensor containing the token IDs.
        ranges: A dictionary mapping token types to their ID ranges.
        pad_idx: Padding token index (excluded from "total" count).

    Returns:
        A dictionary mapping token types to their counts.
    """
    counts = {}
    for token_type, (start, end) in ranges.items():
        counts[token_type] = ((tokens >= start) & (tokens <= end)).sum().item()
    counts["total"] = (tokens != pad_idx).sum().item()
    return counts
