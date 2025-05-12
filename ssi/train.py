import logging
import math
import os
import sys
import time
from collections import defaultdict
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import StateDict
from torchtune import training
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, scale_grads
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.metric_logging import WandBLogger
from torchtune.training.precision import PRECISION_STR_TO_DTYPE
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi._version import __version__
from ssi.checkpoint import FullModelHFCheckpointer, resolve_checkpointer_output_dir
from ssi.constants import EPOCHS_KEY, MODEL_KEY, OPTIMIZER_KEY, SEED, SEED_KEY, STEPS_KEY, SUPPORTED_DTYPES
from ssi.data import setup_sft_data, setup_text_completion_data
from ssi.eval import compute_dataset_loss
from ssi.llama_configs import ConfigLlama3_2
from ssi.loss import compute_loss
from ssi.lr_schedule import setup_lr_scheduler
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def validate_train_cfg(cfg: DictConfig) -> None:
    if PRECISION_STR_TO_DTYPE.get(cfg.dtype) not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Supported dtypes: {SUPPORTED_DTYPES}")

    if cfg.optimizer_in_bwd or cfg.enable_activation_checkpointing or cfg.enable_activation_offloading:
        raise NotImplementedError

    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Missing keys in config: {missing_keys}")


def resume_training_state(ckpt_dict: dict[str, Any]) -> tuple[int, int, StateDict]:
    if SEED != ckpt_dict[SEED_KEY]:
        raise ValueError("Config value for seed does not match the checkpoint value")
    return ckpt_dict[EPOCHS_KEY], ckpt_dict[STEPS_KEY], ckpt_dict[OPTIMIZER_KEY]


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
    """
    Count the number of tokens of each type in the given tensor.

    Args:
        tokens (Tensor): The tensor containing the token IDs.
        ranges (dict[str, tuple[int, int]]): A dictionary mapping token types to their ID ranges.

    Returns:
        dict[str, int]: A dictionary mapping token types to their counts.
    """
    counts = {}
    for token_type, (start, end) in ranges.items():
        counts[token_type] = ((tokens >= start) & (tokens <= end)).sum().item()
    counts["total"] = (tokens != pad_idx).sum().item()
    return counts


def train(cfg: DictConfig) -> None:
    validate_train_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)
    DEVICE: torch.device = get_device(cfg.device)
    DTYPE: torch.dtype = get_dtype(cfg.dtype)
    wandb_logger = WandBLogger(**cfg.wandb, tags=[__version__])
    if cfg.checkpointer.output_dir is None:
        cfg.checkpointer.output_dir = resolve_checkpointer_output_dir(cfg, wandb_logger)
        LOGGER.info(f"No checkpointer output dir provided. Resolved to: {cfg.checkpointer.output_dir!s}")
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    model, llama_config = setup_llama3_2_1b(
        cfg=cfg,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=DTYPE,
        device_default=DEVICE,
    )
    model.to(device=DEVICE)
    model.train()
    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    token_type_ranges = get_token_type_ranges(llama_config)
    epochs_run, global_step, optimizer_state = 0, 0, None
    if checkpointer.recipe_checkpoint is not None:  # cfg.checkpoint.recipe_checkpoint Path
        epochs_run, global_step, optimizer_state = resume_training_state(ckpt_dict)
    optimizer: AdamW = setup_optimizer(cfg, model, optimizer_state)
    lr_scheduler: LambdaLR | None = setup_lr_scheduler(
        cfg=cfg,
        optimizer=optimizer,
        global_step=-1 if global_step == 0 else global_step,
        num_training_steps=cfg.max_steps,
    )
    loss_fn = CEWithChunkedOutputLoss()
    if cfg.compile:
        training.compile_loss(loss_fn)
    if isinstance(loss_fn, CEWithChunkedOutputLoss):
        model.set_num_output_chunks(loss_fn.num_output_chunks)
    # TODO clean this up later -> use hydra.utils.instantiate (requires refactoring configs)
    if cfg.config_name == "sft.yaml":
        data_train, sampler_train = setup_sft_data(cfg_dataset=cfg.data.train, model_tokenizer=tokenizer)
        data_dev, sampler_dev = setup_sft_data(cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer)
    elif cfg.config_name == "cpt.yaml":
        data_train, sampler_train = setup_text_completion_data(cfg.data.train, tokenizer)
        data_dev, sampler_dev = setup_text_completion_data(cfg.data.dev, tokenizer)
    else:
        raise NotImplementedError
    optimizer.zero_grad()  # zero gradients before training # NOTE make conditional for optimizer_in_bwd
    t_train_start = time.perf_counter()
    t0 = time.perf_counter()
    loss_running = 0.0
    token_type_counts_total = defaultdict(int)
    num_tokens_step = 0
    tokens_train_total: int = 0
    steps_per_epoch = len(data_train) // cfg.gradient_accumulation_steps
    n_epochs = math.ceil(cfg.max_steps / steps_per_epoch)
    LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    wandb_logger.log_config(cfg)  # log config after parameter resolution + overrides
    for epoch in range(epochs_run, n_epochs):
        sampler_train.set_epoch(epoch)  # distinct seed each epoch
        for i, batch in tqdm(enumerate(data_train), total=len(data_train)):
            batch_to_device(batch, DEVICE)  # in-place
            for tt, ttcnt in count_token_types(batch["tokens"], token_type_ranges, tokenizer.pad_id).items():
                token_type_counts_total[tt] += ttcnt
            num_tokens_iter = (batch["labels"] != loss_fn.ignore_index).sum()
            num_tokens_step += num_tokens_iter
            # loss is normalized -> multiply by number of tokens for renormalization later for grad. accum.
            loss_batch = compute_loss(batch, model, loss_fn) * num_tokens_iter
            loss_running += loss_batch
            loss_batch.backward()
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scale_grads(model, 1 / num_tokens_step)
                if cfg.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.clip_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                global_step += 1
                loss_to_log = loss_running.item() / num_tokens_step  # loss per token
                tokens_train_total += num_tokens_step  # total number of tokens trained on so far
                # log metrics to console
                LOGGER.info(
                    " | ".join(
                        (
                            f"Epoch {epoch + 1:03d}",
                            f"Iter {i:0{len(str(steps_per_epoch))}d} / {steps_per_epoch}",
                            f"Global Step {global_step:0{len(str(steps_per_epoch))}d}",  # TODO bad zero padding
                            f"Loss: {loss_to_log:.4f}",
                            f"Tokens (num_tokens_step): {num_tokens_step}",
                            *[f"Tokens ({tt}): {ttcnt}" for tt, ttcnt in token_type_counts_total.items()],
                        )
                    )
                )
                # validate (evaluate on dev set)
                if global_step % cfg.eval_steps == 0:
                    dev_loss = compute_dataset_loss(
                        model, data_dev, loss_fn, epoch, global_step, steps_per_epoch, DEVICE
                    )
                else:
                    dev_loss = None
                # log metrics to wandb
                if global_step % cfg.log_interval == 0:
                    dur_step = time.perf_counter() - t0
                    log_dict = {
                        "loss": loss_to_log,
                        "lr": get_lr(optimizer),
                        "duration_step": dur_step,
                        "tokens_per_second_per_gpu": num_tokens_step / dur_step,
                        "tokens_total": tokens_train_total,
                        "train_clock_time": (time.perf_counter() - t_train_start) / (60**2),
                        **{f"n_tokens.{tt}": ttcnt for tt, ttcnt in token_type_counts_total.items()},
                    }
                    if cfg.clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                    if dev_loss is not None:
                        log_dict.update({"dev_loss": dev_loss})
                    wandb_logger.log_dict(log_dict, step=global_step)
                # reset step-level tracker variables
                loss_running = 0.0
                num_tokens_step = 0
                t0 = time.perf_counter()
                # Save checkpoint
                if global_step != 0 and global_step % cfg.save_steps == 0:
                    checkpointer.save_checkpoint(
                        model_state_dict=model.state_dict(),
                        optimizer_state_dict=optimizer.state_dict(),
                        epoch=epoch,
                        global_step=global_step,
                        seed=SEED,
                    )
                    LOGGER.info(f"Checkpoint saved at step {global_step:0{len(str(steps_per_epoch))}d}")  # TODO 0s pad
                if global_step >= cfg.max_steps:
                    LOGGER.info("Training completed.")
                    break
            del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
            torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM
