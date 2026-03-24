from collections import defaultdict
import itertools
import logging
import math
import os
import time
from typing import Any

from omegaconf import DictConfig, OmegaConf
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchtune import training
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, get_world_size_and_rank, scale_grads
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.precision import PRECISION_STR_TO_DTYPE
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi._version import __version__
from ssi.checkpoint import FullModelHFCheckpointer, resolve_checkpointer_output_dir, restore_rng_states
from ssi.constants import (
    CHECKPOINT_VERSION,
    CHECKPOINT_VERSION_KEY,
    CONSUMED_SAMPLES_KEY,
    CUMULATIVE_METRICS_KEY,
    DEBUGGING_TAG,
    GLOBAL_STEP_KEY,
    LR_SCHEDULER_KEY,
    MODEL_KEY,
    OPTIMIZER_KEY,
    RNG_KEY,
    SEED,
    SEED_KEY,
    SUPPORTED_DTYPES,
    TRAINING_HPARAMS_KEY,
)
from ssi.data import setup_sft_data, setup_text_completion_data
from ssi.eval import compute_dataset_loss
from ssi.llama_configs import ConfigLlama3_2, configllama3_2_1b
from ssi.loss import compute_loss
from ssi.lr_schedule import setup_lr_scheduler
from ssi.metric_logging import WandBLoggerPatched as WandBLogger
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer


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
    world_size, _ = get_world_size_and_rank()
    wandb_tags = [__version__, cfg.config_name]
    if os.getenv("SLURM_JOB_QOS") == "gpu-debug":
        wandb_tags += [DEBUGGING_TAG]
    wandb_logger = WandBLogger(**cfg.wandb, tags=wandb_tags)
    if cfg.checkpointer.output_dir is None:
        cfg.checkpointer.output_dir = resolve_checkpointer_output_dir(cfg, wandb_logger)
        LOGGER.info(f"No checkpointer output dir provided. Resolved to: {cfg.checkpointer.output_dir!s}")
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    configllama3_2_1b.update_from_speech_cfg(cfg.speech)  # in-place
    model = setup_llama3_2_1b(
        cfg=cfg,
        llama_config=configllama3_2_1b,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=DTYPE,
        device_default=DEVICE,
    )
    model.to(device=DEVICE)
    model.train()
    tokenizer, _special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    token_type_ranges = get_token_type_ranges(llama_config=configllama3_2_1b)

    # === Resume state ===
    global_step = 0
    consumed_samples = 0
    resume_state: dict[str, Any] | None = None
    if checkpointer.training_state_checkpoint is not None:
        resume_state = resume_training_state(ckpt_dict)
        global_step = resume_state["global_step"]
        consumed_samples = resume_state["consumed_samples"]

    optimizer: AdamW = setup_optimizer(cfg, model, resume_state["optimizer_state"] if resume_state else None)
    lr_scheduler: LambdaLR | None = setup_lr_scheduler(
        cfg=cfg,
        optimizer=optimizer,
        global_step=global_step - 1,  # see setup_lr_scheduler: LambdaLR steps once on init
        num_training_steps=cfg.max_steps,
    )
    if resume_state and lr_scheduler is not None:
        lr_scheduler.load_state_dict(resume_state["lr_scheduler_state"])

    loss_fn = CEWithChunkedOutputLoss()
    if cfg.compile:
        training.compile_loss(loss_fn)
    if isinstance(loss_fn, CEWithChunkedOutputLoss):
        model.set_num_output_chunks(loss_fn.num_output_chunks)
    # TODO clean this up later -> use hydra.utils.instantiate (requires refactoring configs)
    if cfg.config_name == "sft":
        data_train, sampler_train = setup_sft_data(cfg_dataset=cfg.data.train, model_tokenizer=tokenizer)
        data_dev, _sampler_dev = setup_sft_data(cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer)
    elif cfg.config_name == "cpt":
        data_train, sampler_train = setup_text_completion_data(cfg.data.train, tokenizer)
        data_dev, _sampler_dev = setup_text_completion_data(cfg.data.dev, tokenizer)
    else:
        raise NotImplementedError

    # === Derived training geometry ===
    batch_size = cfg.data.train.dataloader.batch_size
    batches_per_epoch = len(data_train)
    remainder_batches = batches_per_epoch % cfg.gradient_accumulation_steps
    if remainder_batches > 0:
        LOGGER.warning(
            f"batches_per_epoch ({batches_per_epoch}) is not divisible by "
            f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}): "
            f"{remainder_batches} remainder batches will be discarded at each epoch boundary."
        )
    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    assert steps_per_epoch > 0, (
        f"batches_per_epoch ({batches_per_epoch}) < gradient_accumulation_steps ({cfg.gradient_accumulation_steps})"
    )
    n_epochs = math.ceil(cfg.max_steps / steps_per_epoch)

    # === Validate hparams on resume ===
    if resume_state:
        validate_resume_hparams(
            ckpt_hparams=resume_state["training_hparams"],
            current_hparams={
                "batch_size": batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "world_size": world_size,
                "steps_per_epoch": steps_per_epoch,
            },
            force_resume=cfg.get("force_resume", False),
        )

    # === Resume position ===
    epochs_run = global_step // steps_per_epoch
    batches_to_skip = (global_step % steps_per_epoch) * cfg.gradient_accumulation_steps

    # usable_batches: only full accumulation windows — remainder batches are not processed
    usable_batches = steps_per_epoch * cfg.gradient_accumulation_steps

    # === Cumulative metrics (restore or initialize) ===
    tokens_train_total: int = 0
    token_type_counts_total: defaultdict[str, int] = defaultdict(int)
    wall_clock_offset: float = 0.0
    if resume_state:
        cm = resume_state["cumulative_metrics"]
        tokens_train_total = cm["tokens_train_total"]
        for k, v in cm["token_type_counts"].items():
            token_type_counts_total[k] = v
        wall_clock_offset = cm["wall_clock_seconds"]

    # === Restore framework RNG states (after all setup, before training loop) ===
    if resume_state:
        restore_rng_states(resume_state["rng_state"])
        LOGGER.info("Restored framework RNG states from checkpoint.")

    # === Training loop ===
    optimizer.zero_grad()
    t_train_start = time.perf_counter()
    t0 = time.perf_counter()
    loss_running = 0.0
    num_tokens_step = 0
    max_seq_len_step = 0
    LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    wandb_logger.log_config(cfg)  # log config after parameter resolution + overrides
    for epoch in range(epochs_run, n_epochs):
        sampler_train.set_epoch(epoch)  # distinct seed each epoch
        if hasattr(data_train.dataset, "set_epoch"):
            data_train.dataset.set_epoch(epoch)
        # Skip already-processed batches on resume -> neatly eliminates need to zero grads for these
        if epoch == epochs_run and batches_to_skip > 0:
            LOGGER.info(f"Resuming: skipping {batches_to_skip} batches in epoch {epoch}")
            data_iter = itertools.islice(enumerate(data_train), batches_to_skip, usable_batches)
            n_batches = usable_batches - batches_to_skip
        else:
            data_iter = itertools.islice(enumerate(data_train), usable_batches)
            n_batches = usable_batches
        for i, batch in tqdm(data_iter, total=n_batches):
            batch_to_device(batch, DEVICE)  # in-place
            for tt, ttcnt in count_token_types(batch["tokens"], token_type_ranges, tokenizer.pad_id).items():
                token_type_counts_total[tt] += ttcnt
            max_seq_len_step = max(max_seq_len_step, batch["tokens"].size(1))
            num_tokens_iter = int((batch["labels"] != loss_fn.ignore_index).sum().item())
            num_tokens_step += num_tokens_iter
            # loss is normalized -> multiply by number of tokens for renormalization later for grad. accum.
            loss_batch = compute_loss(batch, model, loss_fn) * num_tokens_iter
            loss_running += loss_batch
            loss_batch.backward()
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scale_grads(model, torch.tensor(1 / num_tokens_step))
                if cfg.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.clip_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                global_step += 1
                consumed_samples += cfg.gradient_accumulation_steps * batch_size * world_size
                loss_to_log = loss_running.item() / num_tokens_step  # loss per token
                tokens_train_total += num_tokens_step  # total number of tokens trained on so far
                # log metrics to console
                LOGGER.info(
                    " | ".join(
                        (
                            f"Epoch {epoch + 1:03d}",
                            f"Iteration {i:0{len(str(batches_per_epoch))}d} / {batches_per_epoch}",
                            f"Global Step {global_step}",
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
                        "train_clock_time": (wall_clock_offset + (time.perf_counter() - t_train_start)) / (60**2),
                        "max_seq_len_step": max_seq_len_step,
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
                max_seq_len_step = 0
                t0 = time.perf_counter()
                # Save checkpoint
                if global_step != 0 and global_step % cfg.save_steps == 0:
                    checkpointer.save_model_checkpoint(model.state_dict(), global_step)
                    checkpointer.save_training_state(
                        optimizer_state_dict=optimizer.state_dict(),
                        lr_scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
                        global_step=global_step,
                        seed=SEED,
                        training_hparams={
                            "batch_size": batch_size,
                            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                            "world_size": world_size,
                            "steps_per_epoch": steps_per_epoch,
                        },
                        consumed_samples=consumed_samples,
                        cumulative_metrics={
                            "tokens_train_total": tokens_train_total,
                            "token_type_counts": dict(token_type_counts_total),
                            "wall_clock_seconds": wall_clock_offset + (time.perf_counter() - t_train_start),
                        },
                    )
                    LOGGER.info(f"Checkpoint saved at step {global_step}")
                if global_step >= cfg.max_steps:
                    LOGGER.info("Training completed.")
                    return
            del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
            torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM
