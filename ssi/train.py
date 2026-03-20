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


class Trainer:
    """Stateful trainer for speech-language model continued pretraining and supervised fine-tuning."""

    def __init__(self, cfg: DictConfig) -> None:
        validate_train_cfg(cfg)
        self.cfg = cfg
        self.device = get_device(cfg.device)
        self.dtype = get_dtype(cfg.dtype)
        self.world_size, _ = get_world_size_and_rank()

    def setup(self) -> None:
        cfg = self.cfg
        training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)

        # === Logging ===
        wandb_tags = [__version__, cfg.config_name]
        if os.getenv("SLURM_JOB_QOS") == "gpu-debug":
            wandb_tags += [DEBUGGING_TAG]
        self.wandb_logger = WandBLogger(**cfg.wandb, tags=wandb_tags)
        if cfg.checkpointer.output_dir is None:
            cfg.checkpointer.output_dir = resolve_checkpointer_output_dir(cfg, self.wandb_logger)
            LOGGER.info(f"No checkpointer output dir provided. Resolved to: {cfg.checkpointer.output_dir!s}")

        # === Checkpointer ===
        self.checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
        ckpt_dict = self.checkpointer.load_checkpoint()

        # === Model ===
        configllama3_2_1b.update_from_speech_cfg(cfg.speech)  # in-place
        self.model = setup_llama3_2_1b(
            cfg=cfg,
            llama_config=configllama3_2_1b,
            model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
            dtype_default=self.dtype,
            device_default=self.device,
        )
        self.model.to(device=self.device)
        self.model.train()

        # === Tokenizer ===
        self.tokenizer, _special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
        self.token_type_ranges = get_token_type_ranges(llama_config=configllama3_2_1b)

        # === Resume state ===
        self.global_step = 0
        self.consumed_samples = 0
        resume_state: dict[str, Any] | None = None
        if self.checkpointer.recipe_checkpoint is not None:
            resume_state = resume_training_state(ckpt_dict)
            self.global_step = resume_state["global_step"]
            self.consumed_samples = resume_state["consumed_samples"]

        # === Optimizer ===
        self.optimizer: AdamW = setup_optimizer(
            cfg, self.model, resume_state["optimizer_state"] if resume_state else None
        )

        # === LR scheduler ===
        self.lr_scheduler: LambdaLR | None = setup_lr_scheduler(
            cfg=cfg,
            optimizer=self.optimizer,
            global_step=self.global_step - 1,  # see setup_lr_scheduler: LambdaLR steps once on init
            num_training_steps=cfg.max_steps,
        )
        if resume_state and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(resume_state["lr_scheduler_state"])

        # === Loss function ===
        self.loss_fn = CEWithChunkedOutputLoss()
        if cfg.compile:
            training.compile_loss(self.loss_fn)
        if isinstance(self.loss_fn, CEWithChunkedOutputLoss):
            self.model.set_num_output_chunks(self.loss_fn.num_output_chunks)

        # === Data ===
        # TODO clean this up later -> use hydra.utils.instantiate (requires refactoring configs)
        if cfg.config_name == "sft":
            self.data_train, self.sampler_train = setup_sft_data(
                cfg_dataset=cfg.data.train, model_tokenizer=self.tokenizer
            )
            self.data_dev, _sampler_dev = setup_sft_data(
                cfg_dataset=cfg.data.dev, model_tokenizer=self.tokenizer
            )
        elif cfg.config_name == "cpt":
            self.data_train, self.sampler_train = setup_text_completion_data(cfg.data.train, self.tokenizer)
            self.data_dev, _sampler_dev = setup_text_completion_data(cfg.data.dev, self.tokenizer)
        else:
            raise NotImplementedError

        # === Derived training geometry ===
        self.batch_size = cfg.data.train.dataloader.batch_size
        self.batches_per_epoch = len(self.data_train)
        remainder_batches = self.batches_per_epoch % cfg.gradient_accumulation_steps
        if remainder_batches > 0:
            LOGGER.warning(
                f"batches_per_epoch ({self.batches_per_epoch}) is not divisible by "
                f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps}): "
                f"{remainder_batches} remainder batches will be discarded at each epoch boundary."
            )
        self.steps_per_epoch = self.batches_per_epoch // cfg.gradient_accumulation_steps
        assert self.steps_per_epoch > 0, (
            f"batches_per_epoch ({self.batches_per_epoch}) < "
            f"gradient_accumulation_steps ({cfg.gradient_accumulation_steps})"
        )
        self.usable_batches = self.steps_per_epoch * cfg.gradient_accumulation_steps
        self.n_epochs = math.ceil(cfg.max_steps / self.steps_per_epoch)

        # === Validate hparams on resume ===
        if resume_state:
            validate_resume_hparams(
                ckpt_hparams=resume_state["training_hparams"],
                current_hparams=self.training_hparams,
                force_resume=cfg.get("force_resume", False),
            )

        # === Resume position ===
        self.epochs_run = self.global_step // self.steps_per_epoch
        self.batches_to_skip = (self.global_step % self.steps_per_epoch) * cfg.gradient_accumulation_steps

        # === Cumulative metrics (restore or initialize) ===
        self.tokens_train_total: int = 0
        self.token_type_counts_total: defaultdict[str, int] = defaultdict(int)
        self.wall_clock_offset: float = 0.0
        if resume_state:
            cm = resume_state["cumulative_metrics"]
            self.tokens_train_total = cm["tokens_train_total"]
            for k, v in cm["token_type_counts"].items():
                self.token_type_counts_total[k] = v
            self.wall_clock_offset = cm["wall_clock_seconds"]

        # === Restore framework RNG states (after all setup, before training loop) ===
        if resume_state:
            restore_rng_states(resume_state["rng_state"])
            LOGGER.info("Restored framework RNG states from checkpoint.")

    def train(self) -> None:
        cfg = self.cfg

        # === Training loop ===
        self.optimizer.zero_grad()
        self._t_train_start = time.perf_counter()
        t0 = time.perf_counter()
        loss_running = 0.0
        num_tokens_step = 0
        max_seq_len_step = 0
        grad_norm = None
        LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
        self.wandb_logger.log_config(cfg)  # log config after parameter resolution + overrides
        for epoch in range(self.epochs_run, self.n_epochs):
            self.epoch = epoch
            self.sampler_train.set_epoch(self.epoch)  # distinct seed each epoch
            if hasattr(self.data_train.dataset, "set_epoch"):
                self.data_train.dataset.set_epoch(self.epoch)
            # Skip already-processed batches on resume -> neatly eliminates need to zero grads for these
            if self.epoch == self.epochs_run and self.batches_to_skip > 0:
                LOGGER.info(f"Resuming: skipping {self.batches_to_skip} batches in epoch {self.epoch}")
                data_iter = itertools.islice(
                    enumerate(self.data_train), self.batches_to_skip, self.usable_batches
                )
                n_batches = self.usable_batches - self.batches_to_skip
            else:
                data_iter = itertools.islice(enumerate(self.data_train), self.usable_batches)
                n_batches = self.usable_batches
            for i, batch in tqdm(data_iter, total=n_batches):
                batch_to_device(batch, self.device)  # in-place
                for tt, ttcnt in count_token_types(
                    batch["tokens"], self.token_type_ranges, self.tokenizer.pad_id
                ).items():
                    self.token_type_counts_total[tt] += ttcnt
                max_seq_len_step = max(max_seq_len_step, batch["tokens"].size(1))
                num_tokens_iter = int((batch["labels"] != self.loss_fn.ignore_index).sum().item())
                num_tokens_step += num_tokens_iter
                # loss is normalized -> multiply by number of tokens for renormalization later for grad. accum.
                loss_batch = compute_loss(batch, self.model, self.loss_fn) * num_tokens_iter
                loss_running += loss_batch
                loss_batch.backward()
                if (i + 1) % cfg.gradient_accumulation_steps == 0:
                    scale_grads(self.model, 1 / num_tokens_step)
                    if cfg.clip_grad_norm is not None:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=float(cfg.clip_grad_norm)
                        )
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()
                    self.global_step += 1
                    self.consumed_samples += cfg.gradient_accumulation_steps * self.batch_size * self.world_size
                    loss_to_log = loss_running.item() / num_tokens_step  # loss per token
                    self.tokens_train_total += num_tokens_step  # total number of tokens trained on so far
                    # log metrics to console
                    LOGGER.info(
                        " | ".join(
                            (
                                f"Epoch {self.epoch + 1:03d}",
                                f"Iteration {i:0{len(str(self.batches_per_epoch))}d} / {self.batches_per_epoch}",
                                f"Global Step {self.global_step}",
                                f"Loss: {loss_to_log:.4f}",
                                f"Tokens (num_tokens_step): {num_tokens_step}",
                                *[
                                    f"Tokens ({tt}): {ttcnt}"
                                    for tt, ttcnt in self.token_type_counts_total.items()
                                ],
                            )
                        )
                    )
                    # evaluate on dev set
                    dev_loss = None
                    if self.global_step % cfg.eval_steps == 0:
                        dev_loss = self.evaluate()
                    # log metrics to wandb
                    if self.global_step % cfg.log_interval == 0:
                        dur_step = time.perf_counter() - t0
                        self.log_metrics(
                            loss=loss_to_log,
                            dev_loss=dev_loss,
                            dur_step=dur_step,
                            grad_norm=grad_norm,
                            num_tokens_step=num_tokens_step,
                            max_seq_len_step=max_seq_len_step,
                        )
                    # reset step-level tracker variables
                    loss_running = 0.0
                    num_tokens_step = 0
                    max_seq_len_step = 0
                    t0 = time.perf_counter()
                    # Save checkpoint
                    if self.global_step != 0 and self.global_step % cfg.save_steps == 0:
                        self.save_checkpoint()
                    if self.global_step >= cfg.max_steps:
                        LOGGER.info("Training completed.")
                        return
                del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
                torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM

    def save_checkpoint(self) -> None:
        self.checkpointer.save_model_checkpoint(self.model.state_dict(), self.global_step)
        self.checkpointer.save_training_state(
            optimizer_state_dict=self.optimizer.state_dict(),
            lr_scheduler_state_dict=self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            global_step=self.global_step,
            seed=SEED,
            training_hparams=self.training_hparams,
            consumed_samples=self.consumed_samples,
            cumulative_metrics=self.cumulative_metrics,
        )
        LOGGER.info(f"Checkpoint saved at step {self.global_step}")

    def evaluate(self) -> float:
        return compute_dataset_loss(
            self.model, self.data_dev, self.loss_fn,
            self.epoch, self.global_step, self.steps_per_epoch, self.device,
        )

    def log_metrics(
        self,
        *,
        loss: float,
        dev_loss: float | None,
        dur_step: float,
        grad_norm: float | None,
        num_tokens_step: int,
        max_seq_len_step: int,
    ) -> None:
        log_dict = {
            "loss": loss,
            "lr": get_lr(self.optimizer),
            "duration_step": dur_step,
            "tokens_per_second_per_gpu": num_tokens_step / dur_step,
            "tokens_total": self.tokens_train_total,
            "train_clock_time": (self.wall_clock_offset + (time.perf_counter() - self._t_train_start)) / (60**2),
            "max_seq_len_step": max_seq_len_step,
            **{f"n_tokens.{tt}": ttcnt for tt, ttcnt in self.token_type_counts_total.items()},
        }
        if self.cfg.clip_grad_norm is not None:
            log_dict.update({"grad_norm": grad_norm})
        if dev_loss is not None:
            log_dict.update({"dev_loss": dev_loss})
        self.wandb_logger.log_dict(log_dict, step=self.global_step)

    @property
    def training_hparams(self) -> dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "world_size": self.world_size,
            "steps_per_epoch": self.steps_per_epoch,
        }

    @property
    def cumulative_metrics(self) -> dict[str, Any]:
        return {
            "tokens_train_total": self.tokens_train_total,
            "token_type_counts": dict(self.token_type_counts_total),
            "wall_clock_seconds": self.wall_clock_offset + (time.perf_counter() - self._t_train_start),
        }


def train(cfg: DictConfig) -> None:
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.train()
