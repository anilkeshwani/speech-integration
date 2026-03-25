"""Stateful Trainer class for speech-integration training.

Encapsulates all training state (model, optimizer, scheduler, dataloaders,
step counters, cumulative metrics) as instance attributes. Provides the same
training logic as the functional ``train()`` in ``ssi.train``, but organized
into composable methods that can be tested and extended independently.

Design reference: torchtune's FTRecipeInterface — class-based for state
encapsulation, self-contained, composition over inheritance.
"""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
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
from torch.utils.data import DataLoader, DistributedSampler
from torchtune import training
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, get_world_size_and_rank, scale_grads
from torchtune.training.lr_schedulers import get_lr
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi._version import __version__
from ssi.checkpoint import FullModelHFCheckpointer, resolve_checkpointer_output_dir, restore_rng_states
from ssi.constants import DEBUGGING_TAG, MODEL_KEY, SEED
from ssi.data import setup_sft_data, setup_text_completion_data
from ssi.eval import compute_dataset_loss
from ssi.llama_configs import configllama3_2_1b
from ssi.loss import compute_loss
from ssi.lr_schedule import setup_lr_scheduler
from ssi.metric_logging import WandBLoggerPatched as WandBLogger
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.train_utils import (
    count_token_types,
    get_token_type_ranges,
    resume_training_state,
    validate_resume_hparams,
    validate_train_cfg,
)


__all__ = ["Trainer", "TrainingGeometry"]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingGeometry:
    """Derived constants that depend on dataset size and gradient accumulation.

    Encapsulates the training geometry computed from the dataloader length,
    gradient accumulation steps, and max training steps.

    Args:
        batch_size: Per-device batch size.
        batches_per_epoch: Total batches per epoch (``len(dataloader)``).
        steps_per_epoch: Optimizer steps per epoch (``batches_per_epoch // gradient_accumulation_steps``).
        usable_batches: Batches actually processed per epoch (``steps_per_epoch * gradient_accumulation_steps``).
        n_epochs: Total epochs needed to reach ``max_steps``.
        gradient_accumulation_steps: Number of micro-batches per optimizer step.
        world_size: Number of distributed processes.
    """

    batch_size: int
    batches_per_epoch: int
    steps_per_epoch: int
    usable_batches: int
    n_epochs: int
    gradient_accumulation_steps: int
    world_size: int

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig,
        dataloader: DataLoader,
        world_size: int,
    ) -> TrainingGeometry:
        batch_size = cfg.data.train.dataloader.batch_size
        batches_per_epoch = len(dataloader)
        gradient_accumulation_steps = cfg.gradient_accumulation_steps

        remainder_batches = batches_per_epoch % gradient_accumulation_steps
        if remainder_batches > 0:
            LOGGER.warning(
                f"batches_per_epoch ({batches_per_epoch}) is not divisible by "
                f"gradient_accumulation_steps ({gradient_accumulation_steps}): "
                f"{remainder_batches} remainder batches will be discarded at each epoch boundary."
            )

        steps_per_epoch = batches_per_epoch // gradient_accumulation_steps
        if steps_per_epoch <= 0:
            raise ValueError(
                f"batches_per_epoch ({batches_per_epoch}) < gradient_accumulation_steps ({gradient_accumulation_steps})"
            )

        usable_batches = steps_per_epoch * gradient_accumulation_steps
        n_epochs = math.ceil(cfg.max_steps / steps_per_epoch)

        return cls(
            batch_size=batch_size,
            batches_per_epoch=batches_per_epoch,
            steps_per_epoch=steps_per_epoch,
            usable_batches=usable_batches,
            n_epochs=n_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )


class Trainer:
    """Stateful trainer for speech-integration experiments.

    Encapsulates model, optimizer, scheduler, dataloaders, checkpointer, logger,
    and all training counters as instance attributes. Mirrors the logic of the
    functional ``ssi.train.train()`` exactly, enabling bit-identical training
    runs while allowing individual phases to be tested in isolation.

    Usage::

        trainer = Trainer(cfg)
        trainer.setup()
        trainer.train()
        trainer.cleanup()

    Args:
        cfg: Hydra DictConfig containing the full training configuration.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

        # Components — populated by setup()
        self.model: TransformerDecoder | None = None
        self.tokenizer: Llama3Tokenizer | None = None
        self.optimizer: AdamW | None = None
        self.lr_scheduler: LambdaLR | None = None
        self.loss_fn: CEWithChunkedOutputLoss | None = None
        self.checkpointer: FullModelHFCheckpointer | None = None
        self.wandb_logger: WandBLogger | None = None

        # Data — populated by setup()
        self.data_train: DataLoader | None = None
        self.sampler_train: DistributedSampler | None = None
        self.data_dev: DataLoader | None = None
        self.token_type_ranges: dict[str, tuple[int, int]] | None = None

        # Geometry — populated by setup()
        self.geometry: TrainingGeometry | None = None

        # Device and dtype — populated by setup()
        self.device: torch.device | None = None
        self.dtype: torch.dtype | None = None
        self.world_size: int | None = None

        # Training state — populated by setup(), updated during train()
        self.global_step: int = 0
        self.consumed_samples: int = 0
        self.tokens_train_total: int = 0
        self.token_type_counts_total: defaultdict[str, int] = defaultdict(int)
        self.wall_clock_offset: float = 0.0

        # Step-level accumulators — reset each optimizer step
        self.loss_running: float = 0.0
        self.num_tokens_step: int = 0
        self.max_seq_len_step: int = 0

        # Timing
        self.t_train_start: float = 0.0
        self.t_step_start: float = 0.0

        # Grad norm (tracked for logging when clip_grad_norm is set)
        self._grad_norm: float | None = None

        # Optional loss log for equivalence testing
        self._loss_log: list[float] | None = None

    # === Setup ===

    def setup(self) -> None:
        """Initialize all components: model, tokenizer, data, optimizer, loss, logging.

        Must be called before ``train()``. Handles checkpoint loading and
        optional resume from a training state checkpoint.
        """
        validate_train_cfg(self.cfg)
        training.set_seed(seed=SEED, debug_mode=self.cfg.debug_mode)
        self.device = get_device(self.cfg.device)
        self.dtype = get_dtype(self.cfg.dtype)
        self.world_size, _ = get_world_size_and_rank()

        self._setup_logging()
        self._setup_model()
        self._setup_tokenizer()
        # Extract resume state early (before optimizer, which needs it)
        self._extract_resume_state()
        self._setup_optimizer()
        self._setup_loss()
        self._setup_data()
        # geometry depends on data_train being set
        self.geometry = TrainingGeometry.from_config(self.cfg, self.data_train, self.world_size)
        # Validate and finalize resume (needs geometry for hparam validation)
        self._finalize_resume()
        # Free checkpoint dict — no longer needed after setup
        del self._ckpt_dict

    def _setup_logging(self) -> None:
        wandb_tags = [__version__, self.cfg.config_name]
        if os.getenv("SLURM_JOB_QOS") == "gpu-debug":
            wandb_tags += [DEBUGGING_TAG]
        self.wandb_logger = WandBLogger(**self.cfg.wandb, tags=wandb_tags)
        if self.cfg.checkpointer.output_dir is None:
            self.cfg.checkpointer.output_dir = resolve_checkpointer_output_dir(self.cfg, self.wandb_logger)
            LOGGER.info(f"No checkpointer output dir provided. Resolved to: {self.cfg.checkpointer.output_dir!s}")

    def _setup_model(self) -> None:
        # Deep-copy the singleton so multiple Trainers with different n_dsus
        # in the same process don't corrupt each other's config.
        self._llama_config = copy.deepcopy(configllama3_2_1b)
        self._llama_config.update_from_speech_cfg(self.cfg.speech)
        self.checkpointer = FullModelHFCheckpointer(
            **self.cfg.checkpointer,
            model_expectations=self._llama_config.checkpoint_expectations,
        )
        self._ckpt_dict = self.checkpointer.load_checkpoint()
        self.model = setup_llama3_2_1b(
            cfg=self.cfg,
            llama_config=self._llama_config,
            model_state_dict=self._ckpt_dict[MODEL_KEY],
            dtype_default=self.dtype,
            device_default=self.device,
        )
        self.model.to(device=self.device)
        self.model.train()

    def _setup_tokenizer(self) -> None:
        self.tokenizer, _special_tokens = setup_llama3_tokenizer(**self.cfg.tokenizer)
        self.token_type_ranges = get_token_type_ranges(llama_config=self._llama_config)

    def _setup_data(self) -> None:
        if self.cfg.config_name == "sft":
            self.data_train, self.sampler_train = setup_sft_data(
                cfg_dataset=self.cfg.data.train, model_tokenizer=self.tokenizer
            )
            self.data_dev, _sampler_dev = setup_sft_data(cfg_dataset=self.cfg.data.dev, model_tokenizer=self.tokenizer)
        elif self.cfg.config_name == "cpt":
            self.data_train, self.sampler_train = setup_text_completion_data(self.cfg.data.train, self.tokenizer)
            self.data_dev, _sampler_dev = setup_text_completion_data(self.cfg.data.dev, self.tokenizer)
        else:
            raise NotImplementedError(f"Unsupported config_name: {self.cfg.config_name}")

    def _extract_resume_state(self) -> None:
        """Extract resume state from checkpoint dict (early phase).

        Sets ``self._resume_state``, ``self.global_step``, and
        ``self.consumed_samples``. Must be called before ``_setup_optimizer``
        (which needs the optimizer state dict from the checkpoint).
        """
        self._resume_state: dict[str, Any] | None = None
        if self.checkpointer.training_state_checkpoint is not None:
            self._resume_state = resume_training_state(self._ckpt_dict)
            self.global_step = self._resume_state["global_step"]
            self.consumed_samples = self._resume_state["consumed_samples"]

    def _setup_optimizer(self) -> None:
        self.optimizer = setup_optimizer(
            self.cfg, self.model, self._resume_state["optimizer_state"] if self._resume_state else None
        )
        self.lr_scheduler = setup_lr_scheduler(
            cfg=self.cfg,
            optimizer=self.optimizer,
            global_step=self.global_step - 1,
            num_training_steps=self.cfg.max_steps,
        )
        if self._resume_state and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(self._resume_state["lr_scheduler_state"])

    def _setup_loss(self) -> None:
        self.loss_fn = CEWithChunkedOutputLoss()
        if self.cfg.compile:
            training.compile_loss(self.loss_fn)
        if isinstance(self.loss_fn, CEWithChunkedOutputLoss):
            self.model.set_num_output_chunks(self.loss_fn.num_output_chunks)

    def _finalize_resume(self) -> None:
        """Validate hparams and restore cumulative metrics + RNG (late phase).

        Must be called after geometry is computed (needs ``steps_per_epoch``
        for hparam validation).
        """
        if self._resume_state is None:
            return
        # Cumulative metrics
        cm = self._resume_state["cumulative_metrics"]
        self.tokens_train_total = cm["tokens_train_total"]
        for k, v in cm["token_type_counts"].items():
            self.token_type_counts_total[k] = v
        self.wall_clock_offset = cm["wall_clock_seconds"]
        # Validate hparams (needs geometry.steps_per_epoch)
        validate_resume_hparams(
            ckpt_hparams=self._resume_state["training_hparams"],
            current_hparams={
                "batch_size": self.geometry.batch_size,
                "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                "world_size": self.world_size,
                "steps_per_epoch": self.geometry.steps_per_epoch,
            },
            force_resume=self.cfg.get("force_resume", False),
        )

    # === Training ===

    def train(self) -> None:
        """Run the full training loop.

        Iterates over epochs and batches, performing gradient accumulation,
        periodic evaluation, logging, and checkpointing. Mirrors the logic
        of ``ssi.train.train()`` exactly.
        """
        self.optimizer.zero_grad()
        self.t_train_start = time.perf_counter()
        self.t_step_start = time.perf_counter()
        self._reset_step_accumulators()

        epochs_run = self.global_step // self.geometry.steps_per_epoch
        batches_to_skip = (self.global_step % self.geometry.steps_per_epoch) * self.cfg.gradient_accumulation_steps

        # Restore RNG states after all setup, before training loop
        if self._resume_state:
            restore_rng_states(self._resume_state["rng_state"])
            LOGGER.info("Restored framework RNG states from checkpoint.")
            self._resume_state = None  # free optimizer/RNG state references

        LOGGER.info(OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=False))
        self.wandb_logger.log_config(self.cfg)

        for epoch in range(epochs_run, self.geometry.n_epochs):
            self._train_epoch(epoch, batches_to_skip if epoch == epochs_run else 0)
            if self.global_step >= self.cfg.max_steps:
                LOGGER.info("Training completed.")
                return

    def _train_epoch(self, epoch: int, batches_to_skip: int = 0) -> None:
        if self.sampler_train is not None:
            self.sampler_train.set_epoch(epoch)
        if hasattr(self.data_train.dataset, "set_epoch"):
            self.data_train.dataset.set_epoch(epoch)

        if batches_to_skip > 0:
            LOGGER.info(f"Resuming: skipping {batches_to_skip} batches in epoch {epoch}")
            data_iter = itertools.islice(enumerate(self.data_train), batches_to_skip, self.geometry.usable_batches)
            n_batches = self.geometry.usable_batches - batches_to_skip
        else:
            data_iter = itertools.islice(enumerate(self.data_train), self.geometry.usable_batches)
            n_batches = self.geometry.usable_batches

        for i, batch in tqdm(data_iter, total=n_batches):
            self._train_step(batch)
            if (i + 1) % self.cfg.gradient_accumulation_steps == 0:
                self._optimizer_step(epoch, i)
                if self.global_step >= self.cfg.max_steps:
                    return
            del batch

    def _train_step(self, batch: dict[str, Tensor]) -> None:
        """Single micro-batch forward + backward pass."""
        batch_to_device(batch, self.device)
        for tt, ttcnt in count_token_types(batch["tokens"], self.token_type_ranges, self.tokenizer.pad_id).items():
            self.token_type_counts_total[tt] += ttcnt
        self.max_seq_len_step = max(self.max_seq_len_step, batch["tokens"].size(1))
        num_tokens_iter = int((batch["labels"] != self.loss_fn.ignore_index).sum().item())
        self.num_tokens_step += num_tokens_iter
        loss_batch = compute_loss(batch, self.model, self.loss_fn) * num_tokens_iter
        loss_batch.backward()
        self.loss_running += loss_batch.item()

    def _optimizer_step(self, epoch: int, iter_idx: int) -> None:
        """Gradient accumulation boundary: scale, clip, step, log, checkpoint."""
        scale_grads(self.model, torch.tensor(1 / self.num_tokens_step))
        if self.cfg.clip_grad_norm is not None:
            self._grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=float(self.cfg.clip_grad_norm)
            )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.global_step += 1
        self.consumed_samples += self.cfg.gradient_accumulation_steps * self.geometry.batch_size * self.world_size
        loss_to_log = self.loss_running / self.num_tokens_step
        self.tokens_train_total += self.num_tokens_step

        # Record loss for equivalence testing
        if self._loss_log is not None:
            self._loss_log.append(loss_to_log)

        self._log_metrics(epoch, iter_idx, loss_to_log)
        self._reset_step_accumulators()
        self._maybe_save_checkpoint()

    def _evaluate(self) -> float:
        """Compute dev set loss."""
        return compute_dataset_loss(
            self.model,
            self.data_dev,
            self.loss_fn,
            epoch=self.global_step // self.geometry.steps_per_epoch,
            global_step=self.global_step,
            steps_per_epoch=self.geometry.steps_per_epoch,
            device=self.device,
        )

    def _log_metrics(self, epoch: int, iter_idx: int, loss_to_log: float) -> None:
        """Log metrics to console and W&B."""
        LOGGER.info(
            " | ".join(
                (
                    f"Epoch {epoch + 1:03d}",
                    f"Iteration {iter_idx:0{len(str(self.geometry.batches_per_epoch))}d}"
                    f" / {self.geometry.batches_per_epoch}",
                    f"Global Step {self.global_step}",
                    f"Loss: {loss_to_log:.4f}",
                    f"Tokens (num_tokens_step): {self.num_tokens_step}",
                    *[f"Tokens ({tt}): {ttcnt}" for tt, ttcnt in self.token_type_counts_total.items()],
                )
            )
        )

        # Evaluate on dev set (kept as if/else to match functional train() structure)
        if self.global_step % self.cfg.eval_steps == 0:  # noqa: SIM108
            dev_loss = self._evaluate()
        else:
            dev_loss = None

        # W&B logging
        if self.global_step % self.cfg.log_interval == 0:
            dur_step = time.perf_counter() - self.t_step_start
            log_dict = {
                "loss": loss_to_log,
                "lr": get_lr(self.optimizer),
                "duration_step": dur_step,
                "tokens_per_second_per_gpu": self.num_tokens_step / dur_step,
                "tokens_total": self.tokens_train_total,
                "train_clock_time": (self.wall_clock_offset + (time.perf_counter() - self.t_train_start)) / (60**2),
                "max_seq_len_step": self.max_seq_len_step,
                **{f"n_tokens.{tt}": ttcnt for tt, ttcnt in self.token_type_counts_total.items()},
            }
            if self.cfg.clip_grad_norm is not None:
                log_dict.update({"grad_norm": self._grad_norm})
            if dev_loss is not None:
                log_dict.update({"dev_loss": dev_loss})
            self.wandb_logger.log_dict(log_dict, step=self.global_step)

    def _maybe_save_checkpoint(self) -> None:
        """Save checkpoint if at a save_steps boundary."""
        if self.global_step != 0 and self.global_step % self.cfg.save_steps == 0:
            self.save_checkpoint()
            LOGGER.info(f"Checkpoint saved at step {self.global_step}")

    def _reset_step_accumulators(self) -> None:
        """Reset per-optimizer-step accumulators."""
        self.loss_running = 0.0
        self.num_tokens_step = 0
        self.max_seq_len_step = 0
        self.t_step_start = time.perf_counter()

    # === Checkpointing ===

    def save_checkpoint(self) -> None:
        """Save model weights and training state."""
        self.checkpointer.save_model_checkpoint(self.model.state_dict(), self.global_step)
        self.checkpointer.save_training_state(
            optimizer_state_dict=self.optimizer.state_dict(),
            lr_scheduler_state_dict=self.lr_scheduler.state_dict() if self.lr_scheduler else None,
            global_step=self.global_step,
            seed=SEED,
            training_hparams={
                "batch_size": self.geometry.batch_size,
                "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
                "world_size": self.world_size,
                "steps_per_epoch": self.geometry.steps_per_epoch,
            },
            consumed_samples=self.consumed_samples,
            cumulative_metrics={
                "tokens_train_total": self.tokens_train_total,
                "token_type_counts": dict(self.token_type_counts_total),
                "wall_clock_seconds": self.wall_clock_offset + (time.perf_counter() - self.t_train_start),
            },
        )

    # === Cleanup ===

    def cleanup(self) -> None:
        """Teardown (currently a no-op; future: DDP process group destruction)."""
        pass
