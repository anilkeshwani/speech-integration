import logging
import math
import os
import sys
import time
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.optimizer import StateDict
from torchtune import training
from torchtune.modules import TransformerDecoder
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, scale_grads
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.metric_logging import WandBLogger
from torchtune.training.precision import PRECISION_STR_TO_DTYPE
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi.checkpoint import FullModelHFCheckpointer, resolve_checkpointer_output_dir
from ssi.constants import EPOCHS_KEY, MODEL_KEY, OPTIMIZER_KEY, SEED, SEED_KEY, STEPS_KEY, SUPPORTED_DTYPES
from ssi.data import setup_sft_data, setup_text_completion_data
from ssi.eval import compute_dataset_loss
from ssi.loss import compute_loss
from ssi.lr_schedule import setup_lr_scheduler
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer


################################################################################
# Global Settings
################################################################################

# Config to use; see conf/ directory
CONFIG_NAME = "sft.yaml"

# Debug mode
"""
`None` -> don't set any PyTorch global values
"default" or 0 -> don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark
"warn" or 1 -> warn on nondeterministic operations and disable PyTorch CuDNN benchmark
"error" or 2 -> error on nondeterministic operations and disable PyTorch CuDNN benchmark
"""
DEBUG_MODE: str | None = None

################################################################################
# Preamble
################################################################################

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)

################################################################################
# Training helper functions
################################################################################


def validate_cfg(cfg: DictConfig) -> None:
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


################################################################################
# Training
################################################################################


@hydra.main(config_path="conf", config_name=CONFIG_NAME, version_base=None)
def train(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=DEBUG_MODE)
    DEVICE: torch.device = get_device(cfg.device)
    DTYPE: torch.dtype = get_dtype(cfg.dtype)
    wandb_logger = WandBLogger(**cfg.wandb)
    if cfg.checkpointer.output_dir is None:
        cfg.checkpointer.output_dir = resolve_checkpointer_output_dir(cfg, wandb_logger)
        LOGGER.info(f"No checkpointer output dir provided. Resolved to: {cfg.checkpointer.output_dir!s}")
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    model: TransformerDecoder = setup_llama3_2_1b(
        cfg=cfg,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=DTYPE,
        device_default=DEVICE,
    )
    model.to(device=DEVICE)
    model.train()
    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
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
    if CONFIG_NAME == "sft.yaml":
        data_train, sampler_train = setup_sft_data(cfg_dataset=cfg.data.train, model_tokenizer=tokenizer)
        data_dev, sampler_dev = setup_sft_data(cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer)
    elif CONFIG_NAME == "cpt.yaml":
        data_train, sampler_train = setup_text_completion_data(cfg.data.train, tokenizer)
        data_dev, sampler_dev = setup_text_completion_data(cfg.data.dev, tokenizer)
    else:
        raise NotImplementedError
    optimizer.zero_grad()  # zero gradients before training # NOTE make conditional for optimizer_in_bwd
    t_train_start = time.perf_counter()
    t0 = time.perf_counter()
    loss_running = 0.0
    num_tokens = 0  # TODO rename to num_tokens_step
    tokens_train_total: int = 0
    steps_per_epoch = len(data_train) // cfg.gradient_accumulation_steps
    n_epochs = math.ceil(cfg.max_steps / steps_per_epoch)
    LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    for epoch in range(epochs_run, n_epochs):
        sampler_train.set_epoch(epoch)  # distinct seed each epoch
        for i, batch in tqdm(enumerate(data_train), total=len(data_train)):
            batch_to_device(batch, DEVICE)  # in-place
            num_tokens_batch = (batch["labels"] != loss_fn.ignore_index).sum()
            num_tokens += num_tokens_batch
            # loss is normalized -> multiply by number of tokens for renormalization later for grad. accum.
            loss_batch = compute_loss(batch, model, loss_fn) * num_tokens_batch
            loss_running += loss_batch
            loss_batch.backward()
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scale_grads(model, 1 / num_tokens)
                if cfg.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.clip_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                global_step += 1
                loss_to_log = loss_running.item() / num_tokens  # loss per token
                tokens_train_total += num_tokens  # total number of tokens trained on so far
                # TODO add separate speech and text token counters
                # log metrics to console
                LOGGER.info(
                    f"Epoch {epoch + 1:03d} | "
                    f"Iter {i:0{len(str(steps_per_epoch))}d} / {steps_per_epoch} | "
                    f"Global Step {global_step:0{len(str(steps_per_epoch))}d} | "  # TODO bad zero padding
                    f"Loss: {loss_to_log:.4f}"
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
                        "tokens_per_second_per_gpu": num_tokens / dur_step,
                        "tokens_total": tokens_train_total,  # TODO check everything OK
                        # TODO add separate speech and text token counters
                        "train_clock_time": (time.perf_counter() - t_train_start) / (60**2),
                    }
                    if cfg.clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                    if dev_loss is not None:
                        log_dict.update({"dev_loss": dev_loss})
                    wandb_logger.log_dict(log_dict, step=global_step)
                # reset step-level tracker variables
                loss_running = 0.0
                num_tokens = 0
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
            # del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
            # torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM


################################################################################
# Script entry point
################################################################################

if __name__ == "__main__":
    train()
