import time
from typing import Any, Callable

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.optim.optimizer import StateDict
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune import config, modules, training
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, scale_grads
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.metric_logging import WandBLogger
from torchtune.training.precision import PRECISION_STR_TO_DTYPE
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi.checkpoint import FullModelHFCheckpointer
from ssi.constants import EPOCHS_KEY, MODEL_KEY, OPTIMIZER_KEY, SEED, SEED_KEY, STEPS_KEY, TOTAL_EPOCHS_KEY
from ssi.data import setup_data
from ssi.data.sft import SFTDataset
from ssi.lr_schedule import setup_lr_scheduler
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer


# Debug mode
# `None` -> don't set any PyTorch global values
# "default" or 0 -> don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark
# "warn" or 1 -> warn on nondeterministic operations and disable PyTorch CuDNN benchmark
# "error" or 2 -> error on nondeterministic operations and disable PyTorch CuDNN benchmark
DEBUG_MODE: str | None = None

SUPPORTED_DTYPES: set[torch.dtype] = {torch.float32, torch.bfloat16}


def compute_loss(batch: dict[str, torch.Tensor], model: TransformerDecoder, loss_fn: Callable) -> torch.Tensor:
    labels = batch.pop("labels")  # shape [b, s] needed for the loss not the model
    logits = model(**batch)  # NOTE add activation offloading context
    labels = torch.hstack((labels[..., 1:], torch.full_like(labels[..., -1], loss_fn.ignore_index)))
    if not isinstance(logits, list):
        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))
    loss = loss_fn(logits, labels)
    del logits  # free logits otherwise peaks backward memory
    return loss


def validate_cfg(cfg: DictConfig) -> None:
    if PRECISION_STR_TO_DTYPE.get(cfg.dtype) not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Supported dtypes: {SUPPORTED_DTYPES}")

    if cfg.optimizer_in_bwd or cfg.enable_activation_checkpointing or cfg.enable_activation_offloading:
        raise NotImplementedError


def resume_training_state(ckpt_dict: dict[str, Any]) -> tuple[int, int, StateDict]:
    if SEED != ckpt_dict[SEED_KEY]:
        raise ValueError("Config value for seed does not match the checkpoint value")
    return ckpt_dict[EPOCHS_KEY], ckpt_dict[STEPS_KEY], ckpt_dict[OPTIMIZER_KEY]


@hydra.main(config_path="conf", config_name="sft.yaml", version_base=None)
def train(cfg: DictConfig) -> None:
    validate_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=DEBUG_MODE)
    DEVICE: torch.device = get_device(cfg.device)
    DTYPE: torch.dtype = get_dtype(cfg.dtype)
    wandb_logger = WandBLogger(cfg.wandb)
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
    optimizer: AdamW = setup_optimizer(cfg.optimizer, model, optimizer_state)
    lr_scheduler: LambdaLR | None = setup_lr_scheduler(
        cfg=cfg,
        optimizer=optimizer,
        last_epoch=-1 if ckpt_dict.get(STEPS_KEY) is None else ckpt_dict["EPOCHS_KEY"],  # if STEPS, require EPOCHS
        current_step=ckpt_dict.get(STEPS_KEY, 0),
        num_training_steps=cfg.max_steps,
    )
    loss_fn = CEWithChunkedOutputLoss()
    if cfg.compile:
        training.compile_loss(loss_fn)
    if isinstance(loss_fn, CEWithChunkedOutputLoss):
        model.set_num_output_chunks(loss_fn.num_output_chunks)
    data_train, sampler_train = setup_data(cfg_dataset=cfg.data.train, model_tokenizer=tokenizer, loss_fn=loss_fn)
    data_dev, sampler_dev = setup_data(cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer, loss_fn=loss_fn)
    optimizer.zero_grad()  # zero gradients before training # NOTE make conditional for optimizer_in_bwd
    t0 = time.perf_counter()
    loss_running = 0.0
    num_tokens = 0
    for epoch in range(epochs_run, cfg.epochs):
        sampler_train.set_epoch(epoch)  # distinct seed each epoch
        for i, batch in tqdm(enumerate(data_train)):
            # TODO time each iteration
            batch_to_device(batch, DEVICE)  # in-place
            # TODO calculate number of non-pad tokens
            num_tokens_curr = (batch["labels"] != loss_fn.ignore_index).sum()
            num_tokens += num_tokens_curr
            # loss is normalized -> multiply by number of tokens for renormalization later for grad. accum.
            loss_curr = compute_loss(batch, model, loss_fn) * num_tokens_curr
            loss_running += loss_curr
            loss_curr.backward()
            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scale_grads(model, 1 / num_tokens)
                if cfg.clip_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.clip_grad_norm))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if lr_scheduler is not None:
                    lr_scheduler.step()
                loss_to_log = loss_running.item() / num_tokens
                global_step += 1

                # log metrics
                if global_step % cfg.log_interval == 0:
                    dur_step = time.perf_counter() - t0
                    log_dict = {
                        "loss_train": loss_to_log,
                        "lr": get_lr(optimizer),
                        "duration_step": dur_step,
                        "tokens_per_second_per_gpu": num_tokens / dur_step,
                    }
                    if cfg.clip_grad_norm is not None:
                        log_dict.update({"grad_norm": grad_norm})
                    wandb_logger.log_dict(log_dict, step=global_step)

                # reset step-level tracker variables
                loss_running = 0.0
                num_tokens = 0
                t0 = time.perf_counter()


if __name__ == "__main__":
    train()
