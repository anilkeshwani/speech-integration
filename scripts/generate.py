#!/usr/bin/env python

import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
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
from ssi.constants import (
    DEBUGGING_TAG,
    EPOCHS_KEY,
    MODEL_KEY,
    OPTIMIZER_KEY,
    SEED,
    SEED_KEY,
    STEPS_KEY,
    SUPPORTED_DTYPES,
)
from ssi.data import setup_sft_data, setup_text_completion_data
from ssi.eval import compute_dataset_loss
from ssi.llama_configs import ConfigLlama3_2
from ssi.loss import compute_loss
from ssi.lr_schedule import setup_lr_scheduler
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.train import count_token_types, get_token_type_ranges, resume_training_state, validate_train_cfg


LOGGER = logging.getLogger(__name__)


def generate(cfg: DictConfig) -> None:
    # NOTE contains additional checks on optimizer_in_bwd, enable_activation_checkpointing, enable_activation_offloading
    validate_train_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)  # unfortunate this comes from "training" but it's good here
    DEVICE: torch.device = get_device(cfg.device)
    DTYPE: torch.dtype = get_dtype(cfg.dtype)
    if cfg.checkpointer.output_dir is None:
        cfg.checkpointer.output_dir = Path(cfg.output_dir, "evaluation", "checkpoints")
        LOGGER.info(f"No checkpointer output dir provided. Resolved to: {cfg.checkpointer.output_dir!s}")
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)  # TODO does this still create the output dir on init?
    ckpt_dict = checkpointer.load_checkpoint()
    model, llama_config = setup_llama3_2_1b(
        cfg=cfg,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=DTYPE,
        device_default=DEVICE,
    )
    model.to(device=DEVICE)
    model.eval()
    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    token_type_ranges = get_token_type_ranges(llama_config)
    loss_fn = CEWithChunkedOutputLoss()
    if cfg.compile:
        training.compile_loss(loss_fn)
    if isinstance(loss_fn, CEWithChunkedOutputLoss):
        model.set_num_output_chunks(loss_fn.num_output_chunks)
    # NOTE for now, we generate via an SFT dataset - this is flexible because we can modify the system prompt and also
    #      template the dataset columns via a PromptTemplate class, which can be specified as a dictionary in the YAML
    data_test, sampler_test = setup_sft_data(cfg_dataset=cfg.data.train, model_tokenizer=tokenizer)
    token_type_counts_total = defaultdict(int)
    tokens_test_total: int = 0
    test_loss_total: float = 0.0
    LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    t0 = time.perf_counter()
    for i, batch in tqdm(enumerate(data_test), total=len(data_test)):
        # Output the number of tokens to the console - do this before moving the batch to the GPU
        for tt, ttcnt in count_token_types(batch["tokens"], token_type_ranges, tokenizer.pad_id).items():
            token_type_counts_total[tt] += ttcnt
        LOGGER.info(" | ".join([f"Tokens ({tt}): {ttcnt}" for tt, ttcnt in token_type_counts_total.items()]))
        LOGGER.info(f"max_seq_len_step: {batch['tokens'].size(1)}")
        batch_to_device(batch, DEVICE)  # in-place
        num_tokens_iter = (batch["labels"] != loss_fn.ignore_index).sum()
        # loss is normalized -> multiply by number of tokens for renormalization later for grad. accum.
        loss_batch = compute_loss(batch, model, loss_fn) * num_tokens_iter
        tokens_test_total += num_tokens_iter  # total number of tokens trained on so far
        test_loss_total += loss_batch.item()  # total loss so far
        # log metrics to console batchwise
        loss_to_log = loss_batch.item() / num_tokens_iter  # loss per token per iteration
        LOGGER.info(" | ".join((f"[Iter {i}] Loss: {loss_to_log:.4f}", f"Tokens (num_tokens_iter): {num_tokens_iter}")))
        LOGGER.info(f"Time elapsed (iteration): {time.perf_counter() - t0:.2f} seconds")
        t0 = time.perf_counter()
        # TODO surely the explicit deletion of the batch is not necessary in inference mode??
        # del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
        # torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM
    LOGGER.info(f"Total test loss (unnormalised): {test_loss_total}")
    LOGGER.info(f"Total test loss (per token): {test_loss_total / tokens_test_total}")
    LOGGER.info(f"Total number of tokens: {tokens_test_total}")
