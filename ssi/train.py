from typing import Callable

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, LRScheduler
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune import config, modules, training
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules import TransformerDecoder
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype
from torchtune.training.precision import PRECISION_STR_TO_DTYPE
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi.constants import MODEL_KEY, OPTIMIZER_KEY, SEED
from ssi.data import setup_data
from ssi.data.sft import SFTDataset
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer


SUPPORTED_DTYPES = [torch.float32, torch.bfloat16]


def compute_loss(batch: dict[str, torch.Tensor], model: TransformerDecoder, loss_fn: Callable) -> torch.Tensor:
    labels = batch.pop("labels")  # shape [b, s] needed for the loss not the model
    logits = model(**batch)  # NOTE add activation offloading context
    labels = torch.hstack((labels[..., 1:], torch.full_like(labels[..., -1], loss_fn.ignore_index)))
    if not isinstance(logits, list):
        labels = labels.reshape(-1)
        logits = logits.reshape(-1, logits.size(-1))
    loss = loss_fn(logits, labels)
    del logits  # free logits otherwise it peaks backward memory
    return loss


def validate_cfg(cfg: DictConfig) -> None:
    if PRECISION_STR_TO_DTYPE.get(cfg.dtype) not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Supported dtypes: {SUPPORTED_DTYPES}")

    if cfg.optimizer_in_bwd or cfg.enable_activation_checkpointing or cfg.enable_activation_offloading:
        raise NotImplementedError


def train(cfg: DictConfig) -> None:
    training.set_seed(seed=cfg.seed, debug_mode=cfg.debug_mode)
    device_default: torch.device = get_device(cfg.device)
    dtype_default: torch.dtype = get_dtype(cfg.dtype)
    model: TransformerDecoder = setup_llama3_2_1b(
        cfg=cfg,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=dtype_default,
        device_default=device_default,
    )
    model.to(device=device_default)
    model.train()

    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)

    optimizer: AdamW = setup_optimizer(cfg.optimizer, model, ckpt_dict.get(OPTIMIZER_KEY))  # NOTE optim. ckpt optional

    loss_fn = CEWithChunkedOutputLoss()

    if cfg.compile:
        training.compile_loss(loss_fn)

    if isinstance(loss_fn, CEWithChunkedOutputLoss):
        model.set_num_output_chunks(loss_fn.num_output_chunks)

    lr_lambda = lambda epoch: 1.0
    lr_scheduler: LRScheduler
    sampler: DistributedSampler

    dataset_train = setup_data(cfg_dataset=cfg.data.train, model_tokenizer=tokenizer)
    dataset_train = setup_data(cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer)

    optimizer.zero_grad()  # zero gradients before training # NOTE make conditional for optimizer_in_bwd

    for epoch in range(cfg.epochs):
        sampler.set_epoch(epoch)  # distinct seed each epoch
        for i, batch in tqdm(enumerate(data)):
            pass
