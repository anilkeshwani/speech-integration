import logging
import os
import sys
from functools import partial

import omegaconf
from omegaconf import DictConfig, ListConfig, OmegaConf
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune import config
from torchtune.config._utils import _get_component_from_path
from torchtune.data import padded_collate_packed, padded_collate_sft
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import PackedDataset
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_world_size_and_rank

from ssi.data.sft import SFTDataset


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def setup_text_completion_data(
    cfg_dataset: DictConfig,
    batch_size: int,
    model_tokenizer: Llama3Tokenizer,
    collate_fn: str | None = None,  # type: ignore # TODO avoid changing type
) -> tuple[DataLoader, DistributedSampler]:
    cfg_dataset_is_struct = OmegaConf.is_struct(cfg_dataset)
    OmegaConf.set_struct(cfg_dataset, False)  # TODO
    if isinstance(cfg_dataset, ListConfig):
        raise NotImplementedError("Support for the shuffle parameter needs to be added to use ConcatDataset.")
    shuffle = cfg_dataset.pop("shuffle")
    ds = config.instantiate(cfg_dataset, model_tokenizer)
    packed = cfg_dataset.get("packed", False)
    if collate_fn is not None:
        if "left_pad_sequence" in collate_fn:
            raise RuntimeError("left_pad_sequence collator is only for inference.")
        collate_fn: Callable = _get_component_from_path(collate_fn)  # type: ignore
    else:
        collate_fn = padded_collate_sft
    sampler = DistributedSampler(ds, num_replicas=1, rank=0, shuffle=shuffle, seed=0)  # TODO
    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,  # dropping last avoids shape issues with compile + flex attention
        collate_fn=(
            partial(collate_fn, padding_idx=model_tokenizer.pad_id, ignore_idx=CROSS_ENTROPY_IGNORE_IDX)
            if not packed
            else padded_collate_packed
        ),
    )
    LOGGER.info(f"Dataset and Sampler initialized from {cfg_dataset.source}.")
    OmegaConf.set_struct(cfg_dataset, cfg_dataset_is_struct)  # TODO
    return dataloader, sampler


def setup_sft_data(
    cfg_dataset: DictConfig,
    model_tokenizer: Llama3Tokenizer,
    loss_fn: CEWithChunkedOutputLoss,
) -> tuple[DataLoader, DistributedSampler]:
    cfg_dataset_is_struct = OmegaConf.is_struct(cfg_dataset)
    OmegaConf.set_struct(cfg_dataset, False)  # TODO
    world_size, rank = get_world_size_and_rank()  # more general
    # NOTE we mutate the cfg_dataset, even if we restore the struct setting
    shuffle = cfg_dataset.pop("shuffle")
    batch_size = cfg_dataset.pop("batch_size")
    drop_last = cfg_dataset.pop("drop_last")
    packed = cfg_dataset.pop("packed", False)

    if isinstance(cfg_dataset, ListConfig):
        raise NotImplementedError
    else:
        dataset = SFTDataset(**cfg_dataset, model_tokenizer=model_tokenizer)

    if packed:
        dataset = pack_dataset(dataset, model_tokenizer)

    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=0)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=drop_last,  # dropping last avoids shape issues with compile + flex attention
        collate_fn=(
            partial(padded_collate_sft, padding_idx=model_tokenizer.pad_id, ignore_idx=loss_fn.ignore_index)
            if not packed
            else padded_collate_packed
        ),
    )
    OmegaConf.set_struct(cfg_dataset, cfg_dataset_is_struct)  # TODO
    return dataloader, sampler


def pack_dataset(dataset: Dataset, tokenizer: Llama3Tokenizer) -> PackedDataset:
    if tokenizer.max_seq_len is None:
        raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
    return PackedDataset(dataset, max_seq_len=tokenizer.max_seq_len)
