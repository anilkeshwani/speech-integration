import logging
import os
import sys
from functools import partial

import omegaconf
from omegaconf import DictConfig, ListConfig, OmegaConf
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune.data import padded_collate_packed, padded_collate_sft
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


def setup_data(
    cfg_dataset: DictConfig,
    model_tokenizer: Llama3Tokenizer,
    loss_fn: CEWithChunkedOutputLoss,
) -> tuple[DataLoader, DistributedSampler]:
    cfg_dataset_is_struct = OmegaConf.is_struct(cfg_dataset)
    OmegaConf.set_struct(cfg_dataset, False)
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
    OmegaConf.set_struct(cfg_dataset, cfg_dataset_is_struct)
    return dataloader, sampler


def pack_dataset(dataset: Dataset, tokenizer: Llama3Tokenizer) -> PackedDataset:
    if tokenizer.max_seq_len is None:
        raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
    return PackedDataset(dataset, max_seq_len=tokenizer.max_seq_len)
