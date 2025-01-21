import logging
import os
import sys
from functools import partial

from omegaconf import DictConfig, ListConfig
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
    world_size, rank = get_world_size_and_rank()  # more general
    shuffle = cfg_dataset.pop("shuffle")
    batch_size = cfg_dataset.pop("batch_size")
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
        drop_last=True,  # dropping last avoids shape issues with compile + flex attention
        collate_fn=(
            partial(padded_collate_sft, padding_idx=model_tokenizer.pad_id, ignore_idx=loss_fn.ignore_index)
            if not packed
            else padded_collate_packed
        ),
    )
    return dataloader, sampler


def pack_dataset(dataset: Dataset, tokenizer: Llama3Tokenizer) -> PackedDataset:
    if tokenizer.max_seq_len is None:
        raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
    return PackedDataset(dataset, max_seq_len=tokenizer.max_seq_len)
