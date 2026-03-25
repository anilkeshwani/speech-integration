from collections.abc import Callable
from functools import partial
import logging
import sys
from typing import Any

import datasets as hf_datasets
from omegaconf import DictConfig, ListConfig
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, DistributedSampler
import torchtune.data
from torchtune.data import padded_collate_packed
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.datasets import PackedDataset
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_world_size_and_rank

from ssi.constants import SEED
from ssi.data.cpt import TextCompletionDataset
from ssi.data.sft import SFTDataset


LOGGER = logging.getLogger(__name__)


def load_dataset_subset(
    source: str,
    n_samples: int,
    **load_dataset_kwargs,
) -> hf_datasets.Dataset:
    """Load the first ``n_samples`` from a HuggingFace dataset via streaming.

    Uses ``streaming=True`` so that only the requested rows are fetched over
    the network — the full dataset is never downloaded to disk.

    Args:
        source: HuggingFace dataset identifier (e.g. ``"anilkeshwani/mls-hubert_large_ll60k-layer_22"``).
        n_samples: Number of samples to materialize.
        **load_dataset_kwargs: Forwarded to ``datasets.load_dataset``
            (e.g. ``split="train"``).

    Returns:
        A standard ``datasets.Dataset`` containing the first *n_samples* rows.
    """
    if "split" not in load_dataset_kwargs:
        raise ValueError("load_dataset_subset requires a 'split' kwarg (e.g. split='train')")
    iterable = hf_datasets.load_dataset(source, streaming=True, **load_dataset_kwargs)
    rows = list(iterable.take(n_samples))
    LOGGER.info(
        f"Streamed {len(rows)}/{n_samples} samples from {source} (split={load_dataset_kwargs.get('split', '?')})"
    )
    return hf_datasets.Dataset.from_list(rows)


def setup_text_completion_data(
    cfg_dataset: DictConfig,
    model_tokenizer: Llama3Tokenizer,
    loss_fn: CEWithChunkedOutputLoss | None = None,
) -> tuple[DataLoader, DistributedSampler]:
    if isinstance(cfg_dataset, ListConfig):
        raise NotImplementedError("Support for the shuffle parameter needs to be added to use ConcatDataset.")
    if cfg_dataset.get("packed", False):
        # Strictly this doesn't have to affect CPT since we don't need to change the collate
        # function (as for SFT) but raised for consistency / as a reminder
        raise NotImplementedError("Need to add a custom collate function to handle the PACKED case - not implemented.")
    dataset = TextCompletionDataset(tokenizer=model_tokenizer, **cfg_dataset.dataset)
    if cfg_dataset.get("packed", False):
        dataset = pack_dataset(dataset, model_tokenizer, split_across_pack=cfg_dataset.get("split_across_pack", False))
        collate_fn = padded_collate_packed
    else:
        ignore_idx = CROSS_ENTROPY_IGNORE_IDX if loss_fn is None else loss_fn.ignore_index
        collate_fn = partial(
            padded_collate_sft,
            padding_idx=model_tokenizer.pad_id,
            ignore_idx=ignore_idx,
            additional_keys=cfg_dataset.dataset.get("additional_keys", []),
        )
    world_size, rank = get_world_size_and_rank()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg_dataset["shuffle"], seed=SEED)
    # NOTE dropping last avoids shape issues w/ compile + flex attention
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset.dataloader.batch_size,
        sampler=sampler,
        drop_last=cfg_dataset.dataloader.drop_last,
        collate_fn=collate_fn,
        num_workers=cfg_dataset.dataloader.get("num_workers", 0),
        persistent_workers=cfg_dataset.dataloader.get("persistent_workers", False),
    )
    LOGGER.info(f"Dataset and Sampler initialized from {cfg_dataset.dataset.source}.")
    return dataloader, sampler


def setup_sft_data(
    cfg_dataset: DictConfig,
    model_tokenizer: Llama3Tokenizer,
    loss_fn: CEWithChunkedOutputLoss | None = None,
) -> tuple[DataLoader, DistributedSampler]:
    if isinstance(cfg_dataset, ListConfig):
        raise NotImplementedError("Support for list of datasets not implemented")
    if cfg_dataset.get("packed", False):
        raise NotImplementedError("Need to add a custom collate function to handle the PACKED case - not implemented.")
    dataset = SFTDataset(model_tokenizer=model_tokenizer, **cfg_dataset.dataset)
    if cfg_dataset.get("packed", False):
        dataset = pack_dataset(dataset, model_tokenizer, split_across_pack=False)
        collate_fn = padded_collate_packed
    else:
        ignore_idx = CROSS_ENTROPY_IGNORE_IDX if loss_fn is None else loss_fn.ignore_index
        collate_fn = partial(
            padded_collate_sft,
            padding_idx=model_tokenizer.pad_id,
            ignore_idx=ignore_idx,
            additional_keys=cfg_dataset.dataset.get("additional_keys", []),
        )
    world_size, rank = get_world_size_and_rank()  # more general
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=cfg_dataset["shuffle"], seed=SEED)
    # NOTE dropping last avoids shape issues w/ compile + flex attention
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg_dataset.dataloader.batch_size,
        sampler=sampler,
        drop_last=cfg_dataset.dataloader.drop_last,
        collate_fn=collate_fn,
        num_workers=cfg_dataset.dataloader.get("num_workers", 0),
        persistent_workers=cfg_dataset.dataloader.get("persistent_workers", False),
    )
    return dataloader, sampler


####################################################################################################
# Collate function - custom to handle passing of additional_keys -> link samples to GT (e.g. ASR)
####################################################################################################


def padded_collate_sft(
    batch: list[dict[str, Any]],  # NOTE list[dict[str, list[int]]] in torchtune.data._collate.padded_collate_sft
    padding_idx: int = 0,
    ignore_idx: int = CROSS_ENTROPY_IGNORE_IDX,
    additional_keys: list[str] | None = None,
) -> dict[str, Tensor] | dict[str, Any]:
    """Pad a batch of sequences to the longest sequence length in the batch, and
    convert integer lists to tensors.

    Args:
        batch (list[dict[str, list[int]]]): A list of dictionaries containing input, label pairs.
        padding_idx (int): Padding index for input ids. Defaults to 0.
        ignore_idx (int): Padding index for labels. Defaults to -100.
        additional_keys (list[str]): Additional keys to collate. Useful for returning sample IDs.

    Returns:
        dict[str, torch.Tensor] | dict[str, Any]: Collated input and label tensors and optionally additional key fields.

    Example:
        >>> token_pairs = [
        >>>    {"tokens": [1, 2, 3], "labels": [4, 5, 6]},
        >>>    {"tokens": [7,], "labels": [10,]},
        >>> ]
        >>> collated = padded_collate(
        >>>    batch=token_pairs,
        >>>    padding_idx=padding_idx,
        >>>    ignore_idx=ignore_idx,
        >>> )
        >>> collated["tokens"]
        >>> tensor([[1, 2, 3], [7, 0, 0]])
        >>> collated["labels"]
        >>> tensor([[4, 5, 6], [10, -100, -100]])
    """
    if additional_keys is None:
        additional_keys = []
    input_ids = pad_sequence(
        [torch.tensor(x["tokens"]) for x in batch],
        batch_first=True,
        padding_value=padding_idx,
    )
    labels = pad_sequence(
        [torch.tensor(x["labels"]) for x in batch],
        batch_first=True,
        padding_value=ignore_idx,
    )

    input_ids_seq_len = input_ids.shape[-1]
    labels_seq_len = labels.shape[-1]

    # Hack to pad correctly and not use max_seq_len, which is costly
    if input_ids_seq_len > labels_seq_len:
        labels = F.pad(labels, (0, input_ids_seq_len - labels_seq_len), value=ignore_idx)
    elif labels_seq_len > input_ids_seq_len:
        input_ids = F.pad(
            input_ids,
            (0, labels_seq_len - input_ids_seq_len),
            value=padding_idx,
        )

    additional_keys_dict = {key: [x[key] for x in batch] for key in additional_keys}
    return {"tokens": input_ids.long(), "labels": labels.long()} | additional_keys_dict


def pack_dataset(dataset: Dataset, tokenizer: Llama3Tokenizer, split_across_pack: bool = False) -> PackedDataset:
    if tokenizer.max_seq_len is None:
        raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
    return PackedDataset(dataset, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack)


####################################################################################################
# Debug -> used to demonstate OOM error during SFT (Alpaca dataset is as provided by torchtune)
####################################################################################################


def setup_alpaca_data(
    tokenizer: Llama3Tokenizer,
    loss_fn: Callable,
    batch_size: int,
    shuffle: bool = True,
    collate_fn: Callable = torchtune.data.padded_collate_sft,
) -> tuple[DataLoader, DistributedSampler]:
    ds = torchtune.datasets.alpaca_dataset(tokenizer=tokenizer, packed=False)
    world_size, rank = get_world_size_and_rank()  # more general
    sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=shuffle, seed=SEED)
    dataloader = DataLoader(
        dataset=ds,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True,  # dropping last avoids shape issues with compile + flex attention
        collate_fn=partial(collate_fn, padding_idx=tokenizer.pad_id, ignore_idx=loss_fn.ignore_index),
    )
    LOGGER.info(f"Dataset and Sampler initialized: {ds._data}.")
    LOGGER.info(f"Data setup performed via: {sys._getframe().f_code.co_name}")
    return dataloader, sampler
