#!/usr/bin/env python

"""
Script to tabulate frequencies of DSUs given a Hugging Face dataset.
"""

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from tqdm import tqdm


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Tabulate DSU frequencies in a Hugging Face dataset")
    parser.add_argument("dataset_name", type=str, help="Hugging Face dataset")
    parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: train)")
    parser.add_argument(
        "--dsus_column", type=str, default="speech_tokens", help="Column name containing DSUs (default: speech_tokens)"
    )
    parser.add_argument(
        "--output_file", type=Path, default=None, help="Output file to save DSU frequencies (default: stdout)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_name, split=args.split)
    LOGGER.info(f"Loaded dataset {args.dataset_name} split {args.split}")

    if args.dsus_column not in dataset.column_names:
        raise ValueError(f"Column '{args.dsus_column}' not found in dataset columns: {dataset.column_names}")

    # Count DSU frequencies
    dsu_counter = Counter()
    for example in tqdm(dataset):
        dsus: list[int] = example[args.dsus_column]
        if not isinstance(dsus, list) and isinstance(dsus[0], int):  # type: ignore
            raise ValueError(f"Expected list of integers in column '{args.dsus_column}', got {type(dsus)}")
        dsu_counter.update(dsus)

    # Sort DSUs by frequency
    sorted_dsus = dsu_counter.most_common()

    # Output results
    if args.output_file:
        LOGGER.info(f"Saving DSU frequencies to {args.output_file}")
        with open(args.output_file, "x") as f:
            for dsu, freq in sorted_dsus:
                f.write(f"{dsu}\t{freq}\n")
    else:
        LOGGER.info("DSU Frequencies:")
        for dsu, freq in sorted_dsus:
            print(f"{dsu}\t{freq}")


if __name__ == "__main__":
    main()
