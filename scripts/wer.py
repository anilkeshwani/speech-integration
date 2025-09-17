#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat

from evaluate import load
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from whisper_normalizer.english import EnglishTextNormalizer

from ssi.constants import SUPPORTED_DATASETS
from ssi.utils import extract_texts_from_generations_jsonl, ref_from_hf_dataset


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Calculate Word Error Rate (WER) from model generations.")
    parser.add_argument("generations_jsonl", type=Path, help="Path to the JSON lines file with generations.")
    parser.add_argument("--dataset", type=str, help="Hugging Face dataset for reference transcripts.")
    parser.add_argument("--split", type=str, help="Hugging Face dataset split for reference transcripts.")
    parser.add_argument(
        "--gt_transcript_colname",
        type=str,
        default="transcript",
        help="Column name for ground truth transcripts in the dataset.",
    )
    parser.add_argument("--normalizer", type=str, default="whisper", choices=["whisper"], help="Text normalizer.")
    return parser.parse_args()


def main(args: Namespace) -> None:
    wer_json = args.generations_jsonl.parent / "wer.json"
    if wer_json.exists():
        with open(wer_json, "r") as f:
            _wer_json_contents = pformat(json.load(f))
        raise FileExistsError(f"Output WER JSON already exists: {wer_json} with contents: " + _wer_json_contents)
    wer_metric = load("wer")
    if args.dataset is None:
        args.dataset = args.generations_jsonl.parents[1].name
        assert args.dataset.split("-")[0] in SUPPORTED_DATASETS
        LOGGER.info(f"Inferred dataset from path: {args.dataset}")
    if args.split is None:
        args.split = args.generations_jsonl.parent.name
        LOGGER.info(f"Inferred split from path: {args.split}")
    generated = extract_texts_from_generations_jsonl(args.generations_jsonl)
    reference = ref_from_hf_dataset(args.dataset, args.split, args.gt_transcript_colname)
    if args.normalizer == "whisper":
        english_normalizer = EnglishTextNormalizer()  # TODO update for other languages
        generated = [english_normalizer(text) for text in generated]
        reference = [english_normalizer(text) for text in reference]
    elif args.normalizer is None:
        LOGGER.info("No normalizer specified, skipping text normalization.")
    else:
        raise NotImplementedError(f"Unsupported normalizer: {args.normalizer}. Supported: 'whisper' or None (null).")
    wer = wer_metric.compute(predictions=generated, references=reference)
    with open(wer_json, "x") as f:
        json.dump({"wer": wer}, f, indent=4)
    LOGGER.info(f"WER: {wer:.5f}.")
    LOGGER.info(f"Saved WER JSON to {wer_json!s}")


if __name__ == "__main__":
    main(parse_args())
