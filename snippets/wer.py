#!/usr/bin/env python

import json
import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import load_dataset
from evaluate import load
from whisper_normalizer.english import EnglishTextNormalizer


LOGGER = logging.getLogger(__name__)

HF_REPO_OWNER = "anilkeshwani"
TRANSCRIPTS_COLNAME = "transcript"


def extract_texts_from_generations_jsonl(generations_jsonl: Path) -> list[str]:
    texts = []
    with open(generations_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            is_single_generation = len(data["outputs"]) == 1
            if is_single_generation:
                texts.append(data.pop("outputs").pop(0).pop("text"))
            else:
                raise NotImplementedError("Multiple generations per prompt are not supported by this script.")
    return texts


def ref_from_generation_path(generations_jsonl: Path) -> list[str]:
    split, repo_name = [par.name for par in generations_jsonl.parents[:2]]
    ds = load_dataset(f"{HF_REPO_OWNER}/{repo_name}", split=split)
    return list(ds[TRANSCRIPTS_COLNAME])  # GT text


def main(args: Namespace) -> None:
    wer_metric = load("wer")
    generations_jsonl = Path(args.generations_jsonl)
    generated = extract_texts_from_generations_jsonl(generations_jsonl)
    reference = ref_from_generation_path(args.generations_jsonl)
    if args.normalizer == "whisper":
        english_normalizer = EnglishTextNormalizer()  # TODO update for other languages
        generated = [english_normalizer(text) for text in generated]
        reference = [english_normalizer(text) for text in reference]
    elif args.normalizer is None:
        LOGGER.info("No normalizer specified, skipping text normalization.")
    else:
        raise NotImplementedError(f"Unsupported normalizer: {args.normalizer}. Supported: 'whisper' or None (null).")
    wer = wer_metric.compute(predictions=generated, references=reference)
    wer_json = generations_jsonl.parent / "wer.json"
    with open(wer_json, "x") as f:
        json.dump({"wer": wer}, f, indent=4)
    LOGGER.info(f"WER: {wer:.3f}. Saved to {wer_json!s}")


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Compute WER from generated text.")
    parser.add_argument("generations_jsonl", type=Path, help="Path to the JSONL file containing generations.")
    parser.add_argument("--normalizer", type=str, default="whisper", help="Text normalizer. Default: 'whisper'")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
