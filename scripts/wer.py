import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

from datasets import load_dataset
from evaluate import load
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from whisper_normalizer.english import EnglishTextNormalizer

from ssi.constants import HF_OWNER, SUPPORTED_DATASETS


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)


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


def ref_from_hf_dataset(dataset: str, split: str, gt_transcript_colname: str = "transcript") -> list[str]:
    repo_id = HF_OWNER + "/" + dataset
    ds = load_dataset(repo_id, split=split)
    return list(ds[gt_transcript_colname])


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
        raise FileExistsError(f"Output WER JSON already exists: {wer_json}")
    wer_metric = load("wer")
    if args.dataset is None:
        args.dataset = args.generations_jsonl.parents[1].name
        assert args.dataset.split("-")[0] in SUPPORTED_DATASETS
    if args.split is None:
        args.split = args.generations_jsonl.parent.name
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
