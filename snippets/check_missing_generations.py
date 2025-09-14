#!/usr/bin/env python

import argparse
from pathlib import Path
from pprint import pprint

from natsort import natsorted


GEN_OUTPUT_FILENAME_DEFAULT: str = "generations.jsonl"


def gen_exists(global_step_dir: Path, dataset: str, split: str) -> bool:
    return (global_step_dir / dataset / split / GEN_OUTPUT_FILENAME_DEFAULT).is_file()


def main():
    parser = argparse.ArgumentParser(description="Check for missing generation directories")
    parser.add_argument("run_dir", type=Path, help="Path to experiment run directory")
    parser.add_argument(
        "--dataset",
        default="mls-hubert_large_ll60k-layer_22",
        help="Dataset name (default: mls-hubert_large_ll60k-layer_22)",
    )
    parser.add_argument("--split", default="dev", help="Split name (default: dev)")
    parser.add_argument(
        "--generations-dir", type=Path, help='Generations directory (default: "${run_dir}/generations)"'
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    checkpoints_dir = args.run_dir / "checkpoints"
    if not checkpoints_dir.is_dir():
        raise ValueError(f"Checkpoints directory does not exist: {checkpoints_dir}")

    if args.generations_dir is None:
        args.generations_dir = checkpoints_dir.parent / "generations"

    ckpts = set(_.parts[-1] for _ in checkpoints_dir.rglob("global_step_*") if _.is_dir())
    gnrts = set(
        _.parts[-1] for _ in args.generations_dir.rglob("global_step_*") if gen_exists(_, args.dataset, args.split)
    )
    missing = ckpts - gnrts

    print(f"Found {len(ckpts)} checkpoint directories")
    print(f"Found {len(gnrts)} generation directories")
    print(f"Missing {len(missing)} generation directories:")
    if args.verbose:
        for m in natsorted(missing):
            print(m)


if __name__ == "__main__":
    main()
