#!/usr/bin/env python

import logging
import os
import sys
from pathlib import Path

import wandb
from omegaconf import OmegaConf
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL

from ssi.constants import TORCHTUNE_CONFIG_FILENAME
from ssi.utils import _parse_model_path, extract_wandb_run_cfg, parse_hf_repo_id


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    # level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    level=logging.INFO,
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)

# Example input:
# /mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-1024-dsus-sft/hearty-gorge-510-id_5bh6o9ir
WB_ENTITY = "anilkeshwani"
WB_PROJECT = "speech-integration"


def main(dir: Path) -> None:
    wandb_api = wandb.Api()
    for rundir in sorted(dir.iterdir()):
        if rundir.name == "wandb":
            LOGGER.info(f"Skipping wandb directory: {rundir!s}")
            continue
        if not rundir.is_dir():
            LOGGER.info(f"Skipping {rundir!s} as it is not a directory")
            continue
        if (rundir / "checkpoints" / TORCHTUNE_CONFIG_FILENAME).exists():
            LOGGER.info(f"Config already exists at {rundir / 'checkpoints' / TORCHTUNE_CONFIG_FILENAME!s}. Skipping")
            continue
        wandb_run_id = rundir.name.split("_")[-1]  # relies on naming like "hearty-gorge-510-id_5bh6o9ir"
        # Initilise a wandb.apis.public.runs.Run object from model metadata NOTE not a wandb.sdk.wandb_run.Run object
        wandb_run = wandb_api.run(f"{WB_ENTITY}/{WB_PROJECT}/{wandb_run_id}")  # hard-coded
        wandb_run_cfg = extract_wandb_run_cfg(wandb_run)
        OmegaConf.save(wandb_run_cfg, rundir / "checkpoints" / TORCHTUNE_CONFIG_FILENAME)
        LOGGER.info(f"Saved config to {rundir / 'checkpoints' / TORCHTUNE_CONFIG_FILENAME!s}")
        # breakpoint()  # toggled during development/testing for manual inspection of saved configs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Impute config files for Llama 3.2 runs")
    parser.add_argument(
        "dir",
        type=Path,
        help="Directory containing the runs to impute configs for. Should contain subdirectories with run names.",
    )
    args = parser.parse_args()
    main(args.dir)
