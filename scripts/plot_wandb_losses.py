#!/usr/bin/env python3
"""
Plot dev_loss and train_loss metrics from a W&B training run given a path like:
/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/hopeful-sound-525-id_5plc1ikb/generations

Usage:
    python scripts/plot_wandb_losses.py /path/to/experiment/run/generations /path/to/output_dir [ext]

Requires:
    wandb
    matplotlib
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import wandb
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL

from ssi.constants import WANDB_ENTITY_DEFAULT, WANDB_PROJECT_DEFAULT


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)


def extract_run_info(generations_dir: Path):
    """Extract W&B run name and run id from the given path."""
    if len(generations_dir.parts) < 2:
        raise ValueError("Path too short to extract run info.")
    run_dir = generations_dir.parts[-2]
    if "-id_" not in run_dir:
        raise ValueError("Run directory does not match expected format '<run_name>-id_<run_id>'")
    run_name, run_id = run_dir.split("-id_")
    return run_name, run_id


def fetch_wandb_run(run_id: str, entity: str, project: str) -> wandb.apis.public.Run:
    """Fetch W&B run object using run_id or run_name. Project and entity may be required."""
    api = wandb.Api()
    return api.run(f"{entity}/{project}/{run_id}")


def extract_wer_data(generations_dir: Path, dataset: str, split: str = "dev") -> list[tuple[int, float]]:
    """Extract WER data from wer.json files in the generations directory."""
    wer_data = []

    # Find all global_step_* directories
    for step_dir in generations_dir.rglob("global_step_*"):
        try:
            step_num = int(step_dir.name.removeprefix("global_step_"))
            wer_file = step_dir / dataset / split / "wer.json"
            if wer_file.exists():
                with open(wer_file, "r") as f:
                    wer_json = json.load(f)
                    wer_value = wer_json.get("wer")
                    if wer_value is not None:
                        wer_data.append((step_num, wer_value))
        except (ValueError, json.JSONDecodeError, KeyError) as e:
            LOGGER.warning(f"Could not parse WER data from {step_dir}: {e}")

    # Sort by step number
    wer_data.sort(key=lambda x: x[0])
    return wer_data


def extract_run_metadata(run):
    """Extract learning rate, warmup steps, and dataset from run config."""
    config = run.config

    # Extract learning rate
    lr = config.get("optimizer", {}).get("lr", "Unknown")

    # Extract warmup steps
    warmup_steps = config.get("lr_scheduler", {}).get("num_warmup_steps", "Unknown")

    # Extract dataset
    dataset_source = "Unknown"
    if "data" in config and "train" in config["data"] and "dataset" in config["data"]["train"]:
        dataset_source = config["data"]["train"]["dataset"].get("source", "Unknown")

    return lr, warmup_steps, dataset_source


def plot_losses(run: wandb.apis.public.Run, output_dir: str, generations_dir: Path, ext: str = "png"):
    """Plot dev_loss and loss from W&B run history and save to file."""
    history = run.history(keys=["dev_loss", "loss"])
    steps = history["_step"]

    # Extract metadata
    lr, warmup_steps, dataset = extract_run_metadata(run)
    ds_owner, ds_name = dataset.split("/")

    # Extract WER data
    wer_data = extract_wer_data(generations_dir, dataset=ds_name)
    wer_steps, wer_values = zip(*wer_data) if wer_data else ([], [])
    # Convert WER to percentage
    wer_values_pct = [v * 100 for v in wer_values] if wer_values else []

    # Create both linear and log scale plots
    for scale_type in ["linear", "log"]:
        fig, ax1 = plt.subplots(figsize=(12, 8))

        # Plot losses on primary y-axis
        lines = []
        if "loss" in history:
            line1 = ax1.plot(steps, history["loss"], label="loss", color="blue")
            lines.extend(line1)
        if "dev_loss" in history:
            line2 = ax1.plot(steps, history["dev_loss"], label="dev_loss", color="orange")
            lines.extend(line2)
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Loss", color="black")
        ax1.tick_params(axis="y")

        # Set log scale if requested
        if scale_type == "log":
            ax1.set_yscale("log")
            # Use explicit number formatting instead of scientific notation
            ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f"{x:.3g}"))

        # Plot WER on secondary y-axis
        if wer_data:
            ax2 = ax1.twinx()
            scat = ax2.scatter(wer_steps, wer_values_pct, color="red", alpha=0.7, s=30, label="WER")
            ax2.set_ylabel("Word Error Rate (%)", color="red")
            ax2.tick_params(axis="y", labelcolor="red")

            # Set log scale for WER axis if requested
            if scale_type == "log":
                ax2.set_yscale("log")
                # Use explicit number formatting for WER axis too
                ax2.yaxis.set_major_formatter(
                    ticker.FuncFormatter(lambda x, p: f"{x:.3g}")
                )  # Combine all legend elements
            labels = [line.get_label() for line in lines] + ["WER"]
            ax1.legend(lines + [scat], labels, loc="upper right")
        else:
            ax1.legend(loc="upper right")

        title_suffix = " (Log Scale)" if scale_type == "log" else ""
        plt.title(f"Losses and WER for W&B run: {run.name}{title_suffix}")

        # Add metadata text box in the plot
        metadata_text = "\n".join((f"LR: {lr}", f"Warmup Steps: {warmup_steps}", f"Dataset: {dataset}"))
        plt.text(
            0.98,
            0.85,
            metadata_text,
            transform=ax1.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8),
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=10,
        )

        plt.tight_layout()

        # Save with appropriate filename
        filename_suffix = "_log" if scale_type == "log" else ""
        out_path = Path(output_dir) / f"run_losses_plot{filename_suffix}.{ext}"
        plt.savefig(str(out_path))
        LOGGER.info(f"Plot saved to {out_path}")
        plt.close()

    LOGGER.info(f"Run metadata - LR: {lr}, Warmup Steps: {warmup_steps}, Dataset: {dataset}")
    if wer_data:
        LOGGER.info(f"Found {len(wer_data)} WER data points")


def main():
    parser = argparse.ArgumentParser(description="Plot dev_loss and train_loss metrics from a W&B training run.")
    parser.add_argument("experiment_path", type=Path, help="Path to experiment run generations directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the plot (default: experiment path's parent directory)",
    )
    parser.add_argument("--ext", type=str, default="png", help="File extension for the plot (default: png)")
    args = parser.parse_args()

    run_name, run_id = extract_run_info(args.experiment_path)
    LOGGER.info(f"Extracted run_name: {run_name}, run_id: {run_id}")
    entity = os.environ.get("WANDB_ENTITY", WANDB_ENTITY_DEFAULT)
    project = os.environ.get("WANDB_PROJECT", WANDB_PROJECT_DEFAULT)
    output_dir = args.output_dir or str(Path(args.experiment_path).parent)
    try:
        run = fetch_wandb_run(run_id=run_id, entity=entity, project=project)
    except Exception as e:
        LOGGER.error(f"Error fetching W&B run: {e}")
        sys.exit(1)
    plot_losses(run, output_dir, args.experiment_path, args.ext)


if __name__ == "__main__":
    main()
