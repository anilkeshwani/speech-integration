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
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

from ssi.constants import WANDB_ENTITY_DEFAULT, WANDB_PROJECT_DEFAULT


def extract_run_info(path):
    """Extract W&B run name and run id from the given path."""
    p = Path(path)
    if len(p.parts) < 2:
        raise ValueError("Path too short to extract run info.")
    run_dir = p.parts[-2]
    if "-id_" not in run_dir:
        raise ValueError("Run directory does not match expected format '<run_name>-id_<run_id>'")
    run_name, run_id = run_dir.split("-id_")
    return run_name, run_id


def fetch_wandb_run(run_id=None, entity=None, project=None):
    """Fetch W&B run object using run_id or run_name. Project and entity may be required."""
    api = wandb.Api()
    if run_id is None:
        raise ValueError("Run ID is required.")
    # Use default entity/project if not provided
    entity = entity or WANDB_ENTITY_DEFAULT
    project = project or WANDB_PROJECT_DEFAULT
    if not (entity and project):
        raise ValueError("Entity and project could not be determined from W&B settings.")
    return api.run(f"{entity}/{project}/{run_id}")


def plot_losses(run, output_dir, ext="png"):
    """Plot dev_loss and loss from W&B run history and save to file."""
    history = run.history(keys=["dev_loss", "loss"])
    breakpoint()
    steps = history["step"] if "step" in history else range(len(history))
    plt.figure(figsize=(10, 6))
    if "loss" in history:
        plt.plot(steps, history["loss"], label="loss")
    if "dev_loss" in history:
        plt.plot(steps, history["dev_loss"], label="dev_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Losses for W&B run: {run.name}")
    plt.legend()
    plt.tight_layout()
    out_path = Path(output_dir) / f"run_losses_plot.{ext}"
    plt.savefig(str(out_path))
    print(f"Plot saved to {out_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Plot dev_loss and train_loss metrics from a W&B training run.")
    parser.add_argument("experiment_path", type=str, help="Path to experiment run generations directory")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the plot (default: experiment path's parent directory)",
    )
    parser.add_argument("--ext", type=str, default="png", help="File extension for the plot (default: png)")
    args = parser.parse_args()

    run_name, run_id = extract_run_info(args.experiment_path)
    print(f"Extracted run_name: {run_name}, run_id: {run_id}")
    # You may need to set your W&B entity and project below
    entity = None  # e.g., 'your-entity'
    project = None  # e.g., 'your-project'
    output_dir = args.output_dir or str(Path(args.experiment_path).parent)
    try:
        run = fetch_wandb_run(run_id=run_id, entity=entity, project=project)
    except Exception as e:
        print(f"Error fetching W&B run: {e}")
        sys.exit(1)
    plot_losses(run, output_dir, args.ext)


if __name__ == "__main__":
    main()
