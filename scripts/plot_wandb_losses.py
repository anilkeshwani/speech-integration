#!/usr/bin/env python3
"""
Plot dev_loss and train_loss metrics from a W&B training run given a path like:
/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/hopeful-sound-525-id_5plc1ikb/generations

Usage:
    python scripts/plot_wandb_losses.py /path/to/experiment/run/generations

Requires:
    wandb
    matplotlib
"""
import os
import re
import sys

import matplotlib.pyplot as plt
import wandb


def extract_run_info(path):
    """Extract W&B run name and run id from the given path."""
    # Example path:
    # /mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/hopeful-sound-525-id_5plc1ikb/generations
    parts = os.path.normpath(path).split(os.sep)
    if len(parts) < 2:
        raise ValueError("Path too short to extract run info.")
    run_name = parts[-2]
    # Try to extract run id from run_name (e.g., hopeful-sound-525-id_5plc1ikb)
    match = re.search(r"id_([a-zA-Z0-9]+)", run_name)
    run_id = match.group(1) if match else None
    return run_name, run_id


def fetch_wandb_run(run_id=None, run_name=None, entity=None, project=None):
    """Fetch W&B run object using run_id or run_name. Project and entity may be required."""
    api = wandb.Api()
    if run_id:
        # If project/entity are not provided, try to infer from environment or config
        runs = api.runs(project=f"{entity}/{project}") if entity and project else api.runs()
        for run in runs:
            if run.id == run_id:
                return run
        raise ValueError(f"Run with id {run_id} not found.")
    elif run_name:
        runs = api.runs(project=f"{entity}/{project}") if entity and project else api.runs()
        for run in runs:
            if run.name == run_name:
                return run
        raise ValueError(f"Run with name {run_name} not found.")
    else:
        raise ValueError("Must provide run_id or run_name.")


def plot_losses(run):
    """Plot dev_loss and train_loss from W&B run history."""
    history = run.history(keys=["dev_loss", "train_loss"])
    steps = history["step"] if "step" in history else range(len(history))
    plt.figure(figsize=(10, 6))
    if "train_loss" in history:
        plt.plot(steps, history["train_loss"], label="train_loss")
    if "dev_loss" in history:
        plt.plot(steps, history["dev_loss"], label="dev_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title(f"Losses for W&B run: {run.name}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/plot_wandb_losses.py /path/to/experiment/run/generations")
        sys.exit(1)
    path = sys.argv[1]
    run_name, run_id = extract_run_info(path)
    print(f"Extracted run_name: {run_name}, run_id: {run_id}")
    # You may need to set your W&B entity and project below
    entity = None  # e.g., 'your-entity'
    project = None  # e.g., 'your-project'
    try:
        run = fetch_wandb_run(run_id=run_id, run_name=run_name, entity=entity, project=project)
    except Exception as e:
        print(f"Error fetching W&B run: {e}")
        sys.exit(1)
    plot_losses(run)


if __name__ == "__main__":
    main()
