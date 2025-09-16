#!/usr/bin/env bash

#SBATCH --job-name=generate
#SBATCH --partition=a6000
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu-short
#SBATCH --time=00:25:00
#SBATCH --output="/mnt/scratch-artemis/anilkeshwani/logs/%x.out"
#SBATCH --error="/mnt/scratch-artemis/anilkeshwani/logs/%x.err"

if [ -z "$1" ]; then
  echo "Usage: sbatch run_generate.sbatch <checkpoint-path>"
  exit 1
fi

CHECKPOINT=$1

conda run -n ssi-dev python /mnt/scratch-artemis/anilkeshwani/speech-integration/scripts/generate.py model="$CHECKPOINT"
