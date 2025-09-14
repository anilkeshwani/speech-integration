#!/usr/bin/env bash

EXP_DIR='/mnt/scratch-artemis/anilkeshwani/experiments/' # for pretty naming of tmux windows

TMUX_SESSION_NAME='generation' # tmux session with this name must already exist
CONDA_ENV='ssi-dev'            # Conda environment to use for generation

# Slurm parameters
SLURM_TIME='01:00:00'
SLURM_QOS='gpu-short'

# Sleep between batches of 4 jobs (max jobs allowed on QOS gpu-short)
SLEEP_SECONDS=5

# # Command line argument (mandatory)
# dir="$1" # directory containing HF model directories; HINT usually has basename epoch_0 or epoch_{N} for some N

# # Iterate over directories sorted by creation time - apparently find is safer than ls
# for file in $(find "${dir}" -mindepth 1 -maxdepth 1 -type d -exec stat --format='%W %n' {} + | sort -n | awk '{print $2}'); do

idx_job=0 # index for job naming
# Iterate over directories sorted by creation time - apparently find is safer than ls
for file in $(find "${EXP_DIR}" -type d -name 'global_step*' -exec stat --format='%W %n' {} + | grep 'sft' | grep -v 'generations' | sort -n | awk '{print $2}'); do
    # Skip the wandb/ directory
    if [[ "$(basename "${file}")" == "wandb" ]]; then
        echo "Skipping wandb directory: ${file}"
        continue
    fi
    echo "Launching Slurm generation job for model: ${file#"${EXP_DIR}"}"
    # Create a new tmux window for each generation task w/ checkpoint directory (minus experiment dir) as window name
    tmux new-window -d -t ${TMUX_SESSION_NAME} -n "generation-${file}" -- bash -c "
        srun --partition a6000 --time=${SLURM_TIME} --qos=${SLURM_QOS} --gres=gpu:1 \
            conda run --live-stream -n ${CONDA_ENV} \
                /mnt/scratch-artemis/anilkeshwani/speech-integration/scripts/generate.py \
                model=${file} \
                gen.split='dev'
        exec bash"
    sleep 1 # sleep to not overwhelm Slurm or tmux
    idx_job=$((idx_job + 1))
    if ((idx_job % 4 == 0)); then
        echo "Launched ${idx_job} generation jobs, sleeping for ${SLEEP_SECONDS} seconds..."
        sleep $SLEEP_SECONDS # sleep to not overwhelm Slurm or tmux
    fi
done
