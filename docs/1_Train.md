# Train

> [!NOTE] 
> To enable debugging, pass `hydra.verbose=true`. This sets the log level of all loggers to `DEBUG` as described in [Logging - Hydra docs](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/logging/). A string or list can be passed to set specific loggers' levels to `DEBUG`; see the documentation for details. This is useful for visualising debug output e.g. prompts constructed in the dataset to be echoed to the console. 

## Continued Pre-training (CPT)

Example call using SpeechTokenizer-encoded (RVQ 0) speech tokens:

```bash
python scripts/train_cpt.py \
    checkpointer.checkpoint_dir="$(realpath "${HOME}/hafh/models/extended/Llama-3.2-1B-5000-dsus")" \
    checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' \
    optimizer.lr=0.0002 \
    lr_scheduler.num_warmup_steps=0 \
    speech.deduplicate=true \
    data=cpt/mls-speechtokenizer-rvq_0
```

To enable debugging output with Hydra, append:

```bash
hydra.verbose=true # results in e.g. prompts constructed in the dataset to be echoed to the console
```

## Supervised Fine-tuning (SFT)

Specify the call as for CPT but using `scripts/train_sft.py` in place of `scripts/train_cpt.py` and specify an appropriate SFT/IT dataset config group e.g. `data=sft_hubert`.

Example call:

```bash
python scripts/train_sft.py \
    checkpointer.checkpoint_dir="$(realpath "${HOME}/hafh/models/extended/Llama-3.2-1B-1024-dsus")" \
    checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' \
    optimizer.lr=0.0002 \
    lr_scheduler.num_warmup_steps=1000 \
    speech.n_dsus=1024 \
    speech.deduplicate=true \
    data=sft/mls-speechtokenizer-rvq_0
```

## Running with Slurm (e.g. on Sardine)

To run interactively with Slurm via `srun` inside a tmux session prefix the run with `srun` and `conda run`, specifying appropriate parameters for each - typically the values shown in the example below. 

In the below example call, the Conda environment used is `ssi-latest`. Specify the appropriate Conda environment to `conda run` under the `-n` option.

```bash
srun \
    --partition a6000 \
    --time=48:00:00 \
    --gres=gpu:1 \
    --qos=gpu-medium \
    conda run --live-stream -n ssi-latest python scripts/train_cpt.py \
        checkpointer.checkpoint_dir="$(realpath "${HOME}/hafh/models/extended/Llama-3.2-1B-5000-dsus")" \
        checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' \
        optimizer.lr=0.0002 \
        lr_scheduler.num_warmup_steps=0 \
        speech.deduplicate=true \
        data=cpt/mls-speechtokenizer-rvq_0
```

Notes:
- Relies on the `hafh -> /mnt/scratch-artemis/anilkeshwani` symlink in the shared `${HOME}` across Artemis and Poseidon.
- Remember to pass `--live-stream` (or equivalently `--no-capture-output`) to prevent buffering/capture of stdout/stderr (standard out/standard error)
