# Speech Integration

Research implementation to investigate methods of integrating the speech modality into pre-trained language models

# Setup

## Clone Repository

```bash
git clone git@github.com:anilkeshwani/speech-integration.git &&
    cd speech-integration &&
    git submodule update --init --recursive --progress
```

## Set Up Environment

Ensure the sox and ffmpeg binaries are installed:

```bash
apt install sox ffmpeg
```

Install the package including development dependencies:

```bash
conda create -n ssi python=3.10.6 -y &&
    conda activate ssi &&
    pip install .["dev"] &&
    pip install --no-dependencies git+https://github.com/anilkeshwani/speech-text-alignment.git
```

> [!NOTE] 
> You may need to enter enter your SSH key passphrase for installation.

Editable install:

```bash
conda create -n ssi-dev python=3.10.6 -y &&
    conda activate ssi-dev &&
    pip install -e .["dev"] &&
    pip install --no-dependencies git+https://github.com/anilkeshwani/speech-text-alignment.git
```

> [!NOTE] 
> A dedicated environment with a static install is recommended for use with Slurm jobs, which (AFAIK) use the environment and package as installed on execution start. This is recommended so intermediate - possibly breaking - changes in an editable project location do not cause run failures or code versioning issues. 

## Setup Extras

Get shell completions for the configurations from Hydra for the duration of the Bash session by running:

```bash
eval "$(python ssi/train.py -sc install=bash)"
```

If you want to use pre-commit remember to install hooks:

```bash
pre-commit install --install-hooks
```

## Type Checking and LSP (Pyrefly)

Pyrefly is the project's canonical type checker and Python language server.

Install/update development dependencies:

```bash
uv sync --extra dev
```

Run type checking (configured for `ssi`):

```bash
uv run pyrefly check
```

For Google Antigravity IDE:
- Install the `Pyrefly` extension (`meta.pyrefly`) from OpenVSX.
- Use the workspace interpreter at `${workspaceFolder}/.venv/bin/python`.
- Keep Pyright/Pylance disabled for this workspace to avoid duplicate diagnostics.

## Download Llama 3.2 Base Model

### Download with Hugging Face CLI

```bash
base_models_dir=/mnt/scratch-artemis/anilkeshwani/models/base/ &&
huggingface-cli download "meta-llama/Llama-3.2-1B" \
    --local-dir ${base_models_dir}/Llama-3.2-1B \
    --exclude "original/consolidated.00.pth" \
    --revision "4e20de362430cd3b72f300e6b0f18e50e7166e08" # specific Git LFS commit
```

## Extend Llama 3.2 Base Model

Extend the base model with the specified number of DSUs (speech tokens). 

It saves the extended model to the specified output directory.

```bash
./scripts/extend_llama3_2.py --n_new_dsus 5000
```

See `./scripts/extend_llama3_2.py --help` for details.

# Train

> [!NOTE] 
> To enable debugging, pass `hydra.verbose=true`. This sets the log level of all loggers to `DEBUG` as described in [Logging - Hydra docs](https://hydra.cc/docs/1.3/tutorials/basic/running_your_app/logging/). A string or list can be passed to set specific loggers' levels to `DEBUG`; see the documentation for details. This is useful for visualising debug output e.g. prompts constructed in the dataset to be echoed to the console. 

## Continued Pre-training (CPT)

Example call using SpeechTokenizer-encoded (RVQ 0) speech tokens:

```bash
uv run scripts/train_cpt.py \
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
uv run scripts/train_sft.py \
    checkpointer.checkpoint_dir="$(realpath "${HOME}/hafh/models/extended/Llama-3.2-1B-2048-dsus")" \
    checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' \
    optimizer.lr=0.0002 \
    lr_scheduler.num_warmup_steps=1000 \
    speech.deduplicate=true \
    speech.n_dsus=2048 \
    data=sft/mls-mimi-srvq_0 \
    gradient_accumulation_steps=1 \
    data.train.dataloader.batch_size=32 \
    save_steps=10000
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
    uv run scripts/train_sft.py \
        checkpointer.checkpoint_dir="$(realpath "${HOME}/hafh/models/extended/Llama-3.2-1B-2048-dsus")" \
        checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' \
        optimizer.lr=0.0002 \
        lr_scheduler.num_warmup_steps=1000 \
        speech.deduplicate=true \
        speech.n_dsus=2048 \
        data=sft/mls-mimi-srvq_0
```

Notes:
- Relies on the `hafh -> /mnt/scratch-artemis/anilkeshwani` symlink in the shared `${HOME}` across Artemis and Poseidon.
- Remember to pass `--live-stream` (or equivalently `--no-capture-output`) to prevent buffering/capture of stdout/stderr (standard out/standard error)

---

# Generation

Generation is performed with vLLM using the Hugging Face-compatible model directories that are written during training time (every `save_steps` global steps). 

Example call:

```bash
python scripts/generate.py \
  model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000
```

## Specifying a Dataset Explicitly

Where `data` is not explicitly specified, the model will generate ASR transcripts for the test split of the training dataset. 

Example call with explicit test dataset:

```bash
python scripts/generate.py \
  model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000 \
  data=sft/voxpopuli-hubert_large_ll60k-layer_22
```

## Specifying Generation Parameters

Parameters affecting generation, e.g. sampling parameters, are available to override via the CLI with Hydra syntax. 

See the _generation.yaml_ config for the available options.

Example call with an override:

```
python scripts/generate.py \
  model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000 \
  sampling_params.max_tokens=1024 # override default value
```

### Specifying Model-specific Parameters

Parameters that are fixed for a given model checkpoint are automatically imputed from the training config, which is saved under the _checkpoints/_ subdirectory of a given run with the name _torchtune_config.yaml_. In the absence of a training configuration, these options can still be specified.

```bash
python scripts/generate.py \
    model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/lyric-butterfly-478-id_3qj03g5e/checkpoints/epoch_0/global_step_70000 \
    speech.deduplicate=false \
    speech.n_dsus=5000 \
    data=sft/mls-hubert_large_ll60k-layer_22
```

## Generation inside Slurm

```
srun --partition a6000 --time=01:00:00 --gres=gpu:1 --qos=gpu-short \
  conda run -n ssi-dev --live-stream \
    ./scripts/generate.py \
      gen.split=dev \
      model=/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/hopeful-sound-525-id_5plc1ikb/checkpoints/epoch_0/global_step_80000
```

## Generation: Extras

There is a hacky script, [generation_launcher.sh](/snippets/generation_launcher.sh), to launch generation for all checkpoints in a given training run (for a given epoch) under [snippets](/snippets)

---

# Evaluate Model Performance

## Word Error Rate (WER)

To compute word error rate, pass the _generations.jsonl_ file containing the output of _generate.py_ as in the following example:

```bash
python scripts/wer.py \
    /mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus-sft/hopeful-sound-525-id_5plc1ikb/generations/epoch_0/global_step_318000/mls-hubert_large_ll60k-layer_22/test/generations.jsonl
```

See `python scripts/wer.py --help` for details.



> [!IMPORTANT] 
> _wer.py_ relies on _generations.jsonl_ being located in nested directories indicating the dataset name and split: `<dataset>/<split>/generations.jsonl`. 
> If this is not the case, pass them in explicitly - see `--help`

_wer.py_ does not use Hydra

---

Help for _wer.py_ at last update:

```bash
usage: wer.py [-h] [--dataset DATASET] [--split SPLIT] [--gt_transcript_colname GT_TRANSCRIPT_COLNAME] [--normalizer {whisper}] generations_jsonl

Calculate Word Error Rate (WER) from model generations.

positional arguments:
  generations_jsonl     Path to the JSON lines file with generations.

options:
  -h, --help            show this help message and exit
  --dataset DATASET     Hugging Face dataset for reference transcripts.
  --split SPLIT         Hugging Face dataset split for reference transcripts.
  --gt_transcript_colname GT_TRANSCRIPT_COLNAME
                        Column name for ground truth transcripts in the dataset.
  --normalizer {whisper}
                        Text normalizer.
```
