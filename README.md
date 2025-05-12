# speech-integration
Research implementation to investigate methods of integrating the speech modality into pre-trained language models

## Setup

### Clone Repository

```bash
git clone git@github.com:anilkeshwani/speech-integration.git &&
    cd speech-integration &&
    git submodule update --init --recursive --progress # future-proof
```

### Set Up Environment

Ensure the necessary binary requirements are installed:

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

> [!NOTE] You may need to enter enter your SSH key passphrase for installation.

Editable install:

```bash
conda create -n ssi-dev python=3.10.6 -y &&
    conda activate ssi-dev &&
    pip install -e .["dev"] &&
    pip install --no-dependencies git+https://github.com/anilkeshwani/speech-text-alignment.git
```

> [!NOTE] A dedicated environment with a static install is recommended for use with Slurm jobs, which (AFAIK) use the environment and package as installed on execution start. This is recommended so intermediate - possibly breaking - changes in an editable project location do not cause run failures or code versioning issues. 

### Setup Extras

Get shell completions for the configurations from Hydra for the duration of the Bash session by running:

```bash
eval "$(python ssi/train.py -sc install=bash)"
```

If you want to use pre-commit remember to install hooks:

```bash
pre-commit install --install-hooks
```

## Download Llama 3.2 Base Model

### Download with Hugging Face CLI

```bash
base_models_dir=/mnt/scratch-artemis/anilkeshwani/models/base/ &&
huggingface-cli download "meta-llama/Llama-3.2-1B" \
    --local-dir ${base_models_dir}/Llama-3.2-1B \
    --exclude "original/consolidated.00.pth" \
    --revision "4e20de362430cd3b72f300e6b0f18e50e7166e08" # specific Git LFS commit
```

### Download with torchtune (`tune download`) - Not Recommended

The training recipe assumes that you've run the following command, substituting relevant variables from the configuration file values, in order to download the Llama 3.2 pre-trained (base) model:

``` bash
tune download meta-llama/${model_name} --output-dir ${base_models_dir}/${model_name} --ignore-patterns "original/consolidated.00.pth"
```

Typically:

```bash
base_models_dir=/mnt/scratch-artemis/anilkeshwani/models/base/ &&
tune download meta-llama/Llama-3.2-1B \
    --output-dir ${base_models_dir}/Llama-3.2-1B \
    --ignore-patterns "original/consolidated.00.pth"
```

<details>
    <summary>Download terminal output</summary>
    ```
    Ignoring files matching the following patterns: original/consolidated.00.pth
    LICENSE.txt: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 7.71k/7.71k [00:00<00:00, 2.99MB/s]
    original/params.json: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 220/220 [00:00<00:00, 2.06MB/s]
    USE_POLICY.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.02k/6.02k [00:00<00:00, 38.1MB/s]
    README.md: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 41.2k/41.2k [00:00<00:00, 13.4MB/s]
    .gitattributes: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1.52k/1.52k [00:00<00:00, 14.1MB/s]
    tokenizer.model: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2.18M/2.18M [00:00<00:00, 25.0MB/s]
    Fetching 12 files: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 12/12 [00:02<00:00,  4.76it/s]
    Successfully downloaded model repo and wrote to the following locations:
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/.gitattributes
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/config.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/LICENSE.txt
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/tokenizer_config.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/tokenizer.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/original
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/.cache
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/README.md
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/generation_config.json
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/model.safetensors
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/USE_POLICY.md
    /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B/special_tokens_map.json
    ```
</details>

## Extend Llama 3.2 Base Model

The following command suffices to extend the base model with the specified number of DSUs (speech tokens). It will save the extended model to the specified output directory.

```bash
./scripts/extend_llama3_2.py --n_new_dsus 5000
```

See `./scripts/extend_llama3_2.py --help` for details.

## Train

### Continued Pre-training (CPT)

```bash
python scripts/train_cpt.py \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus' \
    checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' # slightly weird syntax
```

#### Run in Slurm (e.g. on Sardine)

First enable the correct Conda environment. Then run, for example:

```bash
srun \
    --partition a6000 \
    --time=48:00:00 \
    --gres=gpu:1 \
    --qos=gpu-medium \
    python scripts/train_cpt.py \
        checkpointer.checkpoint_dir="${HOME}/hafh/models/extended/Llama-3.2-1B-5000-dsus" \
        checkpointer.checkpoint_files="['ft-model-00001-of-00001.safetensors']" \
        optimizer.lr=0.0002 \
        lr_scheduler.num_warmup_steps=1000
```

This is an example demo snippet. Note: Relies on the `hafh -> /mnt/scratch-artemis/anilkeshwani` symlink in the shared `${HOME}` across Artemis and Poseidon.

### Supervised Fine-tuning (SFT)

```
python scripts/train_cpt.py \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus' \
    checkpointer.checkpoint_files='["ft-model-00001-of-00001.safetensors"]' # slightly weird syntax
```

## Generation

Example call:

```bash
python ssi/generate.py \
    checkpointer.config_json='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus/config.json' \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/experiments/Llama-3.2-1B-5000-dsus/playful-morning-102-id_rq5tmfca/checkpoints/global-step-053862' \
    checkpointer.checkpoint_files='["hf_model_0001_1.pt"]' \
    output_jsonl='generated-playful-morning-global-step-053862.jsonl'
```
