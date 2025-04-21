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
    pip install -e .["dev"] &&
    pip install --no-dependencies git+https://github.com/anilkeshwani/speech-text-alignment.git
```

Get shell completions for the configurations from Hydra for the duration of the Bash session by running:

```bash
eval "$(python ssi/train.py -sc install=bash)"
```

If you want to use pre-commit remember to install hooks:

```bash
pre-commit install --install-hooks
```

## Continued Pre-training (_CPT_)

```bash
python scripts/train_cpt.py \
    checkpointer.config_json='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus/config.json' \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus' \
    checkpointer.checkpoint_files='["hf_model_0001_0.pt"]' # slightly weird syntax
```

## Supervised Fine-tuning (_SFT_)

```
python scripts/train_cpt.py \
    checkpointer.config_json='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus/config.json' \
    checkpointer.checkpoint_dir='/mnt/scratch-artemis/anilkeshwani/models/extended/Llama-3.2-1B-5000-dsus' \
    checkpointer.checkpoint_files='["hf_model_0001_0.pt"]' # slightly weird syntax
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

