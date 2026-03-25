# Speech Integration

Research codebase investigating approaches to integrating the speech modality into the Llama 3.2 1B language model via discrete speech tokens. See `plans/MASTER PLAN.md` for the full experiment design.

We compare four speech tokenizers (HuBERT, SpeechTokenizer, Mimi, FocalCodec) across four training approaches (CPT interleaved, CPT concatenated, SFT, CPT+SFT) on Multilingual LibriSpeech (MLS), with and without BPE compression of speech tokens.

## Setup

### Prerequisites

- Python 3.10.6
- [uv](https://docs.astral.sh/uv/) (package manager)
- `sox` and `ffmpeg` binaries: `apt install sox ffmpeg`

### Install

```bash
git clone git@github.com:anilkeshwani/speech-integration.git
cd speech-integration
uv sync --extra dev
```

### Pre-commit hooks

```bash
pre-commit install
pre-commit run --all-files  # verify everything passes
```

### Download Llama 3.2 base model

```bash
huggingface-cli download "meta-llama/Llama-3.2-1B" \
    --local-dir /mnt/scratch-artemis/anilkeshwani/models/base/Llama-3.2-1B \
    --exclude "original/consolidated.00.pth" \
    --revision "4e20de362430cd3b72f300e6b0f18e50e7166e08"
```

### Extend base model with speech tokens

Extend the Llama 3.2 1B tokenizer and embedding layer with DSU (discrete speech unit) tokens:

```bash
uv run scripts/extend_llama3_2.py --n_new_dsus 5000
```

This creates the extended model at `models/extended/Llama-3.2-1B-5000-dsus/`. See `--help` for options.

## Training

All training scripts use [Hydra](https://hydra.cc/) for configuration. Data configs are modular — each tokenizer-specific config inherits shared defaults from a base config. Checkpoint files are auto-discovered from the checkpoint directory.

### Continued Pre-Training (CPT)

```bash
uv run scripts/train_cpt.py data=cpt/mls-hubert_large_ll60k-layer_22
```

Available CPT data configs: `cpt/mls-hubert_large_ll60k-layer_22`, `cpt/mls-speechtokenizer-rvq_0`, `cpt/mls-mimi-srvq_0`, `cpt/mls-focalcodec`.

### Supervised Fine-Tuning (SFT)

```bash
uv run scripts/train_sft.py data=sft/mls-hubert_large_ll60k-layer_22
```

Available SFT data configs: `sft/mls-hubert_large_ll60k-layer_22`, `sft/mls-speechtokenizer-rvq_0`, `sft/mls-mimi-srvq_0`, `sft/mls-focalcodec`.

> [!NOTE]
> `speech.n_dsus` (the number of discrete speech unit tokens) is automatically resolved from the data config. Each tokenizer has a fixed codebook size: HuBERT=5000, SpeechTokenizer=1024, Mimi=2048, FocalCodec=8192. Override with `speech.n_dsus=<N>` if needed.

### Common overrides

```bash
# Override optimizer and scheduler
optimizer.lr=1e-4 lr_scheduler.num_warmup_steps=500

# Override batch size and gradient accumulation
data.train.dataloader.batch_size=32 gradient_accumulation_steps=1

# Override checkpointing and evaluation frequency
save_steps=5000 eval_steps=500

# Resume from a training state checkpoint
checkpointer.training_state_checkpoint=/path/to/training_state.pt

# Use only the first 2000 samples (streamed, no full dataset download)
data.train.dataset.n_samples=2000 data.dev.dataset.n_samples=200

# Override checkpoint directory (default: ${extended_models_dir}/${extended_model_name})
checkpointer.checkpoint_dir=/path/to/extended/Llama-3.2-1B-5000-dsus
```

### Running on Slurm

```bash
srun --partition a6000 --time=48:00:00 --gres=gpu:1 --qos=gpu-medium \
    uv run scripts/train_sft.py \
        data=sft/mls-hubert_large_ll60k-layer_22
```

> [!NOTE]
> Pass `--live-stream` to `srun` to prevent buffering of stdout/stderr.

### Debug mode

Append `hydra.verbose=true` to any training command to set all loggers to `DEBUG` level (shows constructed prompts, data pipeline details, etc.).

## Generation

Generation uses vLLM with the HF-compatible model checkpoints saved during training (every `save_steps` steps).

```bash
uv run scripts/generate.py \
    model=/path/to/experiments/hubert/sft_a7b2cdef/step_10000
```

The script auto-detects `speech.n_dsus` and the data config from the training config snapshot (`torchtune_config.yaml`) saved alongside the checkpoints.

### Specifying a dataset

By default, generation runs on the test split of the training dataset. To use a different dataset:

```bash
uv run scripts/generate.py \
    model=/path/to/step_10000 \
    data=sft/mls-hubert_large_ll60k-layer_22
```

### Sampling parameters

```bash
uv run scripts/generate.py \
    model=/path/to/step_10000 \
    sampling_params.temperature=0.7 \
    sampling_params.max_tokens=512
```

See `conf/generate.yaml` for all available options.

## Evaluation

### Word Error Rate (WER)

```bash
uv run scripts/wer.py /path/to/generations.jsonl
```

The script expects `generations.jsonl` to be located under `<dataset>/<split>/generations.jsonl` to auto-detect the reference dataset. Otherwise, pass `--dataset` and `--split` explicitly. See `--help` for details.

## Testing

```bash
uv run pytest              # run all tests
uv run pytest -v           # verbose output
uv run pytest -k "not disk"  # skip tests that load model weights from disk
```

Tests are under `tests/`. Some tests require the Llama 3.2 1B base model at `LLAMA_3_2_1B_BASE_DIR` (see `ssi/constants.py`) and are automatically skipped if not found.

## Project structure

```
conf/                   Hydra configs (common, training, cpt, sft, generate, data/)
scripts/                Entry points (train_cpt, train_sft, generate, extend_llama3_2, wer)
ssi/                    Core library (checkpoint, train, data, model, loss, eval, etc.)
tests/                  Test suite
plans/                  Design documents and research notes
```

## Configuration architecture

```
conf/
  common.yaml           Base config (speech params, paths, W&B, device)
  training.yaml         Training defaults (optimizer, scheduler, checkpointing)
  cpt.yaml              CPT entry point (inherits common + training, selects data)
  sft.yaml              SFT entry point (inherits common + training, selects data)
  generate.yaml         Generation entry point (inherits common, selects data)
  data/
    cpt/
      _base_.yaml       Shared CPT data config (interleave params, dataloader, splits)
      mls-hubert_large_ll60k-layer_22.yaml   (4 lines: source + sampling_rate)
      mls-speechtokenizer-rvq_0.yaml
      mls-mimi-srvq_0.yaml
      mls-focalcodec.yaml
    sft/
      _base_.yaml       Shared SFT data config (system prompt, column map, splits)
      mls-hubert_large_ll60k-layer_22.yaml   (3 lines: source)
      mls-speechtokenizer-rvq_0.yaml
      mls-mimi-srvq_0.yaml
      mls-focalcodec.yaml
```

Adding a new speech tokenizer requires only a 3-4 line data config per training type.
