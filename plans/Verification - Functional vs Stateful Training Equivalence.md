# Verification — Functional vs Stateful Training Equivalence

## Purpose

Before cutting over the training scripts from `ssi.train.train()` to `ssi.trainer.Trainer`, both implementations must be verified to produce identical results on a real W&B-logged training run. This document provides the exact commands.

## Prerequisites

1. Extended Llama 3.2 1B model at `/home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus/`
2. Logged into HuggingFace CLI (`huggingface-cli whoami`)
3. Logged into W&B (`wandb login`)
4. Environment variable `HAFH=/home/ubuntu` (or wherever the extended model lives)

## Hyperparameters

Selected per the literature review (`plans/Research - Optimal hyperparameters based on literature review.md`):

| Parameter | Value | Source |
|---|---|---|
| LR | 5e-5 | Futami (Interspeech 2025) + LLaSA consensus |
| Schedule | Cosine, 3 warmup steps (~3% of 100 steps) | Consensus across papers |
| Effective batch size | 32 (bs=4 × grad_accum=8) | Standard for 1B models |
| Data | First 2000 MLS-HuBERT train, 200 dev | Streamed via `n_samples` |
| Max steps | 100 | ~1.6 epochs; enough for loss curve comparison |

## Run 1: Functional `train()` (baseline)

```bash
HAFH=/home/ubuntu uv run python scripts/train_sft.py \
    speech.n_dsus=5000 \
    checkpointer.checkpoint_dir=/home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus \
    'checkpointer.checkpoint_files=["ft-model-00001-of-00001.safetensors"]' \
    data=sft/mls-hubert_large_ll60k-layer_22 \
    data.train.dataset.n_samples=2000 \
    data.dev.dataset.n_samples=200 \
    data.train.dataloader.batch_size=4 \
    data.dev.dataloader.batch_size=4 \
    gradient_accumulation_steps=8 \
    optimizer.lr=5e-5 \
    lr_scheduler.num_warmup_steps=3 \
    max_steps=100 \
    eval_steps=25 \
    save_steps=50 \
    log_interval=1 \
    wandb.project=sft-equivalence-verification \
    wandb.group=functional
```

## Run 2: Stateful `Trainer` (new implementation)

```bash
HAFH=/home/ubuntu uv run python scripts/train_sft_trainer.py \
    speech.n_dsus=5000 \
    checkpointer.checkpoint_dir=/home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus \
    'checkpointer.checkpoint_files=["ft-model-00001-of-00001.safetensors"]' \
    data=sft/mls-hubert_large_ll60k-layer_22 \
    data.train.dataset.n_samples=2000 \
    data.dev.dataset.n_samples=200 \
    data.train.dataloader.batch_size=4 \
    data.dev.dataloader.batch_size=4 \
    gradient_accumulation_steps=8 \
    optimizer.lr=5e-5 \
    lr_scheduler.num_warmup_steps=3 \
    max_steps=100 \
    eval_steps=25 \
    save_steps=50 \
    log_interval=1 \
    wandb.project=sft-equivalence-verification \
    wandb.group=trainer
```

## What to compare on W&B

In the `sft-equivalence-verification` project, overlay both runs:

| Metric | Expected | Notes |
|---|---|---|
| `loss` | Identical at every step | Primary equivalence criterion |
| `lr` | Identical | Cosine schedule from same config |
| `tokens_total` | Identical | Cumulative token count |
| `n_tokens.text` / `.dsu` / `.modality` | Identical | Per-type token counts |
| `dev_loss` | Identical at steps 25, 50, 75, 100 | Evaluation loss |
| `max_seq_len_step` | Identical | Max sequence length per step |
| `duration_step` | Different | Wall-clock timing — expected to differ |
| `tokens_per_second_per_gpu` | Different | Derived from wall-clock |
| `train_clock_time` | Different | Cumulative wall-clock |

## Notes

- `n_samples=2000` streams only the first 2000 rows from HuggingFace (no full dataset download)
- With bs=4 and 2000 samples: 500 batches/epoch → 62 optimizer steps/epoch → 100 steps ≈ 1.6 epochs
- For fully deterministic (bit-identical) traces, add `optimizer.fused=False debug_mode=2` and set env var `CUBLAS_WORKSPACE_CONFIG=:4096:8`. All three are required: the env var configures cuBLAS workspace, `debug_mode=2` enables `torch.use_deterministic_algorithms(True)`, and `fused=False` avoids non-deterministic fused AdamW kernels. This slows training ~20% but produces byte-identical checkpoint files
- Both scripts accept identical Hydra overrides — the only difference is the implementation path
