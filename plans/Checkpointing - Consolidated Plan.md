# Checkpointing: Consolidated Plan

This document consolidates all checkpoint-related design, implementation status, and future work. It supersedes checkpoint-specific content in `Training Cleanup Tasks.md` and absorbs the key content from `Refactor - Separate Model and Training State Checkpoints.md` and `Plan to Simplify Checkpoint Directory Structure.md`. The companion `Research - Checkpoint and Resume Best Practices.md` contains the web research that informed the design.

Implementation lives in `ssi/checkpoint.py`, `ssi/train.py`, and `ssi/constants.py`.

---

## Table of Contents

1. [Context](#1-context)
2. [Design Principles](#2-design-principles)
3. [Design Decision Log](#3-design-decision-log)
4. [Proposed Checkpoint Structure](#4-proposed-checkpoint-structure)
5. [Downstream Consumers](#5-downstream-consumers)
6. [Future Work](#6-future-work)
7. [Remaining Open Items](#7-remaining-open-items)

---

## 1. Context

This is a research codebase investigating approaches to integrating speech into Llama 3.2 1B via discrete speech tokens (see `MASTER PLAN.md`). We compare four speech tokenizers (HuBERT, SpeechTokenizer, Mimi, FocalCodec) across four training approaches (CPT-Concat, CPT-Interleave, SFT, CPT+SFT; the latter CPT+SFT chooses the better performing CPT) on the Multilingual LibriSpeech dataset (MLS), both using BPE to compress the speech tokens or not.

MLS is 44k hours. Our training budget is 48 GPU-hours on a single A6000. One epoch of MLS will never complete, so every training run ends mid-epoch. The checkpoint system must correctly resume mid-epoch, detect configuration mismatches that would corrupt training, and produce deterministic results across resumes.

The experiment matrix (4 tokenizers × 4 approaches × 2 compression settings) produces many runs. The checkpoint structure must make it easy to identify, browse, and compare results across this matrix.

---

## 2. Design Principles

These principles guide all checkpoint-related decisions. When evaluating future changes, refer back here.

### P1: `global_step` is the canonical resume counter

`global_step` (optimizer updates completed) is the single source of truth for training progress. Epoch number and batch offset are derived from it. This follows the universal standard for LLM pretraining where epochs rarely complete.

### P2: `consumed_samples` is the batch-size-independent progress counter

We also track `consumed_samples` (total micro-batches processed × batch_size) as a secondary counter. This is Megatron-LM's approach and allows future flexibility if batch size ever needs to change between runs. For now, it serves primarily as a monitoring and auditability field.

### P3: Strict validation, not silent corruption

If the training configuration differs between save and resume in a way that would break step-to-data-position mapping, we raise a hard error. This is more rigorous than HuggingFace (silent corruption) and PyTorch Lightning (no validation). A `force_resume` config flag can downgrade to a warning for expert users.

### P4: Save everything needed for exact resume; eliminate state where possible

The checkpoint should contain all state needed to produce a training run bitwise-identical to an uninterrupted run. This includes framework RNG states, scheduler state, and cumulative metrics — not just model weights and optimizer. Where possible, prefer designs that eliminate state entirely (e.g., per-sample deterministic RNG for interleaving) over designs that require saving and restoring additional state.

### P5: Step-based training with epoch bookkeeping

`max_steps` defines training duration. Epochs are a convenience for data shuffling (`sampler.set_epoch(epoch)`), not a training loop control structure. The epoch number is derived from `global_step` on resume, never trusted from the checkpoint.

---

## 3. Design Decision Log

Each decision records what we chose, what we rejected, and why.

### D1: `global_step` as canonical counter (not `consumed_samples`)

**Chose**: `global_step` primary, `consumed_samples` secondary.
**Rejected**: `consumed_samples` as primary (Megatron-LM style).
**Why**: Our codebase is step-based throughout (LR schedule, checkpointing cadence, eval cadence, logging cadence). Switching to `consumed_samples` as primary would require rewriting all of these. `consumed_samples` is tracked for monitoring and future flexibility, but `global_step` drives the training loop.

### D2: Hard error on config mismatch (not warning)

**Chose**: `ValueError` on any mismatch in `batch_size`, `gradient_accumulation_steps`, `world_size`.
**Rejected**: Warning-only (HuggingFace approach).
**Why**: HuggingFace's silent corruption is a known footgun that wastes researcher time. A clear error message telling you exactly what's wrong is more helpful. `force_resume` flag available for expert users.

### D3: islice skip for data position (not StatefulDataLoader)

**Chose**: `itertools.islice` skip for now.
**Rejected**: `torchdata.StatefulDataLoader` (for now — see Future Work F1).
**Why**: islice is zero-dependency and works correctly with `num_workers=0`. StatefulDataLoader requires `torchdata >= 0.8.0` and changes to the data pipeline. The skip cost is acceptable for our current scale. We'll upgrade when we move to `num_workers > 0` or when skip times become a bottleneck.

**Skip cost estimate** (for context on when to revisit): For MLS at batch_size=16, worst case (checkpoint near end of epoch), each skipped batch involves a HF datasets lookup + tokenization + collation. Estimated throughput: ~100-500 batches/second on CPU, giving 1-100 minutes for a full-epoch skip depending on sample lengths.

### D4: Save LR scheduler state_dict (not reconstruct from global_step)

**Chose**: Save and restore `scheduler.state_dict()`.
**Rejected**: Reconstruct from `last_epoch=global_step` (the pre-refactor approach).
**Why**: The reconstruction approach only works for specific schedule types. Saving the state_dict is universal, costs negligible space, and is the standard practice in every framework except torchtune.

### D5: Remove `EPOCHS_KEY` from checkpoint (not keep for compat)

**Chose**: Remove `EPOCHS_KEY` entirely from new checkpoints.
**Rejected**: Keep writing it for human readability.
**Why**: A field that is written but never read on resume is a maintenance hazard. It will inevitably diverge from `global_step // steps_per_epoch` and confuse someone inspecting the checkpoint. The epoch can be trivially derived when needed.

### D6: Per-sample deterministic RNG (not stateful PRNG)

**Chose**: Per-sample RNG seeded by `(seed, epoch, sample_index)` — a pure function of sample identity.
**Rejected alternatives**:
- Module-level global PRNG — shared between train/dev, order-dependent, impossible to checkpoint cleanly, incompatible with `num_workers > 0`.
- Per-instance PRNG on `TextCompletionDataset` — still stateful, still requires checkpointing, still order-dependent, still broken with `num_workers > 0`.

**Why**: The per-sample approach eliminates the entire class of PRNG-state-management problems. There is no state to save, restore, or synchronize. Each sample's interleaving is deterministic given its identity, regardless of processing order, parallelism, or resume point. NumPy's `SeedSequence` handles tuple seeds correctly, producing independent, well-distributed streams with no collision risk.

| Property | Stateful PRNG (before) | Per-sample deterministic RNG (after) |
|----------|----------------------|--------------------------------------|
| State to checkpoint | `PRNG.bit_generator.state` | Nothing |
| islice skip correctness | Depends on `num_workers=0` | Always correct |
| `num_workers > 0` | Broken (shared mutable state) | Correct (pure function) |
| Train/dev contamination | Shared PRNG — dev depends on training | Independent by construction |
| Same sample, different order | Different interleaving | Identical interleaving |
| Reproducibility across resumes | Fragile (requires exact PRNG position) | Guaranteed |

### D7: Drop `epoch` parameter from checkpoint saving

**Chose**: Remove `epoch` from the checkpoint save API; use `step_{global_step}` directory naming.
**Rejected**: Keep `epoch` in the API.
**Why**: Per principle P5, epoch is derived from global_step. Having it as a parameter creates a source of inconsistency. The directory structure `step_N/` is clearer than `epoch_E/global_step_N/`.

### D8: Separate `save_model_checkpoint` and `save_training_state` (not single `save_checkpoint`)

**Chose**: Two public methods with distinct signatures and responsibilities.
**Rejected**: Single `save_checkpoint` with `save_training_state` boolean flag.
**Why**: The old API conflated two operations with different purposes, lifecycles, and storage requirements. Model checkpoints are per-step HF-format directories used for inference/evaluation; training state is a single overwritten file used only for resume. The boolean flag was a code smell — callers like `extend_llama3_2.py` had to pass dummy `optimizer_state_dict=None`, `seed=SEED`, `save_training_state=False`. Optional `None` defaults on training state fields created a latent bug: a caller could write a `recipe_state.pt` missing required keys that would crash on resume. The new API enforces the schema v1 contract via mandatory keyword arguments — it is impossible to create an unresumable checkpoint.

---

## 4. Proposed Checkpoint Structure

### 4.1. Directory Layout

**Current layout:**
```
experiments/
  {base_model_name}-{config_name}/
    {wandb_run_name}-id_{wandb_run_id}/
      checkpoints/
        step_N/
          model-*.safetensors, config.json, tokenizer.model, ...
        recipe_state.pt
```

**Proposed layout** — organized by speech tokenizer, with BPE as a subdirectory:
```
experiments/
  {tokenizer}/                                          <- top-level by speech tokenizer
    base_model/                                         <- extended Llama 3.2 1B for raw DSUs
      model-*.safetensors, config.json, tokenizer.model
    bpe/                                                <- BPE-compressed speech representations
      base_model/                                       <- extended Llama 3.2 1B for BPE vocab
        model-*.safetensors, config.json, tokenizer.model
      {stage}_{wandb_id}/                               <- runs using BPE compression
        step_N/
        recipe_state.pt
        torchtune_config.yaml
    {stage}_{wandb_id}/                                 <- runs using raw DSUs
      step_N/                                           <- model checkpoint (HF-format, self-contained)
        model-*.safetensors
        model.safetensors.index.json
        config.json
        tokenizer.model
        generation_config.json
      recipe_state.pt                                   <- training state (single file, always overwritten)
      torchtune_config.yaml                             <- resolved config snapshot
```

**Concrete example:**
```
experiments/
  hubert/
    base_model/                                         <- Llama 3.2 1B + 5000 HuBERT DSUs
    bpe/
      base_model/                                       <- Llama 3.2 1B + BPE-compressed HuBERT vocab
      cpt_interleaved_c3d4e5f6/
        step_5000/
        recipe_state.pt
        torchtune_config.yaml
      sft_d4e5f6a7/
        ...
    cpt_interleaved_9h3htysc/
      step_5000/
      step_10000/
      recipe_state.pt
      torchtune_config.yaml
    cpt_concatenated_a1b2c3d4/
      ...
    sft_a7b2cdef/
      step_2000/
      recipe_state.pt
      torchtune_config.yaml
  speechtokenizer/
    base_model/
    bpe/
      base_model/
      ...
    cpt_concatenated_f1e2d3c4/
      ...
  mimi/
    ...
  focalcodec/
    ...
```

**Key design decisions:**

1. **Top-level by tokenizer**: Each tokenizer defines a fixed `n_dsus` (e.g., 5000 for HuBERT). Grouping by tokenizer makes it natural to pre-build an extended base model per tokenizer and share it across runs.

2. **`base_model/` per tokenizer**: A pre-extended Llama 3.2 1B with the tokenizer vocabulary and embedding layer sized for that tokenizer's `n_dsus`. Built once by `extend_llama3_2.py`, used as the starting point for all runs under that tokenizer.

3. **`bpe/` as a subdirectory within each tokenizer**: The BPE vocabulary is derived from a specific tokenizer's codebook, so it nests naturally under the tokenizer. BPE runs require a separate `base_model/` because the BPE vocabulary size differs from the raw codebook size. Non-BPE runs (the majority) sit directly under the tokenizer directory with no extra nesting.

4. **`n_dsus` not in directory name**: Since `n_dsus` is determined by the tokenizer (or BPE merge count) and is constant within a given `base_model/`, encoding it in the run name is redundant. It can be read from the config snapshot if needed.

5. **No dedup/modality flags in directory name**: Deduplication and modality tokens are always enabled (hardcoded in `conf/common.yaml`). No flags needed.

6. **Run directory name**: `{stage}_{wandb_id}` where:
   - `stage`: `cpt_interleaved`, `cpt_concatenated`, or `sft`
   - `wandb_id`: 8-character W&B run ID for uniqueness and cross-referencing

7. **No `checkpoints/` nesting**: The constant `checkpoints/` intermediate directory is dropped — `step_N/` directories and `recipe_state.pt` live directly in the run directory.

8. **`torchtune_config.yaml` at run level**: A snapshot of the resolved training config, saved once at training start. Enables auditability and re-running without hunting for the original config.

9. **CPT+SFT has no special directory convention**: The SFT run that follows CPT points to a CPT checkpoint as its starting model. The `torchtune_config.yaml` inside the SFT run records which CPT checkpoint it started from.

### 4.2. Model Checkpoint Contents (`step_N/`)

Each `step_N/` directory is a self-contained HF-format model directory, usable directly by HuggingFace tooling, `generate.py`, and evaluation scripts.

```
step_N/
  model-00001-of-00004.safetensors      <- sharded model weights
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json          <- weight map and metadata
  config.json                           <- model config (copied from source)
  tokenizer.model                       <- tokenizer (copied from source)
  generation_config.json                <- generation defaults (copied from source)
```

Produced by `save_model_checkpoint()`. Files other than weights are copied from the source model directory so each checkpoint is independently loadable.

### 4.3. Training State Contents (`recipe_state.pt`)

A single `recipe_state.pt` file at the run directory level, always overwritten on save. Contains everything needed for exact resume.

**Schema v1** (current):

```python
{
    # --- Versioning ---
    "checkpoint_version": 1,                    # schema version for forward compat
    "timestamp": "2026-03-19T14:30:00+00:00",   # ISO 8601 UTC
    "ssi_version": "0.1.0",                     # package version

    # --- Progress ---
    "global_step": 10000,                       # canonical counter (P1)
    "consumed_samples": 160000,                 # batch-size-independent counter (P2)

    # --- Reproducibility ---
    "seed": 42831,                              # training seed (validated on resume)
    "rng_state": {                              # all framework RNG states
        "python": random.getstate(),
        "numpy": numpy.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
    },

    # --- Optimizer & Scheduler ---
    "optimizer": optimizer.state_dict(),         # full optimizer state
    "lr_scheduler": scheduler.state_dict(),      # or None if no scheduler

    # --- Validation on Resume ---
    "training_hparams": {                       # hard-error on mismatch (P3)
        "batch_size": 16,
        "gradient_accumulation_steps": 4,
        "world_size": 1,
        "steps_per_epoch": 5000,
    },

    # --- Monitoring & Auditability ---
    "cumulative_metrics": {
        "tokens_train_total": 10_000_000,
        "token_type_counts": {
            "text": 8_000_000,
            "dsu": 1_500_000,
            "special_text": 500_000,
        },
        "wall_clock_seconds": 36000.0,
    },
}
```

The only gap vs. the "complete checkpoint" from the Research doc is DataLoader/sampler state (currently handled via islice skip per D3; StatefulDataLoader planned as F1).

---

## 5. Downstream Consumers

When the directory structure changes (§4.1), all downstream code is updated to the new format. No legacy format support is needed — old checkpoints will not be loaded.

### Files requiring updates

| File | What changes | Details |
|------|-------------|---------|
| `ssi/checkpoint.py` | `resolve_checkpointer_output_dir()` | Build path from `{experiments_root_dir}/{tokenizer}/{stage}_{wandb_id}` using `cfg.experiments_root_dir` directly. Add `TOKENIZER_SHORT_NAMES` mapping in `ssi/constants.py`. |
| `ssi/utils.py` | `_parse_model_path()` | Parse new run dir name (`run_dir/step_M`) to extract stage, wandb_id. |
| `scripts/generate.py` | `_resolve_gen_output_dir()` and `main()` | Output dir: `run_dir/generations/step_M`. Config lookup: `Path(cfg.model).parent` (1 level up). |
| `scripts/plot_wandb_losses.py` | Glob pattern | Match `step_*`. |
| `snippets/check_missing_generations.py` | Step dir discovery | Look for step dirs in run_dir directly. |
| `snippets/generation_launcher.sh` | `find` pattern | Match `step_*`. |
| `snippets/impute_config.py` | Config lookup | Check for `torchtune_config.yaml` at run_dir level. |
| `conf/training.yaml` | Comment | Update to reflect new auto-resolution format. |

### Files NOT changed

- `conf/common.yaml` — `output_dir` stays as-is (still used for `wandb.log_dir`). The resolver uses `cfg.experiments_root_dir` directly.
- `ssi/metric_logging.py` — Already saves to `config.checkpointer.output_dir`. Works as-is.
- Data config YAMLs — No `short_name` field needed. Tokenizer short name derived from HF source string via `parse_hf_repo_id()` + `TOKENIZER_SHORT_NAMES`.

---

## 6. Future Work

These items are explicitly out of scope for the current implementation but are documented for planning.

### F1: `torchdata.StatefulDataLoader`

Replace `torch.utils.data.DataLoader` with `torchdata.StatefulDataLoader` to eliminate islice skip cost. Requires `torchdata >= 0.8.0`. The DataLoader state_dict would be saved in the checkpoint alongside the other state. This also enables `num_workers > 0` for the DataLoader itself (the per-sample deterministic RNG in D6 already makes the dataset side safe for multi-worker access).

When enabling `num_workers > 0`, note that islice-based resume skipping is still synchronous and its cost scales with `num_workers=0` assumptions. Switching to `StatefulDataLoader` remains the recommended path for production use with `num_workers > 0`. See `plans/Reference - Memory in Multi-Worker DataLoaders.md` for the memory implications of multi-worker data loading.

### F2: Megatron-style consumed_samples indexing

Pre-shuffle the dataset with a deterministic seed and use `consumed_samples` to index directly into the shuffled sequence. No skip, no stateful dataloader. This is the most robust solution for datasets that never complete an epoch.

### F3: WSD learning rate schedule

Implement Warmup-Stable-Decay schedule for checkpoint-friendly training. The stable phase has trivial scheduler state (constant LR), making resume simpler. Natural branching for producing final models at different compute budgets. No need to know `total_steps` in advance (unlike cosine). See Research doc §2 for details.

### F4: Checkpoint retention policy

Currently all checkpoints are kept. Implement a policy like "keep last N + every Kth checkpoint" to manage disk usage. HuggingFace and PyTorch Lightning both have this built in.

### F5: Async checkpointing

Save checkpoints without blocking training. PyTorch DCP and torchtitan support this. Useful when checkpoint save time is a significant fraction of step time.

### F6: Batch size change support

Using `consumed_samples` as the primary counter would allow changing batch size between runs. The training loop would derive the step count from `consumed_samples / new_effective_batch_size`. This requires F2 (consumed_samples indexing) to work correctly for data position.

### F7: Pre-built extended base models per tokenizer

Build and store an extended Llama 3.2 1B per tokenizer in the `{tokenizer}/base_model/` directory (see §4.1). Each extended model has the tokenizer vocabulary and embedding layer sized for that tokenizer's `n_dsus`. Built once by `extend_llama3_2.py`, shared across all runs for that tokenizer. This is a prerequisite for the directory structure simplification.

---

## 7. Remaining Open Items

| ID | Description | Status |
|----|-------------|--------|
| T17 | GPU smoke test: end-to-end save/resume roundtrip on GPU (few steps → save → resume → verify step continuity, LR continuity, no duplicate batches) | Pending |
| R1 | Directory structure simplification (§4.1 + §5 of this document). Blocked on deciding `n_dsus` per tokenizer and building extended base models (F7). | Pending |
