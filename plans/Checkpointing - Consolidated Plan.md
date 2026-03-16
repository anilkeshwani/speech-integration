# Checkpointing: Consolidated Design and Implementation Plan

This document consolidates and supersedes the following previous plans:
- `Fix B2 - Intra-Epoch Step-Level Resume with Configuration Validation.md` (and its critique notes)
- Checkpoint-related items from `Training Cleanup Tasks.md` (B1, B2, B6, items in Recommendations)
- `Plan to Simplify Checkpoint Directory Structure.md` (checkpoint schema portions)
- Checkpoint-related items from `Plan - Overview.md` (Phase 1 checkpoint stabilization)

Those files are preserved on GitHub for history but should be considered superseded by this document.

The companion `Research - Checkpoint and Resume Best Practices.md` contains the detailed web research that informs the design choices below.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Current State and Its Bugs](#2-current-state-and-its-bugs)
3. [Design Principles](#3-design-principles)
4. [Checkpoint Schema](#4-checkpoint-schema)
5. [Resume Behaviour Specification](#5-resume-behaviour-specification)
6. [Data Position Recovery](#6-data-position-recovery)
7. [RNG State Management](#7-rng-state-management)
8. [Learning Rate Schedule Resume](#8-learning-rate-schedule-resume)
9. [Configuration Validation on Resume](#9-configuration-validation-on-resume)
10. [Monitoring Continuity Across Resume](#10-monitoring-continuity-across-resume)
11. [Implementation Plan](#11-implementation-plan)
12. [Test Plan](#12-test-plan)
13. [Design Decision Log](#13-design-decision-log)
14. [Future Work](#14-future-work)

---

## 1. Problem Statement

MLS is 44k hours. Our training budget is 48 GPU-hours on a single A6000. One epoch of MLS will never complete, so every training run ends mid-epoch. We need checkpointing that:

1. **Correctly resumes mid-epoch** — picking up at the exact dataset position, global step, and optimizer state where training left off.
2. **Detects configuration mismatches** — raising clear errors when batch size, gradient accumulation, or world size differ between save and resume, because these break the step-to-data-position mapping.
3. **Preserves RNG state** — so that stochastic operations (interleaving, dropout, data shuffling) produce the same sequence they would have in an uninterrupted run.
4. **Saves the LR scheduler state** — so the learning rate resumes at the correct point in the schedule.
5. **Maintains monitoring continuity** — so cumulative metrics (tokens seen, training time) don't reset to zero on resume.

---

## 2. Current State and Its Bugs

### What the checkpoint currently saves (`save_checkpoint` in `ssi/checkpoint.py:339-370`)

```python
ckpt_dict = {
    MODEL_KEY: model_state_dict,        # "model"
    EPOCHS_KEY: epoch,                   # "epochs_run" — current loop index, 0-indexed
    GLOBAL_STEP_KEY: global_step,        # "global_step"
    SEED_KEY: seed,                      # "seed"
    training.OPT_KEY: optimizer_state,   # "optimizer" (conditional)
}
```

### What `resume_training_state` reads (`ssi/train.py:65-68`)

```python
return ckpt_dict[EPOCHS_KEY], ckpt_dict[GLOBAL_STEP_KEY], ckpt_dict[OPTIMIZER_KEY]
```

### Bug inventory (checkpoint-related)

| # | Severity | Bug | Root Cause | Current Impact |
|---|----------|-----|------------|----------------|
| **B2** | Critical | Epoch semantics on resume are wrong | `save_checkpoint` writes the current loop index `epoch` (mid-epoch, 0-indexed). On resume, `range(epochs_run, n_epochs)` restarts that epoch from batch 0, re-training already-seen batches and double-incrementing `global_step`. | Data duplication + step count corruption on every resume. |
| **PRNG** | Critical | Module-level PRNG in `cpt.py` not advanced during any skip | `PRNG = np.random.default_rng(SEED)` is module-level. On resume, it starts at position 0 regardless of how many batches were processed before the checkpoint. Even without islice, the PRNG state is never restored. | Every resumed run produces different interleaving than the original run would have for the same data points. |
| **LR** | High | LR scheduler state not checkpointed | `lr_schedule.py` reconstructs the scheduler from `global_step` on resume, but does not save/restore `scheduler.state_dict()`. The current workaround (passing `last_epoch=global_step-1`) only works for the specific cosine-with-warmup schedule and breaks if the schedule has any internal state beyond `last_epoch`. | LR is approximately correct for cosine warmup but will be wrong for any other schedule. |
| **Metrics** | Moderate | `tokens_train_total` and `token_type_counts_total` not checkpointed | Both are initialized to 0 in `train.py:164-167`. | W&B `tokens_total` and `n_tokens.*` metrics reset to 0 on every resume, creating discontinuities in monitoring. |
| **EPOCHS_KEY dead** | Moderate | `EPOCHS_KEY` written to every checkpoint but no longer semantically meaningful | The plan (B2 fix) derives `epochs_run` from `global_step // steps_per_epoch`, making the saved `epochs_run` value redundant and potentially confusing if it disagrees. | No functional bug, but a maintenance hazard. |
| **B6** | Low | Remainder batches silently dropped | When `batches_per_epoch % gradient_accumulation_steps != 0`, the trailing incomplete accumulation window is never stepped on. | A few batches per epoch are wasted. Documented, not a correctness issue for step-based training. |

---

## 3. Design Principles

These principles guide every decision below. When in doubt, refer back here.

### P1: `global_step` is the canonical resume counter

`global_step` (optimizer updates completed) is the single source of truth for training progress. Epoch number and batch offset are derived from it. This follows the universal standard for LLM pretraining where epochs rarely complete.

### P2: `consumed_samples` is the batch-size-independent progress counter

We also track `consumed_samples` (total micro-batches processed × batch_size) as a secondary counter. This is Megatron-LM's approach and allows future flexibility if batch size ever needs to change between runs. For now, it serves primarily as a monitoring and auditability field.

### P3: Strict validation, not silent corruption

If the training configuration differs between save and resume in a way that would break step-to-data-position mapping, we raise a hard error. This is more rigorous than HuggingFace (silent corruption) and PyTorch Lightning (no validation). A `--force-resume` flag can downgrade to a warning for expert users.

### P4: Save everything needed for exact resume

The checkpoint should contain all state needed to produce a training run bitwise-identical to an uninterrupted run. This includes RNG states, scheduler state, and cumulative metrics — not just model weights and optimizer.

### P5: Step-based training with epoch bookkeeping

`max_steps` defines training duration. Epochs are a convenience for data shuffling (`sampler.set_epoch(epoch)`), not a training loop control structure. The epoch number is kept in the checkpoint for human readability but is derived from `global_step` on resume, never trusted from the checkpoint.

---

## 4. Checkpoint Schema

### New schema (version 1)

The `recipe_state.pt` file will contain:

```python
{
    # === Core training state ===
    "global_step": int,                    # optimizer updates completed (canonical counter)
    "consumed_samples": int,               # total samples processed (batch-size independent)
    "optimizer": dict,                     # optimizer.state_dict()
    "lr_scheduler": dict | None,           # scheduler.state_dict() or None if no scheduler
    "seed": int,                           # training seed for validation

    # === RNG states (for exact resume) ===
    "rng_state": {
        "python": <random.getstate()>,
        "numpy_global": <numpy.random.get_state()>,
        "torch_cpu": <torch.get_rng_state()>,
        "torch_cuda": <torch.cuda.get_rng_state_all()>,
        "cpt_prng": <PRNG.bit_generator.state>,      # our custom interleaving PRNG
    },

    # === Configuration validation ===
    "training_hparams": {
        "batch_size": int,
        "gradient_accumulation_steps": int,
        "world_size": int,
        "steps_per_epoch": int,
    },

    # === Monitoring continuity ===
    "cumulative_metrics": {
        "tokens_train_total": int,
        "token_type_counts": dict[str, int],
        "wall_clock_seconds": float,         # cumulative training wall time
    },

    # === Metadata ===
    "checkpoint_version": 1,                 # schema version for forward compat
    "timestamp": str,                        # ISO 8601
    "ssi_version": str,                      # from ssi._version.__version__
}
```

Model weights continue to be saved separately as HF-format safetensors (unchanged from current behaviour).

### Removed fields

| Field | Reason |
|-------|--------|
| `EPOCHS_KEY` ("epochs_run") | Derived from `global_step // steps_per_epoch`. Saving it creates a source of confusion when it disagrees with the derived value. |

### Backward compatibility

When loading a checkpoint that lacks `checkpoint_version` (i.e. any existing checkpoint):
- Emit `LOGGER.warning("Legacy checkpoint format detected. ...")`.
- Fall back to reading `EPOCHS_KEY` and `GLOBAL_STEP_KEY` as before.
- `consumed_samples`, `lr_scheduler`, `rng_state`, `cumulative_metrics` all default to None/zero.
- **Do not** attempt to validate training hparams (no `training_hparams` key).

### Constants (`ssi/constants.py`)

```python
# New keys
TRAINING_HPARAMS_KEY: str = "training_hparams"
LR_SCHEDULER_KEY: str = "lr_scheduler"
RNG_STATE_KEY: str = "rng_state"
CONSUMED_SAMPLES_KEY: str = "consumed_samples"
CUMULATIVE_METRICS_KEY: str = "cumulative_metrics"
CHECKPOINT_VERSION_KEY: str = "checkpoint_version"
CHECKPOINT_VERSION: int = 1
```

---

## 5. Resume Behaviour Specification

### Resume sequence

On resume (when `recipe_checkpoint is not None`):

1. **Load checkpoint** — `checkpointer.load_checkpoint()` (unchanged).
2. **Validate seed** — raise `ValueError` if `ckpt_dict[SEED_KEY] != SEED`.
3. **Restore model weights** — `model.load_state_dict(ckpt_dict[MODEL_KEY])` (unchanged).
4. **Restore optimizer** — `optimizer.load_state_dict(ckpt_dict[OPTIMIZER_KEY])`.
5. **Early validation** — validate `gradient_accumulation_steps` and `world_size` (these are known before data setup).
6. **Set up data** — create DataLoader, compute `batches_per_epoch`, `steps_per_epoch`.
7. **Full validation** — validate `batch_size` and `steps_per_epoch` against checkpoint (requires data setup to have completed).
8. **Restore LR scheduler** — `scheduler.load_state_dict(ckpt_dict[LR_SCHEDULER_KEY])` if present, else reconstruct from `global_step` (legacy fallback).
9. **Restore RNG states** — restore all 5 RNG states if `rng_state` is present.
10. **Compute resume position**:
    ```python
    global_step = ckpt_dict[GLOBAL_STEP_KEY]
    epochs_run = global_step // steps_per_epoch
    batches_to_skip = (global_step % steps_per_epoch) * gradient_accumulation_steps
    ```
11. **Restore cumulative metrics** — `tokens_train_total`, `token_type_counts_total`, `wall_clock_seconds` if present.
12. **Enter training loop** at `epoch=epochs_run`, skipping `batches_to_skip` batches in the first epoch.

### Correctness invariants

| Scenario | Expected behaviour |
|----------|-------------------|
| Fresh start | `global_step=0`, `epochs_run=0`, `batches_to_skip=0` |
| Mid-epoch resume (step N, N % S != 0) | Skip `(N % S) * G` batches in epoch `N // S` |
| Exact epoch boundary (N % S == 0) | Start epoch `N // S` from batch 0, no skip |
| Config mismatch | `ValueError` raised before any training |
| Legacy checkpoint | Warning, proceed with derived values, no hparam validation |

(S = steps_per_epoch, G = gradient_accumulation_steps)

---

## 6. Data Position Recovery

### Current approach: islice skip

```python
if epoch == epochs_run and batches_to_skip > 0:
    data_iter = itertools.islice(enumerate(data_train), batches_to_skip, None)
else:
    data_iter = enumerate(data_train)
```

### Why islice is acceptable for now

- **Simplicity**: No new dependencies, no changes to the DataLoader/Dataset API.
- **Our context**: `num_workers=0`, so each skipped batch is a synchronous CPU load. This is slow but not catastrophically so for our scale.
- **Correctness with RNG fix**: Once we save and restore the `cpt.py` PRNG state (Section 7), the islice approach produces the same RNG sequence as an uninterrupted run, because `__getitem__` is called for each skipped batch and advances the PRNG correctly. **Wait — this is wrong.** `islice` on the DataLoader iterator skips yielded batches but *does* call `__getitem__` on the underlying Dataset for each skipped batch (because `num_workers=0` means the main process calls `__getitem__` synchronously for each batch). So with `num_workers=0`, islice *does* advance the PRNG correctly. **This means the PRNG concern from the critique is actually not an issue for islice specifically** — but it IS an issue if we ever move to `num_workers>0` or to a seek-based approach.

**Correction to the original critique**: With `num_workers=0`, `itertools.islice` on `enumerate(data_train)` does iterate through the DataLoader, which calls `Dataset.__getitem__()` for each batch, which calls `interleave()`, which advances `PRNG`. The PRNG *is* advanced during the skip. The critique was wrong on this point.

**However**, the PRNG state is still not *saved/restored* from the checkpoint. After a resume, the PRNG starts from `PRNG = np.random.default_rng(SEED)` (position 0), and the islice skip advances it by `batches_to_skip * batch_size` calls — which is correct only if the PRNG was at position 0 at the start of the current epoch. Since `DistributedSampler.set_epoch(epoch)` reshuffles the data order per epoch, and the PRNG in `cpt.py` is independent of the sampler, the PRNG state at the start of epoch E depends on all samples processed in epochs 0..E-1. For E=0 (the common case in our single-epoch training), the PRNG starts at position 0 and the islice skip correctly advances it. **For E>0, the PRNG would be wrong.** Since we never complete an epoch in practice, this is not currently a problem, but it must be fixed for correctness.

### Skip cost estimate

For MLS at batch_size=16, worst case (checkpoint near end of epoch):
- `batches_per_epoch` ≈ 44000 * 3600 / (avg_sample_duration * 16) — this depends on sample lengths
- Each skipped batch involves: HF datasets lookup + tokenization + collation
- Estimate: ~100-500 batches/second on CPU → 1-100 minutes for a full-epoch skip

### Future: `torchdata.StatefulDataLoader`

`torchdata >= 0.8.0` provides `StatefulDataLoader` as a drop-in replacement for `torch.utils.data.DataLoader` with `state_dict()` / `load_state_dict()` methods. This eliminates the skip cost entirely by saving the sampler position. It also handles the `num_workers` constraint cleanly. This is the recommended medium-term upgrade path.

### Future: Megatron-style consumed_samples

For the long term, the gold standard is Megatron-LM's approach: a deterministic pre-shuffled index mapping where `consumed_samples` directly indexes into the sequence. No skip, no stateful dataloader — just `dataset[consumed_samples:]`. This requires restructuring the data pipeline but is the most robust solution for datasets that never complete an epoch.

---

## 7. RNG State Management

### What we need to save

| RNG | Used by | Save | Restore |
|-----|---------|------|---------|
| Python `random` | `training.set_seed()` | `random.getstate()` | `random.setstate(state)` |
| NumPy global | `training.set_seed()` | `numpy.random.get_state()` | `numpy.random.set_state(state)` |
| PyTorch CPU | `training.set_seed()`, dropout | `torch.get_rng_state()` | `torch.set_rng_state(state)` |
| PyTorch CUDA | dropout, `DistributedSampler` | `torch.cuda.get_rng_state_all()` | `torch.cuda.set_rng_state_all(states)` |
| `cpt.py` PRNG | interleaving span boundaries, `start_with_text` | `PRNG.bit_generator.state` | `PRNG.bit_generator.state = state` |

### When to save

RNG states are saved as part of `recipe_state.pt` at every checkpoint.

### When to restore

Restore immediately after loading the checkpoint, before entering the training loop. This ensures that any code between restore and the first training batch consumes the same RNG sequence as the original run.

### The `cpt.py` PRNG problem

The module-level `PRNG = np.random.default_rng(SEED)` in `cpt.py:41` creates several issues:
1. It's shared between train and dev `TextCompletionDataset` instances.
2. It's module-level, so saving/restoring it requires reaching into the module's global state.
3. It means dev-set evaluation interleaving depends on training history.

**Fix (this plan)**: Make PRNG a constructor argument to `TextCompletionDataset`, stored as `self._prng`. The train and dev datasets get separate PRNG instances. The train dataset's PRNG state is saved in the checkpoint. The dev dataset's PRNG is re-seeded deterministically each evaluation.

```python
class TextCompletionDataset(Dataset):
    def __init__(self, ..., prng: np.random.Generator | None = None):
        self._prng = prng or np.random.default_rng(SEED)
        ...
```

`interleave()` and `get_span_idxs_binomial()` become methods (or take `prng` as a parameter) instead of using the module-level global.

---

## 8. Learning Rate Schedule Resume

### Current state

`lr_schedule.py` creates a fresh `LambdaLR` with `last_epoch=global_step-1` on every resume. This works for the specific cosine-with-warmup schedule because `LambdaLR` applies `lr_lambda(last_epoch + 1)` = `lr_lambda(global_step)` on the first step. But:
- It doesn't save/restore `scheduler.state_dict()`.
- It won't work for schedules with internal state beyond `last_epoch` (e.g., ReduceLROnPlateau, cyclic schedules).
- It reconstructs the schedule from scratch, which means changing schedule parameters between runs silently takes effect — this might be intentional (warm restart) or a bug (accidental).

### Fix

1. **Save** `scheduler.state_dict()` in the checkpoint under `LR_SCHEDULER_KEY`.
2. **On resume**: create a fresh scheduler (same as now), then `scheduler.load_state_dict(ckpt_dict[LR_SCHEDULER_KEY])`.
3. **Legacy fallback**: if `LR_SCHEDULER_KEY` not in checkpoint, use the current `last_epoch` approach with a warning.
4. **Warm restart detection**: if the user has changed LR schedule parameters in the config vs the checkpoint, warn but proceed. This is a legitimate use case (fine-tuning from a CPT checkpoint with a different schedule).

### Future: WSD schedule

The Warmup-Stable-Decay (WSD) schedule is ideal for our use case:
- The stable phase runs at constant LR indefinitely — checkpointing is trivial (no schedule state needed).
- No need to know `total_steps` in advance (unlike cosine).
- Branch off a stable-phase checkpoint for a short decay phase when you want a final model.

This is a future enhancement, not part of the current implementation plan.

---

## 9. Configuration Validation on Resume

### What we validate

| Field | When checkable | Severity on mismatch |
|-------|---------------|---------------------|
| `seed` | Immediately | Hard error (always) |
| `gradient_accumulation_steps` | Before data setup | Hard error |
| `world_size` | Before data setup | Hard error |
| `batch_size` | After data setup | Hard error |
| `steps_per_epoch` | After data setup | Hard error (derived cross-check) |

### Why hard errors, not warnings

HuggingFace's approach (warn-only on batch_size, ignore grad_accum and world_size) is a known source of silent training corruption. Users who hit this spend hours debugging why their loss spiked or their model diverged. A hard error with a clear message ("your batch_size was 16, now it's 32 — this breaks step-to-data-position mapping") is far more helpful.

### Override mechanism

For expert users who understand the consequences:
```python
# In config:
force_resume: bool = False  # set to True to downgrade validation errors to warnings
```

When `force_resume=True`, mismatches log `LOGGER.warning(...)` instead of raising. The training proceeds but the data position will be wrong (some batches repeated, others skipped). This is acceptable for fine-tuning from a checkpoint with different batch size, where exact data position doesn't matter.

### Early vs late validation

We split validation into two phases to avoid wasting time on data setup when we know the config is wrong:
1. **Early** (before data setup): `seed`, `gradient_accumulation_steps`, `world_size`.
2. **Late** (after data setup): `batch_size`, `steps_per_epoch`.

---

## 10. Monitoring Continuity Across Resume

### Problem

Currently, `tokens_train_total`, `token_type_counts_total`, and training wall time all reset to 0 on resume. This creates discontinuities in W&B charts that make it hard to track cumulative progress.

### Fix

Save and restore cumulative metrics:

```python
"cumulative_metrics": {
    "tokens_train_total": int,
    "token_type_counts": dict[str, int],   # {"text": N, "dsu": M, "special_text": K, ...}
    "wall_clock_seconds": float,
}
```

On resume, initialize from checkpoint values instead of 0. On fresh start, initialize to 0 as before.

---

## 11. Implementation Plan

### Phase 1: Core checkpoint schema (single commit)

**Files modified:**

| File | Changes |
|------|---------|
| `ssi/constants.py` | Add new key constants (`TRAINING_HPARAMS_KEY`, `LR_SCHEDULER_KEY`, `RNG_STATE_KEY`, etc.) |
| `ssi/checkpoint.py` | Update `save_checkpoint` to accept and save all new fields; add `save_rng_states()` / `load_rng_states()` helpers |
| `ssi/train.py` | Update `resume_training_state` to return new fields; add `validate_resume_hparams`; update save call site; update training loop for islice skip; restore cumulative metrics |
| `ssi/lr_schedule.py` | No changes to `setup_lr_scheduler`; scheduler state_dict saved/restored in `train.py` |

**Detailed changes:**

*`ssi/constants.py`*:
```python
TRAINING_HPARAMS_KEY: str = "training_hparams"
LR_SCHEDULER_KEY: str = "lr_scheduler"
RNG_STATE_KEY: str = "rng_state"
CONSUMED_SAMPLES_KEY: str = "consumed_samples"
CUMULATIVE_METRICS_KEY: str = "cumulative_metrics"
CHECKPOINT_VERSION_KEY: str = "checkpoint_version"
CHECKPOINT_VERSION: int = 1
```

*`ssi/checkpoint.py`* — `save_checkpoint` gains parameters:
```python
def save_checkpoint(
    self,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None,
    global_step: int,
    seed: int,
    *,
    lr_scheduler_state_dict: dict[str, Any] | None = None,
    rng_state: dict[str, Any] | None = None,
    training_hparams: dict[str, Any] | None = None,
    consumed_samples: int | None = None,
    cumulative_metrics: dict[str, Any] | None = None,
    save_training_state: bool = True,
    adapter_only: bool = False,
    output_dir: Path | None = None,
    ignore_suffixes: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
```

The `epoch` parameter is removed. The output directory uses `step_{global_step}` instead of `epoch_{epoch}/global_step_{global_step}`.

*`ssi/train.py`* — `resume_training_state` simplified:
```python
def resume_training_state(ckpt_dict: dict[str, Any]) -> dict[str, Any]:
    """Extract resume state from a checkpoint dict. Returns a dict with all resume fields."""
    if SEED != ckpt_dict[SEED_KEY]:
        raise ValueError(f"Seed mismatch: config={SEED}, checkpoint={ckpt_dict[SEED_KEY]}")
    return {
        "global_step": ckpt_dict[GLOBAL_STEP_KEY],
        "optimizer_state": ckpt_dict[OPTIMIZER_KEY],
        "lr_scheduler_state": ckpt_dict.get(LR_SCHEDULER_KEY),
        "rng_state": ckpt_dict.get(RNG_STATE_KEY),
        "training_hparams": ckpt_dict.get(TRAINING_HPARAMS_KEY),
        "consumed_samples": ckpt_dict.get(CONSUMED_SAMPLES_KEY, 0),
        "cumulative_metrics": ckpt_dict.get(CUMULATIVE_METRICS_KEY),
    }
```

*`ssi/train.py`* — training loop changes:
```python
# After data setup
batches_per_epoch = len(data_train)
steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
assert steps_per_epoch > 0, f"batches_per_epoch ({batches_per_epoch}) < gradient_accumulation_steps ({cfg.gradient_accumulation_steps})"

# Compute resume position
epochs_run = global_step // steps_per_epoch
batches_to_skip = (global_step % steps_per_epoch) * cfg.gradient_accumulation_steps
assert batches_to_skip < batches_per_epoch, f"batches_to_skip ({batches_to_skip}) >= batches_per_epoch ({batches_per_epoch})"

# Training loop
for epoch in range(epochs_run, n_epochs):
    sampler_train.set_epoch(epoch)
    if epoch == epochs_run and batches_to_skip > 0:
        LOGGER.info(f"Resuming: skipping {batches_to_skip} batches in epoch {epoch}")
        data_iter = itertools.islice(enumerate(data_train), batches_to_skip, None)
        n_batches = batches_per_epoch - batches_to_skip
    else:
        data_iter = enumerate(data_train)
        n_batches = batches_per_epoch
    for i, batch in tqdm(data_iter, total=n_batches):
        ...
```

### Phase 2: PRNG refactor (separate commit)

Move the module-level `PRNG` in `cpt.py` to a per-instance attribute on `TextCompletionDataset`.

**Files modified:**

| File | Changes |
|------|---------|
| `ssi/data/cpt.py` | Remove module-level `PRNG`; add `prng` parameter to `TextCompletionDataset.__init__`; pass `self._prng` to `interleave` and `get_span_idxs_binomial` |
| `ssi/data/__init__.py` | Pass PRNG instance when constructing `TextCompletionDataset` |
| `ssi/train.py` | Create PRNG instance, pass to data setup, save/restore its state |

### Phase 3: Tests (separate commit)

See Section 12.

---

## 12. Test Plan

### Unit tests (no GPU, no filesystem)

| ID | Test | What it validates |
|----|------|-------------------|
| T1 | `resume_training_state` returns correct dict from well-formed checkpoint | Basic schema contract |
| T2 | Seed mismatch raises `ValueError` | Seed validation |
| T3 | Missing keys raise `KeyError` | Schema completeness |
| T4 | `validate_resume_hparams` passes on matching config | Happy path |
| T5 | `validate_resume_hparams` raises on each mismatch type | batch_size, grad_accum, world_size, steps_per_epoch |
| T6 | `validate_resume_hparams` warns on legacy checkpoint (no hparams key) | Backward compat |
| T7 | `validate_resume_hparams` warns (not errors) with `force_resume=True` | Override mechanism |
| T8 | Resume position arithmetic: mid-epoch | `epochs_run`, `batches_to_skip` correct |
| T9 | Resume position arithmetic: exact epoch boundary | `batches_to_skip == 0` |
| T10 | Resume position arithmetic: fresh start | All zero |
| T11 | `batches_to_skip < batches_per_epoch` assertion | Guard against impossible skip |

### Integration tests (filesystem, no GPU)

| ID | Test | What it validates |
|----|------|-------------------|
| T12 | Round-trip: save then load returns identical state | On-disk serialization correctness |
| T13 | Checkpoint contains `GLOBAL_STEP_KEY`, not `STEPS_KEY` | B1 regression guard |
| T14 | Checkpoint does NOT contain `EPOCHS_KEY` | Schema cleanup verification |
| T15 | Legacy checkpoint (no new keys) loads with warnings | Backward compat |
| T16 | RNG state round-trip: save, restore, generate — matches uninterrupted sequence | RNG preservation |

### Smoke tests (GPU, cluster)

| ID | Test | What it validates |
|----|------|-------------------|
| T17 | Train 10 steps, save, resume, verify step 11 is correct | End-to-end resume |
| T18 | Resume with different `gradient_accumulation_steps` → error | Config validation |
| T19 | W&B `tokens_total` is continuous across resume boundary | Monitoring continuity |

---

## 13. Design Decision Log

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
**Rejected**: `torchdata.StatefulDataLoader` (for now — see Future Work).
**Why**: islice is zero-dependency and works correctly with `num_workers=0`. StatefulDataLoader requires `torchdata >= 0.8.0` and changes to the data pipeline. The skip cost is acceptable for our current scale. We'll upgrade when we move to `num_workers > 0` or when skip times become a bottleneck.

### D4: Save LR scheduler state_dict (not reconstruct from global_step)

**Chose**: Save and restore `scheduler.state_dict()`.
**Rejected**: Reconstruct from `last_epoch=global_step` (current approach).
**Why**: The reconstruction approach only works for specific schedule types. Saving the state_dict is universal, costs negligible space, and is the standard practice in every framework except torchtune.

### D5: Remove `EPOCHS_KEY` from checkpoint (not keep for compat)

**Chose**: Remove `EPOCHS_KEY` entirely from new checkpoints.
**Rejected**: Keep writing it for human readability.
**Why**: A field that is written but never read on resume is a maintenance hazard. It will inevitably diverge from `global_step // steps_per_epoch` and confuse someone inspecting the checkpoint. The epoch can be trivially derived when needed.

### D6: Per-instance PRNG (not module-level)

**Chose**: Constructor-injected `np.random.Generator` per `TextCompletionDataset` instance.
**Rejected**: Module-level global PRNG (current state).
**Why**: Module-level state is: (a) shared between train and dev, making dev evaluation non-deterministic; (b) impossible to save/restore cleanly without reaching into module globals; (c) incompatible with multi-worker DataLoader (each worker would share the same PRNG state).

### D7: Drop `epoch` parameter from `save_checkpoint`

**Chose**: Remove `epoch` from `save_checkpoint` signature; use `step_{global_step}` directory naming.
**Rejected**: Keep `epoch` in the API.
**Why**: Per principle P5, epoch is derived from global_step. Having it as a parameter creates a source of inconsistency. The directory structure `step_N/` is clearer than `epoch_E/global_step_N/`.

---

## 14. Future Work

These items are explicitly out of scope for the current implementation but are documented for planning.

### F1: `torchdata.StatefulDataLoader`
Replace `torch.utils.data.DataLoader` with `torchdata.StatefulDataLoader` to eliminate islice skip cost. Requires `torchdata >= 0.8.0`. The DataLoader state_dict would be saved in the checkpoint alongside the other state. This also enables `num_workers > 0` without PRNG state issues (each worker's state is captured by the StatefulDataLoader).

### F2: Megatron-style consumed_samples indexing
Pre-shuffle the dataset with a deterministic seed and use `consumed_samples` to index directly into the shuffled sequence. No skip, no stateful dataloader. This is the most robust solution for datasets that never complete an epoch.

### F3: WSD learning rate schedule
Implement Warmup-Stable-Decay schedule for checkpoint-friendly training. The stable phase has trivial scheduler state (constant LR), making resume simpler. Natural branching for producing final models at different compute budgets. See research doc section 2 for details.

### F4: Checkpoint retention policy
Currently all checkpoints are kept. Implement a policy like "keep last N + every Kth checkpoint" to manage disk usage. HuggingFace and PyTorch Lightning both have this built in.

### F5: Async checkpointing
Save checkpoints without blocking training. PyTorch DCP and torchtitan support this. Useful when checkpoint save time is a significant fraction of step time.

### F6: Batch size change support
Using `consumed_samples` as the primary counter would allow changing batch size between runs. The training loop would derive the step count from `consumed_samples / new_effective_batch_size`. This requires F2 (consumed_samples indexing) to work correctly for data position.
