# Refactor — Stateful Trainer Class

> **Status: COMPLETE.** All phases implemented, verified, and cut over. The functional `train()` has been removed. The `Trainer` class is the sole training implementation. See commit history `origin/dev..HEAD` on `stateful-trainer` for the full change set.

## Motivation

The training pipeline previously used a single 230-line function (`train()`) in `ssi/train.py` that managed 20+ local variables representing training state. This made the code hard to test, inspect, and extend.

The refactor replaced it with a stateful `Trainer` class in `ssi/trainer.py` following torchtune's `FTRecipeInterface` pattern: class-based state encapsulation, composition over inheritance, explicit `setup()` / `train()` / `save_checkpoint()` / `cleanup()` methods.

---

## Architecture (as implemented)

### Module: `ssi/trainer.py`

Contains `Trainer` and `TrainingGeometry`. Exports: `__all__ = ["Trainer", "TrainingGeometry"]`.

### Module: `ssi/train_utils.py`

Pure utility functions extracted from the former `ssi/train.py`:
- `validate_train_cfg(cfg)` — config validation
- `resume_training_state(ckpt_dict)` — checkpoint dict parsing
- `validate_resume_hparams(...)` — hparam mismatch checking
- `get_token_type_ranges(llama_config)` — vocab range computation
- `count_token_types(tokens, ranges, pad_idx)` — token counting

### Deleted: `ssi/train.py`

The functional `train()` was removed after equivalence was proven. All utility functions were moved to `ssi/train_utils.py`.

---

## Detailed Class Design (as implemented)

### `TrainingGeometry` (dataclass)

```python
@dataclass(frozen=True)
class TrainingGeometry:
    batch_size: int
    batches_per_epoch: int
    steps_per_epoch: int
    usable_batches: int        # steps_per_epoch * gradient_accumulation_steps
    n_epochs: int
    gradient_accumulation_steps: int
    world_size: int
```

Factory: `TrainingGeometry.from_config(cfg, dataloader, world_size)`.

- [x] All 7 fields implemented
- [x] `from_config()` factory with remainder warning and `ValueError` on insufficient batches (improvement over original `assert`)

### `Trainer` Class — Attributes

All 24 planned attributes are present, plus two additions:

| Attribute | Category | Status |
|---|---|---|
| `self.cfg` | Config | [x] Done |
| `self.device` | Config | [x] Done |
| `self.dtype` | Config | [x] Done |
| `self.world_size` | Config | [x] Done |
| `self.model` | Components | [x] Done |
| `self.tokenizer` | Components | [x] Done |
| `self.optimizer` | Components | [x] Done |
| `self.lr_scheduler` | Components | [x] Done |
| `self.loss_fn` | Components | [x] Done |
| `self.checkpointer` | Components | [x] Done |
| `self.wandb_logger` | Components | [x] Done |
| `self.data_train` | Data | [x] Done |
| `self.sampler_train` | Data | [x] Done |
| `self.data_dev` | Data | [x] Done |
| `self.token_type_ranges` | Data | [x] Done |
| `self.global_step` | Training state | [x] Done |
| `self.consumed_samples` | Training state | [x] Done |
| `self.tokens_train_total` | Training state | [x] Done |
| `self.token_type_counts_total` | Training state | [x] Done |
| `self.wall_clock_offset` | Training state | [x] Done |
| `self.loss_running` | Accumulators | [x] Done |
| `self.num_tokens_step` | Accumulators | [x] Done |
| `self.max_seq_len_step` | Accumulators | [x] Done |
| `self.t_train_start` | Timing | [x] Done |
| `self.t_step_start` | Timing | [x] Done |
| `self.geometry` | TrainingGeometry | [x] Done |
| `self._grad_norm` | Logging (added) | [x] Done |
| `self._loss_log` | Testing (added) | [x] Done |

### `Trainer` Class — Methods

```
Trainer(cfg: DictConfig)
├── setup() -> None
│   ├── _setup_logging() -> None
│   ├── _setup_model() -> None
│   ├── _setup_tokenizer() -> None
│   ├── _extract_resume_state() -> None    # deviation: split from _resume()
│   ├── _setup_optimizer() -> None
│   ├── _setup_loss() -> None
│   ├── _setup_data() -> None
│   ├── TrainingGeometry.from_config()
│   ├── _finalize_resume() -> None         # deviation: split from _resume()
│   └── del self._ckpt_dict               # memory cleanup
├── train() -> None
│   ├── restore_rng_states()               # deviation: moved from _finalize_resume
│   └── _train_epoch(epoch, batches_to_skip) -> None
│       ├── _train_step(batch) -> None
│       └── _optimizer_step(epoch, iter_idx) -> None
│           ├── _log_metrics(epoch, iter_idx, loss_to_log) -> None
│           │   └── _evaluate() -> float
│           ├── _reset_step_accumulators() -> None
│           └── _maybe_save_checkpoint() -> None
├── save_checkpoint() -> None
└── cleanup() -> None
```

### Documented Deviations from Original Plan

1. **`_resume()` split**: The plan specified a single `_resume()` method. The implementation splits it into `_extract_resume_state()` (runs before optimizer creation, provides optimizer state dict) and `_finalize_resume()` (runs after geometry computation, validates hparams, restores cumulative metrics). `restore_rng_states()` moved to `train()` — semantically identical (runs after all setup, before loop).

2. **`_train_step` return type**: Plan specified `-> float`, implementation returns `None`. Loss is accumulated into `self.loss_running` and logged in `_optimizer_step`.

3. **`del batch` / `empty_cache()` location**: Plan placed these inside `_train_step`. Implementation places them in `_train_epoch` (after `_train_step` returns). Equivalent behavior.

4. **`_optimizer_step` signature**: Plan specified `(epoch)`, implementation uses `(epoch, iter_idx)`. The `iter_idx` is needed for console log formatting. `_optimizer_step` also subsumes `_log_metrics`, `_reset_step_accumulators`, and `_maybe_save_checkpoint` calls.

5. **`scale_grads` argument**: Plan used `1 / self.num_tokens_step` (float). Implementation uses `torch.tensor(1 / self.num_tokens_step)`. This is a bug fix — torchtune's `scale_grads` requires a `torch.Tensor`, not a float.

6. **`_ckpt_dict` cleanup**: Not in original plan. Added after code review to prevent multi-GB memory leak during training.

7. **Timer ordering**: Code review caught that `_reset_step_accumulators` and `_maybe_save_checkpoint` were swapped relative to the original. Fixed to match: timer resets before checkpoint save.

---

## Implementation Plan — Completion Status

### Phase 1: Scaffolding — [x] COMPLETE

| Step | Description | Commit |
|---|---|---|
| [x] 1.1 | Trainer skeleton | `8f89abf` |
| [x] 1.2 | TrainingGeometry dataclass | `8f89abf` |
| [x] 1.3 | Unit tests | `8f89abf` |

### Phase 2: Setup Methods — [x] COMPLETE

| Step | Description | Commit |
|---|---|---|
| [x] 2.1 | `_setup_model()` | `8f89abf` |
| [x] 2.2 | `_setup_tokenizer()` | `8f89abf` |
| [x] 2.3 | `_setup_data()` | `8f89abf` |
| [x] 2.4 | `_setup_optimizer()` | `8f89abf`, `98da781` (ordering fix) |
| [x] 2.5 | `_setup_loss()` | `8f89abf` |
| [x] 2.6 | `_setup_logging()` | `8f89abf` |
| [x] 2.7 | Resume (`_extract_resume_state` + `_finalize_resume`) | `8f89abf`, `98da781` |
| [x] 2.8 | Wire `setup()` | `8f89abf` |
| [x] 2.9 | TrainingGeometry after data setup | `8f89abf` |

### Phase 3: Training Loop — [x] COMPLETE

| Step | Description | Commit |
|---|---|---|
| [x] 3.1 | `_reset_step_accumulators()` | `8f89abf` |
| [x] 3.2 | `_train_step(batch)` | `8f89abf` |
| [x] 3.3 | `_optimizer_step(epoch, iter_idx)` | `8f89abf` |
| [x] 3.4 | `_evaluate()` | `8f89abf` |
| [x] 3.5 | `_log_metrics(epoch, iter_idx, loss_to_log)` | `8f89abf` |
| [x] 3.6 | `_maybe_save_checkpoint()` | `8f89abf` |
| [x] 3.7 | `_train_epoch(epoch, batches_to_skip)` | `8f89abf` |
| [x] 3.8 | `train()` outer loop | `8f89abf` |
| [x] 3.9 | `save_checkpoint()` | `8f89abf` |
| [x] 3.10 | `cleanup()` | `8f89abf` |

### Phase 4: Tests — [x] COMPLETE

| Test | Description | File | Status |
|---|---|---|---|
| [x] T-U1 | Construction | `test_trainer.py` | Passing |
| [x] T-U2 | TrainingGeometry values | `test_trainer.py` | Passing (4 variants) |
| [x] T-U3 | TrainingGeometry remainder warning | `test_trainer.py` | Passing |
| [x] T-U4 | Geometry insufficient batches raises | `test_trainer.py` | Passing |
| [x] T-U5 | `_extract_resume_state` | `test_trainer.py` | Passing |
| [x] T-U6 | `_reset_step_accumulators` | `test_trainer.py` | Passing |
| [x] T-U7 | `_optimizer_step` counters | `test_trainer.py` | Passing (2 tests) |
| [x] T-U8 | `_maybe_save_checkpoint` | `test_trainer.py` | Passing (3 tests) |
| [x] T-U9 | `save_checkpoint` args | `test_trainer.py` | Passing |
| [x] T-U10 | `_loss_log` recording | `test_trainer.py` | Passing (2 tests) |
| [x] T-I1 | Setup smoke | `test_trainer_gpu.py` | Passing |
| [x] T-I2 | Single train step | `test_trainer_gpu.py` | Passing |
| [x] T-I3 | Optimizer step | `test_trainer_gpu.py` | Passing |
| [x] T-I4 | Evaluation | `test_trainer_gpu.py` | Passing |
| [x] T-I5 | Full train run | `test_trainer_gpu.py` | Passing |
| [x] T-I6 | Functional equivalence | `test_equivalence.py` (tiny model), `test_equivalence_e2e.py` (real model) | Passing — bit-identical losses, weights, W&B metrics, checkpoints verified over 200 steps |
| [x] T-I7 | Resume equivalence | `test_resume_equivalence.py` | Passing — losses and W&B metrics match between interrupted+resumed and uninterrupted runs |
| [x] T-I8 | CPT equivalence | Was `test_cpt_equivalence.py` — passed, then removed after cutover (no functional baseline to compare against). CPT path tested implicitly via T-I1–I5. |

**Note on T-I6/T-I8 test files removed post-cutover:** `test_equivalence_e2e_full.py`, `test_cpt_equivalence.py`, and `verify_full_equivalence.py` were removed in commit `bd11ded` because they compared the functional `train()` against the Trainer — once `train()` was deleted, these tests had no baseline. The equivalence was proven before deletion (200-step verification: 2,394 comparisons, 0 failures, byte-identical SHA256 checkpoints across separate CLI processes).

### Phase 5: Cutover — [x] COMPLETE

| Step | Description | Commit |
|---|---|---|
| [x] 5.1 | Update `scripts/train_sft.py` and `scripts/train_cpt.py` to use Trainer | `c43749c` |
| [x] 5.2 | Add `__all__` export | `96ece6c` |
| [x] 5.3 | Remove functional `train()` (stronger than planned deprecation comment) | `bd11ded` |

**Deviation from plan:** Step 5.3 originally called for a deprecation comment on `train()`. Instead, the entire `ssi/train.py` was deleted and its utility functions moved to `ssi/train_utils.py`. This is a stronger action — cleaner than leaving deprecated code around.

---

## Additional Work Beyond Original Plan

### Bug Fixes

| Fix | Commit | Description |
|---|---|---|
| `scale_grads` type error | `9716628` | Both `train.py` and `trainer.py` passed a Python `float` to `scale_grads()` which requires `torch.Tensor`. Latent bug — would crash on CUDA. |
| Memory leak | `13f4641` | `self._ckpt_dict` (several GB) persisted through training. Fixed by deleting at end of `setup()`. |
| Timer ordering | `13f4641` | `_reset_step_accumulators` and `_maybe_save_checkpoint` were swapped vs original, affecting `duration_step` W&B metric. |

### Hydra Config Fix

| Item | Commit | Description |
|---|---|---|
| `_base_` defaults resolution | `4cad4c9` | The modularization commit `b05b53c` introduced two bugs: (1) Hydra couldn't resolve `defaults: [_base_]` across config group boundaries, (2) `${_source}` interpolation resolved from the wrong scope. Fixed by moving base configs to `conf/data/_sft_base.yaml` and `conf/data/_cpt_base.yaml`, eliminating all cross-file interpolations, using `???` mandatory markers. All config values verified identical. |

### `n_samples` Data Subsetting Feature

| Item | Commit | Description |
|---|---|---|
| Streaming subset | `ffe32ca` | Added `n_samples` parameter to `SFTDataset` and `TextCompletionDataset`. Uses HF `streaming=True` + `.take(n)` to load only the first N samples without downloading the full dataset. Added `load_dataset_subset()` helper to `ssi/data/__init__.py`. Added `n_samples: null` to all Hydra data configs. CLI usage: `data.train.dataset.n_samples=2000`. |

---

## Verification Summary

### Equivalence Proof

The Trainer was proven bit-identical to the functional `train()` across multiple levels:

1. **Tiny model test** (`test_equivalence.py`): 4 optimizer steps, random 256-vocab 2-layer Llama, exact `torch.equal` on every parameter.

2. **Real model test** (`test_equivalence_e2e.py`): 6 steps with Llama 3.2 1B, 2k MLS-HuBERT samples, exact loss and weight match with deterministic CUDA.

3. **Full E2E with W&B + checkpoints** (formerly `test_equivalence_e2e_full.py`): 4 steps through real `train()` and `Trainer.setup()+train()` paths, compared all W&B `log_dict` payloads, safetensors checkpoint weights, and training state optimizer buffers.

4. **200-step standalone verification** (formerly `verify_full_equivalence.py`): 200 optimizer steps with eval at 50/100/150/200, checkpoints at 100/200. **2,394 comparisons, 0 failures**. Dev losses, token counts, optimizer buffers, and model weights all bit-identical.

5. **CLI byte-identical checkpoints**: Both `scripts/train_sft.py` and `scripts/train_sft_trainer.py` run via real CLI (separate processes) with `debug_mode=2 optimizer.fused=False CUBLAS_WORKSPACE_CONFIG=:4096:8`. SHA256 of safetensors files match exactly at step 100 and step 200.

6. **Resume equivalence** (`test_resume_equivalence.py`): 8-step run interrupted at step 4 and resumed produces identical losses at steps 5–8 as an uninterrupted 8-step run.

7. **CPT equivalence** (formerly `test_cpt_equivalence.py`): Functional and Trainer produced identical losses and W&B metrics for CPT with interleaved text-speech sequences.

### Code Review

Code review agent (`origin/dev..HEAD`) confirmed:
- All config values preserved exactly across the restructuring
- No dangling imports after `ssi/train.py` deletion
- No pre-existing test files removed
- Memory leak and timer ordering issues identified and fixed

---

## Final File Inventory

| File | Status |
|---|---|
| `ssi/trainer.py` | **CREATED** — Trainer + TrainingGeometry |
| `ssi/train_utils.py` | **CREATED** — extracted utility functions |
| `ssi/train.py` | **DELETED** — replaced by Trainer |
| `ssi/data/__init__.py` | **MODIFIED** — added `load_dataset_subset()` |
| `ssi/data/sft.py` | **MODIFIED** — added `n_samples` parameter |
| `ssi/data/cpt.py` | **MODIFIED** — added `n_samples` parameter |
| `conf/data/_sft_base.yaml` | **CREATED** — moved from `conf/data/sft/_base_.yaml` |
| `conf/data/_cpt_base.yaml` | **CREATED** — moved from `conf/data/cpt/_base_.yaml` |
| `conf/data/sft/_base_.yaml` | **DELETED** — replaced by `_sft_base.yaml` |
| `conf/data/cpt/_base_.yaml` | **DELETED** — replaced by `_cpt_base.yaml` |
| `conf/data/sft/*.yaml` (4 files) | **MODIFIED** — direct source overrides, no interpolation |
| `conf/data/cpt/*.yaml` (4 files) | **MODIFIED** — direct source + sampling_rate overrides |
| `scripts/train_sft.py` | **MODIFIED** — uses Trainer |
| `scripts/train_cpt.py` | **MODIFIED** — uses Trainer |
| `scripts/train_sft_trainer.py` | **CREATED then DELETED** — was temporary verification script |
| `scripts/plt_embed_tsne.py` | **MODIFIED** — import updated to `ssi.train_utils` |
| `tests/test_trainer.py` | **CREATED** — 16 unit tests |
| `tests/test_trainer_gpu.py` | **CREATED** — 5 GPU integration tests |
| `tests/test_equivalence.py` | **CREATED** — tiny-model equivalence (5 tests) |
| `tests/test_equivalence_e2e.py` | **CREATED** — real-model Trainer tests (4 tests) |
| `tests/test_resume_equivalence.py` | **CREATED** — resume equivalence (2 tests) |
| `tests/conftest.py` | **CREATED** — shared fixtures |
| `tests/test_checkpoint.py` | **MODIFIED** — import updated to `ssi.train_utils` |
| `ssi/eval.py` | Unchanged |
| `ssi/loss.py` | Unchanged |
| `ssi/model.py` | Unchanged |
| `ssi/optimizer.py` | Unchanged |
| `ssi/lr_schedule.py` | Unchanged |
| `ssi/checkpoint.py` | Unchanged |
| `ssi/metric_logging.py` | Unchanged |
| `ssi/constants.py` | Unchanged |
| `tests/test_cpt_deterministic_rng.py` | Unchanged |

### Test Files Removed Post-Cutover

These test files existed during the verification phase and were removed after the functional `train()` was deleted (they compared functional vs stateful — no longer applicable):

| File | Purpose | Removed in |
|---|---|---|
| `tests/test_equivalence_e2e_full.py` | Full E2E with W&B traces + checkpoints | `bd11ded` |
| `tests/test_cpt_equivalence.py` | CPT functional vs stateful | `bd11ded` |
| `tests/verify_full_equivalence.py` | 200-step standalone verification | `bd11ded` |

---

## Test Suite (final state)

72 tests passing:

```bash
# All fast tests (CPU + GPU, ~7s)
uv run pytest tests/ -v --ignore=tests/test_trainer_gpu.py \
    --ignore=tests/test_resume_equivalence.py \
    --ignore=tests/test_equivalence_e2e.py

# GPU integration tests (need extended model + MLS data, ~8 min each)
WANDB_MODE=disabled HAFH=/home/ubuntu uv run pytest tests/test_trainer_gpu.py -v

# Resume equivalence (needs extended model, ~3 min)
WANDB_MODE=disabled HAFH=/home/ubuntu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    uv run pytest tests/test_resume_equivalence.py -v

# Real model E2E (needs extended model, ~1 min)
WANDB_MODE=disabled HAFH=/home/ubuntu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    uv run pytest tests/test_equivalence_e2e.py -v

# Everything
WANDB_MODE=disabled HAFH=/home/ubuntu CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    uv run pytest tests/ -v
```
