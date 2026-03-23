# Refactor — Stateful Trainer Class

## Motivation

The current training pipeline (`ssi/train.py`) uses a single 230-line function (`train()`) that manages 20+ local variables representing training state. This makes the code:

- **Hard to test** — individual phases (setup, step, eval, checkpoint) cannot be tested in isolation
- **Hard to inspect** — there's no way to introspect training state from outside the function
- **Hard to extend** — adding CPT+SFT chaining, new loss functions, or callback hooks requires modifying a monolithic function
- **Inconsistent with the ecosystem** — nearly every major framework (HF Transformers, PyTorch Lightning, Composer, TRL, torchtune itself) uses class-based state encapsulation

### Design Reference: torchtune's FTRecipeInterface

Since this codebase depends on torchtune, we follow their `FTRecipeInterface` Protocol pattern:
- Class-based for state encapsulation
- Self-contained (no deep inheritance trees)
- Composition over inheritance
- Explicit methods: `setup()`, `train()`, `save_checkpoint()`, `load_checkpoint()`

We diverge from torchtune in one key way: we extract a **reusable `Trainer` class** rather than writing duplicate training loops per recipe. This is justified because our CPT and SFT loops are structurally identical — they differ only in data setup — and we want to guarantee they remain in sync.

---

## Architecture

### New Module: `ssi/trainer.py`

A single new file containing the `Trainer` class and the `TrainingGeometry` helper dataclass.

### Principle: 1:1 Functional Equivalence

Every line of logic in the current `train()` must have a direct counterpart in `Trainer`. The two implementations must produce **bit-identical** training runs (same loss sequence, same gradients, same checkpoints) given the same config and data. This is the core verification criterion.

---

## Detailed Class Design

### `TrainingGeometry` (dataclass)

Encapsulates the derived constants that depend on dataset size and grad accumulation. Currently these are computed as local variables in `train()` (lines 218–231).

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

### `Trainer` Class

#### Instance Attributes (State)

The following table maps every local variable in the current `train()` to a `Trainer` attribute:

| Current local variable | Trainer attribute | Category |
|---|---|---|
| `cfg` | `self.cfg` | Config |
| `DEVICE` | `self.device` | Config |
| `DTYPE` | `self.dtype` | Config |
| `world_size` | `self.world_size` | Config |
| `model` | `self.model` | Components |
| `tokenizer` | `self.tokenizer` | Components |
| `optimizer` | `self.optimizer` | Components |
| `lr_scheduler` | `self.lr_scheduler` | Components |
| `loss_fn` | `self.loss_fn` | Components |
| `checkpointer` | `self.checkpointer` | Components |
| `wandb_logger` | `self.wandb_logger` | Components |
| `data_train` | `self.data_train` | Data |
| `sampler_train` | `self.sampler_train` | Data |
| `data_dev` | `self.data_dev` | Data |
| `token_type_ranges` | `self.token_type_ranges` | Data |
| `global_step` | `self.global_step` | Training state |
| `consumed_samples` | `self.consumed_samples` | Training state |
| `tokens_train_total` | `self.tokens_train_total` | Training state |
| `token_type_counts_total` | `self.token_type_counts_total` | Training state |
| `wall_clock_offset` | `self.wall_clock_offset` | Training state |
| `loss_running` | `self.loss_running` | Step-level accumulators |
| `num_tokens_step` | `self.num_tokens_step` | Step-level accumulators |
| `max_seq_len_step` | `self.max_seq_len_step` | Step-level accumulators |
| `t_train_start` | `self.t_train_start` | Timing |
| `t0` | `self.t_step_start` | Timing |
| (derived) | `self.geometry` | `TrainingGeometry` |

#### Methods

```
Trainer(cfg: DictConfig)           # stores cfg only; no heavy init
├── setup() -> None                # all component creation + checkpoint loading + resume
│   ├── _setup_logging() -> None   # W&B logger init, tags, config logging
│   ├── _setup_model() -> None     # checkpointer, model load, llama config update
│   ├── _setup_tokenizer() -> None # tokenizer + token_type_ranges
│   ├── _setup_data() -> None      # train/dev dataloaders + samplers
│   ├── _setup_optimizer() -> None # optimizer + lr_scheduler
│   ├── _setup_loss() -> None      # loss_fn + compile + chunked output
│   └── _resume() -> None          # restore state from training_state.pt if present
├── train() -> None                # outer epoch loop, delegates to _train_epoch
│   ├── _train_epoch(epoch) -> None              # single epoch loop
│   │   ├── _train_step(batch) -> float          # single fwd + bwd (one micro-batch)
│   │   ├── _optimizer_step(epoch) -> None       # grad scale + clip + optim + lr + metrics
│   │   ├── _evaluate() -> float | None          # dev set loss
│   │   ├── _log_metrics(epoch, iter_idx) -> None # console + wandb logging
│   │   └── _maybe_save_checkpoint() -> None     # periodic model + training state save
│   └── _reset_step_accumulators() -> None       # zero running loss/tokens/seq_len
├── save_checkpoint() -> None      # explicit checkpoint (model + training state)
└── cleanup() -> None              # any teardown (currently a no-op, future: DDP)
```

#### Constructor

```python
def __init__(self, cfg: DictConfig) -> None:
    self.cfg = cfg
    # All other attributes initialized to None / sentinels.
    # Actual construction deferred to setup().
```

Rationale: separating `__init__` from `setup()` allows tests to construct a `Trainer` with a mock config without triggering heavy model/data loading.

#### `setup()` Implementation Sketch

The setup method does everything the current `train()` does before the `# === Training loop ===` comment (lines 156–267), organized into the private `_setup_*` methods listed above.

Key detail — `_resume()`:
1. Check if `checkpointer.training_state_checkpoint is not None`
2. Call `resume_training_state(ckpt_dict)` (existing function — keep as-is)
3. Set `self.global_step`, `self.consumed_samples` from resume state
4. Load optimizer state dict
5. Load lr_scheduler state dict
6. Restore cumulative metrics (`tokens_train_total`, `token_type_counts_total`, `wall_clock_offset`)
7. Call `validate_resume_hparams()`
8. Call `restore_rng_states()` **last** (after all setup, before loop)

#### `train()` Implementation Sketch

```python
def train(self) -> None:
    self.optimizer.zero_grad()
    self.t_train_start = time.perf_counter()
    self.t_step_start = time.perf_counter()
    self._reset_step_accumulators()

    epochs_run = self.global_step // self.geometry.steps_per_epoch
    batches_to_skip = (self.global_step % self.geometry.steps_per_epoch) * self.cfg.gradient_accumulation_steps

    LOGGER.info(OmegaConf.to_yaml(self.cfg, resolve=True, sort_keys=False))
    self.wandb_logger.log_config(self.cfg)

    for epoch in range(epochs_run, self.geometry.n_epochs):
        self._train_epoch(epoch, batches_to_skip if epoch == epochs_run else 0)
        if self.global_step >= self.cfg.max_steps:
            LOGGER.info("Training completed.")
            return
```

#### `_train_step(batch)` — Single Micro-Batch

```python
def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
    batch_to_device(batch, self.device)
    for tt, ttcnt in count_token_types(batch["tokens"], self.token_type_ranges, self.tokenizer.pad_id).items():
        self.token_type_counts_total[tt] += ttcnt
    self.max_seq_len_step = max(self.max_seq_len_step, batch["tokens"].size(1))
    num_tokens_iter = int((batch["labels"] != self.loss_fn.ignore_index).sum().item())
    self.num_tokens_step += num_tokens_iter
    loss_batch = compute_loss(batch, self.model, self.loss_fn) * num_tokens_iter
    self.loss_running += loss_batch
    loss_batch.backward()
    del batch
    torch.cuda.empty_cache()
    return loss_batch.item()
```

#### `_optimizer_step(epoch)` — Gradient Accumulation Boundary

```python
def _optimizer_step(self, epoch: int) -> None:
    scale_grads(self.model, 1 / self.num_tokens_step)
    if self.cfg.clip_grad_norm is not None:
        self._grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float(self.cfg.clip_grad_norm)
        )
    self.optimizer.step()
    self.optimizer.zero_grad(set_to_none=True)
    if self.lr_scheduler is not None:
        self.lr_scheduler.step()
    self.global_step += 1
    self.consumed_samples += (
        self.cfg.gradient_accumulation_steps * self.geometry.batch_size * self.world_size
    )
    self.tokens_train_total += self.num_tokens_step
```

### What Stays in `ssi/train.py`

The following **functions remain unchanged** in `ssi/train.py` — they are pure functions with no state, and the Trainer calls them:

- `validate_train_cfg(cfg)` — config validation
- `resume_training_state(ckpt_dict)` — checkpoint dict parsing
- `validate_resume_hparams(...)` — hparam mismatch checking
- `get_token_type_ranges(llama_config)` — vocab range computation
- `count_token_types(tokens, ranges, pad_idx)` — token counting
- `train(cfg)` — **kept as-is** for backwards compatibility and equivalence testing

The functional `train()` is **not deleted**. It remains as the reference implementation during the verification phase. Once equivalence is proven, the entry point scripts (`train_cpt.py`, `train_sft.py`) will be updated to use `Trainer`.

### Entry Point Changes (Deferred to Post-Verification)

```python
# scripts/train_sft.py (after verification)
@hydra.main(config_path="../conf", config_name="sft", version_base=None)
def main(cfg):
    trainer = Trainer(cfg)
    trainer.setup()
    trainer.train()
    trainer.cleanup()
```

---

## Implementation Plan — Step by Step

### Phase 1: Scaffolding (no behavior change)

**Step 1.1** — Create `ssi/trainer.py` with the `Trainer` class skeleton: `__init__`, `setup`, `train`, `save_checkpoint`, `cleanup` as stubs that raise `NotImplementedError`.

**Step 1.2** — Create `TrainingGeometry` dataclass with `from_config()` factory.

**Step 1.3** — Add basic test `tests/test_trainer_unit.py` that verifies `Trainer` can be constructed with a DictConfig without error, and that `TrainingGeometry.from_config()` produces correct values.

### Phase 2: Setup Methods

**Step 2.1** — Implement `_setup_model()`: checkpointer creation, checkpoint loading, llama config update, model construction via `setup_llama3_2_1b`, move to device, set to train mode.

**Step 2.2** — Implement `_setup_tokenizer()`: call `setup_llama3_tokenizer`, compute `token_type_ranges`.

**Step 2.3** — Implement `_setup_data()`: dispatch to `setup_sft_data` or `setup_text_completion_data` based on `config_name`, store train/dev loaders and samplers.

**Step 2.4** — Implement `_setup_optimizer()`: call `setup_optimizer`, `setup_lr_scheduler`.

**Step 2.5** — Implement `_setup_loss()`: create `CEWithChunkedOutputLoss`, handle `compile` and `set_num_output_chunks`.

**Step 2.6** — Implement `_setup_logging()`: W&B logger init with tags, resolve `checkpointer.output_dir` if None.

**Step 2.7** — Implement `_resume()`: restore global_step, consumed_samples, optimizer state, lr_scheduler state, cumulative metrics, RNG states.

**Step 2.8** — Wire `setup()` to call all `_setup_*` methods in the correct order.

**Step 2.9** — Implement `TrainingGeometry` computation after data setup (depends on dataloader length).

### Phase 3: Training Loop

**Step 3.1** — Implement `_reset_step_accumulators()`.

**Step 3.2** — Implement `_train_step(batch)` — single micro-batch forward + backward.

**Step 3.3** — Implement `_optimizer_step(epoch)` — gradient scaling, clipping, optimizer step, LR step, counter increments.

**Step 3.4** — Implement `_evaluate()` — delegates to existing `compute_dataset_loss`.

**Step 3.5** — Implement `_log_metrics(epoch, iter_idx)` — console and W&B logging.

**Step 3.6** — Implement `_maybe_save_checkpoint()` — periodic model + training state save.

**Step 3.7** — Implement `_train_epoch(epoch, batches_to_skip)` — inner loop with gradient accumulation, calling the methods above.

**Step 3.8** — Implement `train()` — outer epoch loop.

**Step 3.9** — Implement `save_checkpoint()` (explicit, non-periodic).

**Step 3.10** — Implement `cleanup()`.

### Phase 4: GPU Equivalence Tests

See "Test Plan" section below.

### Phase 5: Cutover

**Step 5.1** — Update `scripts/train_cpt.py` and `scripts/train_sft.py` to use `Trainer`.

**Step 5.2** — Add `__all__` export of `Trainer` from `ssi/trainer.py`.

**Step 5.3** — Mark the functional `train()` in `ssi/train.py` with a deprecation comment pointing to `Trainer`.

---

## Test Plan

### Test Data Strategy

Tests use Multilingual LibriSpeech data via HuggingFace, as referenced in the existing configs. For speed:

- **Subset size**: First **2,000 samples** from the MLS train split and first **200 samples** from the MLS validation split.
- **Mechanism**: Use a `filter_fn` in the data config or a `datasets.Dataset.select(range(N))` call.
- **Max steps**: 10–20 steps (enough to exercise gradient accumulation, eval, and checkpoint save).
- **Batch size**: 2 (matching existing SFT config).
- **Gradient accumulation**: 2 (so each optimizer step = 2 micro-batches = 4 samples).
- **Eval steps**: 5 (so eval fires twice in a 10-step run).
- **Save steps**: 10 (so checkpoint fires once).

### Existing Tests (Run First)

| Test File | Tests | GPU Required |
|---|---|---|
| `tests/test_checkpoint.py` | T1–T16: checkpoint schema, resume logic, RNG round-trip, disk tests | No (CPU) except T12–T15b (need Llama 3.2 1B base weights on disk) |
| `tests/test_cpt_deterministic_rng.py` | T16b–T16c: per-sample deterministic RNG | No (CPU) |

### New Tests: `tests/test_trainer.py`

#### Unit Tests (CPU, no model weights)

**T-U1: Construction** — `Trainer(cfg)` stores cfg, all other attributes are None.

**T-U2: TrainingGeometry** — `TrainingGeometry.from_config()` computes correct values for known inputs (batches_per_epoch=100, grad_accum=4 → steps_per_epoch=25, etc.).

**T-U3: TrainingGeometry remainder warning** — when batches_per_epoch % grad_accum != 0, a warning is logged.

**T-U4: Validate train cfg** — `validate_train_cfg` rejects invalid configs (negative steps, save_steps not multiple of eval_steps, etc.). (These already exist implicitly but should be explicit.)

#### Integration Tests (GPU, Llama 3.2 1B extended model, MLS subset)

All integration tests use the extended model at `models/extended/Llama-3.2-1B-5000-dsus/` and the `sft/mls-hubert_large_ll60k-layer_22` data config with the 2000-sample subset.

**T-I1: Setup smoke test** — `Trainer.setup()` completes without error. Verify all attributes are non-None. Verify `self.geometry` has sane values.

**T-I2: Single train step** — After `setup()`, call `_train_step(batch)` on one batch. Verify loss is a finite float. Verify `num_tokens_step > 0`. Verify `token_type_counts_total` is populated.

**T-I3: Optimizer step** — Run `gradient_accumulation_steps` micro-batches, then `_optimizer_step()`. Verify `global_step` incremented by 1. Verify optimizer state changed (compare param checksums before/after).

**T-I4: Evaluation** — Call `_evaluate()`. Verify returns a finite float.

**T-I5: Full train run (10 steps)** — `Trainer.train()` for 10 steps. Collect loss sequence.

**T-I6: Functional equivalence** — Run the **functional** `train(cfg)` for 10 steps with the same config, seed, and data. Collect loss sequence. **Assert the two loss sequences are identical** (or within float tolerance `atol=1e-6`).

**T-I7: Checkpoint save + resume equivalence** — Run `Trainer.train()` for 5 steps, save checkpoint. Create new `Trainer`, resume from checkpoint, run 5 more steps. Compare loss sequence at steps 6–10 against an uninterrupted 10-step run.

**T-I8: CPT equivalence** — Same as T-I6 but using `cpt` config and `cpt/mls-hubert_large_ll60k-layer_22` data.

### Test Fixtures and Helpers

```python
# conftest.py additions

@pytest.fixture
def sft_cfg_subset():
    """Hydra-composed SFT config with 2000-sample subset and small step counts."""
    # Uses hydra.compose() to build config programmatically
    # Overrides: max_steps=10, eval_steps=5, save_steps=10,
    #            gradient_accumulation_steps=2, data.train.dataloader.batch_size=2
    ...

@pytest.fixture
def cpt_cfg_subset():
    """Same as above but for CPT."""
    ...

def collect_loss_sequence(train_fn_or_trainer, cfg, max_steps) -> list[float]:
    """Utility to capture per-step loss for comparison."""
    ...
```

### Test Markers

```python
# pytest markers
@pytest.mark.gpu        # requires CUDA
@pytest.mark.disk       # requires model weights on disk
@pytest.mark.slow       # takes > 30 seconds
@pytest.mark.equiv      # equivalence test between functional and stateful
```

### Running Tests

```bash
# All CPU tests (fast)
uv run pytest tests/ -v -m "not gpu and not disk"

# Disk tests (need Llama 3.2 1B base weights)
uv run pytest tests/ -v -m "disk and not gpu"

# GPU integration tests (need GPU + extended model + MLS data)
uv run pytest tests/test_trainer.py -v -m "gpu"

# Equivalence tests only
uv run pytest tests/test_trainer.py -v -m "equiv"

# Everything
uv run pytest tests/ -v
```

---

## Data Subsetting Strategy

For GPU tests, we need a fast-loading subset of MLS. Two approaches (use approach A):

### Approach A: Runtime filtering via `datasets.Dataset.select()`

Add a `_subset_size` private config key to the test data configs:

```python
# In test fixture
with open_dict(cfg):
    cfg.data.train.dataset._subset_size = 2000
    cfg.data.dev.dataset._subset_size = 200
```

Then in both `SFTDataset.__init__` and `TextCompletionDataset.__init__`, after loading:

```python
if hasattr(cfg, '_subset_size') and cfg._subset_size is not None:
    self._data = self._data.select(range(min(cfg._subset_size, len(self._data))))
```

**Alternatively** (simpler, no dataset class changes): use the existing `filter_fn` parameter with a closure:

```python
# In test fixture, override filter_fn to take first N samples
cfg.data.train.dataset.filter_fn = None  # clear any existing
# Then after dataset loads, manually: dataset._data = dataset._data.select(range(2000))
```

The cleanest approach: pass a HuggingFace `split` string with range syntax, which is natively supported:

```python
# In test fixture override
cfg.data.train.dataset.split = "train[:2000]"
cfg.data.dev.dataset.split = "validation[:200]"
```

This requires **no code changes** — HuggingFace `load_dataset` natively supports slice notation in split strings.

### Approach B: Pre-cached subset on disk

Not needed given Approach A's simplicity.

---

## Verification Criteria for Equivalence (T-I6)

Two training runs are equivalent if and only if:

1. **Loss sequence**: `abs(loss_functional[i] - loss_stateful[i]) < 1e-6` for all steps i
2. **Global step**: both reach the same `global_step` value
3. **Model weights**: `torch.allclose(param_f, param_s, atol=1e-6)` for all parameters after N steps
4. **Optimizer state**: momentum/velocity buffers are identical
5. **Consumed samples**: identical count

The tolerance `1e-6` accounts for bf16 rounding; if both run in fp32 the tolerance can be tightened to `1e-7`.

To capture per-step losses from the functional `train()` without modifying it, we can:
- Monkey-patch `LOGGER.info` to intercept the loss log lines, OR
- Add a lightweight loss-capture hook that both implementations use, OR
- (Simplest) Add an optional `loss_log: list[float] | None = None` parameter to both `train()` and `Trainer.train()` that appends per-step loss when provided.

We use the third approach: add an optional `_loss_log` parameter.

---

## File Inventory

| File | Action |
|---|---|
| `ssi/trainer.py` | **CREATE** — new Trainer class |
| `ssi/train.py` | **KEEP** — no changes (reference implementation) |
| `ssi/eval.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/loss.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/model.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/optimizer.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/lr_schedule.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/checkpoint.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/metric_logging.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/data/__init__.py` | **KEEP** — no changes (called by Trainer) |
| `ssi/data/cpt.py` | **KEEP** — no changes |
| `ssi/data/sft.py` | **KEEP** — no changes |
| `ssi/constants.py` | **KEEP** — no changes |
| `tests/test_trainer.py` | **CREATE** — new test suite |
| `tests/conftest.py` | **CREATE** — shared fixtures |
| `scripts/train_cpt.py` | **MODIFY** (Phase 5 only) — switch to Trainer |
| `scripts/train_sft.py` | **MODIFY** (Phase 5 only) — switch to Trainer |

---

## Risk Mitigation

1. **The functional `train()` is never modified.** It remains as the ground truth.
2. **Equivalence tests gate the cutover.** Scripts are not updated until T-I6 and T-I7 pass.
3. **No config schema changes.** The Trainer reads the same Hydra configs.
4. **No checkpoint schema changes.** The Trainer writes identical `training_state.pt` files.
5. **Pure functions stay pure.** `validate_train_cfg`, `resume_training_state`, `get_token_type_ranges`, `count_token_types`, `validate_resume_hparams` remain in `ssi/train.py` and are imported by `Trainer`.

---

## Implementation Order and Estimated Iteration Count

For an automated loop running every ~15 minutes, with GPU tests:

| Iteration | Steps | What Gets Done |
|---|---|---|
| 1 | 1.1–1.3 | Skeleton + TrainingGeometry + unit tests |
| 2 | 2.1–2.3 | Model, tokenizer, data setup |
| 3 | 2.4–2.9 | Optimizer, loss, logging, resume, full setup() |
| 4 | 3.1–3.4 | Step accumulators, train_step, optimizer_step, evaluate |
| 5 | 3.5–3.8 | Logging, checkpointing, epoch loop, outer train() |
| 6 | 3.9–3.10, T-I1–T-I4 | Save/cleanup, GPU smoke tests |
| 7 | T-I5–T-I6 | Full run + functional equivalence test |
| 8 | T-I7–T-I8 | Resume equivalence + CPT equivalence |
| 9 | 5.1–5.3 | Script cutover (if all tests pass) |
