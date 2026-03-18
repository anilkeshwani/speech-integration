# Refactor: Separate Model and Training State Checkpoints

## Motivation

`save_checkpoint` currently conflates two distinct operations with different purposes, lifecycles, and storage requirements:

1. **Model checkpoint** — HF-format safetensors saved per step in `step_N/`. Used for inference, generation, fine-tuning, and evaluation. Lives on disk indefinitely, one directory per checkpoint step. Consumers: `generate.py`, `plt_embed_tsne.py`, downstream evaluation scripts, HuggingFace ecosystem tools.

2. **Training state** — optimizer, LR scheduler, RNG states, training hparams, cumulative metrics, consumed_samples, saved as `recipe_state.pt`. Used exclusively for resuming training (crash recovery, multi-job training). Only the latest one matters — always overwritten. Consumers: `train.py` resume path only.

### Problems with the current design

**P1: The `save_training_state` flag is a code smell.** A single function that behaves fundamentally differently based on a boolean flag is trying to serve two masters. `extend_llama3_2.py` calls `save_checkpoint` with `save_training_state=False`, `optimizer_state_dict=None`, dummy `global_step=0`, dummy `seed=SEED` — it only wants to save model weights but has to navigate a training-oriented API.

**P2: Optional training state fields produce unresumable checkpoints.** The save side (`checkpoint.py:380-425`) declares permissive defaults on training state parameters:

- `lr_scheduler_state_dict: dict[str, Any] | None = None` (line 387)
- `training_hparams: dict[str, Any] | None = None` (line 388)
- `cumulative_metrics: dict[str, Any] | None = None` (line 390)

These `None`-valued parameters are conditionally inserted into the checkpoint dict (lines 408-415):

```python
if lr_scheduler_state_dict is not None:   # line 408 — key omitted when None
    ckpt_dict[LR_SCHEDULER_KEY] = ...
if training_hparams is not None:          # line 411 — key omitted when None
    ckpt_dict[TRAINING_HPARAMS_KEY] = ...
if cumulative_metrics is not None:        # line 414 — key omitted when None
    ckpt_dict[CUMULATIVE_METRICS_KEY] = ...
```

When any of these are left at the default `None`, the corresponding key is **silently omitted** from the saved `recipe_state.pt`.

The load side (`train.py:70-87`, `resume_training_state`) uses direct `[]` subscripts on every one of those keys:

```python
"lr_scheduler_state": ckpt_dict[LR_SCHEDULER_KEY],    # line 82
"training_hparams": ckpt_dict[TRAINING_HPARAMS_KEY],  # line 84
"cumulative_metrics": ckpt_dict[CUMULATIVE_METRICS_KEY],  # line 86
```

There are no `.get()` fallbacks — these are hard subscripts that raise `KeyError` if the key is absent.

**The bug**: A caller can invoke `save_checkpoint(...)` with `save_training_state=True` (the default, line 391) while leaving `training_hparams`, `lr_scheduler_state_dict`, or `cumulative_metrics` at their default `None`. The function writes a `recipe_state.pt` missing required keys. On resume, `resume_training_state` crashes with `KeyError`. This works today only because `train.py` happens to pass all fields — the API signature does not enforce it.

**P3: Everything flows through one dict.** `save_checkpoint` builds a single `ckpt_dict` containing model weights, optimizer state, RNG states, metrics, and metadata. This dict is then passed to `_save_checkpoint`, which passes the whole thing to `save_full_model` (which reads `MODEL_KEY`) and `save_recipe_state` (which filters out `MODEL_KEY`). The model weights and training state never need to be in the same dict — they go to different files in different locations.

**P4: Storage locations are tangled.** Model weights go to `step_N/` subdirectories; training state goes to `self.output_dir/recipe_state.pt`. But the logic for determining these paths is interleaved inside `save_checkpoint` and `_save_checkpoint`. The caller can't control them independently.

---

## Proposed Design

### Two public methods replace `save_checkpoint`

```python
class FullModelHFCheckpointer:

    def save_model_checkpoint(
        self,
        model_state_dict: dict[str, Any],
        global_step: int,
        *,
        output_dir: Path | None = None,
        ignore_suffixes: list[str] | None = None,
    ) -> Path:
        """Save model weights in HF safetensors format to step_N/ directory.

        Copies config.json, tokenizer files, etc. alongside the weights so the
        checkpoint directory is self-contained and usable by HF tooling.

        Returns the output directory path.
        """
        ...

    def save_training_state(
        self,
        *,
        optimizer_state_dict: dict[str, Any],
        lr_scheduler_state_dict: dict[str, Any] | None,
        global_step: int,
        seed: int,
        training_hparams: dict[str, Any],
        consumed_samples: int,
        cumulative_metrics: dict[str, Any],
    ) -> Path:
        """Save training resume state to recipe_state.pt at self.output_dir.

        Always overwrites the previous file. All fields except
        lr_scheduler_state_dict are mandatory — a checkpoint written by this
        method is guaranteed to be resumable.

        Returns the output file path.
        """
        ...
```

#### Key properties

- **`save_model_checkpoint`** takes only what it needs: model weights and a step number. No optimizer, no RNG, no metrics. No training-state parameters to accidentally leave as `None`.

- **`save_training_state`** has all required fields as mandatory keyword arguments (no defaults except `lr_scheduler_state_dict` which is legitimately `None` when no scheduler is configured). The schema v1 contract is enforced by the function signature — it is impossible to call this without providing all required fields.

- **`lr_scheduler_state_dict`** is the one field that can be `None`. It is always written to the checkpoint (as `None`), so `resume_training_state` always finds the key. The resume side already handles `None` correctly (`if resume_state and lr_scheduler is not None: lr_scheduler.load_state_dict(...)`).

- **Each method constructs only the dict it needs.** No shared mega-dict, no filtering of model keys from training state.

### Disk layout (unchanged)

```
{output_dir}/
  step_100/                  <- save_model_checkpoint
    model-00001-of-00001.safetensors
    model.safetensors.index.json
    config.json
    tokenizer.model
    ...
  step_200/
    ...
  recipe_state.pt            <- save_training_state (single file, always overwritten)
```

This is exactly the layout we already have. The refactor changes the code structure, not the disk structure.

### What happens to `save_checkpoint`

Deleted. The two new methods are called independently. There is no wrapper that calls both — the caller decides what to save.

### What happens to `_save_checkpoint`

Deleted. Its responsibilities are absorbed:
- Model save logic → `save_model_checkpoint`
- Adapter save logic → `save_model_checkpoint` (conditional, as today)
- File copy logic → `save_model_checkpoint`
- Recipe state logic → `save_training_state`

### What happens to `save_recipe_state`

Replaced by `save_training_state`. The current `save_recipe_state` takes an opaque dict and filters out model keys — the new method constructs the dict internally from explicit parameters, which is cleaner and self-documenting.

### What happens to `save_full_model`

Unchanged. It remains a private helper called by `save_model_checkpoint`.

---

## Caller Changes

### `ssi/train.py` — training loop checkpoint save

**Before:**
```python
checkpointer.save_checkpoint(
    model_state_dict=model.state_dict(),
    optimizer_state_dict=optimizer.state_dict(),
    global_step=global_step,
    seed=SEED,
    lr_scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
    training_hparams={...},
    consumed_samples=consumed_samples,
    cumulative_metrics={...},
)
```

**After:**
```python
checkpointer.save_model_checkpoint(model.state_dict(), global_step)
checkpointer.save_training_state(
    optimizer_state_dict=optimizer.state_dict(),
    lr_scheduler_state_dict=lr_scheduler.state_dict() if lr_scheduler else None,
    global_step=global_step,
    seed=SEED,
    training_hparams={...},
    consumed_samples=consumed_samples,
    cumulative_metrics={...},
)
```

Two calls, clear intent. The model checkpoint goes to `step_N/`, the training state goes to `recipe_state.pt`.

### `scripts/extend_llama3_2.py` — model export only

**Before:**
```python
checkpointer.save_checkpoint(
    model.state_dict(),
    optimizer_state_dict=None,
    global_step=0,
    seed=SEED,
    save_training_state=False,
    output_dir=args.output_dir,
    ignore_suffixes=ignore_suffixes,
)
```

**After:**
```python
checkpointer.save_model_checkpoint(
    model.state_dict(),
    global_step=0,
    output_dir=args.output_dir,
    ignore_suffixes=ignore_suffixes,
)
```

No dummy parameters, no `save_training_state=False` flag, no confusion.

### `ssi/train.py` — resume path (`load_checkpoint`)

**No change.** `load_checkpoint` already handles both model weights and recipe state in a single load (it merges `recipe_state.pt` into the returned dict when `recipe_checkpoint` is set). This is fine for now — the load side's concern separation is less urgent because it always needs both pieces when resuming. If we later want to load model-only (for evaluation), we just don't pass `recipe_checkpoint`.

---

## Test Changes

### `tests/test_checkpoint.py`

The disk round-trip tests (T12–T15) currently call `save_recipe_state` on a `FullModelHFCheckpointer` instance. These need to call `save_training_state` instead, with explicit arguments rather than a dict.

The T12 test (round-trip recipe state) becomes:
```python
checkpointer.save_training_state(
    optimizer_state_dict={"state": {}, "param_groups": []},
    lr_scheduler_state_dict={"last_epoch": 150, "_last_lr": [2e-4]},
    global_step=150,
    seed=SEED,
    training_hparams=dict(HPARAMS),
    consumed_samples=9600,
    cumulative_metrics={
        "tokens_train_total": 100_000,
        "token_type_counts": {"text": 80_000, "dsu": 15_000, "special_text": 5_000},
        "wall_clock_seconds": 3600.0,
    },
)
```

This is actually better — the test explicitly shows what a valid training state save looks like, rather than passing a kitchen-sink dict and hoping the filtering logic works.

### New test: calling `save_training_state` with a missing required field

Since the signature enforces mandatory kwargs, this is tested at the Python level (calling without a required argument raises `TypeError`). No explicit test needed — the language enforces the contract. This directly resolves the P2 problem.

---

## Implementation Steps

### Step 1: Add `save_model_checkpoint` and `save_training_state` methods

Add both new methods to `FullModelHFCheckpointer`. They can coexist with the old `save_checkpoint` temporarily.

`save_training_state` constructs the full v1 schema dict internally:
```python
def save_training_state(self, *, optimizer_state_dict, lr_scheduler_state_dict,
                        global_step, seed, training_hparams, consumed_samples,
                        cumulative_metrics) -> Path:
    state_dict = {
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION,
        GLOBAL_STEP_KEY: global_step,
        SEED_KEY: seed,
        training.OPT_KEY: optimizer_state_dict,
        LR_SCHEDULER_KEY: lr_scheduler_state_dict,
        RNG_KEY: save_rng_states(),
        TRAINING_HPARAMS_KEY: training_hparams,
        CONSUMED_SAMPLES_KEY: consumed_samples,
        CUMULATIVE_METRICS_KEY: cumulative_metrics,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ssi_version": __version__,
    }
    output_path = self.output_dir / "recipe_state.pt"
    torch.save(state_dict, output_path)
    return output_path
```

`save_model_checkpoint` extracts the model-saving logic from `_save_checkpoint`:
```python
def save_model_checkpoint(self, model_state_dict, global_step, *,
                          output_dir=None, ignore_suffixes=None) -> Path:
    if output_dir is None:
        output_dir = self.output_dir / f"step_{global_step}"
    if ignore_suffixes is None:
        ignore_suffixes = SUFFIXES_TO_NOT_COPY + ["torchtune_config.yaml"]
    state_dict = {training.MODEL_KEY: model_state_dict}
    self.save_full_model(state_dict, output_dir)
    if training.ADAPTER_KEY in state_dict:
        self.save_adapter_weights(state_dict, output_dir)
    if training.ADAPTER_CONFIG in state_dict:
        self.save_adapter_config(state_dict, output_dir)
    copy_files(self.checkpoint_dir, output_dir, ignore_suffixes=ignore_suffixes)
    return output_dir
```

### Step 2: Migrate callers

- `ssi/train.py`: replace `save_checkpoint(...)` with `save_model_checkpoint(...)` + `save_training_state(...)`
- `scripts/extend_llama3_2.py`: replace `save_checkpoint(...)` with `save_model_checkpoint(...)`

### Step 3: Delete old methods

Remove `save_checkpoint`, `_save_checkpoint`, and `save_recipe_state` from `FullModelHFCheckpointer`.

### Step 4: Update tests

Update disk round-trip tests (T12–T15) to call `save_training_state` with explicit arguments.

---

## Out of Scope

- **`load_checkpoint` refactor**: The load side merges model weights and recipe state into one dict. This works and the resume path handles it correctly. Splitting the load is a separate concern and not needed now.
- **Adapter checkpoint handling**: The adapter save/load paths are currently unused (marked with `# NOTE not used currently`). They are preserved as-is inside `save_model_checkpoint` for forward compatibility but not actively tested or changed.
- **Checkpoint retention policy**: Deciding how many `step_N/` directories to keep on disk is orthogonal to this refactor (F4 in Checkpointing - Consolidated Plan).
