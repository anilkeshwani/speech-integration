# B2: Intra-Epoch Step-Level Resume with Configuration Validation

## Context

MLS is 44k hours. The training budget is 48 GPU-hours on a single A6000. One epoch of MLS will never complete, so the training run always ends mid-epoch. The simple `epoch + 1` fix would restart from a fresh shuffle of MLS on every resume, discarding all intra-epoch progress.

`global_step` is the canonical resume state. The epoch number and batch offset within the epoch are derived from it. This requires that `steps_per_epoch` (and therefore `batch_size`, `gradient_accumulation_steps`, and `world_size`) be the same on resume as on save — otherwise the derived skip count is wrong. We therefore save these hparams in the checkpoint and validate them on resume.

**HuggingFace Trainer precedent (and its gap):** HF saves `train_batch_size` and emits a warning-only on mismatch. It does not save `gradient_accumulation_steps` or `world_size` — a known bug: changing either silently corrupts the batch-skip arithmetic. We store all three and raise an error on mismatch (downgrade to warning with an explicit override flag).

---

## Design

### Core formula

```
steps_per_epoch      = batches_per_epoch // gradient_accumulation_steps
epochs_run           = global_step // steps_per_epoch
batches_to_skip      = (global_step % steps_per_epoch) * gradient_accumulation_steps
```

`batches_to_skip` is the number of raw micro-batches already consumed in the current epoch.
Skipping is done with `itertools.islice(enumerate(data_train), batches_to_skip, None)`.

`num_workers=0` throughout, so islice skipping is cheap — no prefetch workers, each discarded batch is a synchronous CPU load.

### What is saved in the checkpoint (new `TRAINING_HPARAMS_KEY` entry)

```python
{
    "batch_size":                  data_train.batch_size,
    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
    "world_size":                  world_size,   # int(os.environ.get("WORLD_SIZE", 1))
    "steps_per_epoch":             steps_per_epoch,   # cross-check convenience field
}
```

Stored under a new key `TRAINING_HPARAMS_KEY = "training_hparams"` in the checkpoint dict.

### Validation on resume

After data setup (so current `batch_size`, `steps_per_epoch` are known):

```python
validate_resume_hparams(ckpt_dict, current_batch_size, cfg.gradient_accumulation_steps, world_size, steps_per_epoch)
```

- Compares saved vs current `batch_size`, `gradient_accumulation_steps`, `world_size`, `steps_per_epoch`.
- If any differ: raises `ValueError` listing all mismatches.
- If checkpoint predates this feature (no `TRAINING_HPARAMS_KEY`): emits a `LOGGER.warning` and proceeds.

---

## Files to modify

| File | Change |
|---|---|
| `ssi/constants.py` | Add `TRAINING_HPARAMS_KEY = "training_hparams"` |
| `ssi/checkpoint.py` | Add `training_hparams` parameter to `save_checkpoint`; include in `ckpt_dict` |
| `ssi/train.py` | Update `resume_training_state`; add `validate_resume_hparams`; compute skip; modify epoch/batch loop |
| `plans/Training Cleanup Tasks.md` | Mark B2 done |

---

## Implementation

### `ssi/constants.py`

Add after `GLOBAL_STEP_KEY`:
```python
TRAINING_HPARAMS_KEY: str = "training_hparams"
```

---

### `ssi/checkpoint.py`

Add `TRAINING_HPARAMS_KEY` to the import from `ssi.constants`.

Extend `save_checkpoint` signature and body:

```python
def save_checkpoint(
    self,
    model_state_dict: dict[str, Any],
    optimizer_state_dict: dict[str, Any] | None,
    epoch: int,
    global_step: int,
    seed: int,
    training_hparams: dict[str, Any] | None = None,   # NEW
    save_training_state: bool = True,
    adapter_only: bool = False,
    output_dir: Path | None = None,
    ignore_suffixes: list[str] | None = None,
) -> tuple[dict[str, Any], Path]:
    ...
    ckpt_dict: dict = {
        MODEL_KEY: model_state_dict,
        EPOCHS_KEY: epoch,
        GLOBAL_STEP_KEY: global_step,
        SEED_KEY: seed,
    }
    if training_hparams is not None:
        ckpt_dict[TRAINING_HPARAMS_KEY] = training_hparams
    ...
```

---

### `ssi/train.py`

**New imports:**
```python
import itertools
```

Add to the constants import block:
```python
TRAINING_HPARAMS_KEY,
```

Remove `EPOCHS_KEY` from the import block (no longer used in train.py).

**Update `resume_training_state` — drop `EPOCHS_KEY`, return only step + optimizer:**

```python
def resume_training_state(ckpt_dict: dict[str, Any]) -> tuple[int, StateDict]:
    if SEED != ckpt_dict[SEED_KEY]:
        raise ValueError("Config value for seed does not match the checkpoint value")
    return ckpt_dict[GLOBAL_STEP_KEY], ckpt_dict[OPTIMIZER_KEY]
```

**New `validate_resume_hparams` function (place near `resume_training_state`):**

```python
def validate_resume_hparams(
    ckpt_dict: dict[str, Any],
    batch_size: int,
    gradient_accumulation_steps: int,
    world_size: int,
    steps_per_epoch: int,
) -> None:
    if TRAINING_HPARAMS_KEY not in ckpt_dict:
        LOGGER.warning(
            "Checkpoint predates training hparam tracking; cannot validate batch_size / "
            "gradient_accumulation_steps / world_size match. Proceeding — verify manually."
        )
        return
    saved = ckpt_dict[TRAINING_HPARAMS_KEY]
    mismatches = []
    for key, current in (
        ("batch_size", batch_size),
        ("gradient_accumulation_steps", gradient_accumulation_steps),
        ("world_size", world_size),
        ("steps_per_epoch", steps_per_epoch),
    ):
        if saved.get(key) != current:
            mismatches.append(f"  {key}: checkpoint={saved.get(key)!r}, current={current!r}")
    if mismatches:
        raise ValueError(
            "Cannot resume: training configuration differs from checkpoint:\n"
            + "\n".join(mismatches)
            + "\nTo override (not recommended), adjust your config to match the checkpoint values."
        )
```

**Update the call site (replace the 3-tuple unpack):**

```python
global_step, optimizer_state = 0, None
if checkpointer.recipe_checkpoint is not None:
    global_step, optimizer_state = resume_training_state(ckpt_dict)
```

**After data setup and `steps_per_epoch` is computed — add hparam prep and validation:**

```python
batches_per_epoch = len(data_train)
steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
n_epochs = math.ceil(cfg.max_steps / steps_per_epoch)
world_size = int(os.environ.get("WORLD_SIZE", 1))
training_hparams = {
    "batch_size": data_train.batch_size,
    "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
    "world_size": world_size,
    "steps_per_epoch": steps_per_epoch,
}
if checkpointer.recipe_checkpoint is not None:
    validate_resume_hparams(ckpt_dict, data_train.batch_size, cfg.gradient_accumulation_steps, world_size, steps_per_epoch)
epochs_run = global_step // steps_per_epoch
batches_to_skip = (global_step % steps_per_epoch) * cfg.gradient_accumulation_steps
if batches_to_skip > 0:
    LOGGER.info(
        f"Resuming from global_step={global_step}: epoch {epochs_run}, "
        f"skipping {batches_to_skip} already-processed batches."
    )
```

**Update `save_checkpoint` call site to pass `training_hparams`:**

```python
if global_step != 0 and global_step % cfg.save_steps == 0:
    checkpointer.save_checkpoint(
        model_state_dict=model.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        epoch=epoch,
        global_step=global_step,
        seed=SEED,
        training_hparams=training_hparams,
    )
```

**Modify the epoch/batch loop:**

```python
for epoch in range(epochs_run, n_epochs):
    sampler_train.set_epoch(epoch)
    if epoch == epochs_run and batches_to_skip > 0:
        data_iter = itertools.islice(enumerate(data_train), batches_to_skip, None)
        n_batches = batches_per_epoch - batches_to_skip
    else:
        data_iter = enumerate(data_train)
        n_batches = batches_per_epoch
    for i, batch in tqdm(data_iter, total=n_batches):
        ...  # inner loop body unchanged; i starts at batches_to_skip for the resume epoch
```

The `i` values from `enumerate` start at `batches_to_skip`. Since `batches_to_skip = K × gradient_accumulation_steps`, the condition `(i + 1) % cfg.gradient_accumulation_steps == 0` fires correctly for the first new optimizer step.

---

## Correctness cases

| Scenario | `global_step` | `epochs_run` | `batches_to_skip` | Result |
|---|---|---|---|---|
| Fresh start | 0 | 0 | 0 | Epoch 0, batch 0 ✓ |
| Mid-epoch resume | N, N % S ≠ 0 | N // S | (N % S) × G | Correct batch offset ✓ |
| Exact epoch boundary | N, N % S == 0 | N // S | 0 | Next epoch, batch 0 ✓ |
| Config mismatch | any | — | — | `ValueError` raised ✓ |

(S = steps_per_epoch, G = gradient_accumulation_steps)

---

## Verification

1. Fresh run to step 10 (grad_accum=2, ~50 batches/epoch): confirm `training_hparams` appears in `recipe_state.pt` with correct values.
2. Resume from that checkpoint: confirm `batches_to_skip=20`, `epochs_run=0`, and that `global_step` increments from 10 to 11 on the first optimizer step after resume.
3. Attempt resume with a different `gradient_accumulation_steps` in config: confirm `ValueError` is raised listing the mismatch.
4. Resume from a legacy checkpoint (no `TRAINING_HPARAMS_KEY`): confirm `LOGGER.warning` fires and training proceeds.

# Notes and Critique on Current Plan

Here are the weaknesses and omissions I found, ordered by severity:

## Findings

### 1. Critical: Module-level PRNG in `cpt.py` is NOT advanced during `islice` skip

`cpt.py:41` defines a module-level stateful PRNG:
```python
PRNG = np.random.default_rng(SEED)
```

This PRNG advances on every `__getitem__` call for the `INTERLEAVED` sequence type — specifically in `interleave()` (calls `PRNG.choice` and `get_span_idxs_binomial` which calls `PRNG.binomial`). When `islice` skips `batches_to_skip` micro-batches, it does so **without calling `__getitem__`**, so the PRNG state stays at position 0.

Consequence: after the skip, every sample in the resumed epoch gets a different interleaving structure (span boundaries, `start_with_text` choice) than it would have had in an uninterrupted run. The dataset indices fed to the model are the same (the DistributedSampler is deterministic), but their content differs.

The plan makes no mention of this. The fix is to save the PRNG state in the checkpoint alongside `training_hparams` and restore it before the epoch loop begins.

---

### 2. Important: `islice` skip time is not quantified and could be large

The plan says `num_workers=0` makes islice "cheap." But each discarded batch still requires a full CPU-side `__getitem__` call per sample in the batch: HuggingFace `datasets` lookup, string processing, tokenization. For MLS at any reasonable batch size, worst-case skip (checkpointing near the end of an epoch) could mean loading tens of thousands of batches and discarding them. The plan should either:
- Estimate this overhead and flag it as a known cost, or
- Mention saving the DataLoader/sampler state as a future optimization to avoid it

This is especially relevant for production training runs where a resume from late in the epoch could block for hours before any gradient steps.

---

### 3. Moderate: `tokens_train_total` and `token_type_counts_total` are not checkpointed

Both are initialized to `0` before the training loop (`train.py:166-167`) and are not restored from the checkpoint. After a resume:
- `tokens_total` in W&B resets to 0, making it useless for tracking cumulative tokens across a resume boundary
- `n_tokens.{tt}` counts similarly reset

The plan doesn't mention these, either to restore them from checkpoint or to document the W&B monitoring discontinuity they create.

---

### 4. Moderate: `EPOCHS_KEY` is orphaned — written to every checkpoint but never read after refactor

The plan removes `EPOCHS_KEY` from `resume_training_state` (correct, since `epochs_run` is now derived from `global_step`). But `save_checkpoint` still writes `ckpt_dict[EPOCHS_KEY] = epoch` on every save. This value is no longer trusted on resume — the plan derives `epochs_run = global_step // steps_per_epoch` instead.

If someone inspects a checkpoint and sees `epochs_run: 3` but `global_step // steps_per_epoch = 5`, they'll be confused. More importantly, the plan doesn't state this key is now dead — it should either be removed from the checkpoint or clearly documented.

---

### 5. Moderate: No explicit guard for `batches_to_skip >= batches_per_epoch`

If `batches_to_skip >= batches_per_epoch` (which can't happen if `steps_per_epoch` validation passes, but isn't guarded by an explicit assertion), `islice` yields an empty iterator and the entire epoch is silently skipped with no batches processed. The plan relies on `steps_per_epoch` validation to prevent this transitively, but a direct assertion would be clearer and safer:
```python
assert batches_to_skip < batches_per_epoch, ...
```

---

### 6. Minor: `validate_resume_hparams` is called after data setup — wasted setup on mismatch

The plan places `validate_resume_hparams` after the `setup_{sft,text_completion}_data` calls. If there's a configuration mismatch (e.g., wrong `batch_size`), the dataset will have been loaded from disk before the error is raised. This is wasteful but not a correctness issue. The `world_size` and `gradient_accumulation_steps` components of the validation could be checked before data setup, with the `batch_size` / `steps_per_epoch` checks deferred.

---

### 7. Minor: Console log `Iteration {i}` will show skipped-past numbers in resume epoch

The existing log at `train.py:203`:
```python
f"Iteration {i:0{len(str(batches_per_epoch))}d} / {batches_per_epoch}",
```
On resume, `i` starts at `batches_to_skip` (not 0), so the first logged iteration is `batches_to_skip / batches_per_epoch` not `1 / batches_per_epoch`. This is technically correct (it reflects the actual batch index) but could be confusing to a reader who expects iterations to start at `~0` in log output after a resume. Worth documenting.

---

### Summary table

| # | Severity | Issue |
|---|---|---|
| 1 | Critical | `PRNG` in `cpt.py` not advanced during islice skip → wrong interleaving on resume |
| 2 | Important | Skip cost not quantified — could block for hours on late-epoch resume of MLS |
| 3 | Moderate | `tokens_train_total` / `token_type_counts_total` not checkpointed → W&B monitoring breaks |
| 4 | Moderate | `EPOCHS_KEY` written but no longer read — silent dead field in every checkpoint |
| 5 | Moderate | No assertion guarding `batches_to_skip < batches_per_epoch` |
| 6 | Minor | Validation happens after expensive data setup |
| 7 | Minor | Console log `Iteration {i}` is confusing on resume (starts mid-range) |
