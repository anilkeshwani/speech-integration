# Training Code Cleanup — Itemized Task List

Produced by exploratory review of all training-related code. Tasks are grouped by category and labelled for easy reference. Items marked ~~like this~~ are done.

Cross-referenced with: `plans/claude-train-critique.md`, `plans/Training Fixes and Refactor.md`, `plans/claude-data-critique.md`, `plans/Plan to Simplify Checkpoint Directory Structure.md`.

---

## Bugs (Incorrect Behavior)

**B1. Checkpoint resume broken — step key mismatch**
- `resume_training_state()` reads `ckpt_dict[STEPS_KEY]` (`"steps_run"`) — `ssi/train.py:60-63`
- `save_checkpoint()` writes `GLOBAL_STEP_KEY` (`"global_step"`) — `ssi/checkpoint.py:534-538`
- Result: resume fails with a missing key or restores wrong step.
- Fix: align save/load to use the same key (`"steps_run"`), add legacy fallback.

**B2. Epoch semantics on resume are wrong**
- Save writes the current loop index `epoch` (mid-epoch) — `ssi/checkpoint.py:536`
- Resume skips that epoch entirely (`range(epochs_run, n_epochs)`) — `ssi/train.py:168`
- Result: remaining batches in the interrupted epoch are dropped.
- Fix: save `epoch + 1` (completed epochs), or switch to step-only training with no epoch loop.

**B3. Gradient normalization uses wrong token count**
- `num_tokens_iter` is counted from unshifted labels — `ssi/train.py:175`
- But `compute_loss` shifts labels by 1 — `ssi/loss.py:16`
- Creates systematic mismatch in loss scaling and gradient renormalization.
- Fix: count valid tokens from the same shifted mask used by loss, or return `(loss, n_valid_tokens)` from `compute_loss`.

**B4. `steps_per_epoch = 0` crashes**
- `steps_per_epoch = batches_per_epoch // gradient_accumulation_steps` — `ssi/train.py:164`
- If `batches_per_epoch < grad_accum_steps`, `steps_per_epoch == 0` and `ceil(max_steps / 0)` crashes.
- Fix: add validation in `validate_train_cfg()` or guard with an early error.

**B5. Divide-by-zero in gradient scaling**
- `scale_grads(model, 1 / num_tokens_step)` — `ssi/train.py:182`
- If all labels in an accumulation window are masked, `num_tokens_step == 0` → ZeroDivisionError.
- Fix: skip optimizer step (with a warning) when `num_tokens_step == 0`.

**B6. Remainder batches silently dropped**
- When `batches_per_epoch % gradient_accumulation_steps != 0`, the trailing incomplete accumulation window is never stepped.
- Fix: flush partial accumulation at epoch end, or document the behavior explicitly.

**B7. `compute_loss` mutates input batch**
- `labels = batch.pop("labels")` — `ssi/loss.py:15`
- Fragile: any caller that needs labels after `compute_loss` will see an empty batch.
- Fix: use `batch["labels"]` (read-only access).

**B8. Dev loss has wrong type**
- `num_tokens_dev` starts as `int` but is incremented by a tensor — `ssi/eval.py:30-31`
- Return type annotation is `-> float` but may return a tensor.
- Fix: call `.item()` when accumulating; cast before dividing.

**B9. CPT custom key parameters silently ignored**
- `tokenized_key`, `alignment_start_time_key`, `alignment_end_time_key`, `speech_tokens_key` resolved in `cpt.py:85-92` but never stored or passed to `interleave()` / `concatenate_speech_text()`.
- Fix: thread keys through to prompt functions, or remove the parameters if intentionally unused.

---

## Dead / Redundant Code

~~**D1. Dead `ignore_idx` assignment in data setup**~~
- ~~`ssi/data/__init__.py:44-46, 81-83` — first `if loss_fn is None: ignore_idx = ...` immediately overwritten by the ternary below.~~
- ~~Fix: remove the redundant `if` block; keep only the ternary.~~

**D2. Unreachable second `packed` block**
- Both `setup_text_completion_data` and `setup_sft_data` raise `NotImplementedError` if `packed=True`, making the second `if packed:` block unreachable.
- Converted to feature spec: `plans/Feature - Packed Dataset Support.md`.
- Fix: implement packing properly per the feature spec; dead stub code will be replaced.

~~**D3. Unreachable code after `NotImplementedError` raises**~~
- ~~`optimizer_in_bwd`, `enable_activation_checkpointing`, `enable_activation_offloading` all raise immediately, with unreachable code beneath them.~~
- ~~Files: `ssi/model.py:33-42`, `ssi/lr_schedule.py:25-42`, `ssi/optimizer.py`~~
- ~~Fix: either remove the dead lines, or collapse the stubs to a single clear comment.~~
- Also removed dead `optimizer_in_bwd` / `optim_ckpt_wrapper` parameters from `lr_schedule.py` and `checkpoint.py`; flattened `else` skeleton in `optimizer.py`.

~~**D4. Self-assignment `optimizer = optimizer`**~~
- ~~No-op in `else` branch of `lr_schedule.py` and `optimizer.py`.~~
- ~~Fix: remove.~~

~~**D5. Unreachable activation checkpointing warning**~~
- ~~`ssi/model.py:41-42` warns if `enable_activation_checkpointing and not enable_activation_offloading`, but checkpointing raises `NotImplementedError` above — warning can never be reached.~~
- ~~Fix: remove (belongs after the feature is implemented).~~

---

## Naming / Typos

~~**N1. Typo: `optimzer_state_dict`**~~
- ~~`ssi/optimizer.py:11` — missing `i` in `optimizer`.~~
- ~~Fix: rename to `optimizer_state_dict` throughout.~~

~~**N2. Confusing `global_step = -1` initialization**~~
- ~~`ssi/train.py:138` — non-obvious `-1` sentinel for LR scheduler's `last_epoch`.~~
- ~~Fix: add a clear comment or compute `last_epoch = global_step - 1` explicitly.~~

---

## Code Clarity / Style

**C1. Hard-coded SLURM QoS check**
- `ssi/train.py:111` — `os.getenv("SLURM_JOB_QOS") == "gpu-debug"` ties behavior to cluster-specific env var.
- Fix: replace with `cfg.debug_mode` (already exists in `common.yaml`).

**C2. TODO: use `hydra.utils.instantiate` for LR scheduler**
- `ssi/train.py:146` — implement or remove the TODO.

**C3. Zero-padding in log format strings**
- `ssi/train.py:198,244`, `ssi/eval.py:36` — `# TODO bad zero padding` with dynamic field width.
- Fix: compute fixed width from `max_steps` once at training start.

**C4. Inconsistent exception types in `sft.py`**
- `ValueError` on line 149, `TypeError` on lines 160,170 — all for boolean type checks.
- Fix: standardize to `TypeError` (wrong type passed).

**C5. `setup_alpaca_data` is debug scaffolding**
- `ssi/data/__init__.py:180-199` — not called from any training script.
- Fix: move to a test/example file or remove.

**C6. Module-level PRNG shared between train and dev datasets**
- `cpt.py:41` — `PRNG = np.random.default_rng(SEED)` is module-level; dev reproducibility depends on training history.
- Fix: create a per-instance RNG, or separate RNGs for train and dev.

**C7. Unconditional `torch.cuda.empty_cache()` every batch**
- `ssi/train.py:249` — harms throughput on CUDA; wrong on other devices.
- Fix: gate behind `cfg.debug_mode` or remove (the `del batch` is sufficient).

**C8. Token-type accounting semantics unclear**
- Counters appear cumulative but logged like per-step values — `train.py:99-101`.
- Padding tokens in `special_text` range inflate that counter while `"total"` excludes padding.
- Fix: clarify per-step vs cumulative; exclude padding consistently from all type buckets.

---

## Config Validation

**V1. No positive-integer validation for critical step fields**
- `gradient_accumulation_steps`, `max_steps`, `log_interval`, `eval_steps`, `save_steps` never validated > 0 in `validate_train_cfg()`.
- Zero values cause modulo/division crashes deep in the training loop.
- Fix: add checks to `validate_train_cfg()`.

**V2. `save_steps` not validated as multiple of `eval_steps`**
- Comment in `conf/training.yaml` says it "should be" — not enforced.
- Fix: add assertion in `validate_train_cfg()`.

---

## Structural Refactoring

**R1. Checkpoint directory structure simplification**
- Full plan in `plans/Plan to Simplify Checkpoint Directory Structure.md`.
- Current: `experiments/.../checkpoints/epoch_N/global_step_M/`
- Target: `experiments/{stage}_{encoder}_{n_dsus}dsus_{flags}_{wandb_id}/step_M/`
- Touches: `ssi/checkpoint.py`, `ssi/utils.py`, `scripts/generate.py`, `snippets/`, `conf/training.yaml`.

~~**R2. LR scheduler `last_epoch` initialization**~~
- ~~Clean up `global_step = -1 if global_step == 0 else global_step` pattern in `train.py:138-146`.~~
- ~~Compute `last_epoch = global_step - 1` explicitly with a comment.~~
