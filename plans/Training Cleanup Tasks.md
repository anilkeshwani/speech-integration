# Training Code Cleanup — Itemized Task List

Produced by exploratory review of all training-related code. Tasks are grouped by category and labelled for easy reference. Items marked ~~like this~~ are done.

Cross-referenced with: `plans/claude-train-critique.md`, `plans/Training Fixes and Refactor.md`, `plans/claude-data-critique.md`, `plans/Plan to Simplify Checkpoint Directory Structure.md`.

**Checkpoint-related items (B1, B2, B6, Recommendation 1) are now consolidated in `plans/Checkpointing - Consolidated Plan.md`.** That document supersedes the checkpoint-specific content below and in `Fix B2 - Intra-Epoch Step-Level Resume with Configuration Validation.md`. The research backing the design is in `plans/Research - Checkpoint and Resume Best Practices.md`.

## Recommendations

1. ~~**Fix checkpoint schema first** → **See `Checkpointing - Consolidated Plan.md` for the full design.**~~
   - ~~Implemented as checkpoint schema v1. All fields aligned, legacy checkpoints rejected, RNG state saved/restored, LR scheduler state saved/restored, cumulative metrics preserved, hparam validation on resume.~~

2. **Fix token-count math**
   - Compute normalization counts on the same shifted label mask used by loss, or derive counts inside `compute_loss` and return both `(loss, n_valid_tokens)`.

3. **Harden config validation**
   - Enforce strict positivity for step/interval fields.
   - Validate `batches_per_epoch >= gradient_accumulation_steps` or support partial-window stepping explicitly.

4. **Guard against zero-token windows**
   - Skip optimizer step when `num_tokens_step == 0`, or fail fast with a descriptive error.

5. **Remove or gate `empty_cache`**
   - At minimum `if DEVICE.type == "cuda" and cfg.debug_empty_cache`.
   - Default off.

6. **Make metric semantics explicit**
   - Decide whether token-type counters are per-step or cumulative and ensure padding is excluded consistently.

7. **Fix eval numeric typing and distributed reduction**
   - Convert token counts to Python ints (`.item()`), and all-reduce `(dev_loss_sum, dev_token_count)` across ranks before division.

8. **Avoid mutating batches in loss**
   - Use `labels = batch["labels"]` and keep batch immutable in compute paths.

9. **Add regression tests** → **T1–T4**
   - Resume from saved checkpoint roundtrip.
   - Gradient normalization with shifted labels.
   - Small dataset where `len(dataloader) < grad_accum`.
   - All-ignored-label batch behavior.

---

## Bugs (Incorrect Behavior)

~~**B1. Checkpoint resume broken — step key mismatch**~~
- ~~`resume_training_state()` reads `ckpt_dict[STEPS_KEY]` (`"steps_run"`) — `ssi/train.py:60-63`~~
- ~~`save_checkpoint()` writes `GLOBAL_STEP_KEY` (`"global_step"`) — `ssi/checkpoint.py:534-538`~~
- ~~Result: resume fails with a missing key or restores wrong step.~~
- ~~Fix: align save/load to use the same key (`"steps_run"`), add legacy fallback for old checkpoints.~~
- For exact resume (bit-identical continuation): also save/load RNG state and intra-epoch dataloader position.

~~**B2. Epoch semantics on resume are wrong**~~
- ~~Fixed by checkpoint schema v1: `epoch` parameter removed from `save_checkpoint`, `EPOCHS_KEY` removed from checkpoints, epoch derived from `global_step // steps_per_epoch`, islice skip for mid-epoch resume. See `Checkpointing - Consolidated Plan.md`.~~

**B3. Gradient normalization uses wrong token count**
- `num_tokens_iter` counted from **unshifted** labels — `ssi/train.py:180`
- But `compute_loss` shifts labels left by 1, appending `ignore_index` at the end — `ssi/loss.py:16`
- This mismatch affects three sites:
  - Per-batch loss scaling: `compute_loss(...) * num_tokens_iter` — `ssi/train.py:183`
  - Gradient renormalization: `scale_grads(model, 1 / num_tokens_step)` — `ssi/train.py:187`
  - Logged loss: `loss_running.item() / num_tokens_step` — `ssi/train.py:195`
- Overcounts by up to `batch_size` valid tokens (one per sequence); bias is small per batch but systematic across training.
- Fix: count valid tokens from the same shifted label mask used by loss, or return `(loss, n_valid_tokens)` from `compute_loss`.

**B4. `steps_per_epoch = 0` crashes**
- `steps_per_epoch = batches_per_epoch // gradient_accumulation_steps` — `ssi/train.py:169`
- If `batches_per_epoch < grad_accum_steps`, `steps_per_epoch == 0` and `ceil(max_steps / 0)` crashes.
- V1 validation guards `gradient_accumulation_steps > 0` but cannot check `batches_per_epoch` at config-validation time.
- Fix: guard after `steps_per_epoch` is computed (raise early if zero) or enforce via dataset size check.

**B5. Divide-by-zero in gradient scaling**
- `scale_grads(model, 1 / num_tokens_step)` — `ssi/train.py:187`
- If all labels in an accumulation window are masked, `num_tokens_step == 0` → ZeroDivisionError.
- Fix: skip optimizer step (with a warning) when `num_tokens_step == 0`.

**B6. Remainder batches silently dropped**
- When `batches_per_epoch % gradient_accumulation_steps != 0`, the trailing incomplete accumulation window is never stepped.
- Fix: flush partial accumulation at epoch end, or document the behavior explicitly.

**B7. `compute_loss` mutates input batch**
- `labels = batch.pop("labels")` — `ssi/loss.py:15`
- Fragile: any caller that needs labels after `compute_loss` will see an empty batch.
- Fix: use `batch["labels"]` (read-only access).

**B8. Dev loss has wrong type and missing distributed aggregation**
- `num_tokens_dev_batch = (...).sum()` yields a tensor — `ssi/eval.py:30`; `num_tokens_dev += num_tokens_dev_batch` makes `num_tokens_dev` a tensor.
- `dev_loss_running / num_tokens_dev` returns a tensor, but `compute_dataset_loss` is annotated `-> float`.
- No cross-rank aggregation: in a distributed setting, each rank reports its own local shard's loss, not the global dev loss.
- Fix: call `.item()` on `num_tokens_dev_batch` at the accumulation site (`eval.py:30`); all-reduce `(dev_loss_sum, dev_token_count)` across ranks before the final division.

**B9. CPT custom key parameters silently ignored**
- `tokenized_key`, `alignment_start_time_key`, `alignment_end_time_key`, `speech_tokens_key` resolved in `cpt.py:85-92` but never stored or passed to `interleave()` / `concatenate_speech_text()`.
- Fix: thread keys through to prompt functions, or remove the parameters if intentionally unused.

**B10. `update_from_speech_cfg` mutates global singleton instead of `self`**
- `ssi/llama_configs.py:48-49` — `update_from_speech_cfg` is an instance method but hardcodes `configllama3_2_1b` instead of `self`.
- Calling it on any instance other than `configllama3_2_1b` silently mutates the wrong object.
- Currently harmless because `train.py:124` always calls it as `configllama3_2_1b.update_from_speech_cfg(...)`, but is a latent bug for tests or multi-config use.
- Fix: replace `configllama3_2_1b.n_dsus = ...` and `configllama3_2_1b.modality_tokens = ...` with `self.n_dsus = ...` and `self.modality_tokens = ...`.

**B11. `train_cfg` NameError in `generate.py` when `cfg.train_yaml` is not None**
- `scripts/generate.py` only assigns `train_cfg` inside the `if cfg.train_yaml is None:` block (lines 152–160).
- `train_cfg` is referenced unconditionally at lines 164, 170, 180 to resolve `cfg.speech.n_dsus`, `cfg.speech.deduplicate`, and `cfg.data`.
- If a caller provides `cfg.train_yaml` explicitly, this raises `NameError: name 'train_cfg' is not defined` at runtime.
- Fix: add the `else` branch that loads `train_cfg = OmegaConf.load(cfg.train_yaml)` when `cfg.train_yaml` is not None.

---

**D0. Dead import `from ast import Not` in `ssi/checkpoint.py:5`**
- Unused import, likely a copy-paste artifact.
- Fix: remove.

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

~~**C6. Module-level PRNG shared between train and dev datasets**~~
- ~~Fixed by per-sample deterministic RNG: module-level `PRNG` deleted, each sample's interleaving is a pure function of `(seed, epoch, sample_index)`. See `Checkpointing - Consolidated Plan.md` Step 1.~~

**C7. Unconditional `torch.cuda.empty_cache()` every batch**
- `ssi/train.py:254` — harms throughput on CUDA; a no-op or error on other devices.
- Fix: remove (the preceding `del batch` is sufficient), or gate behind a dedicated `cfg.debug_empty_cache` flag (default off) — do not reuse `cfg.debug_mode` for this.

**C8. Token-type accounting semantics unclear**
- `token_type_counts_total` is cumulative but logged inline with per-step metrics (loss, num_tokens_step) — `train.py:206`.
- Also logged to wandb as `n_tokens.{tt}` at `train.py:228` — cumulative, but field name and context are ambiguous.
- Padding tokens in `special_text` range inflate that counter while `"total"` excludes padding.

**C9. Mutable default arguments in data functions**
- `ssi/data/__init__.py:108`, `ssi/data/sft.py:121` — `additional_keys: list[str] = []`
- `ssi/data/cpt.py:172` — `dsu_spans: list[str] = []`
- Low practical risk here (lists are not mutated across calls), but violates Python best practice.
- Fix: replace with `= None` and assign `[]` inside the function body, or use `field(default_factory=list)` for dataclasses.
- Fix: clarify per-step vs cumulative; exclude padding consistently from all type buckets.

**C10. No schema validation for required dataset columns**
- `ssi/data/sft.py` and `ssi/data/cpt.py` do not validate that expected columns are present before attempting access.
- Failure mode is an unhelpful `KeyError` deep in the data pipeline rather than an early, descriptive error.
- Fix: add explicit checks for required columns at dataset construction time.

**C11. Training entrypoints are separate scripts with no explicit mode flag**
- `scripts/train.py` (or equivalent) and generation are separate scripts; no unified entrypoint with `--mode cpt|sft|generate`.
- Optional / aspirational: low priority relative to correctness fixes.

**C12. Hydra configs are not minimal or modular**
- Data configs (under `conf/data/`) repeat shared parameters across `train` and `dev` splits for the same dataset and training type (CPT vs SFT).
- More broadly, configs across training types and datasets have not been audited for redundancy or factored into composable base configs + overrides.
- Fix: extract shared parameters into a parent/base config; set only split-specific or dataset-specific parameters in child configs. Use Hydra's config group composition to assemble final configs from minimal, non-redundant parts.

---

## Config Validation

~~**V1. No positive-integer validation for critical step fields**~~
- ~~`gradient_accumulation_steps`, `max_steps`, `log_interval`, `eval_steps`, `save_steps` never validated > 0 in `validate_train_cfg()`.~~
- ~~Zero values cause modulo/division crashes deep in the training loop.~~
- ~~Fix: add checks to `validate_train_cfg()`.~~

~~**V2. `save_steps` not validated as multiple of `eval_steps`**~~
- ~~Comment in `conf/training.yaml` says it "should be" — not enforced.~~
- ~~Fix: add assertion in `validate_train_cfg()`.~~

---

## Structural Refactoring

**R0. Split `ssi/train.py` into logical units**
- Current `train()` function mixes config validation/setup, epoch/step execution, evaluation, logging, and checkpoint policy in one large function.
- Proposed decomposition: separate modules or functions for each concern; introduce an explicit `TrainState` dataclass (epoch, global_step, token counters, timers) to replace scattered local variables; add callback-like hooks so logging/checkpointing/eval triggers are testable in isolation.
- This is a significant structural refactor — do after correctness bugs (B-series) are fixed.

**R1. Checkpoint directory structure simplification**
- Full plan in `plans/Plan to Simplify Checkpoint Directory Structure.md`.
- Current: `experiments/.../checkpoints/epoch_N/global_step_M/`
- Target: `experiments/{stage}_{encoder}_{n_dsus}dsus_{flags}_{wandb_id}/step_M/`
- Touches: `ssi/checkpoint.py`, `ssi/utils.py`, `scripts/generate.py`, `snippets/`, `conf/training.yaml`.
- Notes:
    - We'll come back to this refactoring - specifically I think it makes more sense to structure the experiments into four broad directories according to which speech tokenizer is being used and then automatically set the number of DSUs used in that set of experiments accordingly. For this, we'll probably want to write out an extended version of the Llama 3.2 1B model where the tokenizer and embedding layer is extended by the correct number of types (tokens) in each case e.g. 5000 for HuBERT. 

~~**R2. LR scheduler `last_epoch` initialization**~~
- ~~Clean up `global_step = -1 if global_step == 0 else global_step` pattern in `train.py:138-146`.~~
- ~~Compute `last_epoch = global_step - 1` explicitly with a comment.~~

---

## CI and Documentation

**CI1. Add quality gates to CI pipeline**
- No CI currently enforces formatting, linting, or tests.
- Fix: add ruff (lint + format), and `pytest` to CI on every branch. Consolidate to ruff; drop any separate black/isort/flake8 config.

**CI2. Documentation**
- No architecture overview, config reference, or reproducibility checklist exists.
- Minimum viable docs: architecture overview (module responsibilities), config field reference, reproducibility checklist (seed, RNG, data order), checkpoint compatibility statement (schema version, legacy fallback behaviour).

---

## Regression Tests

Recommended test cases to validate correctness of the above fixes:

**T1. Checkpoint resume roundtrip** — Unit tests done (`tests/test_checkpoint.py` T1, T12–T16); GPU smoke test (T17 in Consolidated Plan) pending.
- Run for N steps, save checkpoint, resume, verify global_step is correct, LR is continuous, and the first post-resume batch is not a duplicate of the last pre-checkpoint batch.

**T2. Gradient normalization with shifted labels**
- Construct a batch with known valid-token count; verify `num_tokens_iter` matches the shifted label mask (not the unshifted one) and that `loss * num_tokens_iter / num_tokens_step` is numerically correct.

**T3. Small dataset: `len(dataloader) < gradient_accumulation_steps`**
- Verify a clear early error is raised (not a silent `ZeroDivisionError` deep in `math.ceil`).

**T4. All-ignored-label batch**
- Construct a batch where every label is `ignore_index`; verify the optimizer step is skipped with a warning rather than producing a `ZeroDivisionError`.
