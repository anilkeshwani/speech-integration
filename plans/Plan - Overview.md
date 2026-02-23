# Sprint Plan: Stabilize, Then Refactor the Speech Integration Codebase

## Summary

This plan prioritizes correctness and reproducibility bugs first (especially checkpointing and data loading), then performs architecture and code-quality refactors.
We allow backward-incompatible changes and prioritise improving final code quality.

## Architecture Review (Current State)

- Runtime entrypoints are thin Hydra wrappers (scripts/train_cpt.py, scripts/train_sft.py, scripts/generate.py) that delegate into ssi/*.
- Core training orchestration is centralized in ssi/train.py, with data setup in ssi/data/__init__.py, model wiring in ssi/model.py, checkpointing in ssi/checkpoint.py, and loss logic in ssi/loss.py.
- Config management is Hydra-based (conf/common.yaml, conf/training.yaml, conf/{cpt,sft,generate}.yaml, and dataset group files).
- Strengths:
    - Clear separation between CPT/SFT dataset implementations.
    - Good use of typed configs and torchtune integrations.
    - Reproducibility intent (seed constants, config logging, deterministic options).
- Key structural weaknesses:
    - Mutable global/singleton config state in model config layer (ssi/llama_configs.py).
    - Training loop mixes orchestration, metrics, checkpoint policy, and token accounting in one large function (ssi/train.py).
    - Incomplete/fragile abstractions around checkpoint schema and resume semantics (ssi/checkpoint.py, ssi/train.py).
    - No automated tests; correctness is currently trust-based.

## High-Priority Findings (Bugs / Risks)

1. Resume checkpoint key mismatch can break training resume.

- ssi/train.py:63 reads STEPS_KEY from checkpoint.
- ssi/checkpoint.py:536 writes GLOBAL_STEP_KEY instead.
- Impact: resume can fail or silently restore wrong step semantics.

2. compute_loss mutates input batch in place.

- ssi/loss.py:15 uses batch.pop("labels").
- Impact: hidden side effects across train/eval flow; brittle if batch reused/logged downstream.

3. ConfigLlama3_2.update_from_speech_cfg updates a global singleton regardless of instance.

- ssi/llama_configs.py:48-49 writes to configllama3_2_1b directly.
- Impact: non-local state mutation, harder testing, cross-run contamination risk.

4. Generation config loading logic has a latent unbound variable path.

- In scripts/generate.py, train_cfg is only set inside if cfg.train_yaml is None branch.
- Later code reads train_cfg unconditionally (scripts/generate.py:163+).
- Impact: passing train_yaml may fail unexpectedly.

5. Checkpoint module has correctness/cleanliness defects that should be fixed before refactor.

- Suspicious unused import from ast import Not (ssi/checkpoint.py:5).
- Output directory creation policy and schema ownership are mixed with conversion logic.
- Impact: maintainability and reliability risk in the known weak area.

6. Missing test coverage for known weak paths.

- No test_* suite found across repository.
- Impact: regressions likely during overhaul.

## Target End-State Architecture

- ssi/app/ (or equivalent):
- train_service.py: end-to-end train orchestration with explicit lifecycle methods.
- generate_service.py: generation workflow with validated config resolution.
- ssi/domain/:
- model_config.py: immutable model/speech config object; no singleton mutation.
- checkpoint_schema.py: versioned checkpoint metadata and keys.
- ssi/infra/:
- checkpoint_io.py: HF/torchtune serialization conversions.
- data_loaders.py: dataset + collate composition.
- ssi/eval/:
- metrics.py, wer_eval.py with explicit contracts.
- tests/:
- unit tests for data transforms, loss contracts, config resolution.
- integration tests for train-resume and generate path.

## Implementation Plan (Decision-Complete)

### Phase 1: Correctness Stabilization (Bug Fixes First)

- Standardize checkpoint resume schema.
- Define canonical key set: epochs_run, global_step, optimizer, seed.
- Update resume_training_state and save paths to one schema.
- Add schema validator in checkpoint load path.
- Remove input mutation in loss.
- Change compute_loss to read labels = batch["labels"] and keep function pure.
- Add explicit docstring contract: input batch is immutable.
- Fix generation config path consistency.
- Ensure train_cfg is always loaded from either explicit cfg.train_yaml or inferred fallback.
- Add clear error messaging on both paths.
- Eliminate singleton side effects in llama config.
- Refactor update_from_speech_cfg to update self only.
- Instantiate model config per run and pass explicitly through setup functions.
- Clean checkpoint module defects.
- Remove invalid/unused imports and dead code.
- Separate I/O directory creation policy from conversion logic.

Acceptance criteria for Phase 1:

- Resume-from-checkpoint works for both fresh and resumed runs.
- compute_loss leaves input batch unchanged.
- scripts/generate.py works with and without explicit train_yaml.
- Unit tests cover these regressions.

### Phase 2: Data Loading Hardening

- Introduce explicit dataset contracts (TypedDict/dataclass) for sample keys per mode.
- Validate required columns at dataset initialization (SFTDataset, TextCompletionDataset).
- Refactor collate API (ssi/data/__init__.py) to avoid mutable defaults and implicit behavior.
- Add deterministic behavior controls for interleaving/random span generation.
- Replace global PRNG (ssi/data/cpt.py) with per-dataset RNG seeded in constructor.
- Add targeted tests:
- speech deduplication behavior.
- modality token insertion.
- interleave span boundary correctness.
- packed/unpacked collation invariants.

Acceptance criteria for Phase 2:

- Data pipeline fails fast on schema mismatch with actionable errors.
- Same seed + config yields stable tokenized output in test fixtures.
- Collate outputs satisfy shape/label invariants.

### Phase 3: Training Loop Refactor

- Split ssi/train.py into:
- config validation/setup
- epoch/step execution
- evaluation
- logging
- checkpoint policy
- Introduce explicit TrainState object (epoch, global_step, token counters, timers).
- Add callback-like hooks for logging/checkpointing/eval triggers.
- Preserve current behavior but with clearer interfaces and test seams.

Acceptance criteria for Phase 3:

- train() complexity reduced and logically decomposed.
- Step accounting and logging behavior verified with integration tests.
- No behavioral regression in checkpoint cadence and eval cadence.

### Phase 4: Interface and Config Cleanup (Breaking Changes Allowed)

- Flatten/normalize Hydra config structure for training and generation.
- Remove deprecated fields and TODO placeholders not used in final architecture.
- Make CLI interfaces explicit:
- scripts/train.py --mode {cpt,sft}
- scripts/generate.py with strict required model/config options
- Document final stable public interfaces in README.

Acceptance criteria for Phase 4:

- One clear training entrypoint and one generation entrypoint.
- Config schema is explicit and validated.
- Documentation matches executable behavior exactly.

### Phase 5: Testing, CI, and Documentation

- Add test stack:
- pytest unit tests for utils/data/loss/config parsing.
- lightweight integration tests using tiny synthetic fixtures/mocks.
- Add quality gates:
- format (black, isort)
- lint (flake8 or ruff; recommend consolidating to ruff)
- tests required in CI.
- Expand docs:
- architecture overview diagram/table
- config reference
- reproducibility checklist
- checkpoint compatibility statement (final schema only)

Acceptance criteria for Phase 5:

- CI runs lint + tests on every branch.
- README + docs are sufficient for a new contributor to run train/generate end-to-end.
- Known weak spots (checkpointing, data loading) have explicit test coverage.

## Public API / Interface Changes

- Replace implicit mutable model config singleton with explicit per-run config object.
- Canonical checkpoint state schema with global_step as the single step field.
- Potential consolidation of training entrypoints into one script with mode flag.
- Stricter generation config contract (train_yaml resolution behavior formalized).
- Data loader interfaces will enforce explicit dataset key requirements.

## Test Cases and Scenarios

- Resume training from saved checkpoint:
- validates epoch, global step, optimizer state, and seed checks.
- Loss function purity:
- batch input remains identical pre/post compute_loss.
- Generation config scenarios:
- explicit train_yaml provided.
- inferred fallback config path.
- failure path emits actionable error.
- Data transforms:
- SFT inference vs training output masking.
- CPT interleave/concatenate modes and deduplication.
- modality token insertion on/off.
- End-to-end smoke tests:
- tiny synthetic dataset for CPT and SFT one-step train.
- one-batch generation JSONL output schema validation.

## Assumptions and Defaults

- Backward compatibility is intentionally relaxed for cleaner final architecture.
- Priority order is fixed: bugs/correctness first, then refactor.
- Primary weak zones to address first are checkpointing and data loading.
- We will standardize on a final published interface and remove transitional complexity.
