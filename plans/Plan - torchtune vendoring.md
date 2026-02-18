# Torchtune Vendoring Plan

Internalize torchtune ownership. 

## Summary

Remove direct torchtune dependency by introducing an internal compatibility layer and gradually owning the required torchtune surface.

## Phases

Phases 1, 2 and 8 are captured in the main Plan.md. Others specifically relate to torchtune vendoring. 


### Phase 0: Baseline + Guardrails

- Freeze a baseline commit and capture reproducibility metadata.
- Add import audit script to track all direct torchtune imports.
- Add temporary CI check to report (not fail yet) direct torchtune imports.
- Exit criteria: baseline reproducibility run artifact + import audit artifact committed.

### Phase 1: Correctness fixes (highest priority)

Note: This phase is mostly captured in the main Plan.md. 

- Fix checkpoint resume key mismatch in ssi/train.py and ssi/checkpoint.py by defining one canonical step key strategy.
- Remove batch mutation in ssi/loss.py (batch.pop("labels")).
- Fix generation config flow in scripts/generate.py so train_cfg is always defined correctly.
- Eliminate global singleton mutation in ssi/llama_configs.py (update_from_speech_cfg should mutate self only).
- Exit criteria: checkpoint resume test passes, loss purity test passes, generation config tests pass.

### Phase 2: Data loading hardening

Note: This phase is mostly captured in the main Plan.md. 

- Add explicit schema validation for required dataset columns in ssi/data/sft.py and ssi/data/cpt.py.
- Replace mutable default args ([]) and other unsafe defaults in dataset/collate code.
- Move CPT PRNG from module-global to instance-owned RNG for deterministic reproducibility.
- Add deterministic fixtures for deduplication, modality token behavior, and interleave span correctness.
- Exit criteria: deterministic tokenization/collation tests pass under fixed seed.

### Phase 3: Torchtune integration boundary (new ownership track start)

- Create ssi/tune/ as the only allowed boundary for torchtune access.
- Replace all from torchtune... imports in ssi/* and scripts/* with ssi.tune.* imports.
- Add hard CI rule: fail on direct imports matching ^from torchtune|^import torchtune outside ssi/tune.
- Exit criteria: direct torchtune imports only exist inside ssi/tune.

### Phase 4: Internalize private torchtune APIs first

- Rehome currently private/internal dependencies behind our API:
- Checkpoint utils currently from torchtune.training.checkpointing._utils
- Checkpointer interface currently from torchtune.training.checkpointing._checkpointer
- Data internals from torchtune.data._common, _messages, _utils
- Tokenizer internals from torchtune.models.llama3._tokenizer
- Import guard from torchtune.utils._import_guard
- Copy minimal required source from third_party/torchtune with attribution headers intact.
- Exit criteria: no imports from torchtune private modules (._*) anywhere in runtime code.

### Phase 5: Own runtime-critical torchtune surface

- Internalize model/tokenizer/training utilities required by SSI runtime:
- llama3_2 builders used by ssi/model.py
- tokenizer classes/constants used by ssi/tokenizer/*
- core modules/loss used in ssi/train.py, ssi/loss.py, ssi/eval.py
- scheduler/precision/device helpers currently used from torchtune training/utils
- Keep module scope minimal to SSI needs; do not vendor unused torchtune subsystems.
- Exit criteria: training and generation run without importing installed torchtune package.

### Phase 6: Checkpoint conversion consolidation

- Replace from torchtune.models import convert_weights dependency path with owned conversion module.
- Narrow supported checkpoint conversion model types to explicitly supported SSI targets.
- Add versioned checkpoint schema and migration logic for old checkpoints.
- Exit criteria: load/save/resume compatibility tests pass with owned conversion path.

### Phase 7: Architecture cleanup + public interface simplification

- Split ssi/train.py into orchestrator + components (train step, eval step, logging, checkpoint policy).
- Normalize Hydra config shape and remove obsolete flags/TODO pathways.
- Optionally unify training entrypoints into single script with explicit mode.
- Exit criteria: smaller modules, clear interfaces, no behavior regressions in smoke runs.

### Phase 8: Testing, docs, and publication readiness

Note: This phase is mostly captured in the main Plan.md. 

- Add unit tests for checkpoint schema, data transforms, loss, tokenizer behavior.
- Add integration smoke tests (tiny fixtures) for CPT/SFT train + generation.
- Document owned dependency policy and third-party attribution for vendored code.
- Update README/developer docs with final architecture and supported interfaces.
- Exit criteria: CI green on lint+tests; docs match actual runtime behavior.

## API/interface changes to expect

- New internal boundary package: ssi.tune.
- Direct torchtune usage removed from application modules.
- Canonical checkpoint schema formalized and versioned.
- Some config keys/CLI behavior may change (breaking changes allowed per your guidance).

## Risk controls

- Keep migration incremental with parity tests at each phase.
- Move private API usages first to reduce fragility quickly.
- Keep third_party/torchtune pinned at v0.5.0 during migration; drop pip dependency only after parity passes.