# Checkpointing: Design Rationale and Future Work

This document records the design principles, decision rationale, and future work for the checkpoint and resume system. Implementation details are in the code itself (`ssi/checkpoint.py`, `ssi/train.py`, `ssi/constants.py`). The companion `Research - Checkpoint and Resume Best Practices.md` contains the web research that informed the design.

---

## Table of Contents

1. [Context](#1-context)
2. [Design Principles](#2-design-principles)
3. [Design Decision Log](#3-design-decision-log)
4. [Future Work](#4-future-work)

---

## 1. Context

MLS is 44k hours. Our training budget is 48 GPU-hours on a single A6000. One epoch of MLS will never complete, so every training run ends mid-epoch. The checkpoint system must correctly resume mid-epoch, detect configuration mismatches that would corrupt training, and produce deterministic results across resumes.

---

## 2. Design Principles

These principles guide all checkpoint-related decisions. When evaluating future changes, refer back here.

### P1: `global_step` is the canonical resume counter

`global_step` (optimizer updates completed) is the single source of truth for training progress. Epoch number and batch offset are derived from it. This follows the universal standard for LLM pretraining where epochs rarely complete.

### P2: `consumed_samples` is the batch-size-independent progress counter

We also track `consumed_samples` (total micro-batches processed x batch_size) as a secondary counter. This is Megatron-LM's approach and allows future flexibility if batch size ever needs to change between runs. For now, it serves primarily as a monitoring and auditability field.

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

---

## 4. Future Work

These items are explicitly out of scope for the current implementation but are documented for planning.

### F1: `torchdata.StatefulDataLoader`

Replace `torch.utils.data.DataLoader` with `torchdata.StatefulDataLoader` to eliminate islice skip cost. Requires `torchdata >= 0.8.0`. The DataLoader state_dict would be saved in the checkpoint alongside the other state. This also enables `num_workers > 0` for the DataLoader itself (the per-sample deterministic RNG in D6 already makes the dataset side safe for multi-worker access).

When enabling `num_workers > 0`, note that islice-based resume skipping is still synchronous and its cost scales with `num_workers=0` assumptions. Switching to `StatefulDataLoader` remains the recommended path for production use with `num_workers > 0`. See `plans/Reference - Memory in Multi-Worker DataLoaders.md` for the memory implications of multi-worker data loading.

### F2: Megatron-style consumed_samples indexing

Pre-shuffle the dataset with a deterministic seed and use `consumed_samples` to index directly into the shuffled sequence. No skip, no stateful dataloader. This is the most robust solution for datasets that never complete an epoch.

### F3: WSD learning rate schedule

Implement Warmup-Stable-Decay schedule for checkpoint-friendly training. The stable phase has trivial scheduler state (constant LR), making resume simpler. Natural branching for producing final models at different compute budgets. No need to know `total_steps` in advance (unlike cosine). See research doc section 2 for details.

### F4: Checkpoint retention policy

Currently all checkpoints are kept. Implement a policy like "keep last N + every Kth checkpoint" to manage disk usage. HuggingFace and PyTorch Lightning both have this built in.

### F5: Async checkpointing

Save checkpoints without blocking training. PyTorch DCP and torchtitan support this. Useful when checkpoint save time is a significant fraction of step time.

### F6: Batch size change support

Using `consumed_samples` as the primary counter would allow changing batch size between runs. The training loop would derive the step count from `consumed_samples / new_effective_batch_size`. This requires F2 (consumed_samples indexing) to work correctly for data position.
