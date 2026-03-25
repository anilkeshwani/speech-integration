# Research: Checkpoint and Resume Best Practices in Deep Learning

This document summarizes web research conducted on checkpoint/resume best practices across major PyTorch-based training frameworks (PyTorch Lightning, HuggingFace Transformers Trainer, torchtune, torchtitan, Megatron-LM, DeepSpeed). The goal is to inform design decisions for our intra-epoch step-level resume implementation.

---

## 1. Effective Batch Size Changes on Resume

### The Problem

Effective batch size = `per_device_batch_size * gradient_accumulation_steps * world_size`. If any of these three factors changes between checkpoint save and resume, the mapping from `global_step` to dataset position breaks: `batches_to_skip = (global_step % steps_per_epoch) * gradient_accumulation_steps` produces the wrong offset.

### How Frameworks Handle This

**HuggingFace Transformers Trainer:**
- Saves `train_batch_size` in `trainer_state.json` but does **not** save `gradient_accumulation_steps` or `world_size` -- this is a known gap ([Issue #21271](https://github.com/huggingface/transformers/issues/21271)).
- On mismatch, the step-skipping arithmetic silently corrupts. The maintainer position is explicit: "That feature is not supported, you should resume training with the exact same hyperparameters or start a new training if you want to change them."
- There was a historical bug where resume with gradient accumulation computed the wrong skip count, fixed in [PR #8624](https://github.com/huggingface/transformers/pull/8624).
- An issue was opened requesting a warning on batch size mismatch, but as of the latest search results, HF still does not reject or clearly warn on this.

**PyTorch Lightning:**
- Does not explicitly validate batch size on resume. The checkpoint saves `global_step` and `current_epoch`, and the training loop uses these directly. Changing batch size between runs is technically allowed but will cause the learning rate schedule and data position to be inconsistent with what the original run would have done.

**Megatron-LM:**
- Tracks `consumed_samples` as the canonical progress metric (not just `global_step`). This is more robust because `consumed_samples` is independent of batch size configuration.
- `global_batch_size = micro_batch_size * data_parallel_size * gradient_accumulation_steps`. Megatron derives `iteration` from `consumed_samples / global_batch_size`.
- Megatron also supports batch size rampup (starting small, linearly increasing to target), which inherently requires consumed_samples tracking.
- Changing `data_parallel_size` or `micro_batch_size` between runs is possible because the resume is based on consumed_samples, not step count. The new configuration simply derives a different step count from the same consumed_samples.

**DeepSpeed:**
- Standard ZeRO checkpoints require the **same number of GPUs** to resume. Each rank saves its own partition of optimizer states, and these partitions are tightly coupled to world_size.
- **Universal Checkpointing** (DeepSpeed >= 0.14.4) solves this by converting ZeRO checkpoints to a consolidated format via `ds_to_universal.py`, then re-sharding at load time for the new GPU count. This supports changing TP, PP, DP, SP, and ZeRO degrees. ([Tutorial](https://www.deepspeed.ai/tutorials/universal-checkpointing/), [Paper: arXiv 2406.18820](https://arxiv.org/html/2406.18820v2))

**torchtune:**
- Does not validate batch size on resume. The recipe state saves optimizer state, epoch, and seed, but not batch size or gradient accumulation configuration. This is a known gap ([Issue #1551](https://github.com/meta-pytorch/torchtune/issues/1551)).

**PyTorch DCP (Distributed Checkpoint):**
- Supports resharding when world_size changes for model and optimizer states. Does not handle the semantic question of what batch size change means for training progress.

### Recommendation

The approach in our plan (save `batch_size`, `gradient_accumulation_steps`, `world_size`, `steps_per_epoch` and validate on resume with a hard error) is **more rigorous than any major framework** except Megatron-LM's consumed_samples approach. The Megatron approach of tracking `consumed_samples` (or equivalently, `consumed_tokens`) is the gold standard because it decouples progress tracking from batch configuration. Consider:
- Primary: Save `consumed_samples` (or `consumed_tokens`) as the canonical progress counter alongside `global_step`.
- Secondary: Also save and validate the batch configuration as we planned, but use `consumed_samples` for data position recovery if batch size changes are ever needed.

### Sources
- [HuggingFace Trainer batch size mismatch issue #21271](https://github.com/huggingface/transformers/issues/21271)
- [HuggingFace Trainer gradient accumulation resume fix PR #8624](https://github.com/huggingface/transformers/pull/8624)
- [HuggingFace Accelerate resume with different batch size discussion](https://discuss.huggingface.co/t/resuming-accelerate-based-pretraining-with-different-batch-size/30845)
- [DeepSpeed Universal Checkpointing tutorial](https://www.deepspeed.ai/tutorials/universal-checkpointing/)
- [DeepSpeed Universal Checkpointing paper](https://arxiv.org/html/2406.18820v2)
- [DeepSpeed resume with different GPU count issue #7628](https://github.com/deepspeedai/DeepSpeed/issues/7628)
- [torchtune checkpoint resume issue #1551](https://github.com/meta-pytorch/torchtune/issues/1551)
- [Megatron-LM checkpoint resume bug #1570](https://github.com/NVIDIA/Megatron-LM/issues/1570)

---

## 2. Learning Rate Schedule Resume

### How Schedulers Should Be Resumed

The correct PyTorch pattern is:

1. Save `scheduler.state_dict()` alongside `optimizer.state_dict()`.
2. On resume, create a fresh scheduler instance, then call `scheduler.load_state_dict(saved_state)`.
3. The scheduler's internal `last_epoch` counter determines where in the schedule you are. **Note on naming**: despite being called `last_epoch`, this counter simply tracks how many times `scheduler.step()` has been called. Whether it counts epochs or optimizer steps depends entirely on the calling convention. Classical PyTorch examples (StepLR, MultiStepLR) call `.step()` per epoch; modern large-model codebases (HuggingFace Trainer, torchtune, fairseq) call `.step()` per optimizer update, making `last_epoch` effectively a step counter. `OneCycleLR` and `CyclicLR` are explicitly documented as requiring per-batch stepping. This naming confusion is an acknowledged open issue in PyTorch ([#69753](https://github.com/pytorch/pytorch/issues/69753), [#37768](https://github.com/pytorch/pytorch/issues/37768)). **In our codebase**, we call `.step()` per optimizer update, so `last_epoch` counts optimizer steps.

**Common pitfalls** ([PyTorch Forum discussion](https://discuss.pytorch.org/t/what-is-the-proper-way-of-resuming-a-scheduler/176350)):
- Initializing the scheduler with `last_epoch=saved_value` directly causes an `initial_lr` parameter error.
- Using `last_epoch=-1` (default) then loading state_dict works correctly.
- Different scheduler types (PolynomialLR, CosineAnnealingWarmRestarts, etc.) have subtly different behavior on resume.

### What If LR Schedule Parameters Change Between Runs?

**PyTorch Lightning:**
- Saves and restores scheduler state_dict. Changing LR schedule parameters between runs is a frequently requested feature but not cleanly supported ([Issue #12118](https://github.com/Lightning-AI/pytorch-lightning/issues/12118), [Issue #19865](https://github.com/Lightning-AI/pytorch-lightning/issues/19865)).
- Workaround: manually override LR in `on_epoch_start()` or use `load_from_checkpoint()` with parameter overrides, then start a "new" training from that model state.
- Known bug: scheduler sometimes uses `global_step=0` instead of the restored step on the first validation call after resume ([Issue #18588](https://github.com/Lightning-AI/pytorch-lightning/issues/18588), [Issue #12812](https://github.com/Lightning-AI/lightning/issues/12812)).

**HuggingFace Trainer:**
- Saves `scheduler.pt` in checkpoints. Restores it on resume. Does not support changing schedule parameters between runs.

**Megatron-LM:**
- Saves LR scheduler state in the checkpoint. The scheduler is tied to `consumed_samples` / iteration count.

**torchtune:**
- **Does not checkpoint the learning rate scheduler state** -- this is a known deficiency. On resume, the LR scheduler restarts from scratch, which means the model gets the wrong learning rate for its position in training. ([Issue #1551](https://github.com/meta-pytorch/torchtune/issues/1551))

### The WSD Schedule Advantage

The Warmup-Stable-Decay (WSD) schedule is increasingly popular for LLM pretraining because of its checkpoint-friendliness:
- The stable phase runs at constant LR indefinitely. You can checkpoint at any point during this phase and resume without any schedule state.
- When you want to produce a final model, you branch from a stable-phase checkpoint and run a short decay phase.
- Unlike cosine schedules, WSD does not require knowing `total_steps` in advance.
- WSD-Simplified (WSD-S) empirically outperforms WSD and Cyclic-Cosine for continual pretraining across 0.1B-1.2B parameter scales.

**Key reference:** [Understanding WSD Learning Rates (arXiv)](https://arxiv.org/html/2410.05192v1)

### Recommendation

- Always save and restore `scheduler.state_dict()`. This is non-negotiable for correct resume.
- For strict resume (same config), the scheduler state_dict handles everything.
- For warm restart (different config), load model + optimizer weights but create a fresh scheduler. This is the "fine-tuning from checkpoint" pattern, not "resume."
- Consider WSD schedule for its resume-friendliness: during the stable phase, the scheduler state is trivial (constant LR), making mid-training resume much simpler.

### Sources
- [PyTorch Forum: Proper way to resume a scheduler](https://discuss.pytorch.org/t/what-is-the-proper-way-of-resuming-a-scheduler/176350)
- [PyTorch Lightning: Allow resume with different LR #12118](https://github.com/Lightning-AI/pytorch-lightning/issues/12118)
- [PyTorch Lightning: Resume scheduler bug #18588](https://github.com/Lightning-AI/pytorch-lightning/issues/18588)
- [PyTorch Lightning: Change LR scheduler on resume #19865](https://github.com/Lightning-AI/pytorch-lightning/issues/19865)
- [torchtune: Missing LR scheduler checkpoint #1551](https://github.com/meta-pytorch/torchtune/issues/1551)
- [WSD Schedule paper](https://arxiv.org/html/2410.05192v1)

---

## 3. Dataset Position / Data Loader State on Resume

### Three Approaches

#### Approach A: Skip batches (islice / skip_first_batches)

**How it works:** On resume, recreate the dataloader from scratch with the same seed/epoch, then fast-forward by discarding `N` batches.

**Who uses it:**
- **HuggingFace Trainer**: Uses `accelerate.data_loader.skip_first_batches()` by default. Controlled by `ignore_data_skip` training argument (default: False = do skip). ([Trainer docs](https://huggingface.co/docs/transformers/en/main_classes/trainer))
- **Our plan**: Uses `itertools.islice(enumerate(data_train), batches_to_skip, None)`.

**Tradeoffs:**
- Pro: Simple, no extra state to save, works with any DataLoader.
- Con: **Slow for late-epoch resume.** Each skipped batch still runs the full CPU-side pipeline (dataset lookup, tokenization, collation). For large datasets like MLS (44k hours), skipping tens of thousands of batches could take hours. HuggingFace users report this as a major pain point ([Issue #21934](https://github.com/huggingface/transformers/issues/21934), [Forum discussion](https://discuss.huggingface.co/t/resume-from-checkpoint-skipping-batches-why-does-the-processing-function-need-to-be-run-for-skipped-batches/31291)).
- Con: Does not advance auxiliary RNG state that is consumed during `__getitem__` but not during islice skip (this is the critical PRNG issue identified in our plan critique).

#### Approach B: Save sampler/dataloader state (StatefulDataLoader)

**How it works:** `torchdata.stateful_dataloader.StatefulDataLoader` is a drop-in replacement for `torch.utils.data.DataLoader` that adds `state_dict()` / `load_state_dict()` methods. ([Tutorial](https://meta-pytorch.org/data/beta/stateful_dataloader_tutorial.html), [README](https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/README.md))

**What state is saved:**
- By default: number of batches yielded (used to fast-forward the sampler or dataset).
- If sampler defines `state_dict()`/`load_state_dict()`: saves sampler position and RNG state. `RandomSampler` and `BatchSampler` from `torch.utils.data` are monkey-patched when you import `torchdata.stateful_dataloader`.
- If dataset defines `state_dict()`/`load_state_dict()`: saves per-worker dataset state (e.g., RNG transform state for data augmentation).

**Constraints:**
- `num_workers` must be the same on resume as on save.
- Available in `torchdata >= 0.8.0`.

**Framework adoption:**
- **HuggingFace**: Active work to integrate StatefulDataLoader ([Issue #31441](https://github.com/huggingface/transformers/issues/31441)). Available in accelerate >= 0.34.0 via `use_stateful_dataloader` config.
- **PyTorch Lightning**: Discussions ongoing ([Discussion #21276](https://github.com/Lightning-AI/pytorch-lightning/discussions/21276), [Discussion #20129](https://github.com/Lightning-AI/pytorch-lightning/discussions/20129)).
- **torchtitan**: Uses StatefulDataLoader for checkpoint state management.

#### Approach C: Re-shuffle and accept data repetition/skipping

**How it works:** On resume, just start a fresh epoch with a new shuffle. Accept that some data points from the interrupted epoch will be repeated and others skipped.

**Who uses it:**
- This is the default behavior if you don't implement either A or B.
- Common in research codebases where exact reproducibility is not a priority.
- HuggingFace Trainer with `ignore_data_skip=True`.

**Tradeoffs:**
- Pro: Trivial to implement, zero resume overhead.
- Con: Statistical bias -- some samples get seen twice, others are never seen in that "epoch."
- Con: For single-epoch training on massive datasets (standard LLM pretraining), this means permanently losing coverage of some data.

#### Megatron-LM Approach

Megatron-LM takes a unique approach: it uses `consumed_samples` to construct a deterministic index mapping. The dataset is pre-shuffled with a fixed seed, and `consumed_samples` directly indexes into this pre-shuffled sequence. There is no need to skip batches or save dataloader state -- the dataloader simply starts reading from position `consumed_samples` in the pre-shuffled index.

### Recommendation

For our use case (MLS 44k hours, never completes an epoch):
1. **Short term**: The islice approach in the current plan is acceptable but needs the PRNG state fix identified in the critique. Add timing/logging to quantify skip cost.
2. **Medium term**: Consider `torchdata.StatefulDataLoader` to avoid the skip cost entirely. This requires sampler state_dict support and is a cleaner long-term solution.
3. **Long term**: The Megatron-LM consumed_samples approach (pre-shuffled index + direct seek) is the gold standard for datasets that never complete an epoch.

### Sources
- [torchdata StatefulDataLoader tutorial](https://meta-pytorch.org/data/beta/stateful_dataloader_tutorial.html)
- [torchdata StatefulDataLoader README](https://github.com/meta-pytorch/data/blob/main/torchdata/stateful_dataloader/README.md)
- [HuggingFace: StatefulDataLoader support #31441](https://github.com/huggingface/transformers/issues/31441)
- [HuggingFace: Faster skipping #21934](https://github.com/huggingface/transformers/issues/21934)
- [HuggingFace: Why does resume run processing for skipped batches?](https://discuss.huggingface.co/t/resume-from-checkpoint-skipping-batches-why-does-the-processing-function-need-to-be-run-for-skipped-batches/31291)
- [HuggingFace: Streaming dataset fast resume discussion](https://discuss.huggingface.co/t/would-it-be-possible-to-implement-and-iterable-dataset-with-streaming-and-fast-resume-no-need-to-skip-batches/56119)
- [PyTorch Lightning: StatefulDataLoader discussion #21276](https://github.com/Lightning-AI/pytorch-lightning/discussions/21276)
- [PyTorch Forum: Resume dataloader from batch_idx](https://discuss.pytorch.org/t/resume-iterating-dataloader-from-checkpoint-batch-idx/60683)

---

## 4. RNG State Preservation

### What RNG States Exist

A complete PyTorch training setup has **four** independent RNG states:

| RNG | Save | Restore |
|-----|------|---------|
| Python `random` | `random.getstate()` | `random.setstate(state)` |
| NumPy | `numpy.random.get_state()` | `numpy.random.set_state(state)` |
| PyTorch CPU | `torch.get_rng_state()` | `torch.set_rng_state(state)` |
| PyTorch CUDA (per device) | `torch.cuda.get_rng_state_all()` | `torch.cuda.set_rng_state_all(states)` |

For `numpy.random.default_rng()` (Generator API, used in our `cpt.py`), the state must be saved/restored via the generator's own `bit_generator.state` property, **not** via the legacy `numpy.random.get_state()`.

### How Important Is It?

**For exact bitwise reproducibility on resume:** Essential. Without RNG state restoration, dropout patterns, data augmentation, and any stochastic operation will diverge from the uninterrupted run.

**For practical training quality:** Less critical for most uses. The model will converge to a similar quality regardless of RNG state, since the randomness is by design beneficial. However:
- **Data shuffling RNG**: Important if you care about data ordering reproducibility.
- **Dropout/augmentation RNG**: Less important for convergence, more for debugging.
- **Our specific case**: The `PRNG` in `cpt.py` that controls interleaving is critical because it determines the *content* of training samples, not just their order.

### What Frameworks Save

**HuggingFace Trainer:** Saves all four RNG states in `rng_state.pth` per checkpoint. In DDP, saves per-rank RNG states. Controlled by `save_only_model` flag (default: save everything).

**PyTorch Lightning:** Saves RNG states as part of the checkpoint. Includes Python, NumPy, PyTorch CPU, and CUDA states.

**Megatron-LM:** Saves all model weights, optimizer states, and RNG states.

**DeepSpeed:** Saves RNG states per rank.

**torchtune:** Does **not** save RNG states ([Issue #1551](https://github.com/meta-pytorch/torchtune/issues/1551) -- only saves optimizer state, epoch, seed).

### DDP-Specific Gotcha

When using DistributedSampler with `shuffle=True`, the shuffle seed is `base_seed + epoch` (via `sampler.set_epoch(epoch)`). This isolates the DataLoader's shuffle RNG from the global state. However, if any code between "restore RNG" and "DataLoader consumes RNG" advances the global state, the DataLoader will get a different RNG sequence. ([PyTorch Forum discussion](https://discuss.pytorch.org/t/save-random-generator-states-and-resume-training-for-ddp/146994))

### Recommendation

1. Save all four RNG states (Python, NumPy global, PyTorch CPU, PyTorch CUDA).
2. **Additionally** save the `cpt.py` PRNG state (`PRNG.bit_generator.state`) since it uses `numpy.random.default_rng()` which is independent of the global NumPy RNG.
3. Restore RNG states as early as possible after checkpoint load, before any code that might consume random numbers.
4. For DDP, save per-rank RNG states.

### Sources
- [PyTorch Reproducibility documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)
- [PyTorch Forum: Save RNG states for DDP training](https://discuss.pytorch.org/t/save-random-generator-states-and-resume-training-for-ddp/146994)
- [What to save in checkpoints (apxml.com)](https://apxml.com/courses/how-to-build-a-large-language-model/chapter-19-checkpointing-fault-tolerance/saving-model-state)
- [Notes on reproducibility in PyTorch (GitHub gist)](https://gist.github.com/Guitaricet/28fbb2a753b1bb888ef0b2731c03c031)

---

## 5. What Should a Checkpoint Contain?

### Comparison Across Frameworks

| Component | PyTorch Lightning | HuggingFace Trainer | Megatron-LM | DeepSpeed | torchtune | torchtitan |
|-----------|:-:|:-:|:-:|:-:|:-:|:-:|
| Model state_dict | Yes | Yes (`pytorch_model.bin`) | Yes (sharded) | Yes (sharded per rank) | Yes | Yes |
| Optimizer state_dict | Yes | Yes (`optimizer.pt`) | Yes | Yes (sharded per rank) | Yes (intermediate only) | Yes (configurable) |
| LR Scheduler state_dict | Yes | Yes (`scheduler.pt`) | Yes | Yes | **No** | Yes |
| Current epoch | Yes | Yes (`trainer_state.json`) | Derived from consumed_samples | Yes | Yes | Yes |
| Global step | Yes | Yes (`trainer_state.json`) | Yes (iteration) | Yes | No (epoch-level only) | Yes |
| Consumed samples/tokens | No | No | **Yes** | No | No | Unknown |
| RNG states (all 4) | Yes | Yes (`rng_state.pth`) | Yes | Yes | **No** | Yes |
| Callback/hook states | Yes | No | N/A | N/A | N/A | N/A |
| DataModule/DataLoader state | Yes (if stateful) | No (uses skip) | Via consumed_samples | No | No | Via StatefulDataLoader |
| Hyperparameters | Yes (init args) | Yes (`training_args.bin`) | Partial (via args) | Via config | No | Via config |
| GradScaler state | Yes | Yes (if using AMP) | Yes | Yes | Unknown | Unknown |
| Training config/metadata | Partial | Yes | Partial | Yes | No | Partial |
| Loss/metrics history | No | Yes (`trainer_state.json`) | No | No | No | No |

### The "Complete" Checkpoint

Based on the research, a checkpoint that enables **perfect, bitwise-identical resume** should contain:

```
1. model.state_dict()
2. optimizer.state_dict()
3. scheduler.state_dict()
4. grad_scaler.state_dict()          # if using mixed precision
5. global_step                        # canonical progress counter
6. consumed_samples (or consumed_tokens)  # batch-size-independent progress
7. epoch                              # for epoch-based bookkeeping
8. RNG states:
   a. random.getstate()
   b. numpy.random.get_state()
   c. torch.get_rng_state()
   d. torch.cuda.get_rng_state_all()
   e. Any custom PRNG states (e.g., numpy.random.Generator)
9. dataloader/sampler state           # or enough info to reconstruct position
10. training_hparams:
    a. batch_size
    b. gradient_accumulation_steps
    c. world_size
    d. steps_per_epoch
11. Metadata:
    a. Checkpoint format version
    b. Timestamp
    c. Training config hash or snapshot
    d. Cumulative metrics (tokens seen, loss history, etc.)
```

### Practical Tiers

**Tier 1 -- Inference only:** Model state_dict only. ~1/3 the size of a full checkpoint.

**Tier 2 -- Resume training (approximate):** Model + optimizer + scheduler + global_step + epoch. Loses exact data position and RNG state but converges similarly.

**Tier 3 -- Resume training (exact):** Everything in the complete list above. Required for bitwise reproducibility and exact data position recovery.

### Sources
- [PyTorch Lightning: Saving and loading checkpoints](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
- [HuggingFace Accelerate: Checkpointing guide](https://huggingface.co/docs/accelerate/usage_guides/checkpoint)
- [PyTorch: Saving and loading a general checkpoint](https://docs.pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)
- [Large model checkpointing tips (Nebius)](https://nebius.com/blog/posts/model-pre-training/large-ml-model-checkpointing-tips)
- [Checkpointing strategies for LLMs (Medium)](https://medium.com/@dpratishraj7991/checkpointing-strategies-for-large-language-models-llms-full-sharded-efficient-restarts-at-0fa026d8a566)
- [DeepSpeed Model Checkpointing docs](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)
- [Megatron Bridge Checkpointing docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/checkpointing.html)

---

## 6. Checkpoint Versioning / Metadata

### What Frameworks Save

**HuggingFace Trainer:**
- `training_args.bin`: Full `TrainingArguments` object serialized via pickle. Contains all training hyperparameters.
- `trainer_state.json`: JSON with `global_step`, `epoch`, `best_metric`, `log_history` (full loss/metric history), and `trial_name`/`trial_params` for HPO.
- `config.json`: Model configuration.
- No explicit checkpoint version number.

**PyTorch Lightning:**
- Saves `hyper_parameters` (the init arguments passed to LightningModule) in the checkpoint.
- Saves `pytorch-lightning_version` in the checkpoint for compatibility checking.
- Saves `state_dict` of all callbacks, enabling stateful callbacks to resume.
- Uses `ModelCheckpoint` callback with configurable filename patterns for versioning.

**Megatron-LM:**
- Saves `train_state.pt` with iteration count and consumed_samples.
- Checkpoint directory names include the iteration number for versioning.
- Training arguments are logged but not fully saved in the checkpoint itself.

**DeepSpeed:**
- Saves `ds_config.json` alongside checkpoint files.
- Checkpoint directory structure: `global_step{N}/` containing per-rank files.
- Supports `client_state` dict for arbitrary user metadata.

**torchtune:**
- Saves `recipe_state.pt` with optimizer state, epoch, and seed.
- Model weights saved in HuggingFace-compatible format (with weight map).
- No explicit metadata or configuration snapshot.

### How Frameworks Handle Loading From Different Configurations

Most frameworks take a **strict** approach: they assume the configuration matches and do not validate. The common workarounds are:

1. **Ignore and hope**: Load what you can, let missing/extra keys cause errors.
2. **Strict mode flag**: PyTorch's `model.load_state_dict(strict=False)` allows partial loading.
3. **Key remapping**: Manual mapping of old key names to new ones (common when model architecture changes).
4. **Conversion scripts**: Dedicated tools like Megatron-LM's `tools/checkpoint/convert.py` or DeepSpeed's `ds_to_universal.py`.

### Recommendation

Save the following metadata in every checkpoint:
- `checkpoint_version`: Integer version for our checkpoint format. Increment when the structure changes.
- `timestamp`: ISO 8601 timestamp of when the checkpoint was saved.
- `training_config`: A snapshot of the full training configuration (or its hash) for auditability.
- `training_hparams`: The batch-size-related parameters needed for validation (already in our plan).
- `cumulative_metrics`: Total tokens processed, total training time, etc., for monitoring continuity across resume boundaries.

### Sources
- [PyTorch Lightning: Checkpointing docs](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)
- [HuggingFace Trainer docs](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [DeepSpeed: Model Checkpointing docs](https://deepspeed.readthedocs.io/en/latest/model-checkpointing.html)
- [torchtune: Checkpointing deep dive](https://docs.pytorch.org/torchtune/0.3/deep_dives/checkpointer.html)

---

## 7. The "Step" vs "Epoch" Debate

### Context

For LLM pretraining, the dataset is typically so large that training runs for less than one epoch. For example, our MLS dataset is 44k hours and the training budget (48 GPU-hours on a single A6000) will never complete a full pass. This makes the "epoch" concept nearly meaningless.

### Industry Practice

**Single-epoch training is the norm for LLM pretraining.** Each token is typically seen only once. Checkpoints are taken every N steps (optimizer updates), not at epoch boundaries. This is the standard approach used by LLaMA, GPT, and other large-scale training runs.

**Step-based training patterns:**
- `max_steps` replaces `num_epochs` as the primary training duration parameter.
- Checkpoints are saved every `save_steps` optimizer steps.
- Evaluation happens every `eval_steps` steps.
- Logging happens every `log_steps` steps.
- Retention policies keep only the last N checkpoints plus periodic historical ones.

### How to Handle Mid-Epoch Resume

**Approach 1: Step-based with epoch bookkeeping (our current plan)**
- Use `global_step` as the canonical counter.
- Derive `epochs_run` and `batches_to_skip` from `global_step` and `steps_per_epoch`.
- Works well when the dataset size is fixed and known.

**Approach 2: Consumed-samples-based (Megatron-LM)**
- Track `consumed_samples` as the canonical counter.
- The dataset provides a deterministic mapping from sample index to data.
- On resume, start reading from `consumed_samples` position.
- Completely decouples progress from batch size and epoch structure.
- More robust when batch size might change between runs.

**Approach 3: Token-based**
- Track `consumed_tokens` instead of samples (relevant when sequence lengths vary).
- Less common but more precise for variable-length training.

### The WSD Schedule Connection

The WSD (Warmup-Stable-Decay) schedule is particularly well-suited for step-based training:
- No need to know `total_steps` in advance (unlike cosine schedule).
- The stable phase can run indefinitely; branch off for decay when ready.
- Checkpoints from the stable phase can be used to spawn multiple decay runs at different compute budgets.
- WSD-Simplified (WSD-S) starts new phases directly from intermediate checkpoints at high LR, outperforming standard WSD for continual pretraining.

### Recommendation

1. Use `global_step` (optimizer updates) as the primary progress counter.
2. Also track and checkpoint `consumed_samples` and/or `consumed_tokens` for batch-size-independent progress tracking and monitoring continuity.
3. Keep `epoch` as a bookkeeping convenience but do not rely on it for resume logic.
4. Use `max_steps` rather than `num_epochs` to define training duration.
5. Consider WSD schedule for its natural fit with step-based, mid-training-resumable workflows.

### Sources
- [Epochs in ML pipelines (Nebius)](https://nebius.com/blog/posts/epochs-in-day-to-day-ml-pipelines)
- [Checkpoint on checkpoints in LLMs (VAST Data)](https://www.vastdata.com/blog/a-checkpoint-on-checkpoints-in-llms)
- [LLM Training Checkpointing & Fault Tolerance (apxml.com)](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-3-llm-training-finetuning-ops/checkpointing-fault-tolerance)
- [Checkpointing for distributed training failures (Medium)](https://medium.com/better-ml/checkpointing-for-distributed-training-failures-603caadb5c96)
- [WSD Schedule paper](https://arxiv.org/html/2410.05192v1)
- [Mid-Training of LLMs survey](https://arxiv.org/html/2510.06826v1)
- [torchtitan checkpoint docs](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md)
- [PyTorch DCP tutorial](https://docs.pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)

---

## Summary of Key Takeaways for Our Implementation

| Decision Point | Recommendation | Rationale |
|---|---|---|
| Progress counter | `global_step` primary, `consumed_samples` secondary | Megatron gold standard; decouples from batch config |
| Batch size change on resume | Validate and reject (current plan) | HF's silent corruption is a known footgun; strictness is correct |
| LR scheduler | Save and restore `state_dict()` | Universal best practice; torchtune's omission is a known bug |
| Dataset position | islice skip (now) -> StatefulDataLoader (later) | islice is simple but slow; StatefulDataLoader is the future |
| RNG state | Save all 4 standard + custom PRNG | Critical for our interleaving PRNG; standard practice in all major frameworks |
| Checkpoint metadata | Add version, timestamp, config snapshot | Goes beyond most frameworks; essential for auditability |
| Epoch vs step | Step-based with `max_steps` | Standard for LLM pretraining; epoch is meaningless for MLS |
| LR schedule | Consider WSD for resume-friendliness | No `total_steps` dependency; natural branching for decay |
