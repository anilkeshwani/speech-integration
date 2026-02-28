# Critique of `ssi/train.py`

## Findings (ordered by severity)

1. **Checkpoint resume is broken due to step-key mismatch**
   - `resume_training_state()` reads `ckpt_dict[STEPS_KEY]` (`"steps_run"`) in [`ssi/train.py:60-63`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:60).
   - `save_checkpoint()` writes `GLOBAL_STEP_KEY` (`"global_step"`), not `STEPS_KEY`, in [`ssi/checkpoint.py:534-538`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/checkpoint.py:534).
   - Result: resume can fail with missing key or restore wrong step state.

2. **Checkpointed epoch semantics are inconsistent with `epochs_run`**
   - Resume assumes `EPOCHS_KEY` means completed epochs (`epochs_run`) in [`ssi/train.py:63`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:63).
   - Save stores current loop index `epoch` mid-epoch in [`ssi/checkpoint.py:536`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/checkpoint.py:536).
   - On resume, training restarts that epoch from the beginning (`range(epochs_run, n_epochs)` in [`ssi/train.py:168`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:168)), duplicating data and breaking exact continuation.

3. **Gradient normalization uses the wrong token count**
   - You compute `num_tokens_iter` from unshifted labels in [`ssi/train.py:175`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:175).
   - But loss is computed on **shifted** labels in [`ssi/loss.py:16`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/loss.py:16).
   - This creates a mathematical mismatch in:
     - scaling of per-batch loss (`* num_tokens_iter`) in [`ssi/train.py:178`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:178)
     - gradient renormalization (`1 / num_tokens_step`) in [`ssi/train.py:182`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:182)
     - logged loss (`loss_running / num_tokens_step`) in [`ssi/train.py:190`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:190)
   - Bias is typically small per batch, but systematic.

4. **`steps_per_epoch` can be zero, causing hard failure**
   - `steps_per_epoch = batches_per_epoch // gradient_accumulation_steps` in [`ssi/train.py:164`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:164).
   - If `batches_per_epoch < grad_accum_steps`, then `steps_per_epoch == 0` and `n_epochs = ceil(max_steps / 0)` crashes at [`ssi/train.py:165`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:165).

5. **Possible divide-by-zero in gradient scaling**
   - `scale_grads(model, 1 / num_tokens_step)` in [`ssi/train.py:182`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:182).
   - If all labels in an accumulation window are ignored, `num_tokens_step == 0` and this blows up.

6. **Unconditional `torch.cuda.empty_cache()` each batch is unsafe/incorrect**
   - Called every iteration in [`ssi/train.py:249`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:249).
   - On non-CUDA device this is wrong; on CUDA it severely hurts performance and can increase fragmentation pressure.

7. **Token-type accounting is internally inconsistent**
   - Type counts include pad tokens if pad ID falls in a token range (likely `special_text`) in [`ssi/train.py:99-101`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:99).
   - But `"total"` excludes pad.
   - So `sum(type_counts)` and `total` are not comparable, and `special_text` is inflated by padding.

8. **Dev-loss type and aggregation correctness issues**
   - In eval, `num_tokens_dev` starts as `int` but is incremented by tensor in [`ssi/eval.py:30-31`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/eval.py:30), so return type may become tensor despite annotation `-> float`.
   - No cross-rank aggregation is done, so distributed eval loss is local-shard loss, not global.

9. **No validation for critical positive integer config fields**
   - Missing checks for `gradient_accumulation_steps`, `max_steps`, `log_interval`, `eval_steps`, `save_steps` > 0.
   - Zero values can cause modulo/division errors at [`ssi/train.py:181`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:181), [`ssi/train.py:206`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:206), [`ssi/train.py:236`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/train.py:236).

10. **`compute_loss` mutates input batch (`pop("labels")`)**
    - In [`ssi/loss.py:15`](/mnt/scratch-artemis/anilkeshwani/speech-integration/ssi/loss.py:15).
    - Currently works, but this is fragile and can cause subtle bugs if any caller needs labels later.

## Recommendations

1. **Fix checkpoint schema first**
   - Save and load the same keys (`steps_run` + `epochs_run`), and keep legacy fallback for old checkpoints.
   - Save/load RNG state and intra-epoch dataloader position if exact resume matters.

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

9. **Add regression tests**
   - Resume from saved checkpoint roundtrip.
   - Gradient normalization with shifted labels.
   - Small dataset where `len(dataloader) < grad_accum`.
   - All-ignored-label batch behavior.
