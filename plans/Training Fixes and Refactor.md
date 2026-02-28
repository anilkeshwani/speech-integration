# Training Critique (Initial Review)

## High Severity
1. ~~Stops only the inner loop; training can continue past `max_steps`.~~
2. ~~`num_tokens_step` becomes a tensor, causing formatting/logging errors.~~
3. Remainder batches never stepped when `batches_per_epoch % gradient_accumulation_steps != 0`.
4. Potential divide-by-zero if a batch has all labels ignored (zero tokens).

## Medium / Low Severity
5. Token-type counters are cumulative but logged like per-step values (misleading).
6. `steps_per_epoch` can be zero when `batches_per_epoch < gradient_accumulation_steps`.
7. `torch.cuda.empty_cache()` each epoch can hurt throughput; consider gating for debug only.
