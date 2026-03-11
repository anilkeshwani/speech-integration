# Test Spec: Checkpoint Save and Load

Tests the round-trip correctness of `save_checkpoint()` → `resume_training_state()` and the key schema introduced by the B1 fix.

**GPU required:** No. All tests operate on CPU tensors and temporary directories.

**Location:** `tests/test_checkpoint.py` ✓ implemented

---

## Fixtures

### `tmp_checkpointer`
A `FullModelHFCheckpointer` instance pointed at a temporary output directory, with `recipe_checkpoint` set to the `recipe_state.pt` that `save_checkpoint` will produce.

Construction requires a real `config.json` and at least one model shard file.  Use a symbolic link or copy of the Llama 3.2 1B `config.json` plus a single-shard random-weight safetensor file as the minimal checkpoint input.  Mark this fixture `scope="module"` to avoid rebuilding it for every test.

> **Alternative (simpler):** Skip `FullModelHFCheckpointer` entirely for the key-schema tests. Construct `ckpt_dict` by hand (a plain `dict`) and pass it directly to `resume_training_state()`. Use a real checkpointer only for the on-disk round-trip test.

### `minimal_ckpt_dict`
```python
{
    SEED_KEY:        SEED,           # canonical seed value
    EPOCHS_KEY:      2,
    GLOBAL_STEP_KEY: 150,
    OPTIMIZER_KEY:   {"state": {}, "param_groups": []},  # empty but structurally valid
    MODEL_KEY:       {"weight": torch.zeros(4, 4)},
}
```

---

## Test Cases

### T-CKP-1: `resume_training_state` returns correct values from a well-formed dict

```
Given  minimal_ckpt_dict
When   resume_training_state(ckpt_dict) is called
Then   returns (2, 150, {"state": {}, "param_groups": []})
       i.e. (EPOCHS_KEY value, GLOBAL_STEP_KEY value, OPTIMIZER_KEY value)
```

### T-CKP-2: Seed mismatch raises `ValueError`

```
Given  minimal_ckpt_dict with SEED_KEY set to SEED + 1 (wrong seed)
When   resume_training_state(ckpt_dict) is called
Then   raises ValueError with message containing "seed"
```

### T-CKP-3: Missing `GLOBAL_STEP_KEY` raises `KeyError`

```
Given  minimal_ckpt_dict with GLOBAL_STEP_KEY removed
When   resume_training_state(ckpt_dict) is called
Then   raises KeyError
```

### T-CKP-4: Missing `EPOCHS_KEY` raises `KeyError`

```
Given  minimal_ckpt_dict with EPOCHS_KEY removed
When   resume_training_state(ckpt_dict) is called
Then   raises KeyError
```

### T-CKP-5: Missing `OPTIMIZER_KEY` raises `KeyError`

```
Given  minimal_ckpt_dict with OPTIMIZER_KEY removed
When   resume_training_state(ckpt_dict) is called
Then   raises KeyError
```

### T-CKP-6: `save_checkpoint` writes `GLOBAL_STEP_KEY`, not `STEPS_KEY`

```
Given  a FullModelHFCheckpointer with a valid output directory
When   save_checkpoint(model_state_dict, optimizer_state_dict, epoch=2, global_step=150, seed=SEED)
       is called
Then   recipe_state.pt loaded from disk contains key "global_step" with value 150
And    recipe_state.pt does NOT contain key "steps_run"
```

This is the regression guard for the B1 bug: confirms the canonical key is written correctly.

### T-CKP-7: On-disk round-trip — save then resume returns identical state

```
Given  a FullModelHFCheckpointer with a valid output directory
When   save_checkpoint(..., epoch=3, global_step=200, seed=SEED) is called
And    recipe_state.pt is loaded from the output directory
And    resume_training_state(loaded_dict) is called
Then   returned (epochs_run, global_step, optimizer_state) == (3, 200, optimizer_state_dict)
```

---

## Notes

- Tests T-CKP-1 through T-CKP-5 require no filesystem access — use a plain dict.
- Tests T-CKP-6 and T-CKP-7 require a real `FullModelHFCheckpointer`.  The minimal viable checkpoint input is:
  - `config.json` — copy from `ssi/constants.py:LLAMA_3_2_1B_BASE_DIR`
  - A single-shard safetensor file containing at least one tensor (can be random weights)
  - Use `pytest`'s `tmp_path` fixture for the output directory
- `FullModelHFCheckpointer` currently expects a `recipe_checkpoint` path to decide whether to resume.  For T-CKP-6 and T-CKP-7, set `recipe_checkpoint` to `None` at construction, then point it at the file written by `save_checkpoint` for loading.
- The optimizer state used in tests can be an empty-but-structurally-valid dict (`{"state": {}, "param_groups": []}`); we are testing key schema, not optimizer correctness.

---

## Running Tests Regularly

### Who should run what

The test suite splits naturally into two tiers based on runtime and infrastructure requirements:

| Tier | Tests | Runtime | Dependencies |
|------|-------|---------|--------------|
| Fast (pure-dict) | `test_resume_training_state_*` (5 tests) | < 1 s | None beyond the package |
| Disk | `test_save_recipe_state_*`, `test_save_and_resume_*` (2 tests) | ~20 s | `LLAMA_3_2_1B_BASE_DIR` on disk |

### Recommended cadence

**On every commit / PR — CI pipeline (GitHub Actions or equivalent)**

Run the full suite. The disk tests should be run here if the runner has access to the model weights (e.g. a self-hosted Artemis runner). If not, run the fast tier only and rely on developer machines for the disk tier.

Suggested CI step:

```yaml
- name: Run checkpoint tests
  run: uv run pytest tests/test_checkpoint.py -v
```

**On every local commit — pre-commit hook**

The fast tier is cheap enough to run as a pre-commit hook. Add a `local` hook to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: checkpoint-tests
      name: Checkpoint unit tests
      language: system
      entry: uv run pytest tests/test_checkpoint.py -k "not disk" -q
      pass_filenames: false
      always_run: true
```

The `-k "not disk"` selector excludes the two disk tests. Alternatively, mark the disk tests with `@pytest.mark.slow` and use `-m "not slow"`.

**Periodic full run on the cluster**

If CI runners do not have access to the Llama 3.2 1B weights, schedule a periodic Slurm job (e.g. nightly or weekly) to run the full suite on a node that does:

```bash
srun --partition cpu --time=00:05:00 \
    uv run pytest tests/test_checkpoint.py -v
```

### Keeping the suite cheap

- The disk tests call `save_recipe_state()` directly rather than `save_checkpoint()`, avoiding a full Llama 3.2 1B weight load. Keep this pattern for any future disk-touching tests.
- If the model directory moves, update `LLAMA_3_2_1B_BASE_DIR` in `ssi/constants.py`; the skip guard in the test file will handle the rest automatically.
