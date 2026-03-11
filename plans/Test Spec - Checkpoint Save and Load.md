# Test Spec: Checkpoint Save and Load

Tests the round-trip correctness of `save_checkpoint()` → `resume_training_state()` and the key schema introduced by the B1 fix.

**GPU required:** No. All tests operate on CPU tensors and temporary directories.

**Proposed location:** `tests/test_checkpoint.py`

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
