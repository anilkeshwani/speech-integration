# Plan: Simplify Checkpoint Directory Structure

## Context

Training runs produce deeply nested checkpoint directories that are hard to browse and query:
```
experiments/Llama-3.2-1B-5000-dsus-sft/rosy-salad-547-id_9h3htysc/checkpoints/epoch_0/global_step_10000/
```
The W&B random name is uninformative, `checkpoints/` is constant, and `epoch_N/` is misleading (training is step-based). We want a flat, self-describing structure:
```
experiments/sft_hubert_5000dsus_dedup_9h3htysc/step_10000/
```

## Target Structure

```
{experiments_root_dir}/
  {stage}_{encoder}_{n_dsus}dsus_{flags}_{wandb_id}/   <- self-describing run dir
    step_{M}/                                            <- step-level checkpoint
      model-*.safetensors, config.json, tokenizer.model, etc.
    recipe_state.pt                                      <- training resume state
    torchtune_config.yaml                                <- resolved config snapshot
```

## Changes

### Phase 1: Core logic (atomic commit)

**1. `ssi/constants.py`** — Add encoder short name mapping

```python
ENCODER_SHORT_NAMES: dict[str, str] = {
    "hubert_large_ll60k": "hubert",
    "speechtokenizer": "speechtokenizer",
    "mimi": "mimi",
}
```
Keys match the `speech_encoder` field returned by `parse_hf_repo_id()` in `ssi/utils.py:95-109`.

**2. `ssi/checkpoint.py`** — Two changes:

**(a) Rewrite `resolve_checkpointer_output_dir()` (line 558-563):**
- Add helpers `_get_encoder_short_name(data_source)` and `_build_run_dir_name(cfg, wandb_id)`
- Build name: `{cfg.config_name}_{encoder_short}_{n_dsus}dsus_{flags}_{wandb_id}`
- Use `cfg.experiments_root_dir` directly (NOT `cfg.output_dir`, which includes model name)
- Drop the trailing `/checkpoints` — it was always redundant
- Flags: `dedup`/`nodedup` always; `nomodtok` only when `use_modality_tokens=False` (True is default)

**(b) Step directory naming — DONE:**
`save_model_checkpoint()` (formerly `save_checkpoint()`) now defaults to `self.output_dir / f"step_{global_step}"`. The old `epoch_{epoch}/global_step_{global_step}` nesting was removed as part of the checkpoint schema v1 refactor. Epoch is derived from `global_step // steps_per_epoch` — no information lost.

### Phase 2: Downstream consumers

All downstream code must handle **both old and new formats** during transition.

**3. `ssi/utils.py` — `_parse_model_path()` (line 55-77)**
- Add format detection: new format has 2 parts (`run_dir/step_M`), legacy has 5 parts with `checkpoints` at index 2
- Parse new run dir name to extract stage, encoder, n_dsus, flags, wandb_id
- Keep legacy parsing as fallback

**4. `scripts/generate.py` — Two locations:**
- `_resolve_gen_output_dir()` (line 37-59): Currently checks `parts[-3] == "checkpoints"`. Add branch for new format: `run_dir/step_M` → `run_dir/generations/step_M`
- `main()` (line 152-156): Currently looks for config at `Path(cfg.model).parents[1]` (2 levels up past `epoch_N/`). New format: `Path(cfg.model).parent` (1 level up). Try new path first, fall back to legacy.

**5. `scripts/plot_wandb_losses.py` (line ~62)** — Update glob from `global_step_*` to match both `step_*` and `global_step_*`

**6. `snippets/check_missing_generations.py` (line 32-39)** — Look for step dirs in run_dir directly (new) and under `checkpoints/` (legacy)

**7. `snippets/generation_launcher.sh` (line 23)** — Update `find` pattern to match both `global_step*` and `step_*`

**8. `snippets/impute_config.py` (line ~42)** — Check for `torchtune_config.yaml` at run_dir level (new) and under `checkpoints/` (legacy)

### Phase 3: Config comment update

**9. `conf/training.yaml` (line 37)** — Update comment to reflect new auto-resolution format

### Files NOT changed

- `conf/common.yaml` — `output_dir` stays as-is (still used for `wandb.log_dir`). The resolver uses `cfg.experiments_root_dir` directly.
- `ssi/metric_logging.py` — Already saves to `config.checkpointer.output_dir` which will be the run dir. Works as-is.
- Data config YAMLs — No `short_name` field needed. The encoder short name is derived from the HF source string via `parse_hf_repo_id()` + `ENCODER_SHORT_NAMES`.

## Key Design Decisions

1. **Code-level mapping over YAML `short_name`**: 3 encoders, 5 data config files. A dict in `constants.py` avoids duplication and inconsistency.
2. **`experiments_root_dir` not `output_dir`**: `output_dir` includes `${base_model_name}-${config_name}` which we're folding into the run dir name. Use `experiments_root_dir` to avoid double-encoding.
3. **Exception-based flags**: Only include `nomodtok` when modality tokens are off. Default state = no flag. Keeps names shorter.
4. **Dual-format support in all consumers**: Old experiments on disk are not renamed. Code handles both formats until legacy support is dropped.

## Verification

1. Run a short CPT and SFT training job (few steps) — verify:
   - Run dir name matches `{stage}_{encoder}_{n_dsus}dsus_{flags}_{wandb_id}`
   - Step dirs use `step_M` naming
   - `recipe_state.pt` and `torchtune_config.yaml` at run dir level
2. Point `generate.py` at a new-format checkpoint — verify generation output dir resolution
3. Test `_parse_model_path()` with both old and new format paths
4. Test `_build_run_dir_name()` with all encoder/flag combinations
