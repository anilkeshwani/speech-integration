# Removed: Adapter and LoRA Checkpoint Support

Adapter/LoRA checkpoint support was removed because this research project uses full-model continued pre-training and supervised fine-tuning only. This document records what was removed, how it worked, and how to restore it if needed.

## What was removed

### `ssi/checkpoint.py` — FullModelHFCheckpointer

**Imports:** `ADAPTER_CONFIG_FNAME`, `ADAPTER_MODEL_FNAME`, `REPO_ID_FNAME` from `torchtune.training.checkpointing._utils`.

**Constructor (`__init__`):**
- `adapter_checkpoint: Path | str | None = None` parameter
- `self.adapter_checkpoint` instance variable
- `repo_id` loading from `{checkpoint_dir}/.repo_id.json` — existed solely to supply `base_model_name_or_path` to `tune_to_peft_adapter_config()`
- `self.repo_id` instance variable
- Logging: `"Resuming adapter from checkpoint: ..."`

**`load_checkpoint()`:**
- Conditional adapter weight loading via `safe_torch_load(self.adapter_checkpoint)` into `converted_state_dict[training.ADAPTER_KEY]`

**`save_adapter_weights()` — entire method:**
Saved adapter weights in two formats:
1. Raw torchtune format as `adapter_model.pt` (for resuming training)
2. PEFT-converted format as `adapter_model.safetensors` or `.bin` (for HF compatibility), via `convert_weights.tune_to_peft_adapter_weights()`

Included a TODO noting that ideally only one format should be saved, but `peft_to_tune` (the reverse conversion) did not exist.

**`save_adapter_config()` — entire method:**
Converted and saved adapter configuration (LoRA rank, target modules, etc.) as `adapter_config.json` via `convert_weights.tune_to_peft_adapter_config()`.

**`save_recipe_state()`:**
`exclude_keys` simplified from `(MODEL_KEY, ADAPTER_KEY, ADAPTER_CONFIG)` to `(MODEL_KEY,)`.

**`_save_checkpoint()`:**
- `adapter_only: bool` parameter removed
- `adapter_only` validation and conditional model-skip logic removed
- Conditional calls to `save_adapter_weights` and `save_adapter_config` removed (these were marked `# NOTE not used currently`)

**`save_checkpoint()`:**
- `adapter_only: bool = False` parameter removed
- Passthrough of `adapter_only` to `_save_checkpoint` removed

### `ssi/constants.py`

```python
ADAPTER_KEY: str = training.ADAPTER_KEY  # adapter weights such as LoRA weights
assert ADAPTER_KEY == "adapter"
```

### `conf/training.yaml`

```yaml
adapter_checkpoint: null
```

### `plans/Refactor - Separate Model and Training State Checkpoints.md`

- Adapter save logic line removed from `_save_checkpoint` responsibilities
- Adapter checks removed from `save_model_checkpoint` code snippet
- Adapter checkpoint handling bullet removed from Out of Scope section

---

## How adapter/LoRA support worked

### Loading
1. `adapter_checkpoint` path passed to `FullModelHFCheckpointer.__init__`
2. `load_checkpoint()` loaded the adapter state dict via `safe_torch_load` and stored it under `training.ADAPTER_KEY` ("adapter") in the returned dict
3. The calling recipe would load these weights into the model's LoRA layers

### Saving
1. `_save_checkpoint` checked for `training.ADAPTER_KEY` in the state dict
2. If present, `save_adapter_weights` saved two copies:
   - Raw torchtune-format weights as `adapter_model.pt`
   - PEFT-converted weights as `adapter_model.safetensors` (via `tune_to_peft_adapter_weights`)
3. If `training.ADAPTER_CONFIG` was present, `save_adapter_config` saved the config as `adapter_config.json` (via `tune_to_peft_adapter_config`)
4. `adapter_only` flag allowed saving only adapter weights without the full model

### PEFT conversion functions (from `torchtune.models.convert_weights`)
- `tune_to_peft_adapter_weights()` — renames LoRA weight keys from torchtune naming (e.g. `layers.0.attn.q_proj.lora_a.weight`) to PEFT/HF naming
- `tune_to_peft_adapter_config()` — converts torchtune adapter config to PEFT's `adapter_config.json` format, including `base_model_name_or_path`

---

## How to restore LoRA support

### Recommended: Separate AdapterCheckpointer class

Keep adapter I/O separate from the model checkpointer:

```python
class AdapterCheckpointer:
    def __init__(self, adapter_config: dict[str, Any]):
        self.adapter_config = adapter_config  # rank, alpha, target_modules, dropout

    def load_adapter(self, path: Path) -> dict[str, torch.Tensor]:
        return safe_torch_load(path)

    def save_adapter(self, adapter_state_dict: dict[str, torch.Tensor], output_dir: Path) -> Path:
        ...  # save in torchtune and/or PEFT format

    def save_adapter_config(self, output_dir: Path) -> Path:
        ...  # save adapter_config.json for PEFT/HF compatibility
```

### Required torchtune imports
- `torchtune.training.ADAPTER_KEY` — state dict key for adapter weights
- `torchtune.training.ADAPTER_CONFIG` — state dict key for adapter config
- `torchtune.models.convert_weights.tune_to_peft_adapter_weights()` — weight conversion
- `torchtune.models.convert_weights.tune_to_peft_adapter_config()` — config conversion
- `torchtune.training.checkpointing._utils.ADAPTER_MODEL_FNAME` — standard filename
- `torchtune.training.checkpointing._utils.ADAPTER_CONFIG_FNAME` — standard filename

### Configuration changes
- Add `adapter_checkpoint` to config YAML (or new adapter-specific config)
- Add LoRA hyperparameters: rank, alpha, target_modules, dropout
- Add adapter-specific training flags (e.g. freeze base model weights)

### Training loop changes
- Model initialization: apply LoRA to target modules, optionally freeze base weights
- Optimizer: include only adapter parameters if training adapter-only
- State dict: extract adapter weights separately from `model.state_dict()`
