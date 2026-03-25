# LLaMA 3.2 1B Checkpoint Structure & Embedding Extension: Complete Analysis

## 1. HuggingFace Repository vs Local Checkpoint

### 1a. What's on HuggingFace (`meta-llama/Llama-3.2-1B`)

Verified via `huggingface_hub.list_repo_tree()` on 2026-03-19 (repo SHA: `4e20de362430cd3b72f300e6b0f18e50e7166e08`, last modified 2024-10-24):

```
meta-llama/Llama-3.2-1B (HuggingFace Hub)
├── config.json                         (843 B)     # HF model config
├── generation_config.json              (185 B)     # HF generation params
├── model.safetensors                   (2.30 GiB)  # HF safetensors weights
├── special_tokens_map.json             (301 B)     # HF special token strings
├── tokenizer_config.json               (49.3 KiB)  # HF tokenizer config (chat template, etc.)
├── tokenizer.json                      (8.7 MiB)   # Full HF tokenizers-library JSON (vocab + merges + config)
├── original/
│   ├── consolidated.00.pth             (2.30 GiB)  # Meta's original PyTorch checkpoint
│   ├── tokenizer.model                 (2.1 MiB)   # tiktoken BPE file (128,000 merges)
│   └── params.json                     (220 B)     # Meta's original params
├── .gitattributes                      (1.5 KiB)
├── LICENSE.txt                         (7.5 KiB)
├── README.md                           (40.3 KiB)
└── USE_POLICY.md                       (5.9 KiB)
```

### 1b. What's on disk locally (`models/base/Llama-3.2-1B/`)

Downloaded via `tune download` (torchtune):

```
Llama-3.2-1B/ (local)
├── config.json                         (843 B)     ✓ matches HF
├── generation_config.json              (185 B)     ✓ matches HF
├── model.safetensors                   (2.30 GiB)  ✓ matches HF
├── special_tokens_map.json             (301 B)     ✓ matches HF
├── tokenizer_config.json               (49.3 KiB)  ✓ matches HF
├── tokenizer.json                      (1.9 KiB)   ⚠ DIFFERENT (see below)
├── original/
│   ├── tokenizer.model                 (2.1 MiB)   ✓ matches HF
│   └── params.json                     (220 B)     ✓ matches HF
│   [consolidated.00.pth]                            ✗ NOT PRESENT (see below)
├── .gitattributes                      (1.5 KiB)   ✓ matches HF
├── LICENSE.txt                         (7.5 KiB)   ✓ matches HF
├── README.md                           (40.3 KiB)  ✓ matches HF
└── USE_POLICY.md                       (5.9 KiB)   ✓ matches HF
```

### 1c. Disparities Explained

**Disparity 1: `original/consolidated.00.pth` (2.30 GiB) -- absent locally**

This is Meta's original PyTorch checkpoint format (a single `.pth` file serialised with `torch.save`). It contains the same model weights as `model.safetensors` but in Meta's key naming convention (e.g. `tok_embeddings.weight` rather than HF's `model.embed_tokens.weight`). The `tune download` command from torchtune excludes this file because the HF safetensors format is preferred -- it's memory-mappable, faster to load, and is what `FullModelHFCheckpointer` reads. **No impact on correctness**: the weights are identical, just in a different serialisation.

**Disparity 2: `tokenizer.json` (8.7 MiB on HF vs 1.9 KiB locally)**

On HuggingFace, `tokenizer.json` is the full HuggingFace `tokenizers` library JSON format containing the complete BPE vocabulary table (128K tokens + merges + special tokens), pre-tokenizer, post-processor, and decoder configs -- hence 8.7 MiB.

Locally, the file is only 1.9 KiB and contains just the pre/post-processor and decoder config (no vocab table). This is because `tune download` produces a stripped-down version -- torchtune uses `original/tokenizer.model` (the tiktoken BPE file) directly, not the HF JSON tokenizer. **No impact on our codebase**: SSI never reads `tokenizer.json`. Our tokenizer path is exclusively through `original/tokenizer.model` -> `load_tiktoken_bpe()` -> `Llama3TokenizerPUA`. Furthermore, `tokenizer.json` is explicitly excluded when creating extended model directories (`scripts/extend_llama3_2.py:92-93`).

### 1d. Why each file exists

| File | Purpose | Who reads it |
|------|---------|-------------|
| `config.json` | Defines architecture (`num_hidden_layers=16`, `hidden_size=2048`, etc.) + **`vocab_size=128256`**, **`bos_token_id=128000`**, **`eos_token_id=128001`** | HF transformers, vLLM, our `FullModelHFCheckpointer` |
| `model.safetensors` | The actual weight tensors including `model.embed_tokens.weight` (shape `[128256, 2048]`) and `lm_head.weight` (tied) | Checkpoint loader via `safe_torch_load()` |
| `original/consolidated.00.pth` | Meta's original PyTorch checkpoint (same weights, different key format). Not downloaded by `tune download`. | Meta's original inference code (not used by SSI) |
| `original/tokenizer.model` | tiktoken BPE file: each line is `<base64-encoded-token> <merge-rank>`. 128,000 lines = 128,000 BPE merges | Our `setup_llama3_tokenizer()` via `load_tiktoken_bpe()` |
| `original/params.json` | Meta's format: `dim=2048`, `n_heads=32`, etc. + `vocab_size=128256` | Our `extend_params()` |
| `generation_config.json` | Default sampling params + `bos_token_id`, `eos_token_id` | vLLM, our `extend_generation_config()` |
| `tokenizer.json` | HF tokenizers-library JSON (full vocab on HF, stripped locally) | Not used by SSI (excluded from extended models) |
| `tokenizer_config.json` | HF tokenizer config (chat template, added_tokens, etc.) | Not used by SSI (excluded from extended models) |
| `special_tokens_map.json` | HF special token string mappings | Copied to extended models but not read by SSI code |

## 2. The Vocabulary Arithmetic

Base LLaMA 3.2 1B:
```
128,000  BPE text tokens        (positions [0, 127999] in tokenizer.model)
+   256  special tokens          (LLAMA3_SPECIAL_TOKENS: <|begin_of_text|>, <|end_of_text|>, etc.)
= 128,256  total vocab_size      (matches config.json)
```

The 256 special tokens are **not** in `tokenizer.model`. They're added programmatically in `setup_llama3_tokenizer()` at `ssi/tokenizer/__init__.py:29-31`:
```python
special_tokens_dynamic = dict(
    zip(LLAMA3_SPECIAL_TOKENS, range(base_vocab_size, base_vocab_size + 256))
)
```

So in the base model:
- `<|begin_of_text|>` = ID 128,000
- `<|end_of_text|>` = ID 128,001

## 3. The Extension Process (`scripts/extend_llama3_2.py`)

Given `n_new_dsus=5000` and `use_modality_tokens=True`:

**Step-by-step execution order:**

1. **Load base checkpoint** via `FullModelHFCheckpointer` -- reads `model.safetensors`, converts HF keys to torchtune keys (`model.embed_tokens.weight` -> `tok_embeddings.weight`, etc.)

2. **Initialize torchtune model** via `simple_setup_model()` -- creates a `llama3_2_1b()` with base `vocab_size=128256`, loads weights

3. **`extend_model()`** -- the critical function (`ssi/extend_llama3_2/__init__.py:80-110`):
   ```
   Original embedding: [128,000 text | 256 special] = [128,256 x 2048]

   Step 1: Clone original embeddings
   Step 2: Split into base_vocab (128,000) and special_tokens (256)
   Step 3: Fit multivariate Gaussian to base_vocab embeddings (sigma_scaling=1e-5)
   Step 4: Sample 5,002 new vectors (5,000 DSUs + 2 modality tokens)
   Step 5: Concatenate: [128,000 text | 5,002 new | 256 special] = [133,258 x 2048]
   Step 6: Replace tok_embeddings with nn.Embedding.from_pretrained(concatenated)
   Step 7: Replace output with TiedLinear(tok_embeddings)
   Step 8: Assert base embeddings unchanged, special embeddings unchanged, size delta correct
   ```

4. **Save extended weights** -- converts back to HF format, writes `ft-model-00001-of-00001.safetensors` (~4.6 GiB, larger because the embedding/lm_head grew), copies non-weight files

5. **`extend_tiktoken()`** -- appends 5,002 new lines to `tokenizer.model` (in the output directory):
   - 5,000 lines for DSU tokens (PUA-encoded via `dsu2pua()`)
   - 2 lines for modality tokens (`MODALITY_TOKEN_TEXT`, `MODALITY_TOKEN_SPEECH`)
   - Each gets the next available merge rank

6. **`extend_config()`** / **`extend_params()`** / **`extend_generation_config()`** -- updates all JSON configs with new `vocab_size=133258` and new `bos_token_id=133002`, `eos_token_id=133003`

## 4. Extended Model on Disk

```
Llama-3.2-1B-5000-dsus/
├── config.json                        # vocab_size: 133258, bos: 133002, eos: 133003
├── generation_config.json             # bos: 133002, eos: 133003
├── ft-model-00001-of-00001.safetensors  # Extended weights (~4.6 GiB)
├── model.safetensors.index.json       # Weight map pointing to ft-model shard
├── special_tokens_map.json            # Copied from base
├── original/
│   ├── tokenizer.model                # 133,002 lines (128K + 5K DSUs + 2 modality)
│   └── params.json                    # vocab_size: 133258
└── torchtune_config.yaml              # Training config snapshot
```

**Critically: `tokenizer_config.json` and `tokenizer.json` are NOT copied** (explicitly excluded at `scripts/extend_llama3_2.py:92-93`). This is because the HF tokenizer JSON format isn't used -- only the tiktoken BPE file is.

## 5. The New Vocabulary Layout

```
Position 0-127,999:        Base BPE text tokens     (PRESERVED from pre-training)
Position 128,000-132,999:  5,000 DSU tokens         (NEWLY SAMPLED from Gaussian)
Position 133,000-133,001:  2 modality tokens         (NEWLY SAMPLED from Gaussian)
Position 133,002-133,257:  256 special tokens        (PRESERVED, but SHIFTED from 128,000-128,255)
```

**The shift is the key insight**: `<|begin_of_text|>` moves from 128,000 to 133,002. That's why `bos_token_id` and `eos_token_id` must be updated in all config files, and why `setup_llama3_tokenizer()` dynamically computes special token IDs from `base_vocab_size` (which now includes DSUs + modality tokens):

```python
# tokenizer/__init__.py:26-31
base_vocab_size = len(mergeable_ranks)  # 133,002 for extended model
special_tokens_dynamic = dict(
    zip(LLAMA3_SPECIAL_TOKENS, range(base_vocab_size, base_vocab_size + 256))
)
# So <|begin_of_text|> = 133,002, <|end_of_text|> = 133,003
```

## 6. Training: How Files Are Loaded

In `ssi/trainer.py` (`Trainer.setup()`), the flow is:

1. **Config**: `configllama3_2_1b.update_from_speech_cfg(cfg.speech)` sets `n_dsus=5000`, `modality_tokens=True` on the singleton config, so `vocab_size` property returns 133,258

2. **Tokenizer**: `setup_llama3_tokenizer(path=extended/original/tokenizer.model)` loads the **extended** BPE file (133,002 merges) and assigns special tokens starting at 133,002

3. **Checkpoint**: `FullModelHFCheckpointer(checkpoint_dir=extended_model_dir, checkpoint_files=[...])` loads the extended safetensors, converts HF->torchtune keys

4. **Model**: `setup_llama3_2_1b(llama_config=configllama3_2_1b, model_state_dict=...)` creates a torchtune `llama3_2` with `vocab_size=133258` and loads the extended state dict

5. **The model already has the extended embeddings baked in** -- there's no extension at training time. Training just loads the pre-extended checkpoint and fine-tunes all parameters (no freezing).

## 7. Generation: How Files Are Loaded

In `scripts/generate.py`:

1. **Auto-resolves** `n_dsus` and `deduplicate` from the training config saved alongside the checkpoint (`torchtune_config.yaml`)

2. **Tokenizer**: Same `setup_llama3_tokenizer()` loading the extended BPE file

3. **vLLM**: `LLM(model=cfg.model, skip_tokenizer_init=True)` -- vLLM reads `config.json` from the checkpoint directory, sees `vocab_size=133258`, and allocates the correct embedding dimensions. It reads the safetensors weight map and loads all shards. `skip_tokenizer_init=True` because the custom PUA tokenizer is used separately.

4. **Data**: `SFTDataset` with `inference=True` -- tokenizes speech input (DSU PUA tokens) into prompt token IDs, but doesn't append the assistant transcript or EOS. These go to vLLM as `TokensPrompt(prompt_token_ids=...)`.

5. **Stop tokens**: Set to `[tokenizer.eom_id, tokenizer.eot_id, tokenizer.eos_id]` -- all at the **shifted** positions (133,002+).

## 8. The PUA Tokenizer Monkeypatch

The reason `Llama3TokenizerPUA` exists: standard tiktoken's regex `CL100K_PATTERN` explicitly **excludes** `\p{Co}` (Private Use Area codepoints). Since DSU tokens are encoded as PUA characters, they'd be split or eaten without this fix.

`CL100K_PATTERN_PUA` adds `|\p{Co}` at the end, ensuring each PUA character matches as a **single token**.

## 9. Correctness Invariants (Assertions in the Code)

The codebase enforces these at extension time (`scripts/extend_llama3_2.py:134-138`):
- `tokenizer.vocab_size == 128000 + 256 + n_dsus + 2*use_modality_tokens`
- `len(model.tok_embeddings.weight) == tokenizer.vocab_size`

And inside `extend_model()` (`ssi/extend_llama3_2/__init__.py:103-105`):
- Base text embeddings (rows 0-127,999) are bitwise identical to original
- Special token embeddings (last 256 rows) are bitwise identical to original
- Size delta == exactly `n_dsus + 2*use_modality_tokens`

## 10. Weight Conversion Round-Trip

```
HF keys (on disk)              torchtune keys (in memory)
-----------------              ----------------------------
model.embed_tokens.weight  <-> tok_embeddings.weight
model.layers.0.self_attn.  <-> layers.0.attn.
model.norm.weight          <-> norm.weight
lm_head.weight             <-> output.weight (TiedLinear -> tok_embeddings.weight)
```

`convert_weights.hf_to_tune()` and `convert_weights.tune_to_hf()` handle this bidirectionally, using `num_heads`, `num_kv_heads`, `dim`, and `head_dim` from `config.json` to reshape attention weight matrices correctly.

## Summary

The system is architecturally sound. The tokenizer and model weights are extended in lockstep, with special tokens shifted to the end. All config files are updated consistently. Training loads pre-extended checkpoints directly. Generation uses vLLM which reads the HF-format checkpoint directory natively. The only custom piece needed at inference time is the PUA tokenizer monkeypatch.

## Key Source Files

| Component | File |
|-----------|------|
| Extension logic | `ssi/extend_llama3_2/__init__.py` |
| Extension script | `scripts/extend_llama3_2.py` |
| Model setup | `ssi/model.py` |
| Tokenizer setup | `ssi/tokenizer/__init__.py` |
| PUA monkeypatch | `ssi/tokenizer/monkeypatch.py` |
| Config schema | `ssi/llama_configs.py` |
| Checkpoint I/O | `ssi/checkpoint.py` |
| Training loop | `ssi/trainer.py` |
| Generation | `scripts/generate.py` |
| Constants | `ssi/constants.py` |
