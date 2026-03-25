Now I have a thorough picture. Let me explain each parameter and the implementation in full.

---

## CPT Configs (`conf/data/cpt/`)

Both CPT configs share the same structure — one for HuBERT tokens, one for SpeechTokenizer tokens. They configure `TextCompletionDataset`.

### Top-level split keys: `train` / `dev`
Consumed in `ssi/trainer.py` (`Trainer._setup_data`):
```python
data_train, sampler_train = setup_text_completion_data(cfg.data.train, tokenizer)
data_dev, sampler_dev   = setup_text_completion_data(cfg.data.dev, tokenizer)
```
Each split becomes a separate `DataLoader` + `DistributedSampler`.

---

### `dataset` block → unpacked as `**cfg_dataset.dataset` into `TextCompletionDataset.__init__` (`ssi/data/__init__.py:39`)

| Parameter | Value | Where used |
|---|---|---|
| `source` | HF dataset path | `load_dataset(source, split=split)` in `cpt.py:81` |
| `split` | `train` / `validation` | `load_dataset(source, split=split)` in `cpt.py:81` |
| `sequence_type` | `interleaved` | `CompletionSequenceType(sequence_type)` → selects `interleave()` as `prompt_fn` (`cpt.py:94-99`) |
| `interleave_kwargs` | dict | `partial(interleave, **interleave_kwargs)` bakes 4 kwargs into the prompt fn |
| `deduplicate` | `${speech.deduplicate}` | passed through `_prepare_sample` → `prompt_fn(..., deduplicate=...)` |
| `use_modality_tokens` | `${speech.use_modality_tokens}` | passed through `_prepare_sample` → `prompt_fn(..., use_modality_tokens=...)` |
| `add_eos` | `True` | `self._tokenizer.encode(..., add_eos=self.add_eos)` in `cpt.py:129` |
| `tokenized_key` | `null` | falls back to `sardalign.constants.TOKENIZED_KEY` |
| `alignment_start_time_key` | `null` | falls back to `sardalign.constants.ALIGNMENT_START_TIME_KEY` |
| `alignment_end_time_key` | `null` | falls back to `sardalign.constants.ALIGNMENT_END_TIME_KEY` |
| `speech_tokens_key` | `null` | falls back to `sardalign.constants.SPEECH_TOKENS_KEY` |

#### `interleave_kwargs` sub-parameters (used in `cpt.py:152-190`):

| Sub-param | HuBERT | SpeechTokenizer | Where used |
|---|---|---|---|
| `sampling_rate` | `16000` | `24000` | `times_to_dsu_idxs((t_start, t_end), sampling_rate, downsampling_ratio)` → converts alignment timestamps to speech token indices |
| `downsampling_ratio` | `320` | `320` | same call; 16000/320=50fps (HuBERT) or 24000/320=75fps (SpeechTokenizer) |
| `mean_seq_len_tokens` | `39.43` | `39.43` | `n` in `Binomial(n, p)` used to generate span boundaries in **text** token space via `get_span_idxs_binomial(int(mean_seq_len_tokens), binom_prob, len(tokens))` |
| `binom_prob` | `0.1` | `0.1` | `p` in `Binomial(n, p)` → expected span size ≈ 39×0.1 = 3.9 text tokens |

`mean_seq_len_tokens=39.43` being identical for both is correct — it parameterises binomial span sizes in *text* token space (word-pieces), which is independent of the speech encoder.

---

### `dataloader` block → `DataLoader` in `ssi/data/__init__.py:56-62`

| Parameter | Train | Dev | Where used |
|---|---|---|---|
| `batch_size` | `16` | `16` | `DataLoader(..., batch_size=...)` |
| `drop_last` | `True` | `False` | `DataLoader(..., drop_last=...)` — True on train to avoid shape issues with compiled models + flex attention |

### `shuffle` → `DistributedSampler(..., shuffle=...)` (`__init__.py:54`)
True for train, False for dev. Correct.

### `packed: False`
If True, raises `NotImplementedError` immediately in `setup_text_completion_data`.

---

## SFT Configs (`conf/data/sft/`)

Three splits: `train`, `dev`, `test`. Configures `SFTDataset`.

### `dataset` block → `SFTDataset(model_tokenizer=tokenizer, **cfg_dataset.dataset)` (`__init__.py:76`)

| Parameter | Value | Where used |
|---|---|---|
| `source` | HF dataset path | `load_dataset(source, **load_dataset_kwargs)` in `sft.py:131`; `split` lands in `load_dataset_kwargs` |
| `split` | `train`/`validation`/`test` | captured by `**load_dataset_kwargs` → passed to `load_dataset` |
| `inference` | `False` (train/dev), `True` (test) | `self._inference` → `InputOutputToMessages(..., inference=...)` and `model_tokenizer(..., inference=...)`. Controls: (a) empty output for generation mode, (b) whether EOS is appended by the Llama3 tokenizer |
| `deduplicate` | `${speech.deduplicate}` | `InputOutputToMessages.__call__`: removes consecutive duplicate DSU tokens if True |
| `filter_fn` | `null` | `None` → no filtering applied |
| `train_on_input` | `True` | `Message(role="user", masked=not train_on_input)` → `masked=False` → model trains on speech token input too (intentional regularisation to prevent forgetting) |
| `column_map` | `{input: speech_tokens, output: transcript}` | `sample[column_map["input"]]` and `sample[column_map["output"]]` in `InputOutputToMessages.__call__` |
| `new_system_prompt` | ASR instruction string | prepended as `Message(role="system", masked=True)` |
| `image_dir` | `null` | `None` → no image loading |
| `use_modality_tokens` | `${speech.use_modality_tokens}` | if True: wraps speech span as `MODALITY_TOKEN_SPEECH + sp_span + MODALITY_TOKEN_TEXT` in `InputOutputToMessages.__call__:308` |

---

## Bug / Issue Findings

### 🐛 BUG 1 (latent): Custom key parameters silently ignored in `TextCompletionDataset`

In `cpt.py:85-92`, `tokenized_key`, `alignment_start_time_key`, `alignment_end_time_key`, `speech_tokens_key` are resolved from `null` to defaults — but then **discarded**. They are never stored as instance variables and never passed to `interleave()` or `concatenate_speech_text()`, which hard-code `sample[TOKENIZED_KEY]`, `sample[ALIGNMENT_START_TIME_KEY]`, etc., directly.

If anyone sets a non-null value for these keys in the config, it will be silently ignored. This is currently safe because all configs use `null`, but the feature is broken.

**Relevant code:** `cpt.py:85-103` — keys resolved but not threaded into `prompt_fn`.

---

~~### 🐛 BUG 2 (dead code): Redundant `ignore_idx` assignment~~
~~Removed. See D1 in Training Cleanup Tasks.md.~~

---

### 🐛 BUG 3 (dead code): Second `packed` check is unreachable

In both `setup_text_completion_data` and `setup_sft_data`, the structure is:

```python
if cfg_dataset.get("packed", False):
    raise NotImplementedError(...)  # ← always raises if packed=True
dataset = ...
if cfg_dataset.get("packed", False):  # ← unreachable; first block always raised
    dataset = pack_dataset(...)
    collate_fn = padded_collate_packed
```

The second `if packed` block can never execute. The guard should be removed from the top and the packing logic implemented properly.

---

~~### ⚠️ CONCERN: SpeechTokenizer `sampling_rate: 24000`~~
~~Confirmed as a bug. SpeechTokenizer sampling_rate should be 16000 (not 24000). Both sampling_rate and downsampling_ratio replaced with `???` mandatory placeholders. Full analysis in `plans/Bugfix - CPT Interleave Config Sampling Parameters.md`.~~

---

~~### ⚠️ Minor concern: Module-level shared PRNG~~
~~Fixed by per-sample deterministic RNG: module-level `PRNG` deleted, each sample's interleaving is a pure function of `(seed, epoch, sample_index)`. See C6 in Training Cleanup Tasks and D6 in Checkpointing - Consolidated Plan.~~

