# Feature Spec: Packed Dataset Support (CPT and SFT)

## Background

Sequence packing concatenates multiple short samples into fixed-length sequences, eliminating padding waste and increasing GPU utilisation. The infrastructure already exists in torchtune (`PackedDataset`, `padded_collate_packed`) and the `pack_dataset` helper is already defined in `ssi/data/__init__.py`. Both `setup_text_completion_data` and `setup_sft_data` have stub code for the packed path, but each raises `NotImplementedError` before reaching it. The dead stub code should be replaced with a working implementation.

## Current State

`ssi/data/__init__.py`:

```python
# In both setup_text_completion_data and setup_sft_data:
if cfg_dataset.get("packed", False):
    raise NotImplementedError("Need to add a custom collate function to handle the PACKED case - not implemented.")
dataset = ...
if cfg_dataset.get("packed", False):   # <- dead: always False here
    dataset = pack_dataset(...)
    collate_fn = padded_collate_packed
else:
    ...
```

The `pack_dataset` helper:

```python
def pack_dataset(dataset, tokenizer, split_across_pack=False) -> PackedDataset:
    if tokenizer.max_seq_len is None:
        raise ValueError("PackedDataset requires a max_seq_len to be set on the tokenizer.")
    return PackedDataset(dataset, max_seq_len=tokenizer.max_seq_len, split_across_pack=split_across_pack)
```

## What Needs to Be Done

### 1. Understand `padded_collate_packed`

`torchtune.data.padded_collate_packed` expects each sample from `PackedDataset` to have:
- `tokens`: `list[int]`
- `labels`: `list[int]`
- `seq_lens`: `list[int]` — lengths of original sequences within the pack (used for block-diagonal attention mask)

Verify that `TextCompletionDataset.__getitem__` and `SFTDataset.__getitem__` return `seq_lens` when wrapped in `PackedDataset`, or whether `PackedDataset` injects it automatically. If not present, either:
  - `PackedDataset` adds it (check torchtune source), or
  - Each dataset's `__getitem__` needs to return it.

### 2. `additional_keys` incompatibility

The custom `padded_collate_sft` in `ssi/data/__init__.py` supports `additional_keys` (e.g. sample IDs for ASR eval). `padded_collate_packed` does not. Decide:
  - **Option A**: Prohibit `additional_keys` when `packed=True` (validate and raise early).
  - **Option B**: Write a `padded_collate_packed_with_keys` that extends `padded_collate_packed` to pass through additional keys.

Option A is simpler and likely sufficient for training; option B is needed if eval-time sample tracking matters.

### 3. CPT specifics

The existing comment notes: "Strictly this doesn't have to affect CPT since we don't need to change the collate function (as for SFT)". Verify whether `padded_collate_packed` is correct for CPT, or whether the default collate works. If the unpacked collate is fine for CPT, packing still requires switching to `padded_collate_packed` to handle `seq_lens` for attention masking (otherwise the model sees cross-sample attention).

The `split_across_pack` option (whether a sample can be split across two packs) should be exposed in the CPT dataset config. For SFT it is hardcoded to `False` in the stub — this seems correct (do not split a conversation across packs).

### 4. Config changes

`conf/training.yaml` (or the dataset sub-configs) should expose:
```yaml
dataset:
  packed: false          # enable sequence packing
  split_across_pack: false  # CPT only; SFT should always be false
```

`max_seq_len` must be set on the tokenizer when packing. Currently it is unclear whether this is set. Validate at `setup_*_data` time or in `validate_train_cfg`.

### 5. Implementation plan

1. Read `torchtune.datasets.PackedDataset.__getitem__` to confirm what keys it returns and whether `seq_lens` is injected automatically.
2. Confirm `TextCompletionDataset.__getitem__` and `SFTDataset.__getitem__` return a dict with `tokens` and `labels` (and optionally `seq_lens`).
3. Resolve the `additional_keys` question (Option A recommended initially).
4. Remove the early `NotImplementedError` raises and the dead second `if packed:` blocks.
5. Restructure both setup functions to a single `if packed / else` branch:

```python
dataset = XDataset(...)
if cfg_dataset.get("packed", False):
    dataset = pack_dataset(dataset, model_tokenizer, split_across_pack=...)
    collate_fn = padded_collate_packed
else:
    ignore_idx = CROSS_ENTROPY_IGNORE_IDX if loss_fn is None else loss_fn.ignore_index
    collate_fn = partial(padded_collate_sft, ...)
```

6. Add config validation: if `packed=True` and `additional_keys` is non-empty, raise a clear error.
7. Add config validation: if `packed=True` and `tokenizer.max_seq_len is None`, raise a clear error.
8. Smoke-test a short CPT and SFT run with `packed: true`.

## Open Questions

- Does `PackedDataset` inject `seq_lens` automatically, or must the underlying dataset return it?
- Is `split_across_pack` ever desirable for CPT in this project? (Speech–text interleaving may make cross-pack splitting semantically wrong.)
- Should the `DistributedSampler` still be used with `PackedDataset`, or does packing change the indexing contract?
