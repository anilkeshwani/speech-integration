# Bugfix — CPT Interleave Config Sampling Parameters

> **Status: PARTIALLY MITIGATED.** The incorrect hardcoded values have been replaced with mandatory `???` placeholders so that Hydra will reject any CPT training run until verified values are explicitly provided. The correct values are documented below but have not yet been committed as defaults — they require manual verification before being set.

## Summary

The CPT interleaved sequence type uses `sampling_rate` and `downsampling_ratio` config values to convert word-level alignment timestamps (in seconds) into speech token indices. Two of the four tokenizer configs have incorrect values, causing the `times_to_dsu_idxs()` function to compute wrong span boundaries during interleaved sequence construction.

| Tokenizer | Config state | Correct `sampling_rate` | Correct `downsampling_ratio` | Correct token rate |
|---|---|---|---|---|
| HuBERT | `???` (was 16000/320 — correct) | 16000 | 320 | 50 Hz |
| SpeechTokenizer | `???` (was 24000/320 — **wrong**) | 16000 | 320 | 50 Hz |
| Mimi | `???` (was 24000/320 — **wrong**) | 24000 | 1920 | 12.5 Hz |
| FocalCodec | `???` (was 16000/320 — unverified) | 16000 | 320 | 50 Hz (unverified) |

---

## Bug 1: SpeechTokenizer `sampling_rate` is 24000, should be 16000

### Affected file

`conf/data/cpt/mls-speechtokenizer-rvq_0.yaml`

### Config state

Both `sampling_rate` and `downsampling_ratio` are currently `???` (mandatory placeholders). The previous incorrect values were `sampling_rate: 24000, downsampling_ratio: 320`.

### What was wrong

The alignment conversion `times_to_dsu_idxs()` would have computed:

```
token_index = time_in_seconds × 24000 / 320 = time_in_seconds × 75
```

But SpeechTokenizer produces tokens at 50 Hz (not 75 Hz), so computed indices would overshoot by 1.5x. For a word at t=10s, the code would compute token index 750 when the correct index is 500.

### Evidence

#### 1. Empirical token counts from the dataset

Streaming the first sample from both HuBERT and SpeechTokenizer datasets (same audio clip ID `4800_10003_000000`, duration 15.81s, 252960 samples at 16 kHz):

```
HuBERT:           790 tokens → 252960 / 790 = 320.2 → 50.0 tokens/s
SpeechTokenizer:  791 tokens → 252960 / 791 = 319.8 → 50.0 tokens/s
```

Both produce ~50 tokens/second from the same audio. The token rate is identical, confirming SpeechTokenizer operates at the same effective rate as HuBERT (16 kHz input, 320x downsampling).

#### 2. SpeechTokenizer model config

From the model checkpoint config at `config/spt_base_cfg.json` in the [SpeechTokenizer repository](https://github.com/ZhangXInFD/SpeechTokenizer):

```json
{
    "sample_rate": 16000,
    "strides": [8, 5, 4, 2]
}
```

- `sample_rate: 16000` — the model expects 16 kHz input
- `strides: [8, 5, 4, 2]` — product is 8×5×4×2 = 320, confirming `downsampling_ratio: 320`

The `model.sample_rate` property reads directly from this config (`speechtokenizer/model.py:38`).

#### 3. Tokenization script confirms no resampling

From [anilkeshwani/SpeechTokenizer](https://github.com/anilkeshwani/SpeechTokenizer), `scripts/mls.py:110–112`:

```python
if sr != model.sample_rate:
    warnings.warn(f"Audio {mls_id} has sample rate {sr} != {model.sample_rate}. Resampling.")
    wav = torchaudio.functional.resample(wav, sr, model.sample_rate)
```

MLS audio is 16 kHz. `model.sample_rate` is 16000. The condition `16000 != 16000` is false — **no resampling occurs**. The audio is processed at its native 16 kHz.

#### 4. HuggingFace model card

The HuggingFace model config at `fnlp/SpeechTokenizer/speechtokenizer_hubert_avg/config.json` contains both `"sampling_rate": 16000` and `"sample_rate": 16000`.

#### 5. SpeechTokenizer paper

"SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models" (arXiv 2308.16692). The GitHub README states: "A model operated at **16khz** on monophonic speech."

### Proposed fix

```yaml
# conf/data/cpt/mls-speechtokenizer-rvq_0.yaml
train:
  dataset:
    source: anilkeshwani/mls-speechtokenizer-rvq_0
    interleave_kwargs:
      sampling_rate: 16000  # SpeechTokenizer model sampling rate
dev:
  dataset:
    source: anilkeshwani/mls-speechtokenizer-rvq_0
    interleave_kwargs:
      sampling_rate: 16000
```

---

## Bug 2: Mimi `downsampling_ratio` inherits incorrect default of 320, should be 1920

### Affected file

`conf/data/cpt/mls-mimi-srvq_0.yaml` (previously inherited `downsampling_ratio: 320` from `conf/data/_cpt_base.yaml`; the base config dev split has since been fixed to `???`)

### Config state

Both `sampling_rate` and `downsampling_ratio` are currently `???` (mandatory placeholders). The previous incorrect values were `sampling_rate: 24000, downsampling_ratio: 320` (320 inherited from `_cpt_base.yaml`).

### What was wrong

The alignment conversion would have computed:

```
token_index = time_in_seconds × 24000 / 320 = time_in_seconds × 75
```

But Mimi produces tokens at 12.5 Hz. The correct computation is:

```
token_index = time_in_seconds × 24000 / 1920 = time_in_seconds × 12.5
```

Computed indices would overshoot by 6x. For a word at t=10s, the code would compute token index 750 when the correct index is 125.

### Evidence

#### 1. Empirical token counts from the dataset

Same audio clip as above (ID `4800_10003_000000`, 15.81s):

```
Mimi: 198 tokens → 15.81 / 198 = 0.0799 s/token = 12.5 tokens/s
```

Compare with HuBERT (50 tokens/s) and SpeechTokenizer (50 tokens/s) — Mimi produces 4x fewer tokens for the same audio, as expected from its higher compression ratio.

#### 2. Effective downsampling ratio from the data

Using the dataset's `num_samples` field (252960 at 16 kHz) and token count:

```
252960 samples / 198 tokens = 1277.6 samples/token (at 16 kHz)
```

Since the tokenization script resamples to 24 kHz before encoding:

```
252960 × (24000 / 16000) = 379440 samples at 24 kHz
379440 / 198 = 1916.4 ≈ 1920 samples/token (at 24 kHz)
```

This confirms `downsampling_ratio: 1920` when paired with `sampling_rate: 24000`.

#### 3. Mimi model architecture

Mimi's encoder uses convolutional blocks with strides `(4, 5, 6, 8)` plus a final stride-2 convolution:

```
4 × 5 × 6 × 8 × 2 = 1920
```

This is documented in the [Moshi paper](https://arxiv.org/abs/2410.00037) and confirmed by the HuggingFace Transformers [Mimi documentation](https://huggingface.co/docs/transformers/model_doc/mimi).

#### 4. Tokenization script confirms resampling to 24 kHz

From [anilkeshwani/mimi-tokenization](https://github.com/anilkeshwani/mimi-tokenization), `mls.py:49–51`:

```python
_mimi_feat_extractor: EncodecFeatureExtractor = AutoFeatureExtractor.from_pretrained(MIMI_REPO_ID)
MIMI_SR = _mimi_feat_extractor.sampling_rate
assert MIMI_SR == 24_000
```

And `mls.py:143–145`:

```python
if sr != MIMI_SR:
    wav = torchaudio.functional.resample(wav, sr, MIMI_SR)
```

MLS audio (16 kHz) is resampled to 24 kHz before Mimi encoding. The `sampling_rate: 24000` in the config correctly reflects this. Only the `downsampling_ratio` is wrong (320 instead of 1920).

#### 5. Mimi feature extractor confirms frame rate

The HuggingFace Mimi feature extractor reports a `frame_rate` that, combined with the 24 kHz sampling rate, implies a downsampling ratio of 1920: `24000 / 12.5 = 1920`.

### Proposed fix

```yaml
# conf/data/cpt/mls-mimi-srvq_0.yaml
train:
  dataset:
    source: anilkeshwani/mls-mimi-srvq_0
    interleave_kwargs:
      sampling_rate: 24000  # Mimi (Moshi) model sampling rate
      downsampling_ratio: 1920  # Mimi encoder strides: 4×5×6×8×2 = 1920
dev:
  dataset:
    source: anilkeshwani/mls-mimi-srvq_0
    interleave_kwargs:
      sampling_rate: 24000
      downsampling_ratio: 1920
```

---

## Note on FocalCodec

`conf/data/cpt/mls-focalcodec.yaml` has `sampling_rate: ???` and `downsampling_ratio: ???`. FocalCodec's 50 Hz variant (used in this project, `lucadellalib/focalcodec_50hz`) operates at 16 kHz with a downsampling factor of 320 (16000/320 = 50 Hz). The correct values are `sampling_rate: 16000, downsampling_ratio: 320`. The HF repo ID still has a `# TODO confirm` comment.

---

## How the bug manifests

The function `times_to_dsu_idxs()` in `ssi/data/cpt.py:181–185` converts word-level alignment timestamps to speech token indices:

```python
start_idx_hu, end_idx_hu = times_to_dsu_idxs(
    (align_t_starts[start_idx], align_t_ends[end_idx - 1]),
    sampling_rate,
    downsampling_ratio,
)
sp_tkns_spn = speech_tokens[start_idx_hu:end_idx_hu]
```

`times_to_dsu_idxs` (from `sardalign.utils.align`) computes:

```python
token_index = int(time_in_seconds * sampling_rate / downsampling_ratio)
```

With the incorrect values, the computed `start_idx_hu` and `end_idx_hu` overshoot the actual speech token positions. This causes:

1. **Misaligned spans**: Text and speech spans in interleaved sequences don't correspond to the same temporal region
2. **Index-out-of-bounds potential**: If the computed index exceeds `len(speech_tokens)`, the slice returns a truncated or empty span
3. **Corrupted training signal**: The CPT model learns incorrect text-speech associations

### Scope of impact

- **SFT configs are NOT affected** — SFT does not use `interleave_kwargs` or `times_to_dsu_idxs()`
- **CPT interleaved mode IS affected** — only when using SpeechTokenizer or Mimi data configs
- **CPT concatenated mode is NOT affected** — concatenation does not use alignment timestamps
- **HuBERT CPT is NOT affected** — its config values are correct

---

## Verification commands

After applying fixes, verify the token rates match empirical data:

```bash
uv run python -c "
from datasets import load_dataset

for name, src, expected_rate in [
    ('HuBERT', 'anilkeshwani/mls-hubert_large_ll60k-layer_22', 50.0),
    ('SpeechTokenizer', 'anilkeshwani/mls-speechtokenizer-rvq_0', 50.0),
    ('Mimi', 'anilkeshwani/mls-mimi-srvq_0', 12.5),
]:
    s = list(load_dataset(src, split='train', streaming=True).take(1))[0]
    rate = len(s['speech_tokens']) / s['duration']
    print(f'{name:20s}: {len(s[\"speech_tokens\"]):4d} tokens / {s[\"duration\"]:.2f}s = {rate:.1f} Hz (expected {expected_rate})')
"
```

Expected output:

```
HuBERT              :  790 tokens / 15.81s = 50.0 Hz (expected 50.0)
SpeechTokenizer     :  791 tokens / 15.81s = 50.0 Hz (expected 50.0)
Mimi                :  198 tokens / 15.81s = 12.5 Hz (expected 12.5)
```
