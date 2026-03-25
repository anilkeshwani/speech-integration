# Tokenizer Architecture: tiktoken vs HF `tokenizer.json`

## Context

The LLaMA 3.2 1B checkpoint on HuggingFace ships with **two independent tokenizer representations** of the same vocabulary. SSI extends one of them (tiktoken) and ignores the other (HF JSON). This document explains both formats, why the current approach was chosen, what the alternative would look like, and what the options are going forward.

---

## 1. The Two Tokenizer Formats

### 1a. `original/tokenizer.model` -- tiktoken BPE format

- **What it is**: A plain-text file where each line is `<base64-encoded-token> <merge-rank>`. 128,000 lines for the base vocabulary.
- **What it does NOT contain**: The 256 special tokens (`<|begin_of_text|>`, `<|end_of_text|>`, etc.). These are added programmatically by `setup_llama3_tokenizer()` (`ssi/tokenizer/__init__.py:29-31`).
- **Who reads it**: tiktoken (via `load_tiktoken_bpe()`), which is wrapped by torchtune's `TikTokenBaseTokenizer`, which is wrapped by our `Llama3TokenizerPUA`.
- **Size**: ~2.1 MiB (base), ~2.2 MiB (extended with 5,000 DSUs + 2 modality tokens).
- **History**: Llama 3 switched from sentencepiece (Llama 2) to tiktoken. Meta kept the `.model` filename, which is confusing because sentencepiece also uses `.model`. They are completely different formats -- this is a tiktoken file, not a sentencepiece file.

### 1b. `tokenizer.json` -- HuggingFace `tokenizers` library format

- **What it is**: A self-contained JSON file used by the HuggingFace `tokenizers` Rust library. Contains everything needed to tokenize text:
  - `model.vocab`: 128,000 entries (the base BPE vocabulary as a `{token_string: id}` dict)
  - `model.merges`: 280,147 merge rules (as `"token_a token_b"` strings)
  - `added_tokens`: 256 entries (the special tokens, IDs 128,000-128,255, each marked `special: true`)
  - `pre_tokenizer`: regex-based split pattern + byte-level fallback
  - `post_processor`: template processing (adds BOS)
  - `decoder`: byte-level decoder
  - `model.ignore_merges: true` (important flag -- see section 3)
- **Who reads it**: HuggingFace `AutoTokenizer.from_pretrained()`, and **vLLM by default** when initializing its tokenizer.
- **Size**: 8.7 MiB on the HuggingFace Hub. Locally only 1.9 KiB (stripped by `tune download` to just the pre/post-processor config, no vocab).
- **Current status in SSI**: Not used. Explicitly excluded from extended model directories (`scripts/extend_llama3_2.py:92-93`).

### 1c. Key structural difference

| Aspect | `tokenizer.model` (tiktoken) | `tokenizer.json` (HF) |
|--------|------------------------------|------------------------|
| Format | Plain text, base64 lines | JSON |
| Vocabulary | 128,000 BPE merges only | 128,000 vocab + 280K merges + 256 special tokens |
| Special tokens | Added programmatically | Embedded in `added_tokens` array |
| Pre-tokenizer regex | Set in Python code (`CL100K_PATTERN`) | Embedded in JSON (`pre_tokenizer.pretokenizers[0].pattern.Regex`) |
| Consumed by | tiktoken (Python/Rust) | HF `tokenizers` (Rust) |
| PUA support | Requires `Llama3TokenizerPUA` monkeypatch | Would require patching the JSON regex |

---

## 2. What SSI Currently Does

### Extension pipeline (`scripts/extend_llama3_2.py`)

1. **Extends `original/tokenizer.model`** by appending new base64-encoded lines for DSU tokens (PUA-encoded) and modality tokens
2. **Does NOT touch `tokenizer.json`** -- explicitly excluded from the output directory
3. Special tokens shift: `<|begin_of_text|>` moves from 128,000 to `base_vocab_size + n_dsus + n_modality_tokens` (e.g. 133,002 for 5,000 DSUs + 2 modality tokens)

### Training (`ssi/train.py`)

- Loads the extended `tokenizer.model` via `setup_llama3_tokenizer()` -> `Llama3TokenizerPUA`
- Tokenizes all data (text + DSU PUA characters) through this single tokenizer
- No HF tokenizer involved

### Generation (`scripts/generate.py`)

- Loads the same extended `tokenizer.model` via `setup_llama3_tokenizer()` -> `Llama3TokenizerPUA`
- Tokenizes input externally, producing `prompt_token_ids`
- Passes pre-tokenized IDs to vLLM as `TokensPrompt(prompt_token_ids=...)`
- **vLLM is initialized with `skip_tokenizer_init=True`** -- it never loads or uses a tokenizer
- Decoding of generated token IDs back to text is done by SSI's tokenizer, not vLLM's

### The PUA monkeypatch (`ssi/tokenizer/monkeypatch.py`)

The standard tiktoken regex `CL100K_PATTERN` does not match Unicode Private Use Area codepoints (`\p{Co}`). Since DSU tokens are encoded as PUA characters (U+E000 onwards via `dsu2pua()`), they would be consumed by the wrong regex branch and split or garbled.

`CL100K_PATTERN_PUA` fixes this with two changes:
1. Adds `\p{Co}` to the exclusion sets in character classes (so PUA chars don't match text/number branches)
2. Adds `|\p{Co}` as a final alternative (so each PUA char matches as a single token)

This is enforced at import time: `assert CL100K_PATTERN_PUA != CL100K_PATTERN`

---

## 3. The `tokenizer.json` Alternative: What It Would Take

### 3a. The regex problem is identical

Verified empirically:

```
HF tokenizer.json pre-tokenizer regex == torchtune CL100K_PATTERN?  True
HF tokenizer.json includes \p{Co}?                                  False
```

The `pre_tokenizer.pretokenizers[0].pattern.Regex` field in `tokenizer.json` is **character-for-character identical** to the `CL100K_PATTERN` that `Llama3TokenizerPUA` exists to fix. So extending `tokenizer.json` without also patching this regex would produce the same PUA tokenization bug.

### 3b. What extending `tokenizer.json` would require

To produce a valid extended `tokenizer.json` that vLLM (or HF `AutoTokenizer`) could load natively:

1. **Patch the pre-tokenizer regex** to include `\p{Co}` -- the same logical fix as `CL100K_PATTERN_PUA`, but applied to the JSON string field
2. **Add DSU tokens to `added_tokens`** as non-special tokens, each with:
   - `content`: the PUA character string (e.g. `"\ue000"` for DSU 0)
   - `id`: the token ID (128,000 onwards)
   - `special: false`
3. **Add modality tokens** similarly (if enabled)
4. **Shift the 256 special tokens** in `added_tokens` -- update all their `id` fields
5. **Leave `model.vocab` and `model.merges` unchanged** -- the base BPE vocab doesn't change
6. **Note**: `model.ignore_merges: true` means the HF tokenizer treats `added_tokens` as atomic (no merge rules applied to them). This is correct behaviour for DSU tokens -- they should never be merged or split. This flag works in our favour.

### 3c. What you'd gain

- **vLLM could load the tokenizer natively**: no `skip_tokenizer_init=True`, no external tokenization, no `TokensPrompt` wrapper
- **`AutoTokenizer.from_pretrained()` would work**: useful if sharing the model externally or using HF `pipeline()`
- **Rust-based tokenization**: the HF `tokenizers` library runs in Rust, which is faster than Python tiktoken -- though tokenization is not a bottleneck in practice
- **Cleaner vLLM integration**: could pass raw text strings to `llm.generate()` instead of pre-tokenized IDs

### 3d. What you'd lose or risk

- **Two tokenizer files to maintain in sync**: Training uses torchtune which reads `original/tokenizer.model` via tiktoken. Generation would use `tokenizer.json` via HF tokenizers. Any divergence between the two = silent correctness bug (training and inference disagree on token IDs).
- **The regex fix is in a JSON string field**: Harder to test and easier to get wrong than the Python-level monkeypatch with its explicit `assert CL100K_PATTERN_PUA != CL100K_PATTERN`. No equivalent assertion mechanism in JSON.
- **Loss of single source of truth**: Currently there is exactly one tokenizer codepath (`original/tokenizer.model` -> `Llama3TokenizerPUA`) used everywhere. The `tokenizer.json` route would introduce a second codepath for inference.
- **torchtune dependency**: Unless torchtune adds native `tokenizer.json` support, training would still need the tiktoken file. You'd be maintaining both indefinitely.

---

## 4. Options

### Option A: Stay the course (recommended for research)

**Keep the current architecture**: extend only `original/tokenizer.model`, use `Llama3TokenizerPUA` everywhere, pass pre-tokenized IDs to vLLM with `skip_tokenizer_init=True`.

**Pros**:
- Single source of truth for tokenization
- Same tokenizer codepath at training and inference -- no risk of divergence
- Already working, tested, and understood
- The `skip_tokenizer_init=True` pattern is not a hack -- it is the documented vLLM approach for custom tokenizers

**Cons**:
- External tokenization means SSI must handle the tokenize-before-vLLM step
- Cannot use vLLM's string-based API (must use `TokensPrompt`)
- Model is not plug-and-play for external consumers (no working `AutoTokenizer`)

**When this is the right choice**: Research and internal experimentation where you control both training and inference, and correctness is paramount.

### Option B: Extend `tokenizer.json` alongside `tokenizer.model`

**Add an `extend_tokenizer_json()` function** to the extension pipeline that produces a valid HF-format `tokenizer.json` with DSU tokens, modality tokens, shifted special tokens, and the patched PUA regex. Ship both files in extended model directories.

**Pros**:
- vLLM can load the tokenizer natively
- Model directories become fully HF-compatible (`AutoTokenizer.from_pretrained()` works)
- Enables simpler vLLM integration (pass strings, not token IDs)

**Cons**:
- Two tokenizer files to maintain in lockstep
- Need a validation step that asserts equivalence between the tiktoken and JSON tokenizers (e.g. encode a suite of test strings through both and compare)
- More complex extension pipeline
- If the test suite doesn't catch a divergence, you have a silent correctness bug

**When this is the right choice**: If you plan to distribute the model to external users, or if the `skip_tokenizer_init` / `TokensPrompt` pattern becomes a friction point for scaling up the generation pipeline.

### Option C: Switch entirely to `tokenizer.json` (high effort, not recommended now)

**Replace the tiktoken codepath entirely** with HF `tokenizers`. This would mean:
- Training uses `AutoTokenizer` instead of torchtune's `Llama3Tokenizer`
- Extension modifies `tokenizer.json` as the single source of truth
- `original/tokenizer.model` becomes a read-only archive of the base tokenizer

**Pros**:
- True single source of truth (but now it's the JSON file, not the tiktoken file)
- Full HF ecosystem compatibility

**Cons**:
- Requires decoupling from torchtune's tokenizer stack (significant refactor)
- torchtune's `Llama3Tokenizer` is deeply integrated: `encode_message()`, `encode_dialog()`, prompt templates all depend on it
- Would need to verify that the HF tokenizer produces byte-identical output to the tiktoken one for all training data
- High risk, high effort, low marginal benefit for research use

**When this is the right choice**: Only if migrating away from torchtune entirely, or if building a production system where HF compatibility is a hard requirement.

### Option D: Generate `tokenizer.json` as a build artefact (read-only, not authoritative)

**At extension time, auto-generate `tokenizer.json` from the extended `tokenizer.model`** for convenience, but treat it as a derived artefact -- not the source of truth. Add a clear warning that it should not be hand-edited.

**Pros**:
- Gets you `AutoTokenizer` compatibility for free
- `tokenizer.model` remains the single source of truth
- vLLM can optionally use native tokenizer init
- No sync problem because the JSON is always regenerated from the tiktoken file

**Cons**:
- Need to write a `tiktoken_to_hf_json()` conversion function (non-trivial but tractable)
- Need to embed the PUA-patched regex in the generated JSON
- Need to verify the generated tokenizer produces identical output (same test suite as Option B)
- Generated file may drift from what HF `transformers` expects if their format evolves

**When this is the right choice**: A pragmatic middle ground if you want HF compatibility without giving up the single-source-of-truth property. Good candidate for a future improvement once the core research is stable.

---

## 5. Recommendation

**For the current research phase: Option A (stay the course).**

The current architecture is sound. The `skip_tokenizer_init=True` + `TokensPrompt` pattern is not a workaround -- it is the correct way to use vLLM with a custom tokenizer. The single-source-of-truth property (one tokenizer file, one codepath, same at training and inference) is the most important invariant for research correctness.

**When the research matures**: Option D (generate `tokenizer.json` as a derived artefact) is the natural next step. It would give you HF compatibility for model sharing without compromising the single-source-of-truth. The key prerequisite is a test harness that encodes a diverse set of strings (text, DSUs, mixed, modality tokens, edge cases) through both tokenizers and asserts identical output.

---

## Appendix: File Locations

| Component | File |
|-----------|------|
| tiktoken extension | `ssi/extend_llama3_2/__init__.py` :: `extend_tiktoken()` |
| PUA regex patch | `ssi/tokenizer/monkeypatch.py` :: `CL100K_PATTERN_PUA` |
| Tokenizer setup | `ssi/tokenizer/__init__.py` :: `setup_llama3_tokenizer()` |
| Extension entry point | `scripts/extend_llama3_2.py` |
| Generation (vLLM) | `scripts/generate.py` :: `generate()` |
| Base tokenizer.model | `models/base/Llama-3.2-1B/original/tokenizer.model` |
| Full HF tokenizer.json (cached) | `~/.cache/huggingface/hub/models--meta-llama--Llama-3.2-1B/.../tokenizer.json` |
