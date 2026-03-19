# Feature: Standalone Sample-Wise Generation

## Motivation

The current generation pipeline (`scripts/generate.py`) is tightly coupled to dataset-driven evaluation. It requires:
- A Hydra config resolving to a specific SFT dataset split
- An experiment directory structure to resolve output paths
- Training config inference for speech parameters (n_dsus, deduplicate)
- The full `SFTDataset` → `DataLoader` → vLLM pipeline

This means there is **no way to generate from an arbitrary prompt** — every generation run requires a dataset config. For research iteration, we need the ability to quickly probe a model checkpoint with ad-hoc prompts: a raw text string, a text string with modality tokens, or a speech-token sequence rendered from a template.

### Prior art (deleted)

Two deleted scripts previously provided related functionality:

1. **`evaluation/generation.py`** (deleted in `06cb8bc`, 2025-06-03) — loaded test data from JSONL, rendered Jinja2 templates with speech tokens and modality token variables (`MODALITY_TOKEN_SPEECH`, `MODALITY_TOKEN_TEXT`, `speech_tokens`), passed the rendered text prompts to vLLM, and wrote structured JSONL output. This was the closest thing to sample-wise generation but still required a JSONL input file.

2. **`scripts/render_via_prompt.py`** (deleted in `b9be043`, 2025-05-19) — a data preprocessing utility that applied Jinja2 templates to JSONL samples, rendering speech tokens and modality tokens into a templated text field. It wrote the rendered output to a new JSONL file (not a generation script itself, but a prompt preparation step).

Both scripts shared:
- A `prompt_templates/` directory of `.jinja` files loaded via `jinja2.FileSystemLoader`
- Template variables: `speech_tokens` (PUA-encoded DSU string), `MODALITY_TOKEN_SPEECH`, `MODALITY_TOKEN_TEXT`, and optionally `text`
- Automatic `.jinja` extension appending when not provided by the user

The `prompt_templates/` directory (7 `.jinja` files) still exists in the repository but is orphaned — no code loads from it.

## Requirements

### Core: generate from arbitrary prompts without a dataset config

The module must support at minimum these prompt input modes:

1. **Raw text string** — e.g. `"The capital of France is"`. Tokenized and sent directly to the model.
2. **Raw token IDs** — a pre-tokenized list of integer token IDs. Sent to the model without further tokenization.
3. **Jinja2 template + variables** — a template (file path or inline string) rendered with user-supplied variables, then tokenized. Template variables should include the speech/modality token constants used in this project.

### Prompt template system

- Templates should be Jinja2 `.jinja` files, consistent with the prior art.
- The renderer must supply the standard project variables automatically: `MODALITY_TOKEN_SPEECH`, `MODALITY_TOKEN_TEXT` (from `sardalign.constants`).
- User-supplied variables (e.g. `speech_tokens`, `text`) are passed at render time.
- Templates may be referenced by name (resolved from a known directory) or by absolute/relative file path.

### Model and tokenizer loading

- Accept a model checkpoint path (the same paths used by the current `scripts/generate.py`).
- Use the project's `setup_llama3_tokenizer` for tokenization.
- Speech parameters (n_dsus, deduplicate) must be specifiable directly — no implicit inference from training configs.

### Generation parameters

- Expose vLLM `SamplingParams` (temperature, top_p, top_k, max_tokens, repetition_penalty, stop tokens, n sequences per prompt).
- Sensible defaults for interactive use: greedy decoding (temperature=0), single sequence (n=1), reasonable max_tokens.

### Output

- Print generated text to stdout by default (for interactive use).
- Optionally write structured JSONL output (prompt, prompt_token_ids, generated text, token_ids, stop_reason, metrics) for reproducibility and downstream analysis.

### Interface

- Usable as a **CLI script** with clear arguments (not Hydra — this should be lightweight and not require a config directory).
- Also importable as a **Python function** for use in notebooks or other scripts.

## Non-requirements (out of scope)

- Batch evaluation over datasets — that is the job of the existing `scripts/generate.py` pipeline.
- WER or other metric computation — that belongs in the evaluation module.
- Multi-GPU / distributed generation — single-GPU vLLM is sufficient for ad-hoc probing.
- Integration with W&B or experiment tracking — this is a lightweight tool.

## Open questions

- **Where should prompt templates live?** The current `prompt_templates/` directory is at the repo top level, which is unusual for assets consumed by source code. Options include: `ssi/templates/`, `conf/templates/`, or keeping the current location. This decision is pending.
- **Should we retain the existing 7 templates?** Most are trivial test prompts. We may want to curate a smaller set of genuinely useful templates or start fresh.
- **Should the script support JSONL input as a batch of prompts?** The deleted `evaluation/generation.py` operated on JSONL files. We could support this as an optional mode without making it required, bridging the gap between fully ad-hoc and fully dataset-driven generation.
