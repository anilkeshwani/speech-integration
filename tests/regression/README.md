# GPU Regression Tests

This suite enforces exact-loss regression checks for speech training configurations.

## Required environment

- `SSI_TEST_MODEL_DIR`: path to the model directory containing:
  - `model.safetensors`
  - `config.json`
  - `original/tokenizer.model`

Optional:

- `SSI_TEST_OUTPUT_DIR` (default: `/tmp`)
- `SSI_TEST_TIMEOUT_SECONDS` (default: `7200`)
- `SSI_REQUIRE_THRESHOLDS=1` to fail if threshold profile is inactive
- `SSI_CAPTURE_REFERENCE=1` to print observed losses for baseline capture

## Selection

Run smoke tier:

```bash
python scripts/tests_gpu.py
```

Run a tokenizer subset:

```bash
python scripts/tests_gpu.py --tokenizer hubert --tokenizer mimi
```

Run by approach:

```bash
python scripts/tests_gpu.py --approach sft
```

Run a specific matrix row:

```bash
python scripts/tests_gpu.py --matrix-id sft_mimi_mls
```

## Baseline capture workflow

1. Enable a matrix row in `tests/regression/matrix.yaml`.
2. Execute with reference capture:

```bash
SSI_CAPTURE_REFERENCE=1 python scripts/tests_gpu.py --matrix-id <row_id> -s
```

3. Copy emitted checkpoints into `tests/regression/thresholds.yaml` under the row profile.
4. Set `active: true` for that profile.
5. Populate `tests/regression/reference_manifest.yaml` metadata.
