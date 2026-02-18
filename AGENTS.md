# Repository Guidelines

## Project Structure & Module Organization
`ssi/` contains the core Python package (training loop, model wiring, data loaders, tokenization, and utilities).  
`scripts/` provides executable entry points such as `train_cpt.py`, `train_sft.py`, `generate.py`, and `wer.py`.  
`conf/` stores Hydra configs (`cpt.yaml`, `sft.yaml`, `generate.yaml`, plus dataset config groups under `conf/data/`).  
`prompt_templates/` and `snippets/` hold reusable templates and operational helpers.  
`third_party/torchtune/` is a git submodule; treat it as vendored code and keep local-project changes in `ssi/`, `scripts/`, and `conf/` unless submodule updates are intentional.

## Build, Test, and Development Commands
- `git submodule update --init --recursive --progress`: initialize vendored dependencies.
- `conda create -n ssi-dev python=3.10.6 -y && conda activate ssi-dev`: create the supported runtime.
- `pip install -e .[dev]`: editable install with formatter/lint tooling.
- `pip install --no-dependencies git+https://github.com/anilkeshwani/speech-text-alignment.git`: required alignment dependency.
- `python scripts/train_cpt.py ...` / `python scripts/train_sft.py ...`: run training with Hydra overrides.
- `python scripts/generate.py model=<checkpoint_dir>`: generate transcripts.
- `python scripts/wer.py <path/to/generations.jsonl>`: compute WER.

## Coding Style & Naming Conventions
Use Python 3.10.6, 4-space indentation, and explicit imports.  
Format with Black (`line-length = 120`) and sort imports with isort (`profile = black`).  
Naming: `snake_case` for files/functions/variables, `PascalCase` for classes, and concise Hydra config names matching existing patterns (for example, `sft/mls-mimi-srvq_0`).  
Run `black ssi scripts && isort ssi scripts` before opening a PR.

## Testing Guidelines
There is currently no first-party pytest suite at repository root. For every change, run fast CLI smoke checks:
- `python scripts/train_sft.py --help`
- `python scripts/generate.py --help`
- `python scripts/wer.py --help`

If you modify `third_party/torchtune/`, run targeted tests inside that submodule (`pytest tests/...`) and report what was exercised.

## Commit & Pull Request Guidelines
Follow the existing history style: short, imperative, sentence-case commit subjects (for example, `Add plans/ directory`).  
Keep commits scoped to one concern.  
PRs should include: purpose, key file/config changes, exact reproduction command(s), and result impact (loss/WER deltas, sample output, or WandB link).  
Do not commit checkpoints, datasets, generated artifacts, or secrets.
