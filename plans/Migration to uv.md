# Migration to uv

## Current status
- Project dependencies are managed by `uv` (`pyproject.toml` + `uv.lock`).
- `sardalign` is added as a Git dependency and pinned to a specific commit in `pyproject.toml`.

## Bootstrap command
- Standard bootstrap:

```bash
uv sync --extra dev
```

## Important caveat for fresh environments
- `sardalign` currently pulls a submodule that uses an SSH URL (`git@github.com:...`).
- On machines without GitHub SSH configured, Git submodule fetch may fail during install.

### Temporary workaround (used here)
- For add/update operations, force Git to rewrite `git@github.com:` URLs to HTTPS for the command:

```bash
GIT_CONFIG_COUNT=1 \
GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf \
GIT_CONFIG_VALUE_0=git@github.com: \
uv add "sardalign @ git+https://github.com/anilkeshwani/speech-text-alignment.git"
```

## Long-term fix
- Update the submodule URL(s) in `speech-text-alignment` from SSH to HTTPS.
- Then bump the pinned `sardalign` revision in this repo.
- After that, `uv sync --extra dev` should bootstrap cleanly without the rewrite workaround.
