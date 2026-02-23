from __future__ import annotations

import math
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "tests" / "regression" / "matrix.yaml"
THRESHOLDS_PATH = ROOT / "tests" / "regression" / "thresholds.yaml"

LOSS_LINE_RE = re.compile(r"Global Step\s+(\d+).+?Loss:\s+([0-9]*\.?[0-9]+)")


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _required_env_missing(required_env: list[str]) -> list[str]:
    return [name for name in required_env if not os.getenv(name)]


def _build_command(matrix: dict, row: dict) -> list[str]:
    defaults = matrix["defaults"]
    overrides = list(defaults.get("base_overrides", []))
    overrides.append(f"data={row['data_config']}")
    overrides.extend(row.get("hydra_overrides", []))
    return [sys.executable, row["script"], *overrides]


def _collect_loss_by_step(output: str) -> dict[int, float]:
    losses: dict[int, float] = {}
    for line in output.splitlines():
        match = LOSS_LINE_RE.search(line)
        if not match:
            continue
        step = int(match.group(1))
        loss = float(match.group(2))
        losses[step] = loss
    return losses


def _assert_thresholds(losses: dict[int, float], profile: dict, row_id: str) -> None:
    checkpoints = profile.get("checkpoints", [])
    tolerances = profile.get("tolerances", {})
    abs_tol = float(tolerances.get("abs", 0.0))
    rel_tol = float(tolerances.get("rel", 0.0))
    for checkpoint in checkpoints:
        step = int(checkpoint["step"])
        expected = float(checkpoint["train_loss"])
        if step not in losses:
            raise AssertionError(f"{row_id}: missing loss for step={step}; captured steps={sorted(losses.keys())}")
        observed = losses[step]
        if not math.isclose(observed, expected, rel_tol=rel_tol, abs_tol=abs_tol):
            raise AssertionError(
                f"{row_id}: step={step} expected={expected:.6f} observed={observed:.6f} "
                f"(rel_tol={rel_tol}, abs_tol={abs_tol})"
            )


@pytest.mark.gpu
@pytest.mark.regression
@pytest.mark.heavy
def test_training_loss_regression(matrix_row: dict) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU is required for this regression test.")

    matrix = _load_yaml(MATRIX_PATH)
    thresholds = _load_yaml(THRESHOLDS_PATH)

    row_id = matrix_row["id"]
    if not matrix_row.get("enabled", False):
        pytest.skip(f"{row_id}: row is disabled in matrix.yaml")
    if not matrix_row.get("runnable", False):
        blocked_by = matrix_row.get("blocked_by", [])
        pytest.skip(f"{row_id}: not runnable (blocked_by={blocked_by})")

    required_env = matrix["defaults"].get("required_env", [])
    missing_env = _required_env_missing(required_env)
    if missing_env:
        pytest.skip(f"{row_id}: missing required env vars: {', '.join(missing_env)}")

    profile_name = matrix_row["threshold_profile"]
    profiles = thresholds.get("profiles", {})
    if profile_name not in profiles:
        raise AssertionError(f"{row_id}: threshold profile not found: {profile_name}")
    profile = profiles[profile_name]

    require_thresholds = os.getenv("SSI_REQUIRE_THRESHOLDS", "0") == "1"
    if not profile.get("active", False):
        if require_thresholds:
            raise AssertionError(
                f"{row_id}: threshold profile '{profile_name}' is inactive. "
                "Capture reference and set active=true before enforcing."
            )
        pytest.skip(f"{row_id}: threshold profile '{profile_name}' is inactive")

    env = os.environ.copy()
    env.update(matrix["defaults"].get("base_env", {}))
    timeout = int(os.getenv("SSI_TEST_TIMEOUT_SECONDS", "7200"))
    cmd = _build_command(matrix, matrix_row)
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=timeout)
    output = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        raise AssertionError(f"{row_id}: training command failed with code {proc.returncode}\n{output}")

    losses = _collect_loss_by_step(output)
    if not losses:
        raise AssertionError(f"{row_id}: no 'Loss:' lines parsed from output.")

    if os.getenv("SSI_CAPTURE_REFERENCE", "0") == "1":
        checkpoints = [{"step": step, "train_loss": value} for step, value in sorted(losses.items())]
        raise AssertionError(
            f"{row_id}: reference capture mode enabled.\n"
            f"Use these checkpoints for profile '{profile_name}':\n{checkpoints}"
        )

    _assert_thresholds(losses, profile, row_id)
