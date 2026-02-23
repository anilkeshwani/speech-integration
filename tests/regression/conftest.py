from __future__ import annotations

from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = ROOT / "tests" / "regression" / "matrix.yaml"


def _load_matrix() -> dict:
    with MATRIX_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption("--matrix-id", action="append", default=[], help="Run only specified matrix ids.")
    parser.addoption("--tokenizer", action="append", default=[], help="Run only rows for tokenizer(s).")
    parser.addoption("--approach", action="append", default=[], help="Run only rows for approach(es).")
    parser.addoption("--include-disabled", action="store_true", help="Include matrix rows marked enabled=false.")


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "gpu: requires CUDA GPU resources")
    config.addinivalue_line("markers", "regression: long-running regression tests")
    config.addinivalue_line("markers", "heavy: expensive end-to-end runs")


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "matrix_row" not in metafunc.fixturenames:
        return

    matrix = _load_matrix()
    rows = matrix["rows"]
    only_ids = set(metafunc.config.getoption("--matrix-id"))
    only_tokenizers = set(metafunc.config.getoption("--tokenizer"))
    only_approaches = set(metafunc.config.getoption("--approach"))
    include_disabled = bool(metafunc.config.getoption("--include-disabled"))

    selected = []
    for row in rows:
        if not include_disabled and not row.get("enabled", False):
            continue
        if only_ids and row["id"] not in only_ids:
            continue
        if only_tokenizers and row.get("tokenizer") not in only_tokenizers:
            continue
        if only_approaches and row.get("approach") not in only_approaches:
            continue
        marks = [getattr(pytest.mark, mark_name) for mark_name in row.get("marks", [])]
        selected.append(pytest.param(row, id=row["id"], marks=marks))

    metafunc.parametrize("matrix_row", selected)
