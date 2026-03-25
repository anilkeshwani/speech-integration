"""Shared fixtures and markers for the test suite.

pytest auto-discovers this file — no import needed. Markers and fixtures
defined here are available to all test modules under tests/.
"""

from pathlib import Path

import pytest
import torch

from ssi.constants import TORCHTUNE_EXTENDED_MODELS_DIR


# ---------------------------------------------------------------------------
# Model path discovery
# ---------------------------------------------------------------------------

EXTENDED_MODEL_DIR = TORCHTUNE_EXTENDED_MODELS_DIR / "Llama-3.2-1B-5000-dsus"

# Also check ~/models/extended/ (for environments without /mnt/scratch-artemis)
_LOCAL_EXTENDED = Path.home() / "models" / "extended" / "Llama-3.2-1B-5000-dsus"
if _LOCAL_EXTENDED.exists():
    EXTENDED_MODEL_DIR = _LOCAL_EXTENDED


def _has_extended_model() -> bool:
    return EXTENDED_MODEL_DIR.exists() and (EXTENDED_MODEL_DIR / "model.safetensors.index.json").exists()


def _has_gpu() -> bool:
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def extended_model_dir() -> Path:
    """Path to the extended Llama 3.2 1B model with 5000 DSUs."""
    return EXTENDED_MODEL_DIR


@pytest.fixture
def has_gpu() -> bool:
    return _has_gpu()


@pytest.fixture
def has_extended_model() -> bool:
    return _has_extended_model()
