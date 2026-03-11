"""
Tests for checkpoint save/load round-trip correctness.

Covers the ``save_recipe_state()`` -> ``resume_training_state()`` round-trip and the key
schema introduced by standardising the per-step counter in recipe
checkpoints to ``GLOBAL_STEP_KEY`` (``"global_step"``).

GPU required
------------
No. All tests use CPU tensors and temporary directories.

Test structure
--------------
Pure-dict tests (no filesystem access):
    Operate on a hand-crafted ``ckpt_dict`` and call ``resume_training_state()`` directly.
    These verify the key schema and error-handling of the resume path in isolation, with no
    I/O.

Disk tests (require ``FullModelHFCheckpointer`` and ``LLAMA_3_2_1B_BASE_DIR``):
    Write a ``recipe_state.pt`` to a temporary directory via ``save_recipe_state()`` and
    reload it from disk, verifying both the on-disk key schema and the end-to-end
    save -> load -> resume round-trip. Skipped automatically when ``LLAMA_3_2_1B_BASE_DIR``
    is absent from the current machine.

    ``save_recipe_state()`` is called directly rather than the full ``save_checkpoint()``,
    which would require loading the Llama 3.2 1B model weights. We are testing key schema
    and resume correctness, not model weight serialisation.

Fixtures
--------
minimal_ckpt_dict:
    A minimal but structurally valid checkpoint dict with the canonical keys ``SEED_KEY``,
    ``EPOCHS_KEY``, ``GLOBAL_STEP_KEY``, ``OPTIMIZER_KEY``, and ``MODEL_KEY``. The
    optimizer state is empty but structurally valid; the model weights are a small
    zero-filled tensor. We are testing key schema, not optimizer or weight correctness.
"""

import pytest
import torch

from ssi.checkpoint import FullModelHFCheckpointer
from ssi.constants import (
    EPOCHS_KEY,
    GLOBAL_STEP_KEY,
    LLAMA_3_2_1B_BASE_DIR,
    MODEL_KEY,
    OPTIMIZER_KEY,
    SEED,
    SEED_KEY,
)
from ssi.train import resume_training_state


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_ckpt_dict():
    return {
        SEED_KEY: SEED,
        EPOCHS_KEY: 2,
        GLOBAL_STEP_KEY: 150,
        OPTIMIZER_KEY: {"state": {}, "param_groups": []},
        MODEL_KEY: {"weight": torch.zeros(4, 4)},
    }


# ---------------------------------------------------------------------------
# Pure-dict tests — no filesystem access
# ---------------------------------------------------------------------------


def test_resume_training_state_returns_correct_tuple(minimal_ckpt_dict):
    """Happy path: returns (epochs_run, global_step, optimizer_state) from a well-formed dict."""
    epochs, global_step, opt_state = resume_training_state(minimal_ckpt_dict)
    assert epochs == 2
    assert global_step == 150
    assert opt_state == {"state": {}, "param_groups": []}


def test_resume_training_state_wrong_seed_raises(minimal_ckpt_dict):
    """Seed mismatch raises ValueError with a message containing 'seed'."""
    minimal_ckpt_dict[SEED_KEY] = SEED + 1
    with pytest.raises(ValueError, match="seed"):
        resume_training_state(minimal_ckpt_dict)


def test_resume_training_state_missing_global_step_raises(minimal_ckpt_dict):
    """Missing GLOBAL_STEP_KEY raises KeyError."""
    del minimal_ckpt_dict[GLOBAL_STEP_KEY]
    with pytest.raises(KeyError):
        resume_training_state(minimal_ckpt_dict)


def test_resume_training_state_missing_epochs_raises(minimal_ckpt_dict):
    """Missing EPOCHS_KEY raises KeyError."""
    del minimal_ckpt_dict[EPOCHS_KEY]
    with pytest.raises(KeyError):
        resume_training_state(minimal_ckpt_dict)


def test_resume_training_state_missing_optimizer_raises(minimal_ckpt_dict):
    """Missing OPTIMIZER_KEY raises KeyError."""
    del minimal_ckpt_dict[OPTIMIZER_KEY]
    with pytest.raises(KeyError):
        resume_training_state(minimal_ckpt_dict)


# ---------------------------------------------------------------------------
# Disk tests — require real FullModelHFCheckpointer and LLAMA_3_2_1B_BASE_DIR
# ---------------------------------------------------------------------------

_skip_disk = pytest.mark.skipif(
    not LLAMA_3_2_1B_BASE_DIR.exists(),
    reason="Llama 3.2 1B base dir not found on this machine",
)


@_skip_disk
def test_save_recipe_state_contains_global_step(tmp_path, minimal_ckpt_dict):
    """Saved recipe_state.pt contains GLOBAL_STEP_KEY and does not contain the legacy 'steps_run' key.

    Regression guard for the B1 bug: confirms the canonical key is written correctly.
    """
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=LLAMA_3_2_1B_BASE_DIR,
        checkpoint_files=["model.safetensors"],
        output_dir=tmp_path,
        recipe_checkpoint=None,
    )
    checkpointer.save_recipe_state(minimal_ckpt_dict)
    loaded = torch.load(tmp_path / "recipe_state.pt", weights_only=False)
    assert loaded[GLOBAL_STEP_KEY] == 150
    assert "steps_run" not in loaded


@_skip_disk
def test_save_and_resume_recipe_state_round_trips(tmp_path, minimal_ckpt_dict):
    """On-disk round-trip: save_recipe_state then resume_training_state returns the original state."""
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=LLAMA_3_2_1B_BASE_DIR,
        checkpoint_files=["model.safetensors"],
        output_dir=tmp_path,
        recipe_checkpoint=None,
    )
    checkpointer.save_recipe_state(minimal_ckpt_dict)
    loaded = torch.load(tmp_path / "recipe_state.pt", weights_only=False)
    epochs, global_step, opt_state = resume_training_state(loaded)
    assert epochs == 2
    assert global_step == 150
    assert opt_state == {"state": {}, "param_groups": []}
