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
# T-CKP-1 through T-CKP-5: pure-dict tests (no filesystem)
# ---------------------------------------------------------------------------

def test_resume_training_state_returns_correct_tuple(minimal_ckpt_dict):
    """T-CKP-1: happy path returns (epochs, global_step, optimizer_state)."""
    epochs, global_step, opt_state = resume_training_state(minimal_ckpt_dict)
    assert epochs == 2
    assert global_step == 150
    assert opt_state == {"state": {}, "param_groups": []}


def test_resume_training_state_wrong_seed_raises(minimal_ckpt_dict):
    """T-CKP-2: mismatched seed raises ValueError mentioning 'seed'."""
    minimal_ckpt_dict[SEED_KEY] = SEED + 1
    with pytest.raises(ValueError, match="seed"):
        resume_training_state(minimal_ckpt_dict)


def test_resume_training_state_missing_global_step_raises(minimal_ckpt_dict):
    """T-CKP-3: missing GLOBAL_STEP_KEY raises KeyError."""
    del minimal_ckpt_dict[GLOBAL_STEP_KEY]
    with pytest.raises(KeyError):
        resume_training_state(minimal_ckpt_dict)


def test_resume_training_state_missing_epochs_raises(minimal_ckpt_dict):
    """T-CKP-4: missing EPOCHS_KEY raises KeyError."""
    del minimal_ckpt_dict[EPOCHS_KEY]
    with pytest.raises(KeyError):
        resume_training_state(minimal_ckpt_dict)


def test_resume_training_state_missing_optimizer_raises(minimal_ckpt_dict):
    """T-CKP-5: missing OPTIMIZER_KEY raises KeyError."""
    del minimal_ckpt_dict[OPTIMIZER_KEY]
    with pytest.raises(KeyError):
        resume_training_state(minimal_ckpt_dict)


# ---------------------------------------------------------------------------
# T-CKP-6 and T-CKP-7: disk tests using real FullModelHFCheckpointer
# ---------------------------------------------------------------------------

_LLAMA_DIR_EXISTS = LLAMA_3_2_1B_BASE_DIR.exists()
_skip_disk = pytest.mark.skipif(not _LLAMA_DIR_EXISTS, reason="Llama 3.2 1B base dir not found on this machine")


@_skip_disk
def test_save_recipe_state_contains_global_step(tmp_path, minimal_ckpt_dict):
    """T-CKP-6: recipe_state.pt contains global_step and does NOT contain steps_run."""
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
    """T-CKP-7: save_recipe_state then resume_training_state gives back (2, 150, opt_state)."""
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
