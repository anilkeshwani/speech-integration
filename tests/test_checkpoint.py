"""
Tests for checkpoint schema v1 and resume logic.

Covers:
- ``resume_training_state()`` schema contract and validation
- ``validate_resume_hparams()`` mismatch detection and force_resume override
- Resume position arithmetic (epochs_run, batches_to_skip)
- Legacy checkpoint rejection
- On-disk round-trip via ``save_recipe_state()``
- Framework RNG state save/restore round-trip

GPU required: No. All tests use CPU tensors and temporary directories.
"""

import random

import numpy as np
import pytest
import torch

from ssi.checkpoint import FullModelHFCheckpointer, restore_rng_states, save_rng_states
from ssi.constants import (
    CHECKPOINT_VERSION,
    CHECKPOINT_VERSION_KEY,
    CONSUMED_SAMPLES_KEY,
    CUMULATIVE_METRICS_KEY,
    EPOCHS_KEY,
    GLOBAL_STEP_KEY,
    LLAMA_3_2_1B_BASE_DIR,
    LR_SCHEDULER_KEY,
    MODEL_KEY,
    OPTIMIZER_KEY,
    RNG_KEY,
    SEED,
    SEED_KEY,
    TRAINING_HPARAMS_KEY,
)
from ssi.train import resume_training_state, validate_resume_hparams


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HPARAMS = {
    "batch_size": 16,
    "gradient_accumulation_steps": 4,
    "world_size": 1,
    "steps_per_epoch": 500,
}


@pytest.fixture
def v1_ckpt_dict():
    """A structurally valid v1 checkpoint dict with all required fields."""
    return {
        CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION,
        SEED_KEY: SEED,
        GLOBAL_STEP_KEY: 150,
        OPTIMIZER_KEY: {"state": {}, "param_groups": []},
        MODEL_KEY: {"weight": torch.zeros(4, 4)},
        LR_SCHEDULER_KEY: {"last_epoch": 150, "_last_lr": [2e-4]},
        RNG_KEY: {
            "python": random.getstate(),
            "numpy_global": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
        },
        TRAINING_HPARAMS_KEY: dict(HPARAMS),
        CONSUMED_SAMPLES_KEY: 9600,
        CUMULATIVE_METRICS_KEY: {
            "tokens_train_total": 100_000,
            "token_type_counts": {"text": 80_000, "dsu": 15_000, "special_text": 5_000},
            "wall_clock_seconds": 3600.0,
        },
    }


# ---------------------------------------------------------------------------
# T1: resume_training_state returns correct dict
# ---------------------------------------------------------------------------


def test_resume_returns_correct_fields(v1_ckpt_dict):
    result = resume_training_state(v1_ckpt_dict)
    assert result["global_step"] == 150
    assert result["optimizer_state"] == {"state": {}, "param_groups": []}
    assert result["lr_scheduler_state"]["last_epoch"] == 150
    assert result["consumed_samples"] == 9600
    assert result["cumulative_metrics"]["tokens_train_total"] == 100_000
    assert result["training_hparams"] == HPARAMS
    assert "python" in result["rng_state"]


# ---------------------------------------------------------------------------
# T2: Seed mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_seed_mismatch_raises(v1_ckpt_dict):
    v1_ckpt_dict[SEED_KEY] = SEED + 1
    with pytest.raises(ValueError, match="Seed mismatch"):
        resume_training_state(v1_ckpt_dict)


# ---------------------------------------------------------------------------
# T3: Missing required keys raise KeyError
# ---------------------------------------------------------------------------

_REQUIRED_KEYS = [
    GLOBAL_STEP_KEY,
    OPTIMIZER_KEY,
    LR_SCHEDULER_KEY,
    RNG_KEY,
    TRAINING_HPARAMS_KEY,
    CONSUMED_SAMPLES_KEY,
    CUMULATIVE_METRICS_KEY,
]


@pytest.mark.parametrize("key", _REQUIRED_KEYS, ids=[k for k in _REQUIRED_KEYS])
def test_missing_key_raises(v1_ckpt_dict, key):
    del v1_ckpt_dict[key]
    with pytest.raises(KeyError):
        resume_training_state(v1_ckpt_dict)


# ---------------------------------------------------------------------------
# T4: validate_resume_hparams passes on matching config
# ---------------------------------------------------------------------------


def test_hparam_validation_passes_on_match():
    validate_resume_hparams(ckpt_hparams=dict(HPARAMS), current_hparams=dict(HPARAMS))


# ---------------------------------------------------------------------------
# T5: validate_resume_hparams raises on each mismatch type
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, bad_value",
    [
        ("batch_size", 32),
        ("gradient_accumulation_steps", 8),
        ("world_size", 2),
        ("steps_per_epoch", 1000),
    ],
)
def test_hparam_mismatch_raises(key, bad_value):
    current = dict(HPARAMS)
    current[key] = bad_value
    with pytest.raises(ValueError, match=key):
        validate_resume_hparams(ckpt_hparams=dict(HPARAMS), current_hparams=current)


# ---------------------------------------------------------------------------
# T6: Legacy checkpoint (no checkpoint_version) raises ValueError
# ---------------------------------------------------------------------------


def test_legacy_checkpoint_raises(v1_ckpt_dict):
    del v1_ckpt_dict[CHECKPOINT_VERSION_KEY]
    with pytest.raises(ValueError, match="Legacy checkpoints are not supported"):
        resume_training_state(v1_ckpt_dict)


# ---------------------------------------------------------------------------
# T7: validate_resume_hparams warns with force_resume=True
# ---------------------------------------------------------------------------


def test_force_resume_warns_instead_of_raising(caplog):
    current = dict(HPARAMS)
    current["batch_size"] = 32
    # Should not raise, but should log a warning
    validate_resume_hparams(
        ckpt_hparams=dict(HPARAMS), current_hparams=current, force_resume=True
    )
    assert any("batch_size" in record.message for record in caplog.records)
    assert any(record.levelname == "WARNING" for record in caplog.records)


# ---------------------------------------------------------------------------
# T8-T10: Resume position arithmetic
# ---------------------------------------------------------------------------


def test_resume_position_mid_epoch():
    """Mid-epoch: global_step=150, steps_per_epoch=500 → epoch 0, skip 600 batches."""
    global_step, steps_per_epoch, grad_accum = 150, 500, 4
    epochs_run = global_step // steps_per_epoch
    batches_to_skip = (global_step % steps_per_epoch) * grad_accum
    assert epochs_run == 0
    assert batches_to_skip == 600


def test_resume_position_epoch_boundary():
    """Exact epoch boundary: global_step=500, steps_per_epoch=500 → epoch 1, skip 0."""
    global_step, steps_per_epoch, grad_accum = 500, 500, 4
    epochs_run = global_step // steps_per_epoch
    batches_to_skip = (global_step % steps_per_epoch) * grad_accum
    assert epochs_run == 1
    assert batches_to_skip == 0


def test_resume_position_fresh_start():
    """Fresh start: global_step=0 → epoch 0, skip 0."""
    global_step, steps_per_epoch, grad_accum = 0, 500, 4
    epochs_run = global_step // steps_per_epoch
    batches_to_skip = (global_step % steps_per_epoch) * grad_accum
    assert epochs_run == 0
    assert batches_to_skip == 0


# ---------------------------------------------------------------------------
# T11: batches_to_skip < batches_per_epoch invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("global_step", [0, 1, 249, 499, 500, 501, 999])
def test_batches_to_skip_within_bounds(global_step):
    steps_per_epoch, grad_accum = 500, 4
    batches_per_epoch = steps_per_epoch * grad_accum
    batches_to_skip = (global_step % steps_per_epoch) * grad_accum
    assert 0 <= batches_to_skip < batches_per_epoch


# ---------------------------------------------------------------------------
# Disk tests — require FullModelHFCheckpointer and LLAMA_3_2_1B_BASE_DIR
# ---------------------------------------------------------------------------

_skip_disk = pytest.mark.skipif(
    not LLAMA_3_2_1B_BASE_DIR.exists(),
    reason="Llama 3.2 1B base dir not found on this machine",
)


@_skip_disk
def test_round_trip_recipe_state(tmp_path, v1_ckpt_dict):
    """T12: save_recipe_state then load returns identical training state."""
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=LLAMA_3_2_1B_BASE_DIR,
        checkpoint_files=["model.safetensors"],
        output_dir=tmp_path,
    )
    checkpointer.save_recipe_state(v1_ckpt_dict)
    loaded = torch.load(tmp_path / "recipe_state.pt", weights_only=False)
    assert loaded[GLOBAL_STEP_KEY] == 150
    assert loaded[CONSUMED_SAMPLES_KEY] == 9600
    assert loaded[CHECKPOINT_VERSION_KEY] == CHECKPOINT_VERSION
    assert loaded[CUMULATIVE_METRICS_KEY]["tokens_train_total"] == 100_000
    # Model weights should be excluded from recipe state
    assert MODEL_KEY not in loaded


@_skip_disk
def test_recipe_state_contains_global_step_not_steps_run(tmp_path, v1_ckpt_dict):
    """T13: Regression guard — GLOBAL_STEP_KEY present, legacy 'steps_run' absent."""
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=LLAMA_3_2_1B_BASE_DIR,
        checkpoint_files=["model.safetensors"],
        output_dir=tmp_path,
    )
    checkpointer.save_recipe_state(v1_ckpt_dict)
    loaded = torch.load(tmp_path / "recipe_state.pt", weights_only=False)
    assert GLOBAL_STEP_KEY in loaded
    assert "steps_run" not in loaded


@_skip_disk
def test_recipe_state_does_not_contain_epochs_key(tmp_path, v1_ckpt_dict):
    """T14: EPOCHS_KEY is not in v1 checkpoints."""
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=LLAMA_3_2_1B_BASE_DIR,
        checkpoint_files=["model.safetensors"],
        output_dir=tmp_path,
    )
    checkpointer.save_recipe_state(v1_ckpt_dict)
    loaded = torch.load(tmp_path / "recipe_state.pt", weights_only=False)
    assert EPOCHS_KEY not in loaded


@_skip_disk
def test_legacy_checkpoint_from_disk_raises(tmp_path):
    """T15: A recipe_state.pt without checkpoint_version is rejected on resume."""
    legacy_dict = {
        SEED_KEY: SEED,
        EPOCHS_KEY: 2,
        GLOBAL_STEP_KEY: 100,
        OPTIMIZER_KEY: {"state": {}, "param_groups": []},
    }
    torch.save(legacy_dict, tmp_path / "recipe_state.pt")
    loaded = torch.load(tmp_path / "recipe_state.pt", weights_only=False)
    with pytest.raises(ValueError, match="Legacy checkpoints are not supported"):
        resume_training_state(loaded)


# ---------------------------------------------------------------------------
# T16: Framework RNG state round-trip
# ---------------------------------------------------------------------------


def test_rng_state_round_trip():
    """Save RNG states, advance them, restore, and verify generation matches."""
    # Set a known state
    random.seed(12345)
    np.random.seed(12345)
    torch.manual_seed(12345)

    saved = save_rng_states()

    # Record what the next random values would be
    py_val = random.random()
    np_val = np.random.rand()
    torch_val = torch.rand(1).item()

    # Advance the RNG further (contaminate)
    for _ in range(100):
        random.random()
        np.random.rand()
        torch.rand(1)

    # Restore and verify the same values come out
    restore_rng_states(saved)
    assert random.random() == py_val
    assert np.random.rand() == np_val
    assert torch.rand(1).item() == torch_val
