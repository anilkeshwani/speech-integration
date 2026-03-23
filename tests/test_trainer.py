"""
Tests for the stateful Trainer class and TrainingGeometry.

Unit tests (CPU, no model weights):
- T-U1: Trainer construction stores cfg, all components are None
- T-U2: TrainingGeometry computes correct values for known inputs
- T-U3: TrainingGeometry warns on remainder batches
- T-U4: TrainingGeometry raises on insufficient batches
"""

from unittest.mock import MagicMock

from omegaconf import OmegaConf
import pytest

from ssi.trainer import Trainer, TrainingGeometry


# ---------------------------------------------------------------------------
# T-U1: Trainer construction
# ---------------------------------------------------------------------------


def test_trainer_construction_stores_cfg():
    """Trainer(cfg) stores cfg and initializes all component attributes to None."""
    cfg = OmegaConf.create({"dummy": True})
    trainer = Trainer(cfg)
    assert trainer.cfg is cfg
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.optimizer is None
    assert trainer.lr_scheduler is None
    assert trainer.loss_fn is None
    assert trainer.checkpointer is None
    assert trainer.wandb_logger is None
    assert trainer.data_train is None
    assert trainer.sampler_train is None
    assert trainer.data_dev is None
    assert trainer.token_type_ranges is None
    assert trainer.geometry is None
    assert trainer.device is None
    assert trainer.dtype is None
    assert trainer.world_size is None


def test_trainer_initial_training_state():
    """Trainer starts with zeroed training state."""
    cfg = OmegaConf.create({"dummy": True})
    trainer = Trainer(cfg)
    assert trainer.global_step == 0
    assert trainer.consumed_samples == 0
    assert trainer.tokens_train_total == 0
    assert trainer.wall_clock_offset == 0.0
    assert trainer.loss_running == 0.0
    assert trainer.num_tokens_step == 0
    assert trainer.max_seq_len_step == 0
    assert dict(trainer.token_type_counts_total) == {}


def test_trainer_loss_log_default_none():
    """_loss_log is None by default, can be set to a list for testing."""
    cfg = OmegaConf.create({"dummy": True})
    trainer = Trainer(cfg)
    assert trainer._loss_log is None
    trainer._loss_log = []
    assert trainer._loss_log == []


# ---------------------------------------------------------------------------
# T-U2: TrainingGeometry computation
# ---------------------------------------------------------------------------


def _make_mock_dataloader(length: int) -> MagicMock:
    dl = MagicMock(spec=["__len__"])
    dl.__len__ = MagicMock(return_value=length)
    return dl


def test_geometry_basic():
    """Standard case: 100 batches, grad_accum=4 -> 25 steps/epoch."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 16}}},
        "gradient_accumulation_steps": 4,
        "max_steps": 100,
    })
    dl = _make_mock_dataloader(100)
    geo = TrainingGeometry.from_config(cfg, dl, world_size=1)
    assert geo.batch_size == 16
    assert geo.batches_per_epoch == 100
    assert geo.steps_per_epoch == 25
    assert geo.usable_batches == 100
    assert geo.n_epochs == 4  # ceil(100 / 25)
    assert geo.gradient_accumulation_steps == 4
    assert geo.world_size == 1


def test_geometry_partial_epoch():
    """max_steps not a multiple of steps_per_epoch -> n_epochs rounds up."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 8}}},
        "gradient_accumulation_steps": 2,
        "max_steps": 30,
    })
    dl = _make_mock_dataloader(50)
    geo = TrainingGeometry.from_config(cfg, dl, world_size=1)
    assert geo.steps_per_epoch == 25
    assert geo.n_epochs == 2  # ceil(30 / 25)


def test_geometry_single_step_epoch():
    """Edge case: exactly 1 step per epoch."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 1}}},
        "gradient_accumulation_steps": 10,
        "max_steps": 5,
    })
    dl = _make_mock_dataloader(10)
    geo = TrainingGeometry.from_config(cfg, dl, world_size=1)
    assert geo.steps_per_epoch == 1
    assert geo.usable_batches == 10
    assert geo.n_epochs == 5


def test_geometry_multi_gpu():
    """world_size is stored correctly."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 4}}},
        "gradient_accumulation_steps": 1,
        "max_steps": 10,
    })
    dl = _make_mock_dataloader(20)
    geo = TrainingGeometry.from_config(cfg, dl, world_size=4)
    assert geo.world_size == 4
    assert geo.steps_per_epoch == 20


# ---------------------------------------------------------------------------
# T-U3: TrainingGeometry remainder warning
# ---------------------------------------------------------------------------


def test_geometry_remainder_warning(caplog):
    """When batches_per_epoch % grad_accum != 0, a warning is logged."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 8}}},
        "gradient_accumulation_steps": 3,
        "max_steps": 10,
    })
    dl = _make_mock_dataloader(10)  # 10 % 3 = 1 remainder
    geo = TrainingGeometry.from_config(cfg, dl, world_size=1)
    assert geo.steps_per_epoch == 3  # 10 // 3
    assert geo.usable_batches == 9  # 3 * 3
    assert any("remainder batches will be discarded" in record.message for record in caplog.records)


def test_geometry_no_remainder_no_warning(caplog):
    """When batches_per_epoch % grad_accum == 0, no warning."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 8}}},
        "gradient_accumulation_steps": 5,
        "max_steps": 10,
    })
    dl = _make_mock_dataloader(20)  # 20 % 5 = 0
    TrainingGeometry.from_config(cfg, dl, world_size=1)
    assert not any("remainder" in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# T-U4: TrainingGeometry raises on insufficient batches
# ---------------------------------------------------------------------------


def test_geometry_insufficient_batches_raises():
    """batches_per_epoch < gradient_accumulation_steps should raise."""
    cfg = OmegaConf.create({
        "data": {"train": {"dataloader": {"batch_size": 8}}},
        "gradient_accumulation_steps": 10,
        "max_steps": 5,
    })
    dl = _make_mock_dataloader(5)  # 5 < 10
    with pytest.raises(ValueError, match=r"batches_per_epoch.*gradient_accumulation_steps"):
        TrainingGeometry.from_config(cfg, dl, world_size=1)
