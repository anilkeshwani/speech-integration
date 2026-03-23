"""
Tests for the stateful Trainer class and TrainingGeometry.

Unit tests (CPU, no model weights):
- T-U1: Trainer construction stores cfg, all components are None
- T-U2: TrainingGeometry computes correct values for known inputs
- T-U3: TrainingGeometry warns on remainder batches
- T-U4: TrainingGeometry raises on insufficient batches
- T-U5: _extract_resume_state initializes attribute
- T-U6: _reset_step_accumulators zeroes all step-level state
- T-U7: _optimizer_step increments counters and records loss
- T-U8: _maybe_save_checkpoint delegates at save_steps boundaries
- T-U9: save_checkpoint calls checkpointer methods with correct args
- T-U10: _loss_log records per-step loss when enabled
"""

import time
from unittest.mock import MagicMock, patch

from omegaconf import OmegaConf
import pytest
import torch

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


# ---------------------------------------------------------------------------
# T-U5: Setup ordering — _extract_resume_state sets _resume_state before optimizer
# ---------------------------------------------------------------------------


def test_extract_resume_state_initializes_attribute():
    """_extract_resume_state sets _resume_state to None when no training_state_checkpoint."""
    from unittest.mock import MagicMock

    cfg = OmegaConf.create({"dummy": True})
    trainer = Trainer(cfg)
    # Simulate post-_setup_model state
    trainer.checkpointer = MagicMock()
    trainer.checkpointer.training_state_checkpoint = None
    trainer._ckpt_dict = {}

    trainer._extract_resume_state()

    assert trainer._resume_state is None
    assert trainer.global_step == 0
    assert trainer.consumed_samples == 0


# ---------------------------------------------------------------------------
# T-U6: _reset_step_accumulators
# ---------------------------------------------------------------------------


def test_reset_step_accumulators():
    """_reset_step_accumulators zeroes loss_running, num_tokens_step, max_seq_len_step."""
    cfg = OmegaConf.create({"dummy": True})
    trainer = Trainer(cfg)
    trainer.loss_running = 42.0
    trainer.num_tokens_step = 1000
    trainer.max_seq_len_step = 512

    trainer._reset_step_accumulators()

    assert trainer.loss_running == 0.0
    assert trainer.num_tokens_step == 0
    assert trainer.max_seq_len_step == 0
    assert trainer.t_step_start > 0  # timer was reset


# ---------------------------------------------------------------------------
# T-U7: _optimizer_step increments counters and records loss
# ---------------------------------------------------------------------------


def _make_trainer_for_optimizer_step():
    """Create a Trainer with mocked components ready for _optimizer_step."""
    cfg = OmegaConf.create({
        "gradient_accumulation_steps": 2,
        "clip_grad_norm": None,
        "eval_steps": 100,
        "log_interval": 1,
        "save_steps": 100,
    })
    trainer = Trainer(cfg)
    trainer.world_size = 1
    trainer.device = torch.device("cpu")

    # Mock model with parameters
    param = torch.nn.Parameter(torch.randn(4, 4))
    trainer.model = MagicMock()
    trainer.model.parameters.return_value = [param]
    trainer.model.named_parameters.return_value = [("weight", param)]

    # Mock optimizer with param_groups (needed by get_lr)
    trainer.optimizer = MagicMock()
    trainer.optimizer.param_groups = [{"lr": 2e-4}]
    trainer.lr_scheduler = MagicMock()
    trainer.wandb_logger = MagicMock()
    trainer.checkpointer = MagicMock()

    # Set geometry
    trainer.geometry = TrainingGeometry(
        batch_size=2, batches_per_epoch=20, steps_per_epoch=10,
        usable_batches=20, n_epochs=1, gradient_accumulation_steps=2, world_size=1,
    )

    # Set accumulators to simulate 2 micro-batches of work
    trainer.loss_running = torch.tensor(5.0)
    trainer.num_tokens_step = 100
    trainer.max_seq_len_step = 256
    trainer.t_train_start = time.perf_counter()
    trainer.t_step_start = time.perf_counter()
    trainer.token_type_counts_total = {"text": 80, "dsu": 20}

    return trainer


def test_optimizer_step_increments_counters():
    """_optimizer_step increments global_step, consumed_samples, tokens_train_total."""
    trainer = _make_trainer_for_optimizer_step()
    assert trainer.global_step == 0
    assert trainer.consumed_samples == 0
    assert trainer.tokens_train_total == 0

    with patch("ssi.trainer.scale_grads"):
        trainer._optimizer_step(epoch=0, iter_idx=1)

    assert trainer.global_step == 1
    assert trainer.consumed_samples == 2 * 2 * 1  # grad_accum * batch_size * world_size
    assert trainer.tokens_train_total == 100
    trainer.optimizer.step.assert_called_once()
    trainer.optimizer.zero_grad.assert_called_once_with(set_to_none=True)
    trainer.lr_scheduler.step.assert_called_once()


def test_optimizer_step_resets_accumulators():
    """After _optimizer_step, step-level accumulators are reset."""
    trainer = _make_trainer_for_optimizer_step()

    with patch("ssi.trainer.scale_grads"):
        trainer._optimizer_step(epoch=0, iter_idx=1)

    assert trainer.loss_running == 0.0
    assert trainer.num_tokens_step == 0
    assert trainer.max_seq_len_step == 0


# ---------------------------------------------------------------------------
# T-U8: _maybe_save_checkpoint
# ---------------------------------------------------------------------------


def test_maybe_save_checkpoint_at_boundary():
    """_maybe_save_checkpoint calls save_checkpoint at save_steps boundaries."""
    cfg = OmegaConf.create({"save_steps": 10, "gradient_accumulation_steps": 2})
    trainer = Trainer(cfg)
    trainer.global_step = 10
    trainer.consumed_samples = 200
    trainer.tokens_train_total = 1000
    trainer.token_type_counts_total = {"text": 800, "dsu": 200}
    trainer.wall_clock_offset = 0.0
    trainer.checkpointer = MagicMock()
    trainer.model = MagicMock()
    trainer.optimizer = MagicMock()
    trainer.lr_scheduler = MagicMock()
    trainer.geometry = TrainingGeometry(
        batch_size=2, batches_per_epoch=20, steps_per_epoch=10,
        usable_batches=20, n_epochs=1, gradient_accumulation_steps=2, world_size=1,
    )
    trainer.world_size = 1
    trainer.t_train_start = time.perf_counter()

    trainer._maybe_save_checkpoint()

    trainer.checkpointer.save_model_checkpoint.assert_called_once()
    trainer.checkpointer.save_training_state.assert_called_once()


def test_maybe_save_checkpoint_not_at_boundary():
    """_maybe_save_checkpoint does NOT call save at non-boundary steps."""
    cfg = OmegaConf.create({"save_steps": 10})
    trainer = Trainer(cfg)
    trainer.global_step = 7  # not a multiple of 10
    trainer.checkpointer = MagicMock()

    trainer._maybe_save_checkpoint()

    trainer.checkpointer.save_model_checkpoint.assert_not_called()
    trainer.checkpointer.save_training_state.assert_not_called()


def test_maybe_save_checkpoint_not_at_step_zero():
    """_maybe_save_checkpoint does NOT save at step 0 (even though 0 % N == 0)."""
    cfg = OmegaConf.create({"save_steps": 10})
    trainer = Trainer(cfg)
    trainer.global_step = 0
    trainer.checkpointer = MagicMock()

    trainer._maybe_save_checkpoint()

    trainer.checkpointer.save_model_checkpoint.assert_not_called()


# ---------------------------------------------------------------------------
# T-U9: save_checkpoint calls checkpointer with correct args
# ---------------------------------------------------------------------------


def test_save_checkpoint_passes_correct_args():
    """save_checkpoint passes model state, global_step, seed, hparams, metrics."""
    from ssi.constants import SEED

    cfg = OmegaConf.create({"gradient_accumulation_steps": 4})
    trainer = Trainer(cfg)
    trainer.global_step = 50
    trainer.consumed_samples = 800
    trainer.tokens_train_total = 5000
    trainer.token_type_counts_total = {"text": 4000, "dsu": 1000}
    trainer.wall_clock_offset = 100.0
    trainer.t_train_start = time.perf_counter()
    trainer.world_size = 1
    trainer.model = MagicMock()
    trainer.model.state_dict.return_value = {"weight": torch.zeros(4)}
    trainer.optimizer = MagicMock()
    trainer.optimizer.state_dict.return_value = {"state": {}}
    trainer.lr_scheduler = MagicMock()
    trainer.lr_scheduler.state_dict.return_value = {"last_epoch": 50}
    trainer.checkpointer = MagicMock()
    trainer.geometry = TrainingGeometry(
        batch_size=2, batches_per_epoch=100, steps_per_epoch=25,
        usable_batches=100, n_epochs=2, gradient_accumulation_steps=4, world_size=1,
    )

    trainer.save_checkpoint()

    # Verify model checkpoint saved with correct step
    trainer.checkpointer.save_model_checkpoint.assert_called_once()
    model_args = trainer.checkpointer.save_model_checkpoint.call_args
    assert model_args[0][1] == 50  # global_step

    # Verify training state saved with correct structure
    trainer.checkpointer.save_training_state.assert_called_once()
    ts_kwargs = trainer.checkpointer.save_training_state.call_args[1]
    assert ts_kwargs["global_step"] == 50
    assert ts_kwargs["seed"] == SEED
    assert ts_kwargs["consumed_samples"] == 800
    assert ts_kwargs["training_hparams"]["batch_size"] == 2
    assert ts_kwargs["training_hparams"]["gradient_accumulation_steps"] == 4
    assert ts_kwargs["training_hparams"]["steps_per_epoch"] == 25
    assert ts_kwargs["cumulative_metrics"]["tokens_train_total"] == 5000
    assert ts_kwargs["cumulative_metrics"]["token_type_counts"] == {"text": 4000, "dsu": 1000}


# ---------------------------------------------------------------------------
# T-U10: _loss_log records per-step loss when enabled
# ---------------------------------------------------------------------------


def test_loss_log_records_when_enabled():
    """When _loss_log is a list, _optimizer_step appends the per-token loss."""
    trainer = _make_trainer_for_optimizer_step()
    trainer._loss_log = []

    with patch("ssi.trainer.scale_grads"):
        trainer._optimizer_step(epoch=0, iter_idx=1)

    assert len(trainer._loss_log) == 1
    assert isinstance(trainer._loss_log[0], float)
    # loss = 5.0 / 100 = 0.05
    assert abs(trainer._loss_log[0] - 0.05) < 1e-6


def test_loss_log_not_recorded_when_none():
    """When _loss_log is None, no recording happens (no AttributeError)."""
    trainer = _make_trainer_for_optimizer_step()
    assert trainer._loss_log is None

    with patch("ssi.trainer.scale_grads"):
        trainer._optimizer_step(epoch=0, iter_idx=1)

    assert trainer._loss_log is None
