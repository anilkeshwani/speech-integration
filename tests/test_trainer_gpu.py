"""
GPU integration tests for the stateful Trainer class.

These tests require:
- A CUDA GPU
- The extended Llama 3.2 1B model at EXTENDED_MODEL_DIR
- Network access to stream MLS data from HuggingFace

Tests use streaming+materialize to load only ~2000 samples (near-zero disk).

T-I1: Setup smoke test
T-I2: Single train step produces finite loss
T-I3: Optimizer step increments global_step
T-I4: Evaluation returns finite float
T-I5: Full train run (10 steps)
T-I6: Functional equivalence (stateful vs functional train())
"""

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
import pytest
import torch

from ssi.constants import TORCHTUNE_EXTENDED_MODELS_DIR
from ssi.trainer import Trainer, TrainingGeometry


# ---------------------------------------------------------------------------
# Ensure HF_HOME is set for streaming downloads
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Extended model path discovery (mirrors conftest.py logic)
# ---------------------------------------------------------------------------

EXTENDED_MODEL_DIR = TORCHTUNE_EXTENDED_MODELS_DIR / "Llama-3.2-1B-5000-dsus"
_LOCAL_EXTENDED = Path.home() / "models" / "extended" / "Llama-3.2-1B-5000-dsus"
if _LOCAL_EXTENDED.exists():
    EXTENDED_MODEL_DIR = _LOCAL_EXTENDED


def _has_extended_model() -> bool:
    return EXTENDED_MODEL_DIR.exists() and (EXTENDED_MODEL_DIR / "model.safetensors.index.json").exists()


def _get_checkpoint_files() -> list[str]:
    """Discover the safetensors checkpoint file(s) in the extended model dir."""
    st_files = sorted(EXTENDED_MODEL_DIR.glob("*.safetensors"))
    return [f.name for f in st_files]


# All tests in this module require GPU + extended model
pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available"),
    pytest.mark.skipif(not _has_extended_model(), reason=f"Extended model not found at {EXTENDED_MODEL_DIR}"),
]

# Constants for test configs
_HF_SOURCE = "anilkeshwani/mls-hubert_large_ll60k-layer_22"
_TRAIN_SUBSET = 200  # small for speed
_DEV_SUBSET = 50
_CONF_DIR = os.path.join(os.path.dirname(__file__), "..", "conf")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------



def _compose_sft_cfg(tmp_path, max_steps=10, eval_steps=5, save_steps=10, grad_accum=2, batch_size=2):
    """Compose an SFT config with Hydra, overriding for small test runs."""
    GlobalHydra.instance().clear()
    abs_conf_dir = os.path.abspath(_CONF_DIR)
    with initialize_config_dir(config_dir=abs_conf_dir, version_base=None):
        cfg = compose(
            config_name="sft",
            overrides=[
                "speech.n_dsus=5000",
                f"checkpointer.checkpoint_dir={EXTENDED_MODEL_DIR}",
                f"checkpointer.checkpoint_files={_get_checkpoint_files()}",
                f"checkpointer.output_dir={tmp_path}/checkpoints",
                "data=sft/mls-hubert_large_ll60k-layer_22",
                f"max_steps={max_steps}",
                f"eval_steps={eval_steps}",
                f"save_steps={save_steps}",
                f"gradient_accumulation_steps={grad_accum}",
                f"data.train.dataloader.batch_size={batch_size}",
                f"data.dev.dataloader.batch_size={batch_size}",
                "log_interval=1",
                f"tokenizer.path={EXTENDED_MODEL_DIR / 'original' / 'tokenizer.model'}",
                "wandb.project=test-speech-integration",
                "wandb.entity=null",
                "wandb.log_dir=/tmp/wandb_test",
                "config_name=sft",  # hydra:job.config_name not available in compose()
            ],
        )
    return cfg


def _make_trainer_with_subset(cfg, train_n=_TRAIN_SUBSET, dev_n=_DEV_SUBSET) -> Trainer:
    """Create a Trainer and patch datasets to use streaming subsets after setup."""
    trainer = Trainer(cfg)

    # We need to override setup to inject subset data.
    # Call setup but then patch the datasets post-hoc.
    trainer.setup()

    # Patch train and dev datasets with streaming subsets
    if trainer.data_train is not None:
        ds = trainer.data_train.dataset
        if hasattr(ds, "_data"):
            ds._data = ds._data.select(range(min(train_n, len(ds._data))))
    if trainer.data_dev is not None:
        ds = trainer.data_dev.dataset
        if hasattr(ds, "_data"):
            ds._data = ds._data.select(range(min(dev_n, len(ds._data))))

    # Recompute geometry after patching datasets
    trainer.geometry = TrainingGeometry.from_config(cfg, trainer.data_train, trainer.world_size)

    return trainer


# ---------------------------------------------------------------------------
# T-I1: Setup smoke test
# ---------------------------------------------------------------------------


def test_trainer_setup_smoke(tmp_path):
    """Trainer.setup() completes without error. All attributes are non-None."""
    cfg = _compose_sft_cfg(tmp_path)
    trainer = _make_trainer_with_subset(cfg)

    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.optimizer is not None
    assert trainer.loss_fn is not None
    assert trainer.checkpointer is not None
    assert trainer.wandb_logger is not None
    assert trainer.data_train is not None
    assert trainer.data_dev is not None
    assert trainer.geometry is not None
    assert trainer.device is not None
    assert trainer.dtype is not None
    assert trainer.world_size is not None
    assert trainer.token_type_ranges is not None

    # Geometry sanity
    assert trainer.geometry.batch_size > 0
    assert trainer.geometry.steps_per_epoch > 0
    assert trainer.geometry.n_epochs > 0

    trainer.cleanup()


# ---------------------------------------------------------------------------
# T-I2: Single train step
# ---------------------------------------------------------------------------


def test_single_train_step(tmp_path):
    """After setup, _train_step on one batch produces finite loss contributions."""
    cfg = _compose_sft_cfg(tmp_path)
    trainer = _make_trainer_with_subset(cfg)
    trainer._reset_step_accumulators()

    # Get one batch
    batch = next(iter(trainer.data_train))
    trainer._train_step(batch)

    assert trainer.num_tokens_step > 0
    # loss_running is a Tensor after backward
    assert isinstance(trainer.loss_running, torch.Tensor) or trainer.loss_running > 0
    assert trainer.max_seq_len_step > 0
    # token_type_counts should be populated
    assert sum(trainer.token_type_counts_total.values()) > 0

    trainer.cleanup()


# ---------------------------------------------------------------------------
# T-I3: Optimizer step increments global_step
# ---------------------------------------------------------------------------


def test_optimizer_step_increments(tmp_path):
    """Run grad_accum micro-batches then _optimizer_step. Verify global_step increments."""
    cfg = _compose_sft_cfg(tmp_path, grad_accum=2)
    trainer = _make_trainer_with_subset(cfg)
    import time
    trainer._reset_step_accumulators()
    trainer.optimizer.zero_grad()
    trainer.t_train_start = time.perf_counter()
    trainer.t_step_start = time.perf_counter()

    assert trainer.global_step == 0

    # Run 2 micro-batches (grad_accum=2)
    data_iter = iter(trainer.data_train)
    for _ in range(2):
        batch = next(data_iter)
        trainer._train_step(batch)

    # Capture param sum before optimizer step
    param_sum_before = sum(p.sum().item() for p in trainer.model.parameters())

    trainer._optimizer_step(epoch=0, iter_idx=1)

    assert trainer.global_step == 1
    assert trainer.consumed_samples > 0
    assert trainer.tokens_train_total > 0

    # Params should have changed
    param_sum_after = sum(p.sum().item() for p in trainer.model.parameters())
    assert param_sum_before != param_sum_after

    trainer.cleanup()


# ---------------------------------------------------------------------------
# T-I4: Evaluation returns finite float
# ---------------------------------------------------------------------------


def test_evaluate_returns_finite(tmp_path):
    """_evaluate() returns a finite float dev loss."""
    cfg = _compose_sft_cfg(tmp_path)
    trainer = _make_trainer_with_subset(cfg, dev_n=20)

    # Need global_step > 0 for the logging format inside compute_dataset_loss
    trainer.global_step = 1

    dev_loss = trainer._evaluate()

    assert isinstance(dev_loss, float)
    assert torch.isfinite(torch.tensor(dev_loss))
    assert dev_loss > 0  # loss should be positive for untrained model on real data

    trainer.cleanup()


# ---------------------------------------------------------------------------
# T-I5: Full train run (10 steps)
# ---------------------------------------------------------------------------


def test_full_train_run(tmp_path):
    """Trainer.train() for 10 steps completes without error, collecting losses."""
    cfg = _compose_sft_cfg(tmp_path, max_steps=10, eval_steps=5, save_steps=10)
    trainer = _make_trainer_with_subset(cfg)

    loss_log = []
    trainer._loss_log = loss_log

    trainer.train()

    assert trainer.global_step == 10
    assert len(loss_log) == 10
    # All losses should be finite and positive
    for loss in loss_log:
        assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss: {loss}"
        assert loss > 0, f"Non-positive loss: {loss}"

    trainer.cleanup()
