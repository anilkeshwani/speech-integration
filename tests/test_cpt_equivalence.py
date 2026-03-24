"""
T-I8: CPT functional vs stateful equivalence.

Same as T-I6 but for Continued Pre-Training (CPT) with interleaved
text-speech sequences, verifying that the TextCompletionDataset path
through both train() and Trainer produces identical results.

Uses the real Llama 3.2 1B extended model and 2k MLS-HuBERT samples
streamed via n_samples.

Prerequisites:
    /home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus/
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf
import pytest
import torch

from ssi.constants import SEED
from ssi.train import train as functional_train
from ssi.trainer import Trainer


LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXTENDED_MODEL_DIR = Path("/home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus")
CONF_DIR = Path(__file__).resolve().parent.parent / "conf"
_HF_SOURCE = "anilkeshwani/mls-hubert_large_ll60k-layer_22"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
    pytest.mark.skipif(
        not EXTENDED_MODEL_DIR.exists(),
        reason=f"Extended model not found at {EXTENDED_MODEL_DIR}",
    ),
]

# ---------------------------------------------------------------------------
# Deterministic CUDA
# ---------------------------------------------------------------------------


def _enable_deterministic_cuda():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


_enable_deterministic_cuda()

# ---------------------------------------------------------------------------
# Test parameters
#
# CPT uses larger batch sizes (16 default) but shorter seq_len (768).
# We use batch_size=4, grad_accum=2 for speed.
# ---------------------------------------------------------------------------

_N_TRAIN_SAMPLES = 2000
_N_DEV_SAMPLES = 200
_MAX_STEPS = 4
_BATCH_SIZE = 4
_GRAD_ACCUM = 2


# ---------------------------------------------------------------------------
# Config composition — CPT variant
# ---------------------------------------------------------------------------


def _compose_cpt_cfg(output_dir: Path) -> DictConfig:
    conf = CONF_DIR
    common = OmegaConf.load(conf / "common.yaml")
    training_cfg = OmegaConf.load(conf / "training.yaml")
    cpt = OmegaConf.load(conf / "cpt.yaml")
    data_base = OmegaConf.load(conf / "data" / "_cpt_base.yaml")
    data_override = OmegaConf.load(conf / "data" / "cpt" / "mls-hubert_large_ll60k-layer_22.yaml")
    data_cfg = OmegaConf.merge(data_base, data_override)
    cfg = OmegaConf.merge(common, training_cfg, cpt, {"data": data_cfg})

    overrides = OmegaConf.create({
        "speech": {"n_dsus": 5000, "use_modality_tokens": True, "deduplicate": True},
        "config_name": "cpt",
        "checkpointer": {
            "checkpoint_dir": str(EXTENDED_MODEL_DIR),
            "checkpoint_files": ["ft-model-00001-of-00001.safetensors"],
            "output_dir": str(output_dir / "checkpoints"),
            "config_json": None,
            "training_state_checkpoint": None,
            "safe_serialization": True,
        },
        "tokenizer": {
            "path": str(EXTENDED_MODEL_DIR / "original" / "tokenizer.model"),
            "max_seq_len": 768,
            "prompt_template": None,
            "verbose": False,
        },
        "max_steps": _MAX_STEPS,
        "eval_steps": 999,
        "save_steps": 999,
        "gradient_accumulation_steps": _GRAD_ACCUM,
        "optimizer": {"lr": 5e-5, "fused": False},
        "lr_scheduler": None,
        "clip_grad_norm": None,
        "data": {
            "train": {
                "dataset": {"n_samples": _N_TRAIN_SAMPLES},
                "dataloader": {"batch_size": _BATCH_SIZE},
            },
            "dev": {
                "dataset": {"n_samples": _N_DEV_SAMPLES},
                "dataloader": {"batch_size": _BATCH_SIZE},
            },
        },
        "log_interval": 1,
        "wandb": {
            "project": "test-cpt-equiv",
            "entity": None,
            "log_dir": str(output_dir / "wandb"),
        },
        "compile": False,
        "debug_mode": None,
        "device": "cuda",
        "dtype": "bf16",
    })
    cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)

    def _strip(d):
        if OmegaConf.is_dict(d):
            for key in list(d):
                if key == "defaults":
                    del d[key]
                else:
                    _strip(d[key])

    _strip(cfg)
    return cfg


# ---------------------------------------------------------------------------
# W&B capture
# ---------------------------------------------------------------------------


class WandBCapture:
    def __init__(self, **kwargs):
        self.logged: list[tuple[dict, int]] = []
        self._wandb = MagicMock()
        self._wandb.run = MagicMock()
        self._wandb.run.name = "test-run"
        self._wandb.run.id = "test-id"
        self.config_allow_val_change = True

    def log_dict(self, payload, step):
        self.logged.append((dict(payload), step))

    def log_config(self, config):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _run_functional(cfg: DictConfig, wandb_capture: WandBCapture) -> list[float]:
    """Run functional train() capturing losses via W&B mock."""
    # Capture losses from the W&B log_dict calls
    with patch("ssi.train.WandBLogger", return_value=wandb_capture):
        with patch("ssi.train.resolve_checkpointer_output_dir",
                   return_value=Path(cfg.checkpointer.output_dir)):
            functional_train(cfg)
    return [p["loss"] for p, _ in wandb_capture.logged]


def _run_trainer(cfg: DictConfig, wandb_capture: WandBCapture) -> list[float]:
    """Run Trainer.setup() + train() capturing losses."""
    with patch("ssi.trainer.WandBLogger", return_value=wandb_capture):
        with patch("ssi.trainer.resolve_checkpointer_output_dir",
                   return_value=Path(cfg.checkpointer.output_dir)):
            trainer = Trainer(cfg)
            trainer.setup()
            trainer._loss_log = []
            trainer.train()
            losses = list(trainer._loss_log)
            trainer.cleanup()
    return losses


# ===========================================================================
# Tests
# ===========================================================================


class TestCPTEquivalence:
    """T-I8: CPT functional vs stateful equivalence with interleaved
    text-speech sequences."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmp_path = Path(tempfile.mkdtemp(prefix="test_cpt_equiv_"))
        yield
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_cpt_loss_sequences_identical(self):
        """Functional and Trainer produce identical CPT loss sequences."""
        from torchtune import training

        # --- Functional ---
        training.set_seed(seed=SEED)
        cfg_f = _compose_cpt_cfg(self.tmp_path / "functional")
        wandb_f = WandBCapture()
        losses_f = _run_functional(cfg_f, wandb_f)

        # --- Trainer ---
        training.set_seed(seed=SEED)
        cfg_t = _compose_cpt_cfg(self.tmp_path / "trainer")
        wandb_t = WandBCapture()
        losses_t = _run_trainer(cfg_t, wandb_t)

        assert len(losses_f) == len(losses_t) == _MAX_STEPS, (
            f"Expected {_MAX_STEPS} steps, got functional={len(losses_f)}, "
            f"trainer={len(losses_t)}"
        )
        for step, (lf, lt) in enumerate(zip(losses_f, losses_t)):
            assert lf == lt, (
                f"CPT loss mismatch at step {step}: "
                f"functional={lf}, trainer={lt}, diff={abs(lf - lt)}"
            )

    def test_cpt_wandb_metrics_identical(self):
        """All non-timing W&B metrics match between functional and Trainer."""
        from torchtune import training

        training.set_seed(seed=SEED)
        cfg_f = _compose_cpt_cfg(self.tmp_path / "functional2")
        wandb_f = WandBCapture()
        _run_functional(cfg_f, wandb_f)

        training.set_seed(seed=SEED)
        cfg_t = _compose_cpt_cfg(self.tmp_path / "trainer2")
        wandb_t = WandBCapture()
        _run_trainer(cfg_t, wandb_t)

        timing_keys = {"duration_step", "tokens_per_second_per_gpu", "train_clock_time"}

        assert len(wandb_f.logged) == len(wandb_t.logged)
        for i, ((pf, sf), (pt, st)) in enumerate(
            zip(wandb_f.logged, wandb_t.logged)
        ):
            assert sf == st
            for key in pf:
                if key in timing_keys:
                    continue
                assert pf[key] == pt[key], (
                    f"CPT metric '{key}' mismatch at step {sf}: "
                    f"functional={pf[key]}, trainer={pt[key]}"
                )

    def test_cpt_losses_finite_and_positive(self):
        """Sanity: CPT losses are finite and positive."""
        from torchtune import training

        training.set_seed(seed=SEED)
        cfg = _compose_cpt_cfg(self.tmp_path / "sanity")
        wandb = WandBCapture()
        losses = _run_trainer(cfg, wandb)

        for step, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), f"Non-finite CPT loss at step {step}"
            assert loss > 0, f"Non-positive CPT loss at step {step}"
