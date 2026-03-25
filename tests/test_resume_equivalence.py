"""
T-I7: Checkpoint save + resume equivalence.

Verifies that a Trainer run interrupted at step N/2 and resumed from
checkpoint produces bit-identical losses at steps N/2+1 … N compared to
an uninterrupted N-step run.

Uses the real Llama 3.2 1B extended model and 2k MLS-HuBERT samples
streamed via n_samples.

Prerequisites:
    ~/models/extended/Llama-3.2-1B-5000-dsus/ (or HAFH env var)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import shutil
import tempfile
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf
import pytest
import torch

from ssi.constants import SEED
from ssi.trainer import Trainer
from tests.conftest import EXTENDED_MODEL_DIR, _has_extended_model


LOGGER = logging.getLogger(__name__)

CONF_DIR = Path(__file__).resolve().parent.parent / "conf"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
    pytest.mark.skipif(not _has_extended_model(), reason=f"Extended model not found at {EXTENDED_MODEL_DIR}"),
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
# ---------------------------------------------------------------------------

_N_TRAIN_SAMPLES = 2000
_N_DEV_SAMPLES = 200
_TOTAL_STEPS = 8
_SAVE_AT_STEP = 4  # checkpoint mid-run
_BATCH_SIZE = 2
_GRAD_ACCUM = 2
_EVAL_STEPS = 4  # must divide save_steps


# ---------------------------------------------------------------------------
# Config composition
# ---------------------------------------------------------------------------


def _compose_cfg(output_dir: Path, max_steps: int, save_steps: int, eval_steps: int = _EVAL_STEPS) -> DictConfig:
    conf = CONF_DIR
    common = OmegaConf.load(conf / "common.yaml")
    training_cfg = OmegaConf.load(conf / "training.yaml")
    sft = OmegaConf.load(conf / "sft.yaml")
    data_base = OmegaConf.load(conf / "data" / "_sft_base.yaml")
    data_override = OmegaConf.load(conf / "data" / "sft" / "mls-hubert_large_ll60k-layer_22.yaml")
    data_cfg = OmegaConf.merge(data_base, data_override)
    cfg = OmegaConf.merge(common, training_cfg, sft, {"data": data_cfg})

    overrides = OmegaConf.create(
        {
            "speech": {"n_dsus": 5000, "use_modality_tokens": True, "deduplicate": True},
            "config_name": "sft",
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
                "max_seq_len": 2048,
                "prompt_template": None,
                "verbose": False,
            },
            "max_steps": max_steps,
            "eval_steps": eval_steps,
            "save_steps": save_steps,
            "gradient_accumulation_steps": _GRAD_ACCUM,
            "optimizer": {"lr": 5e-5, "fused": False},
            "lr_scheduler": None,  # constant LR — avoids schedule divergence from different max_steps
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
                "project": "test-resume",
                "entity": None,
                "log_dir": str(output_dir / "wandb"),
            },
            "compile": False,
            "debug_mode": None,
            "device": "cuda",
            "dtype": "bf16",
        }
    )
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
# Helpers
# ---------------------------------------------------------------------------


def _run_trainer(cfg: DictConfig, wandb_capture: WandBCapture) -> list[float]:
    """Run Trainer.setup() + train() with captured W&B, return loss list."""
    with (
        patch("ssi.trainer.WandBLogger", return_value=wandb_capture),
        patch("ssi.trainer.resolve_checkpointer_output_dir", return_value=Path(cfg.checkpointer.output_dir)),
    ):
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


class TestResumeEquivalence:
    """T-I7: Run N steps interrupted at N/2, resume, compare against
    uninterrupted N-step run."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.tmp_path = Path(tempfile.mkdtemp(prefix="test_resume_"))
        yield
        shutil.rmtree(self.tmp_path, ignore_errors=True)

    def test_resumed_losses_match_uninterrupted(self):
        """Losses from a resumed run match the corresponding steps of a
        continuous run exactly."""
        from torchtune import training

        # --- Run A: uninterrupted for TOTAL_STEPS ---
        training.set_seed(seed=SEED)
        cfg_full = _compose_cfg(
            self.tmp_path / "full",
            max_steps=_TOTAL_STEPS,
            save_steps=9996,  # effectively never (must be multiple of eval_steps)
        )
        wandb_full = WandBCapture()
        losses_full = _run_trainer(cfg_full, wandb_full)
        assert len(losses_full) == _TOTAL_STEPS

        # --- Run B1: first half, checkpoint at SAVE_AT_STEP ---
        training.set_seed(seed=SEED)
        cfg_b1 = _compose_cfg(
            self.tmp_path / "b1",
            max_steps=_SAVE_AT_STEP,
            save_steps=_SAVE_AT_STEP,
        )
        wandb_b1 = WandBCapture()
        losses_b1 = _run_trainer(cfg_b1, wandb_b1)
        assert len(losses_b1) == _SAVE_AT_STEP

        # Verify first half matches
        for i in range(_SAVE_AT_STEP):
            assert losses_full[i] == losses_b1[i], (
                f"Pre-resume loss mismatch at step {i}: full={losses_full[i]}, b1={losses_b1[i]}"
            )

        # --- Run B2: resume from checkpoint, finish remaining steps ---
        ckpt_dir = self.tmp_path / "b1" / "checkpoints"
        training_state_path = ckpt_dir / "training_state.pt"
        model_ckpt_dir = ckpt_dir / f"step_{_SAVE_AT_STEP}"

        assert training_state_path.exists(), f"Training state not found: {training_state_path}"
        assert model_ckpt_dir.exists(), f"Model checkpoint not found: {model_ckpt_dir}"

        # Find the saved model checkpoint filename
        st_files = sorted(model_ckpt_dir.glob("*.safetensors"))
        assert len(st_files) > 0, "No safetensors files in checkpoint dir"

        training.set_seed(seed=SEED)
        cfg_b2 = _compose_cfg(
            self.tmp_path / "b2",
            max_steps=_TOTAL_STEPS,
            save_steps=9996,
        )
        # Point at the checkpoint from B1
        OmegaConf.update(cfg_b2, "checkpointer.checkpoint_dir", str(model_ckpt_dir))
        OmegaConf.update(cfg_b2, "checkpointer.checkpoint_files", [f.name for f in st_files])
        OmegaConf.update(cfg_b2, "checkpointer.training_state_checkpoint", str(training_state_path))

        wandb_b2 = WandBCapture()
        losses_b2 = _run_trainer(cfg_b2, wandb_b2)

        # B2 should produce the SECOND half of the losses
        expected_remaining = _TOTAL_STEPS - _SAVE_AT_STEP
        assert len(losses_b2) == expected_remaining, (
            f"Expected {expected_remaining} steps after resume, got {len(losses_b2)}"
        )

        # Compare resumed losses against uninterrupted run
        for i, (l_full, l_resumed) in enumerate(zip(losses_full[_SAVE_AT_STEP:], losses_b2, strict=True)):
            step = _SAVE_AT_STEP + i + 1
            assert l_full == l_resumed, (
                f"Resume loss mismatch at step {step}: "
                f"full={l_full}, resumed={l_resumed}, diff={abs(l_full - l_resumed)}"
            )

    def test_resumed_wandb_metrics_match(self):
        """W&B logged metrics (loss, lr, tokens) from resumed run match."""
        from torchtune import training

        # Uninterrupted
        training.set_seed(seed=SEED)
        cfg_full = _compose_cfg(
            self.tmp_path / "full2",
            max_steps=_TOTAL_STEPS,
            save_steps=9996,
        )
        wandb_full = WandBCapture()
        _run_trainer(cfg_full, wandb_full)

        # First half + checkpoint
        training.set_seed(seed=SEED)
        cfg_b1 = _compose_cfg(
            self.tmp_path / "b1_2",
            max_steps=_SAVE_AT_STEP,
            save_steps=_SAVE_AT_STEP,
        )
        wandb_b1 = WandBCapture()
        _run_trainer(cfg_b1, wandb_b1)

        # Resume
        ckpt_dir = self.tmp_path / "b1_2" / "checkpoints"
        model_ckpt_dir = ckpt_dir / f"step_{_SAVE_AT_STEP}"
        st_files = sorted(model_ckpt_dir.glob("*.safetensors"))

        training.set_seed(seed=SEED)
        cfg_b2 = _compose_cfg(
            self.tmp_path / "b2_2",
            max_steps=_TOTAL_STEPS,
            save_steps=9996,
        )
        OmegaConf.update(cfg_b2, "checkpointer.checkpoint_dir", str(model_ckpt_dir))
        OmegaConf.update(cfg_b2, "checkpointer.checkpoint_files", [f.name for f in st_files])
        OmegaConf.update(cfg_b2, "checkpointer.training_state_checkpoint", str(ckpt_dir / "training_state.pt"))

        wandb_b2 = WandBCapture()
        _run_trainer(cfg_b2, wandb_b2)

        # Compare the second half of full run against resumed run
        timing_keys = {"duration_step", "tokens_per_second_per_gpu", "train_clock_time"}
        full_second_half = wandb_full.logged[_SAVE_AT_STEP:]

        assert len(full_second_half) == len(wandb_b2.logged), (
            f"Log count mismatch: full_second_half={len(full_second_half)}, resumed={len(wandb_b2.logged)}"
        )

        for i, ((pf, sf), (pr, sr)) in enumerate(zip(full_second_half, wandb_b2.logged, strict=True)):
            assert sf == sr, f"Step mismatch at log {i}: {sf} vs {sr}"
            for key in pf:
                if key in timing_keys:
                    continue
                assert pf[key] == pr[key], f"Metric '{key}' mismatch at step {sf}: full={pf[key]}, resumed={pr[key]}"
