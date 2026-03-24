"""
T-I6 (full end-to-end): Functional train() vs stateful Trainer — identical
W&B traces AND checkpoints.

Runs both loops through their *real* setup paths (config validation, model
loading, data loading via n_samples streaming, W&B logging, checkpointing)
and asserts:
    (a) identical W&B loss and token-count traces at every logged step
    (b) identical model checkpoint weights on disk
    (c) identical training-state checkpoint fields (optimizer, step, metrics)

Uses the REAL Llama 3.2 1B extended model and the first 2000 samples of
MLS-HuBERT streamed via the new ``n_samples`` config parameter.

Hyperparameters follow the literature review (plans/Research - Optimal
hyperparameters based on literature review.md):
    - LR 5e-5 (Futami Interspeech 2025 + LLaSA consensus)
    - Cosine schedule with 3% warmup
    - Effective batch size 32 (batch_size=4, grad_accum=8)

Prerequisites:
    /home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus/   (extended model)
"""

from __future__ import annotations

import copy
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf
import pytest
from safetensors.torch import load_file as load_safetensors
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
# Test parameters
#
# Hyperparameters:
#   LR 5e-5       — consensus for full-param speech FT (Futami + LLaSA)
#   Cosine sched   — 3% warmup of max_steps
#   batch_size 4   — fits A6000 48GB with bf16 and seq_len ~1500
#   grad_accum 8   — effective batch size 32
#   4 steps        — enough to checkpoint once and compare
# ---------------------------------------------------------------------------

_N_TRAIN_SAMPLES = 2000
_N_DEV_SAMPLES = 200
_MAX_STEPS = 4
_BATCH_SIZE = 4
_GRAD_ACCUM = 8
_EVAL_STEPS = 4
_SAVE_STEPS = 4
_LR = 5e-5
_WARMUP_STEPS = 1  # ceil(0.03 * 4) — 3% warmup of max_steps


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
# Config composition (same as test_equivalence_e2e.py — merge real YAMLs)
# ---------------------------------------------------------------------------


def _compose_cfg(output_dir: Path) -> DictConfig:
    conf = CONF_DIR
    common = OmegaConf.load(conf / "common.yaml")
    training_cfg = OmegaConf.load(conf / "training.yaml")
    sft = OmegaConf.load(conf / "sft.yaml")
    data_base = OmegaConf.load(conf / "data" / "_sft_base.yaml")
    data_override = OmegaConf.load(conf / "data" / "sft" / "mls-hubert_large_ll60k-layer_22.yaml")
    data_cfg = OmegaConf.merge(data_base, data_override)
    cfg = OmegaConf.merge(common, training_cfg, sft, {"data": data_cfg})

    overrides = OmegaConf.create({
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
        "max_steps": _MAX_STEPS,
        "eval_steps": _EVAL_STEPS,
        "save_steps": _SAVE_STEPS,
        "gradient_accumulation_steps": _GRAD_ACCUM,
        "optimizer": {"lr": _LR, "fused": False},
        "lr_scheduler": {"num_warmup_steps": _WARMUP_STEPS, "num_cycles": 0.5},
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
            "project": "test-equivalence-full",
            "entity": None,
            "log_dir": str(output_dir / "wandb"),
        },
        "compile": False,
        "debug_mode": None,
        "device": "cuda",
        "dtype": "bf16",
        "_source": _HF_SOURCE,
    })
    cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)

    # Strip Hydra-internal keys
    def _strip(d):
        if OmegaConf.is_dict(d):
            for key in list(d):
                if key == "defaults":
                    del d[key]
                else:
                    _strip(d[key])

    _strip(cfg)
    if "_source" in cfg:
        del cfg["_source"]
    if "_source" in cfg.get("data", {}):
        del cfg.data["_source"]

    return cfg


# ---------------------------------------------------------------------------
# W&B capture mock
# ---------------------------------------------------------------------------


class WandBCapture:
    """Drop-in replacement for WandBLoggerPatched that captures all log_dict calls."""

    def __init__(self, **kwargs):
        self.logged: list[tuple[dict, int]] = []
        self._wandb = MagicMock()
        # Make wandb.run truthy so log_dict/log_config proceed
        self._wandb.run = MagicMock()
        self._wandb.run.name = "test-run"
        self._wandb.run.id = "test-id"
        self.config_allow_val_change = True

    def log_dict(self, payload, step):
        # Store a deep copy to avoid mutation
        self.logged.append((dict(payload), step))

    def log_config(self, config):
        pass  # no-op for test

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Run helpers
# ---------------------------------------------------------------------------


def _run_functional(cfg: DictConfig, wandb_capture: WandBCapture):
    """Run the real functional train() with our W&B capture."""
    with patch("ssi.train.WandBLogger", return_value=wandb_capture):
        with patch("ssi.train.resolve_checkpointer_output_dir", return_value=Path(cfg.checkpointer.output_dir)):
            functional_train(cfg)


def _run_trainer(cfg: DictConfig, wandb_capture: WandBCapture):
    """Run the real Trainer.setup() + Trainer.train() with our W&B capture."""
    with patch("ssi.trainer.WandBLogger", return_value=wandb_capture):
        with patch("ssi.trainer.resolve_checkpointer_output_dir", return_value=Path(cfg.checkpointer.output_dir)):
            trainer = Trainer(cfg)
            trainer.setup()
            trainer.train()
            trainer.cleanup()


# ===========================================================================
# Tests
# ===========================================================================


class TestFullEndToEnd:
    """True end-to-end: run both paths through their real setup, W&B logging,
    and checkpointing, then compare everything."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.dir_f = tmp_path / "functional"
        self.dir_t = tmp_path / "trainer"
        self.dir_f.mkdir()
        self.dir_t.mkdir()

    def _run_both(self):
        """Run functional and trainer loops, return their W&B captures."""
        # --- Functional ---
        from torchtune import training

        cfg_f = _compose_cfg(self.dir_f)
        training.set_seed(seed=SEED)
        wandb_f = WandBCapture()
        _run_functional(cfg_f, wandb_f)

        # --- Trainer ---
        cfg_t = _compose_cfg(self.dir_t)
        training.set_seed(seed=SEED)
        wandb_t = WandBCapture()
        _run_trainer(cfg_t, wandb_t)

        return wandb_f, wandb_t

    def test_wandb_loss_traces_identical(self):
        """Every W&B log_dict call has identical loss values and steps."""
        wandb_f, wandb_t = self._run_both()

        assert len(wandb_f.logged) == len(wandb_t.logged), (
            f"Different number of log_dict calls: functional={len(wandb_f.logged)}, "
            f"trainer={len(wandb_t.logged)}"
        )

        for i, ((payload_f, step_f), (payload_t, step_t)) in enumerate(
            zip(wandb_f.logged, wandb_t.logged)
        ):
            assert step_f == step_t, f"Step mismatch at log {i}: {step_f} vs {step_t}"
            assert payload_f["loss"] == payload_t["loss"], (
                f"Loss mismatch at step {step_f}: "
                f"functional={payload_f['loss']}, trainer={payload_t['loss']}, "
                f"diff={abs(payload_f['loss'] - payload_t['loss'])}"
            )

    def test_wandb_token_counts_identical(self):
        """W&B logged token counts match at every step."""
        wandb_f, wandb_t = self._run_both()

        for i, ((pf, sf), (pt, st)) in enumerate(zip(wandb_f.logged, wandb_t.logged)):
            assert pf["tokens_total"] == pt["tokens_total"], (
                f"tokens_total mismatch at step {sf}: {pf['tokens_total']} vs {pt['tokens_total']}"
            )
            # Compare per-type token counts
            for key in pf:
                if key.startswith("n_tokens."):
                    assert pf[key] == pt[key], (
                        f"{key} mismatch at step {sf}: {pf[key]} vs {pt[key]}"
                    )

    def test_wandb_all_metrics_identical(self):
        """Every numeric metric in W&B matches (except wall-clock timings)."""
        wandb_f, wandb_t = self._run_both()

        # Keys that are inherently timing-dependent
        timing_keys = {"duration_step", "tokens_per_second_per_gpu", "train_clock_time"}

        for i, ((pf, sf), (pt, st)) in enumerate(zip(wandb_f.logged, wandb_t.logged)):
            for key in pf:
                if key in timing_keys:
                    continue
                assert pf[key] == pt[key], (
                    f"Metric '{key}' mismatch at step {sf}: functional={pf[key]}, trainer={pt[key]}"
                )

    def test_checkpoint_model_weights_identical(self):
        """Model checkpoint safetensors files are bit-identical."""
        self._run_both()

        ckpt_dir_f = self.dir_f / "checkpoints" / f"step_{_SAVE_STEPS}"
        ckpt_dir_t = self.dir_t / "checkpoints" / f"step_{_SAVE_STEPS}"

        assert ckpt_dir_f.exists(), f"Functional checkpoint not found at {ckpt_dir_f}"
        assert ckpt_dir_t.exists(), f"Trainer checkpoint not found at {ckpt_dir_t}"

        # Find safetensor files
        st_files_f = sorted(ckpt_dir_f.glob("*.safetensors"))
        st_files_t = sorted(ckpt_dir_t.glob("*.safetensors"))
        assert len(st_files_f) == len(st_files_t) > 0

        for sf, st in zip(st_files_f, st_files_t):
            weights_f = load_safetensors(str(sf))
            weights_t = load_safetensors(str(st))
            assert set(weights_f.keys()) == set(weights_t.keys())
            for key in weights_f:
                assert torch.equal(weights_f[key], weights_t[key]), (
                    f"Checkpoint weight mismatch in '{key}': "
                    f"max_diff={(weights_f[key] - weights_t[key]).abs().max().item()}"
                )

    def test_checkpoint_training_state_identical(self):
        """Training state checkpoints have identical optimizer, step, metrics."""
        self._run_both()

        state_f = torch.load(self.dir_f / "checkpoints" / "training_state.pt", weights_only=False)
        state_t = torch.load(self.dir_t / "checkpoints" / "training_state.pt", weights_only=False)

        # Fields that should match exactly
        assert state_f["global_step"] == state_t["global_step"]
        assert state_f["seed"] == state_t["seed"]
        assert state_f["consumed_samples"] == state_t["consumed_samples"]
        assert state_f["training_hparams"] == state_t["training_hparams"]
        assert state_f["cumulative_metrics"]["tokens_train_total"] == state_t["cumulative_metrics"]["tokens_train_total"]
        assert state_f["cumulative_metrics"]["token_type_counts"] == state_t["cumulative_metrics"]["token_type_counts"]

        # Optimizer state: compare parameter tensors
        opt_f = state_f["optimizer"]
        opt_t = state_t["optimizer"]
        for pg_f, pg_t in zip(opt_f["state"].values(), opt_t["state"].values()):
            for key in ("exp_avg", "exp_avg_sq"):
                if key in pg_f:
                    assert torch.equal(pg_f[key], pg_t[key]), (
                        f"Optimizer state '{key}' mismatch"
                    )
