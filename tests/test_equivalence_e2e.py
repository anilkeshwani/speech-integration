"""
T-I6 (end-to-end): Functional equivalence — stateful Trainer vs functional train() loop.

Uses the REAL Llama 3.2 1B extended model, real SFT configs (composed from the
project YAML files via OmegaConf), and a small streamed subset of MLS-HuBERT data
(~2000 train, ~200 dev samples materialized to disk).

Runs N optimizer steps through each loop and asserts bit-identical loss sequences
and final model weights.

Prerequisites (created by the test setup script / CI):
    /home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus/   (extended model)
    /home/ubuntu/data/mls-hubert-subset/train/              (Arrow dataset)
    /home/ubuntu/data/mls-hubert-subset/validation/         (Arrow dataset)
"""

from __future__ import annotations

import copy
from collections import defaultdict
import itertools
import logging
import math
from pathlib import Path
import time
from unittest.mock import MagicMock, patch

from omegaconf import DictConfig, OmegaConf
import pytest
import torch
from torchtune import training
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, get_world_size_and_rank, scale_grads
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi.checkpoint import FullModelHFCheckpointer
from ssi.constants import MODEL_KEY, SEED
from ssi.data import setup_sft_data
from ssi.llama_configs import configllama3_2_1b
from ssi.loss import compute_loss
from ssi.model import setup_llama3_2_1b
from ssi.optimizer import setup_optimizer
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.train_utils import count_token_types, get_token_type_ranges, validate_train_cfg
from ssi.trainer import Trainer, TrainingGeometry


LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXTENDED_MODEL_DIR = Path("/home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus")
MLS_SUBSET_DIR = Path("/home/ubuntu/data/mls-hubert-subset")
CONF_DIR = Path(__file__).resolve().parent.parent / "conf"

# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU"),
    pytest.mark.skipif(
        not EXTENDED_MODEL_DIR.exists(),
        reason=f"Extended model not found at {EXTENDED_MODEL_DIR}",
    ),
    pytest.mark.skipif(
        not (MLS_SUBSET_DIR / "train").exists(),
        reason=f"MLS subset not found at {MLS_SUBSET_DIR}",
    ),
]

# ---------------------------------------------------------------------------
# Test parameters — small enough to run quickly, large enough to be meaningful
# ---------------------------------------------------------------------------

_MAX_STEPS = 6
_GRAD_ACCUM = 2
_BATCH_SIZE = 2
_EVAL_STEPS = 999  # never evaluate mid-run (we're testing training math)
_SAVE_STEPS = 999  # never checkpoint mid-run

# HuggingFace dataset source (used in config, redirected by monkeypatch)
_HF_SOURCE = "anilkeshwani/mls-hubert_large_ll60k-layer_22"


def _enable_deterministic_cuda():
    """Force fully deterministic CUDA execution.

    Without this, bf16 matmuls and cuBLAS reductions use non-deterministic
    algorithms that make even two identical runs of the same code diverge
    from the second optimizer step onwards.
    """
    import os

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# Enable determinism at module load time (before any CUDA ops)
_enable_deterministic_cuda()


# ---------------------------------------------------------------------------
# Config composition — merge YAML files via OmegaConf (bypasses Hydra
# defaults resolution bug with _base_ in nested config groups)
# ---------------------------------------------------------------------------


def _compose_cfg(tmp_path: Path) -> DictConfig:
    """Build the full SFT config by merging YAML files the way Hydra would.

    Loads common.yaml + training.yaml + sft.yaml + data/_sft_base.yaml +
    data/sft/mls-hubert…yaml, merges them in order, resolves interpolations,
    and applies test-specific overrides.
    """
    conf = CONF_DIR

    # Layer 1: common.yaml (base)
    common = OmegaConf.load(conf / "common.yaml")

    # Layer 2: training.yaml (optimizer, schedule, checkpointing)
    training_cfg = OmegaConf.load(conf / "training.yaml")

    # Layer 3: sft.yaml (SFT-specific)
    sft = OmegaConf.load(conf / "sft.yaml")

    # Layer 4: data/_sft_base.yaml (base data config)
    data_base = OmegaConf.load(conf / "data" / "_sft_base.yaml")

    # Layer 5: data/sft/mls-hubert…yaml (tokenizer-specific override)
    data_override = OmegaConf.load(conf / "data" / "sft" / "mls-hubert_large_ll60k-layer_22.yaml")

    # Merge data layers: base + override
    data_cfg = OmegaConf.merge(data_base, data_override)

    # Merge all into one config
    cfg = OmegaConf.merge(common, training_cfg, sft, {"data": data_cfg})

    # Apply test-specific overrides
    overrides = OmegaConf.create({
        "speech": {"n_dsus": 5000, "use_modality_tokens": True, "deduplicate": True},
        "config_name": "sft",
        "checkpointer": {
            "checkpoint_dir": str(EXTENDED_MODEL_DIR),
            "checkpoint_files": ["ft-model-00001-of-00001.safetensors"],
            "output_dir": str(tmp_path / "checkpoints"),
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
        "data": {
            "train": {"dataloader": {"batch_size": _BATCH_SIZE}},
            "dev": {"dataloader": {"batch_size": _BATCH_SIZE}},
        },
        "log_interval": 1,
        "wandb": {
            "project": "test-equivalence",
            "entity": None,
            "log_dir": str(tmp_path / "wandb"),
        },
        "optimizer": {"fused": False},  # fused AdamW is non-deterministic with bf16
        "lr_scheduler": None,
        "compile": False,
        "clip_grad_norm": None,
        "debug_mode": None,
        "device": "cuda",
        "dtype": "bf16",
        "_source": _HF_SOURCE,
    })
    cfg = OmegaConf.merge(cfg, overrides)

    # Resolve all interpolations
    OmegaConf.resolve(cfg)

    # Strip Hydra-internal keys that leaked from YAML defaults sections
    def _strip_hydra_keys(d):
        if OmegaConf.is_dict(d):
            for key in list(d):
                if key == "defaults":
                    del d[key]
                else:
                    _strip_hydra_keys(d[key])

    _strip_hydra_keys(cfg)

    # Remove the private _source key used for Hydra interpolation
    if "_source" in cfg:
        del cfg["_source"]
    if "_source" in cfg.get("data", {}):
        del cfg.data["_source"]

    return cfg


# ---------------------------------------------------------------------------
# Monkeypatch: redirect datasets.load_dataset to our local Arrow subset
# ---------------------------------------------------------------------------


def _patched_load_dataset(source, *args, split=None, **kwargs):
    """Load from local Arrow files instead of HuggingFace Hub."""
    from datasets import load_from_disk

    local_path = MLS_SUBSET_DIR / split
    if not local_path.exists():
        raise FileNotFoundError(
            f"Local MLS subset not found at {local_path}. "
            "Run the streaming materialisation step first."
        )
    LOGGER.info(f"[test] Loading local Arrow subset: {local_path}")
    return load_from_disk(str(local_path))


# ---------------------------------------------------------------------------
# Shared setup: build all components once, return cloneable state
# ---------------------------------------------------------------------------


def _shared_setup(cfg):
    """Perform the full setup once: model checkpoint, tokenizer, data.

    Returns everything both loops need.  The model checkpoint is loaded once
    and the state dict kept on CPU.  Each loop creates a **fresh** model
    instance from this state dict to avoid CUDA-level non-determinism that
    arises when reusing parameter tensors modified by fused AdamW kernels.
    """
    validate_train_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)
    device = get_device(cfg.device)
    dtype = get_dtype(cfg.dtype)
    world_size, _ = get_world_size_and_rank()

    # Load checkpoint once (the expensive part)
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    configllama3_2_1b.update_from_speech_cfg(cfg.speech)

    # Keep model state dict on CPU for cloning into fresh models
    init_state_dict = copy.deepcopy(ckpt_dict[MODEL_KEY])

    # Tokenizer
    tokenizer, _ = setup_llama3_tokenizer(**cfg.tokenizer)
    token_type_ranges = get_token_type_ranges(llama_config=configllama3_2_1b)

    # Data — patched to use local Arrow subset
    with patch("ssi.data.sft.load_dataset", side_effect=_patched_load_dataset):
        data_train, sampler_train = setup_sft_data(
            cfg_dataset=cfg.data.train, model_tokenizer=tokenizer
        )
        data_dev, _ = setup_sft_data(
            cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer
        )

    return {
        "tokenizer": tokenizer,
        "token_type_ranges": token_type_ranges,
        "data_train": data_train,
        "sampler_train": sampler_train,
        "data_dev": data_dev,
        "device": device,
        "dtype": dtype,
        "world_size": world_size,
        "init_state_dict": init_state_dict,
        "checkpointer": checkpointer,
    }


def _make_fresh_model(cfg, components):
    """Create a completely fresh model instance from the saved state dict.

    Ensures no CUDA-level state is shared between runs.
    """
    model = setup_llama3_2_1b(
        cfg=cfg,
        llama_config=configllama3_2_1b,
        model_state_dict=copy.deepcopy(components["init_state_dict"]),
        dtype_default=components["dtype"],
        device_default=components["device"],
    )
    model.to(device=components["device"])
    model.train()
    loss_fn = CEWithChunkedOutputLoss()
    model.set_num_output_chunks(loss_fn.num_output_chunks)
    return model, loss_fn


# ---------------------------------------------------------------------------
# Functional loop — extracted from ssi/train.py
# ---------------------------------------------------------------------------


def _run_functional_loop(cfg, components) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Run the functional train() inner loop, recording losses."""
    tokenizer = components["tokenizer"]
    token_type_ranges = components["token_type_ranges"]
    data_train = components["data_train"]
    sampler_train = components["sampler_train"]
    device = components["device"]

    # Fresh model — completely independent parameter tensors
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)
    model, loss_fn = _make_fresh_model(cfg, components)

    # Optimizer (fresh, from initial model params)
    optimizer = setup_optimizer(cfg, model, None)

    # Geometry
    batches_per_epoch = len(data_train)
    steps_per_epoch = batches_per_epoch // cfg.gradient_accumulation_steps
    n_epochs = math.ceil(cfg.max_steps / steps_per_epoch)
    usable_batches = steps_per_epoch * cfg.gradient_accumulation_steps

    # Training loop state
    global_step = 0
    loss_log: list[float] = []
    token_type_counts_total: defaultdict[str, int] = defaultdict(int)

    optimizer.zero_grad()
    loss_running = 0.0
    num_tokens_step = 0

    for epoch in range(n_epochs):
        sampler_train.set_epoch(epoch)
        if hasattr(data_train.dataset, "set_epoch"):
            data_train.dataset.set_epoch(epoch)
        data_iter = itertools.islice(enumerate(data_train), usable_batches)

        for i, batch in tqdm(data_iter, total=usable_batches, desc=f"[functional] epoch {epoch}"):
            batch_to_device(batch, device)
            for tt, ttcnt in count_token_types(batch["tokens"], token_type_ranges, tokenizer.pad_id).items():
                token_type_counts_total[tt] += ttcnt
            num_tokens_iter = int((batch["labels"] != loss_fn.ignore_index).sum().item())
            num_tokens_step += num_tokens_iter
            loss_batch = compute_loss(batch, model, loss_fn) * num_tokens_iter
            loss_running += loss_batch
            loss_batch.backward()

            if (i + 1) % cfg.gradient_accumulation_steps == 0:
                scale_grads(model, torch.tensor(1 / num_tokens_step))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                loss_to_log = loss_running.item() / num_tokens_step
                loss_log.append(loss_to_log)
                loss_running = 0.0
                num_tokens_step = 0
                if global_step >= cfg.max_steps:
                    break

            del batch
            torch.cuda.empty_cache()

        if global_step >= cfg.max_steps:
            break

    final_weights = {k: v.clone() for k, v in model.state_dict().items()}
    return loss_log, final_weights


# ---------------------------------------------------------------------------
# Stateful Trainer loop
# ---------------------------------------------------------------------------


def _run_trainer_loop(cfg, components) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Run training through the Trainer class, recording losses."""
    tokenizer = components["tokenizer"]
    token_type_ranges = components["token_type_ranges"]
    data_train = components["data_train"]
    sampler_train = components["sampler_train"]
    data_dev = components["data_dev"]
    device = components["device"]
    dtype = components["dtype"]
    world_size = components["world_size"]

    # Fresh model — completely independent parameter tensors
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)
    model, loss_fn = _make_fresh_model(cfg, components)

    trainer = Trainer(cfg)

    # Inject real components (bypass setup())
    trainer.model = model
    trainer.tokenizer = tokenizer
    trainer.token_type_ranges = token_type_ranges
    trainer.loss_fn = loss_fn
    trainer.data_train = data_train
    trainer.sampler_train = sampler_train
    trainer.data_dev = data_dev
    trainer.device = device
    trainer.dtype = dtype
    trainer.world_size = world_size
    trainer.checkpointer = MagicMock()
    trainer.wandb_logger = MagicMock()

    # Fresh optimizer from same initial params
    trainer.optimizer = setup_optimizer(cfg, model, None)
    trainer.lr_scheduler = None

    # Geometry
    trainer.geometry = TrainingGeometry.from_config(cfg, data_train, world_size)

    # Resume state (none — fresh start)
    trainer._resume_state = None

    # Enable loss logging
    loss_log: list[float] = []
    trainer._loss_log = loss_log

    # Run training
    trainer.train()

    final_weights = {k: v.clone() for k, v in model.state_dict().items()}
    return loss_log, final_weights


# ===========================================================================
# Tests
# ===========================================================================


class TestEndToEndEquivalence:
    """Prove bit-identical behaviour between functional and stateful loops
    using the real Llama 3.2 1B model and real MLS-HuBERT data."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        """Compose config and build all shared components once."""
        self.cfg = _compose_cfg(tmp_path)
        self.components = _shared_setup(self.cfg)
        yield
        # Cleanup
        torch.cuda.empty_cache()

    def test_loss_sequences_identical(self):
        """Both loops produce the exact same per-step loss values."""
        losses_f, _ = _run_functional_loop(self.cfg, self.components)
        losses_t, _ = _run_trainer_loop(self.cfg, self.components)

        assert len(losses_f) == len(losses_t) == _MAX_STEPS, (
            f"Expected {_MAX_STEPS} steps, got functional={len(losses_f)}, trainer={len(losses_t)}"
        )
        for step, (lf, lt) in enumerate(zip(losses_f, losses_t)):
            assert lf == lt, (
                f"Loss mismatch at step {step}: functional={lf}, trainer={lt}, "
                f"diff={abs(lf - lt)}"
            )

    def test_final_weights_identical(self):
        """Both loops produce the exact same model parameters after training."""
        _, weights_f = _run_functional_loop(self.cfg, self.components)
        _, weights_t = _run_trainer_loop(self.cfg, self.components)

        assert set(weights_f.keys()) == set(weights_t.keys())
        for key in weights_f:
            assert torch.equal(weights_f[key], weights_t[key]), (
                f"Weight mismatch in '{key}': "
                f"max_abs_diff={(weights_f[key] - weights_t[key]).abs().max().item()}"
            )

    def test_losses_finite_and_positive(self):
        """Sanity: all losses are finite positive floats."""
        losses_f, _ = _run_functional_loop(self.cfg, self.components)
        for step, loss in enumerate(losses_f):
            assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss at step {step}"
            assert loss > 0, f"Non-positive loss at step {step}"

    def test_step_count_matches_config(self):
        """Trainer runs exactly max_steps optimizer steps."""
        losses_t, _ = _run_trainer_loop(self.cfg, self.components)
        assert len(losses_t) == _MAX_STEPS
