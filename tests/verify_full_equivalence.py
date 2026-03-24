#!/usr/bin/env python
"""
Comprehensive equivalence verification: 200-step deterministic training run.

Runs both functional train() and stateful Trainer for 200 optimizer steps
with identical configs, deterministic CUDA, and real MLS-HuBERT data.
Compares:
  1. Per-step loss values (exact match)
  2. All W&B-logged metrics (exact match, except wall-clock timing)
  3. Model checkpoint weights at step 100 and step 200 (exact match)
  4. Training state checkpoint fields (optimizer, consumed_samples, etc.)
  5. Dev loss at eval steps (exact match)
  6. Token counts — cumulative and per-type (exact match)

Expected runtime: ~30 minutes total (~15 min per implementation).
"""

import copy
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Force deterministic CUDA before any torch import
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file as load_safetensors

# Patch safe_torch_load for OmegaConf types in training_state.pt
import ssi.checkpoint as _ssi_ckpt
_original_stl = _ssi_ckpt.safe_torch_load
def _stl_permissive(p, **kw):
    try:
        return _original_stl(p, **kw)
    except ValueError:
        kw.pop("mmap", None)
        return torch.load(p, weights_only=False, **kw)
_ssi_ckpt.safe_torch_load = _stl_permissive

from torchtune import training
from ssi.constants import SEED
from ssi.train import train as functional_train
from ssi.trainer import Trainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXTENDED_MODEL_DIR = Path("/home/ubuntu/models/extended/Llama-3.2-1B-5000-dsus")
CONF_DIR = Path(__file__).resolve().parent.parent / "conf"

# ---------------------------------------------------------------------------
# Parameters — substantial run
# ---------------------------------------------------------------------------
MAX_STEPS = 200
EVAL_STEPS = 50
SAVE_STEPS = 100    # checkpoints at step 100 and 200
BATCH_SIZE = 4
GRAD_ACCUM = 8      # effective batch = 32
N_TRAIN = 2000
N_DEV = 200
LR = 5e-5
WARMUP = 6          # ~3% of 200 steps


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def compose_cfg(output_dir: Path) -> DictConfig:
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
            "config_json": None, "training_state_checkpoint": None,
            "safe_serialization": True,
        },
        "tokenizer": {
            "path": str(EXTENDED_MODEL_DIR / "original" / "tokenizer.model"),
            "max_seq_len": 2048, "prompt_template": None, "verbose": False,
        },
        "max_steps": MAX_STEPS,
        "eval_steps": EVAL_STEPS,
        "save_steps": SAVE_STEPS,
        "gradient_accumulation_steps": GRAD_ACCUM,
        "optimizer": {"lr": LR, "fused": False},
        "lr_scheduler": {"num_warmup_steps": WARMUP, "num_cycles": 0.5},
        "clip_grad_norm": None,
        "data": {
            "train": {"dataset": {"n_samples": N_TRAIN}, "dataloader": {"batch_size": BATCH_SIZE}},
            "dev": {"dataset": {"n_samples": N_DEV}, "dataloader": {"batch_size": BATCH_SIZE}},
        },
        "log_interval": 1,
        "wandb": {"project": "verify-equiv", "entity": None, "log_dir": str(output_dir / "wandb")},
        "compile": False, "debug_mode": None, "device": "cuda", "dtype": "bf16",
    })
    cfg = OmegaConf.merge(cfg, overrides)
    OmegaConf.resolve(cfg)
    def strip(d):
        if OmegaConf.is_dict(d):
            for k in list(d):
                if k == "defaults": del d[k]
                else: strip(d[k])
    strip(cfg)
    return cfg


# ---------------------------------------------------------------------------
# W&B capture
# ---------------------------------------------------------------------------
class WandBCapture:
    def __init__(self, **kw):
        self.logged = []
        self._wandb = MagicMock()
        self._wandb.run = MagicMock()
        self._wandb.run.name = "verify"
        self._wandb.run.id = "verify"
        self.config_allow_val_change = True

    def log_dict(self, payload, step):
        self.logged.append((dict(payload), step))

    def log_config(self, config):
        pass

    def close(self):
        pass


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_functional(output_dir):
    training.set_seed(seed=SEED)
    cfg = compose_cfg(output_dir)
    wb = WandBCapture()
    t0 = time.time()
    with patch("ssi.train.WandBLogger", return_value=wb):
        with patch("ssi.train.resolve_checkpointer_output_dir", return_value=Path(cfg.checkpointer.output_dir)):
            functional_train(cfg)
    elapsed = time.time() - t0
    return wb.logged, elapsed


def run_trainer(output_dir):
    training.set_seed(seed=SEED)
    cfg = compose_cfg(output_dir)
    wb = WandBCapture()
    t0 = time.time()
    with patch("ssi.trainer.WandBLogger", return_value=wb):
        with patch("ssi.trainer.resolve_checkpointer_output_dir", return_value=Path(cfg.checkpointer.output_dir)):
            trainer = Trainer(cfg)
            trainer.setup()
            trainer._loss_log = []
            trainer.train()
            trainer.cleanup()
    elapsed = time.time() - t0
    return wb.logged, elapsed


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
TIMING_KEYS = {"duration_step", "tokens_per_second_per_gpu", "train_clock_time"}


def compare_metrics(logged_f, logged_t):
    """Compare all W&B logged metrics. Returns (n_compared, failures)."""
    failures = []
    n_compared = 0

    if len(logged_f) != len(logged_t):
        failures.append(f"Log count mismatch: functional={len(logged_f)}, trainer={len(logged_t)}")
        return n_compared, failures

    for i, ((pf, sf), (pt, st)) in enumerate(zip(logged_f, logged_t)):
        if sf != st:
            failures.append(f"Step mismatch at log {i}: {sf} vs {st}")
            continue
        for key in pf:
            if key in TIMING_KEYS:
                continue
            n_compared += 1
            if key not in pt:
                failures.append(f"Key '{key}' missing from trainer at step {sf}")
            elif pf[key] != pt[key]:
                failures.append(f"Step {sf}, '{key}': functional={pf[key]}, trainer={pt[key]}, diff={abs(pf[key] - pt[key]) if isinstance(pf[key], (int, float)) else 'N/A'}")

    return n_compared, failures


def compare_checkpoints(dir_f, dir_t, step):
    """Compare model checkpoint weights at a given step."""
    failures = []
    ckpt_f = dir_f / "checkpoints" / f"step_{step}"
    ckpt_t = dir_t / "checkpoints" / f"step_{step}"

    if not ckpt_f.exists():
        failures.append(f"Functional checkpoint missing at step {step}")
        return 0, failures
    if not ckpt_t.exists():
        failures.append(f"Trainer checkpoint missing at step {step}")
        return 0, failures

    st_f = sorted(ckpt_f.glob("*.safetensors"))
    st_t = sorted(ckpt_t.glob("*.safetensors"))
    if len(st_f) != len(st_t):
        failures.append(f"Step {step}: different number of safetensor files")
        return 0, failures

    n_compared = 0
    for sf, st in zip(st_f, st_t):
        wf = load_safetensors(str(sf))
        wt = load_safetensors(str(st))
        for key in wf:
            n_compared += 1
            if key not in wt:
                failures.append(f"Step {step}: key '{key}' missing from trainer checkpoint")
            elif not torch.equal(wf[key], wt[key]):
                diff = (wf[key].float() - wt[key].float()).abs().max().item()
                failures.append(f"Step {step}: weight '{key}' differs, max_abs_diff={diff}")

    return n_compared, failures


def compare_training_state(dir_f, dir_t):
    """Compare training_state.pt fields."""
    failures = []
    ts_f = dir_f / "checkpoints" / "training_state.pt"
    ts_t = dir_t / "checkpoints" / "training_state.pt"

    if not ts_f.exists() or not ts_t.exists():
        failures.append(f"Training state missing: f={ts_f.exists()}, t={ts_t.exists()}")
        return 0, failures

    sf = torch.load(ts_f, weights_only=False)
    st = torch.load(ts_t, weights_only=False)

    n_compared = 0
    # Scalar fields
    for key in ("global_step", "seed", "consumed_samples"):
        n_compared += 1
        if sf.get(key) != st.get(key):
            failures.append(f"training_state '{key}': f={sf.get(key)}, t={st.get(key)}")

    # Training hparams
    n_compared += 1
    if sf.get("training_hparams") != st.get("training_hparams"):
        failures.append(f"training_hparams differ")

    # Cumulative metrics
    cm_f = sf.get("cumulative_metrics", {})
    cm_t = st.get("cumulative_metrics", {})
    for key in ("tokens_train_total",):
        n_compared += 1
        if cm_f.get(key) != cm_t.get(key):
            failures.append(f"cumulative_metrics '{key}': f={cm_f.get(key)}, t={cm_t.get(key)}")

    n_compared += 1
    if cm_f.get("token_type_counts") != cm_t.get("token_type_counts"):
        failures.append(f"token_type_counts differ")

    # Optimizer state tensors
    opt_f = sf.get("optimizer", {}).get("state", {})
    opt_t = st.get("optimizer", {}).get("state", {})
    for param_id in opt_f:
        for buf_key in ("exp_avg", "exp_avg_sq"):
            if buf_key in opt_f[param_id]:
                n_compared += 1
                tf = opt_f[param_id][buf_key]
                tt = opt_t.get(param_id, {}).get(buf_key)
                if tt is None:
                    failures.append(f"optimizer state param {param_id}/{buf_key} missing from trainer")
                elif not torch.equal(tf, tt):
                    diff = (tf.float() - tt.float()).abs().max().item()
                    failures.append(f"optimizer state param {param_id}/{buf_key} differs, max_abs_diff={diff}")

    return n_compared, failures


# ===========================================================================
# Main
# ===========================================================================
def main():
    assert EXTENDED_MODEL_DIR.exists(), f"Model not found: {EXTENDED_MODEL_DIR}"
    assert torch.cuda.is_available(), "No GPU"

    base_dir = Path(tempfile.mkdtemp(prefix="verify_equiv_"))
    dir_f = base_dir / "functional"
    dir_t = base_dir / "trainer"
    dir_f.mkdir()
    dir_t.mkdir()

    print(f"Output directory: {base_dir}")
    print(f"Config: {MAX_STEPS} steps, bs={BATCH_SIZE}, grad_accum={GRAD_ACCUM}, "
          f"lr={LR}, warmup={WARMUP}, eval@{EVAL_STEPS}, save@{SAVE_STEPS}")
    print(f"Data: {N_TRAIN} train, {N_DEV} dev samples (streamed from HF)")
    print(f"Deterministic CUDA: ON, fused AdamW: OFF")
    print("=" * 70)

    # --- Run functional ---
    print(f"\n[1/2] Running functional train() for {MAX_STEPS} steps...")
    logged_f, time_f = run_functional(dir_f)
    print(f"  Done in {time_f:.1f}s ({time_f/60:.1f} min), {len(logged_f)} log entries")

    # --- Run trainer ---
    print(f"\n[2/2] Running Trainer for {MAX_STEPS} steps...")
    logged_t, time_t = run_trainer(dir_t)
    print(f"  Done in {time_t:.1f}s ({time_t/60:.1f} min), {len(logged_t)} log entries")

    # --- Comparisons ---
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    total_compared = 0
    total_failures = []

    # 1. W&B metrics
    n, fails = compare_metrics(logged_f, logged_t)
    total_compared += n
    total_failures.extend(fails)
    print(f"\n[W&B Metrics] {n} values compared, {len(fails)} failures")
    for f in fails[:10]:
        print(f"  FAIL: {f}")

    # 2. Loss values specifically
    losses_f = [p["loss"] for p, _ in logged_f]
    losses_t = [p["loss"] for p, _ in logged_t]
    loss_matches = sum(1 for a, b in zip(losses_f, losses_t) if a == b)
    print(f"\n[Loss Sequence] {loss_matches}/{len(losses_f)} steps match exactly")
    if loss_matches < len(losses_f):
        for i, (a, b) in enumerate(zip(losses_f, losses_t)):
            if a != b:
                print(f"  First mismatch at step {i}: f={a}, t={b}")
                break

    # 3. Dev loss
    dev_f = [(s, p.get("dev_loss")) for p, s in logged_f if "dev_loss" in p]
    dev_t = [(s, p.get("dev_loss")) for p, s in logged_t if "dev_loss" in p]
    dev_matches = sum(1 for (_, a), (_, b) in zip(dev_f, dev_t) if a == b)
    print(f"\n[Dev Loss] {dev_matches}/{len(dev_f)} eval points match exactly")
    for (sf, vf), (st, vt) in zip(dev_f, dev_t):
        status = "OK" if vf == vt else f"MISMATCH (diff={abs(vf-vt):.2e})"
        print(f"  Step {sf}: f={vf:.6f}, t={vt:.6f} — {status}")

    # 4. Token counts
    tokens_f = [(s, p["tokens_total"]) for p, s in logged_f]
    tokens_t = [(s, p["tokens_total"]) for p, s in logged_t]
    tok_matches = sum(1 for (_, a), (_, b) in zip(tokens_f, tokens_t) if a == b)
    print(f"\n[Token Counts] {tok_matches}/{len(tokens_f)} steps match")

    # 5. Checkpoints
    for step in (SAVE_STEPS, MAX_STEPS):
        n, fails = compare_checkpoints(dir_f, dir_t, step)
        total_compared += n
        total_failures.extend(fails)
        print(f"\n[Checkpoint step={step}] {n} weight tensors compared, {len(fails)} failures")
        for f in fails[:5]:
            print(f"  FAIL: {f}")

    # 6. Training state
    n, fails = compare_training_state(dir_f, dir_t)
    total_compared += n
    total_failures.extend(fails)
    print(f"\n[Training State] {n} fields compared, {len(fails)} failures")
    for f in fails[:5]:
        print(f"  FAIL: {f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    if total_failures:
        print(f"RESULT: FAILED — {len(total_failures)} failures out of {total_compared} comparisons")
        for f in total_failures:
            print(f"  - {f}")
    else:
        print(f"RESULT: PASSED — {total_compared} comparisons, 0 failures")
        print(f"  Functional: {time_f:.1f}s, Trainer: {time_t:.1f}s")
        print(f"  {len(losses_f)} loss values, {len(dev_f)} dev evals, "
              f"2 checkpoints, all bit-identical")

    print(f"\nArtifacts at: {base_dir}")

    # Cleanup on success
    if not total_failures:
        shutil.rmtree(base_dir, ignore_errors=True)
        print("(cleaned up — no failures)")

    return len(total_failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
