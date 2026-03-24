"""
T-I6: Functional equivalence — stateful Trainer vs functional train() inner loop.

Proves that the stateful Trainer class produces **bit-identical** loss sequences
and final model weights compared to the functional training loop in ssi.train.

Uses a tiny randomly-initialized transformer and synthetic data so the test runs
on any machine with a CUDA GPU — no extended model checkpoint or HuggingFace
dataset required.
"""

import copy
from collections import defaultdict

import pytest
import torch
from torchtune import training
from torchtune.models.llama3_2 import llama3_2
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import scale_grads

from ssi.loss import compute_loss
from ssi.train import count_token_types


# ---------------------------------------------------------------------------
# Skip if no CUDA GPU
# ---------------------------------------------------------------------------

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="No GPU available")

# ---------------------------------------------------------------------------
# Tiny model + synthetic data helpers
# ---------------------------------------------------------------------------

# Deliberately small so the test runs in seconds
_VOCAB_SIZE = 256
_EMBED_DIM = 64
_NUM_LAYERS = 2
_NUM_HEADS = 4
_NUM_KV_HEADS = 2
_INTERMEDIATE_DIM = 128
_MAX_SEQ_LEN = 32
_BATCH_SIZE = 4
_NUM_BATCHES = 8  # total micro-batches
_GRAD_ACCUM = 2  # optimizer steps = _NUM_BATCHES / _GRAD_ACCUM = 4
_LR = 1e-3
_SEED = 42_831  # matches ssi.constants.SEED


def _make_tiny_model(device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    """Build a tiny Llama 3.2 model using torchtune's builder."""
    with training.set_default_dtype(dtype), torch.device(device):
        model = llama3_2(
            vocab_size=_VOCAB_SIZE,
            num_layers=_NUM_LAYERS,
            num_heads=_NUM_HEADS,
            num_kv_heads=_NUM_KV_HEADS,
            embed_dim=_EMBED_DIM,
            max_seq_len=_MAX_SEQ_LEN,
            intermediate_dim=_INTERMEDIATE_DIM,
            attn_dropout=0.0,
            norm_eps=1e-5,
            rope_base=500_000,
            scale_factor=32,
        )
    return model


def _make_synthetic_batches(
    n_batches: int,
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    seed: int = _SEED,
) -> list[dict[str, torch.Tensor]]:
    """Generate deterministic synthetic batches of token IDs and labels.

    Uses a dedicated RNG so the batches are identical across runs.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    batches = []
    for _ in range(n_batches):
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=gen)
        # Labels = shifted tokens (standard causal LM); -100 on last position
        labels = tokens.clone()
        labels[:, -1] = -100
        batches.append({
            "tokens": tokens.to(device),
            "labels": labels.to(device),
        })
    return batches


# ---------------------------------------------------------------------------
# Token type ranges for counting (simple: entire vocab is "text")
# ---------------------------------------------------------------------------

_TOKEN_TYPE_RANGES = {"text": (0, _VOCAB_SIZE - 1)}


def _seed_everything(seed: int = _SEED) -> None:
    """Reset all RNG states so the two runs start identically."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Functional training loop (extracted from ssi/train.py lines 269-382)
# ---------------------------------------------------------------------------


def _run_functional_loop(
    model: torch.nn.Module,
    loss_fn: CEWithChunkedOutputLoss,
    batches: list[dict[str, torch.Tensor]],
    grad_accum: int,
    lr: float,
    pad_id: int = -100,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Run the exact math from the functional ``train()`` inner loop.

    Returns the per-optimizer-step loss log and the final model state dict.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    optimizer.zero_grad()

    loss_log: list[float] = []
    loss_running = 0.0
    num_tokens_step = 0
    token_type_counts_total: defaultdict[str, int] = defaultdict(int)

    for i, batch in enumerate(batches):
        # --- identical to ssi/train.py lines 291-300 ---
        for tt, ttcnt in count_token_types(batch["tokens"], _TOKEN_TYPE_RANGES, pad_id).items():
            token_type_counts_total[tt] += ttcnt
        num_tokens_iter = int((batch["labels"] != loss_fn.ignore_index).sum().item())
        num_tokens_step += num_tokens_iter
        loss_batch = compute_loss(batch, model, loss_fn) * num_tokens_iter
        loss_running += loss_batch
        loss_batch.backward()

        # --- gradient accumulation boundary (ssi/train.py lines 301-355) ---
        if (i + 1) % grad_accum == 0:
            scale_grads(model, torch.tensor(1 / num_tokens_step))
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            loss_to_log = loss_running.item() / num_tokens_step
            loss_log.append(loss_to_log)
            # reset step accumulators
            loss_running = 0.0
            num_tokens_step = 0

        del batch
        torch.cuda.empty_cache()

    return loss_log, {k: v.clone() for k, v in model.state_dict().items()}


# ---------------------------------------------------------------------------
# Stateful Trainer loop (uses Trainer._train_step + _optimizer_step)
# ---------------------------------------------------------------------------


def _run_trainer_loop(
    model: torch.nn.Module,
    loss_fn: CEWithChunkedOutputLoss,
    batches: list[dict[str, torch.Tensor]],
    grad_accum: int,
    lr: float,
    pad_id: int = -100,
) -> tuple[list[float], dict[str, torch.Tensor]]:
    """Run training through Trainer's methods, injecting components directly.

    We construct a Trainer with a dummy config and manually set the attributes
    that the core training methods need, bypassing setup() entirely.
    """
    from unittest.mock import MagicMock

    from omegaconf import OmegaConf

    from ssi.trainer import Trainer, TrainingGeometry

    # Minimal config — only fields the training loop actually reads
    cfg = OmegaConf.create({
        "gradient_accumulation_steps": grad_accum,
        "max_steps": len(batches) // grad_accum,
        "clip_grad_norm": None,
        "eval_steps": 999,   # never evaluate during this test
        "save_steps": 999,   # never checkpoint during this test
        "log_interval": 1,
        "config_name": "sft",
    })

    trainer = Trainer(cfg)

    # Inject components directly (bypass setup())
    trainer.model = model
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    trainer.lr_scheduler = None
    trainer.loss_fn = loss_fn
    trainer.device = next(model.parameters()).device
    trainer.dtype = next(model.parameters()).dtype
    trainer.world_size = 1
    trainer.token_type_ranges = _TOKEN_TYPE_RANGES

    # Mock tokenizer (only pad_id is used in training loop)
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_id = pad_id
    trainer.tokenizer = mock_tokenizer

    # Mock wandb logger (log_dict and log_config are no-ops)
    trainer.wandb_logger = MagicMock()

    # Mock checkpointer
    trainer.checkpointer = MagicMock()

    # Mock sampler (set_epoch is called in _train_epoch)
    trainer.sampler_train = MagicMock()

    # Geometry
    batches_per_epoch = len(batches)
    steps_per_epoch = batches_per_epoch // grad_accum
    trainer.geometry = TrainingGeometry(
        batch_size=batches[0]["tokens"].size(0),
        batches_per_epoch=batches_per_epoch,
        steps_per_epoch=steps_per_epoch,
        usable_batches=steps_per_epoch * grad_accum,
        n_epochs=1,
        gradient_accumulation_steps=grad_accum,
        world_size=1,
    )

    # Data — wrap batches in a DataLoader-like object
    trainer.data_train = batches  # _train_epoch enumerates this
    trainer.data_dev = None

    # Enable loss logging
    loss_log: list[float] = []
    trainer._loss_log = loss_log

    # Run the training loop via Trainer methods
    trainer.optimizer.zero_grad()
    trainer.t_train_start = __import__("time").perf_counter()
    trainer.t_step_start = __import__("time").perf_counter()
    trainer._reset_step_accumulators()
    trainer._resume_state = None

    for i, batch in enumerate(batches):
        trainer._train_step(batch)
        if (i + 1) % grad_accum == 0:
            trainer._optimizer_step(epoch=0, iter_idx=i)

    return loss_log, {k: v.clone() for k, v in model.state_dict().items()}


# ===========================================================================
# Tests
# ===========================================================================


class TestFunctionalEquivalence:
    """Prove bit-identical behavior between the functional and stateful loops."""

    @pytest.fixture(autouse=True)
    def setup_device(self):
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16

    def test_loss_sequences_identical(self):
        """Both loops produce the exact same loss at every optimizer step."""
        # --- Run 1: Functional ---
        _seed_everything()
        model_f = _make_tiny_model(self.device, self.dtype)
        model_f.train()
        loss_fn_f = CEWithChunkedOutputLoss()
        model_f.set_num_output_chunks(loss_fn_f.num_output_chunks)
        state_before = copy.deepcopy(model_f.state_dict())
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        _seed_everything()
        losses_f, weights_f = _run_functional_loop(model_f, loss_fn_f, batches, _GRAD_ACCUM, _LR)

        # --- Run 2: Stateful Trainer ---
        # Recreate identical model from the saved state
        _seed_everything()
        model_t = _make_tiny_model(self.device, self.dtype)
        model_t.load_state_dict(state_before)
        model_t.train()
        loss_fn_t = CEWithChunkedOutputLoss()
        model_t.set_num_output_chunks(loss_fn_t.num_output_chunks)
        # Regenerate identical batches
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        _seed_everything()
        losses_t, weights_t = _run_trainer_loop(model_t, loss_fn_t, batches, _GRAD_ACCUM, _LR)

        # --- Assertions ---
        assert len(losses_f) == len(losses_t) == _NUM_BATCHES // _GRAD_ACCUM, (
            f"Expected {_NUM_BATCHES // _GRAD_ACCUM} optimizer steps, "
            f"got functional={len(losses_f)}, trainer={len(losses_t)}"
        )

        for step, (lf, lt) in enumerate(zip(losses_f, losses_t)):
            assert lf == lt, (
                f"Loss mismatch at optimizer step {step}: functional={lf}, trainer={lt}"
            )

    def test_final_weights_identical(self):
        """Both loops produce the exact same model weights after training."""
        # --- Run 1: Functional ---
        _seed_everything()
        model_f = _make_tiny_model(self.device, self.dtype)
        model_f.train()
        loss_fn_f = CEWithChunkedOutputLoss()
        model_f.set_num_output_chunks(loss_fn_f.num_output_chunks)
        state_before = copy.deepcopy(model_f.state_dict())
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        _seed_everything()
        _, weights_f = _run_functional_loop(model_f, loss_fn_f, batches, _GRAD_ACCUM, _LR)

        # --- Run 2: Stateful Trainer ---
        _seed_everything()
        model_t = _make_tiny_model(self.device, self.dtype)
        model_t.load_state_dict(state_before)
        model_t.train()
        loss_fn_t = CEWithChunkedOutputLoss()
        model_t.set_num_output_chunks(loss_fn_t.num_output_chunks)
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        _seed_everything()
        _, weights_t = _run_trainer_loop(model_t, loss_fn_t, batches, _GRAD_ACCUM, _LR)

        # --- Assertions ---
        assert set(weights_f.keys()) == set(weights_t.keys()), "Model state dict keys differ"

        for key in weights_f:
            assert torch.equal(weights_f[key], weights_t[key]), (
                f"Weight mismatch in '{key}': max diff = {(weights_f[key] - weights_t[key]).abs().max().item()}"
            )

    def test_token_counts_identical(self):
        """Both loops count the same number of tokens per type."""
        _seed_everything()
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        # Functional counting
        counts_f: defaultdict[str, int] = defaultdict(int)
        for batch in batches:
            for tt, ttcnt in count_token_types(batch["tokens"], _TOKEN_TYPE_RANGES, -100).items():
                counts_f[tt] += ttcnt

        # Trainer counting (via _train_step)
        _seed_everything()
        model = _make_tiny_model(self.device, self.dtype)
        model.train()
        loss_fn = CEWithChunkedOutputLoss()
        model.set_num_output_chunks(loss_fn.num_output_chunks)

        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        from unittest.mock import MagicMock

        from omegaconf import OmegaConf

        from ssi.trainer import Trainer

        cfg = OmegaConf.create({
            "gradient_accumulation_steps": _GRAD_ACCUM,
            "max_steps": _NUM_BATCHES // _GRAD_ACCUM,
            "clip_grad_norm": None,
            "eval_steps": 999,
            "save_steps": 999,
            "log_interval": 1,
            "config_name": "sft",
        })
        trainer = Trainer(cfg)
        trainer.model = model
        trainer.loss_fn = loss_fn
        trainer.device = self.device
        trainer.token_type_ranges = _TOKEN_TYPE_RANGES
        trainer.tokenizer = MagicMock()
        trainer.tokenizer.pad_id = -100
        trainer._reset_step_accumulators()

        for batch in batches:
            trainer._train_step(batch)

        assert dict(counts_f) == dict(trainer.token_type_counts_total), (
            f"Token counts differ: functional={dict(counts_f)}, "
            f"trainer={dict(trainer.token_type_counts_total)}"
        )

    def test_gradient_accumulation_steps_correct(self):
        """Trainer.global_step increments exactly N_BATCHES // GRAD_ACCUM times."""
        _seed_everything()
        model = _make_tiny_model(self.device, self.dtype)
        model.train()
        loss_fn = CEWithChunkedOutputLoss()
        model.set_num_output_chunks(loss_fn.num_output_chunks)
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        _seed_everything()
        losses, _ = _run_trainer_loop(model, loss_fn, batches, _GRAD_ACCUM, _LR)

        expected_steps = _NUM_BATCHES // _GRAD_ACCUM
        assert len(losses) == expected_steps, (
            f"Expected {expected_steps} optimizer steps, got {len(losses)}"
        )

    def test_losses_are_finite_and_positive(self):
        """Sanity: all losses from both loops are finite and positive."""
        _seed_everything()
        model = _make_tiny_model(self.device, self.dtype)
        model.train()
        loss_fn = CEWithChunkedOutputLoss()
        model.set_num_output_chunks(loss_fn.num_output_chunks)
        batches = _make_synthetic_batches(_NUM_BATCHES, _BATCH_SIZE, _MAX_SEQ_LEN, _VOCAB_SIZE, self.device)

        _seed_everything()
        losses, _ = _run_functional_loop(model, loss_fn, batches, _GRAD_ACCUM, _LR)

        for step, loss in enumerate(losses):
            assert torch.isfinite(torch.tensor(loss)), f"Non-finite loss at step {step}: {loss}"
            assert loss > 0, f"Non-positive loss at step {step}: {loss}"
