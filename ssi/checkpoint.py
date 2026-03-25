from datetime import datetime, timezone
import gc
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
import torch
from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._checkpointer import _CheckpointerInterface
from torchtune.training.checkpointing._utils import (
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
    FormattedCheckpointFiles,
    check_outdir_not_in_ckptdir,
    copy_files,
    get_path,
    safe_torch_load,
)
from torchtune.training.metric_logging import WandBLogger

from ssi._version import __version__
from ssi.constants import (
    CHECKPOINT_VERSION,
    CHECKPOINT_VERSION_KEY,
    CONSUMED_SAMPLES_KEY,
    CUMULATIVE_METRICS_KEY,
    GLOBAL_STEP_KEY,
    LLAMA_3_2_CONFIG_RELPATH,
    LR_SCHEDULER_KEY,
    RNG_KEY,
    SEED_KEY,
    TRAINING_HPARAMS_KEY,
)


LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint auto-discovery and validation
# ---------------------------------------------------------------------------


def discover_safetensor_files(checkpoint_dir: Path) -> list[str]:
    """Auto-discover safetensors model shard files in a checkpoint directory.

    Finds ``*.safetensors`` files, excluding index files
    (``model.safetensors.index.json`` is not a shard). Raises if no files
    are found or if the directory contains ambiguous shard naming (e.g.
    both ``model-*`` and ``ft-model-*`` files).

    Args:
        checkpoint_dir: Directory to search.

    Returns:
        Sorted list of shard filenames (not full paths).

    Raises:
        FileNotFoundError: If the directory does not exist or is empty.
        ValueError: If no safetensors files found, or ambiguous naming.
    """
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    st_files = sorted(f.name for f in checkpoint_dir.glob("*.safetensors"))
    if not st_files:
        contents = sorted(f.name for f in checkpoint_dir.iterdir())
        raise ValueError(f"No safetensors files found in {checkpoint_dir}. Directory contents: {contents}")

    # Check for ambiguous naming: both model-* and ft-model-* present
    model_files = [f for f in st_files if f.startswith("model-")]
    ft_files = [f for f in st_files if f.startswith("ft-model-")]
    if model_files and ft_files:
        raise ValueError(
            f"Ambiguous checkpoint files in {checkpoint_dir}: "
            f"found both base shards {model_files} and fine-tuned shards {ft_files}. "
            f"Specify checkpoint_files explicitly to disambiguate."
        )

    LOGGER.info(f"Auto-discovered checkpoint file(s): {st_files}")
    return st_files


def validate_checkpoint_dir(checkpoint_dir: Path, config: dict[str, Any], expectations: Any | None = None) -> None:
    """Validate checkpoint directory contents against config.json and model expectations.

    Runs non-destructive checks before any weights are loaded. Failures
    are ``ValueError`` with actionable error messages.

    Args:
        checkpoint_dir: Directory containing the checkpoint.
        config: Parsed ``config.json`` dict from the checkpoint directory.
        expectations: A ``ModelCheckpointExpectations`` (or any object with
            ``model_name``, ``n_shards``, ``num_layers``, ``hidden_size``,
            ``vocab_size`` attributes). If ``None``, model-specific checks
            are skipped.

    Raises:
        ValueError: If any check fails.
    """
    # Layer 1: config.json existence (already loaded by caller, but check it's non-empty)
    if not config:
        raise ValueError(f"config.json in {checkpoint_dir} is empty or could not be parsed.")

    # Layer 2: index file vs shard consistency
    index_path = checkpoint_dir / SAFETENSOR_INDEX_FNAME
    if index_path.exists():
        index_data = json.loads(index_path.read_text())
        expected_shards = set(index_data.get("weight_map", {}).values())
        actual_shards = {f.name for f in checkpoint_dir.glob("*.safetensors")}
        missing = expected_shards - actual_shards
        if missing:
            raise ValueError(
                f"Shard mismatch in {checkpoint_dir}: index file lists shards "
                f"{sorted(expected_shards)} but directory is missing {sorted(missing)}."
            )

    # Layer 3: model-specific structural checks
    if expectations is None:
        return

    # Shard count
    st_files = sorted(checkpoint_dir.glob("*.safetensors"))
    n_shards = len(st_files)
    if n_shards != expectations.n_shards:
        raise ValueError(
            f"Expected {expectations.n_shards} model shard(s) for "
            f"{expectations.model_name} but found {n_shards} in {checkpoint_dir}. "
            f"Check that checkpoint_dir points to the correct model."
        )

    # Architecture fields from config.json
    config_layers = config.get("num_hidden_layers")
    if config_layers is not None and config_layers != expectations.num_layers:
        raise ValueError(
            f"config.json reports num_hidden_layers={config_layers} but "
            f"{expectations.model_name} has {expectations.num_layers}. Wrong model?"
        )

    config_hidden = config.get("hidden_size")
    if config_hidden is not None and config_hidden != expectations.hidden_size:
        raise ValueError(
            f"config.json reports hidden_size={config_hidden} but "
            f"{expectations.model_name} has {expectations.hidden_size}. Wrong model?"
        )

    config_vocab = config.get("vocab_size")
    if config_vocab is not None and config_vocab != expectations.vocab_size:
        raise ValueError(
            f"Vocab size mismatch: config.json has vocab_size={config_vocab}, "
            f"expected {expectations.vocab_size} for {expectations.model_name} "
            f"with current speech config. Was the model extended with different "
            f"n_dsus or modality token settings?"
        )

    LOGGER.info(
        f"Checkpoint validation passed for {expectations.model_name} "
        f"({n_shards} shard(s), {config_layers} layers, vocab_size={config_vocab})"
    )


def validate_checkpoint_files(checkpoint_files: list[str], input_dir: Path, missing_ok=False) -> list[Path]:
    """Validate that checkpoint files exist in input_dir and return paths sorted by name."""
    checkpoint_paths: list[Path] = []
    for f in checkpoint_files:
        checkpoint_path = get_path(input_dir, f, missing_ok)
        checkpoint_paths.append(checkpoint_path)
    return sorted(checkpoint_paths)


def get_model_checkpoint_paths(checkpoint_files: list[str] | dict[str, str], checkpoint_dir: Path) -> list[Path]:
    """Resolve checkpoint file names to sorted, validated paths under checkpoint_dir."""
    if not isinstance(checkpoint_files, list):
        formatted_checkpoint_files = FormattedCheckpointFiles.from_dict(checkpoint_files)
        checkpoint_files = formatted_checkpoint_files.build_checkpoint_filenames()
    return validate_checkpoint_files(checkpoint_files, input_dir=checkpoint_dir, missing_ok=False)


def save_rng_states() -> dict[str, Any]:
    """Capture Python, NumPy, and PyTorch RNG states including CUDA if available."""
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def restore_rng_states(rng_state: dict[str, Any]) -> None:
    """Restore Python, NumPy, and PyTorch RNG states from a dict produced by save_rng_states."""
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy_global"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


class FullModelHFCheckpointer(_CheckpointerInterface):
    """Reads and writes HF-format checkpoints with torchtune key conversion for Llama 3.2.

    Checkpoint files are sorted by shard ID before reading. Conversion between HF and
    torchtune key formats uses model params read from ``config.json``.

    Args:
        checkpoint_dir: Directory containing the source checkpoint files.
        checkpoint_files: Checkpoint file names (list), a dict with keys
            ``filename_format`` and ``max_filename``, or ``None`` to
            auto-discover safetensors files in ``checkpoint_dir``.
        config_json: Path to the model ``config.json``. Defaults to the Llama 3.2
            config relative path under checkpoint_dir.
        output_dir: Root directory for saved checkpoints and training state.
        training_state_checkpoint: Path to a ``training_state.pt`` file for resuming training.
            None when starting from scratch.
        safe_serialization: If True (default), save weights as safetensors; otherwise
            save as ``.bin`` via ``torch.save``.
        model_expectations: Optional ``ModelCheckpointExpectations`` (or duck-typed
            equivalent) for model-specific validation. When provided, the
            checkpoint directory is validated against expected shard count,
            layer count, hidden size, and vocab size before loading.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        checkpoint_files: list[str] | dict[str, str] | None = None,
        *,
        config_json: Path | str | None = None,
        output_dir: Path | str,
        training_state_checkpoint: Path | str | None = None,
        safe_serialization: bool = True,
        model_expectations: Any | None = None,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.safe_serialization = safe_serialization
        self.output_dir = Path(output_dir)  # idempotent
        self.training_state_checkpoint = (
            Path(training_state_checkpoint) if training_state_checkpoint is not None else None
        )
        if isinstance(checkpoint_files, ListConfig):
            checkpoint_files = OmegaConf.to_object(checkpoint_files)

        check_outdir_not_in_ckptdir(ckpt_dir=self.checkpoint_dir, out_dir=self.output_dir)

        if self.training_state_checkpoint is not None and not self.training_state_checkpoint.is_file():
            raise FileNotFoundError(f"Recipe checkpoint file {self.training_state_checkpoint} not found.")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # weight_map: state_dict key -> checkpoint mapping to partition state dict into output checkpoint files
        self._weight_map: dict[str, str] | None = None  # NOTE initialised to None; updated during checkpoint loading

        if config_json is None:
            config_json = self.checkpoint_dir / LLAMA_3_2_CONFIG_RELPATH
        config_json = Path(config_json)
        if not config_json.exists():
            raise FileNotFoundError(f"No config.json found at {config_json} — expected an HF-format model directory.")
        self._config = json.loads(config_json.read_text())

        # Auto-discover checkpoint files if not provided
        if checkpoint_files is None:
            checkpoint_files = discover_safetensor_files(self.checkpoint_dir)

        # Validate checkpoint directory against model expectations
        validate_checkpoint_dir(self.checkpoint_dir, self._config, model_expectations)

        self._checkpoint_paths = get_model_checkpoint_paths(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self.checkpoint_dir,
        )

        LOGGER.info(f"Resuming from checkpoint(s): {[str(path) for path in self._checkpoint_paths]}")
        if self.training_state_checkpoint is not None:
            LOGGER.info(f"Resuming optimizer and training state from: {self.training_state_checkpoint}")
        else:
            LOGGER.info("No training state checkpoint passed. Will initialize optimizer state from scratch.")

    def load_checkpoint(self) -> dict[str, Any]:
        """Load and merge HF checkpoint shards, converting to torchtune key format.

        Populates ``_weight_map`` (key -> shard ID) so that ``save_model_checkpoint``
        can partition weights back into the same shard layout. If ``training_state_checkpoint``
        was provided at init, the training state is merged into the returned dict.

        Raises:
            ValueError: If any value in a checkpoint file is not a ``torch.Tensor``.
        """
        self._weight_map = {}

        # merged state_dict contains keys and weights from all the checkpoint files
        merged_state_dict: dict[str, torch.Tensor] = {}

        # converted_state_dict is the final state_dict after keys are converted into
        # the torchtune format. This optionally also contains the training state.
        converted_state_dict: dict[str, Any] = {}

        # _checkpoint_paths are already sorted so simply enumerate to generate the right id
        for cpt_idx, cpt_path in enumerate(self._checkpoint_paths):
            state_dict = safe_torch_load(cpt_path)
            for key, value in state_dict.items():
                # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                # will break downstream code
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Expected all values in the state dict to be torch.Tensor. Found {type(value)} instead."
                    )
                # idx is written in the 4 digit format (eg: 0001, 0002, etc.)
                self._weight_map[key] = f"{cpt_idx + 1:04}"
            merged_state_dict.update(state_dict)

            # delete the state_dict to free up memory; TODO check if this del is needed
            del state_dict
            gc.collect()

        converted_state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config.get("head_dim", None),
        )

        if self.training_state_checkpoint is not None:
            training_state = safe_torch_load(self.training_state_checkpoint, mmap=False)
            converted_state_dict.update(training_state)

        return converted_state_dict

    def save_full_model(self, state_dict: dict[str, Any], output_dir: Path) -> None:
        """Convert torchtune weights to HF format and write sharded checkpoint files.

        Uses ``_weight_map`` (populated by ``load_checkpoint``) to partition weights
        across shards. Format (safetensors vs .bin) is controlled by
        ``self.safe_serialization``. Does not modify ``state_dict``.

        Raises:
            ValueError: If ``_weight_map`` has not been initialised by a prior load.
        """
        if self._weight_map is None:
            raise ValueError("Weight map is not initialized. Please load a checkpoint before saving.")

        hf_model_state_dict = convert_weights.tune_to_hf(
            state_dict[training.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config.get("head_dim", None),
        )

        # split the state_dict into separate dicts, one for each output checkpoint file, by _weight_map
        split_state_dicts: dict[str, dict[str, torch.Tensor]] = {}
        total_size = 0
        for key, weight in hf_model_state_dict.items():
            cpt_idx = self._weight_map[key]
            # initialize dict
            if cpt_idx not in split_state_dicts:
                split_state_dicts[cpt_idx] = {}
            split_state_dicts[cpt_idx].update({key: weight})
            total_size += weight.numel() * weight.element_size()

        # write the partitioned state dicts to the right checkpoint file
        # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc.
        num_shards = len(split_state_dicts)
        map_original_name_to_new_name = {}
        output_dir.mkdir(parents=True, exist_ok=True)
        for cpt_idx, model_state_dict in split_state_dicts.items():
            # TODO: We should probably use the original shard name and just add a prefix
            # however, having the SHARD_FNAME standardizes our checkpoints
            shard_name = SHARD_FNAME.format(cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5))
            map_original_name_to_new_name[cpt_idx] = shard_name
            output_path = output_dir / shard_name
            if not self.safe_serialization:
                output_path = output_path.with_suffix(".bin")
                torch.save(model_state_dict, output_path)
            else:
                output_path = output_path.with_suffix(".safetensors")
                save_file(model_state_dict, output_path, metadata={"format": "pt"})
            _ckpt_sz = os.path.getsize(output_path) / 1024**3
            LOGGER.info(f"Model checkpoint of size {_ckpt_sz:.2f} GiB saved to {output_path}")

        # Save the appropriate index file based on serialization format; example:
        # {
        #     "metadata": {"total_size": 1234},
        #     "weight_map": {"key1": "model_0001.safetensors", "key2": "model_0002.safetensors"},
        # }
        if self.safe_serialization:
            weight_map = {
                k: map_original_name_to_new_name[cpt_idx] + ".safetensors" for k, cpt_idx in self._weight_map.items()
            }
            index_file_name = SAFETENSOR_INDEX_FNAME
        else:
            weight_map = {k: map_original_name_to_new_name[cpt_idx] + ".bin" for k, cpt_idx in self._weight_map.items()}
            index_file_name = TORCH_INDEX_FNAME

        index_path = output_dir / index_file_name
        index_data = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(index_path, "w") as f:
            json.dump(index_data, f, indent=2)

        LOGGER.info(f"The full model checkpoint has been saved to {output_dir}")

    def save_model_checkpoint(
        self,
        model_state_dict: dict[str, Any],
        global_step: int,
        *,
        output_dir: Path | None = None,
        ignore_suffixes: list[str] | None = None,
    ) -> Path:
        """Save model weights to a self-contained ``step_N/`` directory.

        Writes sharded HF-format weights (safetensors or .bin per
        ``self.safe_serialization``) and copies config, tokenizer, and other
        non-weight files so the directory is directly usable by HF tooling.
        """
        if output_dir is None:
            output_dir = self.output_dir / f"step_{global_step}"
        if ignore_suffixes is None:
            ignore_suffixes = [*SUFFIXES_TO_NOT_COPY, "torchtune_config.yaml"]
        state_dict = {training.MODEL_KEY: model_state_dict}
        self.save_full_model(state_dict, output_dir)
        copy_files(self.checkpoint_dir, output_dir, ignore_suffixes=ignore_suffixes)
        return output_dir

    def save_training_state(
        self,
        *,
        optimizer_state_dict: dict[str, Any],
        lr_scheduler_state_dict: dict[str, Any] | None,
        global_step: int,
        seed: int,
        training_hparams: dict[str, Any],
        consumed_samples: int,
        cumulative_metrics: dict[str, Any],
    ) -> Path:
        """Save training resume state to ``training_state.pt`` at ``self.output_dir``.

        Always overwrites the previous file. All fields except
        ``lr_scheduler_state_dict`` are mandatory — a checkpoint written by this
        method is guaranteed to be resumable.
        """
        state_dict = {
            CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION,
            GLOBAL_STEP_KEY: global_step,
            SEED_KEY: seed,
            training.OPT_KEY: optimizer_state_dict,
            LR_SCHEDULER_KEY: lr_scheduler_state_dict,
            RNG_KEY: save_rng_states(),
            TRAINING_HPARAMS_KEY: training_hparams,
            CONSUMED_SAMPLES_KEY: consumed_samples,
            CUMULATIVE_METRICS_KEY: cumulative_metrics,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ssi_version": __version__,
        }
        output_path = self.output_dir / "training_state.pt"
        torch.save(state_dict, output_path)
        LOGGER.info(f"Training state ({os.path.getsize(output_path) / 1024**3:.2f} GiB) saved to {output_path}")
        return output_path


def resolve_checkpointer_output_dir(cfg: DictConfig, wandb_logger: WandBLogger) -> Path:
    """Build the checkpoint output path as ``{cfg.output_dir}/{run_name}-id_{run_id}/checkpoints``."""
    if wandb_logger._wandb.run is None:
        raise RuntimeError("wandb run not initialized")
    run_name = wandb_logger._wandb.run.name
    run_id = wandb_logger._wandb.run.id
    return Path(cfg.output_dir, f"{run_name}-id_{run_id}", "checkpoints")
