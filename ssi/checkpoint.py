import gc
import json
import logging
import os
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._checkpointer import _CheckpointerInterface
from torchtune.training.checkpointing._utils import (
    check_outdir_not_in_ckptdir,
    copy_files,
    FormattedCheckpointFiles,
    get_path,
    safe_torch_load,
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
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
    MODEL_KEY,
    RNG_KEY,
    SEED_KEY,
    TRAINING_HPARAMS_KEY,
)


LOGGER = logging.getLogger(__name__)


def validate_checkpoint_files(checkpoint_files: list[str], input_dir: Path, missing_ok=False) -> list[Path]:
    """Validates that the checkpoint files exist and sorts based on ID"""
    checkpoint_paths: list[Path] = []
    for f in checkpoint_files:
        checkpoint_path = get_path(input_dir, f, missing_ok)
        checkpoint_paths.append(checkpoint_path)
    return sorted(checkpoint_paths)


def get_model_checkpoint_paths(checkpoint_files: list[str] | dict[str, str], checkpoint_dir: Path) -> list[Path]:
    if not isinstance(checkpoint_files, list):
        formatted_checkpoint_files = FormattedCheckpointFiles.from_dict(checkpoint_files)
        checkpoint_files = formatted_checkpoint_files.build_checkpoint_filenames()
    return validate_checkpoint_files(checkpoint_files, input_dir=checkpoint_dir, missing_ok=False)


def save_rng_states() -> dict[str, Any]:
    """Capture the 4 standard framework RNG states for checkpointing."""
    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy_global": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return rng_state


def restore_rng_states(rng_state: dict[str, Any]) -> None:
    """Restore the 4 standard framework RNG states from a checkpoint."""
    random.setstate(rng_state["python"])
    np.random.set_state(rng_state["numpy_global"])
    torch.set_rng_state(rng_state["torch_cpu"])
    if "torch_cuda" in rng_state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_state["torch_cuda"])


class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in HF's format for Llama 3.2 1B.

    Note:
        HF checkpoint names are usually ordered by ID (eg: 0001_of_0003, 0002_of_0003, etc.) To ensure \
        we read the files in the right order, we sort the checkpoint file names before reading.

    Note:
        Checkpoint conversion to and from HF's format requires access to model params which are \
        read directly from the ``config.json`` file. This helps ensure we either load the weights \
        correctly or error out in case of discrepancy between the HF checkpoint file and torchtune's \
        model implementations.

    Args:
        checkpoint_dir (Path): Directory containing the checkpoint files
        checkpoint_files (list[str] | dict[str, str]): List of checkpoint files to load or a dictionary
            containing the keys keys ["filename_format", "max_filename"]. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter.
        config_json (Path): Path to the model config JSON file. Required for state dict conversion.
        output_dir (str): Directory to save the checkpoint files
        recipe_checkpoint (Path | None): Path to the recipe state checkpoint file. Default is None.
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`.
            Default is True.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        checkpoint_files: list[str] | dict[str, str],
        *,
        config_json: Path | str | None = None,
        output_dir: Path | str,
        recipe_checkpoint: Path | str | None = None,
        safe_serialization: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.safe_serialization = safe_serialization
        self.output_dir = Path(output_dir)  # idempotent
        self.recipe_checkpoint = Path(recipe_checkpoint) if recipe_checkpoint is not None else None
        if isinstance(checkpoint_files, ListConfig):
            checkpoint_files = OmegaConf.to_object(checkpoint_files)

        check_outdir_not_in_ckptdir(ckpt_dir=self.checkpoint_dir, out_dir=self.output_dir)

        if self.recipe_checkpoint is not None and not self.recipe_checkpoint.is_file():
            raise FileNotFoundError(f"Recipe checkpoint file {self.recipe_checkpoint} not found.")

        self.output_dir.mkdir(parents=True, exist_ok=True)  # TODO

        # weight_map: state_dict key -> checkpoint mapping to partition state dict into output checkpoint files
        self._weight_map: dict[str, str] | None = None  # NOTE initialised to None; updated during checkpoint loading

        if config_json is None:
            config_json = self.checkpoint_dir / LLAMA_3_2_CONFIG_RELPATH
        self._config = json.loads(Path(config_json).read_text())  # gives model params needed for state dict conversion

        self._checkpoint_paths = get_model_checkpoint_paths(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self.checkpoint_dir,
        )

        LOGGER.info(f"Resuming from checkpoint(s): {[str(path) for path in self._checkpoint_paths]}")
        if self.recipe_checkpoint is not None:
            LOGGER.info(f"Resuming optimizer and recipe state from: {self.recipe_checkpoint}")
        else:
            LOGGER.info("No recipe state checkpoint passed. Will initialize optimizer state from scratch.")

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        Llama 3.2 conversion.

        Returns:
            state_dict (Dict[str, Any]): torchtune checkpoint state dict

        Raises:
            ValueError: If the values in the input state_dict are not Tensors
        """
        self._weight_map = {}

        # merged state_dict contains keys and weights from all the checkpoint files
        merged_state_dict: Dict[str, torch.Tensor] = {}

        # converted_state_dict is the final state_dict passed to the recipe after the
        # keys are converted into the torchtune format. This optionally also contains
        # the recipe state
        converted_state_dict: Dict[str, Dict[str, torch.Tensor]] = {}

        # _checkpoint_paths are already sorted so simply enumerate to generate the right id
        for cpt_idx, cpt_path in enumerate(self._checkpoint_paths):
            state_dict = safe_torch_load(cpt_path)
            for key, value in state_dict.items():
                # Ensure that the state dict is a flat dict of keys and tensors. Breaking this assumption
                # will break recipe code
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"Expected all values in the state dict to be torch.Tensor. " f"Found {type(value)} instead."
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

        if self.recipe_checkpoint is not None:
            recipe_state = safe_torch_load(self.recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)

        return converted_state_dict

    def save_full_model(self, state_dict: dict[str, Any], output_dir: Path) -> None:
        if self._weight_map is None:
            raise ValueError("Weight map is not initialized. Please load a checkpoint before saving.")

        state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
            state_dict[training.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config.get("head_dim", None),
        )

        # split the state_dict into separate dicts, one for each output checkpoint file, by _weight_map
        split_state_dicts: Dict[str, Dict[str, torch.Tensor]] = {}
        total_size = 0
        for key, weight in state_dict[training.MODEL_KEY].items():
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

    def save_recipe_state(self, state_dict: dict[str, Any]) -> None:
        """Save training state to a single file at the top-level output directory.

        Always overwrites the previous file — only the latest training state is kept
        on disk. Model weights are saved separately per checkpoint step.
        """
        output_path = self.output_dir / "recipe_state.pt"
        exclude_keys = (training.MODEL_KEY,)
        torch.save({k: v for k, v in state_dict.items() if k not in exclude_keys}, output_path)
        LOGGER.info(f"Recipe state ({os.path.getsize(output_path) / 1024**3:.2f} GiB) saved to {output_path}")

    def _save_checkpoint(
        self,
        state_dict: dict[str, Any],
        output_dir: Path,
        save_training_state: bool,
        ignore_suffixes: list[str],
    ) -> None:
        self.save_full_model(state_dict, output_dir)

        # Save all files in ckpt_dir except model weights and mapping -> facilitate inference
        copy_files(self.checkpoint_dir, output_dir, ignore_suffixes=ignore_suffixes)

        if save_training_state:
            self.save_recipe_state(state_dict)
        else:
            LOGGER.info("No training state saved.")

    def save_checkpoint(
        self,
        model_state_dict: dict[str, Any],
        optimizer_state_dict: dict[str, Any] | None,
        global_step: int,
        seed: int,
        *,
        lr_scheduler_state_dict: dict[str, Any] | None = None,
        training_hparams: dict[str, Any] | None = None,
        consumed_samples: int = 0,
        cumulative_metrics: dict[str, Any] | None = None,
        save_training_state: bool = True,
        output_dir: Path | None = None,
        ignore_suffixes: list[str] | None = None,
    ) -> tuple[dict[str, Any], Path]:
        if ignore_suffixes is None:
            ignore_suffixes = SUFFIXES_TO_NOT_COPY + ["torchtune_config.yaml"]
        ckpt_dict: dict[str, Any] = {
            MODEL_KEY: model_state_dict,
            GLOBAL_STEP_KEY: global_step,
            SEED_KEY: seed,
            CHECKPOINT_VERSION_KEY: CHECKPOINT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ssi_version": __version__,
        }
        if optimizer_state_dict is not None:
            ckpt_dict[training.OPT_KEY] = optimizer_state_dict
        if lr_scheduler_state_dict is not None:
            ckpt_dict[LR_SCHEDULER_KEY] = lr_scheduler_state_dict
        ckpt_dict[RNG_KEY] = save_rng_states()
        if training_hparams is not None:
            ckpt_dict[TRAINING_HPARAMS_KEY] = training_hparams
        ckpt_dict[CONSUMED_SAMPLES_KEY] = consumed_samples
        if cumulative_metrics is not None:
            ckpt_dict[CUMULATIVE_METRICS_KEY] = cumulative_metrics
        if output_dir is None:
            output_dir = self.output_dir / f"step_{global_step}"
        self._save_checkpoint(
            ckpt_dict,
            output_dir=output_dir,
            save_training_state=save_training_state,
            ignore_suffixes=ignore_suffixes,
        )
        return ckpt_dict, output_dir


def resolve_checkpointer_output_dir(cfg: DictConfig, wandb_logger: WandBLogger) -> Path:
    if wandb_logger._wandb.run is None:
        raise RuntimeError("wandb run not initialized")
    run_name = wandb_logger._wandb.run.name
    run_id = wandb_logger._wandb.run.id
    return Path(cfg.output_dir, f"{run_name}-id_{run_id}", "checkpoints")
