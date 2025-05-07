import gc
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from safetensors.torch import save_file
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torchtune import training
from torchtune.models import convert_weights
from torchtune.training.checkpointing._checkpointer import _CheckpointerInterface
from torchtune.training.checkpointing._utils import (
    ADAPTER_CONFIG_FNAME,
    ADAPTER_MODEL_FNAME,
    check_outdir_not_in_ckptdir,
    copy_files,
    FormattedCheckpointFiles,
    get_path,
    ModelType,
    REPO_ID_FNAME,
    safe_torch_load,
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
)
from torchtune.training.metric_logging import WandBLogger

import ssi.constants


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
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


class FullModelHFCheckpointer(_CheckpointerInterface):
    """
    Checkpointer which reads and writes checkpoints in HF's format. For LoRA models this includes
    saving checkpoints in a format that can be loaded into PEFT via e.g. ``from_pretrained``. Examples include
    the Llama-2-7b-hf model from the meta-llama repo (https://huggingface.co/meta-llama/Llama-2-7b-hf).

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
        adapter_checkpoint (Path | None): Path to the adapter weights. Default is None.
        recipe_checkpoint (Path | None): Path to the recipe state checkpoint file. Default is None.
        model_type (ModelType): Model type. Default is ModelType.LLAMA3_2
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`.
            Default is True.
    """

    def __init__(
        self,
        checkpoint_dir: Path | str,
        checkpoint_files: list[str] | dict[str, str],
        config_json: Path | str,
        output_dir: Path | str,
        recipe_checkpoint: Path | str | None = None,
        adapter_checkpoint: Path | str | None = None,
        model_type: str = "llama3_2",
        safe_serialization: bool = True,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.safe_serialization = safe_serialization
        self.model_type: ModelType = ModelType(model_type)
        self.output_dir = Path(output_dir)  # idempotent
        self.recipe_checkpoint = Path(recipe_checkpoint) if recipe_checkpoint is not None else None
        self.adapter_checkpoint = Path(adapter_checkpoint) if adapter_checkpoint else None
        if isinstance(checkpoint_files, ListConfig):
            checkpoint_files = OmegaConf.to_object(checkpoint_files)

        check_outdir_not_in_ckptdir(ckpt_dir=self.checkpoint_dir, out_dir=self.output_dir)

        if self.recipe_checkpoint is not None and not self.recipe_checkpoint.is_file():
            raise FileNotFoundError(f"Recipe checkpoint file {self.recipe_checkpoint} not found.")

        self.output_dir.mkdir(parents=True, exist_ok=True)  # TODO

        # weight_map: state_dict key -> checkpoint mapping to partition state dict into output checkpoint files
        self._weight_map: dict[str, str] | None = None  # NOTE initialised to None; updated during checkpoint loading

        self._config = json.loads(Path(config_json).read_text())  # gives model params needed for state dict conversion

        # repo_id necessary to save adapter config for HF compatibility. JSON produced and saved at download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = Path.joinpath(self.checkpoint_dir, REPO_ID_FNAME).with_suffix(".json")
        self.repo_id = None
        if repo_id_path.exists():
            with open(repo_id_path, "r") as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

        self._checkpoint_paths = get_model_checkpoint_paths(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self.checkpoint_dir,
        )

        LOGGER.info(f"Resuming from checkpoint(s): {[str(path) for path in self._checkpoint_paths]}")
        if self.recipe_checkpoint is not None:
            LOGGER.info(f"Resuming optimizer and recipe state from: {self.recipe_checkpoint}")
        else:
            LOGGER.info("No recipe state checkpoint passed. Will initialize optimizer state from scratch.")
        if self.adapter_checkpoint:
            LOGGER.info(f"Resuming adapter from checkpoint: {self.adapter_checkpoint}")

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        the appropriate convert_weights function (depending on ``self.model_type``).

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
        # the recipe state and adapter weights
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

        match self.model_type:
            case ModelType.PHI3_MINI:
                self.phi3_hf_to_tune(merged_state_dict, converted_state_dict)
            case ModelType.REWARD:
                self.reward_hf_to_tune(merged_state_dict, converted_state_dict)
            case ModelType.QWEN2:
                self.qwen2_hf_to_tune(merged_state_dict, converted_state_dict)
            case ModelType.LLAMA3_VISION:
                self.llama3_vision_hf_to_tune(merged_state_dict, converted_state_dict)
            case ModelType.CLIP_TEXT:
                self.clip_text_hf_to_tune(merged_state_dict, converted_state_dict)
            case ModelType.GEMMA2:
                self.gemma2_hf_to_tune(merged_state_dict, converted_state_dict)
            case ModelType.LLAMA2 | ModelType.LLAMA3 | ModelType.LLAMA3_2:
                converted_state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
                    merged_state_dict,
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        if self.adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self.adapter_checkpoint)
            converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self.recipe_checkpoint is not None:
            recipe_state = safe_torch_load(self.recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)

        return converted_state_dict

    def phi3_hf_to_tune(self, merged_state_dict, converted_state_dict):
        LOGGER.info(
            "Converting Phi-3 Mini weights from HF format."
            "Note that conversion of adapter weights into PEFT format is not supported."
        )
        from torchtune.models.phi3._convert_weights import phi3_hf_to_tune

        converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(merged_state_dict)
        return converted_state_dict

    def phi3_tune_to_hf(self, state_dict):
        from torchtune.models.phi3._convert_weights import phi3_tune_to_hf

        state_dict[training.MODEL_KEY] = phi3_tune_to_hf(state_dict[training.MODEL_KEY])

    def reward_hf_to_tune(self, merged_state_dict, converted_state_dict):
        from torchtune.rlhf.utils import reward_hf_to_tune

        converted_state_dict[training.MODEL_KEY] = reward_hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
        )
        return converted_state_dict

    def reward_tune_to_hf(self, state_dict):
        from torchtune.rlhf.utils import reward_tune_to_hf

        state_dict[training.MODEL_KEY] = reward_tune_to_hf(
            state_dict[training.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
        )

    def qwen2_hf_to_tune(self, merged_state_dict, converted_state_dict):
        from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune

        converted_state_dict[training.MODEL_KEY] = qwen2_hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            tie_word_embeddings=self._config["tie_word_embeddings"],
        )
        return converted_state_dict

    def qwen2_tune_to_hf(self, state_dict):
        from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

        state_dict[training.MODEL_KEY] = qwen2_tune_to_hf(
            state_dict[training.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            tie_word_embeddings=self._config["tie_word_embeddings"],
        )

    def llama3_vision_hf_to_tune(self, merged_state_dict, converted_state_dict):
        from torchtune.models.llama3_2_vision._convert_weights import llama3_vision_hf_to_tune

        text_config = self._config.get("text_config", {})
        vision_config = self._config.get("vision_config", {})
        converted_state_dict[training.MODEL_KEY] = llama3_vision_hf_to_tune(
            merged_state_dict,
            num_heads=text_config["num_attention_heads"],
            num_kv_heads=text_config["num_key_value_heads"],
            dim=text_config["hidden_size"],
            head_dim=text_config.get("head_dim", None),
            vocab_size=text_config["vocab_size"],
            cross_attention_layers=text_config.get("cross_attention_layers", None),
            encoder_dim=vision_config["hidden_size"],
            tile_size=vision_config["image_size"],
            num_tiles=vision_config["max_num_tiles"],
            supported_aspect_ratios=vision_config.get("supported_aspect_ratios", None),
        )
        return converted_state_dict

    def llama3_vision_tune_to_hf(self, state_dict):
        from torchtune.models.llama3_2_vision._convert_weights import llama3_vision_tune_to_hf

        text_config = self._config.get("text_config", {})
        vision_config = self._config.get("vision_config", {})
        state_dict[training.MODEL_KEY] = llama3_vision_tune_to_hf(
            state_dict[training.MODEL_KEY],
            num_heads=text_config["num_attention_heads"],
            num_kv_heads=text_config["num_key_value_heads"],
            dim=text_config["hidden_size"],
            head_dim=text_config.get("head_dim", None),
            vocab_size=text_config["vocab_size"],
            cross_attention_layers=text_config.get("cross_attention_layers", None),
            encoder_dim=vision_config["hidden_size"],
            tile_size=vision_config["image_size"],
            num_tiles=vision_config["max_num_tiles"],
            supported_aspect_ratios=vision_config.get("supported_aspect_ratios", None),
        )

    def clip_text_hf_to_tune(self, merged_state_dict, converted_state_dict):
        from torchtune.models.clip._convert_weights import clip_text_hf_to_tune

        converted_state_dict[training.MODEL_KEY] = clip_text_hf_to_tune(merged_state_dict)
        return converted_state_dict

    def clip_text_tune_to_hf(self, state_dict):
        raise NotImplementedError

    def gemma2_hf_to_tune(self, merged_state_dict, converted_state_dict):
        from torchtune.models.gemma2._convert_weights import gemma2_hf_to_tune

        converted_state_dict[training.MODEL_KEY] = gemma2_hf_to_tune(
            merged_state_dict,
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config.get("head_dim", None),
        )
        return converted_state_dict

    def gemma2_tune_to_hf(self, state_dict):
        from torchtune.models.gemma2._convert_weights import gemma2_tune_to_hf

        state_dict[training.MODEL_KEY] = gemma2_tune_to_hf(
            state_dict[training.MODEL_KEY],
            num_heads=self._config["num_attention_heads"],
            num_kv_heads=self._config["num_key_value_heads"],
            dim=self._config["hidden_size"],
            head_dim=self._config.get("head_dim", None),
        )

    def save_full_model(self, state_dict: dict[str, Any], output_dir: Path) -> None:
        if self._weight_map is None:
            raise ValueError("Weight map is not initialized. Please load a checkpoint before saving.")

        match self.model_type:
            case ModelType.PHI3_MINI:
                self.phi3_tune_to_hf(state_dict)
            case ModelType.REWARD:
                self.reward_tune_to_hf(state_dict)
            case ModelType.QWEN2:
                self.qwen2_tune_to_hf(state_dict)
            case ModelType.LLAMA3_VISION:
                self.llama3_vision_tune_to_hf(state_dict)
            case ModelType.GEMMA2:
                self.gemma2_tune_to_hf(state_dict)
            case ModelType.CLIP_TEXT:
                raise NotImplementedError("Clip text conversion is not supported yet")  # strangely
            case ModelType.LLAMA2 | ModelType.LLAMA3 | ModelType.LLAMA3_2:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

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

    def save_adapter_weights(self, state_dict: dict[str, Any], output_dir: Path) -> None:
        # TODO [Meta] saving it "as is" is a requirement because, if we only save with
        # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
        # convert_weights.peft_to_tune. The .pt format is not needed, but
        # it is an easy way to distinguish the adapters. Ideally we should save only one.
        output_path = (output_dir / ADAPTER_MODEL_FNAME).with_suffix(".pt")
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state_dict[training.ADAPTER_KEY], output_path)
        _ckpt_sz = os.path.getsize(output_path) / 1024**3
        LOGGER.info(f"Adapter checkpoint of size {_ckpt_sz:.2f} GiB saved to {output_path}")

        if self.model_type == ModelType.PHI3_MINI:
            LOGGER.warning("Phi-3 Mini adapter to PEFT conversion unsupported. Saved in torchtune format")
        elif self.model_type == ModelType.LLAMA3_VISION:
            LOGGER.warning("Llama3.2 Vision adapter to PEFT conversion unsupported. Saved in torchtune format")
        else:
            state_dict[training.ADAPTER_KEY] = convert_weights.tune_to_peft_adapter_weights(
                state_dict[training.ADAPTER_KEY],
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )
            output_path = output_dir / ADAPTER_MODEL_FNAME
            output_dir.mkdir(parents=True, exist_ok=True)
            if not self.safe_serialization:
                output_path = output_path.with_suffix(".bin")
                torch.save(state_dict[training.ADAPTER_KEY], output_path)
            else:
                output_path = output_path.with_suffix(".safetensors")
                save_file(state_dict[training.ADAPTER_KEY], output_path, metadata={"format": "pt"})
            _ckpt_sz = os.path.getsize(output_path) / 1024**3
            LOGGER.info(f"Adapter checkpoint of size {_ckpt_sz:.2f} GiB saved to {output_path}")

    def save_adapter_config(self, state_dict: dict[str, Any], output_dir: Path) -> None:
        if self.model_type == ModelType.PHI3_MINI:
            LOGGER.warning("PEFT integration for Phi-3 Mini is not supported. Skipping adapter config save")
        elif self.model_type == ModelType.LLAMA3_VISION:
            LOGGER.warning("PEFT integration for Llama3.2 Vision is not supported. Skipping adapter config save")
        else:
            state_dict[training.ADAPTER_CONFIG] = convert_weights.tune_to_peft_adapter_config(
                adapter_config=state_dict[training.ADAPTER_CONFIG],
                base_model_name_or_path=self.repo_id,
            )
            output_path = (output_dir / ADAPTER_CONFIG_FNAME).with_suffix(".json")
            with open(output_path, "w") as f:
                json.dump(state_dict[training.ADAPTER_CONFIG], f)
            LOGGER.info(f"Adapter config saved to {output_path}")

    def save_recipe_state(self, state_dict: dict[str, Any]) -> None:
        output_path = self.output_dir / "recipe_state.pt"  # NOTE dropped added subdir resp. Meta code
        exclude_keys = (training.MODEL_KEY, training.ADAPTER_KEY, training.ADAPTER_CONFIG)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save({k: v for k, v in state_dict.items() if k not in exclude_keys}, output_path)
        LOGGER.info(f"Recipe checkpoint ({os.path.getsize(output_path) / 1024**3:.2f} GiB) saved to {output_path}")

    def _save_checkpoint(
        self,
        state_dict: dict[str, Any],
        output_dir: Path,
        save_training_state: bool,
        adapter_only: bool,
        ignore_suffixes: list[str],
    ) -> None:
        # convert the state_dict back to hf format; do this in place
        if adapter_only:
            if training.ADAPTER_KEY not in state_dict:
                raise ValueError("Adapter checkpoint not in state_dict. Ensure the state_dict contains adapter weights")
            LOGGER.info("Note: Set adapter_only=True so only adapter weights will be saved.")
        else:
            self.save_full_model(state_dict, output_dir)

        # NOTE not used currently; Save the adapter weights if present (even when adapter_only is False)
        if training.ADAPTER_KEY in state_dict:
            self.save_adapter_weights(state_dict, output_dir)

        # NOTE not used currently
        if training.ADAPTER_CONFIG in state_dict:
            self.save_adapter_config(state_dict, output_dir)

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
        epoch: int,
        global_step: int,
        seed: int,
        save_training_state: bool = True,
        adapter_only: bool = False,
        optimizer_in_bwd: bool = False,  # TODO not implemented
        optim_ckpt_wrapper=None,  # TODO typing if/when implemented; not implemented
        output_dir: Path | None = None,
        ignore_suffixes: list[str] = SUFFIXES_TO_NOT_COPY,
    ) -> tuple[dict[str, Any], Path]:
        ckpt_dict: dict = {
            ssi.constants.MODEL_KEY: model_state_dict,
            ssi.constants.EPOCHS_KEY: epoch,
            ssi.constants.GLOBAL_STEP_KEY: global_step,
            ssi.constants.SEED_KEY: seed,
        }
        if optimizer_state_dict is not None:
            if optimizer_in_bwd:
                ckpt_dict[training.OPT_KEY] = optim_ckpt_wrapper.state_dict()  # type: ignore # TODO
            else:
                ckpt_dict[training.OPT_KEY] = optimizer_state_dict
        if output_dir is None:
            output_dir = self.output_dir / f"epoch_{epoch}" / f"global_step_{global_step}"
        self._save_checkpoint(
            ckpt_dict,
            output_dir=output_dir,
            save_training_state=save_training_state,
            adapter_only=adapter_only,
            ignore_suffixes=ignore_suffixes,
        )
        return ckpt_dict, output_dir


def resolve_checkpointer_output_dir(cfg: DictConfig, wandb_logger: WandBLogger) -> Path:
    if wandb_logger._wandb.run is None:
        raise RuntimeError("wandb run not initialized")
    run_name = wandb_logger._wandb.run.name
    run_id = wandb_logger._wandb.run.id
    return Path(cfg.output_dir, f"{run_name}-id_{run_id}", "checkpoints")
