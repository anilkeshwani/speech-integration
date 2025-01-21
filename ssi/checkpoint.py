import gc
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union

import torch
from safetensors.torch import save_file
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
    RECIPE_STATE_DIRNAME,
    REPO_ID_FNAME,
    safe_torch_load,
    SAFETENSOR_INDEX_FNAME,
    SHARD_FNAME,
    SUFFIXES_TO_NOT_COPY,
    TORCH_INDEX_FNAME,
)
from torchtune.utils._logging import get_logger, log_rank_zero


logger = get_logger("DEBUG")


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
        checkpoint_dir: Path,
        checkpoint_files: list[str] | dict[str, str],
        config_json: Path,
        output_dir: Path,
        recipe_checkpoint: Path | None = None,
        adapter_checkpoint: Path | None = None,
        model_type: str = "llama3_2",
        safe_serialization: bool = True,
    ) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._safe_serialization = safe_serialization
        self._model_type: ModelType = ModelType(model_type)
        self._output_dir = output_dir
        self._recipe_checkpoint = recipe_checkpoint
        self._adapter_checkpoint = adapter_checkpoint

        check_outdir_not_in_ckptdir(ckpt_dir=self._checkpoint_dir, out_dir=self._output_dir)

        if self._recipe_checkpoint is not None and not self._recipe_checkpoint.is_file():
            raise FileNotFoundError(f"Recipe checkpoint file {recipe_checkpoint} not found.")

        self._output_dir.mkdir(parents=True, exist_ok=True)  # TODO

        # weight_map: state_dict key -> checkpoint mapping to partition state dict into output checkpoint files
        self._weight_map: dict[str, str] | None = None  # NOTE initialised to None; updated during checkpoint loading

        self._config = json.loads(config_json.read_text())  # contains model params needed for state dict conversion

        # repo_id necessary to save adapter config for HF compatibility. JSON produced and saved at download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = Path.joinpath(self._checkpoint_dir, REPO_ID_FNAME).with_suffix(".json")
        self.repo_id = None
        if repo_id_path.exists():
            with open(repo_id_path, "r") as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

        self._checkpoint_paths = get_model_checkpoint_paths(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self._checkpoint_dir,
        )

        logger.info(f"Resuming from checkpoint(s): {[str(path) for path in self._checkpoint_paths]}")
        logger.info(
            f"Resuming optimizer and recipe state from: {self._recipe_checkpoint}"
            if self._recipe_checkpoint
            else "Initializing optimizer and recipe state from cold."
        )
        if self._adapter_checkpoint:
            logger.info(f"Resuming adapter from checkpoint: {self._adapter_checkpoint}")

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Load HF checkpoint from file.

        The keys and weights from across all checkpoint files are merged into a single state_dict.
        We preserve the "state_dict key" <-> "checkpoint file" mapping in weight_map so we can
        write the state dict correctly in ``save_checkpoint``.

        Before returning, the model state dict is converted to a torchtune-compatible format using
        the appropriate convert_weights function (depending on ``self._model_type``).

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

        match self._model_type:
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
                raise ValueError(f"Unsupported model type: {self._model_type}")

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._recipe_checkpoint is not None:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)

        return converted_state_dict

    def phi3_hf_to_tune(self, merged_state_dict, converted_state_dict):
        log_rank_zero(
            logger=logger,
            msg="Converting Phi-3 Mini weights from HF format."
            "Note that conversion of adapter weights into PEFT format is not supported.",
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

    def save_checkpoint(
        self,
        state_dict: dict[str, Any],
        epoch: int,
        save_training_state: bool,
        adapter_only: bool = False,
    ) -> None:
        """
        Save HF checkpoint to file. If ``save_training_state`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        The state_dict is first converted back to the HF format and then partitioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            save_training_state (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        if self._weight_map is None:
            raise ValueError("Weight map is not initialized. Please load a checkpoint before saving.")

        # convert the state_dict back to hf format; do this inplace
        if not adapter_only:
            if self._model_type == ModelType.PHI3_MINI:
                self.phi3_tune_to_hf(state_dict)
            elif self._model_type == ModelType.REWARD:
                self.reward_tune_to_hf(state_dict)
            elif self._model_type == ModelType.QWEN2:
                self.qwen2_tune_to_hf(state_dict)
            elif self._model_type == ModelType.LLAMA3_VISION:
                self.llama3_vision_tune_to_hf(state_dict)
            elif self._model_type == ModelType.GEMMA2:
                self.gemma2_tune_to_hf(state_dict)
            elif self._model_type == ModelType.CLIP_TEXT:
                raise NotImplementedError("Clip text conversion is not supported yet")  # strangely
            elif self._model_type in {ModelType.LLAMA2, ModelType.LLAMA3, ModelType.LLAMA3_2}:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
            else:
                raise ValueError(f"Unsupported model type: {self._model_type}")

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
            # e.g. model-00001-of-00004.safetensors, model-00002-of-00004.safetensors, etc
            num_shards = len(split_state_dicts)
            map_original_name_to_new_name = {}
            for cpt_idx, model_state_dict in split_state_dicts.items():
                # TODO: We should probably use the original shard name and just add a prefix
                # however, having the SHARD_FNAME standardizes our checkpoints
                shard_name = SHARD_FNAME.format(cpt_idx=f"{cpt_idx}".zfill(5), num_shards=f"{num_shards}".zfill(5))
                map_original_name_to_new_name[cpt_idx] = shard_name
                output_path = Path.joinpath(self._output_dir, f"epoch_{epoch}", shard_name)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._safe_serialization:
                    output_path = output_path.with_suffix(".bin")
                    torch.save(model_state_dict, output_path)
                else:
                    output_path = output_path.with_suffix(".safetensors")
                    save_file(model_state_dict, output_path, metadata={"format": "pt"})

                logger.info(
                    "Model checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

            # Save the appropriate index file based on serialization format
            # e.g. {metadata: {total_size: 1234}, weight_map: {"key1": "model_0001.safetensors", "key2": "model_0002.safetensors"}}
            if self._safe_serialization:
                weight_map = {
                    k: map_original_name_to_new_name[cpt_idx] + ".safetensors"
                    for k, cpt_idx in self._weight_map.items()
                }
                index_file_name = SAFETENSOR_INDEX_FNAME
            else:
                weight_map = {
                    k: map_original_name_to_new_name[cpt_idx] + ".bin" for k, cpt_idx in self._weight_map.items()
                }
                index_file_name = TORCH_INDEX_FNAME

            index_path = Path.joinpath(self._output_dir, f"epoch_{epoch}", index_file_name)

            index_data = {
                "metadata": {"total_size": total_size},
                "weight_map": weight_map,
            }
            with open(index_path, "w") as f:
                json.dump(index_data, f, indent=2)

        if training.ADAPTER_KEY in state_dict:

            # TODO: saving it "as is" is a requirement because, if we only save with
            # convert_weights.tune_to_peft_adapter_weights, we do NOT have a fn
            # convert_weights.peft_to_tune. The .pt format is not needed, but
            # it is an easy way to distinguish the adapters. Ideally we should save only one.
            output_path = Path.joinpath(self._output_dir, f"epoch_{epoch}", ADAPTER_MODEL_FNAME).with_suffix(".pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict[training.ADAPTER_KEY], output_path)
            logger.info(
                "Adapter checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )

            if self._model_type == ModelType.PHI3_MINI:
                logger.warning(
                    "Saving Phi-3 Mini adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning(
                    "Saving Llama3.2 Vision adapter weights to PEFT format is not supported, saving to torchtune format instead"
                )
            else:
                state_dict[training.ADAPTER_KEY] = convert_weights.tune_to_peft_adapter_weights(
                    state_dict[training.ADAPTER_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
                output_path = Path.joinpath(self._output_dir, f"epoch_{epoch}", ADAPTER_MODEL_FNAME)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if not self._safe_serialization:
                    output_path = output_path.with_suffix(".bin")
                    torch.save(state_dict[training.ADAPTER_KEY], output_path)
                else:
                    output_path = output_path.with_suffix(".safetensors")
                    save_file(
                        state_dict[training.ADAPTER_KEY],
                        output_path,
                        metadata={"format": "pt"},
                    )
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )
        elif adapter_only:
            raise ValueError(
                "Adapter checkpoint not found in state_dict. Please ensure that the state_dict contains adapter weights."
            )

        if training.ADAPTER_CONFIG in state_dict:
            if self._model_type == ModelType.PHI3_MINI:
                logger.warning("PEFT integration for Phi-3 Mini is not supported, skipping adapter config save")
            elif self._model_type == ModelType.LLAMA3_VISION:
                logger.warning("PEFT integration for Llama3.2 Vision is not supported, skipping adapter config save")
            else:
                state_dict[training.ADAPTER_CONFIG] = convert_weights.tune_to_peft_adapter_config(
                    adapter_config=state_dict[training.ADAPTER_CONFIG],
                    base_model_name_or_path=self.repo_id,
                )

                output_path = Path.joinpath(self._output_dir, f"epoch_{epoch}", ADAPTER_CONFIG_FNAME).with_suffix(
                    ".json"
                )
                with open(output_path, "w") as f:
                    json.dump(state_dict[training.ADAPTER_CONFIG], f)
                logger.info(
                    "Adapter checkpoint of size "
                    f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                    f"saved to {output_path}"
                )

        # Save all files in ckpt_dir, except model weights and mapping, to output_dir/epoch_{epoch}
        # So its easy to run inference with the model using this epoch's checkpoint
        copy_files(
            self._checkpoint_dir,
            Path.joinpath(self._output_dir, f"epoch_{epoch}"),
            ignore_suffixes=SUFFIXES_TO_NOT_COPY,
        )

        # If the recipe state needs to be output, first remove the model state dict
        # and if it exists, remove the adapter state dict as well
        if save_training_state:
            _ = state_dict.pop(training.MODEL_KEY, None)
            _ = state_dict.pop(training.ADAPTER_KEY, None)
            _ = state_dict.pop(training.ADAPTER_CONFIG, None)
            output_path = Path.joinpath(self._output_dir, RECIPE_STATE_DIRNAME, "recipe_state.pt")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, output_path)
            logger.info(
                "Recipe checkpoint of size "
                f"{os.path.getsize(output_path) / 1024**3:.2f} GiB "
                f"saved to {output_path}"
            )
        else:
            logger.info("Saving final epoch checkpoint.")
            if adapter_only:
                logger.info(
                    "Please note that you have set adapter_only=True, so only adapter weights will be saved."
                    "You need to merge the adapter weights into your base model for further use. "
                    f"See {self.__class__.__name__}.save_checkpoint for more details."
                )
            else:
                logger.info(
                    "The full model checkpoint, including all weights and configurations, has been saved successfully."
                    "You can now use this checkpoint for further training or inference."
                )
