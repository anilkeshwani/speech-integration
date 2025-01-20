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
    get_adapter_checkpoint_path,
    get_model_checkpoint_path,
    get_recipe_checkpoint_path,
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
        checkpoint_dir (str): Directory containing the checkpoint files
        checkpoint_files (Union[List[str], Dict[str, str]]): List of checkpoint files to load or a dictionary
            containing the keys keys ["filename_format", "max_filename"]. Since the checkpointer takes care
            of sorting by file ID, the order in this list does not matter.
        model_type (str): Model type of the model for which the checkpointer is being loaded, e.g. LLAMA3.
        output_dir (str): Directory to save the checkpoint files
        adapter_checkpoint (Optional[str]): Path to the adapter weights. If None,
            and `resume_from_checkpoint=True`, then look for adapter_model.pt in output_dir/epoch_{largest_epoch}.
            Default is None.
        recipe_checkpoint (Optional[str]): Path to the recipe state checkpoint file. If None,
            and `resume_from_checkpoint=True`, then look for recipe_state.pt in output_dir/RECIPE_STATE_DIRNAME.
            Default is None.
        resume_from_checkpoint (bool): If True, the checkpointer will load the additional checkpoint files to
            resume training from a previous run. Default is False
        safe_serialization (bool): If True, the checkpointer will save the checkpoint file using `safetensors`.
            Default is True.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        checkpoint_files: Union[List[str], Dict[str, str]],
        model_type: str,
        output_dir: str,
        adapter_checkpoint: Optional[str] = None,
        recipe_checkpoint: Optional[str] = None,
        resume_from_checkpoint: bool = False,
        safe_serialization: bool = True,
    ) -> None:

        self._resume_from_checkpoint = resume_from_checkpoint
        self._safe_serialization = safe_serialization
        self._checkpoint_dir = Path(checkpoint_dir)
        self._model_type = ModelType[model_type]
        self._output_dir = Path(output_dir)
        check_outdir_not_in_ckptdir(ckpt_dir=self._checkpoint_dir, out_dir=self._output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # weight_map contains the state_dict key -> checkpoint file mapping so we can correctly
        # parition the state dict into output checkpoint files. This is updated during checkpoint
        # load
        self._weight_map: Dict[str, str] = None

        # the config.json file contains model params needed for state dict conversion
        self._config = json.loads(Path.joinpath(self._checkpoint_dir, "config.json").read_text())

        # repo_id is necessary for when saving an adapter config, so its compatible with HF.
        # This json file is produced and saved in the download step.
        # contents are {"repo_id": "some_model/some_model_version"}
        repo_id_path = Path.joinpath(self._checkpoint_dir, REPO_ID_FNAME).with_suffix(".json")
        self.repo_id = None
        if repo_id_path.exists():
            with open(repo_id_path, "r") as json_file:
                data = json.load(json_file)
                self.repo_id = data.get("repo_id")

        #  resume from adapter_model ckpt
        self._adapter_checkpoint = get_adapter_checkpoint_path(
            output_dir=self._output_dir,
            adapter_checkpoint=adapter_checkpoint,
            resume_from_checkpoint=self._resume_from_checkpoint,
            pattern=r"^epoch_(\d+)",
        )

        # resume recipe_state ckpt
        self._recipe_checkpoint = get_recipe_checkpoint_path(
            output_dir=self._output_dir,
            recipe_checkpoint=recipe_checkpoint,
            resume_from_checkpoint=self._resume_from_checkpoint,
        )

        # get ckpt paths
        self._checkpoint_paths = get_model_checkpoint_path(
            checkpoint_files=checkpoint_files,
            checkpoint_dir=self._checkpoint_dir,
            output_dir=self._output_dir,
            resume_from_checkpoint=self._resume_from_checkpoint,
            has_adapter_checkpoint=self._adapter_checkpoint is not None,
        )

        if self._resume_from_checkpoint:
            logger.info(
                "Resuming from checkpoint using:"
                f"\n\tcheckpoint_paths: {[str(path) for path in self._checkpoint_paths]}"
                f"\n\trecipe_checkpoint: {self._recipe_checkpoint}"
                f"\n\tadapter_checkpoint: {self._adapter_checkpoint}"
            )

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
        if self._model_type == ModelType.PHI3_MINI:
            log_rank_zero(
                logger=logger,
                msg="Converting Phi-3 Mini weights from HF format."
                "Note that conversion of adapter weights into PEFT format is not supported.",
            )
            from torchtune.models.phi3._convert_weights import phi3_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = phi3_hf_to_tune(merged_state_dict)
        elif self._model_type == ModelType.REWARD:
            from torchtune.rlhf.utils import reward_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = reward_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
            )
        elif self._model_type == ModelType.QWEN2:
            from torchtune.models.qwen2._convert_weights import qwen2_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = qwen2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                tie_word_embeddings=self._config["tie_word_embeddings"],
            )
        elif self._model_type == ModelType.LLAMA3_VISION:
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
        elif self._model_type == ModelType.CLIP_TEXT:
            from torchtune.models.clip._convert_weights import clip_text_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = clip_text_hf_to_tune(
                merged_state_dict,
            )
        elif self._model_type == ModelType.GEMMA2:
            from torchtune.models.gemma2._convert_weights import gemma2_hf_to_tune

            converted_state_dict[training.MODEL_KEY] = gemma2_hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )
        else:
            converted_state_dict[training.MODEL_KEY] = convert_weights.hf_to_tune(
                merged_state_dict,
                num_heads=self._config["num_attention_heads"],
                num_kv_heads=self._config["num_key_value_heads"],
                dim=self._config["hidden_size"],
                head_dim=self._config.get("head_dim", None),
            )

        if self._adapter_checkpoint:
            adapter_state_dict = safe_torch_load(self._adapter_checkpoint)
            converted_state_dict[training.ADAPTER_KEY] = adapter_state_dict

        if self._resume_from_checkpoint:
            recipe_state = safe_torch_load(self._recipe_checkpoint, mmap=False)
            converted_state_dict.update(recipe_state)

        return converted_state_dict

    def save_checkpoint(
        self,
        state_dict: Dict[str, Any],
        epoch: int,
        intermediate_checkpoint: bool = False,
        adapter_only: bool = False,
    ) -> None:
        """
        Save HF checkpoint to file. If ``intermediate_checkpoint`` is True, an additional
        checkpoint file ``recipe_state.pt`` is created in ``_output_dir/RECIPE_STATE_DIRNAME``
        which contains the recipe state.

        The state_dict is first converted back to the HF format and then partitioned based on the
        ``_weight_map`` into separate checkpoint files.

        Args:
            state_dict (Dict[str, Any]): Checkpoint state dict to be written out to file
            epoch (int): Epoch number. Used to create the checkpoint file name
            intermediate_checkpoint (bool): If True, an additional checkpoint files for recipe state
                and (if applicable) adapter weights are created. Default is False
            adapter_only (bool): If True, only save the adapter weights. Default is False

        Raises:
            ValueError: if ``adapter_only`` is True and adapter checkpoint not found in state_dict.
        """
        # convert the state_dict back to hf format; do this inplace
        if not adapter_only:
            if self._model_type == ModelType.PHI3_MINI:
                from torchtune.models.phi3._convert_weights import phi3_tune_to_hf

                state_dict[training.MODEL_KEY] = phi3_tune_to_hf(state_dict[training.MODEL_KEY])
            elif self._model_type == ModelType.REWARD:
                from torchtune.rlhf.utils import reward_tune_to_hf

                state_dict[training.MODEL_KEY] = reward_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                )
            elif self._model_type == ModelType.QWEN2:
                from torchtune.models.qwen2._convert_weights import qwen2_tune_to_hf

                state_dict[training.MODEL_KEY] = qwen2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    tie_word_embeddings=self._config["tie_word_embeddings"],
                )
            elif self._model_type == ModelType.LLAMA3_VISION:
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
            elif self._model_type == ModelType.GEMMA2:
                from torchtune.models.gemma2._convert_weights import gemma2_tune_to_hf

                state_dict[training.MODEL_KEY] = gemma2_tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )
            else:
                state_dict[training.MODEL_KEY] = convert_weights.tune_to_hf(
                    state_dict[training.MODEL_KEY],
                    num_heads=self._config["num_attention_heads"],
                    num_kv_heads=self._config["num_key_value_heads"],
                    dim=self._config["hidden_size"],
                    head_dim=self._config.get("head_dim", None),
                )

            # split the state_dict into separate dicts, one for each output checkpoint file
            # e.g. split_state_dicts= {
            #       "0001": {"key1": tensor1, "key2": tensor2},
            #       "0002": {"key3": tensor3}
            #       }
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
        if intermediate_checkpoint:
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
