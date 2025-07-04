#!/usr/bin/env python

import shlex
import subprocess
from pathlib import Path

import wandb
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ssi.utils import _parse_model_path, extract_wandb_run_cfg, parse_hf_repo_id


# Constants
TEST_CONFIG_GROUPS_SUBDIR: str = "data/sft"  # NOTE extracted from class to ease modification if conf/ structure changes
PROJECT_ROOT = Path(__file__).parents[1]
GENERATE_SCRIPT = (PROJECT_ROOT / "scripts" / "generate.py").resolve(strict=True)
CONDA_ENV = "ssi-latest"


class GenerateAutomation:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg  # generation hyperparameters excl. model-specific parameters (speech encoder, deduplicate, etc.)
        self.wandb_api = wandb.Api()
        self.global_hydra = GlobalHydra.instance()

        if not self.global_hydra.is_initialized():
            raise RuntimeError("Hydra not initialized.")

        slurm_call = "srun --partition a6000 --time=01:00:00 --gres=gpu:1 --qos=gpu-short"
        conda_run_call = f"conda run --live-stream -n {CONDA_ENV}" ""
        script_call = f"python {GENERATE_SCRIPT!s}"
        missing_args_template = "data={test_dataset} speech.n_dsus={n_dsus} speech.deduplicate={deduplicate}"

        self.call_template = " ".join((slurm_call, conda_run_call, script_call, missing_args_template))

    @property
    def test_cfg_groups(self) -> list[str]:
        return self.global_hydra.config_loader().get_group_options(TEST_CONFIG_GROUPS_SUBDIR)

    def __call__(self, model: str) -> None:
        # Obtain model metadata from the model path
        model_metadata = _parse_model_path(Path(model), Path(self.cfg.experiments_root_dir))
        # Initilise a wandb.apis.public.runs.Run object from model metadata NOTE not a wandb.sdk.wandb_run.Run object
        wandb_run_id = model_metadata["wandb_run_id"]
        wandb_run = self.wandb_api.run(f"{self.cfg.wandb.entity}/{self.cfg.wandb.project}/{wandb_run_id}")
        wandb_run_cfg = extract_wandb_run_cfg(wandb_run)
        # Check if the model metadata matches the W&B run config
        model_base_name_clash = model_metadata["model_base_name"] != wandb_run_cfg.base_model_name
        training_type_clash = model_metadata["training_type"] != wandb_run_cfg.config_name
        if any((model_base_name_clash, training_type_clash)):
            raise AssertionError("Model metadata does not match W&B run config.")
        # Obtain training data metadata; NOTE assumes internally-consistent config structure to obtain HF repo ID
        train_data_metadata = parse_hf_repo_id(wandb_run_cfg.data.train.dataset.source)
        # Filter test datasets - based on speech encoder and representation layer (DSUs) used for model training
        test_datasets_suffix = "-".join((train_data_metadata[k] for k in ("speech_encoder", "encoder_layer")))
        test_datasets = tuple(c for c in self.test_cfg_groups if c.endswith(test_datasets_suffix))

        # Enqueue Slurm jobs for each test dataset specifying the test dataset via the CLI
        for test_dataset in test_datasets:
            cli_call = self.call_template.format(
                test_dataset=test_dataset,
                n_dsus=wandb_run_cfg.speech.n_dsus,
                deduplicate=wandb_run_cfg.speech.deduplicate,
            )
