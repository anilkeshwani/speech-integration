#!/usr/bin/env python

import json
import logging
from pathlib import Path

import hydra
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchtune import training
from tqdm import tqdm
from vllm import CompletionOutput, LLM, RequestOutput, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sequence import RequestMetrics
from wandb.apis.public.runs import Run

from ssi._version import __version__
from ssi.constants import SEED
from ssi.data.sft import SFTDataset
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.utils import hash_cfg, parse_model_path


LOGGER = logging.getLogger(__name__)


def resolve_gen_output_dir(cfg) -> str:
    # Only used for automatic resolution of null generation output directory; user can always choose to pass value
    example_model_dir = (
        "/mnt/scratch-artemis/anilkeshwani/experiments/"
        "Llama-3.2-1B-5000-dsus-cpt/absurd-sound-475-id_cqjfkkwf/"
        "checkpoints/epoch_0/global_step_2000"
    )
    model_dir = Path(cfg.model).resolve(strict=True)
    if not model_dir.is_relative_to(cfg.experiments_root_dir):
        raise ValueError("Could not resolve null output_dir. Model {cfg.model} not in {cfg.experiments_root_dir}")
    if model_dir.parts[-3] != "checkpoints":
        raise ValueError(f"Could not resolve null output_dir. Expect model directory of form: {example_model_dir}.")
    model_relpath = model_dir.relative_to(cfg.experiments_root_dir)
    ...


def validate_generate_config(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Missing keys in config: {missing_keys}")

    # NOTE Constraint due to downstream WER eval regime under greedy decoding and top k=1 - relax later if needed
    if cfg.sampling_params.n != 1:
        raise NotImplementedError("Sampling multiple sequences per prompt (sampling_params.n > 1) is not supported.")

    # NOTE Model directory constrained to be in the experiments root directory to allow standardised parsing of
    #      model directory to get: W&B run ID, model name, training type, epoch, and global step
    if not Path(cfg.model).is_relative_to(cfg.experiments_root_dir):
        raise NotImplementedError(
            "Script only supports models in the experiments root directory. "
            f"Got model: {cfg.model}. Experiments root directory set to: {cfg.experiments_root_dir}"
        )


def sanitize_wandb_run_json_config(run_json_config: dict) -> dict:
    """W&B format for top-level keys: {"key": {"desc": <str>, "value": <any>}}; "_wandb" field holds run metadata"""
    return {k: v["value"] for k, v in run_json_config.items() if k != "_wandb"}


def extract_run_cfg(run: Run) -> DictConfig:
    return OmegaConf.create(sanitize_wandb_run_json_config(json.loads(run.json_config)))


def parse_train_repo_id(cfg: DictConfig) -> dict[str, str]:
    train_repo_id: str = cfg.data.train.dataset.source  # e.g. anilkeshwani/mls-speechtokenizer-rvq_0
    owner, train_dataset_name = train_repo_id.split("/")
    dataset, speech_encoder, encoder_layer = train_dataset_name.split("-")
    return {"dataset": dataset, "speech_encoder": speech_encoder, "encoder_layer": encoder_layer, "repo_owner": owner}


def generate(cfg: DictConfig) -> None:
    validate_generate_config(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)  # torch, numpy, random + cuDNN deterministic if debug_mode
    if cfg.get("output_dir") is None:
        cfg.output_dir = resolve_gen_output_dir(cfg)  # type: ignore
    # TODO add MLS ID i.e. test dataset additional_keys functionality
    if cfg.sampling_params.stop_token_ids is None:
        cfg.sampling_params.stop_token_ids = [tokenizer.eom_id, tokenizer.eot_id, tokenizer.eos_id]
    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    special_int2str = {v: k for k, v in special_tokens.items()}
    # NOTE SFT dataset - Used for generation for now for flexibility - we can modify the system prompt and template
    #      dataset columns via a PromptTemplate class, which can be specified as a dictionary in the YAML
    data = DataLoader(
        SFTDataset(model_tokenizer=tokenizer, **cfg.data.test.dataset),
        batch_size=cfg.vllm_batch_size,
        collate_fn=lambda batch: [TokensPrompt(prompt_token_ids=sample["tokens"]) for sample in batch],
        shuffle=False,
        drop_last=False,
    )
    # vLLM SamplingParams & LLM initialization
    llm = LLM(model=cfg.model, skip_tokenizer_init=True)
    sampling_params = SamplingParams(**cfg.sampling_params)
    # Set up configuration-specific output directory
    cfg_hash = hash_cfg(cfg)
    output_dir = Path(cfg.output_dir) / cfg_hash
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write generation parameters to a YAML file and log to console
    cfg_yaml_unsorted = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False)
    with open(output_dir / cfg.generation_config_filename, "w") as f:
        f.write(cfg_yaml_unsorted)
    LOGGER.info(cfg_yaml_unsorted)
    # Obtain model metadata from the model path and W&B run config
    model_metadata = parse_model_path(Path(cfg.model), Path(cfg.experiments_root_dir))
    wandb_api = wandb.Api()
    wandb_run = wandb_api.run(f"{cfg.wandb.entity}/{cfg.wandb.project}/{model_metadata['wandb_run_id']}")
    wandb_run_cfg = extract_run_cfg(wandb_run)
    # model_base_name_clash = model_metadata["model_base_name"] != wandb_run_cfg.base_model_name
    # training_type_clash = model_metadata["training_type"] != wandb_run_cfg.config_name
    # if any((model_base_name_clash, training_type_clash)):
    #     raise AssertionError("Model metadata does not match W&B run config.")
    # Obtain training data metadata -> filter test sets used for generation based on speech encoder and encoder layer
    train_data_metadata = parse_train_repo_id(wandb_run_cfg)
    # Use Hydra Compose API to
    # Generate
    with open(output_dir / cfg.output_filename, "w") as f:
        # TODO optionally refactor this to delegate batching to vLLM per their docs (generate all prompts upfront)
        for i, prompt_token_ids in enumerate(tqdm(data)):
            outputs: list[RequestOutput] = llm.generate(prompt_token_ids, sampling_params, use_tqdm=False)
            model_generations_s: list[list[CompletionOutput]] = [output.outputs for output in outputs]
            observability_metrics: list[RequestMetrics | None] = [output.metrics for output in outputs]
            outputs_json_serialisable = []
            for output, generations, observability in zip(outputs, model_generations_s, observability_metrics):
                output_d = {k: v for k, v in vars(output).items() if k not in ("outputs", "metrics")}
                # Manually add decoded prompt text
                output_d["prompt"] = tokenizer.decode(output_d["prompt_token_ids"], **cfg.tokenizer_decoding)
                generations_d = {"outputs": [vars(generation) for generation in generations]}
                # Manually add decoded generated text
                for generation in generations_d["outputs"]:
                    generation["text"] = tokenizer.decode(generation["token_ids"], **cfg.tokenizer_decoding)
                    stop_reason = generation["stop_reason"]
                    generation["stop_reason_text"] = special_int2str[stop_reason] if stop_reason is not None else None
                metrics_d = {"metrics": vars(observability)} if cfg.observability else {}
                outputs_json_serialisable.append(output_d | generations_d | metrics_d)
            # Write outputs to JSONL file
            for output in outputs_json_serialisable:
                f.write(json.dumps(output, ensure_ascii=False) + "\n")
            # del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
            # torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM
    LOGGER.info(f"Wrote outputs to {cfg.output_dir!s}")


@hydra.main(config_path="../conf", config_name="generate", version_base=None)
def main(cfg):
    generate(cfg)


if __name__ == "__main__":
    main()
