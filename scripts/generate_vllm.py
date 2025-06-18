#!/usr/bin/env python

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchtune import training
from torchtune.modules.loss import CEWithChunkedOutputLoss
from torchtune.training import get_dtype, scale_grads
from torchtune.training.lr_schedulers import get_lr
from torchtune.training.precision import PRECISION_STR_TO_DTYPE
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm
from vllm import CompletionOutput, LLM, RequestOutput, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sequence import RequestMetrics

from ssi._version import __version__
from ssi.constants import SEED
from ssi.data import setup_sft_data
from ssi.data.sft import SFTDataset
from ssi.llama_configs import configllama3_2_1b
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.train import count_token_types, get_token_type_ranges, validate_train_cfg


LOGGER = logging.getLogger(__name__)


def resolve_gen_output_dir(cfg) -> str:
    if not Path(cfg.model).is_relative_to(cfg.experiments_root_dir):
        raise ValueError(
            f"Model dir {cfg.model} must be relative to experiment directory {cfg.experiments_root_dir}. "
            "Consider setting output dir manually in the config."
        )
    model_relpath = Path(cfg.model).relative_to(cfg.experiments_root_dir)
    _day, _time = time.strftime("%Y-%m-%d"), time.strftime("%H-%M-%S")
    return str(Path(cfg.generations_root_dir, model_relpath, _day, _time).resolve())


def validate_generate_vllm_config(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Missing keys in config: {missing_keys}")

    if cfg.sampling_params.n != 1:  # NOTE eval regime under greedy decoding and top k=1 -> relax later if needed
        raise NotImplementedError("Sampling multiple sequences per prompt (sampling_params.n > 1) is not supported.")


MODEL = (
    "/mnt/scratch-artemis/anilkeshwani/experiments/"
    "Llama-3.2-1B-5000-dsus-cpt/absurd-sound-475-id_cqjfkkwf/"
    "checkpoints/epoch_0/global_step_2000"
)


def generate(cfg: DictConfig) -> None:
    # TODO timestamp / this + add a YAML or JSON with the config so we know the generation/sampling parameters
    # TODO change to output just the text, stop reason and maybe prompt?
    # TODO add MLS ID
    if cfg.get("output_dir") is None:
        cfg.output_dir = resolve_gen_output_dir(cfg)  # type: ignore
    output_dir = Path(cfg.output_dir)
    # TODO contains additional checks on optimizer_in_bwd, enable_activation_checkpointing, enable_activation_offloading
    validate_train_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)  # TODO does this actually implement reproducibility?
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
    # vLLM: Sampling Settings & Model Instantiation
    if cfg.sampling_params.stop_token_ids is None:
        cfg.sampling_params.stop_token_ids = [tokenizer.eom_id, tokenizer.eot_id, tokenizer.eos_id]
    sampling_params = SamplingParams(**cfg.sampling_params)
    llm = LLM(model=cfg.model, skip_tokenizer_init=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Write generation parameters to a YAML file and log to console
    with open(output_dir / "generation_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    # Generate
    with open(output_dir / cfg.output_filename, "w") as f:
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


@hydra.main(config_path="../conf", config_name="generate_vllm", version_base=None)
def main(cfg):
    generate(cfg)


if __name__ == "__main__":
    main()
