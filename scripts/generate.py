#!/usr/bin/env python

import json
import logging
from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import DataLoader
from torchtune import training
from tqdm import tqdm
from vllm import CompletionOutput, LLM, RequestOutput, SamplingParams
from vllm.inputs.data import TokensPrompt
from vllm.sequence import RequestMetrics

from ssi._version import __version__
from ssi.constants import SEED, TORCHTUNE_CONFIG_FILENAME
from ssi.data.sft import SFTDataset
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.utils import hash_cfg


LOGGER = logging.getLogger(__name__)

# Constants (module-specific)
TEST_CONFIG_GROUPS_SUBDIR: str = "data/sft"  # extracted to ease modification if conf/ directory structure changes


def _resolve_gen_output_dir(cfg) -> str:
    """Resolve the generation output directory based on the config according to internal conventions."""
    model_dir = Path(cfg.model).resolve(strict=True)
    experiments_root_dir = Path(cfg.experiments_root_dir).resolve(strict=True)
    if not model_dir.is_relative_to(experiments_root_dir):
        raise ValueError(
            "Could not resolve null generation output directory. Model {cfg.model} not in {cfg.experiments_root_dir}"
            "Specify a generation output directory in the config or check your model path."
        )
    if model_dir.parts[-3] != "checkpoints":
        example_model_dir = (
            "/mnt/scratch-artemis/anilkeshwani/experiments/"
            "Llama-3.2-1B-5000-dsus-cpt/absurd-sound-475-id_cqjfkkwf/"
            "checkpoints/epoch_0/global_step_2000"
        )
        raise ValueError(
            f"Could not resolve null generation output directory. Expect model directory of form: {example_model_dir}."
        )
    gen_output_dir_parts = list(model_dir.parts)
    gen_output_dir_parts[-3] = "generations"  # "checkpoints" -> "generations"
    gen_output_dir = str(Path(*gen_output_dir_parts).resolve(strict=False))
    LOGGER.info(f"Resolved null generation output directory to: {gen_output_dir}")
    return gen_output_dir


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


def generate(cfg: DictConfig) -> Path:
    # TODO add MLS ID i.e. test dataset additional_keys functionality
    validate_generate_config(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)  # torch, numpy, random + cuDNN deterministic if debug_mode
    if cfg.gen.get("output_dir") is None:
        cfg.gen.output_dir = _resolve_gen_output_dir(cfg)
    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    special_int2str = {v: k for k, v in special_tokens.items()}
    # Set stop token IDs (in sampling_params) from tokenizer if not specified in config
    if cfg.sampling_params.stop_token_ids is None:
        cfg.sampling_params.stop_token_ids = [tokenizer.eom_id, tokenizer.eot_id, tokenizer.eos_id]
    # Set dataset- and configuration-specific output directory - NOTE only after all config parameters are resolved
    _owner, test_dataset_name = (cfg.data.test.dataset.source).split("/")  # e.g. anilkeshwani/mls-speechtokenizer-rvq_0
    gen_output_dir = Path(cfg.gen.output_dir) / test_dataset_name
    if cfg.gen.use_cfg_hash_subdir:
        gen_output_dir = gen_output_dir / hash_cfg(cfg)
    gen_output_dir.mkdir(parents=True, exist_ok=False)  # NOTE fail early if output directory already exists
    # Write generation parameters to a YAML file and log to console
    cfg_yaml_nosort = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False)
    with open(gen_output_dir / cfg.gen.output_config_filename, "x") as f:
        f.write(cfg_yaml_nosort)
    LOGGER.info(cfg_yaml_nosort)
    # NOTE SFT dataset - Used for generation for now for flexibility - we can modify the system prompt and template
    #      dataset columns via a PromptTemplate class, which can be specified as a dictionary in the YAML
    data = DataLoader(
        SFTDataset(model_tokenizer=tokenizer, **cfg.data.test.dataset),  # NOTE generation hard-coded to use test set
        batch_size=cfg.vllm_batch_size,
        collate_fn=lambda batch: [TokensPrompt(prompt_token_ids=sample["tokens"]) for sample in batch],
        shuffle=False,
        drop_last=False,
    )
    # vLLM SamplingParams & LLM initialization; NOTE deferred to allow early fail in open if output file exists
    llm = LLM(model=cfg.model, skip_tokenizer_init=True)
    sampling_params = SamplingParams(**cfg.sampling_params)
    # Generate
    with open(gen_output_dir / cfg.gen.output_filename, "x") as f:
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
    LOGGER.info(f"Wrote outputs to {gen_output_dir!s}")
    return gen_output_dir


@hydra.main(config_path="../conf", config_name="generate", version_base=None)
def main(cfg):
    global_hydra = GlobalHydra.instance()
    if not global_hydra.is_initialized():
        raise RuntimeError("Hydra not initialized.")

    if cfg.train_config is None:
        cfg.train_config = Path(cfg.model).parents[1] / TORCHTUNE_CONFIG_FILENAME  # per internal ckpt convention

    train_cfg = OmegaConf.load(cfg.train_config)

    if cfg.speech.n_dsus is None:
        cfg.speech.n_dsus = train_cfg.speech.n_dsus
    if cfg.speech.deduplicate is None:
        cfg.speech.deduplicate = train_cfg.speech.deduplicate

    # NOTE cfg.speech options must be resolved before cfg.data due to interpolation in cfg.data fields
    if cfg.get("data") is None:
        config_sources = HydraConfig.get().runtime.config_sources  # calls HydraConfig.instance
        config_source_main = Path([s.path for s in config_sources if s.provider == "main"].pop())
        _owner, train_dataset = (train_cfg.data.train.dataset.source).split("/")  # HF repo ID
        test_config_groups = global_hydra.config_loader().get_group_options(TEST_CONFIG_GROUPS_SUBDIR)
        if train_dataset not in test_config_groups:
            raise RuntimeError("Cannot find test dataset config.")
        cfg_data_path = (config_source_main / TEST_CONFIG_GROUPS_SUBDIR / train_dataset).with_suffix(".yaml")
        with open_dict(cfg):  # temporarily remove struct flag -> allow addition of new keys
            cfg.data = OmegaConf.load(cfg_data_path)

    generate(cfg)


if __name__ == "__main__":
    main()
