#!/usr/bin/env python

import logging
import os
import sys

import hydra
import torch
from gguf import Path
from omegaconf import DictConfig
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torchtune.training.precision import PRECISION_STR_TO_DTYPE

from ssi.constants import SUPPORTED_DTYPES


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)


def validate_gen_cfg(cfg: DictConfig) -> None:
    if PRECISION_STR_TO_DTYPE.get(cfg.dtype) not in SUPPORTED_DTYPES:
        raise ValueError(f"Unsupported dtype: {cfg.dtype}. Supported dtypes: {SUPPORTED_DTYPES}")

    if PRECISION_STR_TO_DTYPE.get(cfg.dtype) != torch.float32:
        raise NotImplementedError("Only float32 is supported for generation.")  # TODO can we support bf16?

    if missing_keys := OmegaConf.missing_keys(cfg):
        raise ValueError(f"Missing keys in config: {missing_keys}")


def resolve_gen_cfg(cfg: DictConfig) -> DictConfig:
    cfg.output_jsonl = Path(cfg.output_jsonl)  # TODO can use a Hydra structured config (but they're not great)
    return cfg


@hydra.main(config_path="../conf", config_name="generate.yaml", version_base=None)
def main(cfg: DictConfig):
    validate_gen_cfg(cfg)
    cfg = resolve_gen_cfg(cfg)
    if cfg.output_jsonl.exists():
        raise FileExistsError(f"Output JSON lines file {args.output_jsonl!s} exists.")
    test_data = read_jsonl(args.test_jsonl)
    jinja_env = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT_TEMPLATES_DIR))
    prompt_template = jinja_env.get_template(args.prompt_template)
    prompts = [
        prompt_template.render(
            {
                "MODALITY_TOKEN_SPEECH": MODALITY_TOKEN_SPEECH,
                "MODALITY_TOKEN_TEXT": MODALITY_TOKEN_TEXT,
                "speech_tokens": "".join((dsu2pua(dsu) for dsu in s[SPEECH_TOKENS_KEY])),
            }
        )
        for s in test_data
    ]
    sampling_params = SamplingParams(
        n=10,  # 1
        temperature=1.0,  # 0.8
        top_p=1,  # default is 1; nucleus sampling probability set to 0.95 in vLLM docs; NOTE sum_k(prob) >= p
        max_tokens=128,
        stop=[],  # TODO DEBUG
        # stop=[r"<\s>"],  # TODO DEBUG
        # stop=[r"<\s>", "\n"],
    )
    llm = LLM(model=args.model, tokenizer_mode=args.tokenizer_mode)
    outputs: list[RequestOutput] = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)
    # NOTE the outputs attr of a RequestOutput object is a **list** of CompletionOutput objects
    model_generations_s: list[list[CompletionOutput]] = [output.outputs for output in outputs]  # "outputs" list attr
    observability_metrics: list[RequestMetrics | None] = [output.metrics for output in outputs]  # "metrics" attr
    outputs_json_serialisable = [
        {k: v for k, v in vars(output).items() if k not in ("outputs", "metrics")}
        | {"outputs": [vars(generation) for generation in generations]}
        | {"metrics": vars(observability)}
        for output, generations, observability in zip(outputs, model_generations_s, observability_metrics)
    ]
    write_jsonl(args.output_jsonl, outputs_json_serialisable)
    LOGGER.info(f"Wrote outputs to {args.output_jsonl!s}")


if __name__ == "__main__":
    main()
