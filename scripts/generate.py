#!/usr/bin/env python

import logging
import time
from collections import defaultdict

import torch
from omegaconf import DictConfig, OmegaConf
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
from ssi.llama_configs import configllama3_2_1b
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.train import count_token_types, get_token_type_ranges, validate_train_cfg


LOGGER = logging.getLogger(__name__)

LOSS_FN_IGNORE_INDEX_DEFAULT: int = CEWithChunkedOutputLoss().ignore_index  # -100 # TODO move to constants?


def generate(cfg: DictConfig) -> None:
    # TODO contains additional checks on optimizer_in_bwd, enable_activation_checkpointing, enable_activation_offloading
    validate_train_cfg(cfg)
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)  # TODO
    DEVICE: torch.device = get_device(cfg.device)
    configllama3_2_1b.update_from_speech_cfg(cfg.speech)  # in-place
    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    # special_id2str = {v: k for k, v in special_tokens.items()}
    token_type_ranges = get_token_type_ranges(llama_config=configllama3_2_1b)
    # NOTE for now, we generate via an SFT dataset - this is flexible because we can modify the system prompt and also
    #      template the dataset columns via a PromptTemplate class, which can be specified as a dictionary in the YAML
    data_test, sampler_test = setup_sft_data(
        cfg_dataset=cfg.data.dev,  # TODO dev -> test
        model_tokenizer=tokenizer,
    )
    # vLLM: Sampling Settings & Model Instantiation
    sampling_params = SamplingParams(
        n=1,  # 1; TODO
        temperature=0.0,  # 0.8; TODO
        top_p=1,  # default is 1; nucleus sampling probability set to 0.95 in vLLM docs; NOTE sum_k(prob) >= p; TODO
        max_tokens=128,  # TODO
        # stop=[special_id2str[eid] for eid in [tokenizer.eom_id, tokenizer.eot_id, tokenizer.eos_id]],  # TODO
        stop_token_ids=[tokenizer.eom_id, tokenizer.eot_id, tokenizer.eos_id],  # TODO can replace the above? TODO
        # NOTE the following are added with default values TODO modify if potentially beneficial
        presence_penalty=0,
        frequency_penalty=0,
        repetition_penalty=1,
    )
    llm = LLM(
        model=cfg.model,
        # tokenizer_mode=args.tokenizer_mode,
        skip_tokenizer_init=True,
    )

    token_type_counts_total = defaultdict(int)
    tokens_test_total: int = 0
    LOGGER.info(OmegaConf.to_yaml(cfg, resolve=True, sort_keys=False))
    t0 = time.perf_counter()
    for i, batch in tqdm(enumerate(data_test), total=len(data_test)):
        for tt, ttcnt in count_token_types(batch["tokens"], token_type_ranges, tokenizer.pad_id).items():
            token_type_counts_total[tt] += ttcnt
        LOGGER.info(" | ".join([f"Tokens ({tt}): {ttcnt}" for tt, ttcnt in token_type_counts_total.items()]))
        LOGGER.info(f"max_seq_len_step: {batch['tokens'].size(1)}")
        num_tokens_iter = (batch["labels"] != LOSS_FN_IGNORE_INDEX_DEFAULT).sum()
        tokens_test_total += num_tokens_iter  # total number of tokens trained on so far

        batch_to_device(batch, DEVICE)  # in-place

        # --- vLLM generation ---

        prompt_token_ids = TokensPrompt(prompt_token_ids=batch["tokens"])
        outputs: list[RequestOutput] = llm.generate(prompt_token_ids, sampling_params=sampling_params, use_tqdm=True)
        # NOTE the outputs attr of a RequestOutput object is a **list** of CompletionOutput objects
        model_generations_s: list[list[CompletionOutput]] = [output.outputs for output in outputs]  # outputs list attr
        observability_metrics: list[RequestMetrics | None] = [output.metrics for output in outputs]  # metrics attr
        outputs_json_serialisable = []
        for output, generations, observability in zip(outputs, model_generations_s, observability_metrics):
            outputs_json_serialisable.append(
                {k: v for k, v in vars(output).items() if k not in ("outputs", "metrics")}
                | {"outputs": [vars(generation) for generation in generations]}
                | {"metrics": vars(observability)}
            )

        breakpoint()

        # write_jsonl(args.output_jsonl, outputs_json_serialisable)  # TODO
        # LOGGER.info(f"Wrote outputs to {args.output_jsonl!s}")

        # --- End vLLM generation ---

        LOGGER.info(f"Time elapsed (iteration): {time.perf_counter() - t0:.2f} seconds")
        t0 = time.perf_counter()

        # TODO surely the explicit deletion of the batch is not necessary in inference mode??
        # del batch  # Explicitly delete the batch to free memory; attempt to debug OOM
        # torch.cuda.empty_cache()  # Release all unoccupied cached memory; attempt to debug OOM
    LOGGER.info(f"Total number of tokens: {tokens_test_total}")


# --- Entrypoint ---

import hydra


@hydra.main(config_path="../conf", config_name="generate", version_base=None)
def main(cfg):
    generate(cfg)


if __name__ == "__main__":
    main()
