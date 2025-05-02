#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import jinja2
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from vllm import CompletionOutput, LLM, RequestOutput, SamplingParams
from vllm.sequence import RequestMetrics

from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT, PROMPT_TEMPLATES_DIR, SPEECH_TOKENS_KEY
from sardalign.utils import dsu2pua, read_jsonl, write_jsonl


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("model", type=str, help="Hugging Face checkpoint path or model name")
    # TODO deprecate this later if deciding on a standard set of test benchmarks / test sets
    # e.g. LS-clean, LS-other, WSJ, GigaSpeech, VoxPopuli, etc.
    parser.add_argument(
        "--test-jsonl", required=True, type=Path, help="Test JSON lines file containing text and DSU fields"
    )
    parser.add_argument("--output-jsonl", type=Path, required=True, help="Path for model outputs JSON lines file")
    parser.add_argument(
        "--text-key",
        type=str,
        required=True,
        help="Text field key in input JSON lines file. Used as the ASR reference transcription",
    )
    parser.add_argument(
        "--tokenizer-mode",
        type=str,
        default="slow",
        help="Tokenizer mode. "
        "Passing 'auto' will use the fast tokenizer if available, and 'slow' will always use the slow tokenizer.",
        choices=["auto", "slow"],
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default=None,
        help="Path to Jinja2 template for prompt generation. Default is None which uses all available templates.",
    )
    args = parser.parse_args()

    if not args.prompt_template.endswith(".jinja"):
        args.prompt_template += ".jinja"

    return args


def main(args: Namespace):
    if args.output_jsonl.exists():
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
    outputs_json_serialisable = []
    for output, generations, observability in zip(outputs, model_generations_s, observability_metrics):
        outputs_json_serialisable.append(
            {k: v for k, v in vars(output).items() if k not in ("outputs", "metrics")}
            | {"outputs": [vars(generation) for generation in generations]}
            | {"metrics": vars(observability)}
        )
    write_jsonl(args.output_jsonl, outputs_json_serialisable)
    LOGGER.info(f"Wrote outputs to {args.output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(args)

# TODO: Test both the text and text_normalized fields for VoxPopuli
# TODO: Add ASR metric per https://huggingface.co/learn/audio-course/en/chapter5/evaluation
#       (taking Normalizer from transformers to minimize dependencies)

# NOTE Request Output structure:
# RequestOutput(
#     request_id=0, # <- NOTE
#     prompt="The capital of France is", # <- NOTE
#     prompt_token_ids=[1, 450, 7483, 310, 3444, 338], # <- NOTE
#     encoder_prompt=None,
#     encoder_prompt_token_ids=None,
#     prompt_logprobs=None, # <- NOTE
#     outputs=[ # <- NOTE
#         CompletionOutput(
#             index=0,
#             text=" Paris.",
#             token_ids=(3681, 29889, 13),
#             cumulative_logprob=None,
#             logprobs=None,
#             finish_reason=stop,
#             stop_reason="\n",
#         )
#     ],
#     finished=True, # <- NOTE
#     metrics=RequestMetrics(
#         arrival_time=1728338377.7350004,
#         last_token_time=1728338377.7350004,
#         first_scheduled_time=1728338377.73668,
#         first_token_time=1728338377.754303,
#         time_in_queue=0.0016796588897705078,
#         finished_time=1728338377.765628,
#         scheduler_time=0.000719655305147171,
#         model_forward_time=None,
#         model_execute_time=None,
#     ),
#     lora_request=None,
# )
