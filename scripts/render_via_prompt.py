#!/usr/bin/env python

import json
import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import jinja2
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT, SPEECH_TOKENS_KEY
from sardalign.constants.megatron import MEGATRON_TEXT_KEY
from sardalign.utils import count_lines, dsu2pua
from tqdm import tqdm


PROMPT_FILENAME_INDICATOR: str = "templated"

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__file__)

# Jinja Prompt Templates
PROMPT_TEMPLATES_DIR = Path(__file__).parents[2] / "prompt_templates"  # TODO modularise if retained; shared with gen.


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("input_jsonl", type=Path, help="Path to jsonl with text alignments and HuBERT speech tokens")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=None,
        help="Path to write the templated output data. "
        f"Defaults to input path with '{PROMPT_FILENAME_INDICATOR}' and prompt template name appended to file stem.",
    )
    parser.add_argument(
        "--text-key",
        type=str,
        required=True,
        help="Text field key in input JSON lines file. "
        "Exercise discretion but typically this would be the field considered the ASR reference transcription",
    )
    parser.add_argument(
        "--prompt-template", type=str, required=True, help="Path to Jinja2 template for prompt generation"
    )
    args = parser.parse_args()

    if not args.prompt_template.endswith(".jinja"):
        args.prompt_template += ".jinja"

    if args.output_jsonl is None:
        prompt_name = args.prompt_template.removesuffix(".jinja")
        args.output_jsonl = args.input_jsonl.with_stem(
            "_".join((args.input_jsonl.stem, PROMPT_FILENAME_INDICATOR, prompt_name))
        )

    return args


def main(input_jsonl: Path, output_jsonl: Path, text_key: str, prompt_template: str):
    if output_jsonl.exists():
        raise FileExistsError(f"Output JSON lines file {output_jsonl!s} exists.")
    env_j2 = jinja2.Environment(loader=jinja2.FileSystemLoader(PROMPT_TEMPLATES_DIR))
    prompt_template_j2 = env_j2.get_template(prompt_template)

    with open(input_jsonl, mode="r") as f, open(output_jsonl, mode="x") as g:
        for i, line in enumerate(tqdm(f, desc="Templating text-speech samples", total=count_lines(input_jsonl))):
            sample = json.loads(line)
            prompt = prompt_template_j2.render(
                {
                    "text": sample[text_key],
                    "speech_tokens": "".join((dsu2pua(dsu) for dsu in sample[SPEECH_TOKENS_KEY])),
                    "MODALITY_TOKEN_SPEECH": MODALITY_TOKEN_SPEECH,
                    "MODALITY_TOKEN_TEXT": MODALITY_TOKEN_TEXT,
                }
            )
            g.write(json.dumps({MEGATRON_TEXT_KEY: prompt}) + "\n")
    LOGGER.info(f"Wrote templated data to {output_jsonl!s}")


if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))
