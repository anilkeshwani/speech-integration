#!/usr/bin/env python

import logging
import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

import torchtune.training
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.utils import seed_everything

# NOTE we use the torchtune native FullModelHFCheckpointer (not the custom checkpointer in ssi) for simplicity
from torchtune.training.checkpointing import FullModelHFCheckpointer  # NOTE also exported by torchtune.training

from ssi.extend_llama3_2 import extend_config, extend_model, extend_tiktoken, simple_setup_model
from ssi.extend_llama3_2.constants import (
    BASE_VOCAB_SIZE,
    LLAMA_3_2_1B_BASE_DIR,
    LLAMA_3_2_CONFIG_RELPATH,
    LLAMA_3_2_TOKENIZER_RELPATH,
    LLAMA_BOS_TOKEN,
    LLAMA_EOS_TOKEN,
    SEED,
    TORCHTUNE_EXTENDED_MODELS_DIR,
)
from ssi.llama_configs import configllama3_2_1b
from ssi.tokenizer import setup_llama3_tokenizer


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Extend a tokenizer.model and model.safetensors file for DSUs")
    parser.add_argument("--n_new_dsus", type=int, required=True, help="Number of DSUs to add as tokens")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=LLAMA_3_2_1B_BASE_DIR,
        help="Input Llama 3.2 directory from tune download." f" Default: {LLAMA_3_2_1B_BASE_DIR}",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory to save the extended files")
    parser.add_argument(
        "--no-modality-tokens",
        action="store_false",
        dest="use_modality_tokens",
        help="Do no prepend special modality tokens to spans of text/speech tokens",
    )
    parser.add_argument("--base_vocab_size", type=int, default=BASE_VOCAB_SIZE, help="Tokenizer base vocabulary size")
    args = parser.parse_args()
    if args.output_dir is None:
        mdl_tks_flg = "" if args.use_modality_tokens else "no_modality_tokens-"
        dirname = f"{args.input_dir.name}-{args.n_new_dsus}-{mdl_tks_flg}dsus"
        args.output_dir = TORCHTUNE_EXTENDED_MODELS_DIR / dirname
    return args


def main(args: Namespace) -> None:
    seed_everything(SEED)  # reproducibility
    extend_tiktoken(
        args.n_new_dsus,
        args.use_modality_tokens,
        args.input_dir / LLAMA_3_2_TOKENIZER_RELPATH,
        args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH,
    )
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=str(args.input_dir),
        checkpoint_files=["model.safetensors"],
        model_type="LLAMA3_2",  # NOTE only supports LLAMA3_2 for now; this is a torchtune.training.ModelType
        output_dir=str(args.output_dir),
        adapter_checkpoint=None,
        recipe_checkpoint=None,
        resume_from_checkpoint=False,
        safe_serialization=False,
    )
    ckpt_dict: dict[str, Any] = checkpointer.load_checkpoint()
    model = simple_setup_model(model_state_dict=ckpt_dict[torchtune.training.MODEL_KEY])
    LOGGER.info(f"Model loaded successfully: {model}")
    tokenizer_extended, special_tokens = setup_llama3_tokenizer(args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH)
    # NOTE FullModelHFCheckpointer writes the input config.json to the output_dir on __init__ -> forced to overwrite
    extend_config(
        args.output_dir / LLAMA_3_2_CONFIG_RELPATH,
        bos_token_id=special_tokens[LLAMA_BOS_TOKEN],
        eos_token_id=special_tokens[LLAMA_EOS_TOKEN],
        vocab_size=tokenizer_extended.vocab_size,
        llama_config=configllama3_2_1b,
    )
    extend_model(args.n_new_dsus, args.use_modality_tokens, model, tokenizer_extended, llama_config=configllama3_2_1b)
    LOGGER.info(f"Model extended successfully: {model}")
    ckpt_dict_extended = {torchtune.training.MODEL_KEY: model.state_dict()}
    checkpointer.save_checkpoint(ckpt_dict_extended, epoch=0, intermediate_checkpoint=False, adapter_only=False)


if __name__ == "__main__":
    main(parse_args())
