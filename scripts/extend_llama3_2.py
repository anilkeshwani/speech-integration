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
from torchtune.training.checkpointing._utils import SUFFIXES_TO_NOT_COPY

from ssi.checkpoint import FullModelHFCheckpointer
from ssi.constants import (
    LLAMA_3_2_1B_BASE_DIR,
    LLAMA_3_2_CONFIG_RELPATH,
    LLAMA_3_2_GENERATION_CONFIG_RELPATH,
    LLAMA_3_2_PARAMS_RELPATH,
    LLAMA_3_2_TOKENIZER_RELPATH,
    LLAMA_BOS_TOKEN,
    LLAMA_EOS_TOKEN,
    SEED,
    TORCHTUNE_EXTENDED_MODELS_DIR,
)
from ssi.extend_llama3_2 import (
    extend_config,
    extend_generation_config,
    extend_model,
    extend_params,
    extend_tiktoken,
    simple_setup_model,
)
from ssi.llama_configs import configllama3_2_1b
from ssi.tokenizer import setup_llama3_tokenizer


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
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
    args = parser.parse_args()
    if args.output_dir is None:
        dirname = f"{args.input_dir.name}-{args.n_new_dsus}-dsus"
        if not args.use_modality_tokens:
            dirname += "-no_modality_tokens"
        args.output_dir = TORCHTUNE_EXTENDED_MODELS_DIR / dirname
    return args


def main(args: Namespace) -> None:
    # Preamble
    seed_everything(SEED)  # reproducibility
    LLAMA_CFG = configllama3_2_1b  # NOTE script currently supports 1B model
    # Load base checkpoint
    checkpointer = FullModelHFCheckpointer(
        checkpoint_dir=args.input_dir,
        checkpoint_files=["model.safetensors"],
        config_json=args.input_dir / LLAMA_3_2_CONFIG_RELPATH,
        output_dir=args.output_dir,
    )
    ckpt_dict: dict[str, Any] = checkpointer.load_checkpoint()
    # Initialize model
    model = simple_setup_model(model_state_dict=ckpt_dict[torchtune.training.MODEL_KEY])
    # Extend model
    extend_model(args.n_new_dsus, args.use_modality_tokens, model, llama_config=LLAMA_CFG)
    # Save extended model
    HF_TOKENIZER_CONFIGS = ["tokenizer_config.json", "tokenizer.json"]
    ignore_suffixes: list[str] = SUFFIXES_TO_NOT_COPY + [".txt", ".md"] + HF_TOKENIZER_CONFIGS
    checkpointer.save_checkpoint(
        model.state_dict(),
        optimizer_state_dict=None,
        epoch=0,
        global_step=0,
        seed=SEED,
        save_training_state=False,
        output_dir=args.output_dir,
        ignore_suffixes=ignore_suffixes,
    )
    # Extend tokenizer (in place)
    extend_tiktoken(
        args.n_new_dsus,
        args.use_modality_tokens,
        args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH,  # passing output_dir effectively performs extension in-place
        args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH,
    )
    # Load extended tokenizer (for checks and config extension)
    tokenizer_extended, special_tokens = setup_llama3_tokenizer(args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH)
    # Extend configuration files:
    # - config.json (Hugging Face)
    # - params.json (Meta Llama)
    # - generation_config.json (Hugging Face; vLLM)
    extend_config(
        args.output_dir / LLAMA_3_2_CONFIG_RELPATH,
        bos_token_id=special_tokens[LLAMA_BOS_TOKEN],
        eos_token_id=special_tokens[LLAMA_EOS_TOKEN],
        vocab_size=tokenizer_extended.vocab_size,
        llama_config=LLAMA_CFG,
    )
    extend_params(
        args.output_dir / LLAMA_3_2_PARAMS_RELPATH,
        vocab_size=tokenizer_extended.vocab_size,
        llama_config=LLAMA_CFG,
    )
    extend_generation_config(
        args.output_dir / LLAMA_3_2_GENERATION_CONFIG_RELPATH,
        bos_token_id=special_tokens[LLAMA_BOS_TOKEN],
        eos_token_id=special_tokens[LLAMA_EOS_TOKEN],
    )

    # Checks
    base_vocab_size: int = LLAMA_CFG._base_vocab_size_txt
    special_tokens_size: int = LLAMA_CFG._n_special_txt
    assert (
        tokenizer_extended.vocab_size
        == base_vocab_size + special_tokens_size + args.n_new_dsus + 2 * args.use_modality_tokens
    )
    assert len(model.tok_embeddings.weight.data) == tokenizer_extended.vocab_size


if __name__ == "__main__":
    main(parse_args())
