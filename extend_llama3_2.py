#!/usr/bin/env python

import base64
import hashlib
import json
import logging
import os
from argparse import ArgumentParser, Namespace
from pathlib import Path
from pprint import pformat, pp
from typing import Any

import torch
import torchtune.training
from sardalign.utils import dsu2pua, multivariate_normal_from_weights, seed_everything
from tiktoken.load import load_tiktoken_bpe
from torch import nn
from torchtune import utils
from torchtune.data import PromptTemplate
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.models.llama3._tokenizer import LLAMA3_SPECIAL_TOKENS
from torchtune.models.llama3_2 import llama3_2_1b
from torchtune.modules import TiedLinear, TransformerDecoder
from torchtune.training import ModelType  # noqa: F401
from torchtune.training.checkpointing import FullModelHFCheckpointer  # NOTE also exported by torchtune.training


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
LOGGER = logging.getLogger(__name__)

# Constants
HAFH_DIR = Path(os.environ.get("HAFH", "/mnt/scratch-artemis/anilkeshwani/"))
TORCHTUNE_BASE_MODELS_DIR = HAFH_DIR / "models" / "base" / "torchtune"
TORCHTUNE_EXTENDED_MODELS_DIR = HAFH_DIR / "models" / "extended" / "torchtune"
LLAMA_3_2_1B_BASE_DIR = TORCHTUNE_BASE_MODELS_DIR / "Llama-3.2-1B"

# Tokenizer (tiktoken) and model (HF safetensors) path relative to Llama 3.2 directory (from tune download)
LLAMA_3_2_TOKENIZER_RELPATH = Path("original", "tokenizer.model")
LLAMA_3_2_MODEL_RELPATH = Path("model.safetensors")
LLAMA_3_2_CONFIG_RELPATH = Path("config.json")

BASE_VOCAB_SIZE: int = 128_000
SPECIAL_TOKENS_SIZE = len(LLAMA3_SPECIAL_TOKENS)
assert SPECIAL_TOKENS_SIZE == 256, "Unexpected number of special tokens in Llama 3.2 1B. Has the API changed?"

LLAMA_BOS_TOKEN = "<|begin_of_text|>"
LLAMA_EOS_TOKEN = "<|end_of_text|>"

SEED: int = 428_987


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Extend a tokenizer.model and model.safetensors file for DSUs")
    # core
    parser.add_argument("--n_new_dsus", type=int, required=True, help="Number of DSUs to add as tokens")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=LLAMA_3_2_1B_BASE_DIR,
        help="Input Llama 3.2 directory from tune download." f" Default: {LLAMA_3_2_1B_BASE_DIR}",
    )
    parser.add_argument("--output_dir", type=Path, default=None, help="Output directory to save the extended files")
    # other
    parser.add_argument("--base_vocab_size", type=int, default=BASE_VOCAB_SIZE, help="Tokenizer base vocabulary size")

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = TORCHTUNE_EXTENDED_MODELS_DIR / f"{args.input_dir.name}-{args.n_new_dsus}-dsus"

    return args


def add_dsus_to_tiktoken(n_new_dsus: int, tokenizer_model: Path, output_path: Path) -> None:
    """
    Appends new base64-encoded tokens to a base64-encoded ASCII tokenizer.model file as used by tiktoken.

    Arguments:
        - tokenizer_model: Path to the tokenizer.model file
        - n_new_dsus: Number of DSUs to add as tokens. Converted to PUA tokens via dsu2pua.
    """
    if output_path.exists():
        raise FileExistsError(f"Extended tokenizer output already exists at: {output_path}")

    # Obtain the new DSUs to add
    new_dsu_tkns = [dsu2pua(i) for i in range(n_new_dsus)]  # NOTE in future can specify start/end idxs for new DSUs

    with open(tokenizer_model, "r") as file:
        base_tokenizer_lines = file.readlines()

    # Create a dict[bytes, int] dictionary of the current vocabulary - to test for duplicates
    vocabulary: dict[bytes, int] = {}
    for line in base_tokenizer_lines:
        token, rank = line.split()
        vocabulary[base64.b64decode(token.encode("utf-8"))] = int(rank)

    # Get next merge rank
    rank = max(vocabulary.values()) + 1  # in case tokenizer.model is not sorted by merge rank

    # Prepare new lines with base64-encoded tokens
    new_tokenizer_lines = []
    for i, token in enumerate(new_dsu_tkns):
        token_bytes: bytes = token.encode("utf-8")
        if token_bytes in vocabulary:
            LOGGER.warning(f"Token {token} (idx: {i}) already exists in the vocabulary")
            continue
        token_b64_ascii = base64.b64encode(token_bytes).decode("utf-8")
        new_tokenizer_lines.append(f"{token_b64_ascii} {rank}\n")
        rank += 1

    print(f"Added {len(new_tokenizer_lines)} tokens to {tokenizer_model}")

    # Write the extended tokenizer.model file to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x") as file:
        file.writelines(base_tokenizer_lines + new_tokenizer_lines)

    print(f"Extended tokenizer.model saved to {output_path}")


def setup_model(model_state_dict: dict[str, Any], device: str = "cpu") -> nn.Module:
    """Set up and load a model with the given state_dict"""
    with torchtune.training.set_default_dtype(torch.float32), utils.get_device(device=device):
        model = llama3_2_1b()
    model.load_state_dict(model_state_dict)
    torchtune.training.validate_expected_param_dtype(model.named_parameters(), dtype=torch.float32)  # check fp32
    return model


def setup_llama3_tokenizer(
    tokenizer_model: Path,
    max_seq_len: int | None = None,
    prompt_template: PromptTemplate | None = None,
    verbose: bool = True,
) -> tuple[Llama3Tokenizer, dict[str, int]]:
    with open(tokenizer_model, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    mergeable_ranks = load_tiktoken_bpe(str(tokenizer_model), expected_hash)  # load BPE merges from tokenizer.model
    base_vocab_size = len(mergeable_ranks)
    assert base_vocab_size == max(mergeable_ranks.values()) + 1, "Requirement: base vocab to contiguous and 0-indexed"
    special_tokens_dynamic = {
        k: v
        for k, v in zip(LLAMA3_SPECIAL_TOKENS, range(base_vocab_size, base_vocab_size + len(LLAMA3_SPECIAL_TOKENS)))
    }
    tokenizer = Llama3Tokenizer(
        path=str(tokenizer_model),
        special_tokens=special_tokens_dynamic,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
    )
    if verbose:
        print(f"Loaded Llama 3 tiktoken tokenizer from: {tokenizer_model}")
    pretty_special_tokens = pformat(special_tokens_dynamic, sort_dicts=False, underscore_numbers=True)
    if verbose:
        print(f"Llama3 special tokens (dynamic) added to tokenizer: {pretty_special_tokens}")
        print(f"Tokenizer base vocabulary size (BPE merges file): {base_vocab_size}")
        print(f"Llama 3 tiktoken tokenizer vocabulary size: {tokenizer.vocab_size}")
    return tokenizer, special_tokens_dynamic


def extend_model(
    n_new_dsus: int,
    model: TransformerDecoder,
    extended_tokenizer,
    base_vocab_size: int = BASE_VOCAB_SIZE,
    special_tokens_size: int = SPECIAL_TOKENS_SIZE,
) -> None:
    """Extends a Llama 3 1B model's input embedding layer and tied output layer in place"""
    emb_orig = model.tok_embeddings.weight.data.clone()  # retain original embeddings
    # TODO check whether Llama 3.2 3B has the same embedding size - update comments/docstrings if 3B supported
    assert emb_orig.size() == torch.Size([128_256, 2048]), "Unexpected embedding size for Llama 3.2 1B"
    embeddings = model.tok_embeddings.weight.data
    base_vocab_embeddings = embeddings[:base_vocab_size, :]
    special_tokens_embeddings = embeddings[base_vocab_size:, :]
    assert extended_tokenizer.vocab_size == base_vocab_size + special_tokens_size + n_new_dsus
    mvgaussian = multivariate_normal_from_weights(base_vocab_embeddings, sigma_scaling=1e-5)  # 1e-5 is the default
    new_token_embeddings = mvgaussian.sample(torch.Size((n_new_dsus,)))
    # NOTE TransformerDecoder needs an nn.Embedding module as tok_embeddings and input to TiedLinear
    model.tok_embeddings = nn.Embedding.from_pretrained(
        torch.cat((base_vocab_embeddings, new_token_embeddings, special_tokens_embeddings), dim=0)
    )
    model.output = TiedLinear(model.tok_embeddings)  # F.linear(x, self.tied_module.weight)
    # validate new embeddings
    assert model.tok_embeddings.weight.data[:base_vocab_size, :].equal(emb_orig[:base_vocab_size, :])
    assert model.tok_embeddings.weight.data[-special_tokens_size:, :].equal(emb_orig[-special_tokens_size:, :])
    assert len(model.tok_embeddings.weight.data) == extended_tokenizer.vocab_size
    assert len(model.tok_embeddings.weight.data) == base_vocab_size + special_tokens_size + n_new_dsus  # redundant
    assert len(model.tok_embeddings.weight.data) - len(emb_orig) == n_new_dsus

    print(f"Added {n_new_dsus} new embeddings to the model (in memory)")


def extend_config(config_json: Path, bos_token_id: int, eos_token_id: int, vocab_size: int) -> None:
    with open(config_json, "r") as f:
        config = json.load(f)
    assert config.pop("bos_token_id") == 128_000
    assert config.pop("eos_token_id") == 128_001
    assert config.pop("vocab_size") == BASE_VOCAB_SIZE + SPECIAL_TOKENS_SIZE == 128256
    config["bos_token_id"] = bos_token_id
    config["eos_token_id"] = eos_token_id
    config["vocab_size"] = vocab_size
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Updated config.json with new bos_token_id, eos_token_id, and vocab_size: {config_json}")


def main(args: Namespace) -> None:
    seed_everything(SEED)  # reproducibility
    add_dsus_to_tiktoken(
        args.n_new_dsus, args.input_dir / LLAMA_3_2_TOKENIZER_RELPATH, args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH
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
    model = setup_model(model_state_dict=ckpt_dict[torchtune.training.MODEL_KEY])
    print(f"Model loaded successfully: {model}")
    tokenizer_extended, special_tokens = setup_llama3_tokenizer(args.output_dir / LLAMA_3_2_TOKENIZER_RELPATH)
    # NOTE FullModelHFCheckpointer writes the input config.json to the output_dir on __init__ -> forced to overwrite
    extend_config(
        args.output_dir / LLAMA_3_2_CONFIG_RELPATH,
        bos_token_id=special_tokens[LLAMA_BOS_TOKEN],
        eos_token_id=special_tokens[LLAMA_EOS_TOKEN],
        vocab_size=tokenizer_extended.vocab_size,
    )
    extend_model(args.n_new_dsus, model, tokenizer_extended)  # in place
    print(f"Model extended successfully: {model}")
    ckpt_dict_extended = {torchtune.training.MODEL_KEY: model.state_dict()}
    checkpointer.save_checkpoint(ckpt_dict_extended, epoch=0, intermediate_checkpoint=False, adapter_only=False)


if __name__ == "__main__":
    main(parse_args())
