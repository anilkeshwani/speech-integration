import base64
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import torchtune.training
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sardalign.constants import MODALITY_TOKEN_SPEECH, MODALITY_TOKEN_TEXT
from sardalign.utils import dsu2pua, multivariate_normal_from_weights
from torch import nn
from torchtune import utils
from torchtune.models.llama3 import Llama3Tokenizer
from torchtune.models.llama3_2 import llama3_2_1b
from torchtune.modules import TiedLinear, TransformerDecoder

from ssi.llama_configs import ConfigLlama3_2


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


def extend_tiktoken(n_new_dsus: int, use_modality_tokens: bool, tokenizer_model: Path, output_path: Path) -> None:
    """
    Appends new base64-encoded tokens to a base64-encoded ASCII tokenizer.model file as used by tiktoken.

    Arguments:
        - tokenizer_model: Path to the tokenizer.model file
        - n_new_dsus: Number of DSUs to add as tokens. Converted to PUA tokens via dsu2pua.
    """
    if output_path.exists():
        raise FileExistsError(f"Extended tokenizer output already exists at: {output_path}")

    with open(tokenizer_model, "r") as file:
        base_tokenizer_lines: list[str] = file.readlines()

    # Create a dict[bytes, int] dictionary of the current vocabulary - to test for duplicates
    vocabulary: dict[bytes, int] = {}
    for line in base_tokenizer_lines:
        token, rnk = line.split()
        vocabulary[base64.b64decode(token.encode("utf-8"))] = int(rnk)

    # Get next merge rank
    rank: int = max(vocabulary.values()) + 1  # in case tokenizer.model is not sorted by merge rank

    # Prepare new lines with base64-encoded tokens
    def _create_token_list(tks: list[str]) -> list[str]:
        nonlocal rank  # NOTE vocabulary also used from enclosing scope but not mutated
        tokenizer_lines = []
        for i, token in enumerate(tks):
            token_bytes: bytes = token.encode("utf-8")
            if token_bytes in vocabulary:
                raise RuntimeError(f"Token {token} (idx: {i}) already exists in the vocabulary")
            token_b64_ascii = base64.b64encode(token_bytes).decode("utf-8")
            tokenizer_lines.append(f"{token_b64_ascii} {rank}\n")
            rank += 1
        return tokenizer_lines

    # Add DSU tokens
    dsu_tkns = [dsu2pua(i) for i in range(n_new_dsus)]  # TODO in future specify start/end idxs for new DSUs?
    dsu_tokenizer_lines = _create_token_list(dsu_tkns)
    LOGGER.info(f"Adding {len(dsu_tokenizer_lines)} DSU tokens to {tokenizer_model!s}")

    # Add modality tokens
    if use_modality_tokens:
        modality_tokens: list[str] = [MODALITY_TOKEN_TEXT, MODALITY_TOKEN_SPEECH]
        modality_tokenizer_lines = _create_token_list(modality_tokens)
        LOGGER.info(f"Adding {len(modality_tokenizer_lines)} modality tokens to {tokenizer_model!s}")
    else:
        LOGGER.info(f"No Modality Tokens added to {tokenizer_model}")

    # Write the extended tokenizer.model file to disk
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "x") as file:
        file.writelines(base_tokenizer_lines + dsu_tokenizer_lines + modality_tokenizer_lines)

    LOGGER.info(f"Extended tokenizer.model saved to {output_path}")


def extend_model(
    n_new_dsus: int,
    use_modality_tokens: bool,
    model: TransformerDecoder,
    extended_tokenizer: Llama3Tokenizer,
    llama_config: ConfigLlama3_2,
) -> None:
    """Extends a Llama 3 1B model's input embedding layer and tied output layer in place"""
    base_vocab_size: int = llama_config._base_vocab_size_txt
    special_tokens_size: int = llama_config._n_special_txt
    emb_orig = model.tok_embeddings.weight.data.clone()  # retain original embeddings
    # TODO check whether Llama 3.2 3B has the same embedding size - update comments/docstrings if 3B supported
    assert emb_orig.size() == torch.Size([128_256, 2048]), "Unexpected embedding size for Llama 3.2 1B"
    embeddings = model.tok_embeddings.weight.data
    base_vocab_embeddings = embeddings[:base_vocab_size, :]
    special_tokens_embeddings = embeddings[base_vocab_size:, :]
    assert extended_tokenizer.vocab_size == base_vocab_size + special_tokens_size + n_new_dsus + 2 * use_modality_tokens
    mvgaussian = multivariate_normal_from_weights(base_vocab_embeddings, sigma_scaling=1e-5)  # 1e-5 is the default
    new_token_embeddings = mvgaussian.sample(torch.Size((n_new_dsus + 2 * use_modality_tokens,)))
    # NOTE TransformerDecoder needs an nn.Embedding module as tok_embeddings and input to TiedLinear
    model.tok_embeddings = nn.Embedding.from_pretrained(
        torch.cat((base_vocab_embeddings, new_token_embeddings, special_tokens_embeddings), dim=0)
    )
    model.output = TiedLinear(model.tok_embeddings)  # F.linear(x, self.tied_module.weight)
    # validate new embeddings
    assert model.tok_embeddings.weight.data[:base_vocab_size, :].equal(emb_orig[:base_vocab_size, :])
    assert model.tok_embeddings.weight.data[-special_tokens_size:, :].equal(emb_orig[-special_tokens_size:, :])
    assert len(model.tok_embeddings.weight.data) == extended_tokenizer.vocab_size
    assert len(model.tok_embeddings.weight.data) - len(emb_orig) == n_new_dsus + 2 * use_modality_tokens
    LOGGER.info(f"Added {n_new_dsus} new DSU embeddings to the model (in memory)")
    if use_modality_tokens:
        LOGGER.info("Added embeddings for modality tokens to the model embedding weights (in memory)")
    else:
        LOGGER.info("No embeddings for modality tokens added to model embedding weights")


def extend_config(
    config_json: Path,
    bos_token_id: int,
    eos_token_id: int,
    vocab_size: int,
    llama_config: ConfigLlama3_2,
) -> None:
    base_vocab_size: int = llama_config._base_vocab_size_txt
    special_tokens_size: int = llama_config._n_special_txt
    with open(config_json, "r") as f:
        config = json.load(f)
    assert config.pop("bos_token_id") == 128_000
    assert config.pop("eos_token_id") == 128_001
    assert config.pop("vocab_size") == base_vocab_size + special_tokens_size
    config["bos_token_id"] = bos_token_id
    config["eos_token_id"] = eos_token_id
    config["vocab_size"] = vocab_size
    with open(config_json, "w") as f:
        json.dump(config, f, indent=2)
    LOGGER.info(f"Updated config.json with new bos_token_id, eos_token_id, and vocab_size: {config_json}")


def simple_setup_model(model_state_dict: dict[str, Any], device: str = "cpu") -> TransformerDecoder:
    """Set up and load a model with the given state_dict"""
    with torchtune.training.set_default_dtype(torch.float32), utils.get_device(device=device):
        model = llama3_2_1b()
    model.load_state_dict(model_state_dict)
    torchtune.training.validate_expected_param_dtype(model.named_parameters(), dtype=torch.float32)  # check fp32
    return model
