import hashlib
from pathlib import Path
from pprint import pformat

from tiktoken.load import load_tiktoken_bpe
from torchtune.data import PromptTemplate
from torchtune.models.llama3._tokenizer import LLAMA3_SPECIAL_TOKENS

# this is the only module in ssi where we instantiate a Llama3Tokenizer (all other references are type hints)
from ssi.tokenizer.monkeypatch import Llama3TokenizerPUA as Llama3Tokenizer


# NOTE Check hard-coded to detect breaking API changes from torchtune
if len(LLAMA3_SPECIAL_TOKENS) != 256:
    raise RuntimeError("Unexpected number of special tokens in Llama 3.2 1B. Has the API changed?")


def setup_llama3_tokenizer(
    path: Path,
    max_seq_len: int | None = None,
    prompt_template: PromptTemplate | None = None,
    verbose: bool = True,
) -> tuple[Llama3Tokenizer, dict[str, int]]:
    with open(path, "rb") as f:
        expected_hash = hashlib.sha256(f.read()).hexdigest()
    mergeable_ranks = load_tiktoken_bpe(str(path), expected_hash)  # load BPE merges from tokenizer.model
    base_vocab_size = len(mergeable_ranks)
    assert base_vocab_size == max(mergeable_ranks.values()) + 1, "Requirement: base vocab to contiguous and 0-indexed"
    special_tokens_dynamic = {
        k: v
        for k, v in zip(LLAMA3_SPECIAL_TOKENS, range(base_vocab_size, base_vocab_size + len(LLAMA3_SPECIAL_TOKENS)))
    }
    tokenizer = Llama3Tokenizer(
        path=str(path),
        special_tokens=special_tokens_dynamic,
        max_seq_len=max_seq_len,
        prompt_template=prompt_template,
    )
    if verbose:
        print(f"Loaded Llama 3 tiktoken tokenizer from: {path}")
    pretty_special_tokens = pformat(special_tokens_dynamic, sort_dicts=False, underscore_numbers=True)
    if verbose:
        print(f"Llama3 special tokens (dynamic) added to tokenizer: {pretty_special_tokens}")
        print(f"Tokenizer base vocabulary size (BPE merges file): {base_vocab_size}")
        print(f"Llama 3 tiktoken tokenizer vocabulary size: {tokenizer.vocab_size}")
    return tokenizer, special_tokens_dynamic
