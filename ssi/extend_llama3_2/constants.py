import os
from pathlib import Path

from torchtune.models.llama3._tokenizer import LLAMA3_SPECIAL_TOKENS


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

from sardalign.constants import SEED  # noqa F401
