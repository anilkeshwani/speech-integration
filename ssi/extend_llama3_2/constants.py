import os
from pathlib import Path

from sardalign.constants import SEED  # noqa F401


# Constants
HAFH_DIR = Path(os.environ.get("HAFH", "/mnt/scratch-artemis/anilkeshwani/"))
TORCHTUNE_BASE_MODELS_DIR = HAFH_DIR / "models" / "base"
TORCHTUNE_EXTENDED_MODELS_DIR = HAFH_DIR / "models" / "extended"
LLAMA_3_2_1B_BASE_DIR = TORCHTUNE_BASE_MODELS_DIR / "Llama-3.2-1B"
LLAMA_3_2_3B_BASE_DIR = TORCHTUNE_BASE_MODELS_DIR / "Llama-3.2-3B"

# Tokenizer (tiktoken) and model (HF safetensors) path relative to Llama 3.2 directory (from tune download)
LLAMA_3_2_TOKENIZER_RELPATH = Path("original", "tokenizer.model")
LLAMA_3_2_MODEL_RELPATH = Path("model.safetensors")
LLAMA_3_2_CONFIG_RELPATH = Path("config.json")
LLAMA_3_2_GENERATION_CONFIG_RELPATH = Path("generation_config.json")
LLAMA_3_2_PARAMS_RELPATH = Path("original", "params.json")

# Llama 3.2 tokenizer
LLAMA_BOS_TOKEN = "<|begin_of_text|>"
LLAMA_EOS_TOKEN = "<|end_of_text|>"
