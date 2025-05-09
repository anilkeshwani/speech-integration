import os
from pathlib import Path

import torch


####################################################################################################
# Constants - Keys from sardalign.constants
####################################################################################################

from sardalign.constants import SEED  # isort: skip # noqa: E402

assert SEED == 42831

####################################################################################################
# Constants - Keys from torchtune.training constrained to values as of v0.5.0
####################################################################################################

from torchtune import training  # isort: skip # noqa: E402

ADAPTER_KEY: str = training.ADAPTER_KEY  # adapter weights such as LoRA weights
assert ADAPTER_KEY == "adapter"

EPOCHS_KEY: str = training.EPOCHS_KEY  # number of epochs completed thus far
assert EPOCHS_KEY == "epochs_run"

MODEL_KEY: str = training.MODEL_KEY  # model weights
assert MODEL_KEY == "model"
OPTIMIZER_KEY: str = training.OPT_KEY  # optimizer state NOTE renamed: OPT_KEY -> OPTIMIZER_KEY
assert OPTIMIZER_KEY == "optimizer"
SEED_KEY: str = training.SEED_KEY  # seed for ensuring reproducibility
assert SEED_KEY == "seed"

TOTAL_EPOCHS_KEY: str = training.TOTAL_EPOCHS_KEY  # total number of epochs
assert TOTAL_EPOCHS_KEY == "total_epochs"

# NOTE torchtune.training exports STEPS_KEY = "steps_run" # number of steps completed thus far - for PPO
STEPS_KEY: str = training.STEPS_KEY  # number of steps completed thus far
assert STEPS_KEY == "steps_run"

# NOTE entirely different meaning cf STEPS_KEY exported by torchtune (used in PPO stage)
GLOBAL_STEP_KEY: str = "global_step"

RNG_KEY: str = training.RNG_KEY  # rng state for ensuring correct training resuming (original use in PPO impl.)
assert RNG_KEY == "rng_state"

# TODO remove this - strongly dislike this from torchtune - included as a reminder of functionality conflict
# MAX_STEPS_KEY: str = training.MAX_STEPS_KEY
# assert MAX_STEPS_KEY == "max_steps_per_epoch"

# Keys required in the batch as accepted by torchtune's collate functions - used to avoid conflicts when returning
# additional fields from a dataset (e.g. sample IDs to relate generations to ground truth transcripts in ASR evaluation)
RESERVED_BATCH_KEYS: set[str] = {"tokens", "mask", "labels"}

####################################################################################################
# Constants - General
####################################################################################################

SUPPORTED_DTYPES: set[torch.dtype] = {torch.float32, torch.bfloat16}

####################################################################################################
# Constants - Checkpoints and Artefacts
####################################################################################################

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
