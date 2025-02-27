from sardalign.constants import SEED
from torchtune import training


# Keys: based on torchtune.training constrained to values as of v0.5.0
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
