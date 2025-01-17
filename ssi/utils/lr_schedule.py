import math


def lr_lambda(
    current_step: int,
    num_warmup_steps: int,
    num_training_steps: int,
    num_cycles: float,
) -> float:
    # linear warmup phase
    if current_step < num_warmup_steps:
        return current_step / max(1, num_warmup_steps)
    # cosine
    progress = (current_step - num_warmup_steps) / max(1, num_training_steps - num_warmup_steps)
    cosine_lr_multiple = 0.5 * (1.0 + math.cos(math.pi * num_cycles * 2.0 * progress))
    return max(0.0, cosine_lr_multiple)
