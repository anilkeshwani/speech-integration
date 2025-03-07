import itertools
import logging
import os
import sys
import time

import hydra
import torch
from omegaconf import DictConfig
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from torchtune import generation
from torchtune.data._common import CROSS_ENTROPY_IGNORE_IDX
from torchtune.modules import TransformerDecoder
from torchtune.training import get_dtype
from torchtune.utils import batch_to_device, get_device
from tqdm import tqdm

from ssi.checkpoint import FullModelHFCheckpointer
from ssi.constants import MODEL_KEY
from ssi.data import setup_sft_data
from ssi.model import setup_llama3_2_1b
from ssi.tokenizer import setup_llama3_tokenizer


################################################################################
# Config to use; see conf/ directory
################################################################################

CONFIG_NAME = "generate.yaml"  # generate.yaml <- sft.yaml <- cpt.yaml (inheritance)

IGNORE_IDX: int = CROSS_ENTROPY_IGNORE_IDX  # change if using different loss ignore idx

# Debug mode
"""
`None` -> don't set any PyTorch global values
"default" or 0 -> don't error or warn on nondeterministic operations and additionally enable PyTorch CuDNN benchmark
"warn" or 1 -> warn on nondeterministic operations and disable PyTorch CuDNN benchmark
"error" or 2 -> error on nondeterministic operations and disable PyTorch CuDNN benchmark
"""
DEBUG_MODE: str | None = None

################################################################################
# Preamble
################################################################################

logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOGLEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
)

LOGGER = logging.getLogger(__name__)


@hydra.main(config_path="conf", config_name=CONFIG_NAME, version_base=None)
@torch.inference_mode()
def generate(cfg: DictConfig):
    DEVICE: torch.device = get_device(cfg.device)
    DTYPE: torch.dtype = get_dtype(cfg.dtype)
    custom_generate_next_token = None
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    model: TransformerDecoder = setup_llama3_2_1b(
        cfg=cfg,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=DTYPE,
        device_default=DEVICE,
    )
    model.to(device=DEVICE)
    model.eval()

    tokenizer, special_tokens = setup_llama3_tokenizer(**cfg.tokenizer)
    data_dev, sampler_dev = setup_sft_data(cfg_dataset=cfg.data.dev, model_tokenizer=tokenizer)
    data_dev.inference = True  # TODO BUG this sets inference on the DataLoader instance - meaningless
    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    # TODO ensure this check doesn't cause the DL to skip the first sample
    if next(iter(data_dev))["tokens"].size(0) != 1:
        raise ValueError("Generation only supports batch size 1")
    for i, batch in tqdm(enumerate(data_dev), total=len(data_dev)):
        t0 = time.perf_counter()
        batch_to_device(batch, DEVICE)  # in-place
        prompt = batch["tokens"]  # batch["tokens"].masked_select(batch["tokens"] != tokenizer.pad_id) redundant if bs=1
        if cfg.enable_kv_cache:
            with DEVICE:
                model.setup_caches(batch_size=1, dtype=DTYPE, decoder_max_seq_len=prompt.numel() + cfg.max_new_tokens)
        generated_tokens, logits = generation.generate(
            model=model,
            prompt=prompt,
            max_generated_tokens=cfg.max_new_tokens,
            pad_id=tokenizer.pad_id,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            stop_tokens=tokenizer.stop_tokens,
            custom_generate_next_token=custom_generate_next_token,
        )

        print(f"{prompt.size() = }")
        print(f"{generated_tokens.size() = }")
        print(f"{logits.size() = }")
        prompt_length = prompt.size(1)
        print(f"{prompt_length = }")
        print(tokenizer.decode(generated_tokens[0, prompt_length:].tolist()))
        generated_tokens_list = generated_tokens.tolist()
        t = time.perf_counter() - t0

        promptdec = tokenizer.decode(prompt.squeeze(0).tolist())
        gendec = tokenizer.decode(generated_tokens.squeeze(0).tolist(), truncate_at_eos=True, skip_special_tokens=True)
        gendec_notrunc = tokenizer.decode(
            generated_tokens.squeeze(0).tolist(), truncate_at_eos=False, skip_special_tokens=True
        )
        gendec_notrunc_incspec = tokenizer.decode(
            generated_tokens.squeeze(0).tolist(), truncate_at_eos=False, skip_special_tokens=False
        )

        LOGGER.info(f"{promptdec = }")
        LOGGER.info(f"{gendec = }")
        LOGGER.info(f"{gendec_notrunc = }")
        LOGGER.info(f"{gendec_notrunc_incspec = }")

        if cfg.observability:
            tokens_generated = len(generated_tokens[0]) - prompt.size(0)
            tokens_sec = tokens_generated / t
            LOGGER.info(f"Time for inference: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
            LOGGER.info(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
            LOGGER.info(f"Memory used: {torch.cuda.max_memory_allocated() / 1e9:.02f} GB")

        # # Explicitly release all unoccupied (inc. cache allocated) memory
        # del batch, generated_tokens, logits
        # torch.cuda.empty_cache()


if __name__ == "__main__":
    generate()
