import json
import logging
from pathlib import Path

import hydra
from datasets import load_dataset
from evaluate import load
from omegaconf import DictConfig, OmegaConf
from whisper_normalizer.english import EnglishTextNormalizer


LOGGER = logging.getLogger(__name__)


def validate_wer_config(cfg: DictConfig) -> None:
    missing_keys = OmegaConf.missing_keys(cfg)
    if missing_keys:
        raise ValueError(f"Missing keys in config: {missing_keys}")


def extract_texts_from_generations_jsonl(generations_jsonl: Path) -> list[str]:
    texts = []
    with open(generations_jsonl, "r") as f:
        for line in f:
            data = json.loads(line)
            is_single_generation = len(data["outputs"]) == 1
            if is_single_generation:
                texts.append(data.pop("outputs").pop(0).pop("text"))
            else:
                raise NotImplementedError("Multiple generations per prompt are not supported by this script.")
    return texts


def ref_from_hf_dataset(cfg_dataset: DictConfig) -> list[str]:
    ds = load_dataset(cfg_dataset.source, split=cfg_dataset.split)
    return list(ds[cfg_dataset.column_map.output])  # "output" maps to the GT text


# fmt: off
# TODO Implement resumpion of W&B run for logging of WER per config to runs per step
import wandb  # noqa: E402 # isort: skip
from ssi.utils import parse_model_path  # noqa: E402 # isort: skip
@hydra.main(version_base=None, config_path="../conf", config_name="wer")
def _main(cfg: DictConfig) -> None:
    model_metadata = parse_model_path(Path(cfg.model), Path(cfg.experiments_root_dir))
    wandb_run_id = model_metadata["wandb_run_id"]
    wandb.init(
        resume="must",
        resume_from=wandb_run_id,
        **cfg.wandb,  # TODO check this is comptible with the other arguments specified here
        settings=wandb.Settings(_disable_stats=True, console="off")

        # Code from wandb.Settings i.e. wandb.sdk.wandb_settings.Settings

        # console: Literal["auto", "off", "wrap", "redirect", "wrap_raw", "wrap_emu"] = Field(
        #     default="auto",
        #     validate_default=True,
        # )
        # """The type of console capture to be applied.

        # Possible values are:
        # "auto" - Automatically selects the console capture method based on the
        # system environment and settings.

        # "off" - Disables console capture.

        # "redirect" - Redirects low-level file descriptors for capturing output.

        # "wrap" - Overrides the write methods of sys.stdout/sys.stderr. Will be
        # mapped to either "wrap_raw" or "wrap_emu" based on the state of the system.

        # "wrap_raw" - Same as "wrap" but captures raw output directly instead of
        # through an emulator. Derived from the `wrap` setting and should not be set manually.

        # "wrap_emu" - Same as "wrap" but captures output through an emulator.
        # Derived from the `wrap` setting and should not be set manually.
        # """

        # x_disable_meta: bool = False
        # """Flag to disable the collection of system metadata."""

        # x_disable_setproctitle: bool = False
        # """Flag to disable using setproctitle for the internal process in the legacy service.

        # This is deprecated and will be removed in future versions.
        # """

        # x_disable_stats: bool = False
        # """Flag to disable the collection of system metrics."""

        # x_disable_viewer: bool = False
        # """Flag to disable the early viewer query."""

        # x_disable_machine_info: bool = False
        # """Flag to disable automatic machine info collection."""
    )
# fmt: on


@hydra.main(version_base=None, config_path="../conf", config_name="wer")
def main(cfg: DictConfig) -> None:
    wer_metric = load("wer")
    generations_jsonl = Path(cfg.generations_jsonl)
    generated = extract_texts_from_generations_jsonl(generations_jsonl)
    reference = ref_from_hf_dataset(cfg.data.dev.dataset)
    if cfg.normalizer == "whisper":
        english_normalizer = EnglishTextNormalizer()  # TODO update for other languages
        generated = [english_normalizer(text) for text in generated]
        reference = [english_normalizer(text) for text in reference]
    elif cfg.normalizer is None:
        LOGGER.info("No normalizer specified, skipping text normalization.")
    else:
        raise NotImplementedError(f"Unsupported normalizer: {cfg.normalizer}. Supported: 'whisper' or None (null).")
    wer = wer_metric.compute(predictions=generated, references=reference)
    wer_json = generations_jsonl.parent / "wer.json"
    with open(wer_json, "w") as f:
        json.dump({"wer": wer}, f, indent=4)
    LOGGER.info(f"WER: {wer:.3f}. Saved to {wer_json!s}")


if __name__ == "__main__":
    main()
