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
