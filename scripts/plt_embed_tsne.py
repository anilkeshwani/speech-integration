#!/usr/bin/env python

"""
Script to generate t-SNE plots of model embeddings from saved checkpoints.
Visualizes different token types (text, DSU, modality, special) in the embedding space.
"""

import atexit
import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig
from sardalign.config import LOG_DATEFMT, LOG_FORMAT, LOG_LEVEL
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchtune import training
from torchtune.training import get_dtype
from torchtune.utils import get_device

from ssi.checkpoint import FullModelHFCheckpointer
from ssi.constants import MODEL_KEY, SEED
from ssi.llama_configs import configllama3_2_1b
from ssi.model import setup_llama3_2_1b
from ssi.tokenizer import setup_llama3_tokenizer
from ssi.train import get_token_type_ranges, validate_train_cfg


logging.basicConfig(
    format=LOG_FORMAT,
    datefmt=LOG_DATEFMT,
    level=os.environ.get("LOG_LEVEL", LOG_LEVEL).upper(),
    stream=sys.stdout,
    force=True,
)

LOGGER = logging.getLogger(__name__)


def extract_llama3_2_embeddings(model: torch.nn.Module) -> np.ndarray:
    # Get the embedding layer (typically model.tok_embeddings.weight)
    if hasattr(model, "tok_embeddings"):
        embeddings: torch.Tensor = model.tok_embeddings.weight.data
    elif hasattr(model, "embed_tokens"):
        raise NotImplementedError("LLaMA3 uses 'tok_embeddings', not 'embed_tokens'")
        # embeddings = model.embed_tokens.weight.data  # not used for Llama 3.2 1B
    else:
        for name, module in model.named_modules():
            if "embed" in name.lower() and hasattr(module, "weight"):
                embeddings = module.weight.data
                LOGGER.info(f"Found possible embeddings in layer: {name} with dim {embeddings.size()}")
                break
        raise ValueError("Could not find embedding layer in model")
    return embeddings.to(torch.float32).cpu().numpy()


def create_token_type_labels(ranges: Dict[str, Tuple[int, int]], vocab_size: int) -> Tuple[np.ndarray, list]:
    """Create labels for each token based on its type."""
    labels = np.full(vocab_size, -1, dtype=int)
    label_names = []

    for i, (token_type, (start, end)) in enumerate(ranges.items()):
        labels[start : end + 1] = i
        label_names.append(token_type)

    return labels, label_names


def plot_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    label_names: list,
    output_dir: Path | str,
    perplexity: int,
    n_components: int = 2,
    n_pca_components: int = -1,
    tsne_init="pca",
    tsne_lr="auto",
):
    LOGGER.info(f"Running t-SNE with perplexity={perplexity}, n_components={n_components}")

    # Optionally pre-reduce dimensionality with PCA to reduce compute for large embeddings NOTE 2048 for Llama 3.2 1B
    if n_pca_components > 0 and embeddings.shape[1] > n_pca_components:
        LOGGER.info("Reducing dimensionality with PCA first...")
        pca = PCA(n_components=n_pca_components, random_state=SEED)
        embeddings_reduced = pca.fit_transform(embeddings)
        LOGGER.info(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
    else:
        embeddings_reduced = embeddings

    # Run t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=SEED,
        init=tsne_init,
        learning_rate=tsne_lr,
    )
    embeddings_tsne = tsne.fit_transform(embeddings_reduced)

    # Create plot
    plt.figure(figsize=(12, 10))

    # Color palette
    colors = sns.color_palette("husl", len(label_names))

    for i, label_name in enumerate(label_names):
        mask = labels == i
        if mask.sum() > 0:
            plt.scatter(
                embeddings_tsne[mask, 0],
                embeddings_tsne[mask, 1],
                c=[colors[i]],
                label=f"{label_name} ({mask.sum()} tokens)",
                alpha=0.7,
                s=20,
            )

    plt.title(f"t-SNE Visualization of Model Embeddings (perplexity={perplexity})")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save plot
    output_file = output_dir / f"embeddings_tsne_perplexity_{perplexity}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved t-SNE plot to: {output_file}")
    plt.close()


def plot_embeddings_by_token_type(embeddings: np.ndarray, ranges: Dict[str, Tuple[int, int]], output_dir: Path):
    """Create separate plots for each token type."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for i, (token_type, (start, end)) in enumerate(ranges.items()):
        if i >= len(axes):
            break

        token_embeddings = embeddings[start : end + 1]

        if len(token_embeddings) > 1:
            # Run t-SNE on this token type only
            if len(token_embeddings) > 30:  # Only if we have enough samples
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, len(token_embeddings) // 3),
                    random_state=SEED,
                    init="pca",
                    learning_rate="auto",
                )
                token_tsne = tsne.fit_transform(token_embeddings)

                axes[i].scatter(token_tsne[:, 0], token_tsne[:, 1], alpha=0.7, s=20)
                axes[i].set_title(f"{token_type.capitalize()} Tokens ({len(token_embeddings)} tokens)")
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(
                    0.5,
                    0.5,
                    f"Too few {token_type} tokens\nfor t-SNE ({len(token_embeddings)})",
                    ha="center",
                    va="center",
                    transform=axes[i].transAxes,
                )
                axes[i].set_title(f"{token_type.capitalize()} Tokens")
        else:
            axes[i].text(0.5, 0.5, f"No {token_type} tokens", ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"{token_type.capitalize()} Tokens")

    # Hide unused subplots
    for i in range(len(ranges), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    output_file = output_dir / "embeddings_by_token_type.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    LOGGER.info(f"Saved token type plots to: {output_file}")
    plt.close()


@hydra.main(config_path="../conf", config_name="tsne", version_base=None)
def main(cfg: DictConfig) -> None:
    validate_train_cfg(cfg)  # NOTE fine for our purposes here
    training.set_seed(seed=SEED, debug_mode=cfg.debug_mode)
    DEVICE: torch.device = get_device("cpu")  # NOTE removed get_device(cfg.device)
    DTYPE: torch.dtype = get_dtype(cfg.dtype)
    output_dir = Path(cfg.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"Created output directory: {output_dir}")
    # One-off tmpdir for checkpointer output and other junk - avoid writing to 100GB space-limited /tmp on Artemis
    tmpdir = Path(tempfile.mkdtemp(prefix="plot_embeddings_"))
    atexit.register(lambda: shutil.rmtree(tmpdir, ignore_errors=True))
    if cfg.checkpointer.output_dir is None:
        cfg.checkpointer.output_dir = tmpdir
    checkpointer = FullModelHFCheckpointer(**cfg.checkpointer)
    ckpt_dict = checkpointer.load_checkpoint()
    configllama3_2_1b.update_from_speech_cfg(cfg.speech)  # in-place
    model = setup_llama3_2_1b(
        cfg=cfg,
        llama_config=configllama3_2_1b,
        model_state_dict=ckpt_dict[MODEL_KEY],  # NOTE require model ckpt
        dtype_default=DTYPE,
        device_default=DEVICE,
    )
    model.to(device=DEVICE)
    model.eval()

    # Extract embeddings
    embeddings = extract_llama3_2_embeddings(model)
    LOGGER.info(f"Embeddings shape: {embeddings.shape}")

    # Token type ranges and labels
    ranges = get_token_type_ranges(configllama3_2_1b)
    LOGGER.info("Token type ranges:")
    for token_type, (start, end) in ranges.items():
        LOGGER.info(f"  {token_type}: {start}-{end} ({end-start+1} tokens)")
    labels, label_names = create_token_type_labels(ranges, embeddings.shape[0])

    # Generate t-SNE plots with different perplexity values
    for perplexity in cfg.tsne.perplexities:
        try:
            plot_embeddings(embeddings, labels, label_names, output_dir, perplexity)
        except Exception as e:
            LOGGER.error(f"Failed to create t-SNE plot with perplexity {perplexity}: {e}")

    # Generate plots by token type
    try:
        plot_embeddings_by_token_type(embeddings, ranges, output_dir)
    except Exception as e:
        LOGGER.error(f"Failed to create token type plots: {e}")

    LOGGER.info("Embedding visualization complete!")


if __name__ == "__main__":
    main()
