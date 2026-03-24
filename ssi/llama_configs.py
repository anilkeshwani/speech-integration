from dataclasses import asdict, dataclass
from typing import NamedTuple

from omegaconf import DictConfig


class ModelCheckpointExpectations(NamedTuple):
    """Expected checkpoint structure for a model architecture.

    Used by checkpoint validation to verify that a checkpoint directory
    matches the model specified in the config. Designed to be produced by
    any model config class — not specific to Llama.

    Attributes:
        model_name: Human-readable model name (for error messages).
        n_shards: Expected number of safetensors shard files.
        num_layers: Expected ``num_hidden_layers`` in config.json.
        hidden_size: Expected ``hidden_size`` in config.json.
        vocab_size: Expected ``vocab_size`` in config.json (after extension).
    """

    model_name: str
    n_shards: int
    num_layers: int
    hidden_size: int
    vocab_size: int


@dataclass
class ConfigLlama3_2:
    """Llama 3.2 model configuration for use with torchtune's ``llama3_2`` builder.

    Computes ``vocab_size`` dynamically from base text vocab, special tokens,
    DSU tokens, and optional modality tokens. The ``parameters`` property returns
    only the fields accepted by ``torchtune.models.llama3_2.llama3_2``.

    Args:
        _base_vocab_size_txt: Number of base text tokens in the tokenizer.
        _n_special_txt: Number of reserved special text tokens.
        num_layers: Number of transformer layers.
        num_heads: Number of query attention heads.
        num_kv_heads: Number of key/value attention heads (GQA).
        embed_dim: Model hidden dimension.
        max_seq_len: Maximum sequence length.
        intermediate_dim: FFN intermediate dimension.
        attn_dropout: Attention dropout probability.
        norm_eps: RMSNorm epsilon.
        rope_base: RoPE base frequency.
        scale_factor: RoPE scaling factor.
        _n_dsus: Number of discrete speech unit tokens. Default 0.
        _modality_tokens: Whether modality boundary tokens are enabled. Default False.
    """

    # NOTE attributes are hidden to prevent being exposed by parameters property to llama3_2 function
    _base_vocab_size_txt: int
    _n_special_txt: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    embed_dim: int
    max_seq_len: int
    intermediate_dim: int
    attn_dropout: float
    norm_eps: float
    rope_base: int
    scale_factor: int
    _n_dsus: int = 0
    _modality_tokens: bool = False

    @property
    def n_dsus(self) -> int:
        return self._n_dsus

    @n_dsus.setter
    def n_dsus(self, n_dsus: int) -> None:
        if not isinstance(n_dsus, int) or (n_dsus < 0):
            raise ValueError("n_dsus must be a non-negative integer")
        self._n_dsus = n_dsus

    @property
    def modality_tokens(self) -> bool:
        return self._modality_tokens

    @modality_tokens.setter
    def modality_tokens(self, enable: bool) -> None:
        if not isinstance(enable, bool):
            raise ValueError("modality_tokens must be boolean")
        self._modality_tokens = enable

    def update_from_speech_cfg(self, cfg_speech: DictConfig) -> None:
        """In-place update of speech-specific hyperparameters from DictConfig"""
        if not isinstance(cfg_speech, DictConfig):
            raise TypeError("cfg_speech must be a DictConfig object")
        configllama3_2_1b.n_dsus = cfg_speech.n_dsus
        configllama3_2_1b.modality_tokens = cfg_speech.use_modality_tokens

    @property
    def vocab_size(self) -> int:
        return self._base_vocab_size_txt + self._n_special_txt + self.n_dsus + (2 * self._modality_tokens)

    @property
    def checkpoint_expectations(self) -> ModelCheckpointExpectations:
        """Expected checkpoint structure for validation.

        Llama 3.2 1B and 3B both fit in a single safetensors shard.
        ``vocab_size`` reflects the current extension state (base + DSUs +
        modality tokens).
        """
        size_label = {2048: "1B", 3072: "3B"}.get(self.embed_dim, f"{self.embed_dim}d")
        return ModelCheckpointExpectations(
            model_name=f"Llama 3.2 {size_label}",
            n_shards=1,
            num_layers=self.num_layers,
            hidden_size=self.embed_dim,
            vocab_size=self.vocab_size,
        )

    @property
    def parameters(self) -> dict:
        """Return (only) the parameters needed to initialise a model with torchtune.models.llama3_2.llama3_2"""
        # NOTE asdict returns a dict of all fields including "private" ones; but not properties or methods
        return {"vocab_size": self.vocab_size} | {k: v for k, v in asdict(self).items() if not k.startswith("_")}


# Singletons for the configs

configllama3_2_1b = ConfigLlama3_2(
    _base_vocab_size_txt=128_000,
    _n_special_txt=256,
    num_layers=16,
    num_heads=32,
    num_kv_heads=8,
    embed_dim=2048,
    max_seq_len=131072,
    intermediate_dim=8192,
    attn_dropout=0.0,
    norm_eps=1e-5,
    rope_base=500_000,
    scale_factor=32,
)

configllama3_2_3b = ConfigLlama3_2(
    _base_vocab_size_txt=128_000,
    _n_special_txt=256,
    num_layers=28,
    num_heads=24,
    num_kv_heads=8,
    embed_dim=3072,
    max_seq_len=131072,
    intermediate_dim=8192,
    attn_dropout=0.0,
    norm_eps=1e-5,
    rope_base=500_000,
    scale_factor=32,
)
