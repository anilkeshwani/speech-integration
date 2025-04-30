from dataclasses import asdict, dataclass


@dataclass
class ConfigLlama3_2:
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

    @property
    def vocab_size(self) -> int:
        return self._base_vocab_size_txt + self._n_special_txt + self.n_dsus + (2 * self._modality_tokens)

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
