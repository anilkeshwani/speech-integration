from dataclasses import asdict, dataclass


@dataclass
class ConfigLlama3_2:
    vocab_size: int
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

    @property
    def n_dsus(self) -> int:
        return self._n_dsus

    @n_dsus.setter
    def n_dsus(self, n_dsus: int) -> None:
        if not isinstance(n_dsus, int) or (n_dsus < 0):
            raise ValueError("n_dsus must be a non-negative integer")
        self._n_dsus = n_dsus
    

    @property
    def parameters(self) -> dict:
        _dict = asdict(self)
        _dict.pop("n_dsus")
        return _dict


# Singletons for the configs

configllama3_2_1b = ConfigLlama3_2(
    vocab_size=128_256,
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
    vocab_size=128_256,
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
