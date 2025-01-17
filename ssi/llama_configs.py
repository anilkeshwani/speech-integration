from dataclasses import dataclass


@dataclass
class ConfigLlama3_2_1B:
    vocab_size: int = 128_256
    num_layers: int = 16
    num_heads: int = 32
    num_kv_heads: int = 8
    embed_dim: int = 2048
    max_seq_len: int = 131072
    intermediate_dim: int = 8192
    attn_dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_base: int = 500_000
    scale_factor: int = 32


@dataclass
class ConfigLlama3_2_3B:
    vocab_size: int = 128_256
    num_layers: int = 28
    num_heads: int = 24
    num_kv_heads: int = 8
    embed_dim: int = 3072
    max_seq_len: int = 131072
    intermediate_dim: int = 8192
    attn_dropout: float = 0.0
    norm_eps: float = 1e-5
    rope_base: int = 500_000
    scale_factor: int = 32
