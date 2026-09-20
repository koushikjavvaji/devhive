from dataclasses import dataclass


@dataclass
class GPTConfig:
    vocab_size: int = 8192
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    n_kv_heads: int = 2
    d_ff: int = 1024
    max_seq_len: int = 1024
    rms_eps: float = 1e-6
