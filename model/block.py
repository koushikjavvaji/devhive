import torch
import torch.nn as nn

from model.attention import GroupedQueryAttention
from model.feedforward import SwiGLU
from model.norm import RMSNorm


class TransformerBlock(nn.Module):

    def __init__(self, d_model, n_heads, n_kv_heads, d_ff, max_seq_len=2048, eps=1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(d_model, eps=eps)
        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, max_seq_len)
        self.ffn_norm = RMSNorm(d_model, eps=eps)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x, kv_cache=None, start_pos=0):
        attn_out, kv_cache = self.attn(self.attn_norm(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv_cache


if __name__ == "__main__":
    d_model = 8
    n_heads = 4
    n_kv_heads = 2
    d_ff = 32
    seq_len = 5
    batch = 2

    block = TransformerBlock(d_model, n_heads, n_kv_heads, d_ff)
    x = torch.randn(batch, seq_len, d_model)

    output, _ = block(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
