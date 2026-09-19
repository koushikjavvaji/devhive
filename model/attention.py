import torch
import torch.nn as nn
import torch.nn.functional as F

from model.rope import RoPE


class GroupedQueryAttention(nn.Module):

    def __init__(self, d_model, n_heads, n_kv_heads, max_seq_len=2048):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # share each kv head across n_rep query heads
        k = k.repeat_interleave(self.n_rep, dim=1)
        v = v.repeat_interleave(self.n_rep, dim=1)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        weights = F.softmax(scores, dim=-1)
        out = weights @ v

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out)


if __name__ == "__main__":
    d_model = 8
    n_heads = 4
    n_kv_heads = 2
    seq_len = 5
    batch = 2

    attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    x = torch.randn(batch, seq_len, d_model)

    output = attn(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
