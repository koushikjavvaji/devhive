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
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

        self.rope = RoPE(self.head_dim, max_seq_len)

    def forward(self, x, kv_cache=None, start_pos=0):
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, start_pos)
        k = self.rope(k, start_pos)

        if kv_cache is not None:
            # decode-time: only the new token(s) were projected above, so glue
            # the previously cached keys/values back on before attending
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)

        # fused kernel handles the kv-head grouping internally, so k/v stay
        # at n_kv_heads and we keep the memory savings GQA is for.
        #
        # is_causal=True only works when q and k are the same length (prefill).
        # During decode, q is just the new token(s) but k also holds the whole
        # cache, so is_causal=True would top-left-align the mask and only let
        # the new token see the very first cached position. Build the mask
        # explicitly instead: query i (0-indexed among the new tokens) may
        # attend to key j iff j <= i + (k_len - q_len), i.e. bottom-right-aligned.
        q_len, k_len = q.shape[-2], k.shape[-2]
        if q_len == k_len:
            attn_mask, is_causal = None, True
        else:
            attn_mask = torch.tril(
                torch.ones(q_len, k_len, dtype=torch.bool, device=q.device),
                diagonal=k_len - q_len,
            )
            is_causal = False

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            is_causal=is_causal,
            enable_gqa=(self.n_kv_heads != self.n_heads),
        )

        out = out.transpose(1, 2).contiguous().view(batch, seq_len, self.n_heads * self.head_dim)
        return self.out_proj(out), (k, v)


if __name__ == "__main__":
    d_model = 8
    n_heads = 4
    n_kv_heads = 2
    seq_len = 5
    batch = 2

    attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    x = torch.randn(batch, seq_len, d_model)

    output, _ = attn(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
