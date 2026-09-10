import torch
import torch.nn as nn


class RoPE(nn.Module):

    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        i = torch.arange(0, d_model, 2).float()
        freqs = 1.0 / (10000 ** (i / d_model))
        positions = torch.arange(max_seq_len).float()
        angles = torch.outer(positions, freqs)
        self.register_buffer("cos", torch.cos(angles))
        self.register_buffer("sin", torch.sin(angles))

    def forward(self, x):
        seq_len = x.shape[-2]
        cos = self.cos[:seq_len]
        sin = self.sin[:seq_len]
        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd  = x_even * sin + x_odd * cos
        return torch.stack([x_rotated_even, x_rotated_odd], dim=-1).flatten(-2)


if __name__ == "__main__":
    d_model = 4
    seq_len = 5

    rope = RoPE(d_model)

    Q = torch.tensor([[
        [0.3, 0.7, 0.1, 0.9],
        [0.5, 0.2, 0.8, 0.4],
        [0.1, 0.6, 0.3, 0.7],
        [0.9, 0.3, 0.5, 0.2],
        [0.4, 0.8, 0.2, 0.6],
    ]])

    print("Q shape:", Q.shape)
    Q_rotated = rope(Q)
    print("Q rotated shape:", Q_rotated.shape)
    print("Q rotated:\n", Q_rotated)


