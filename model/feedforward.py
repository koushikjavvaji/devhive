import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


if __name__ == "__main__":
    d_model = 8
    d_ff = 32

    ffn = SwiGLU(d_model, d_ff)
    x = torch.randn(2, 5, d_model)

    output = ffn(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
