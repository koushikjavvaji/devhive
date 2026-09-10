import torch
import torch.nn as nn


class RMSNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.gamma


if __name__ == "__main__":
    d_model = 4
    norm = RMSNorm(d_model)

    x = torch.tensor([[2.0, 4.0, 4.0, 6.0],
                       [1.0, 2.0, 3.0, 4.0]])

    output = norm(x)
    print("Input shape:", x.shape)
    print("Output shape:", output.shape)
    print("Output:\n", output)

