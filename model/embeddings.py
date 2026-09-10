import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, token_ids):
        return self.weight[token_ids]


if __name__ == "__main__":
    vocab_size = 5
    d_model = 3

    embedding = TokenEmbedding(vocab_size, d_model)

    tokens = torch.tensor([
        [1, 3, 0, 2],
        [4, 0, 2, 1],
    ])

    output = embedding(tokens)

    print("Input shape:", tokens.shape)
    print("Output shape:", output.shape)
    print("Output:\n", output)
