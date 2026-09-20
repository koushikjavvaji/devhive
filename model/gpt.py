import torch
import torch.nn as nn
import torch.nn.functional as F

from model.block import TransformerBlock
from model.config import GPTConfig
from model.embeddings import TokenEmbedding
from model.norm import RMSNorm


class GPT(nn.Module):

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                config.d_model,
                config.n_heads,
                config.n_kv_heads,
                config.d_ff,
                config.max_seq_len,
                config.rms_eps,
            )
            for _ in range(config.n_layers)
        ])
        self.final_norm = RMSNorm(config.d_model, eps=config.rms_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # tie output projection to the input embedding
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear) and module is not self.lm_head:
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, token_ids, targets=None, kv_caches=None, start_pos=0):
        x = self.token_emb(token_ids)

        if kv_caches is None:
            kv_caches = [None] * len(self.blocks)

        new_kv_caches = []
        for block, kv_cache in zip(self.blocks, kv_caches):
            x, kv_cache = block(x, kv_cache=kv_cache, start_pos=start_pos)
            new_kv_caches.append(kv_cache)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss, new_kv_caches

    def num_params(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    config = GPTConfig(
        vocab_size=300,
        d_model=32,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=64,
        max_seq_len=64,
    )

    model = GPT(config)
    print(f"Params: {model.num_params():,}")

    batch, seq_len = 2, 10
    tokens = torch.randint(0, config.vocab_size, (batch, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch, seq_len))

    logits, loss, _ = model(tokens, targets)
    print("Logits shape:", logits.shape)
    print("Loss:", loss.item())
