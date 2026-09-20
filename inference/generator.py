import torch
import torch.nn.functional as F

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.byte_bpe import ByteTokenizer

torch.serialization.add_safe_globals([GPTConfig])


class LocalGenerator:
    """Loads our from-scratch checkpoint + tokenizer once and generates code from a comment."""

    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = ByteTokenizer.load(tokenizer_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = GPT(checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.iter = checkpoint["iter"]
        self.val_loss = checkpoint["val_loss"]

    @torch.no_grad()
    def generate_code(self, comment, max_new_tokens=150, temperature=0.8, top_k=50):
        prompt_tokens = [self.tokenizer.comment_token]
        prompt_tokens.extend(self.tokenizer.encode(comment))
        prompt_tokens.append(self.tokenizer.code_token)

        tokens = list(prompt_tokens)
        max_seq_len = self.model.config.max_seq_len

        for _ in range(max_new_tokens):
            context = tokens[-max_seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=self.device)

            logits, _ = self.model(x)
            logits = logits[0, -1, :] / temperature

            if top_k is not None:
                top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < top_values[-1]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()

            if next_token == self.tokenizer.end_token:
                break

            tokens.append(next_token)

        code_tokens = tokens[len(prompt_tokens):]
        return self.tokenizer.decode(code_tokens)
