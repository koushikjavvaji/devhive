import torch
import torch.nn.functional as F

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.byte_bpe import ByteTokenizer

torch.serialization.add_safe_globals([GPTConfig])


class LocalGenerator:
    """Loads our from-scratch checkpoint + tokenizer once and generates code from a comment.

    Uses a KV cache: the prompt is a single prefill forward pass, then each new token is a
    forward pass over just that one token (attending to the cached keys/values from every
    token before it) instead of re-forwarding the whole growing sequence every step."""

    def __init__(self, checkpoint_path, tokenizer_path, device=None):
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")

        self.tokenizer = ByteTokenizer.load(tokenizer_path)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = GPT(checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

        self.iter = checkpoint["iter"]
        self.val_loss = checkpoint["val_loss"]

    def _sample(self, logits, temperature, top_k):
        logits = logits / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).item()

    @torch.no_grad()
    def generate_code(self, comment, max_new_tokens=150, temperature=0.8, top_k=50):
        max_seq_len = self.model.config.max_seq_len

        # trim the comment itself, never the whole prompt: chopping the front off the
        # prompt drops the <comment> marker the model was trained to start from, and a
        # prompt that fills the context leaves no room to generate anything at all
        room_to_generate = min(max_new_tokens, max_seq_len // 2)
        comment_budget = max_seq_len - room_to_generate - 2  # minus <comment> and <code>
        comment_tokens = self.tokenizer.encode(comment)[:comment_budget]

        prompt_tokens = [self.tokenizer.comment_token, *comment_tokens, self.tokenizer.code_token]

        x = torch.tensor([prompt_tokens], dtype=torch.long, device=self.device)
        logits, _, kv_caches = self.model(x, start_pos=0)
        next_token = self._sample(logits[0, -1, :], temperature, top_k)

        generated = []
        pos = len(prompt_tokens)

        while next_token != self.tokenizer.end_token and len(generated) < max_new_tokens and pos < max_seq_len:
            generated.append(next_token)

            x = torch.tensor([[next_token]], dtype=torch.long, device=self.device)
            logits, _, kv_caches = self.model(x, start_pos=pos, kv_caches=kv_caches)
            next_token = self._sample(logits[0, -1, :], temperature, top_k)
            pos += 1

        return self.tokenizer.decode(generated)
