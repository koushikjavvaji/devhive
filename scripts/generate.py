import torch
import torch.nn.functional as F

from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.byte_bpe import ByteTokenizer

torch.serialization.add_safe_globals([GPTConfig])

CHECKPOINT_PATH = "checkpoints/ckpt.pt"
TOKENIZER_PATH = "tokenizer/vocab.json"

MAX_NEW_TOKENS = 200
TEMPERATURE = 0.8
TOP_K = 50

device = "mps" if torch.backends.mps.is_available() else "cpu"

tokenizer = ByteTokenizer.load(TOKENIZER_PATH)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
model = GPT(checkpoint["config"]).to(device)
model.load_state_dict(checkpoint["model"])
model.eval()

print(f"Loaded checkpoint from iter {checkpoint['iter']} (val loss {checkpoint['val_loss']:.4f})")


@torch.no_grad()
def generate(prompt_tokens, max_new_tokens, temperature, top_k):
    tokens = list(prompt_tokens)
    max_seq_len = model.config.max_seq_len

    for _ in range(max_new_tokens):
        context = tokens[-max_seq_len:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        logits, _ = model(x)
        logits = logits[0, -1, :] / temperature

        if top_k is not None:
            top_values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < top_values[-1]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        if next_token == tokenizer.end_token:
            break

        tokens.append(next_token)

    return tokens


if __name__ == "__main__":
    comment = "Add two numbers"

    prompt_tokens = [tokenizer.comment_token]
    prompt_tokens.extend(tokenizer.encode(comment))
    prompt_tokens.append(tokenizer.code_token)

    output_tokens = generate(prompt_tokens, MAX_NEW_TOKENS, TEMPERATURE, TOP_K)
    code_tokens = output_tokens[len(prompt_tokens):]

    print("Comment:", comment)
    print("Generated code:")
    print(tokenizer.decode(code_tokens))
