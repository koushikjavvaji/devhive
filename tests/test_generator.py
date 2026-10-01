import torch

from inference.generator import LocalGenerator
from model.config import GPTConfig
from model.gpt import GPT
from tokenizer.byte_bpe import ByteTokenizer


def make_generator(tmp_path, max_seq_len=32):
    tokenizer = ByteTokenizer()  # untrained: plain bytes + special tokens is enough here
    tokenizer.save(tmp_path / "vocab.json")

    config = GPTConfig(vocab_size=tokenizer.vocab_size, d_model=16, n_layers=1, n_heads=2,
                       n_kv_heads=1, d_ff=32, max_seq_len=max_seq_len)
    torch.save({"model": GPT(config).state_dict(), "config": config, "iter": 0, "val_loss": 0.0},
               tmp_path / "ckpt.pt")

    return LocalGenerator(tmp_path / "ckpt.pt", tmp_path / "vocab.json", device="cpu")


def test_long_prompt_keeps_the_comment_marker_and_still_generates(tmp_path):
    """Regression test for a real bug: a pasted traceback longer than the context window
    got its *front* chopped off — taking the <comment> marker with it — and filled the
    whole context, so the model returned an empty string for any realistic traceback."""
    generator = make_generator(tmp_path)
    generator._sample = lambda logits, temperature, top_k: ord("A")  # never emits <end>

    seen_prompts = []
    forward = generator.model.forward

    def recording_forward(x, **kwargs):
        if kwargs.get("start_pos") == 0:
            seen_prompts.append(x[0].tolist())
        return forward(x, **kwargs)

    generator.model.forward = recording_forward

    code = generator.generate_code("x" * 500, max_new_tokens=10)

    prompt = seen_prompts[0]
    assert prompt[0] == generator.tokenizer.comment_token
    assert prompt[-1] == generator.tokenizer.code_token
    assert code == "A" * 10
