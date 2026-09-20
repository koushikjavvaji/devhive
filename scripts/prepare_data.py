import numpy as np
from datasets import load_dataset

from tokenizer.byte_bpe import ByteTokenizer

NUM_EXAMPLES = 50_000
VAL_FRACTION = 0.01
TOKENIZER_PATH = "tokenizer/vocab.json"
OUT_DIR = "data"


tokenizer = ByteTokenizer.load(TOKENIZER_PATH)
print("Loaded tokenizer, vocab size:", tokenizer.vocab_size)

dataset = load_dataset(
    "sentence-transformers/codesearchnet",
    split=f"train[:{NUM_EXAMPLES}]"
)

tokens = []

for i, row in enumerate(dataset):
    tokens.extend(tokenizer.encode_example(row["comment"], row["code"]))

    if (i + 1) % 20_000 == 0:
        print(f"Encoded {i + 1}/{len(dataset)} examples, {len(tokens):,} tokens so far")

tokens = np.array(tokens, dtype=np.uint16)
print("Total tokens:", len(tokens))

n_val = int(len(tokens) * VAL_FRACTION)
train_tokens = tokens[:-n_val]
val_tokens = tokens[-n_val:]

train_tokens.tofile(f"{OUT_DIR}/train.bin")
val_tokens.tofile(f"{OUT_DIR}/val.bin")

print(f"Train tokens: {len(train_tokens):,}")
print(f"Val tokens: {len(val_tokens):,}")
