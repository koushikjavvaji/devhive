from datasets import load_dataset

from tokenizer.byte_bpe import ByteTokenizer

NUM_EXAMPLES = 50_000
NUM_MERGES = 2_000  # -> vocab_size = 259 + NUM_MERGES = 2259
OUT_PATH = "tokenizer/vocab.json"


dataset = load_dataset(
    "sentence-transformers/codesearchnet",
    split=f"train[:{NUM_EXAMPLES}]"
)

examples = [
    (row["comment"], row["code"])
    for row in dataset
]

print("Examples:", len(examples))

tokenizer = ByteTokenizer()

tokenizer.train(
    examples,
    num_merges=NUM_MERGES
)

tokenizer.save(OUT_PATH)

print("Learned merges:", len(tokenizer.merges))
print("Vocab size:", tokenizer.vocab_size)
print("Saved to:", OUT_PATH)

test_text = dataset[0]["code"]
encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)
print("Round-trip match:", test_text == decoded)
