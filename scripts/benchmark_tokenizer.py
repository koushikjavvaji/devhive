import time

from datasets import load_dataset

from tokenizer.byte_bpe import ByteTokenizer


dataset = load_dataset(
    "sentence-transformers/codesearchnet",
    split="train[:1000]"
)

examples = [
    (
        row["comment"],
        row["code"]
    )
    for row in dataset
]

tokenizer = ByteTokenizer()

start = time.perf_counter()

tokenizer.train(
    examples,
    num_merges=100
)

elapsed = time.perf_counter() - start

print()
print(f"Training time: {elapsed:.2f} seconds")
print(f"Learned merges: {len(tokenizer.merges)}")