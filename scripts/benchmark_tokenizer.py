import time

from datasets import load_dataset
from tokenizer.byte_bpe import ByteTokenizer


start = time.perf_counter()

dataset = load_dataset(
    "sentence-transformers/codesearchnet",
    split="train[:50000]"
)

examples = [
    (
        row["comment"],
        row["code"]
    )
    for row in dataset
]

load_time = time.perf_counter() - start

print(f"Dataset loading/preparation time: {load_time:.2f} seconds")
print(f"Examples: {len(examples)}")

tokenizer = ByteTokenizer()

start = time.perf_counter()

tokenizer.train(
    examples,
    num_merges=1000
)

train_time = time.perf_counter() - start

print(f"Tokenizer training time: {train_time:.2f} seconds")
print(f"Learned merges: {len(tokenizer.merges)}")