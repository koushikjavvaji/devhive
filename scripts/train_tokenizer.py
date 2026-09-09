from datasets import load_dataset

from tokenizer.byte_bpe import ByteTokenizer


dataset = load_dataset(
    "sentence-transformers/codesearchnet",
    split="train[:1000]"
)

examples = []

for row in dataset:
    examples.append(
        (
            row["comment"],
            row["code"]
        )
    )

print("Examples:", len(examples))

tokenizer = ByteTokenizer()

tokenizer.train(
    examples,
    num_merges=100
)

print("Learned merges:", len(tokenizer.merges))

test_text = dataset[0]["code"]

encoded = tokenizer.encode(test_text)
decoded = tokenizer.decode(encoded)

print()
print("Original:")
print(test_text)

print()
print("Encoded:")
print(encoded)

print()
print("Decoded:")
print(decoded)

print()
print("Match:", test_text == decoded)