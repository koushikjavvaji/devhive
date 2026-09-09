from datasets import load_dataset

dataset = load_dataset("sentence-transformers/codesearchnet", split="train")

print(dataset[0])
print(f"Dataset size: {len(dataset)}")
print(dataset)