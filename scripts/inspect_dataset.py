from datasets import load_dataset


dataset = load_dataset(
    "sentence-transformers/codesearchnet",
    split="train"
)

print(dataset)
print()

for i in range(5):
    print("----- Example", i, "-----")
    print("Comment:")
    print(dataset[i]["comment"])
    print()
    print("Code:")
    print(dataset[i]["code"])
    print()