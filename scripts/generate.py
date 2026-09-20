from inference.generator import LocalGenerator

CHECKPOINT_PATH = "checkpoints/ckpt.pt"
TOKENIZER_PATH = "tokenizer/vocab.json"


if __name__ == "__main__":
    generator = LocalGenerator(CHECKPOINT_PATH, TOKENIZER_PATH)
    print(f"Loaded checkpoint from iter {generator.iter} (val loss {generator.val_loss:.4f})")

    comment = "Add two numbers"
    code = generator.generate_code(comment)

    print("Comment:", comment)
    print("Generated code:")
    print(code)
