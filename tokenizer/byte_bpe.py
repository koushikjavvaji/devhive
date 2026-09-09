from collections import Counter


class ByteTokenizer:

    def __init__(self):
        self.merges = {}
        self.token_to_pair = {}

        self.end_token = 256
        self.comment_token = 257
        self.code_token = 258

        self.next_token = 259

        self.special_tokens = {
            self.end_token,
            self.comment_token,
            self.code_token,
        }

    def encode_bytes(self, text):
        return list(text.encode("utf-8"))

    def decode_bytes(self, tokens):
        return bytes(tokens).decode("utf-8")

    def get_pair_counts(self, tokens):
        pairs = Counter()

        for left, right in zip(tokens, tokens[1:]):
            if (
                left in self.special_tokens
                or right in self.special_tokens
            ):
                continue

            pairs[(left, right)] += 1

        return pairs

    def merge_pair(self, tokens, pair, new_token):
        merged = []
        i = 0

        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and pair[0] not in self.special_tokens
                and pair[1] not in self.special_tokens
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                merged.append(new_token)
                i += 2
            else:
                merged.append(tokens[i])
                i += 1

        return merged

    def encode_training_example(self, comment, code):
        tokens = []

        tokens.append(self.comment_token)
        tokens.extend(self.encode_bytes(comment))

        tokens.append(self.code_token)
        tokens.extend(self.encode_bytes(code))

        tokens.append(self.end_token)

        return tokens

    def train(self, examples, num_merges):
        tokens = []

        for comment, code in examples:
            example_tokens = self.encode_training_example(
                comment,
                code
            )
            tokens.extend(example_tokens)

        for merge_number in range(num_merges):
            pair_counts = self.get_pair_counts(tokens)

            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]

            new_token = self.next_token
            self.next_token += 1

            self.merges[best_pair] = new_token
            self.token_to_pair[new_token] = best_pair

            tokens = self.merge_pair(
                tokens,
                best_pair,
                new_token
            )

            print(
                f"Merge {merge_number + 1}: "
                f"{best_pair} -> {new_token} "
                f"(count={count})"
            )

        return tokens

    def encode(self, text):
        tokens = self.encode_bytes(text)

        for pair, new_token in self.merges.items():
            tokens = self.merge_pair(
                tokens,
                pair,
                new_token
            )

        return tokens

    def expand_token(self, token):
        if token < 256:
            return [token]

        if token in self.special_tokens:
            return [token]

        pair = self.token_to_pair[token]

        left = self.expand_token(pair[0])
        right = self.expand_token(pair[1])

        return left + right

    def decode(self, tokens):
        decoded_bytes = []

        for token in tokens:
            if token in self.special_tokens:
                continue

            decoded_bytes.extend(
                self.expand_token(token)
            )

        return bytes(decoded_bytes).decode("utf-8")


if __name__ == "__main__":

    tokenizer = ByteTokenizer()

    examples = [
        (
            "Say hello",
            "print('hello')"
        ),
        (
            "Say goodbye",
            "print('goodbye')"
        ),
        (
            "Add two numbers",
            "return a + b"
        )
    ]

    tokenizer.train(
        examples,
        num_merges=5
    )

    print()
    print("Learned merges:")
    print(tokenizer.merges)

    print()
    print("Token to pair:")
    print(tokenizer.token_to_pair)

    new_text = "hello world"

    encoded = tokenizer.encode(new_text)

    print()
    print("Encoded:")
    print(encoded)

    decoded = tokenizer.decode(encoded)

    print()
    print("Decoded:")
    print(decoded)