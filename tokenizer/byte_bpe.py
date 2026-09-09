from collections import Counter, defaultdict
import heapq
import numpy as np


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

        tokens = np.array(tokens, dtype=np.int32)
        n = len(tokens)
        prev_arr = np.arange(-1, n - 1, dtype=np.int32)
        next_arr = np.arange(1, n + 1, dtype=np.int32)

        pair_counts = Counter()
        pair_positions = defaultdict(set)

        for i in range(n):
            j = next_arr[i]
            if j < n:
                if tokens[i] not in self.special_tokens and tokens[j] not in self.special_tokens:
                    pair = (tokens[i], tokens[j])
                    pair_counts[pair] += 1
                    pair_positions[pair].add(i)

        for merge_number in range(num_merges):
            if not pair_counts:
                break

            best_pair, count = pair_counts.most_common(1)[0]

            new_token = self.next_token
            self.next_token += 1

            self.merges[best_pair] = new_token
            self.token_to_pair[new_token] = best_pair

            positions = sorted(pair_positions.pop(best_pair))
            del pair_counts[best_pair]

            for i in positions:
                if tokens[i] != best_pair[0]:
                    continue
                j = next_arr[i]
                if j >= n or tokens[j] != best_pair[1]:
                    continue

                j_next = next_arr[j]

                p = prev_arr[i]
                if p >= 0 and tokens[p] not in self.special_tokens:
                    old_left = (tokens[p], best_pair[0])
                    pair_counts[old_left] -= 1
                    if pair_counts[old_left] == 0:
                        del pair_counts[old_left]
                        del pair_positions[old_left]
                    else:
                        pair_positions[old_left].discard(p)

                    new_left = (tokens[p], new_token)
                    pair_counts[new_left] += 1
                    pair_positions[new_left].add(p)

                if j_next < n and tokens[j_next] not in self.special_tokens:
                    old_right = (best_pair[1], tokens[j_next])
                    if old_right in pair_counts:
                        pair_counts[old_right] -= 1
                        if pair_counts[old_right] == 0:
                            del pair_counts[old_right]
                            del pair_positions[old_right]
                        else:
                            pair_positions[old_right].discard(j)

                    new_right = (new_token, tokens[j_next])
                    pair_counts[new_right] += 1
                    pair_positions[new_right].add(i)

                tokens[i] = new_token
                tokens[j] = -1

                next_arr[i] = j_next
                if j_next < n:
                    prev_arr[j_next] = i

            print(
                f"Merge {merge_number + 1}: "
                f"{best_pair} -> {new_token} "
                f"(count={count})"
            )

        return tokens[tokens != -1].tolist()

    def encode(self, text):
        tokens = self.encode_bytes(text)
        n = len(tokens)

        if not self.merges or n < 2:
            return tokens

        merge_rank = {pair: rank for rank, pair in enumerate(self.merges)}

        prev_arr = list(range(-1, n - 1))
        next_arr = list(range(1, n + 1))

        heap = []
        for i in range(n - 1):
            pair = (tokens[i], tokens[i + 1])
            if pair in merge_rank:
                heapq.heappush(heap, (merge_rank[pair], i))

        while heap:
            rank, i = heapq.heappop(heap)
            j = next_arr[i]

            if j >= n:
                continue

            pair = (tokens[i], tokens[j])

            if pair not in merge_rank or merge_rank[pair] != rank:
                continue

            new_token = self.merges[pair]
            j_next = next_arr[j]

            tokens[i] = new_token
            tokens[j] = -1

            next_arr[i] = j_next
            if j_next < n:
                prev_arr[j_next] = i

            p = prev_arr[i]
            if p >= 0:
                new_pair = (tokens[p], new_token)
                if new_pair in merge_rank:
                    heapq.heappush(heap, (merge_rank[new_pair], p))

            if j_next < n:
                new_pair = (new_token, tokens[j_next])
                if new_pair in merge_rank:
                    heapq.heappush(heap, (merge_rank[new_pair], i))

        return [t for t in tokens if t != -1]

    def expand_token(self, token):
        result = []
        stack = [token]

        while stack:
            t = stack.pop()

            if t < 256 or t in self.special_tokens:
                result.append(t)
            else:
                left, right = self.token_to_pair[t]
                stack.append(right)
                stack.append(left)

        return result

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