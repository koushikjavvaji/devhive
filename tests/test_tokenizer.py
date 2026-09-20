import numpy as np

from tokenizer.byte_bpe import ByteTokenizer


def test_encode_decode_round_trip_without_merges():
    tokenizer = ByteTokenizer()
    text = "hello world"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_train_then_round_trip():
    tokenizer = ByteTokenizer()
    examples = [
        ("Say hello", "print('hello')"),
        ("Say goodbye", "print('goodbye')"),
        ("Add two numbers", "return a + b"),
    ]
    tokenizer.train(examples, num_merges=20)

    for comment, code in examples:
        tokens = tokenizer.encode_example(comment, code)
        assert tokenizer.decode(tokens) == comment + code


def test_save_load_round_trip(tmp_path):
    tokenizer = ByteTokenizer()
    tokenizer.train([("Add two numbers", "return a + b")], num_merges=10)

    path = tmp_path / "vocab.json"
    tokenizer.save(path)
    loaded = ByteTokenizer.load(path)

    assert loaded.merges == tokenizer.merges
    assert loaded.vocab_size == tokenizer.vocab_size

    text = "return a + b"
    assert loaded.encode(text) == tokenizer.encode(text)


def test_initial_pair_counts_matches_naive_reference():
    """Regression test for the numpy-vectorized pair counting: it replaced a plain
    python loop for speed, so pin it against that loop's (obviously correct) output."""
    tokenizer = ByteTokenizer()
    rng = np.random.default_rng(0)
    # range covers 0-259 so special tokens (256-258) actually show up and exercise
    # the boundary-filtering logic, not just ordinary byte tokens
    tokens = rng.integers(0, 260, size=500).astype(np.int32)

    pair_counts, pair_positions = tokenizer._initial_pair_counts(tokens)

    naive_counts = {}
    naive_positions = {}
    for i in range(len(tokens) - 1):
        left, right = int(tokens[i]), int(tokens[i + 1])
        if left in tokenizer.special_tokens or right in tokenizer.special_tokens:
            continue
        pair = (left, right)
        naive_counts[pair] = naive_counts.get(pair, 0) + 1
        naive_positions.setdefault(pair, set()).add(i)

    assert dict(pair_counts) == naive_counts
    assert dict(pair_positions) == naive_positions


def test_merges_never_cross_special_tokens():
    tokenizer = ByteTokenizer()
    examples = [("a", "b")] * 20  # short/repetitive enough to tempt cross-boundary merges
    tokenizer.train(examples, num_merges=10)

    for left, right in tokenizer.merges:
        assert left not in tokenizer.special_tokens
        assert right not in tokenizer.special_tokens
