import torch


def test_tokenizer_returns_real_ids(adapter):
    tokens, input_ids, _ = adapter.tokenize("Explain artificial intelligence")
    assert input_ids.shape == (1, len(tokens))
    assert all(0 <= t.id < adapter.metadata.vocab_size for t in tokens)
    assert tokens[0].position == 0
    assert len(tokens) >= 3


def test_tokenizer_special_tokens(adapter):
    tokens, _, _ = adapter.tokenize("Hello", use_chat_template=True)
    assert any(t.is_special for t in tokens)


def test_tokenizer_no_chat_template(adapter):
    tokens, _, _ = adapter.tokenize("Hello world", use_chat_template=False)
    assert not any(t.is_special for t in tokens)
    assert tokens[0].text.replace("Ġ", " ").strip().lower() == "hello"


def test_tokenizer_max_tokens(adapter):
    tokens, _, _ = adapter.tokenize("hello world " * 50, max_tokens=10)
    assert len(tokens) == 10


def test_decode_roundtrip(adapter):
    tokens, ids, _ = adapter.tokenize("The quick brown fox")
    text = adapter.decode(ids[0].tolist())
    assert "quick" in text
