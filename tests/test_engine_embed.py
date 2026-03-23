"""Tests for pp_llm/engine_embed.py"""
from __future__ import annotations
import math
import sys

from pp_llm.engine_embed import EmbedEngine, _l2_normalize


def test_l2_normalize():
    result = _l2_normalize([3.0, 4.0])
    assert len(result) == 2
    assert abs(result[0] - 0.6) < 1e-9
    assert abs(result[1] - 0.8) < 1e-9
    # Verify unit norm
    norm = math.sqrt(sum(x * x for x in result))
    assert abs(norm - 1.0) < 1e-9


def test_l2_normalize_zero_vector():
    result = _l2_normalize([0.0, 0.0])
    assert result == [0.0, 0.0]


def test_encode_calls_model(monkeypatch):
    """Mock mlx_embeddings.utils.load, mock model/tokenizer, verify encode() calls them."""
    # Build a fake embedding tensor that supports indexing and .tolist()
    fake_vec = [0.1, 0.2, 0.9]

    class FakeEmbeddings:
        def __getitem__(self, idx):
            return self

        def tolist(self):
            return list(fake_vec)

    fake_outputs = FakeEmbeddings()

    class FakeModel:
        def __call__(self, input_ids, attention_mask=None):
            return fake_outputs

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2, 3]] * len(texts)}

    load_calls = []

    def mock_embed_load(path):
        load_calls.append(path)
        return FakeModel(), FakeTokenizer()

    # Patch the utils module
    embed_utils_mod = sys.modules["mlx_embeddings.utils"]
    embed_utils_mod.load = mock_embed_load

    engine = EmbedEngine()
    result = engine.encode("test/embed-model", ["hello world"])

    assert len(load_calls) == 1
    assert len(result) == 1
    assert isinstance(result[0], list)
    assert len(result[0]) == len(fake_vec)


def test_encode_returns_normalized(monkeypatch):
    """Verify that encode() with normalize=True returns unit-norm vectors."""
    raw_vec = [3.0, 4.0]  # norm = 5.0

    class FakeEmbeddings:
        def __getitem__(self, idx):
            return self

        def tolist(self):
            return list(raw_vec)

    class FakeModel:
        def __call__(self, input_ids, attention_mask=None):
            return FakeEmbeddings()

    class FakeTokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": [[1, 2]] * len(texts)}

    embed_utils_mod = sys.modules["mlx_embeddings.utils"]
    embed_utils_mod.load = lambda path: (FakeModel(), FakeTokenizer())

    engine = EmbedEngine()
    result = engine.encode("test/embed-model2", ["some text"], normalize=True)

    assert len(result) == 1
    vec = result[0]
    norm = math.sqrt(sum(x * x for x in vec))
    assert abs(norm - 1.0) < 1e-6
