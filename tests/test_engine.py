"""Tests for ppmlx.engine — all mlx_lm calls are mocked."""
from __future__ import annotations
import sys
import types
from unittest.mock import MagicMock

import pytest


def _make_fake_mlx_lm(generate_return="Hello!", stream_chunks=None):
    """Build a fake mlx_lm module with controllable return values."""
    fake = types.ModuleType("mlx_lm")
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template.return_value = "mocked prompt"
    mock_tokenizer.encode.return_value = [1, 2, 3]

    fake.load = MagicMock(return_value=(mock_model, mock_tokenizer))
    fake.generate = MagicMock(return_value=generate_return)

    if stream_chunks is None:
        stream_chunks = ["Hello", ", ", "world!"]

    def _stream_gen(model, tokenizer, **kwargs):
        for chunk in stream_chunks:
            obj = MagicMock()
            obj.text = chunk
            yield obj

    fake.stream_generate = MagicMock(side_effect=_stream_gen)
    return fake, mock_model, mock_tokenizer


def _install_fake(fake):
    """Install fake mlx_lm into sys.modules and return it."""
    sys.modules["mlx_lm"] = fake
    return fake


# ---------------------------------------------------------------------------
# 1. LRU eviction
# ---------------------------------------------------------------------------
def test_lru_eviction():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.load("model/a")
    engine.load("model/b")
    engine.load("model/c")  # should evict model/a

    loaded = engine.list_loaded()
    assert "model/a" not in loaded
    assert "model/b" in loaded
    assert "model/c" in loaded
    assert len(loaded) == 2


# ---------------------------------------------------------------------------
# 2. Load already-loaded model moves to end
# ---------------------------------------------------------------------------
def test_load_already_loaded_moves_to_end():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.load("model/a")
    engine.load("model/b")
    engine.load("model/a")  # touch A → B becomes LRU

    # Now loading model/c should evict B (LRU), not A
    engine.load("model/c")
    loaded = engine.list_loaded()
    assert "model/b" not in loaded
    assert "model/a" in loaded
    assert "model/c" in loaded


# ---------------------------------------------------------------------------
# 3. generate calls mlx
# ---------------------------------------------------------------------------
def test_generate_calls_mlx():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="Hello!")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    text, reasoning, pt, ct = engine.generate(
        "some/model", [{"role": "user", "content": "hi"}]
    )

    fake.generate.assert_called_once()
    call_kwargs = fake.generate.call_args
    assert call_kwargs is not None
    # model and tokenizer are positional args
    args, kwargs = call_kwargs
    assert args[0] is mock_model
    assert args[1] is mock_tokenizer


# ---------------------------------------------------------------------------
# 4. generate returns correct tuple structure
# ---------------------------------------------------------------------------
def test_generate_returns_text():
    fake, _, _ = _make_fake_mlx_lm(generate_return="Hello there!")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = engine.generate("some/model", [{"role": "user", "content": "hi"}])
    assert isinstance(result, tuple)
    assert len(result) == 4
    text, reasoning, pt, ct = result
    assert isinstance(text, str)
    assert reasoning is None
    assert isinstance(pt, int)
    assert isinstance(ct, int)


# ---------------------------------------------------------------------------
# 5. generate strips <think> blocks
# ---------------------------------------------------------------------------
def test_generate_strips_thinking():
    fake, _, _ = _make_fake_mlx_lm(generate_return="<think>reasoning here</think>answer text")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    text, reasoning, pt, ct = engine.generate(
        "some/model", [{"role": "user", "content": "hi"}], strip_thinking=True
    )
    assert text == "answer text"
    assert reasoning == "reasoning here"


# ---------------------------------------------------------------------------
# 6. stream_generate yields chunks
# ---------------------------------------------------------------------------
def test_stream_generate_yields_chunks():
    chunks = ["tok1", "tok2", "tok3"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = list(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert result == chunks


# ---------------------------------------------------------------------------
# 7. _strip_thinking helper
# ---------------------------------------------------------------------------
def test_strip_thinking_helper():
    from ppmlx.engine import _strip_thinking

    # Basic case
    text, reasoning = _strip_thinking("<think>inner reasoning</think>final answer")
    assert text == "final answer"
    assert reasoning == "inner reasoning"

    # Multiline thinking
    text, reasoning = _strip_thinking("<think>\nline1\nline2\n</think>response")
    assert text == "response"
    assert reasoning == "line1\nline2"

    # Multiple think blocks — only first matched by search
    text, reasoning = _strip_thinking("<think>r1</think>mid<think>r2</think>end")
    # sub removes all, search finds first
    assert "mid" in text or "end" in text
    assert reasoning == "r1"


# ---------------------------------------------------------------------------
# 8. No thinking tag — text unchanged
# ---------------------------------------------------------------------------
def test_no_thinking_unchanged():
    from ppmlx.engine import _strip_thinking

    original = "This is a plain response."
    text, reasoning = _strip_thinking(original)
    assert text == original
    assert reasoning is None


# ---------------------------------------------------------------------------
# 8b. Template-injected think block (no opening <think>)
# ---------------------------------------------------------------------------
def test_strip_thinking_no_opening_tag():
    """Qwen3 template injects <think> into the prompt, so the model output
    starts inside the think block — only </think> appears in the text."""
    from ppmlx.engine import _strip_thinking

    text, reasoning = _strip_thinking("reasoning content\n</think>\nactual answer")
    assert text == "actual answer"
    assert reasoning == "reasoning content"

    # Only closing tag, no reasoning
    text, reasoning = _strip_thinking("</think>answer only")
    assert text == "answer only"
    assert reasoning is None


# ---------------------------------------------------------------------------
# 9. unload
# ---------------------------------------------------------------------------
def test_unload():
    fake, _, _ = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.load("model/a")
    assert "model/a" in engine.list_loaded()

    result = engine.unload("model/a")
    assert result is True
    assert engine.list_loaded() == []

    # Unloading again returns False
    assert engine.unload("model/a") is False


# ---------------------------------------------------------------------------
# 10. unload_all
# ---------------------------------------------------------------------------
def test_unload_all():
    fake, _, _ = _make_fake_mlx_lm()
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=3)

    engine.load("model/a")
    engine.load("model/b")
    assert len(engine.list_loaded()) == 2

    engine.unload_all()
    assert engine.list_loaded() == []


# ---------------------------------------------------------------------------
# 11. list_loaded empty on fresh engine
# ---------------------------------------------------------------------------
def test_list_loaded_empty():
    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)
    assert engine.list_loaded() == []


# ---------------------------------------------------------------------------
# 12. Singleton get_engine / reset_engine
# ---------------------------------------------------------------------------
def test_singleton():
    from ppmlx.engine import get_engine, reset_engine

    reset_engine()
    e1 = get_engine()
    e2 = get_engine()
    assert e1 is e2

    reset_engine()
    e3 = get_engine()
    assert e3 is not e1


# ---------------------------------------------------------------------------
# 13. _apply_chat_template fallback (no apply_chat_template method)
# ---------------------------------------------------------------------------
def test_apply_chat_template_fallback():
    from ppmlx.engine import TextEngine, LoadedModel, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    # Tokenizer without apply_chat_template
    tokenizer = object()  # plain object has no apply_chat_template

    lm = LoadedModel(repo_id="test/model", model=MagicMock(), tokenizer=tokenizer)
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
    prompt = engine._apply_chat_template(lm, messages)

    assert "system: You are helpful." in prompt
    assert "user: Hello!" in prompt
    assert prompt.endswith("assistant:")


# ---------------------------------------------------------------------------
# 14. apply_chat_template called on tokenizer when available
# ---------------------------------------------------------------------------
def test_chat_template_applied():
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="response")
    mock_tokenizer.apply_chat_template.return_value = "TEMPLATED PROMPT"
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    messages = [{"role": "user", "content": "test"}]
    engine.generate("some/model", messages)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# ---------------------------------------------------------------------------
# 15. stream_generate strips <think> blocks by default
# ---------------------------------------------------------------------------
def test_stream_generate_strips_think_blocks():
    chunks = ["<think>", "reasoning", " here", "</think>", "answer", " text"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert "<think>" not in result
    assert "</think>" not in result
    assert "reasoning" not in result
    assert "answer text" == result


def test_stream_generate_strips_think_split_across_chunks():
    # Tags split across token boundaries
    chunks = ["<thi", "nk>", "internal", "</thi", "nk>", "visible"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert result == "visible"


def test_stream_generate_no_think_tags_passes_through():
    chunks = ["Hello", " ", "world"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}]
    ))
    assert result == "Hello world"


def test_stream_generate_strip_thinking_false():
    chunks = ["<think>", "reason", "</think>", "answer"]
    fake, _, _ = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    result = "".join(engine.stream_generate(
        "some/model", [{"role": "user", "content": "hi"}],
        strip_thinking=False,
    ))
    assert "<think>" in result
    assert "reason" in result


# ---------------------------------------------------------------------------
# 16. Speculative decoding: generate passes draft_model to mlx-lm
# ---------------------------------------------------------------------------
def test_generate_with_draft_model():
    """When draft_model is specified, both models are loaded and
    draft_model + num_draft_tokens are passed to mlx_lm.generate."""
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="fast answer")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=3)

    text, reasoning, pt, ct = engine.generate(
        "target/model",
        [{"role": "user", "content": "hi"}],
        draft_model="draft/model",
        num_draft_tokens=4,
    )

    # mlx_lm.generate should have been called with draft_model and num_draft_tokens
    call_kwargs = fake.generate.call_args
    assert call_kwargs is not None
    _, kwargs = call_kwargs
    assert "draft_model" in kwargs
    assert kwargs["num_draft_tokens"] == 4

    # Both target and draft should be loaded in the engine cache
    loaded = engine.list_loaded()
    assert "target/model" in loaded
    assert "draft/model" in loaded

    # Result should still come through correctly
    assert text == "fast answer"
    assert reasoning is None


# ---------------------------------------------------------------------------
# 17. Speculative decoding: stream_generate passes draft_model to mlx-lm
# ---------------------------------------------------------------------------
def test_stream_generate_with_draft_model():
    """When draft_model is specified, stream_generate passes it to mlx-lm."""
    chunks = ["fast", " ", "stream"]
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(stream_chunks=chunks)
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=3)

    result = list(engine.stream_generate(
        "target/model",
        [{"role": "user", "content": "hi"}],
        draft_model="draft/model",
        num_draft_tokens=3,
    ))

    # mlx_lm.stream_generate should have been called with draft_model
    call_kwargs = fake.stream_generate.call_args
    assert call_kwargs is not None
    _, kwargs = call_kwargs
    assert "draft_model" in kwargs
    assert kwargs["num_draft_tokens"] == 3

    assert result == chunks

    # Both models should be in the cache
    loaded = engine.list_loaded()
    assert "target/model" in loaded
    assert "draft/model" in loaded


# ---------------------------------------------------------------------------
# 18. No draft_model = normal generation (backward compatible)
# ---------------------------------------------------------------------------
def test_generate_without_draft_model_no_speculative_kwargs():
    """Without draft_model, mlx_lm.generate should NOT receive speculative kwargs."""
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="normal")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    engine.generate(
        "some/model",
        [{"role": "user", "content": "hi"}],
    )

    call_kwargs = fake.generate.call_args
    assert call_kwargs is not None
    _, kwargs = call_kwargs
    assert "draft_model" not in kwargs
    assert "num_draft_tokens" not in kwargs


# ---------------------------------------------------------------------------
# 19. Draft model uses LRU cache (same model object returned)
# ---------------------------------------------------------------------------
def test_draft_model_uses_lru_cache():
    """The draft model should be cached and reused across calls."""
    fake, mock_model, mock_tokenizer = _make_fake_mlx_lm(generate_return="ok")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=3)

    engine.generate(
        "target/model",
        [{"role": "user", "content": "call 1"}],
        draft_model="draft/model",
    )

    # mlx_lm.load was called twice (target + draft)
    assert fake.load.call_count == 2

    engine.generate(
        "target/model",
        [{"role": "user", "content": "call 2"}],
        draft_model="draft/model",
    )

    # On second call, both models should be cached — no additional loads
    assert fake.load.call_count == 2


# ---------------------------------------------------------------------------
# 20. Draft model participates in LRU eviction
# ---------------------------------------------------------------------------
def test_draft_model_lru_eviction():
    """With max_loaded=2, loading target + draft fills the cache; loading a
    third model evicts the LRU entry."""
    fake, _, _ = _make_fake_mlx_lm(generate_return="ok")
    _install_fake(fake)

    from ppmlx.engine import TextEngine, reset_engine
    reset_engine()
    engine = TextEngine(max_loaded=2)

    # Load target and draft (fills cache to 2)
    engine.generate(
        "target/model",
        [{"role": "user", "content": "hi"}],
        draft_model="draft/model",
    )
    assert len(engine.list_loaded()) == 2

    # Loading a third model should evict the LRU
    engine.load("other/model")
    loaded = engine.list_loaded()
    assert len(loaded) == 2
    assert "other/model" in loaded
    # target/model was loaded first and not touched after draft, so it might be LRU
    # depending on access pattern. The key point is the cache respects max_loaded.
    assert "target/model" not in loaded or "draft/model" not in loaded
