"""Tests for ppmlx/schema.py — OpenAI-compatible Pydantic v2 models."""
from __future__ import annotations
import json
import pytest
from pydantic import ValidationError

from ppmlx.schema import (
    ChatMessage,
    ContentPart,
    ImageURL,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    CompletionTokensDetails,
    DeltaMessage,
    Usage,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    EmbeddingRequest,
    EmbeddingData,
    EmbeddingUsage,
    EmbeddingResponse,
    ModelInfo,
    ModelListResponse,
    ErrorDetail,
    ErrorResponse,
    make_request_id,
    make_completion_id,
    now_ts,
)


def test_chat_message_string_content():
    msg = ChatMessage(role="user", content="Hello, world!")
    data = json.loads(msg.model_dump_json())
    assert data["role"] == "user"
    assert data["content"] == "Hello, world!"
    # Round-trip
    msg2 = ChatMessage.model_validate(data)
    assert msg2.role == "user"
    assert msg2.content == "Hello, world!"


def test_chat_message_multipart_content():
    part = ContentPart(type="text", text="hi")
    msg = ChatMessage(role="user", content=[part])
    data = json.loads(msg.model_dump_json())
    assert isinstance(data["content"], list)
    assert data["content"][0]["type"] == "text"
    assert data["content"][0]["text"] == "hi"


def test_image_url_content_part():
    img = ImageURL(url="https://example.com/image.png")
    part = ContentPart(type="image_url", image_url=img)
    assert part.type == "image_url"
    assert part.image_url is not None
    assert part.image_url.url == "https://example.com/image.png"
    assert part.text is None
    # Also test base64 data URL
    data_part = ContentPart(
        type="image_url",
        image_url=ImageURL(url="data:image/jpeg;base64,/9j/4AAQ=="),
    )
    assert data_part.image_url.url.startswith("data:image/")


def test_chat_completion_request_defaults():
    req = ChatCompletionRequest(
        model="llama3",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.stream is False
    assert req.temperature is None
    assert req.top_p is None
    assert req.max_tokens is None
    assert req.stop is None
    assert req.seed is None
    assert req.repetition_penalty is None


def test_chat_completion_request_with_all_fields():
    req = ChatCompletionRequest(
        model="llama3",
        messages=[ChatMessage(role="system", content="You are helpful."),
                  ChatMessage(role="user", content="Hello")],
        temperature=0.7,
        top_p=0.9,
        max_tokens=256,
        stream=True,
        stop=["<|end|>", "\n\n"],
        seed=42,
        repetition_penalty=1.1,
    )
    assert req.temperature == 0.7
    assert req.top_p == 0.9
    assert req.max_tokens == 256
    assert req.stream is True
    assert req.stop == ["<|end|>", "\n\n"]
    assert req.seed == 42
    assert req.repetition_penalty == 1.1


def test_chat_completion_response_structure():
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    msg = ChatMessage(role="assistant", content="Hello back!")
    choice = ChatCompletionChoice(index=0, message=msg)
    resp = ChatCompletionResponse(
        model="llama3",
        choices=[choice],
        usage=usage,
    )
    assert resp.object == "chat.completion"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "Hello back!"
    assert resp.usage.total_tokens == 30
    assert resp.id.startswith("chatcmpl-")
    assert isinstance(resp.created, int)


def test_chat_completion_chunk_streaming():
    # First chunk with role
    chunk1 = ChatCompletionChunk(
        model="llama3",
        choices=[ChatCompletionChunkChoice(
            delta=DeltaMessage(role="assistant", content=""),
            finish_reason=None,
        )],
    )
    assert chunk1.object == "chat.completion.chunk"
    assert chunk1.choices[0].delta.role == "assistant"
    assert chunk1.choices[0].finish_reason is None

    # Content chunk
    chunk2 = ChatCompletionChunk(
        model="llama3",
        choices=[ChatCompletionChunkChoice(
            delta=DeltaMessage(content="Hello"),
            finish_reason=None,
        )],
    )
    assert chunk2.choices[0].delta.content == "Hello"
    assert chunk2.choices[0].finish_reason is None

    # Final chunk
    chunk3 = ChatCompletionChunk(
        model="llama3",
        choices=[ChatCompletionChunkChoice(
            delta=DeltaMessage(),
            finish_reason="stop",
        )],
    )
    assert chunk3.choices[0].finish_reason == "stop"
    assert chunk3.choices[0].delta.content is None


def test_completion_request():
    req = CompletionRequest(model="llama3", prompt="Once upon a time")
    assert req.prompt == "Once upon a time"
    assert req.model == "llama3"
    assert req.stream is False
    assert req.max_tokens is None


def test_embedding_request_string_input():
    req = EmbeddingRequest(model="nomic-embed", input="Hello world")
    assert req.input == "Hello world"
    assert req.encoding_format == "float"


def test_embedding_request_list_input():
    req = EmbeddingRequest(model="nomic-embed", input=["Hello", "World"])
    assert isinstance(req.input, list)
    assert len(req.input) == 2
    assert req.input[0] == "Hello"


def test_embedding_response():
    data = EmbeddingData(embedding=[0.1, 0.2, 0.3], index=0)
    usage = EmbeddingUsage(prompt_tokens=3, total_tokens=3)
    resp = EmbeddingResponse(
        data=[data],
        model="nomic-embed",
        usage=usage,
    )
    assert resp.object == "list"
    assert isinstance(resp.data[0].embedding, list)
    assert resp.data[0].embedding[0] == pytest.approx(0.1)
    assert resp.data[0].object == "embedding"
    assert resp.data[0].index == 0


def test_model_list_response():
    models = [
        ModelInfo(id="llama3"),
        ModelInfo(id="mistral"),
    ]
    resp = ModelListResponse(data=models)
    assert resp.object == "list"
    assert len(resp.data) == 2
    assert resp.data[0].id == "llama3"
    assert resp.data[0].object == "model"
    assert resp.data[0].owned_by == "ppmlx"
    assert isinstance(resp.data[0].created, int)


def test_error_response():
    detail = ErrorDetail(
        message="Model not found",
        type="invalid_request_error",
        param="model",
        code="model_not_found",
    )
    resp = ErrorResponse(error=detail)
    assert resp.error.message == "Model not found"
    assert resp.error.type == "invalid_request_error"
    assert resp.error.param == "model"
    assert resp.error.code == "model_not_found"


def test_make_request_id():
    rid = make_request_id()
    assert rid.startswith("chatcmpl-")
    assert len(rid) > 10
    # Each call produces a unique id
    assert make_request_id() != make_request_id()


def test_usage_total_tokens():
    usage = Usage(prompt_tokens=15, completion_tokens=35, total_tokens=50)
    assert usage.total_tokens == usage.prompt_tokens + usage.completion_tokens


# ── Thinking model support ─────────────────────────────────────────────


def test_think_param_accepted():
    req = ChatCompletionRequest(
        model="llama3",
        messages=[ChatMessage(role="user", content="hi")],
        think=True,
    )
    assert req.think is True


def test_think_param_default_none():
    req = ChatCompletionRequest(
        model="llama3",
        messages=[ChatMessage(role="user", content="hi")],
    )
    assert req.think is None


def test_reasoning_budget_validation():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="llama3",
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_budget=0,
        )
    req = ChatCompletionRequest(
        model="llama3",
        messages=[ChatMessage(role="user", content="hi")],
        reasoning_budget=1000,
    )
    assert req.reasoning_budget == 1000


def test_reasoning_budget_max():
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            model="llama3",
            messages=[ChatMessage(role="user", content="hi")],
            reasoning_budget=131073,
        )


def test_usage_with_completion_tokens_details():
    details = CompletionTokensDetails(reasoning_tokens=50)
    usage = Usage(
        prompt_tokens=10,
        completion_tokens=100,
        total_tokens=110,
        completion_tokens_details=details,
    )
    data = json.loads(usage.model_dump_json())
    assert data["completion_tokens_details"]["reasoning_tokens"] == 50


def test_usage_without_completion_tokens_details():
    usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)
    assert usage.completion_tokens_details is None
    data = json.loads(usage.model_dump_json())
    assert data["completion_tokens_details"] is None


def test_completion_request_think_param():
    req = CompletionRequest(
        model="llama3",
        prompt="Once upon a time",
        think=True,
        reasoning_budget=500,
    )
    assert req.think is True
    assert req.reasoning_budget == 500
