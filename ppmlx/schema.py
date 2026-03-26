from __future__ import annotations
import time
import uuid
from typing import Literal
from pydantic import BaseModel, Field


def make_request_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex[:12]


def make_completion_id() -> str:
    return "cmpl-" + uuid.uuid4().hex[:12]


def now_ts() -> int:
    return int(time.time())


# ── Content parts (for multimodal) ─────────────────────────────────────

class ImageURL(BaseModel):
    url: str  # https:// URL or data:image/...;base64,...


class ContentPart(BaseModel):
    type: Literal["text", "image_url"]
    text: str | None = None
    image_url: ImageURL | None = None


# ── Chat messages ───────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str | list[ContentPart] | None = None
    reasoning: str | None = None  # ppmlx extension: populated for <think> models


# ── Chat completion request ─────────────────────────────────────────────

class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    repetition_penalty: float | None = None  # ppmlx extension
    draft_model: str | None = None  # ppmlx extension: speculative decoding
    speculative_tokens: int | None = None  # ppmlx extension: draft tokens per step


# ── Chat completion response ────────────────────────────────────────────

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "tool_calls"] | None = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=make_request_id)
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=now_ts)
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


# ── Streaming chunk ─────────────────────────────────────────────────────

class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None
    reasoning: str | None = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str = Field(default_factory=make_request_id)
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=now_ts)
    model: str
    choices: list[ChatCompletionChunkChoice]


# ── Text completion (legacy) ────────────────────────────────────────────

class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


class CompletionChoice(BaseModel):
    text: str
    index: int = 0
    finish_reason: str | None = "stop"
    logprobs: None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=make_completion_id)
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=now_ts)
    model: str
    choices: list[CompletionChoice]
    usage: Usage


# ── Embeddings ──────────────────────────────────────────────────────────

class EmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: Literal["float", "base64"] = "float"


class EmbeddingData(BaseModel):
    object: Literal["embedding"] = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage


# ── Model list ──────────────────────────────────────────────────────────

class ModelInfo(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=now_ts)
    owned_by: str = "ppmlx"


class ModelListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelInfo]


# ── Errors ──────────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
