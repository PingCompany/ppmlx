from __future__ import annotations
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class LoadedModel:
    """A model that has been loaded into memory."""
    repo_id: str
    model: Any          # mlx_lm model object
    tokenizer: Any      # mlx_lm tokenizer object
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _strip_thinking(text: str) -> tuple[str, str | None]:
    """
    Strip <think>...</think> blocks from text.
    Returns (answer_text, reasoning_text | None).
    reasoning_text is the content between <think> and </think>.
    """
    match = _THINK_PATTERN.search(text)
    if match:
        reasoning = match.group(1).strip()
        answer = _THINK_PATTERN.sub("", text).strip()
        return answer, reasoning
    return text, None


def _resolve_model_path(repo_id: str) -> str:
    """
    Resolve a repo_id to a local path if available, otherwise return the repo_id
    for direct HuggingFace loading.
    """
    try:
        from pp_llm.models import get_model_path
        local = get_model_path(repo_id)
        if local:
            return str(local)
    except ImportError:
        pass
    return repo_id


class TextEngine:
    """
    Wraps mlx-lm for text generation with LRU model caching.

    Thread-safe: uses a lock for model loading/unloading.
    Multiple concurrent requests for the same loaded model are fine.
    """

    def __init__(self, max_loaded: int = 2):
        self._max_loaded = max_loaded
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()

    def _load_impl(self, repo_id: str) -> LoadedModel:
        """Actually load a model using mlx_lm.load. Called under lock."""
        from mlx_lm import load as mlx_load
        path = _resolve_model_path(repo_id)
        model, tokenizer = mlx_load(path)
        return LoadedModel(repo_id=repo_id, model=model, tokenizer=tokenizer)

    def load(self, repo_id: str) -> LoadedModel:
        """
        Load a model into the LRU cache.
        Evicts least-recently-used model if cache is full.
        """
        evicted_ids: list[str] = []
        with self._lock:
            if repo_id in self._models:
                self._models.move_to_end(repo_id)
                lm = self._models[repo_id]
                lm.last_used = time.time()
                return lm

            # Evict LRU if at capacity
            while len(self._models) >= self._max_loaded:
                evicted_id, _ = self._models.popitem(last=False)
                evicted_ids.append(evicted_id)

            lm = self._load_impl(repo_id)
            self._models[repo_id] = lm

        for evicted_id in evicted_ids:
            self._emit_event("unload", evicted_id)
        self._emit_event("load", repo_id)
        return lm

    def _get_or_load(self, repo_id: str) -> LoadedModel:
        """Get from cache or load; moves to end of LRU."""
        return self.load(repo_id)

    def _emit_event(self, event: str, repo_id: str) -> None:
        """Log model lifecycle event to DB (best-effort)."""
        try:
            from pp_llm.db import get_db
            get_db().log_model_event(event=event, model_repo=repo_id)
        except Exception:
            pass

    def _apply_chat_template(self, lm: LoadedModel, messages: list[dict]) -> str:
        """Apply the model's chat template to produce a prompt string."""
        tokenizer = lm.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        # Fallback: simple concatenation
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            parts.append(f"{role}: {content}")
        parts.append("assistant:")
        return "\n".join(parts)

    def generate(
        self,
        repo_id: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        strip_thinking: bool = True,
    ) -> tuple[str, str | None, int, int]:
        """
        Generate a response.
        Returns (answer_text, reasoning_text | None, prompt_tokens, completion_tokens).

        reasoning_text is populated for <think>...</think> models.
        prompt_tokens and completion_tokens are estimates (token count from encode).
        """
        from mlx_lm import generate as mlx_generate

        lm = self._get_or_load(repo_id)
        prompt = self._apply_chat_template(lm, messages)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": False,
            "temp": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            kwargs["seed"] = seed

        text = mlx_generate(lm.model, lm.tokenizer, **kwargs)

        # Estimate token counts
        try:
            prompt_tokens = len(lm.tokenizer.encode(prompt))
            completion_tokens = len(lm.tokenizer.encode(text))
        except Exception:
            prompt_tokens = len(prompt.split())
            completion_tokens = len(text.split())

        if strip_thinking:
            text, reasoning = _strip_thinking(text)
        else:
            reasoning = None

        return text, reasoning, prompt_tokens, completion_tokens

    def stream_generate(
        self,
        repo_id: str,
        messages: list[dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048,
        stop: list[str] | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[str]:
        """
        Stream token-by-token generation.
        Yields text chunks. Handles <think> tags by yielding them transparently
        (caller can strip if needed).
        """
        from mlx_lm import stream_generate as mlx_stream

        lm = self._get_or_load(repo_id)
        prompt = self._apply_chat_template(lm, messages)

        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temp": temperature,
            "top_p": top_p,
        }
        if seed is not None:
            kwargs["seed"] = seed

        for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
            if hasattr(response, "text"):
                yield response.text
            elif isinstance(response, str):
                yield response

    def unload(self, repo_id: str) -> bool:
        """Unload a specific model from cache. Returns True if it was loaded."""
        with self._lock:
            if repo_id not in self._models:
                return False
            del self._models[repo_id]
        self._emit_event("unload", repo_id)
        return True

    def unload_all(self) -> None:
        """Unload all models from cache."""
        with self._lock:
            repo_ids = list(self._models.keys())
            self._models.clear()
        for repo_id in repo_ids:
            self._emit_event("unload", repo_id)

    def list_loaded(self) -> list[str]:
        """Return list of currently loaded model repo IDs."""
        with self._lock:
            return list(self._models.keys())


_engine_instance: TextEngine | None = None
_engine_lock = threading.Lock()


def get_engine(max_loaded: int = 2) -> TextEngine:
    """Return the module-level singleton TextEngine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = TextEngine(max_loaded=max_loaded)
    return _engine_instance


def reset_engine() -> None:
    """Reset singleton (for testing)."""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.unload_all()
    _engine_instance = None
