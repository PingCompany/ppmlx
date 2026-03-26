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
    last_used_at: float = field(default_factory=time.monotonic)


_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)

# Sanity cap: never auto-set max_tokens above this even on huge-context models.
_MAX_AUTO_TOKENS = 32_768


def _context_size(lm: "LoadedModel") -> int:
    """Return the model's context window size from its tokenizer."""
    tok = lm.tokenizer
    inner = getattr(tok, "tokenizer", tok)
    for obj in (tok, inner):
        val = getattr(obj, "model_max_length", None)
        if val and isinstance(val, int) and val < 10 ** 9:
            return val
    return 4096  # conservative fallback


def _strip_thinking(text: str) -> tuple[str, str | None]:
    """
    Strip <think>...</think> blocks from text.
    Returns (answer_text, reasoning_text | None).

    Handles two cases:
    1. Standard: ``<think>reasoning</think>answer``
    2. Template-injected: output starts *inside* a think block (Qwen3 adds
       ``<think>\\n`` to the generation prompt, so the model output begins with
       the reasoning directly followed by ``</think>``).
    """
    match = _THINK_PATTERN.search(text)
    if match:
        reasoning = match.group(1).strip()
        answer = _THINK_PATTERN.sub("", text).strip()
        return answer, reasoning
    # Fallback: no opening <think> but a closing </think> exists — the model
    # started inside a think block injected by the chat template.
    end_idx = text.find("</think>")
    if end_idx != -1:
        reasoning = text[:end_idx].strip()
        answer = text[end_idx + len("</think>"):].strip()
        return answer, (reasoning or None)
    return text, None


def _resolve_model_path(repo_id: str) -> str:
    """
    Resolve a repo_id to a local path if available, otherwise return the repo_id
    for direct HuggingFace loading.
    """
    try:
        from ppmlx.models import get_model_path
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

    _REAPER_INTERVAL = 30  # seconds between TTL reaper sweeps

    def __init__(self, max_loaded: int = 2, ttl_seconds: int = 0):
        self._max_loaded = max_loaded
        self._ttl_seconds = ttl_seconds
        self._models: OrderedDict[str, LoadedModel] = OrderedDict()
        self._lock = threading.Lock()
        self._reaper_stop = threading.Event()
        self._reaper_thread: threading.Thread | None = None
        if self._ttl_seconds > 0:
            self._start_reaper()

    def _start_reaper(self) -> None:
        """Start the background TTL reaper thread."""
        self._reaper_stop.clear()
        t = threading.Thread(target=self._ttl_reaper, daemon=True, name="ppmlx-ttl-reaper")
        t.start()
        self._reaper_thread = t

    def _ttl_reaper(self) -> None:
        """Background loop that unloads models idle longer than ttl_seconds."""
        while not self._reaper_stop.wait(timeout=self._REAPER_INTERVAL):
            self._reap_expired()

    def _reap_expired(self) -> list[str]:
        """Single sweep: unload models idle longer than ttl_seconds. Returns expired IDs."""
        if self._ttl_seconds <= 0:
            return []
        now = time.monotonic()
        expired: list[str] = []
        with self._lock:
            for repo_id, lm in list(self._models.items()):
                if (now - lm.last_used_at) >= self._ttl_seconds:
                    expired.append(repo_id)
            for repo_id in expired:
                del self._models[repo_id]
        for repo_id in expired:
            self._emit_event("unload", repo_id)
        return expired

    def stop_reaper(self) -> None:
        """Signal the reaper thread to stop and wait for it to finish."""
        self._reaper_stop.set()
        if self._reaper_thread is not None:
            self._reaper_thread.join(timeout=5)
            self._reaper_thread = None

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
                lm.last_used_at = time.monotonic()
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
            from ppmlx.db import get_db
            get_db().log_model_event(event=event, model_repo=repo_id)
        except Exception:
            pass

    def _apply_chat_template(
        self, lm: LoadedModel, messages: list[dict],
        enable_thinking: bool = True, tools: list[dict] | None = None,
    ) -> str:
        """Apply the model's chat template to produce a prompt string."""
        tokenizer = lm.tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if tools:
                kwargs["tools"] = tools
            if not enable_thinking:
                try:
                    return tokenizer.apply_chat_template(
                        messages, **kwargs, enable_thinking=False
                    )
                except TypeError:
                    pass  # model doesn't support the flag — fall through
            try:
                return tokenizer.apply_chat_template(messages, **kwargs)
            except TypeError:
                # Tokenizer doesn't support 'tools' kwarg — retry without it
                if tools and "tools" in kwargs:
                    del kwargs["tools"]
                    return tokenizer.apply_chat_template(messages, **kwargs)
                raise
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
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        strip_thinking: bool = True,
        enable_thinking: bool = True,
        tools: list[dict] | None = None,
    ) -> tuple[str, str | None, int, int]:
        """
        Generate a response.
        Returns (answer_text, reasoning_text | None, prompt_tokens, completion_tokens).

        reasoning_text is populated for <think>...</think> models.
        prompt_tokens and completion_tokens are estimates (token count from encode).
        enable_thinking=False suppresses the thinking phase for models that support it (e.g. Qwen3).
        max_tokens=None means 50% of the model's context window (capped at _MAX_AUTO_TOKENS).
        """
        from mlx_lm import generate as mlx_generate

        lm = self._get_or_load(repo_id)
        if max_tokens is None:
            max_tokens = min(_context_size(lm) // 2, _MAX_AUTO_TOKENS)
        prompt = self._apply_chat_template(lm, messages, enable_thinking=enable_thinking, tools=tools)

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            sampler = None
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "verbose": False,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
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
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        seed: int | None = None,
        repetition_penalty: float | None = None,
        enable_thinking: bool = True,
        strip_thinking: bool = True,
        tools: list[dict] | None = None,
    ) -> Iterator[str]:
        """
        Stream token-by-token generation.
        Yields text chunks. When strip_thinking=True (default), <think>...</think>
        blocks are silently consumed and not yielded.
        enable_thinking=False suppresses the thinking phase for models that support it (e.g. Qwen3).
        max_tokens=None means 50% of the model's context window (capped at _MAX_AUTO_TOKENS).
        """
        from mlx_lm import stream_generate as mlx_stream

        lm = self._get_or_load(repo_id)
        if max_tokens is None:
            max_tokens = min(_context_size(lm) // 2, _MAX_AUTO_TOKENS)
        prompt = self._apply_chat_template(lm, messages, enable_thinking=enable_thinking, tools=tools)

        try:
            from mlx_lm.sample_utils import make_sampler
            sampler = make_sampler(temp=temperature, top_p=top_p)
        except Exception:
            sampler = None
        kwargs: dict[str, Any] = {
            "prompt": prompt,
            "max_tokens": max_tokens,
        }
        if sampler is not None:
            kwargs["sampler"] = sampler
        if seed is not None:
            kwargs["seed"] = seed

        if not strip_thinking:
            for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
                if hasattr(response, "text"):
                    yield response.text
                elif isinstance(response, str):
                    yield response
            return

        # State machine to strip <think>...</think> blocks from streamed tokens.
        # Buffers partial tag matches and only yields text outside think blocks.
        # Qwen3's chat template injects "<think>\n" into the generation prompt,
        # so the model output begins *inside* a think block.
        #
        # State machine to strip <think>...</think> blocks from streamed tokens.
        # Buffers partial tag matches and only yields text outside think blocks.
        # Qwen3's chat template injects "<think>\n" into the generation prompt,
        # so the model output begins *inside* a think block.
        #
        # Models that inject <think> but never emit </think> (e.g. GLM-4.7-Flash)
        # are detected by a token budget: if we suppress more than
        # _THINK_PASSTHROUGH_TOKENS chars without finding </think>, we assume the
        # model doesn't use think-tag pairs and switch to pass-through mode.
        _THINK_PASSTHROUGH_TOKENS = 50  # ~50 tokens worth of chars before giving up

        inside_think = bool(re.search(r"<think>\s*$", prompt))
        buf = ""
        think_chars = 0  # chars suppressed while inside_think=True

        for response in mlx_stream(lm.model, lm.tokenizer, **kwargs):
            chunk = response.text if hasattr(response, "text") else response
            buf += chunk

            while buf:
                if inside_think:
                    # Look for </think> closing tag
                    end_idx = buf.find("</think>")
                    if end_idx != -1:
                        # Properly closed — discard thinking content
                        think_chars = 0
                        buf = buf[end_idx + len("</think>"):]
                        inside_think = False
                        continue
                    # Check if buf ends with a partial match for </think>
                    tag = "</think>"
                    keep = 0
                    for i in range(1, min(len(tag), len(buf)) + 1):
                        if buf.endswith(tag[:i]):
                            keep = i
                    suppressed = buf[:-keep] if keep else buf
                    think_chars += len(suppressed)
                    buf = buf[-keep:] if keep else ""

                    # If we've suppressed too much without a close tag, this
                    # model doesn't use proper think-tag pairs — yield as plain text.
                    if think_chars > _THINK_PASSTHROUGH_TOKENS:
                        inside_think = False
                        think_chars = 0
                        if suppressed:
                            yield suppressed
                        continue
                    break
                else:
                    # Look for <think> opening tag
                    start_idx = buf.find("<think>")
                    if start_idx != -1:
                        # Yield everything before the tag
                        if start_idx > 0:
                            yield buf[:start_idx]
                        buf = buf[start_idx + len("<think>"):]
                        inside_think = True
                        continue
                    # Check if buf ends with a partial match for "<think>"
                    tag = "<think>"
                    keep = 0
                    for i in range(1, min(len(tag), len(buf)) + 1):
                        if buf.endswith(tag[:i]):
                            keep = i
                    if keep:
                        safe = buf[:-keep]
                        if safe:
                            yield safe
                        buf = buf[-keep:]
                    else:
                        if buf:
                            yield buf
                        buf = ""
                    break

        # Flush any remaining buffer outside think blocks
        if buf and not inside_think:
            yield buf

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

    def list_loaded_info(self) -> list[dict]:
        """Return detailed info for each loaded model (for ps command)."""
        now = time.monotonic()
        with self._lock:
            result = []
            for repo_id, lm in self._models.items():
                idle_seconds = now - lm.last_used_at
                info: dict = {
                    "repo_id": repo_id,
                    "loaded_at": lm.loaded_at,
                    "last_used": lm.last_used,
                    "idle_seconds": round(idle_seconds, 1),
                }
                if self._ttl_seconds > 0:
                    remaining = max(0, self._ttl_seconds - idle_seconds)
                    info["ttl_remaining_seconds"] = round(remaining, 1)
                result.append(info)
            return result


_engine_instance: TextEngine | None = None
_engine_lock = threading.Lock()


def get_engine(max_loaded: int = 2, ttl_seconds: int = 0) -> TextEngine:
    """Return the module-level singleton TextEngine."""
    global _engine_instance
    if _engine_instance is None:
        with _engine_lock:
            if _engine_instance is None:
                _engine_instance = TextEngine(max_loaded=max_loaded, ttl_seconds=ttl_seconds)
    return _engine_instance


def reset_engine() -> None:
    """Reset singleton (for testing)."""
    global _engine_instance
    if _engine_instance is not None:
        _engine_instance.stop_reaper()
        _engine_instance.unload_all()
    _engine_instance = None
