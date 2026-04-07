"""LRU prompt cache for KV-cache reuse across requests.

Stores prefilled KV-cache states keyed by tokenized prompt prefixes.
On subsequent requests with the same prefix, the cached KV-cache is
deep-copied and extended — skipping the expensive prefill phase.

This is the single biggest speedup for agent workflows where every
request shares the same system prompt.
"""
from __future__ import annotations

import copy
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("ppmlx.engine")


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    """An immutable KV-cache snapshot keyed by token sequence."""
    tokens: tuple[int, ...]  # prompt tokens that produced this cache
    cache: list[Any]         # KV-cache objects (one per transformer layer)


class PromptCacheStore:
    """Thread-safe LRU cache for prompt KV-cache states.

    Supports prefix matching: a cached sequence [A, B, C] can accelerate
    a new prompt [A, B, C, D, E] by reusing the KV-cache for tokens A-C
    and only prefilling D-E.
    """

    def __init__(self, max_entries: int = 4):
        self._max_entries = max_entries
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────

    def find_prefix(
        self, repo_id: str, tokens: list[int],
    ) -> tuple[list[Any] | None, list[int]]:
        """Find the longest cached prefix for the given prompt tokens.

        Returns ``(deep_copy_of_cache, remaining_tokens)`` on hit,
        or ``(None, tokens)`` on miss.

        For exact hits (all tokens cached), the cache is trimmed by 1
        so that ``generate_step`` always receives ≥1 token.
        """
        with self._lock:
            best_key: str | None = None
            best_len = 0
            token_tuple = tuple(tokens)

            for key, entry in self._entries.items():
                if not key.startswith(repo_id + ":"):
                    continue
                elen = len(entry.tokens)
                if elen > best_len and elen <= len(token_tuple):
                    if token_tuple[:elen] == entry.tokens:
                        best_key = key
                        best_len = elen

            if best_key is None:
                return None, tokens

            entry = self._entries[best_key]
            self._entries.move_to_end(best_key)

        # Deep-copy outside the lock (can be slow for large caches)
        cached = copy.deepcopy(entry.cache)
        remaining = tokens[best_len:]

        # Exact hit: generate_step requires ≥1 prompt token
        if not remaining:
            try:
                from mlx_lm.models.cache import (
                    can_trim_prompt_cache,
                    trim_prompt_cache,
                )
                if can_trim_prompt_cache(cached):
                    trim_prompt_cache(cached, 1)
                    remaining = [tokens[-1]]
                else:
                    # Rotating cache can't be trimmed — fall through to no-cache
                    return None, tokens
            except ImportError:
                return None, tokens

        log.info(
            "Prompt cache hit: %d/%d tokens cached, %d remaining",
            len(tokens) - len(remaining), len(tokens), len(remaining),
        )
        return cached, remaining

    def store(
        self, repo_id: str, tokens: list[int], cache: list[Any],
        *, _skip_copy: bool = False,
    ) -> None:
        """Store a prompt KV-cache snapshot.

        By default the cache is deep-copied so the caller can keep mutating
        their copy.  Pass ``_skip_copy=True`` when the caller will NOT use
        *cache* after this call (avoids a potentially expensive copy of
        hundreds of MB of KV tensors).
        """
        token_tuple = tuple(tokens)
        key = f"{repo_id}:{len(token_tuple)}:{hash(token_tuple)}"

        snapshot = cache if _skip_copy else copy.deepcopy(cache)

        with self._lock:
            self._entries[key] = _CacheEntry(tokens=token_tuple, cache=snapshot)
            self._entries.move_to_end(key)

            while len(self._entries) > self._max_entries:
                evicted_key, evicted = self._entries.popitem(last=False)
                log.debug("Prompt cache evicted: %s", evicted_key)

    def clear(self, repo_id: str | None = None) -> None:
        """Clear all entries, or only entries for a specific model."""
        with self._lock:
            if repo_id is None:
                self._entries.clear()
            else:
                keys = [k for k in self._entries if k.startswith(repo_id + ":")]
                for k in keys:
                    del self._entries[k]

    @property
    def is_enabled(self) -> bool:
        """Return True if prompt caching is active (limit > 0)."""
        return self._max_entries > 0

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        with self._lock:
            entries: list[dict[str, Any]] = []
            for key, entry in self._entries.items():
                repo_id = key.split(":")[0]
                entries.append({
                    "repo_id": repo_id,
                    "tokens": len(entry.tokens),
                })
            return {
                "entries": len(self._entries),
                "max_entries": self._max_entries,
                "models": entries,
            }
