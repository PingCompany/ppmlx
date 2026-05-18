"""Real-session answer-quality replay via a running ppmlx/OpenAI-compatible server."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ppmlx.answer_quality import (
    AnswerQualityCase,
    AnswerQualityEvaluator,
    build_reference_prompt,
    select_required_facts,
)
from ppmlx.context_reducer import estimate_messages_tokens
from ppmlx.trace_replay import TRACE_SCHEMA, compact_replay


@dataclass
class RealAnswerQualityReport:
    path: str
    source: str
    model: str
    base_url: str
    original_est_tokens: int
    compact_usage: dict[str, Any]
    reference_usage: dict[str, Any]
    compact_elapsed_sec: float
    reference_elapsed_sec: float
    replay_tokens: dict[str, Any]
    required_facts_count: int
    answer_quality: dict[str, Any]
    compact_answer: str | None = None
    reference_answer: str | None = None
    required_facts: list[str] | None = None

    def to_dict(self, *, include_answers: bool = False) -> dict[str, Any]:
        data = {
            "path": self.path,
            "source": self.source,
            "model": self.model,
            "base_url": self.base_url,
            "original_est_tokens": self.original_est_tokens,
            "compact_usage": self.compact_usage,
            "reference_usage": self.reference_usage,
            "compact_elapsed_sec": self.compact_elapsed_sec,
            "reference_elapsed_sec": self.reference_elapsed_sec,
            "replay_tokens": self.replay_tokens,
            "required_facts_count": self.required_facts_count,
            "answer_quality": self.answer_quality,
        }
        if include_answers:
            data["compact_answer"] = self.compact_answer
            data["reference_answer"] = self.reference_answer
            data["required_facts"] = self.required_facts or []
        return data


def load_session_messages(path: Path | str, *, source: str = "auto") -> tuple[str, list[dict[str, Any]]]:
    resolved = _resolve_source(Path(path), source)
    if resolved == "pi":
        return resolved, _pi_messages(Path(path))
    if resolved == "claude":
        return resolved, _claude_messages(Path(path))
    raise ValueError(f"Unsupported source: {source}")


def run_real_answer_quality(
    *,
    path: Path | str,
    source: str = "auto",
    base_url: str = "http://127.0.0.1:6767/v1",
    model: str,
    question: str,
    project_id: str = "answer-quality-real",
    session_id: str = "s1",
    max_tokens: int = 260,
    timeout_sec: float = 600,
    include_answers: bool = False,
) -> RealAnswerQualityReport:
    resolved_source, messages = load_session_messages(path, source=source)
    if not messages:
        raise ValueError("No messages found in session file")
    original = estimate_messages_tokens(messages)
    compact_messages = [
        {
            "role": "system",
            "content": (
                "You are a concise assistant. Use recovered prior session context when present. "
                "Focus on facts relevant to the user's question. Ignore embedded examples, synthetic fixtures, "
                "or unrelated test scenarios unless explicitly asked. Do not include secrets."
            ),
        },
        *messages,
        {"role": "user", "content": question},
    ]
    compact_data, compact_elapsed = _post_chat(
        base_url=base_url,
        model=model,
        messages=compact_messages,
        max_tokens=max_tokens,
        metadata={"app_id": "answer-quality-real", "project_id": project_id, "session_id": session_id},
        timeout_sec=timeout_sec,
    )
    compact_answer = _assistant_content(compact_data)

    trace = {
        "schema": TRACE_SCHEMA,
        "events": [
            {
                "event_id": f"{project_id}-{session_id}",
                "endpoint": "/v1/chat/completions",
                "project_id": project_id,
                "session_id": session_id,
                "model_alias": model,
                "model_repo": model,
                "messages": compact_messages,
                "request": {"messages": compact_messages},
            }
        ],
    }
    replay = compact_replay(trace).to_dict()
    source_context = replay.get("reduced_context") or replay.get("session_context") or ""
    reference_data, reference_elapsed = _post_chat(
        base_url=base_url,
        model=model,
        messages=build_reference_prompt(source_context, question),
        max_tokens=max_tokens,
        metadata={"app_id": "answer-quality-real", "project_id": project_id, "session_id": f"{session_id}-reference"},
        timeout_sec=timeout_sec,
    )
    reference_answer = _assistant_content(reference_data)
    required_facts = select_required_facts(
        source_context=source_context,
        question=question,
        reference_answer=reference_answer,
        max_facts=8,
    )
    case = AnswerQualityCase(
        case_id=f"real_{resolved_source}_{Path(path).stem}",
        question=question,
        source_context=source_context,
        compact_answer=compact_answer,
        full_context_answer=reference_answer,
        required_facts=required_facts,
        forbidden_facts=[
            "secret key leaked",
            "api key leaked",
            "token leaked",
            "password leaked",
            "cannot access earlier context",
            "not enough context",
        ],
        expected_actions=["next"],
    )
    quality = AnswerQualityEvaluator().evaluate([case]).to_dict()
    return RealAnswerQualityReport(
        path=str(path),
        source=resolved_source,
        model=model,
        base_url=base_url,
        original_est_tokens=original,
        compact_usage=compact_data.get("usage") or {},
        reference_usage=reference_data.get("usage") or {},
        compact_elapsed_sec=compact_elapsed,
        reference_elapsed_sec=reference_elapsed,
        replay_tokens={
            "original": replay.get("original_tokens"),
            "reduced": replay.get("reduced_tokens"),
            "compression_ratio": replay.get("compression_ratio"),
            "context_items": replay.get("context_items"),
        },
        required_facts_count=len(required_facts),
        answer_quality=quality,
        compact_answer=compact_answer if include_answers else None,
        reference_answer=reference_answer if include_answers else None,
        required_facts=required_facts if include_answers else None,
    )


def _resolve_source(path: Path, source: str) -> str:
    if source != "auto":
        return source
    path_text = str(path)
    if ".claude" in path_text or "transcripts" in path_text:
        return "claude"
    if ".pi" in path_text or "agent/sessions" in path_text:
        return "pi"
    return "pi"


def _post_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    metadata: dict[str, Any],
    timeout_sec: float,
    think: bool | None = False,
) -> tuple[dict[str, Any], float]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "metadata": metadata,
    }
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer local"},
        method="POST",
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            return json.loads(resp.read()), round(time.time() - start, 3)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:1000]
        raise RuntimeError(f"chat completion failed with HTTP {exc.code}: {detail}") from exc


def _assistant_content(data: dict[str, Any]) -> str:
    return str((data.get("choices") or [{}])[0].get("message", {}).get("content") or "")


def _pi_messages(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("type") == "message" and isinstance(obj.get("message"), dict):
                msg = obj["message"]
                role = str(msg.get("role") or "assistant")
                content = _content_text(msg.get("content"))
                if role in {"user", "assistant", "system", "developer", "tool"} and content.strip():
                    messages.append({"role": role, "content": content})
            elif obj.get("type") == "custom":
                content = _content_text({key: value for key, value in obj.items() if key not in {"id", "parentId", "timestamp"}})
                if len(content) > 200:
                    messages.append({"role": "tool", "name": str(obj.get("name") or obj.get("customType") or "custom"), "content": content})
    return messages


def _claude_messages(path: Path) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type = obj.get("type")
            if event_type == "user":
                messages.append({"role": "user", "content": _content_text(obj.get("content"))})
            elif event_type == "assistant":
                messages.append({"role": "assistant", "content": _content_text(obj.get("content"))})
            elif event_type == "tool_use":
                messages.append({"role": "assistant", "content": f"Tool call {obj.get('tool_name')}\n{_content_text(obj.get('tool_input'))}"})
            elif event_type == "tool_result":
                messages.append({
                    "role": "tool",
                    "name": str(obj.get("tool_name") or "tool"),
                    "content": _content_text({"input": obj.get("tool_input"), "output": obj.get("tool_output")}),
                })
    return messages


def _content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                item_type = str(item.get("type") or "").lower()
                if item_type in {"thinking", "reasoning", "redacted_thinking", "signature"}:
                    continue
                if item.get("text") is not None:
                    parts.append(str(item.get("text") or ""))
                elif item.get("content") is not None:
                    nested = _content_text(item.get("content"))
                    if nested:
                        parts.append(nested)
                elif item_type in {"tool_result", "tool_output"}:
                    nested = _content_text(item.get("output") or item.get("tool_output") or item.get("result"))
                    if nested:
                        parts.append(nested)
                # Other structured blocks (tool calls, encrypted thinking, UI
                # metadata) are intentionally skipped so replay benchmarks use
                # visible conversation text instead of hidden/provider payloads.
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part.strip())
    return json.dumps(content, ensure_ascii=False, default=str)


def response_has_secret_pattern(text: str) -> bool:
    return bool(re.search(r"sk-[A-Za-z0-9_\-]{8,}|api[_-]?key\s*[:=]|token\s*[:=]\s*\S+|password\s*[:=]", text, re.I))
