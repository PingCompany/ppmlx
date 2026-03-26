from __future__ import annotations
import asyncio
import json
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager

log = logging.getLogger("ppmlx.server")

from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ppmlx import __version__

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    try:
        from ppmlx.db import get_db
        db = get_db()
        db.init()
        app.state.db = db
    except ImportError:
        app.state.db = None

    try:
        from ppmlx.config import load_config
        cfg = load_config()
        interval = cfg.logging.snapshot_interval_seconds
    except ImportError:
        interval = 60

    snapshot_task = asyncio.create_task(_snapshot_loop(interval))

    yield

    snapshot_task.cancel()
    try:
        await snapshot_task
    except asyncio.CancelledError:
        pass

    try:
        if app.state.db:
            app.state.db.flush()
            app.state.db.close()
    except Exception:
        pass


app = FastAPI(
    title="ppmlx",
    version=__version__,
    description="OpenAI-compatible LLM API for Apple Silicon via MLX",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _snapshot_loop(interval_seconds: int) -> None:
    """Periodically log system snapshots to the database."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            from ppmlx.memory import get_system_ram_gb
            from ppmlx.db import get_db
            from ppmlx.engine import get_engine
            ram_total = get_system_ram_gb()
            loaded = get_engine().list_loaded()
            uptime = int(time.time() - _start_time)
            get_db().log_system_snapshot(
                memory_total_gb=ram_total,
                memory_used_gb=0.0,  # hard to get used memory without psutil
                loaded_models=loaded,
                uptime_seconds=uptime,
            )
        except Exception:
            pass


def _route_engine(repo_id: str, has_images: bool) -> str:
    """Determine which engine to use: 'text', 'vision', or 'embed'."""
    try:
        from ppmlx.models import is_vision_model, is_embed_model
        if is_embed_model(repo_id):
            return "embed"
        if has_images and is_vision_model(repo_id):
            return "vision"
    except ImportError:
        pass
    return "text"


def _has_images(messages: list[dict]) -> bool:
    """Check if any message contains an image_url content part."""
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "image_url":
                    return True
    return False


def _log_request(_, **kwargs) -> None:
    """Log a request to the DB (best-effort)."""
    try:
        from ppmlx.db import get_db
        get_db().log_request(**kwargs)
    except Exception:
        pass


def _merge_system_messages(messages: list[dict]) -> list[dict]:
    """Merge all system messages into a single one at the start.

    Many models (e.g. Qwen) only accept one system message and require it
    to be the first message.  When clients like Codex send both an
    ``instructions`` system prompt *and* developer-role messages (also
    mapped to system), we merge them so the model template doesn't error.
    """
    system_parts: list[str] = []
    other: list[dict] = []
    for msg in messages:
        if msg.get("role") == "system":
            c = msg.get("content", "")
            if c:
                system_parts.append(c)
        else:
            other.append(msg)
    if not system_parts:
        return messages
    merged = [{"role": "system", "content": "\n\n".join(system_parts)}]
    merged.extend(other)
    return merged


# ── Tool-call parsing ───────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

# JSON format: <tool_call>{"name": "fn", "arguments": {...}}</tool_call>
_TC_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# XML-like format: <function=name>\n<parameter=key>\nvalue\n</parameter>\n</function>
_TC_XML_FUNC_RE = re.compile(
    r"<function=(\w+)>(.*?)</function>", re.DOTALL
)
_TC_XML_PARAM_RE = re.compile(
    r"<parameter=(\w+)>(.*?)</parameter>", re.DOTALL
)


def _parse_tool_call_body(body: str) -> dict | None:
    """Parse the content inside a ``<tool_call>`` block.

    Supports two formats:
    1. JSON: ``{"name": "fn", "arguments": {"key": "val"}}``
    2. XML-like (Qwen3.5): ``<function=fn><parameter=k>v</parameter></function>``
    """
    body = body.strip()

    # Try JSON first
    jm = _TC_JSON_RE.search(body)
    if jm:
        try:
            data = json.loads(jm.group(0))
            name = data.get("name", "")
            args = data.get("arguments", data.get("parameters", {}))
            if isinstance(args, dict):
                args = json.dumps(args)
            elif not isinstance(args, str):
                args = json.dumps(args)
            return {"name": name, "arguments": args}
        except (json.JSONDecodeError, ValueError):
            pass

    # Try XML-like format
    fm = _TC_XML_FUNC_RE.search(body)
    if fm:
        name = fm.group(1)
        func_body = fm.group(2)
        params = {}
        for pm in _TC_XML_PARAM_RE.finditer(func_body):
            params[pm.group(1)] = pm.group(2).strip()
        return {"name": name, "arguments": json.dumps(params)}

    return None


# Core tools that Codex/Claude Code always need — everything else is optional.
_CORE_TOOL_NAMES = {
    "exec_command", "apply_patch", "write_stdin", "update_plan",
    "request_user_input", "view_image",
    # Anthropic / Claude Code tools
    "bash", "read", "edit", "write", "computer",
}

# Maximum estimated tokens for all tools combined.  Large tool lists
# (Codex sends 66) cause extremely slow prefill on local models
# (e.g. 50s+ for ~20k tool tokens on a 9b model), which makes
# clients timeout.  Keeping the budget small ensures fast first-token.
_MAX_TOOLS_TOKENS = 3000


def _limit_tools(tools: list[dict] | None) -> list[dict] | None:
    """Trim the tools list to keep prompt prefill fast on local models.

    Prioritises core coding tools and drops MCP / agent tools first.
    """
    if not tools:
        return tools
    # Estimate tokens per tool (~4 chars per token)
    total = sum(len(json.dumps(t)) for t in tools) // 4
    if total <= _MAX_TOOLS_TOKENS:
        return tools

    # Filter out non-function tools (e.g. web_search with name=None)
    # and split into core vs non-core
    core = []
    extra = []
    for t in tools:
        name = t.get("name") or t.get("function", {}).get("name", "")
        if not name:
            continue  # skip tools without a name
        if name in _CORE_TOOL_NAMES:
            core.append(t)
        else:
            extra.append(t)

    # Start with core, add extras until budget reached
    result = list(core)
    budget = _MAX_TOOLS_TOKENS - sum(len(json.dumps(t)) for t in result) // 4
    for t in extra:
        cost = len(json.dumps(t)) // 4
        if cost <= budget:
            result.append(t)
            budget -= cost
        if budget <= 0:
            break

    log.info("_limit_tools: %d → %d tools (est %d → %d tokens)",
             len(tools), len(result), total,
             sum(len(json.dumps(t)) for t in result) // 4)
    return result


def _parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Extract ``<tool_call>`` blocks from model output.

    Returns *(remaining_text, tool_calls)* where each tool call is
    ``{"name": "...", "arguments": "..."}`` (arguments as a JSON string).
    """
    calls: list[dict] = []
    for m in _TOOL_CALL_RE.finditer(text):
        tc = _parse_tool_call_body(m.group(1))
        if tc:
            calls.append(tc)
    remaining = _TOOL_CALL_RE.sub("", text).strip()
    return remaining, calls


def _normalize_tool_messages(messages: list[dict]) -> list[dict]:
    """Convert OpenAI tool_calls message format to plain-text format.

    Qwen's chat template doesn't understand OpenAI's ``tool_calls`` list
    structure on assistant messages.  It expects the tool call to appear as
    ``<tool_call>`` JSON blocks in the content string.  Similarly, ``tool``
    role messages are not understood — they must be re-wrapped.

    This function converts:
    - assistant messages with ``tool_calls`` → content with ``<tool_call>`` blocks
    - ``tool`` role messages → keep as-is (Qwen template maps them to
      ``<tool_response>`` via the ``name`` field)
    """
    out: list[dict] = []
    for msg in messages:
        role = msg.get("role", "")
        tc_list = msg.get("tool_calls")

        if role == "assistant" and tc_list:
            # Convert tool_calls list to <tool_call> text blocks
            parts: list[str] = []
            content = msg.get("content") or ""
            if content:
                parts.append(content)
            for tc in tc_list:
                fn = tc.get("function", tc)
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args_obj = json.loads(args)
                    except (json.JSONDecodeError, ValueError):
                        args_obj = args
                else:
                    args_obj = args
                parts.append(
                    f'<tool_call>\n{{"name": "{name}", "arguments": {json.dumps(args_obj)}}}\n</tool_call>'
                )
            out.append({"role": "assistant", "content": "\n".join(parts)})
        else:
            out.append(msg)
    return out


def _speculative_kwargs(
    draft_model: str | None, speculative_tokens: int | None,
) -> dict:
    """Build keyword arguments for speculative decoding (empty if disabled)."""
    if draft_model is None:
        return {}
    kw: dict = {"draft_model": draft_model}
    if speculative_tokens is not None:
        kw["num_draft_tokens"] = speculative_tokens
    return kw


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    try:
        from ppmlx.memory import get_system_ram_gb
        ram_gb = get_system_ram_gb()
    except Exception:
        ram_gb = 0.0

    registry_info = {}
    try:
        from ppmlx.registry import registry_meta
        registry_info = registry_meta()
    except Exception:
        pass

    return {
        "status": "ok",
        "version": __version__,
        "loaded_models": loaded,
        "uptime_seconds": int(time.time() - _start_time),
        "system": {
            "memory_total_gb": round(ram_gb, 1),
        },
        "registry": registry_info,
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint — returns JSON stats from the DB."""
    try:
        from ppmlx.db import get_db
        stats = get_db().get_stats()
    except Exception:
        stats = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}

    try:
        from ppmlx.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    return {**stats, "loaded_models": loaded}


def _model_metadata(model_id: str) -> dict:
    """Build rich model metadata for Codex / OpenAI-compatible clients."""
    meta = {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "ppmlx",
        "slug": model_id,
    }
    # Try to pull extra info from registry
    try:
        from ppmlx.registry import registry_lookup
        entry = registry_lookup(model_id)
        if entry:
            meta["owned_by"] = entry.get("lab", "ppmlx")
    except Exception:
        entry = None
    return meta


@app.get("/v1/models")
async def list_models():
    """List available models (local + aliases)."""
    try:
        from ppmlx.models import list_local_models, all_aliases
        local = list_local_models()
        aliases = all_aliases()

        data = []
        seen = set()

        for m in local:
            mid = m.get("alias") or m.get("repo_id", "unknown")
            if mid not in seen:
                seen.add(mid)
                data.append(_model_metadata(mid))

        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                data.append(_model_metadata(alias))

        return {"object": "list", "data": data, "models": data}
    except ImportError:
        return {"object": "list", "data": [], "models": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()

    # Append request summary for debugging (best-effort)
    try:
        import pathlib
        _cc_log = pathlib.Path("/tmp/ppmlx_chatcompletions_debug.jsonl")
        log_entry = {
            "ts": time.time(),
            "model": body.get("model"),
            "stream": body.get("stream"),
            "tools": len(body.get("tools") or []),
            "tool_names": [t.get("function", {}).get("name") or t.get("name")
                           for t in (body.get("tools") or [])],
            "messages_count": len(body.get("messages", [])),
        }
        with open(_cc_log, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    model_name = body.get("model", "")
    messages = body.get("messages", [])
    # Normalize messages for model compatibility
    tools = _limit_tools(body.get("tools") or None)
    for msg in messages:
        if msg.get("role") == "developer":

            msg["role"] = "system"
        # Ensure content is never None (breaks some chat templates)
        if msg.get("content") is None and not msg.get("tool_calls"):
            msg["content"] = ""
    if tools:
        messages = _normalize_tool_messages(messages)
    messages = _merge_system_messages(messages)
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = body.get("max_tokens")
    stop = body.get("stop")
    seed = body.get("seed")
    repetition_penalty = body.get("repetition_penalty")

    # ppmlx extension: speculative decoding via draft model
    draft_model = body.get("draft_model")
    speculative_tokens = body.get("speculative_tokens")

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    # Resolve draft model alias (reuses same resolve_alias)
    draft_repo_id: str | None = None
    if draft_model:
        try:
            draft_repo_id = resolve_alias(draft_model)
        except Exception:
            draft_repo_id = draft_model

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    request_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    start_ts = time.time()
    created = int(start_ts)

    if stream:
        return _stream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts, tools,
            draft_model=draft_repo_id,
            speculative_tokens=speculative_tokens,
        )
    else:
        return await _nonstream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts, tools,
            draft_model=draft_repo_id,
            speculative_tokens=speculative_tokens,
        )


def _stream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts, tools=None,
    draft_model=None, speculative_tokens=None,
):
    """Return streaming SSE response."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        first_token_ts = None
        is_first_chunk = True
        full_text = ""
        try:
            if engine_type == "text":
                from ppmlx.engine import get_engine
                engine = get_engine()
                gen = engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=max_tokens,
                    seed=seed,
                    tools=tools,
                    **_speculative_kwargs(draft_model, speculative_tokens),
                )
                async for chunk in _async_iter_sync_gen(gen):
                    full_text += chunk
                    if first_token_ts is None:
                        first_token_ts = time.time()
                    if is_first_chunk:
                        delta = {"role": "assistant", "content": chunk}
                        is_first_chunk = False
                    else:
                        delta = {"content": chunk}
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            elif engine_type == "vision":
                from ppmlx.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(repo_id, messages, max_tokens=max_tokens or 1024)
                full_text = text
                if first_token_ts is None:
                    first_token_ts = time.time()
                delta = {"role": "assistant", "content": text} if is_first_chunk else {"content": text}
                is_first_chunk = False
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            err = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n"

        # Parse tool calls if tools were provided
        _, tool_calls = _parse_tool_calls(full_text) if tools else ("", [])

        if tool_calls:
            # Emit tool_calls in streaming format
            tc_list = []
            for i, tc in enumerate(tool_calls):
                call_id = "call_" + uuid.uuid4().hex[:24]
                tc_list.append({
                    "index": i,
                    "id": call_id,
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                })
            delta = {"tool_calls": tc_list}
            data = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"
            finish_reason = "tool_calls"
        else:
            finish_reason = "stop"

        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
        yield f"data: {json.dumps(final)}\n\n"
        yield "data: [DONE]\n\n"

        total_dur = (time.time() - start_ts) * 1000
        ttft = (first_token_ts - start_ts) * 1000 if first_token_ts else None
        _log_request(
            request,
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias=model_name,
            model_repo=repo_id,
            stream=True,
            total_duration_ms=total_dur,
            time_to_first_token_ms=ttft,
            messages_count=len(messages),
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts, tools=None,
    draft_model=None, speculative_tokens=None,
):
    """Return non-streaming JSON response."""
    try:
        if engine_type == "text":
            from ppmlx.engine import get_engine
            engine = get_engine()
            text, reasoning, prompt_tokens, completion_tokens = engine.generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=max_tokens,
                seed=seed,
                repetition_penalty=repetition_penalty,
                tools=tools,
                **_speculative_kwargs(draft_model, speculative_tokens),
            )
        elif engine_type == "vision":
            from ppmlx.engine_vlm import get_vision_engine
            engine = get_vision_engine()
            text, prompt_tokens, completion_tokens = engine.generate(repo_id, messages)
            reasoning = None
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is an embedding model.")
    except HTTPException:
        raise
    except Exception as e:
        _log_request(
            request,
            request_id=request_id,
            endpoint="/v1/chat/completions",
            model_alias=model_name,
            model_repo=repo_id,
            stream=False,
            status="error",
            error_message=str(e),
        )
        raise HTTPException(status_code=503, detail=str(e))

    total_dur = (time.time() - start_ts) * 1000

    # Parse tool calls if tools were provided
    remaining_text, tool_calls = _parse_tool_calls(text) if tools else (text, [])

    message: dict = {"role": "assistant", "content": remaining_text or None}
    if reasoning:
        message["reasoning"] = reasoning

    finish_reason = "stop"
    if tool_calls:
        finish_reason = "tool_calls"
        message["tool_calls"] = [
            {
                "id": "call_" + uuid.uuid4().hex[:24],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in tool_calls
        ]

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    _log_request(
        request,
        request_id=request_id,
        endpoint="/v1/chat/completions",
        model_alias=model_name,
        model_repo=repo_id,
        stream=False,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        total_duration_ms=total_dur,
        messages_count=len(messages),
    )

    return JSONResponse(response)


@app.post("/v1/completions")
async def completions(request: Request):
    """OpenAI-compatible text completions endpoint."""
    body = await request.json()
    model_name = body.get("model", "")
    prompt = body.get("prompt", "")
    max_tokens = body.get("max_tokens")
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    messages = [{"role": "user", "content": prompt}]

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    request_id = "cmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    start_ts = time.time()

    try:
        from ppmlx.engine import get_engine
        engine = get_engine()
        text, reasoning, prompt_tokens, completion_tokens = engine.generate(
            repo_id, messages,
            temperature=0.7 if temperature is None else temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    return JSONResponse({
        "id": request_id,
        "object": "text_completion",
        "created": created,
        "model": model_name,
        "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    })


@app.post("/v1/embeddings")
async def embeddings(request: Request):
    """OpenAI-compatible embeddings endpoint."""
    body = await request.json()
    model_name = body.get("model", "")
    input_text = body.get("input", "")

    if isinstance(input_text, str):
        texts = [input_text]
    else:
        texts = list(input_text)

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    try:
        from ppmlx.engine_embed import get_embed_engine
        engine = get_embed_engine()
        vectors = engine.encode(repo_id, texts)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Embeddings require the 'mlx-embeddings' package. Install with: pip install ppmlx[embeddings]",
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    data = [{"object": "embedding", "embedding": vec, "index": i} for i, vec in enumerate(vectors)]
    total_tokens = sum(len(t.split()) for t in texts)

    _log_request(
        request,
        request_id="embed-" + uuid.uuid4().hex[:8],
        endpoint="/v1/embeddings",
        model_alias=model_name,
        model_repo=repo_id,
        stream=False,
        prompt_tokens=total_tokens,
        total_tokens=total_tokens,
    )

    return JSONResponse({
        "object": "list",
        "data": data,
        "model": model_name,
        "usage": {"prompt_tokens": total_tokens, "total_tokens": total_tokens},
    })


# ── OpenAI Responses API (/v1/responses) ─────────────────────────────


def _responses_input_to_messages(input_data) -> list[dict]:
    """Convert Responses API 'input' field to chat messages list."""
    if isinstance(input_data, str):
        return [{"role": "user", "content": input_data}]
    if isinstance(input_data, list):
        messages = []
        for item in input_data:
            if isinstance(item, str):
                messages.append({"role": "user", "content": item})
            elif isinstance(item, dict):
                role = item.get("role", "user")
                if role == "developer":
                    role = "system"
                content = item.get("content", "")
                # Handle content array (e.g. [{type: "input_text", text: "..."}])
                if isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict):
                            t = part.get("text", "")
                            if t:
                                text_parts.append(t)
                    content = "\n".join(text_parts) if text_parts else ""
                messages.append({"role": role, "content": content})
        return messages
    return [{"role": "user", "content": str(input_data)}]


@app.post("/v1/responses")
async def responses(request: Request):
    """OpenAI Responses API endpoint (used by Codex and newer OpenAI tools)."""
    body = await request.json()

    # Append each request to a debug log (best-effort)
    try:
        import pathlib
        _log_path = pathlib.Path("/tmp/ppmlx_responses_debug.jsonl")
        input_data_raw = body.get("input", "")
        # Summarize input for logging
        if isinstance(input_data_raw, list):
            input_summary = []
            for item in input_data_raw:
                role = item.get("role", "?")
                content = item.get("content", "")
                if isinstance(content, list):
                    text = " ".join(p.get("text", "")[:200] for p in content if isinstance(p, dict))
                elif isinstance(content, str):
                    text = content[:200]
                else:
                    text = str(content)[:200]
                input_summary.append({"role": role, "text": text[:200]})
        else:
            input_summary = str(input_data_raw)[:200]
        log_entry = {
            "ts": time.time(),
            "model": body.get("model"),
            "stream": body.get("stream"),
            "tools": len(body.get("tools") or []),
            "input_summary": input_summary,
            "instructions_len": len(body.get("instructions", "") or ""),
        }
        with open(_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass

    model_name = body.get("model", "")
    input_data = body.get("input", "")
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = body.get("max_output_tokens") or body.get("max_tokens")
    # instructions field acts as a system prompt
    instructions = body.get("instructions")

    tools = _limit_tools(body.get("tools") or None)
    log.info("POST /v1/responses model=%s stream=%s tools=%d",
             model_name, stream, len(tools) if tools else 0)

    messages = _responses_input_to_messages(input_data)
    if instructions:
        messages.insert(0, {"role": "system", "content": instructions})

    # Merge consecutive system messages into one (some models only allow a
    # single system message at the start, e.g. Qwen).
    messages = _merge_system_messages(messages)

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    resp_id = "resp_" + uuid.uuid4().hex[:24]
    msg_id = "msg_" + uuid.uuid4().hex[:24]
    created = int(time.time())

    if stream:
        return _stream_responses(
            resp_id, msg_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, request, tools,
        )
    else:
        return await _nonstream_responses(
            resp_id, msg_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, request, tools,
        )


def _sse(event: str, data: dict, seq: list[int] | None = None) -> str:
    """Format an SSE event.

    *seq* is a mutable 1-element list used as an auto-incrementing counter
    (pass the same list across calls to get monotonic sequence numbers).
    """
    if "type" not in data:
        data = {**data, "type": event}
    if seq is not None:
        data["sequence_number"] = seq[0]
        seq[0] += 1
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


_SENTINEL = object()


async def _async_iter_sync_gen(sync_gen, loop=None):
    """Yield items from a blocking sync generator without blocking the event loop.

    Runs the generator in a background thread and bridges items to the async
    world via an :class:`asyncio.Queue`.
    """
    if loop is None:
        loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    def _producer():
        try:
            for item in sync_gen:
                loop.call_soon_threadsafe(q.put_nowait, item)
        except Exception as exc:
            loop.call_soon_threadsafe(q.put_nowait, exc)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, _SENTINEL)

    import threading
    t = threading.Thread(target=_producer, daemon=True)
    t.start()

    while True:
        item = await q.get()
        if item is _SENTINEL:
            break
        if isinstance(item, Exception):
            raise item
        yield item


def _stream_responses(
    resp_id, msg_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, request,
    tools=None,
):
    """Streaming SSE for the Responses API."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        log.info("responses stream start model=%s tools=%d msgs=%d",
                 model_name, len(tools) if tools else 0, len(messages))
        seq = [0]  # mutable counter for sequence_number

        # Build the full response object (matches Ollama's structure)
        resp_obj = {
            "id": resp_id,
            "object": "response",
            "created_at": created,
            "completed_at": None,
            "model": model_name,
            "status": "in_progress",
            "output": [],
            "usage": None,
            "error": None,
            "instructions": None,
            "tools": tools or [],
            "tool_choice": "auto",
            "parallel_tool_calls": True,
            "truncation": "disabled",
            "temperature": temperature,
            "top_p": top_p,
            "max_output_tokens": max_tokens,
            "previous_response_id": None,
            "metadata": {},
            "text": {"format": {"type": "text"}},
        }

        # response.created — wrapped in "response" key (Codex/Ollama format)
        yield _sse("response.created", {"response": resp_obj}, seq)

        # response.in_progress
        yield _sse("response.in_progress", {"response": resp_obj}, seq)

        full_text = ""
        reasoning_text = ""
        start_ts = time.time()
        # Track output index: reasoning (if any) is at 0, message at 0 or 1
        reasoning_idx: int | None = None
        msg_output_idx = 0

        try:
            if engine_type == "text":
                from ppmlx.engine import get_engine
                engine = get_engine()
                gen = engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=max_tokens,
                    tools=tools,
                    strip_thinking=False,  # Handle thinking in server
                )

                # Qwen3 template injects <think> into the prompt, so model
                # output starts inside a thinking block.  Start in thinking
                # mode and emit a reasoning output item immediately.
                in_thinking = True
                msg_item_emitted = False
                buf = ""
                reasoning_idx = 0
                msg_output_idx = 1
                rs_id = "rs_" + uuid.uuid4().hex[:12]
                yield _sse("response.output_item.added", {
                    "output_index": reasoning_idx,
                    "item": {"id": rs_id, "type": "reasoning", "summary": []},
                }, seq)

                async for chunk in _async_iter_sync_gen(gen):
                    buf += chunk
                    while buf:
                        if not in_thinking:
                            think_pos = buf.find("<think>")
                            close_pos = buf.find("</think>")
                            if think_pos == 0:
                                in_thinking = True
                                buf = buf[len("<think>"):]
                                if reasoning_idx is None:
                                    reasoning_idx = 0
                                    msg_output_idx = 1
                                    rs_id = "rs_" + uuid.uuid4().hex[:12]
                                    yield _sse("response.output_item.added", {
                                        "output_index": reasoning_idx,
                                        "item": {"id": rs_id, "type": "reasoning", "summary": []},
                                    }, seq)
                                continue
                            elif close_pos == 0:
                                buf = buf[len("</think>"):]
                                if reasoning_idx is not None:
                                    yield _sse("response.output_item.done", {
                                        "output_index": reasoning_idx,
                                        "item": {"id": rs_id, "type": "reasoning",
                                                 "summary": [{"type": "summary_text", "text": reasoning_text}]},
                                    }, seq)
                                continue
                            else:
                                # Check for partial tag
                                partial = any(buf.endswith(t[:i])
                                              for t in ("<think>", "</think>")
                                              for i in range(1, len(t)))
                                if partial:
                                    break
                                # Plain text — buffer it (don't stream yet).
                                # We'll emit clean text after parsing tool calls.
                                text_chunk = buf[:think_pos] if think_pos > 0 else buf
                                buf = buf[len(text_chunk):]
                                if text_chunk:
                                    full_text += text_chunk
                                if not buf:
                                    break
                        else:
                            close_pos = buf.find("</think>")
                            if close_pos >= 0:
                                think_chunk = buf[:close_pos]
                                buf = buf[close_pos + len("</think>"):]
                                in_thinking = False
                                if think_chunk:
                                    reasoning_text += think_chunk
                                    yield _sse("response.reasoning_summary_text.delta", {
                                        "output_index": reasoning_idx,
                                        "delta": think_chunk,
                                    }, seq)
                                yield _sse("response.reasoning_summary_text.done", {
                                    "output_index": reasoning_idx,
                                    "text": reasoning_text,
                                }, seq)
                                yield _sse("response.output_item.done", {
                                    "output_index": reasoning_idx,
                                    "item": {"id": rs_id, "type": "reasoning",
                                             "summary": [{"type": "summary_text", "text": reasoning_text}]},
                                }, seq)
                                continue
                            else:
                                # Partial check
                                partial_len = 0
                                for i in range(1, len("</think>")):
                                    if buf.endswith("</think>"[:i]):
                                        partial_len = i
                                        break
                                safe = buf[:len(buf) - partial_len] if partial_len else buf
                                buf = buf[len(safe):] if partial_len else ""
                                if safe:
                                    reasoning_text += safe
                                    if reasoning_idx is None:
                                        # Template-injected thinking
                                        reasoning_idx = 0
                                        msg_output_idx = 1
                                        rs_id = "rs_" + uuid.uuid4().hex[:12]
                                        in_thinking = True
                                        yield _sse("response.output_item.added", {
                                            "output_index": reasoning_idx,
                                            "item": {"id": rs_id, "type": "reasoning", "summary": []},
                                        }, seq)
                                    yield _sse("response.reasoning_summary_text.delta", {
                                        "output_index": reasoning_idx,
                                        "delta": safe,
                                    }, seq)
                                break

                # Flush remaining buf as text (buffered, not streamed)
                if buf and not in_thinking:
                    full_text += buf
            elif engine_type == "vision":
                from ppmlx.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(
                    repo_id, messages,
                    max_tokens=max_tokens,
                )
                full_text = text
                yield _sse("response.output_text.delta", {
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                }, seq)
        except Exception as e:
            log.exception("responses stream error")
            yield _sse("error", {
                "type": "server_error",
                "message": str(e),
            }, seq)
            return

        gen_dur = time.time() - start_ts
        log.info("responses generation done in %.1fs, %d chars", gen_dur, len(full_text))

        # Parse tool calls from the model output
        remaining_text, tool_calls = _parse_tool_calls(full_text)
        log.info("parsed %d tool_calls, remaining_text=%d chars",
                 len(tool_calls), len(remaining_text))

        # ── Emit text message (clean, without <tool_call> blocks) ────
        msg_item = {
            "type": "message", "id": msg_id,
            "status": "in_progress", "role": "assistant", "content": [],
        }
        yield _sse("response.output_item.added", {
            "output_index": msg_output_idx, "item": msg_item,
        }, seq)
        yield _sse("response.content_part.added", {
            "output_index": msg_output_idx, "content_index": 0,
            "part": {"type": "output_text", "text": "", "annotations": []},
        }, seq)
        if remaining_text:
            yield _sse("response.output_text.delta", {
                "output_index": msg_output_idx, "content_index": 0,
                "delta": remaining_text,
            }, seq)
        yield _sse("response.output_text.done", {
            "output_index": msg_output_idx,
            "content_index": 0,
            "text": remaining_text,
        }, seq)
        yield _sse("response.content_part.done", {
            "output_index": msg_output_idx,
            "content_index": 0,
            "part": {"type": "output_text", "text": remaining_text, "annotations": []},
        }, seq)
        done_msg = {
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
        }
        yield _sse("response.output_item.done", {
            "output_index": msg_output_idx,
            "item": done_msg,
        }, seq)

        output_items: list[dict] = []
        # Include reasoning item if we emitted one
        if reasoning_idx is not None:
            output_items.append({
                "type": "reasoning",
                "id": rs_id,
                "summary": [{"type": "summary_text", "text": reasoning_text}],
                "encrypted_content": reasoning_text,
            })
        output_items.append(done_msg)
        output_idx = msg_output_idx + 1

        # ── Emit function call items ─────────────────────────────────
        for tc in tool_calls:
            fc_id = "fc_" + uuid.uuid4().hex[:24]
            call_id = "call_" + uuid.uuid4().hex[:24]
            fc_item = {
                "type": "function_call",
                "id": fc_id,
                "call_id": call_id,
                "name": tc["name"],
                "arguments": "",
                "status": "in_progress",
            }
            yield _sse("response.output_item.added", {
                "output_index": output_idx,
                "item": fc_item,
            }, seq)
            yield _sse("response.function_call_arguments.delta", {
                "output_index": output_idx,
                "delta": tc["arguments"],
            }, seq)
            yield _sse("response.function_call_arguments.done", {
                "output_index": output_idx,
                "arguments": tc["arguments"],
            }, seq)
            done_fc = {**fc_item, "arguments": tc["arguments"], "status": "completed"}
            yield _sse("response.output_item.done", {
                "output_index": output_idx,
                "item": done_fc,
            }, seq)
            output_items.append(done_fc)
            output_idx += 1

        # ── Wrap up ──────────────────────────────────────────────────
        prompt_tokens = sum(len(m.get("content", "").split()) for m in messages if isinstance(m.get("content"), str))
        completion_tokens = len(full_text.split())
        usage = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        }

        resp_obj["status"] = "completed"
        resp_obj["completed_at"] = int(time.time())
        resp_obj["output"] = output_items
        resp_obj["usage"] = usage
        yield _sse("response.completed", {"response": resp_obj}, seq)
        log.info("responses stream completed, %d output items", len(output_items))

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_responses(
    resp_id, msg_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, request,
    tools=None,
):
    """Non-streaming Responses API."""
    try:
        if engine_type == "text":
            from ppmlx.engine import get_engine
            engine = get_engine()
            text, reasoning, prompt_tokens, completion_tokens = engine.generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=max_tokens,
                tools=tools,
            )
        elif engine_type == "vision":
            from ppmlx.engine_vlm import get_vision_engine
            engine = get_vision_engine()
            text, prompt_tokens, completion_tokens = engine.generate(repo_id, messages)
        else:
            raise HTTPException(status_code=400, detail=f"Model '{model_name}' is an embedding model.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    remaining_text, tool_calls = _parse_tool_calls(text)

    output: list[dict] = []
    if remaining_text:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
        })
    for tc in tool_calls:
        output.append({
            "type": "function_call",
            "id": "fc_" + uuid.uuid4().hex[:24],
            "call_id": "call_" + uuid.uuid4().hex[:24],
            "name": tc["name"],
            "arguments": tc["arguments"],
            "status": "completed",
        })
    # Fallback: if no tool calls and no remaining text, emit original text
    if not output:
        output.append({
            "type": "message",
            "id": msg_id,
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        })

    return JSONResponse({
        "id": resp_id,
        "object": "response",
        "created_at": created,
        "completed_at": int(time.time()),
        "model": model_name,
        "status": "completed",
        "output": output,
        "error": None,
        "tools": tools or [],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "truncation": "disabled",
        "metadata": {},
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    })


# ── Anthropic Messages API (/v1/messages) ─────────────────────────────
#
# Claude Code talks to this endpoint when ANTHROPIC_BASE_URL is set.
# Format mirrors the Anthropic Messages API (not OpenAI).


def _anthropic_sse(data: dict) -> str:
    event_type = data.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


@app.post("/v1/messages")
async def anthropic_messages(request: Request):
    """Anthropic Messages API endpoint (used by Claude Code)."""
    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    max_tokens = body.get("max_tokens")
    temperature = body.get("temperature", 0.7)
    system_prompt = body.get("system")
    tools = _limit_tools(body.get("tools") or None)

    # Build chat messages
    chat_messages: list[dict] = []
    if system_prompt:
        if isinstance(system_prompt, list):
            system_prompt = "\n".join(
                p.get("text", "") for p in system_prompt if isinstance(p, dict)
            )
        chat_messages.append({"role": "system", "content": system_prompt})

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # Anthropic content can be a list of blocks
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "tool_result":
                        # Tool results in Anthropic format
                        tc_content = part.get("content", "")
                        if isinstance(tc_content, list):
                            tc_content = "\n".join(
                                p.get("text", "") for p in tc_content
                                if isinstance(p, dict)
                            )
                        chat_messages.append({"role": "user", "content": tc_content})
                        continue
                    elif part.get("type") == "tool_use":
                        # Previous assistant tool call — convert
                        chat_messages.append({
                            "role": "assistant",
                            "content": f'<tool_call>\n{{"name": "{part.get("name","")}", '
                                       f'"arguments": {json.dumps(part.get("input", {}))}}}\n</tool_call>',
                        })
                        continue
            if text_parts:
                chat_messages.append({"role": role, "content": "\n".join(text_parts)})
        else:
            chat_messages.append({"role": role, "content": content})

    chat_messages = _merge_system_messages(chat_messages)

    # Convert Anthropic tools format to OpenAI format for apply_chat_template
    oai_tools = None
    if tools:
        oai_tools = []
        for t in tools:
            oai_tools.append({
                "type": "function",
                "function": {
                    "name": t.get("name", ""),
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {}),
                },
            })

    try:
        from ppmlx.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    msg_id = "msg_" + uuid.uuid4().hex[:24]

    if stream:
        return _stream_anthropic(
            msg_id, model_name, repo_id, chat_messages,
            temperature, max_tokens, oai_tools, tools,
        )
    else:
        return await _nonstream_anthropic(
            msg_id, model_name, repo_id, chat_messages,
            temperature, max_tokens, oai_tools, tools,
        )


def _stream_anthropic(
    msg_id, model_name, repo_id, messages,
    temperature, max_tokens, oai_tools, orig_tools,
):
    from fastapi.responses import StreamingResponse

    async def event_generator():
        # message_start
        yield _anthropic_sse({
            "type": "message_start",
            "message": {
                "id": msg_id,
                "type": "message",
                "role": "assistant",
                "model": model_name,
                "content": [],
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        full_text = ""
        content_idx = 0

        # State machine: track whether we're inside <think> or in text.
        # We use strip_thinking=False so we get raw tokens including
        # <think>...</think>, then emit them as thinking_delta / text_delta.
        # Start in thinking mode — Qwen3 template injects <think> into
        # the prompt so the model starts generating inside a thinking block.
        in_thinking = True
        thinking_started = True
        text_started = False
        buf = ""

        try:
            from ppmlx.engine import get_engine
            engine = get_engine()
            gen = engine.stream_generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                max_tokens=max_tokens,
                tools=oai_tools,
                strip_thinking=False,  # We handle thinking/text separation here
            )

            # Emit thinking block start immediately
            yield _anthropic_sse({
                "type": "content_block_start",
                "index": content_idx,
                "content_block": {"type": "thinking", "thinking": ""},
            })

            async for chunk in _async_iter_sync_gen(gen):
                buf += chunk

                # Detect transitions between thinking and text
                while buf:
                    if not in_thinking:
                        # Look for <think> to start thinking block
                        think_pos = buf.find("<think>")
                        close_pos = buf.find("</think>")

                        if think_pos == 0:
                            # Start thinking block
                            in_thinking = True
                            buf = buf[len("<think>"):]
                            if not thinking_started:
                                thinking_started = True
                                yield _anthropic_sse({
                                    "type": "content_block_start",
                                    "index": content_idx,
                                    "content_block": {"type": "thinking", "thinking": ""},
                                })
                            continue
                        elif close_pos == 0:
                            # Closing tag without opening — template injected
                            # <think> into prompt, so we started inside thinking
                            buf = buf[len("</think>"):]
                            if thinking_started:
                                yield _anthropic_sse({
                                    "type": "content_block_stop",
                                    "index": content_idx,
                                })
                                content_idx += 1
                                thinking_started = False
                            continue
                        elif think_pos > 0:
                            # Text before <think>
                            text_chunk = buf[:think_pos]
                            buf = buf[think_pos:]
                            if text_chunk.strip():
                                if not text_started:
                                    text_started = True
                                    yield _anthropic_sse({
                                        "type": "content_block_start",
                                        "index": content_idx,
                                        "content_block": {"type": "text", "text": ""},
                                    })
                                full_text += text_chunk
                                yield _anthropic_sse({
                                    "type": "content_block_delta",
                                    "index": content_idx,
                                    "delta": {"type": "text_delta", "text": text_chunk},
                                })
                            continue
                        else:
                            # No tag found — might be partial tag at end of buf
                            # Check for partial "<thi" or "</thi" at end
                            partial = False
                            for tag in ("<think>", "</think>"):
                                for i in range(1, len(tag)):
                                    if buf.endswith(tag[:i]):
                                        partial = True
                                        break
                                if partial:
                                    break
                            if partial:
                                break  # Wait for more data

                            # Plain text, no partial tags
                            text_chunk = buf
                            buf = ""
                            if text_chunk:
                                if not text_started:
                                    text_started = True
                                    yield _anthropic_sse({
                                        "type": "content_block_start",
                                        "index": content_idx,
                                        "content_block": {"type": "text", "text": ""},
                                    })
                                full_text += text_chunk
                                yield _anthropic_sse({
                                    "type": "content_block_delta",
                                    "index": content_idx,
                                    "delta": {"type": "text_delta", "text": text_chunk},
                                })
                            break
                    else:
                        # Inside thinking block — look for </think>
                        close_pos = buf.find("</think>")
                        if close_pos >= 0:
                            think_chunk = buf[:close_pos]
                            buf = buf[close_pos + len("</think>"):]
                            in_thinking = False
                            if think_chunk:
                                yield _anthropic_sse({
                                    "type": "content_block_delta",
                                    "index": content_idx,
                                    "delta": {"type": "thinking_delta", "thinking": think_chunk},
                                })
                            yield _anthropic_sse({
                                "type": "content_block_stop",
                                "index": content_idx,
                            })
                            content_idx += 1
                            thinking_started = False
                            continue
                        else:
                            # Check for partial "</thi" at end
                            partial = False
                            for i in range(1, len("</think>")):
                                if buf.endswith("</think>"[:i]):
                                    partial = True
                                    break
                            if partial:
                                # Emit everything except the partial tag
                                safe = buf[:len(buf) - i]
                                buf = buf[len(buf) - i:]
                            else:
                                safe = buf
                                buf = ""
                            if safe:
                                if not thinking_started:
                                    # Template injected <think>, we start mid-think
                                    thinking_started = True
                                    in_thinking = True
                                    yield _anthropic_sse({
                                        "type": "content_block_start",
                                        "index": content_idx,
                                        "content_block": {"type": "thinking", "thinking": ""},
                                    })
                                yield _anthropic_sse({
                                    "type": "content_block_delta",
                                    "index": content_idx,
                                    "delta": {"type": "thinking_delta", "thinking": safe},
                                })
                            break

            # Flush remaining buffer
            if buf:
                if in_thinking or thinking_started:
                    if buf.strip():
                        yield _anthropic_sse({
                            "type": "content_block_delta",
                            "index": content_idx,
                            "delta": {"type": "thinking_delta", "thinking": buf},
                        })
                    yield _anthropic_sse({"type": "content_block_stop", "index": content_idx})
                    content_idx += 1
                else:
                    if not text_started:
                        text_started = True
                        yield _anthropic_sse({
                            "type": "content_block_start",
                            "index": content_idx,
                            "content_block": {"type": "text", "text": ""},
                        })
                    full_text += buf
                    yield _anthropic_sse({
                        "type": "content_block_delta",
                        "index": content_idx,
                        "delta": {"type": "text_delta", "text": buf},
                    })
            elif thinking_started:
                yield _anthropic_sse({"type": "content_block_stop", "index": content_idx})
                content_idx += 1

            # Close text block if open
            if text_started:
                yield _anthropic_sse({"type": "content_block_stop", "index": content_idx})
                content_idx += 1

        except Exception as e:
            yield _anthropic_sse({
                "type": "error",
                "error": {"type": "server_error", "message": str(e)},
            })
            return

        # Parse tool calls from the collected text output
        remaining_text, tool_calls = _parse_tool_calls(full_text)

        # Emit tool_use blocks
        stop_reason = "end_turn"
        if tool_calls:
            stop_reason = "tool_use"
            for tc in tool_calls:
                tc_id = "toolu_" + uuid.uuid4().hex[:24]
                try:
                    args_obj = json.loads(tc["arguments"])
                except (json.JSONDecodeError, ValueError):
                    args_obj = tc["arguments"]
                yield _anthropic_sse({
                    "type": "content_block_start",
                    "index": content_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc_id,
                        "name": tc["name"],
                        "input": {},
                    },
                })
                yield _anthropic_sse({
                    "type": "content_block_delta",
                    "index": content_idx,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": json.dumps(args_obj),
                    },
                })
                yield _anthropic_sse({
                    "type": "content_block_stop",
                    "index": content_idx,
                })
                content_idx += 1

        prompt_tokens = sum(
            len(m.get("content", "").split()) for m in messages
            if isinstance(m.get("content"), str)
        )
        completion_tokens = len(full_text.split())

        yield _anthropic_sse({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason},
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
            },
        })
        yield _anthropic_sse({"type": "message_stop"})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


async def _nonstream_anthropic(
    msg_id, model_name, repo_id, messages,
    temperature, max_tokens, oai_tools, orig_tools,
):
    try:
        from ppmlx.engine import get_engine
        engine = get_engine()
        text, reasoning, prompt_tokens, completion_tokens = engine.generate(
            repo_id, messages,
            temperature=0.7 if temperature is None else temperature,
            max_tokens=max_tokens,
            tools=oai_tools,
        )
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

    remaining_text, tool_calls = _parse_tool_calls(text)

    content: list[dict] = []
    if reasoning:
        content.append({"type": "thinking", "thinking": reasoning})
    if remaining_text:
        content.append({"type": "text", "text": remaining_text})

    stop_reason = "end_turn"
    for tc in tool_calls:
        stop_reason = "tool_use"
        try:
            args_obj = json.loads(tc["arguments"])
        except (json.JSONDecodeError, ValueError):
            args_obj = tc["arguments"]
        content.append({
            "type": "tool_use",
            "id": "toolu_" + uuid.uuid4().hex[:24],
            "name": tc["name"],
            "input": args_obj,
        })

    if not content:
        content.append({"type": "text", "text": text})

    return JSONResponse({
        "id": msg_id,
        "type": "message",
        "role": "assistant",
        "model": model_name,
        "content": content,
        "stop_reason": stop_reason,
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        },
    })


# ── WebSocket transport for Responses API (Codex) ────────────────────


@app.websocket("/v1/responses")
async def responses_ws(websocket: WebSocket):
    """WebSocket transport for the Responses API (used by Codex CLI)."""
    await websocket.accept()
    try:
        while True:
            body = await websocket.receive_json()
            msg_type = body.get("type", "")
            if msg_type != "response.create":
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "invalid_request", "message": f"Unknown message type: {msg_type}"},
                })
                continue

            # Extract fields from the response.create message
            response_body = body.get("response", body)
            model_name = response_body.get("model", body.get("model", ""))
            input_data = response_body.get("input", body.get("input", ""))
            temperature = response_body.get("temperature", body.get("temperature", 0.7))
            top_p = response_body.get("top_p", body.get("top_p", 1.0))
            max_tokens = (
                response_body.get("max_output_tokens")
                or body.get("max_output_tokens")
                or response_body.get("max_tokens")
                or body.get("max_tokens", 4096)
            )
            instructions = response_body.get("instructions", body.get("instructions"))
            tools = response_body.get("tools", body.get("tools")) or None

            messages = _responses_input_to_messages(input_data)
            if instructions:
                messages.insert(0, {"role": "system", "content": instructions})
            messages = _merge_system_messages(messages)

            try:
                from ppmlx.models import resolve_alias
                repo_id = resolve_alias(model_name)
            except Exception:
                repo_id = model_name

            has_imgs = _has_images(messages)
            engine_type = _route_engine(repo_id, has_imgs)

            resp_id = "resp_" + uuid.uuid4().hex[:24]
            msg_id = "msg_" + uuid.uuid4().hex[:24]
            created = int(time.time())

            resp_obj = {
                "id": resp_id,
                "object": "response",
                "created_at": created,
                "model": model_name,
                "status": "in_progress",
                "output": [],
                "usage": None,
            }
            await websocket.send_json({"type": "response.created", **resp_obj})

            full_text = ""
            try:
                if engine_type == "text":
                    from ppmlx.engine import get_engine
                    engine = get_engine()
                    if tools:
                        text, _, _, _ = engine.generate(
                            repo_id, messages,
                            temperature=0.7 if temperature is None else temperature,
                            top_p=1.0 if top_p is None else top_p,
                            max_tokens=max_tokens,
                            tools=tools,
                        )
                        full_text = text
                    else:
                        for chunk in engine.stream_generate(
                            repo_id, messages,
                            temperature=0.7 if temperature is None else temperature,
                            top_p=1.0 if top_p is None else top_p,
                            max_tokens=max_tokens,
                        ):
                            full_text += chunk
                elif engine_type == "vision":
                    from ppmlx.engine_vlm import get_vision_engine
                    engine = get_vision_engine()
                    text, _, _ = engine.generate(
                        repo_id, messages,
                        max_tokens=max_tokens,
                    )
                    full_text = text
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "error": {"type": "server_error", "message": str(e)},
                })
                continue

            remaining_text, tool_calls = _parse_tool_calls(full_text)
            output_items: list[dict] = []
            output_idx = 0

            # Text message
            if remaining_text:
                msg_item = {
                    "type": "message", "id": msg_id,
                    "status": "in_progress", "role": "assistant", "content": [],
                }
                await websocket.send_json({
                    "type": "response.output_item.added",
                    "output_index": output_idx, "item": msg_item,
                })
                await websocket.send_json({
                    "type": "response.content_part.added",
                    "output_index": output_idx, "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                })
                await websocket.send_json({
                    "type": "response.output_text.delta",
                    "output_index": output_idx, "content_index": 0,
                    "delta": remaining_text,
                })
                await websocket.send_json({
                    "type": "response.output_text.done",
                    "output_index": output_idx, "content_index": 0,
                    "text": remaining_text,
                })
                await websocket.send_json({
                    "type": "response.content_part.done",
                    "output_index": output_idx, "content_index": 0,
                    "part": {"type": "output_text", "text": remaining_text, "annotations": []},
                })
                done_msg = {
                    "type": "message", "id": msg_id, "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": remaining_text, "annotations": []}],
                }
                await websocket.send_json({
                    "type": "response.output_item.done",
                    "output_index": output_idx, "item": done_msg,
                })
                output_items.append(done_msg)
                output_idx += 1

            # Function calls
            for tc in tool_calls:
                fc_id = "fc_" + uuid.uuid4().hex[:24]
                call_id = "call_" + uuid.uuid4().hex[:24]
                fc_item = {
                    "type": "function_call", "id": fc_id, "call_id": call_id,
                    "name": tc["name"], "arguments": "", "status": "in_progress",
                }
                await websocket.send_json({
                    "type": "response.output_item.added",
                    "output_index": output_idx, "item": fc_item,
                })
                await websocket.send_json({
                    "type": "response.function_call_arguments.delta",
                    "output_index": output_idx, "delta": tc["arguments"],
                })
                await websocket.send_json({
                    "type": "response.function_call_arguments.done",
                    "output_index": output_idx, "arguments": tc["arguments"],
                })
                done_fc = {**fc_item, "arguments": tc["arguments"], "status": "completed"}
                await websocket.send_json({
                    "type": "response.output_item.done",
                    "output_index": output_idx, "item": done_fc,
                })
                output_items.append(done_fc)
                output_idx += 1

            prompt_tokens = sum(
                len(m.get("content", "").split()) for m in messages
                if isinstance(m.get("content"), str)
            )
            completion_tokens = len(full_text.split())
            usage = {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }

            resp_obj["status"] = "completed"
            resp_obj["output"] = output_items or [{
                "type": "message", "id": msg_id, "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": full_text, "annotations": []}],
            }]
            resp_obj["usage"] = usage
            await websocket.send_json({"type": "response.completed", **resp_obj})

    except WebSocketDisconnect:
        pass
    except Exception:
        pass
