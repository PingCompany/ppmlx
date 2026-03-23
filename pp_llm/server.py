from __future__ import annotations
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: startup and shutdown."""
    try:
        from pp_llm.db import get_db
        db = get_db()
        db.init()
        app.state.db = db
    except ImportError:
        app.state.db = None

    try:
        from pp_llm.config import load_config
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
    title="pp-llm",
    version="0.1.0",
    description="Ollama-style OpenAI-compatible LLM API for Apple Silicon",
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
            from pp_llm.memory import get_system_ram_gb
            from pp_llm.db import get_db
            from pp_llm.engine import get_engine
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
        from pp_llm.models import is_vision_model, is_embed_model
        if is_embed_model(repo_id):
            return "embed"
        if has_images or is_vision_model(repo_id):
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
        from pp_llm.db import get_db
        get_db().log_request(**kwargs)
    except Exception:
        pass


# ── Endpoints ───────────────────────────────────────────────────────────

@app.get("/health")
async def health(request: Request):
    """Health check endpoint."""
    try:
        from pp_llm.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    try:
        from pp_llm.memory import get_system_ram_gb
        ram_gb = get_system_ram_gb()
    except Exception:
        ram_gb = 0.0

    return {
        "status": "ok",
        "version": "0.1.0",
        "loaded_models": loaded,
        "uptime_seconds": int(time.time() - _start_time),
        "system": {
            "memory_total_gb": round(ram_gb, 1),
        },
    }


@app.get("/metrics")
async def metrics():
    """Metrics endpoint — returns JSON stats from the DB."""
    try:
        from pp_llm.db import get_db
        stats = get_db().get_stats()
    except Exception:
        stats = {"total_requests": 0, "avg_duration_ms": None, "by_model": []}

    try:
        from pp_llm.engine import get_engine
        loaded = get_engine().list_loaded()
    except Exception:
        loaded = []

    return {**stats, "loaded_models": loaded}


@app.get("/v1/models")
async def list_models():
    """List available models (local + aliases)."""
    try:
        from pp_llm.models import list_local_models, all_aliases
        local = list_local_models()
        aliases = all_aliases()

        now = int(time.time())
        data = []
        seen = set()

        for m in local:
            mid = m.get("alias") or m.get("repo_id", "unknown")
            if mid not in seen:
                seen.add(mid)
                data.append({"id": mid, "object": "model", "created": now, "owned_by": "pp-llm"})

        for alias in aliases:
            if alias not in seen:
                seen.add(alias)
                data.append({"id": alias, "object": "model", "created": now, "owned_by": "pp-llm"})

        return {"object": "list", "data": data}
    except ImportError:
        return {"object": "list", "data": []}


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI-compatible chat completions endpoint."""
    body = await request.json()
    model_name = body.get("model", "")
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    temperature = body.get("temperature", 0.7)
    top_p = body.get("top_p", 1.0)
    max_tokens = body.get("max_tokens", 2048)
    stop = body.get("stop")
    seed = body.get("seed")
    repetition_penalty = body.get("repetition_penalty")

    try:
        from pp_llm.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    has_imgs = _has_images(messages)
    engine_type = _route_engine(repo_id, has_imgs)

    request_id = "chatcmpl-" + uuid.uuid4().hex[:12]
    start_ts = time.time()
    created = int(start_ts)

    if stream:
        return _stream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts
        )
    else:
        return await _nonstream_chat(
            request_id, created, model_name, repo_id, messages,
            engine_type, temperature, top_p, max_tokens, stop, seed,
            repetition_penalty, request, start_ts
        )


def _stream_chat(
    request_id, created, model_name, repo_id, messages,
    engine_type, temperature, top_p, max_tokens, stop, seed,
    repetition_penalty, request, start_ts
):
    """Return streaming SSE response."""
    from fastapi.responses import StreamingResponse

    async def event_generator():
        first_token_ts = None
        try:
            if engine_type == "text":
                from pp_llm.engine import get_engine
                engine = get_engine()
                for chunk in engine.stream_generate(
                    repo_id, messages,
                    temperature=0.7 if temperature is None else temperature,
                    top_p=1.0 if top_p is None else top_p,
                    max_tokens=2048 if max_tokens is None else max_tokens,
                    seed=seed,
                ):
                    if first_token_ts is None:
                        first_token_ts = time.time()
                    delta = {"role": "assistant", "content": chunk}
                    data = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": model_name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
                    }
                    yield f"data: {json.dumps(data)}\n\n"
            elif engine_type == "vision":
                from pp_llm.engine_vlm import get_vision_engine
                engine = get_vision_engine()
                text, _, _ = engine.generate(repo_id, messages, max_tokens=1024 if max_tokens is None else max_tokens)
                if first_token_ts is None:
                    first_token_ts = time.time()
                data = {
                    "id": request_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {"role": "assistant", "content": text}, "finish_reason": None}],
                }
                yield f"data: {json.dumps(data)}\n\n"
        except Exception as e:
            err = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n"

        final = {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
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
    repetition_penalty, request, start_ts
):
    """Return non-streaming JSON response."""
    try:
        if engine_type == "text":
            from pp_llm.engine import get_engine
            engine = get_engine()
            text, reasoning, prompt_tokens, completion_tokens = engine.generate(
                repo_id, messages,
                temperature=0.7 if temperature is None else temperature,
                top_p=1.0 if top_p is None else top_p,
                max_tokens=2048 if max_tokens is None else max_tokens,
                seed=seed,
                repetition_penalty=repetition_penalty,
            )
        elif engine_type == "vision":
            from pp_llm.engine_vlm import get_vision_engine
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

    message = {"role": "assistant", "content": text}
    if reasoning:
        message["reasoning"] = reasoning

    response = {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
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
    max_tokens = body.get("max_tokens", 2048)
    temperature = body.get("temperature", 0.7)
    stream = body.get("stream", False)

    messages = [{"role": "user", "content": prompt}]

    try:
        from pp_llm.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    request_id = "cmpl-" + uuid.uuid4().hex[:12]
    created = int(time.time())
    start_ts = time.time()

    try:
        from pp_llm.engine import get_engine
        engine = get_engine()
        text, reasoning, prompt_tokens, completion_tokens = engine.generate(
            repo_id, messages,
            temperature=0.7 if temperature is None else temperature,
            max_tokens=2048 if max_tokens is None else max_tokens,
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
        from pp_llm.models import resolve_alias
        repo_id = resolve_alias(model_name)
    except Exception:
        repo_id = model_name

    try:
        from pp_llm.engine_embed import get_embed_engine
        engine = get_embed_engine()
        vectors = engine.encode(repo_id, texts)
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
