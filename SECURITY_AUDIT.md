# ppmlx Security Audit

**Date:** 2026-03-26
**Scope:** Full codebase review (server.py, engine.py, engine_vlm.py, engine_embed.py, db.py, models.py, config.py, cli.py, quantize.py, memory.py, schema.py, registry.py)

---

## Critical

### C1. Debug JSONL Logs to /tmp/ Expose Request Data
**File:** `server.py` lines 427-442, 837-866
**Description:** Two endpoints (`/v1/chat/completions` and `/v1/responses`) write debug logs to world-readable `/tmp/ppmlx_chatcompletions_debug.jsonl` and `/tmp/ppmlx_responses_debug.jsonl`. These logs contain model names, tool names, message counts, instructions lengths, and input summaries. Any local user can read these files, and they grow unboundedly.
**Impact:** Information disclosure of user prompts and API usage to any local process.
**Fix:** Remove debug JSONL logging entirely.

### C2. CORS Wildcard Allows Cross-Origin Attacks
**File:** `server.py` lines 64-70
**Description:** CORS is configured with `allow_origins=["*"]` and `allow_credentials=True`. This allows any website to make authenticated cross-origin requests to the server, enabling prompt injection attacks from malicious web pages.
**Impact:** A malicious website can invoke the LLM API on behalf of any user whose browser can reach the server.
**Fix:** Default CORS origins to `["http://localhost:*", "http://127.0.0.1:*"]`. Make configurable via config.toml. The `allow_credentials=True` with `allow_origins=["*"]` combination is actually rejected by browsers, but the intent is clearly wrong.

### C3. Vision Engine file:// URL Allows Arbitrary File Read (SSRF/LFI)
**File:** `engine_vlm.py` lines 74-78
**Description:** The `_extract_images` method accepts `file://` URLs and bare absolute paths (including `~` expansion) from API request bodies. This allows any API client to read arbitrary files from the server's filesystem by submitting them as "images" to the vision model.
**Impact:** Arbitrary local file read via API requests.
**Fix:** Reject `file://` URLs and absolute paths from remote API requests. Only allow http/https URLs and base64 data URIs.

---

## High

### H1. No Request Body Size Limit
**File:** `server.py` — all POST endpoints
**Description:** No middleware or endpoint-level limit on request body size. An attacker can send multi-GB payloads to exhaust server memory.
**Impact:** Denial of service via memory exhaustion.
**Fix:** Add request body size limiting middleware (e.g., 10 MB default, configurable).

### H2. No max_tokens Cap Allows Memory Exhaustion
**File:** `server.py` — all generation endpoints
**Description:** While the engine has `_MAX_AUTO_TOKENS = 32_768` for the `None` case, a client can explicitly set `max_tokens` to any value (e.g., 10,000,000). This can cause the model to allocate enormous KV cache and exhaust unified memory.
**Impact:** Denial of service via memory exhaustion.
**Fix:** Enforce a configurable server-side maximum (default 32768). Clamp client-requested values.

### H3. `_flush_port` Kills Arbitrary PIDs
**File:** `cli.py` lines 599-623
**Description:** `_flush_port` runs `lsof` to find PIDs on a port, then `os.kill(pid, 9)` on each without verifying ownership. A race condition or port reuse could kill unrelated processes.
**Impact:** Denial of service to unrelated processes on the system.
**Fix:** Validate that the target PID belongs to a ppmlx process before killing.

### H4. Unbounded Embedding Input Allows Resource Exhaustion
**File:** `server.py` lines 748-797
**Description:** The embeddings endpoint accepts an unbounded list of input texts. An attacker can send thousands of texts in a single request.
**Impact:** Resource exhaustion.
**Fix:** Limit the number of texts per embedding request.

### H5. WebSocket Has No Message Size Limit or Rate Limiting
**File:** `server.py` lines 1793-1986
**Description:** The WebSocket endpoint accepts unlimited messages of unlimited size with no rate limiting. A single client can monopolize the server.
**Impact:** Denial of service.
**Fix:** Add message size limits and basic rate limiting to the WebSocket handler.

---

## Medium

### M1. Error Responses Leak Internal Paths and Stack Traces
**File:** `server.py` lines 648, 732, 776, 1294, 1749
**Description:** Exception messages are passed directly to HTTP error responses via `detail=str(e)`. This can leak internal file paths, model paths, and library internals.
**Recommendation:** Sanitize error messages; log full exceptions server-side.

### M2. No Input Validation on Model Names
**File:** `server.py` — all endpoints accepting model names
**Description:** Model names from request bodies are passed directly to `resolve_alias()` without validation. While `resolve_alias` does check for known aliases and `/` in the name, extremely long or malformed names could cause unexpected behavior.
**Recommendation:** Validate model name length and character set.

### M3. System Prompts Stored in DB Without Length Limits
**File:** `db.py` lines 144-180
**Description:** The `system_prompt` field is stored without any length limit. A very large system prompt could bloat the database.
**Recommendation:** Truncate stored system prompts.

### M4. Subprocess in quantize.py Passes User Input
**File:** `quantize.py` lines 104-130
**Description:** `_try_subprocess` constructs a command list from `repo_id` and config values. While it uses list-based `subprocess.run` (not shell=True), the `repo_id` comes from user input through `resolve_alias` and could contain unusual characters.
**Recommendation:** Already using list-based subprocess (not shell injection). Low practical risk.

### M5. No Authentication Mechanism
**Description:** The API accepts any request without authentication. While this is by design for a local development tool, users binding to `0.0.0.0` expose the API to the entire network.
**Recommendation:** Add optional API key authentication, warn when binding to non-localhost.

---

## Low

### L1. Temporary Files in Vision Engine Not Cleaned Up
**File:** `engine_vlm.py` lines 68-71
**Description:** Base64-decoded images are written to temp files with `delete=False` and never cleaned up.
**Recommendation:** Clean up temp files after use.

### L2. No Logging of Security Events
**Description:** Failed requests, rate limit hits, and authentication failures are not logged distinctly for security monitoring.
**Recommendation:** Add security event logging.

### L3. SQLite Database Has No Size Limit
**File:** `db.py`
**Description:** The request log database grows indefinitely.
**Recommendation:** Add periodic cleanup or rotation.

### L4. Config File Permissions Not Checked
**File:** `config.py`
**Description:** The config file containing HF tokens is not checked for proper permissions (should be 600).
**Recommendation:** Warn if config file is world-readable.

---

## Fixed Issues

The following Critical and High issues have been fixed in this commit:

- **C1**: Removed debug JSONL logging from `/v1/chat/completions` and `/v1/responses`
- **C2**: Made CORS origins configurable, defaulting to localhost-only patterns
- **C3**: Restricted vision engine image sources to reject `file://` URLs and absolute paths from API requests
- **H1**: Added request body size limiting middleware (default 10 MB, configurable)
- **H2**: Added configurable server-side max_tokens cap (default 32768)
- **H3**: Added PID ownership validation before killing processes in `_flush_port`
- **H4**: Limited embedding input to 256 texts per request
- **H5**: Added message size limits to WebSocket handler
