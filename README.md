# qasr-mt

**Multi-tenant streaming ASR server for [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).**

A production-safe drop-in replacement for the upstream `qwen-asr-demo-streaming`
Flask demo — which is explicitly documented as "Single stream only (no
batching)" and leaks tokenizer/state across concurrent users.

This server keeps the same HTTP API shape (so existing clients Just Work) but
rewires the backend onto vLLM's `AsyncLLMEngine` with proper per-session
isolation. Verified concurrent transcription with zero cross-talk and zero
language-ID metadata leaks.

```
Existing client ─► POST /api/start
                ─► POST /api/chunk (f32 PCM) ─► {language, text}
                ─► POST /api/finish          ─► {language, text}
```

## Why this exists

Upstream state as of April 2026:

| Option | Status |
|--------|--------|
| `qwen-asr-demo-streaming` (Flask) | ⚠️ Single stream only. Documented limitation. Race on concurrent use. |
| `vllm serve` + `/v1/audio/transcriptions` | ✅ Multi-tenant safe but **batch-only** — no partial transcriptions. |
| `vllm serve` + `/v1/realtime` WebSocket | ⚠️ Known bugs: LID prefix leaks, 5 s segment boundary overlap, no rollback protocol. See [vllm#35767](https://github.com/vllm-project/vllm/issues/35767). |
| `vllm#35894` (community fix) | ⏳ Draft PR, 36 days inactive (as of 2026-04-23), architectural concerns in [vllm#35908](https://github.com/vllm-project/vllm/issues/35908). |

If you need **streaming transcription with partial results shown live, served
to multiple concurrent users, in production, today**, none of the above fit.

**qasr-mt** takes the SDK's battle-tested rolling-decode / prefix-rollback
streaming logic and wires it to `AsyncLLMEngine`, giving you the demo's
UX with production safety.

## Quickstart

```bash
git clone <repo-url>
cd qwen3-asr-mt
docker compose up -d           # first boot downloads the 0.6B model (~1.5 GB)
docker compose logs -f          # wait for "Application startup complete"

# smoke test (2 concurrent sessions, different audio slices)
python tests/smoke_concurrent.py http://localhost:8111 path/to/test.wav
```

Requires:

* NVIDIA GPU with CUDA 12.8+ driver (tested RTX 3090, should work on any Ampere+).
* NVIDIA Container Toolkit.
* Roughly 8–10 GB VRAM at `QASR_GPU_MEM_UTIL=0.35`, scale down for smaller cards.

## API

Identical surface to `qwen-asr-demo-streaming`. Audio body is raw float32
little-endian PCM at 16 kHz, mono, no WAV header.

### `POST /api/start`

Query params (optional): `language=Chinese|English|...` forces text-only output.

Response:
```json
{"session_id": "cb3a53d4bf1f42558b4fd2f65f3376b2"}
```

### `POST /api/chunk?session_id=<id>`

Header: `Content-Type: application/octet-stream`. Body: arbitrary-length
float32 PCM. Server buffers and runs a rolling decode when
`chunk_size_sec` worth of audio has accumulated.

Response: current accumulated result (not just the delta).
```json
{"language": "Chinese", "text": "這是 Qwen3-ASR 串流辨識的測試句"}
```

### `POST /api/finish?session_id=<id>`

Flushes any tail audio shorter than one chunk, runs a final decode, deletes
the session.

Response: same shape as `/api/chunk`, representing the final text.

### `GET /health`

Returns `{"ready": true, "sessions": 0}` once the engine finishes warmup.
Use as Docker/Kubernetes readiness probe.

### `GET /api/sessions`

Debugging aid: returns currently-live session IDs.

## Configuration

All knobs are environment variables (or CLI flags if running standalone):

| Variable | Default | Meaning |
|----------|---------|---------|
| `QASR_MODEL` | `Qwen/Qwen3-ASR-0.6B` | HF repo id or local path. |
| `QASR_CHUNK_SIZE_SEC` | `1.0` | Decode a new segment every N seconds of audio. |
| `QASR_UNFIXED_CHUNK_NUM` | `4` | First N chunks skip prefix conditioning. |
| `QASR_UNFIXED_TOKEN_NUM` | `5` | Roll back last K tokens each decode for boundary revision. |
| `QASR_GPU_MEM_UTIL` | `0.35` | vLLM GPU memory fraction; lower when sharing GPU. |
| `QASR_MAX_MODEL_LEN` | `8192` | Max context length; tune for very long audio. |
| `QASR_MAX_NEW_TOKENS` | `32` | Tokens generated per decode step. |
| `QASR_SESSION_TTL_SEC` | `600` | Idle session GC. |
| `QASR_DTYPE` | `auto` | `auto`/`bfloat16`/`float16`/`float32`. |
| `QASR_PORT` | `8000` | Listen port (container-internal). |

## Concurrency architecture

```
HTTP request ─► FastAPI handler (async)
                │
                ▼
        SESSIONS dict  ← asyncio.Lock (protects dict)
                │
                ▼
        per-session state  ← asyncio.Lock (serialises chunks in one session)
                │
                ▼
        AsyncLLMEngine.generate(request_id=sid:chunk_id)
                │
                ▼
        vLLM batch scheduler  (multi-tenant safe by design)
```

Two invariants make this work:

1. **Engine isolation** — every `engine.generate()` call uses a unique
   `request_id` (`{session}:c{chunk}`). vLLM's scheduler interleaves
   different sessions' requests in its own batch but keeps KV cache
   strictly per-request. This is the contract `AsyncLLMEngine` is designed
   around; no application-level serialisation is needed.
2. **State isolation** — each session owns its own `audio_accum`,
   `_raw_decoded`, chunk counter, and prefix buffer. A per-session
   `asyncio.Lock` prevents chunks from the same caller from re-entering
   the decode loop while one is still in flight.

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full discussion,
including why the `vllm.LLM` (offline) path inside the upstream demo is not
safe to share across threads.

## Benchmarks

Measured on a single RTX 3090 with `QASR_GPU_MEM_UTIL=0.35`:

| Scenario | Per-chunk latency | Cold-start |
|----------|------------------|------------|
| 1 concurrent session | 0.5 – 0.6 s | ~8 s (graph compile) |
| 2 concurrent sessions | 0.5 – 0.6 s | same |

vLLM's own benchmark claims up to 128 concurrent streams on the 0.6B model
with RTF 0.064; we have not yet pushed beyond 2.

## Comparison with upstream demo

|  | `qwen-asr-demo-streaming` | **qasr-mt** |
|---|---|---|
| Engine | `vllm.LLM` (offline API) | `vllm.AsyncLLMEngine` |
| Web framework | Flask (sync, `threaded=True`) | FastAPI (native async) |
| Concurrent streams | **Unsafe** (documented as single-stream) | Safe, tested |
| LID metadata leak to client | Observed on rolling-decode boundary | Stripped (`parse_asr_output` + regex) |
| `force_language` | Yes | Yes |
| Rolling decode (`unfixed_chunk_num`/`unfixed_token_num`) | Yes | Yes — logic ported verbatim |
| Timestamps | No | No |
| HTTP API shape | `/api/{start,chunk,finish}` | **Same** (drop-in) |

## Credits

* [**Qwen3-ASR**](https://github.com/QwenLM/Qwen3-ASR) by the Qwen team —
  the model and the SDK's rolling-decode algorithm this server reuses.
* [**vLLM**](https://github.com/vllm-project/vllm) — `AsyncLLMEngine`,
  chunked prefill, PagedAttention.
* vLLM community discussion [#35767](https://github.com/vllm-project/vllm/issues/35767)
  and [#35908](https://github.com/vllm-project/vllm/issues/35908) — clarified
  what the upstream realtime endpoint does and does not solve.

## License

[Apache License 2.0](LICENSE). Same licence as Qwen3-ASR and vLLM.
