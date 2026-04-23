"""
qasr-mt — Multi-tenant streaming ASR server for Qwen3-ASR.

Drop-in replacement for ``qwen-asr-demo-streaming`` (the single-stream Flask
demo shipped with ``qwen-asr``). The demo is explicitly documented as
``Single stream only (no batching)`` and is **unsafe** under concurrent use:
raw LID metadata leaks, tokenizer races, and overlapping generations can
mix one user's transcription into another's response.

This server replaces the demo with the production vLLM path:

* Each HTTP session owns an isolated streaming state (audio buffer,
  accumulated audio, rolling-decode prefix tokens).
* A single ``vllm.AsyncLLMEngine`` is shared across sessions; vLLM's
  internal batch scheduler handles concurrency safely, one ``request_id``
  per call.
* A per-session ``asyncio.Lock`` guarantees chunks from the same caller
  do not interleave.
* ``parse_asr_output`` is invoked before every response, with a
  belt-and-suspenders regex that strips any residual ``language X<asr_text>``
  metadata even on malformed edge-case outputs.

The HTTP surface is intentionally identical to ``qwen_asr.cli.demo_streaming``:
existing clients can switch by changing only the endpoint URL.

    POST /api/start                          -> {"session_id": "..."}
    POST /api/chunk?session_id=...  (f32 PCM) -> {"language": "...", "text": "..."}
    POST /api/finish?session_id=...          -> {"language": "...", "text": "..."}

Audio format for /api/chunk body: mono 16 kHz PCM, float32 little-endian, no header.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import sys
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams

from qwen_asr.inference.qwen3_asr import (
    SAMPLE_RATE,
    Qwen3ASRModel,
    Qwen3ASRProcessor,
    parse_asr_output,
)

log = logging.getLogger("qasr")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class Config:
    model: str = os.environ.get("QASR_MODEL", "Qwen/Qwen3-ASR-0.6B")
    host: str = os.environ.get("QASR_HOST", "0.0.0.0")
    port: int = int(os.environ.get("QASR_PORT", "8000"))
    chunk_size_sec: float = float(os.environ.get("QASR_CHUNK_SIZE_SEC", "1.0"))
    unfixed_chunk_num: int = int(os.environ.get("QASR_UNFIXED_CHUNK_NUM", "4"))
    unfixed_token_num: int = int(os.environ.get("QASR_UNFIXED_TOKEN_NUM", "5"))
    max_new_tokens: int = int(os.environ.get("QASR_MAX_NEW_TOKENS", "32"))
    gpu_memory_utilization: float = float(os.environ.get("QASR_GPU_MEM_UTIL", "0.35"))
    max_model_len: int = int(os.environ.get("QASR_MAX_MODEL_LEN", "8192"))
    session_ttl_sec: int = int(os.environ.get("QASR_SESSION_TTL_SEC", "600"))
    dtype: str = os.environ.get("QASR_DTYPE", "auto")


CFG = Config()


# ---------------------------------------------------------------------------
# Streaming state (per session)
# ---------------------------------------------------------------------------
@dataclass
class StreamingState:
    session_id: str
    chunk_size_samples: int
    unfixed_chunk_num: int
    unfixed_token_num: int
    chunk_id: int = 0
    buffer: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    audio_accum: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    prompt_raw: str = ""
    force_language: Optional[str] = None
    language: str = ""
    text: str = ""
    _raw_decoded: str = ""
    last_seen: float = field(default_factory=time.time)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


# ---------------------------------------------------------------------------
# Globals (set up in lifespan)
# ---------------------------------------------------------------------------
engine: Optional[AsyncLLMEngine] = None
processor: Optional[Qwen3ASRProcessor] = None
prompt_stub: Optional[Qwen3ASRModel] = None  # only used for _build_text_prompt
sampling_params: Optional[SamplingParams] = None

SESSIONS: Dict[str, StreamingState] = {}
SESSIONS_LOCK = asyncio.Lock()


# Belt-and-suspenders: even if parse_asr_output misses a malformed case,
# peel off any residual Qwen3-ASR metadata delimiter before sending to the client.
_LID_LEAK_RE = re.compile(r"(?:^|\s)language\s+[A-Za-z\-]+<asr_text>")


def _sanitize(text: str) -> str:
    if not text:
        return ""
    t = text
    while True:
        m = _LID_LEAK_RE.search(t)
        if not m:
            break
        t = t[m.end():]
    if t.startswith("<asr_text>"):
        t = t[len("<asr_text>"):]
    return t.lstrip()


def _gc_sessions_locked() -> int:
    now = time.time()
    dead = [sid for sid, s in SESSIONS.items() if now - s.last_seen > CFG.session_ttl_sec]
    for sid in dead:
        SESSIONS.pop(sid, None)
    if dead:
        log.info("gc: purged %d idle sessions", len(dead))
    return len(dead)


async def _run_decode(state: StreamingState) -> None:
    """One rolling-decode step. Caller must hold ``state.lock``.

    Ports Qwen3ASRModel.streaming_transcribe's rolling-window / prefix-rollback
    logic to the AsyncLLMEngine generate() API.
    """
    prefix = ""
    if state.chunk_id >= state.unfixed_chunk_num and state._raw_decoded:
        cur_ids = processor.tokenizer.encode(state._raw_decoded)
        end_idx = max(1, len(cur_ids) - state.unfixed_token_num)
        prefix = processor.tokenizer.decode(cur_ids[:end_idx])

    prompt_text = state.prompt_raw + prefix
    inp = {"prompt": prompt_text, "multi_modal_data": {"audio": [state.audio_accum]}}

    request_id = f"{state.session_id}:c{state.chunk_id}"
    final = None
    async for out in engine.generate(
        prompt=inp,
        sampling_params=sampling_params,
        request_id=request_id,
    ):
        final = out

    gen_text = final.outputs[0].text if (final and final.outputs) else ""
    state._raw_decoded = prefix + gen_text

    lang, txt = parse_asr_output(state._raw_decoded, user_language=state.force_language)
    state.language = lang or state.language
    state.text = _sanitize(txt)
    state.chunk_id += 1


# ---------------------------------------------------------------------------
# FastAPI app + lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine, processor, prompt_stub, sampling_params
    log.info("loading model=%s  gpu_mem=%.2f  max_len=%d",
             CFG.model, CFG.gpu_memory_utilization, CFG.max_model_len)

    engine_args = AsyncEngineArgs(
        model=CFG.model,
        gpu_memory_utilization=CFG.gpu_memory_utilization,
        max_model_len=CFG.max_model_len,
        limit_mm_per_prompt={"audio": 1},
        dtype=CFG.dtype,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    processor = Qwen3ASRProcessor.from_pretrained(CFG.model, fix_mistral_regex=True)

    # Build a minimal Qwen3ASRModel just for its _build_text_prompt() helper
    # (it uses only ``self.processor``; we avoid loading a second model copy).
    prompt_stub = Qwen3ASRModel.__new__(Qwen3ASRModel)
    prompt_stub.backend = "vllm"
    prompt_stub.processor = processor

    sampling_params = SamplingParams(temperature=0.0, max_tokens=CFG.max_new_tokens)
    log.info("ready: chunk=%.2fs unfixed_chunk=%d unfixed_token=%d",
             CFG.chunk_size_sec, CFG.unfixed_chunk_num, CFG.unfixed_token_num)

    yield

    log.info("shutdown: active_sessions=%d", len(SESSIONS))
    SESSIONS.clear()


app = FastAPI(
    title="qasr-mt",
    description="Multi-tenant streaming ASR server for Qwen3-ASR (vLLM AsyncLLMEngine).",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "service": "qasr-mt",
        "model": CFG.model,
        "sessions": len(SESSIONS),
        "chunk_size_sec": CFG.chunk_size_sec,
    }


@app.get("/health")
async def health():
    ready = engine is not None and processor is not None
    return {"ready": ready, "sessions": len(SESSIONS)}


@app.get("/api/sessions")
async def api_sessions():
    async with SESSIONS_LOCK:
        _gc_sessions_locked()
        return {"count": len(SESSIONS), "ids": list(SESSIONS.keys())}


@app.post("/api/start")
async def api_start(language: Optional[str] = None):
    """Create a new streaming session.

    Query params:
      language: optional forced language (e.g. ``Chinese``, ``English``).
                If provided, passed to Qwen3-ASR to force text-only output.
    """
    session_id = uuid.uuid4().hex
    prompt_raw = prompt_stub._build_text_prompt(context="", force_language=language)
    chunk_samples = int(round(CFG.chunk_size_sec * SAMPLE_RATE))
    state = StreamingState(
        session_id=session_id,
        chunk_size_samples=chunk_samples,
        unfixed_chunk_num=CFG.unfixed_chunk_num,
        unfixed_token_num=CFG.unfixed_token_num,
        prompt_raw=prompt_raw,
        force_language=language,
    )
    async with SESSIONS_LOCK:
        _gc_sessions_locked()
        SESSIONS[session_id] = state
    log.debug("start: sid=%s lang=%s", session_id[:8], language)
    return {"session_id": session_id}


@app.post("/api/chunk")
async def api_chunk(session_id: str, request: Request):
    """Append audio chunk to the session; run rolling decode if enough buffered.

    Body: raw float32 little-endian PCM at 16 kHz (mono). Any length accepted.
    Response: current accumulated {language, text}.
    """
    if request.headers.get("content-type", "").split(";")[0].strip() \
            not in ("application/octet-stream", ""):
        raise HTTPException(415, "expect application/octet-stream (f32 PCM)")

    async with SESSIONS_LOCK:
        state = SESSIONS.get(session_id)
    if state is None:
        raise HTTPException(400, "invalid session_id")
    state.last_seen = time.time()

    body = await request.body()
    if len(body) % 4 != 0:
        raise HTTPException(400, "float32 bytes length not multiple of 4")
    wav = np.frombuffer(body, dtype=np.float32).reshape(-1)

    async with state.lock:
        if wav.size > 0:
            state.buffer = (
                np.concatenate([state.buffer, wav]) if state.buffer.size else wav.copy()
            )
        while state.buffer.shape[0] >= state.chunk_size_samples:
            chunk = state.buffer[: state.chunk_size_samples]
            state.buffer = state.buffer[state.chunk_size_samples :]
            state.audio_accum = (
                np.concatenate([state.audio_accum, chunk]) if state.audio_accum.size else chunk.copy()
            )
            await _run_decode(state)
    return {"language": state.language, "text": state.text}


@app.post("/api/finish")
async def api_finish(session_id: str):
    """Flush tail audio (shorter than one chunk) and finalise the session."""
    async with SESSIONS_LOCK:
        state = SESSIONS.pop(session_id, None)
    if state is None:
        raise HTTPException(400, "invalid session_id")

    async with state.lock:
        if state.buffer.size > 0:
            tail = state.buffer
            state.buffer = np.zeros((0,), dtype=np.float32)
            state.audio_accum = (
                np.concatenate([state.audio_accum, tail]) if state.audio_accum.size else tail
            )
            await _run_decode(state)
    log.debug("finish: sid=%s text_len=%d", session_id[:8], len(state.text))
    return {"language": state.language, "text": state.text}


# ---------------------------------------------------------------------------
# Entry point (standalone; compose/Docker use uvicorn directly)
# ---------------------------------------------------------------------------
def _parse_args(argv):
    p = argparse.ArgumentParser(prog="qasr-mt")
    p.add_argument("--host", default=CFG.host)
    p.add_argument("--port", type=int, default=CFG.port)
    p.add_argument("--model", default=CFG.model)
    p.add_argument("--chunk-size-sec", type=float, default=CFG.chunk_size_sec)
    p.add_argument("--unfixed-chunk-num", type=int, default=CFG.unfixed_chunk_num)
    p.add_argument("--unfixed-token-num", type=int, default=CFG.unfixed_token_num)
    p.add_argument("--max-new-tokens", type=int, default=CFG.max_new_tokens)
    p.add_argument("--gpu-memory-utilization", type=float, default=CFG.gpu_memory_utilization)
    p.add_argument("--max-model-len", type=int, default=CFG.max_model_len)
    p.add_argument("--session-ttl-sec", type=int, default=CFG.session_ttl_sec)
    p.add_argument("--dtype", default=CFG.dtype)
    p.add_argument("--log-level", default="info")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # Push CLI overrides into env so config picks them up consistently
    for k, v in vars(args).items():
        if k in ("host", "port", "log_level"):
            continue
        env_key = "QASR_" + k.upper().replace("-", "_")
        os.environ[env_key] = str(v)
    # Rebuild CFG from env now
    global CFG
    CFG = Config()

    import uvicorn
    uvicorn.run(
        "qasr_server.server:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
