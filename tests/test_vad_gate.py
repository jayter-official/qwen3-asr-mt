"""Integration tests for the evidence-driven decode gate in ``_run_decode``.

Heavy deps (vllm / qwen_asr / fastapi) are stubbed in ``sys.modules`` *before*
importing the server, and the LLM engine is replaced with a fake async generator.
This exercises the real gate wiring + rolling-decode plumbing with NO GPU, so it
runs anywhere -- the whole point being to verify the fix without touching 104.
"""

import asyncio
import os
import sys
import types

import numpy as np

SR = 16000


# --------------------------------------------------------------------------- #
# Stub the import-time heavy dependencies so ``qasr_server.server`` imports.
# --------------------------------------------------------------------------- #
def _install_stubs() -> None:
    vllm = types.ModuleType("vllm")
    vllm.AsyncEngineArgs = type("AsyncEngineArgs", (), {})
    vllm.AsyncLLMEngine = type("AsyncLLMEngine", (), {})
    vllm.SamplingParams = type("SamplingParams", (), {"__init__": lambda self, **k: None})
    sys.modules["vllm"] = vllm

    qa = types.ModuleType("qwen_asr")
    qai = types.ModuleType("qwen_asr.inference")
    qaiq = types.ModuleType("qwen_asr.inference.qwen3_asr")
    qaiq.SAMPLE_RATE = SR
    qaiq.Qwen3ASRModel = type("Qwen3ASRModel", (), {})
    qaiq.Qwen3ASRProcessor = type("Qwen3ASRProcessor", (), {})
    qaiq.parse_asr_output = lambda raw, user_language=None: ("Chinese", raw)
    qa.inference = qai
    qai.qwen3_asr = qaiq
    sys.modules["qwen_asr"] = qa
    sys.modules["qwen_asr.inference"] = qai
    sys.modules["qwen_asr.inference.qwen3_asr"] = qaiq

    fa = types.ModuleType("fastapi")

    class _App:
        def __init__(self, *a, **k):
            pass

        def _deco(self, *a, **k):
            def wrap(fn):
                return fn
            return wrap

        post = _deco
        get = _deco

    class _HTTPException(Exception):
        def __init__(self, status_code=400, detail=""):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    fa.FastAPI = _App
    fa.HTTPException = _HTTPException
    fa.Request = type("Request", (), {})
    sys.modules["fastapi"] = fa

    sys.modules.setdefault("uvicorn", types.ModuleType("uvicorn"))


_install_stubs()
from qasr_server import server  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes for the LLM engine
# --------------------------------------------------------------------------- #
class _Out:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text)]


class _FakeEngine:
    def __init__(self, text):
        self._text = text
        self.calls = 0

    async def generate(self, prompt=None, sampling_params=None, request_id=None):
        self.calls += 1
        yield _Out(self._text)


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def _tone(sec: float, amp: float = 0.1, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _fresh(audio: np.ndarray, vad_enabled: bool = True):
    """Wire globals + return a StreamingState carrying ``audio`` as accum."""
    server.engine = _FakeEngine("這是123。")
    server.sampling_params = object()
    server.processor = None  # unused for chunk_id < unfixed_chunk_num
    server.CFG.vad_enabled = vad_enabled
    server.CFG.vad_rms_threshold = 0.01
    server.CFG.vad_lookback_sec = 0.5
    server.CFG.trace_enabled = False  # off by default; trace tests enable explicitly
    st = server.StreamingState(
        session_id="t",
        chunk_size_samples=SR,
        unfixed_chunk_num=4,
        unfixed_token_num=5,
        prompt_raw="PROMPT",
    )
    st.audio_accum = audio
    return st


def _run(coro):
    return asyncio.run(coro)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_silence_is_gated_no_engine_call():
    st = _fresh(_silence(1.0))
    _run(server._run_decode(st))
    assert server.engine.calls == 0       # model never invoked
    assert st.speech_started is False
    assert st.text == ""                  # no phantom echo
    assert st.chunk_id == 0               # gated chunks don't advance the window


def test_silence_accum_trimmed_to_lookback():
    st = _fresh(_silence(2.0))            # 2 s silence
    _run(server._run_decode(st))
    # bounded to ~0.5 s lookback, not unbounded growth
    assert st.audio_accum.size <= int(0.5 * SR)
    assert st.audio_accum.size > 0        # but onset lookback retained


def test_speech_opens_gate_and_decodes():
    st = _fresh(_tone(0.6, amp=0.1))
    _run(server._run_decode(st))
    assert server.engine.calls == 1
    assert st.speech_started is True
    assert st.text == "這是123。"
    assert st.chunk_id == 1


def test_latch_stays_open_for_trailing_silence():
    st = _fresh(_tone(0.6, amp=0.1))
    _run(server._run_decode(st))                       # speech opens gate
    assert st.speech_started and st.chunk_id == 1
    st.audio_accum = np.concatenate([st.audio_accum, _silence(1.0)])
    _run(server._run_decode(st))                       # quiet tail still decodes
    assert server.engine.calls == 2
    assert st.chunk_id == 2                             # not re-gated mid-utterance


def test_kill_switch_disables_gate():
    st = _fresh(_silence(1.0), vad_enabled=False)
    _run(server._run_decode(st))
    assert server.engine.calls == 1                    # decodes silence when VAD off
    assert st.chunk_id == 1


# --- black-box trace ------------------------------------------------------- #
import json  # noqa: E402


def test_trace_written_on_real_decode(tmp_path):
    tp = str(tmp_path / "decode_trace.jsonl")
    st = _fresh(_tone(0.6, amp=0.1))
    server.CFG.trace_enabled = True
    server.CFG.trace_path = tp
    _run(server._run_decode(st))
    rec = json.loads(open(tp, encoding="utf-8").read().strip().split("\n")[-1])
    assert rec["vad_gated"] is False
    assert rec["empty"] is False
    assert rec["parsed"] == "這是123。"
    assert rec["chunk_id"] == 0
    assert "decode_ms" in rec and "active_sessions" in rec and "audio_sec" in rec


def test_trace_written_on_vad_gate(tmp_path):
    tp = str(tmp_path / "decode_trace.jsonl")
    st = _fresh(_silence(1.0))
    server.CFG.trace_enabled = True
    server.CFG.trace_path = tp
    _run(server._run_decode(st))
    rec = json.loads(open(tp, encoding="utf-8").read().strip().split("\n")[-1])
    assert rec["vad_gated"] is True
    assert rec["empty"] is True
    assert rec["speech_started"] is False


def test_trace_disabled_writes_nothing(tmp_path):
    tp = str(tmp_path / "decode_trace.jsonl")
    st = _fresh(_tone(0.6, amp=0.1))
    server.CFG.trace_enabled = False
    server.CFG.trace_path = tp
    _run(server._run_decode(st))
    assert not os.path.exists(tp)
