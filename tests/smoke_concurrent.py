#!/usr/bin/env python3
"""
Concurrent multi-tenant smoke test.

Spins up two parallel sessions against the server, each streaming a
different slice of the supplied WAV file, and asserts:

  * Both sessions return a non-empty final transcription.
  * Neither response contains the raw `language X<asr_text>` metadata.
  * Session A's text does not appear inside session B's, and vice versa.
    (This catches cross-session state leakage.)

Usage:
    python smoke_concurrent.py <wav-file> [base-url]

Exit codes:
    0  all assertions pass
    1  an assertion failed (output includes diagnostics)
    2  bad arguments
"""
from __future__ import annotations

import asyncio
import re
import sys
import time
import wave

import aiohttp
import numpy as np

TARGET_SR = 16000
CHUNK_SEC = 0.5
LID_LEAK_RE = re.compile(r"language\s+[A-Za-z\-]+<asr_text>")


def load_pcm(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1:
        x = x.reshape(-1, nch).mean(axis=1)
    if sr != TARGET_SR:
        new_n = int(round(x.size * TARGET_SR / sr))
        idx = np.linspace(0, x.size - 1, new_n)
        x = np.interp(idx, np.arange(x.size), x).astype(np.float32)
    return x


async def run_session(tag: str, base: str, audio: np.ndarray,
                      start_sec: float, end_sec: float) -> dict:
    segment = audio[int(start_sec * TARGET_SR): int(end_sec * TARGET_SR)]
    chunk_samples = int(CHUNK_SEC * TARGET_SR)

    async with aiohttp.ClientSession() as s:
        async with s.post(f"{base}/api/start") as r:
            sid = (await r.json())["session_id"]
        print(f"[{tag}] sid={sid[:8]} slice=[{start_sec:.1f},{end_sec:.1f}]s")

        pos = 0
        t0 = time.time()
        last_partial = ""
        while pos < segment.size:
            chunk = segment[pos: pos + chunk_samples].astype(np.float32)
            pos += chunk_samples
            async with s.post(
                f"{base}/api/chunk?session_id={sid}",
                data=chunk.tobytes(),
                headers={"Content-Type": "application/octet-stream"},
            ) as r:
                j = await r.json()
            partial = j.get("text", "")
            if partial != last_partial:
                print(f"[{tag}] +{time.time()-t0:5.2f}s  {partial[:80]!r}")
                last_partial = partial
            await asyncio.sleep(CHUNK_SEC)

        async with s.post(f"{base}/api/finish?session_id={sid}") as r:
            final = await r.json()
        print(f"[{tag}] FINAL: {final.get('text','')!r}")
        return final


async def main():
    if len(sys.argv) < 2:
        print("usage: smoke_concurrent.py <wav-file> [base-url]", file=sys.stderr)
        sys.exit(2)
    wav = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8111"

    audio = load_pcm(wav)
    dur = audio.size / TARGET_SR
    if dur < 15:
        print(f"[warn] audio is only {dur:.1f}s; slices will overlap",
              file=sys.stderr)

    a = run_session("A", base, audio, 0, min(10, dur))
    b = run_session("B", base, audio, max(0, dur - 10), dur)
    results = await asyncio.gather(a, b, return_exceptions=True)

    failures: list[str] = []
    for tag, res in zip(("A", "B"), results):
        if isinstance(res, Exception):
            failures.append(f"session {tag} errored: {res!r}")
            continue
        text = res.get("text", "")
        if not text:
            failures.append(f"session {tag} returned empty text")
        if LID_LEAK_RE.search(text):
            failures.append(f"session {tag} leaked LID metadata: {text!r}")

    if all(not isinstance(r, Exception) for r in results):
        a_text = results[0].get("text", "")
        b_text = results[1].get("text", "")
        # Only check non-trivial overlap (10 characters is arbitrary but
        # catches obvious cross-contamination without triggering on common
        # short words).
        if a_text and b_text and len(a_text) > 10 and a_text in b_text:
            failures.append(f"A leaked into B: A={a_text!r} ⊂ B={b_text!r}")
        if a_text and b_text and len(b_text) > 10 and b_text in a_text:
            failures.append(f"B leaked into A: B={b_text!r} ⊂ A={a_text!r}")

    print("\n" + "=" * 60)
    if failures:
        print("FAIL:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: both sessions transcribed cleanly, no leaks.")


if __name__ == "__main__":
    asyncio.run(main())
