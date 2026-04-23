#!/usr/bin/env python3
"""
Minimal async Python client for qasr-mt.

Reads a WAV file, resamples to 16 kHz mono float32 if needed, streams it to
the server in real-time-ish chunks, and prints partial transcriptions.

Usage:
    python python_client.py <wav-file> [base-url] [chunk-sec]

Dependencies:
    pip install aiohttp numpy
"""
from __future__ import annotations

import asyncio
import sys
import time
import wave

import aiohttp
import numpy as np

TARGET_SR = 16000


def load_pcm(path: str) -> np.ndarray:
    """Load a WAV file, return float32 mono samples at 16 kHz."""
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


async def transcribe(base: str, audio: np.ndarray, chunk_sec: float = 0.5) -> str:
    chunk_samples = int(chunk_sec * TARGET_SR)
    async with aiohttp.ClientSession() as s:
        # open session
        async with s.post(f"{base}/api/start") as r:
            sid = (await r.json())["session_id"]
        print(f"[session] {sid[:8]}... opened")

        # stream chunks
        pos = 0
        t0 = time.time()
        while pos < audio.size:
            chunk = audio[pos: pos + chunk_samples].astype(np.float32)
            pos += chunk_samples
            async with s.post(
                f"{base}/api/chunk?session_id={sid}",
                data=chunk.tobytes(),
                headers={"Content-Type": "application/octet-stream"},
            ) as r:
                partial = (await r.json()).get("text", "")
            print(f"[+{time.time()-t0:5.2f}s] {partial[:100]!r}")
            await asyncio.sleep(chunk_sec)  # simulate real-time arrival

        # finalise
        async with s.post(f"{base}/api/finish?session_id={sid}") as r:
            final = await r.json()
        print(f"[final] language={final.get('language')!r}")
        print(f"[final] text={final.get('text')!r}")
        return final.get("text", "")


def main():
    if len(sys.argv) < 2:
        print("usage: python_client.py <wav-file> [base-url] [chunk-sec]", file=sys.stderr)
        sys.exit(2)
    wav = sys.argv[1]
    base = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8111"
    chunk_sec = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5

    audio = load_pcm(wav)
    print(f"[audio] {wav} {audio.size/TARGET_SR:.2f}s @ 16kHz mono f32")
    asyncio.run(transcribe(base, audio, chunk_sec))


if __name__ == "__main__":
    main()
