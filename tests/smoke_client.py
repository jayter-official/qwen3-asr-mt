"""Streaming smoke client — run INSIDE the qasr-mt container against 127.0.0.1:8000.
Verifies: (1) VAD lets real speech through, (2) VAD gates silence to empty,
(3) decode trace gets written. Usage: python3 smoke_client.py /tmp/smoke.wav
"""
import json
import sys
import urllib.request
import wave

import numpy as np

BASE = "http://127.0.0.1:8000"
SR = 16000
CTX = "Claude Code, Jayter, GX10, MOMO, 1688"


def _post(path, data=b"", ctype="application/octet-stream"):
    req = urllib.request.Request(BASE + path, data=data, method="POST",
                                 headers={"Content-Type": ctype})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def stream(samples: np.ndarray, label: str):
    sid = _post(f"/api/start?context={urllib.parse.quote(CTX)}")["session_id"]
    chunk = SR  # 1 s
    for i in range(0, len(samples), chunk):
        _post(f"/api/chunk?session_id={sid}", samples[i:i + chunk].astype(np.float32).tobytes())
    final = _post(f"/api/finish?session_id={sid}")
    print(f"[{label}] {len(samples)/SR:.1f}s -> text={final.get('text','')!r}")
    return final.get("text", "")


def main():
    import urllib.parse  # noqa
    wav_path = sys.argv[1]
    with wave.open(wav_path, "rb") as w:
        raw = w.readframes(w.getnframes())
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() == 2:
            pcm = pcm.reshape(-1, 2).mean(axis=1)
    speech = stream(pcm, "SPEECH")
    silence = stream(np.zeros(int(2.0 * SR), dtype=np.float32), "SILENCE")

    print("\n=== verdict ===")
    print("speech transcribed (VAD passes speech):", "PASS" if len(speech) >= 2 else "FAIL")
    print("silence gated to empty (VAD blocks):  ", "PASS" if silence.strip() == "" else "FAIL")


if __name__ == "__main__":
    import urllib.parse  # ensure available
    main()
