"""Append-only decode trace -- a black box for retroactive diagnosis.

Intermittent streaming failures (the ~20% empty returns, >5 s slow decodes,
suspected truncation) can't be reproduced on demand. So we record every decode's
ground truth as it happens; when a bad result is later reported, the evidence is
already on disk and the cause is a query away.

Pure stdlib. A failed trace write must NEVER break recognition, so every call is
wrapped defensively -- on any error we silently drop the line.
"""

from __future__ import annotations

import json
import os
import threading

_LOCK = threading.Lock()
_dir_ready: set[str] = set()


def write_trace(path: str, record: dict) -> None:
    """Append ``record`` as one JSON line to ``path``. Never raises."""
    try:
        d = os.path.dirname(path)
        if d and d not in _dir_ready:
            os.makedirs(d, exist_ok=True)
            _dir_ready.add(d)
        line = json.dumps(record, ensure_ascii=False)
        with _LOCK:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        # Logging must never take down the ASR path.
        pass
