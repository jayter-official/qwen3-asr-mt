"""Zero-dependency energy VAD (voice-activity detection).

First-principles rationale
--------------------------
Qwen3-ASR biases recognition by *prepending the hotword/brand context as a text
prompt*.  When ``_run_decode`` fires on a chunk that contains little or no speech
(the first rolling window almost always does), the model has no acoustic evidence
to transcribe, so greedy decode (temperature=0) falls back to continuing the most
salient tokens it just saw -- the context terms themselves.  The client then sees
``"Claude Code, Jayter, GX10, ..."`` as a phantom transcription.

The robust fix is not to filter that output after the fact, but to make decoding
*evidence-driven instead of clock-driven*: never ask the model to transcribe a
window that has no voiced speech.  This module provides the speech test.

We deliberately use a frame-level energy threshold rather than the mean RMS of the
whole clip: a short voiced burst surrounded by silence must still register, and a
long silent lead-in must not dilute it below threshold.  No ML model, no extra
dependency -- just numpy -- so it adds zero cold-start / memory cost to the shared
GPU service and cannot itself fail to load.
"""

from __future__ import annotations

import numpy as np

# 30 ms analysis frame: long enough for a stable energy estimate of a voiced
# phone, short enough to catch a brief burst.
FRAME_SEC: float = 0.03


def has_speech(audio: np.ndarray, rms_threshold: float, sample_rate: int) -> bool:
    """Return True if *any* ~30 ms frame of ``audio`` has RMS >= ``rms_threshold``.

    Parameters
    ----------
    audio : float32/float64 mono PCM in [-1, 1].
    rms_threshold : per-frame RMS gate. Voiced speech frames are typically
        0.02-0.2; silence / mic noise floor is usually < 0.005.
    sample_rate : Hz (e.g. 16000).

    Notes
    -----
    * Empty / sub-frame input -> evaluated as a single frame; all-zero -> False.
    * Frame-localised (max over frames), so leading silence never dilutes a real
      voiced burst below threshold.
    """
    if audio is None or audio.size == 0:
        return False

    n = audio.size
    frame = int(round(FRAME_SEC * sample_rate))
    if frame <= 0:
        frame = n
    # If shorter than one frame, evaluate the whole buffer as one frame.
    if n < frame:
        frame = n

    usable = (n // frame) * frame
    if usable == 0:
        return False

    frames = audio[:usable].astype(np.float64).reshape(-1, frame)
    frame_rms = np.sqrt(np.mean(frames * frames, axis=1))
    return bool(np.max(frame_rms) >= rms_threshold)
