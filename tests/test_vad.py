"""Unit tests for the energy VAD. Pure numpy -- no model, no GPU, runs anywhere."""

import numpy as np

from qasr_server.vad import has_speech

SR = 16000
THR = 0.01


def _silence(sec: float) -> np.ndarray:
    return np.zeros(int(sec * SR), dtype=np.float32)


def _tone(sec: float, amp: float = 0.1, freq: float = 220.0) -> np.ndarray:
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_pure_silence_is_not_speech():
    assert has_speech(_silence(1.0), THR, SR) is False


def test_normal_tone_is_speech():
    assert has_speech(_tone(0.5, amp=0.1), THR, SR) is True


def test_quiet_burst_amid_silence_is_speech():
    # 0.1 s voiced burst between long silences must still register (frame-level).
    audio = np.concatenate([_silence(0.9), _tone(0.1, amp=0.08), _silence(0.6)])
    assert has_speech(audio, THR, SR) is True


def test_below_threshold_tone_is_not_speech():
    # Mic noise floor level -> below gate.
    assert has_speech(_tone(0.5, amp=0.002), THR, SR) is False


def test_empty_is_not_speech():
    assert has_speech(np.zeros((0,), dtype=np.float32), THR, SR) is False


def test_sub_frame_input_handled():
    # Shorter than one 30 ms frame: must not crash; silence -> False.
    assert has_speech(_silence(0.005), THR, SR) is False
    # loud sub-frame click -> True
    assert has_speech(np.full(80, 0.2, dtype=np.float32), THR, SR) is True


def test_threshold_is_respected():
    tone = _tone(0.5, amp=0.05)
    assert has_speech(tone, 0.02, SR) is True   # below the 0.05 RMS-ish energy
    assert has_speech(tone, 0.5, SR) is False    # absurdly high gate -> nothing passes
