"""Microphone recording using sounddevice + NumPy float32 PCM."""

from __future__ import annotations

from pathlib import Path
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore[misc, assignment]


SAMPLE_RATE = 16000
CHANNELS = 1


def _require_sounddevice() -> None:
    if sd is None:
        raise RuntimeError(
            "sounddevice is required for audio capture. Install: pip install sounddevice numpy"
        )


def record_seconds(seconds: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Record mono float32 audio in [-1, 1], shape (frames,)."""
    _require_sounddevice()
    n = max(1, int(seconds * sample_rate))
    frames = sd.rec(n, samplerate=sample_rate, channels=CHANNELS, dtype="float32")
    sd.wait()
    return frames.reshape(-1)


def _rms(block: np.ndarray) -> float:
    if block.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(block))))


def record_until_silence(
    max_seconds: float = 30.0,
    sample_rate: int = SAMPLE_RATE,
    chunk_ms: float = 100.0,
    silence_ms: float = 800.0,
    silence_rms: float = 0.015,
    min_speech_ms: float = 300.0,
) -> np.ndarray:
    """Record until RMS stays below threshold for silence_ms (after optional initial speech)."""
    _require_sounddevice()
    chunk = max(1, int(sample_rate * chunk_ms / 1000))
    silence_chunks = max(1, int(silence_ms / chunk_ms))
    min_chunks = max(1, int(min_speech_ms / chunk_ms))

    chunks: list[np.ndarray] = []
    silent_run = 0
    spoken = False
    max_chunks = int(max_seconds / (chunk_ms / 1000))

    for _ in range(max_chunks):
        block = sd.rec(chunk, samplerate=sample_rate, channels=CHANNELS, dtype="float32")
        sd.wait()
        b = block.reshape(-1)
        chunks.append(b)
        level = _rms(b)
        if level > silence_rms:
            spoken = True
            silent_run = 0
        else:
            silent_run += 1
            if spoken and len(chunks) >= min_chunks and silent_run >= silence_chunks:
                break

    if not chunks:
        return np.array([], dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def write_wav_pcm16(path: Path | str, audio_float: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write WAV PCM16 mono (for ffmpeg/piper tooling). Standard library only."""
    import wave
    import struct

    path = Path(path)
    clipped = np.clip(audio_float.astype(np.float64), -1.0, 1.0)
    pcm16 = (clipped * 32767.0).astype(np.int16)

    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm16.tobytes())
