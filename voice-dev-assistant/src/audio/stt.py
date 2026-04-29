"""Local speech-to-text (faster-whisper or openai-whisper fallback)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


class WhisperSTT:
    """Load once; transcribe mono float32 16 kHz PCM in [-1, 1]."""

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device
        self._compute_type = compute_type
        self._fw = None
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel

            dev = self._device or "cpu"
            self._fw = WhisperModel(self._model_size, device=dev, compute_type=self._compute_type)
            self._backend = "faster-whisper"
        except Exception:
            import whisper as ow

            self._ow_model = ow.load_model(self._model_size)
            self._backend = "openai-whisper"
            self._fw = None

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        if getattr(self, "_backend", "") == "faster-whisper" and self._fw is not None:
            segments, _ = self._fw.transcribe(
                audio,
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            return " ".join(s.text.strip() for s in segments).strip()

        import whisper as ow

        audio_f32 = np.clip(audio.astype(np.float64), -1.0, 1.0)
        result = ow.transcribe(
            self._ow_model,
            audio_f32,
            fp16=False,
            language="en",
        )
        return (result.get("text") or "").strip()


def transcribe_fast(text_hint: bool = False) -> str:
    """Placeholder — use WhisperSTT instance instead."""
    del text_hint
    return ""
