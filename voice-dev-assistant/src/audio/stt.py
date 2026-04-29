"""Local speech-to-text (faster-whisper or openai-whisper fallback)."""

from __future__ import annotations

from typing import Any

import numpy as np


class WhisperSTT:
    """Load once; transcribe mono float32 16 kHz PCM in [-1, 1]."""

    def __init__(
        self,
        model_size: str = "base",
        device: str | None = None,
        compute_type: str = "int8",
    ) -> None:
        self._model_size = model_size
        self._device = device or "cpu"
        self._compute_type = compute_type
        self._backend = ""
        self._fw: Any = None
        self._ow: Any = None
        self._load()

    def _load(self) -> None:
        try:
            from faster_whisper import WhisperModel

            self._fw = WhisperModel(
                self._model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
            self._backend = "faster-whisper"
        except Exception:
            import whisper as ow

            self._ow = ow.load_model(self._model_size)
            self._backend = "openai-whisper"

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        if audio.size == 0:
            return ""
        if self._backend == "faster-whisper" and self._fw is not None:
            seg_iter, _info = self._fw.transcribe(
                audio.astype(np.float32),
                beam_size=5,
                language="en",
                vad_filter=True,
            )
            return " ".join(s.text.strip() for s in seg_iter).strip()

        if self._ow is None:
            raise RuntimeError("Whisper backend not initialized")
        af = np.clip(audio.astype(np.float64), -1.0, 1.0).astype(np.float32)
        result = self._ow.transcribe(af, fp16=False, language="en")
        return (result.get("text") or "").strip()
