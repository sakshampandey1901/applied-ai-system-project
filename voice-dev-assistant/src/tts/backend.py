"""Text-to-speech: Piper CLI by default (pluggable)."""

from __future__ import annotations

import shutil
import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable


DEFAULT_PIPER_CMD = shutil.which("piper") or "piper"


def _minimal_silent_wav_bytes(duration_sec: float = 0.3, sample_rate: int = 16000) -> bytes:
    n = int(sample_rate * duration_sec)
    import io

    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(b"\x00\x00" * n)
    return buf.getvalue()


class TTSBackend(ABC):
    @abstractmethod
    def synth_to_wav(self, text: str, out_wav: Path) -> None:
        ...

    def say(self, text: str, out_wav: Path, play_audio: Callable[[Path], None]) -> None:
        self.synth_to_wav(text, out_wav)
        play_audio(out_wav)


class PiperTTS(TTSBackend):
    """Calls `piper` with --model; writes WAV via --output_file."""

    def __init__(
        self,
        model_path: Path | str | None = None,
        piper_cmd: str = DEFAULT_PIPER_CMD,
    ) -> None:
        import os

        mp = model_path or os.environ.get("ATLAS_PIPER_MODEL") or ""
        self.model_path = Path(mp) if mp else None
        self.piper_cmd = piper_cmd

    def synth_to_wav(self, text: str, out_wav: Path) -> None:
        if not text.strip():
            text = " "
        if not (self.model_path and self.model_path.is_file()):
            raise FileNotFoundError(
                "Piper voice model missing. Set ATLAS_PIPER_MODEL to the .onnx path."
            )
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.piper_cmd,
            "--model",
            str(self.model_path),
            "--output_file",
            str(out_wav),
        ]
        proc = subprocess.run(
            cmd,
            input=text.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if proc.returncode != 0:
            stderr = proc.stderr.decode("utf-8", errors="replace")
            raise RuntimeError(f"Piper failed ({proc.returncode}): {stderr}")


class SilentWavTTS(TTSBackend):
    """Quiet placeholder WAV for CI or runs without Piper model."""

    def synth_to_wav(self, text: str, out_wav: Path) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(_minimal_silent_wav_bytes())
