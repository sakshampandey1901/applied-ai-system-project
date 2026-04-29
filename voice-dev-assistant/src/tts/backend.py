"""Text-to-speech: Piper Python API (preferred) or Piper CLI fallback."""

from __future__ import annotations

import shutil
import subprocess
import wave
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

DEFAULT_PIPER_CMD = shutil.which("piper") or "piper"


def _load_piper_voice_class():
    try:
        from piper.voice import PiperVoice as PV
        return PV
    except ImportError:
        pass
    try:
        from piper import PiperVoice as PV
        return PV
    except ImportError as e:
        raise ImportError(
            "Piper Python API not installed. Run: pip install piper-tts"
        ) from e


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


class PiperVoiceTTS(TTSBackend):
    """Piper ONNX via bundled Python bindings (`pip install piper-tts`)."""

    __slots__ = ("_voice",)

    def __init__(self, model_path: Path | str) -> None:
        mp = Path(model_path)
        if not mp.is_file():
            raise FileNotFoundError(f"Piper voice model not found: {mp}")
        pv = _load_piper_voice_class()
        self._voice = pv.load(str(mp))

    def synth_to_wav(self, text: str, out_wav: Path) -> None:
        t = text.strip() or " "
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(out_wav), "wb") as wav_file:
            if hasattr(self._voice, "synthesize_wav"):
                self._voice.synthesize_wav(t, wav_file)
            else:
                self._voice.synthesize(t, wav_file)


class PiperCliTTS(TTSBackend):
    """Calls `piper` executable with --model (no Python piper-tts package)."""

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


# Back-compat name used in older docs
PiperTTS = PiperCliTTS


class SilentWavTTS(TTSBackend):
    """Quiet placeholder WAV for CI or runs without Piper model."""

    def synth_to_wav(self, text: str, out_wav: Path) -> None:
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(_minimal_silent_wav_bytes())


def make_piper_backend(model_path: Path) -> TTSBackend:
    """Prefer Python PiperVoice; fall back to `piper` CLI."""
    try:
        return PiperVoiceTTS(model_path)
    except ImportError:
        pass
    if shutil.which("piper"):
        return PiperCliTTS(model_path=model_path)
    raise ImportError(
        "Install Piper: pip install piper-tts  OR put the `piper` CLI on PATH"
    )
