"""Play WAV/audio via OS tools."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def resolve_player() -> str | None:
    for cand in ("afplay", "ffplay"):
        path = shutil.which(cand)
        if path:
            return path
    return shutil.which("aplay")


def play_wav(path: Path) -> None:
    """Play synchronously; raises if no player."""
    player = resolve_player()
    if not player:
        raise RuntimeError("No audio player found (install afplay on macOS, or ffplay from ffmpeg)")
    path = Path(path)
    if player.endswith("ffplay"):
        subprocess.run([player, "-nodisp", "-autoexit", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        subprocess.run([player, str(path)], check=False)


def announce_text(text: str) -> None:
    """Stdout for visibility when TTS unavailable."""
    print(text, flush=True)
