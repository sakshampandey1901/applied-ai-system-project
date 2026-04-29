#!/usr/bin/env python3
"""Atlas Voice Developer Assistant — I/O glue only (mic/STT/console → OpenClaw → Piper)."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from atlas_context.reader import project_root  # noqa: E402
from openclaw.orchestrator import OpenClawOrchestrator  # noqa: E402
from output.player import play_wav  # noqa: E402
from state_machine import State  # noqa: E402
from tts.backend import PiperTTS, SilentWavTTS, TTSBackend  # noqa: E402


def build_tts() -> TTSBackend:
    if os.environ.get("ATLAS_VOICE", "").lower() in ("silent", "0", "off"):
        return SilentWavTTS()
    mp = os.environ.get("ATLAS_PIPER_MODEL", "").strip()
    if not mp:
        print(
            "[Atlas] ATLAS_PIPER_MODEL not set — using silent WAV. "
            "Set ATLAS_PIPER_MODEL=/path/to/voice.onnx for Piper.",
            file=sys.stderr,
        )
        return SilentWavTTS()
    p = Path(mp)
    if not p.is_file():
        print(
            f"[Atlas] Piper model not found: {p} — using silent WAV.",
            file=sys.stderr,
        )
        return SilentWavTTS()
    return PiperTTS(model_path=p)


def speak(tts: TTSBackend, text: str) -> None:
    """Print + Piper synthesis + playback — TTS performs no interpretation."""
    print(text, flush=True)
    path = Path(tempfile.gettempdir()) / f"atlas-{uuid.uuid4().hex}.wav"
    try:
        tts.synth_to_wav(text, path)
        play_wav(path)
    except RuntimeError as e:
        print(f"[Atlas] TTS playback: {e}", file=sys.stderr)
    except FileNotFoundError as e:
        print(f"[Atlas] TTS: {e}", file=sys.stderr)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def run_text_loop(tts: TTSBackend, orch: OpenClawOrchestrator) -> None:
    print("Atlas text mode. Commands: wake/sleep/shutdown + intent phrases. Empty line exits.")
    while orch.state != State.SHUTDOWN:
        try:
            line = input("> ").strip()
        except EOFError:
            orch.shutdown()
            speak(tts, "Shutting down")
            break
        if not line:
            orch.shutdown()
            speak(tts, "Shutting down")
            break

        turn = orch.handle_transcript(line)
        for phrase in turn.speech_sequence:
            speak(tts, phrase)
        if turn.exit_process:
            break

        if (
            not turn.speech_sequence
            and orch.state in (State.IDLE, State.SLEEP)
            and line.strip()
        ):
            print("(Say wake phrase first)")


def run_voice_loop(tts: TTSBackend, orch: OpenClawOrchestrator, stt: object) -> None:
    from audio.capture import record_seconds, record_until_silence

    wake_seconds = float(os.environ.get("ATLAS_WAKE_CHUNK_SEC", "4"))
    idle_like = (State.IDLE, State.SLEEP)

    while orch.state != State.SHUTDOWN:
        try:
            if orch.state in idle_like:
                audio = record_seconds(wake_seconds)
                transcript = stt.transcribe(audio).strip()
            else:
                audio = record_until_silence()
                transcript = stt.transcribe(audio).strip()

            if not transcript:
                continue

            turn = orch.handle_transcript(transcript)

            for phrase in turn.speech_sequence:
                speak(tts, phrase)
            if turn.exit_process:
                break

        except KeyboardInterrupt:
            print("\n[Atlas] Interrupted.", file=sys.stderr)
            orch.shutdown()
            speak(tts, "Shutting down")
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas Voice Developer Assistant")
    ap.add_argument(
        "--text",
        action="store_true",
        help="Text REPL instead of microphone",
    )
    args = ap.parse_args()

    os.environ.setdefault("ATLAS_PROJECT_ROOT", str(project_root()))

    def _on_transition(old: State, new: State) -> None:
        del old, new

    orch = OpenClawOrchestrator(initial=State.IDLE, on_transition=_on_transition)
    tts = build_tts()

    if args.text:
        run_text_loop(tts, orch)
        if orch.state == State.SHUTDOWN:
            sys.exit(0)
        return

    from audio.stt import WhisperSTT

    print(
        f"[Atlas] Project: {project_root()} — say 'Atlas wake up' to begin.",
        flush=True,
    )
    whisper_model = os.environ.get("ATLAS_WHISPER_MODEL", "base")
    device = os.environ.get("ATLAS_WHISPER_DEVICE")
    ctype = os.environ.get("ATLAS_WHISPER_COMPUTE", "int8")
    stt = WhisperSTT(model_size=whisper_model, device=device, compute_type=ctype)
    run_voice_loop(tts, orch, stt)
    sys.exit(0)


if __name__ == "__main__":
    main()
