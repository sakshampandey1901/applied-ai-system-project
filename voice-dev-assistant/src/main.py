#!/usr/bin/env python3
"""Atlas Voice Developer Assistant — entry point."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import uuid
from pathlib import Path

# This file lives inside `src/`; that directory is the import root for `audio`, `agent`, …
_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from agent.router import classify_intent, run_intent  # noqa: E402
from atlas_context.reader import project_root  # noqa: E402
from output.player import play_wav  # noqa: E402
from state_machine import State, VoiceStateMachine, extract_control_command  # noqa: E402
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


def run_text_loop(tts: TTSBackend, sm: VoiceStateMachine) -> None:
    """Non-audio mode for tests and environments without a mic."""
    print("Atlas text mode. Commands: wake/sleep/shutdown + intent phrases. Empty line exits.")
    while sm.state != State.SHUTDOWN:
        try:
            line = input("> ").strip()
        except EOFError:
            sm.shutdown()
            speak(tts, "Shutting down")
            break
        if not line:
            sm.shutdown()
            speak(tts, "Shutting down")
            break
        ctrl = extract_control_command(line)
        if ctrl == "wake":
            sm.apply_control_command("wake")
            if sm.state == State.ACTIVE:
                speak(tts, "Atlas is now active")
            continue
        if ctrl == "sleep":
            sm.apply_control_command("sleep")
            if sm.state == State.SLEEP:
                speak(tts, "Going to sleep")
            continue
        if ctrl == "shutdown":
            sm.shutdown()
            speak(tts, "Shutting down")
            break
        if sm.state != State.ACTIVE:
            print("(Say wake phrase first)")
            continue
        speak(tts, "Thinking...")
        intent = classify_intent(line)
        speak(tts, "Responding...")
        try:
            reply = run_intent(intent, line)
        except Exception as e:
            reply = f"Error: {e}"
        speak(tts, reply)


def run_voice_loop(tts: TTSBackend, sm: VoiceStateMachine, stt: object) -> None:
    from audio.capture import record_seconds, record_until_silence

    wake_seconds = float(os.environ.get("ATLAS_WAKE_CHUNK_SEC", "4"))
    idle_states = (State.IDLE, State.SLEEP)

    while sm.state != State.SHUTDOWN:
        try:
            if sm.state in idle_states:
                audio = record_seconds(wake_seconds)
                transcript = stt.transcribe(audio)
                if not transcript.strip():
                    continue
                ctrl = extract_control_command(transcript)
                if ctrl == "shutdown":
                    sm.shutdown()
                    speak(tts, "Shutting down")
                    break
                if ctrl == "wake":
                    sm.apply_control_command("wake")
                    speak(tts, "Atlas is now active")
                continue

            if sm.state == State.ACTIVE:
                speak(tts, "Thinking...")
                audio = record_until_silence()
                transcript = stt.transcribe(audio)
                if not transcript.strip():
                    continue
                ctrl = extract_control_command(transcript)
                if ctrl == "sleep":
                    sm.apply_control_command("sleep")
                    speak(tts, "Going to sleep")
                    continue
                if ctrl == "shutdown":
                    sm.shutdown()
                    speak(tts, "Shutting down")
                    break
                if ctrl == "wake":
                    speak(tts, "Atlas is now active")
                    continue

                speak(tts, "Responding...")
                intent = classify_intent(transcript)
                try:
                    reply = run_intent(intent, transcript)
                except Exception as e:
                    reply = f"I could not complete that: {e}"
                speak(tts, reply)

        except KeyboardInterrupt:
            print("\n[Atlas] Interrupted.", file=sys.stderr)
            sm.shutdown()
            speak(tts, "Shutting down")
            break


def main() -> None:
    ap = argparse.ArgumentParser(description="Atlas Voice Developer Assistant")
    ap.add_argument(
        "--text",
        action="store_true",
        help="Text REPL instead of microphone (for testing / no audio hardware)",
    )
    args = ap.parse_args()

    os.environ.setdefault("ATLAS_PROJECT_ROOT", str(project_root()))

    def _on_transition(old: State, new: State) -> None:
        del old, new

    sm = VoiceStateMachine(initial=State.IDLE, on_transition=_on_transition)
    tts = build_tts()

    if args.text:
        run_text_loop(tts, sm)
        if sm.state == State.SHUTDOWN:
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
    run_voice_loop(tts, sm, stt)
    sys.exit(0)


if __name__ == "__main__":
    main()
