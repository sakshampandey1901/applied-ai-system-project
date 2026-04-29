# Atlas — Voice Developer Assistant

Local-first voice assistant: **Whisper (STT)** → **OpenClaw (orchestration)** → **`atlas_context` (reads)** → **Ollama (inference)** → **OpenClaw (speech plan)** → **Piper (TTS)**. Control phrases override other logic.

## Layer boundaries (mandatory pipeline)

1. **Input** — microphone or `main.py --text` produces text.
2. **OpenClaw** — intent + control parsing, state (`IDLE` / `ACTIVE` / `SLEEP` / `SHUTDOWN`), routing only. Calls **Context** and **Ollama**; does not synthesize speech; does not run STT/LLM internals.
3. **`atlas_context`** — returns raw `ContextPayload` (file/snippet bytes as text + descriptor). No LLM, no TTS, no intent logic.
4. **Ollama** — `infer_messages(...)` only (HTTP inference). No filesystem, no state, no audio.
5. **OpenClaw** — builds ordered `speech_sequence` (fixed cues + model reply text). No rewriting of model output except error passthrough strings.
6. **Piper** — `tts` + `speak()` in **main**: synthesizes and plays finalized strings only.

## Project layout

```
voice-dev-assistant/
  requirements.txt
  README.md
  src/
    main.py           # Entry: python src/main.py (I/O only)
    state_machine.py # Used by OpenClaw for transitions / phrase extraction
    openclaw/         # Orchestration (intent + routing + prompt assembly)
    audio/
    atlas_context/    # Local file/snippet reads (pkgname avoids Python stdlib `context`)
    llm/
    tts/
    output/
  tests/
```


## Quick start

```bash
cd voice-dev-assistant
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 1. Ollama (local LLM)

Install [Ollama](https://ollama.com/) and pull the default model (**`llama3.1:8b`** — override with `export ATLAS_OLLAMA_MODEL=…`):

```bash
ollama pull llama3.1:8b
```

API base defaults to `http://127.0.0.1:11434` (override via `OLLAMA_HOST`). On macOS the app usually keeps the daemon running; use `ollama serve` only if nothing is listening.

### 2. Whisper (speech-to-text)

The app prefers **faster-whisper** (from `requirements.txt`). Default model is **base** via `ATLAS_WHISPER_MODEL`.

```bash
export ATLAS_WHISPER_MODEL=small
export ATLAS_WHISPER_DEVICE=cpu
export ATLAS_WHISPER_COMPUTE=int8      # float32 if int8 unsupported
```

Fallback: install **openai-whisper** (`pip install openai-whisper`).

### 3. Piper TTS

1. `pip install -r requirements.txt` includes **`piper-tts`** (Python `PiperVoice` API — same idea as loading `models/*.onnx` with `wave.open` + `voice.synthesize` / `synthesize_wav`).
2. Download a voice **`.onnx`** (+ matching `.json` next to it, usually auto-picked).
3. Point Atlas at it:

```bash
export ATLAS_PIPER_MODEL="/full/path/to/models/en_US-joe-medium.onnx"
```

For a project-local model, keep the file under a directory such as
`voice-dev-assistant/models/` and resolve a relative model path from that base:

```bash
export ATLAS_PIPER_PROJECT="/full/path/to/voice-dev-assistant"
export ATLAS_PIPER_MODEL="models/en_US-joe-medium.onnx"
```

Atlas **prefers the Python bindings** (`PiperVoiceTTS`). If `piper-tts` isn’t installed, it tries the **`piper` CLI** on `PATH` ([releases](https://github.com/rhasspy/piper/releases)).

If unset or invalid, Atlas uses brief silent WAV playback and prints all cues to stdout. Force silent:

```bash
export ATLAS_VOICE=silent
```

### 4. Project scope / context

- `ATLAS_PROJECT_ROOT` — safe read root (default: cwd).
- `ATLAS_CURRENT_FILE` — “current file” under that root.
- `ATLAS_SELECTED_CODE` — selected snippet **or** `.atlas/selection.txt`.
- Mention a project file in speech (e.g. `readme.md`): **context resolves that path first** before `ATLAS_CURRENT_FILE`/fallback.

Reads stay under the project root; risky path name patterns are skipped.

### 5. Run

From **`voice-dev-assistant`** as the current directory (your shell prompt usually ends with that folder name):

```bash
python3 src/main.py              # microphone + Whisper voice loop
python3 src/main.py --text      # REPL for testing (no microphone)
python3 -m pytest tests -v
```

If you are already there, **do not** run `cd voice-dev-assistant` again (nested path does not exist).

If Piper is unset, Atlas prints `[Atlas] ATLAS_PIPER_MODEL…` once to stderr and continues with silent WAV plus **printed** text—that is normal.


## Control phrases

| Phrase                     | Transition / effect                              |
|---------------------------|--------------------------------------------------|
| **Atlas wake up**          | IDLE or SLEEP → ACTIVE                           |
| **Atlas go to sleep**      | ACTIVE → SLEEP                                  |
| **Atlas shut down**        | Shutdown (exit)                                 |
| **Atlas close agent**      | Shutdown (exit)                                 |

Order of checks: shutdown → sleep → wake. Audio prompts: “Atlas is now active”, “Going to sleep”, “Shutting down”, “Thinking…”, “Responding…”.

## Coding intents (ACTIVE only)

- Explain this function / Summarize this file / Fix this error / Refactor this code / What does this code do?

Answers use concise bullets routed by OpenClaw; **no arbitrary shell or destructive writes**.

## TTS backends

Implement `tts.backend.TTSBackend` for future providers (ElevenLabs, etc.) and instantiate it from `main.build_tts()`.
