# Atlas — Voice Developer Assistant

Local-first voice assistant for developers: Whisper (STT) → routed intents → **Ollama** (LLM) → **Piper** (TTS). Control phrases **override** other logic.

## Project layout

```
voice-dev-assistant/
  requirements.txt
  README.md
  src/
    main.py           # Entry: python src/main.py
    state_machine.py
    audio/
    agent/
    atlas_context/    # Local file context (pkgname avoids Python stdlib `context`)
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

Install [Ollama](https://ollama.com/) and pull a model (e.g. Llama 3):

```bash
ollama pull llama3
export ATLAS_OLLAMA_MODEL=llama3
ollama serve
```

API base defaults to `http://127.0.0.1:11434` (override via `OLLAMA_HOST`).

### 2. Whisper (speech-to-text)

The app prefers **faster-whisper** (from `requirements.txt`). Default model is **base** via `ATLAS_WHISPER_MODEL`.

```bash
export ATLAS_WHISPER_MODEL=small
export ATLAS_WHISPER_DEVICE=cpu
export ATLAS_WHISPER_COMPUTE=int8      # float32 if int8 unsupported
```

Fallback: install **openai-whisper** (`pip install openai-whisper`).

### 3. Piper TTS

1. Install the `piper` CLI ([rhasspy/piper releases](https://github.com/rhasspy/piper/releases)) on `PATH`.
2. Download a voice `.onnx` (English voice packs from Piper docs/releases).
3. Set the model:

```bash
export ATLAS_PIPER_MODEL=/absolute/path/to/en_US-lessac-medium.onnx
```

If unset or invalid, Atlas uses brief silent WAV playback and prints all cues to stdout. Force silent:

```bash
export ATLAS_VOICE=silent
```

### 4. Project scope / context

- `ATLAS_PROJECT_ROOT` — safe read root (default: cwd).
- `ATLAS_CURRENT_FILE` — “current file” under that root.
- `ATLAS_SELECTED_CODE` — selected snippet **or** `.atlas/selection.txt`.

Reads stay under the project root; risky path name patterns are skipped.

### 5. Run

```bash
python3 src/main.py              # microphone + Whisper voice loop
python3 src/main.py --text      # REPL for testing (no microphone)
python3 -m pytest tests -v
```

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

Answers use **structured** bullets; the agent layer does **not** run arbitrary shell commands or destructive file operations.

## TTS backends

Implement `tts.backend.TTSBackend` for future providers (ElevenLabs, etc.) and instantiate it from `main.build_tts()`.
