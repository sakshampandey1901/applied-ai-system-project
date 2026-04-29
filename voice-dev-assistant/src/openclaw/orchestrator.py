"""OpenClaw: orchestration — intent, routing, context scope, speech plan for TTS (no Piper)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from atlas_context.reader import load_context_bundle
from llm.ollama_client import infer_messages
from openclaw.intents import classify_intent
from state_machine import State, VoiceStateMachine, extract_control_command

# Policy text owned by orchestration — passed to Ollama as system message only.
OLLAMA_SYSTEM_POLICY = """You are Atlas, a voice developer assistant.
Rules:
- Answer only from the provided code context. If missing, say what is missing briefly.
- Be concise. Prefer short bullet points. No long preambles.
- Do not suggest shell commands, rm, format disk, or editing files unless the user explicitly asks to plan edits — and never destructive actions.
- Do not access paths beyond the supplied context."""

_TASK_SPECS: dict[str, tuple[str, str]] = {
    "summarize_file": (
        "Summarize this file for a developer listening on audio.",
        "Keep 3–6 bullets of the most important points.",
    ),
    "explain_function": (
        "Explain the main function or core logic in this snippet.",
        "Start with purpose, then key steps (bullets).",
    ),
    "fix_error": (
        "Diagnose likely issues and propose a minimal fix outline.",
        "Do not claim certainty without evidence from context. Bullet possible causes and fixes.",
    ),
    "refactor": (
        "Suggest a safe refactor outline for readability or structure.",
        "Bullets only; no sweeping rewrites.",
    ),
    "what_code": (
        "Describe what this code does.",
        "Short bullets; state inputs/outputs if obvious.",
    ),
    "generic_code": (
        "Answer the developer request using the provided code context.",
        "Bullets; keep under ~150 spoken words.",
    ),
}

# Chunk long replies so main+Piper can play sentence-sized units (streaming-style UX, sync inference).
_REPLY_CHUNK_SOFT_MAX = 300


def _build_user_message_for_ollama(
    task_label: str,
    constraint: str,
    payload_raw: str,
    source_descriptor: str,
    user_said: str,
) -> str:
    return (
        f"{task_label}\n"
        f"Source: {source_descriptor}\n\n"
        f"--- Context ---\n{payload_raw}\n--- End ---\n\n"
        f"Instruction:\nUser said: {user_said.strip()}\n{constraint}"
    ).strip()


def _split_reply_for_tts_dispatch(model_reply: str) -> list[str]:
    """Break long prose into sequential speak units (main calls Piper per segment)."""
    raw = model_reply.strip()
    if not raw:
        return []
    if len(raw) <= _REPLY_CHUNK_SOFT_MAX:
        return [raw]

    parts: list[str] = []
    buf = ""
    for para in raw.split("\n"):
        para = para.strip()
        if not para:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", para):
            s = sent.strip()
            if not s:
                continue
            cand = f"{buf} {s}".strip() if buf else s
            if len(cand) <= _REPLY_CHUNK_SOFT_MAX:
                buf = cand
            else:
                if buf:
                    parts.append(buf)
                buf = s[: _REPLY_CHUNK_SOFT_MAX] if len(s) > _REPLY_CHUNK_SOFT_MAX else s
    if buf:
        parts.append(buf)
    return parts if parts else [raw[:_REPLY_CHUNK_SOFT_MAX]]


@dataclass(frozen=True, slots=True)
class TurnResult:
    """Ordered strings for Piper; main iterates speak() → TTS dispatcher."""

    speech_sequence: list[str]
    exit_process: bool


class OpenClawOrchestrator:
    """STT yields text → handle_transcript() → ordered speech cues + LLM chunks (single sync inference call)."""

    __slots__ = ("_sm",)

    def __init__(
        self,
        initial: State = State.IDLE,
        on_transition: Callable[[State, State], None] | None = None,
    ) -> None:
        self._sm = VoiceStateMachine(initial=initial, on_transition=on_transition)

    @property
    def state(self) -> State:
        return self._sm.state

    def shutdown(self) -> None:
        self._sm.shutdown()

    def handle_transcript(self, transcript: str) -> TurnResult:
        """Pipeline: blank → no-op | control cues (immediate) | intent → scoped context → infer → speech plan."""
        text = transcript.strip()
        if not text:
            return TurnResult([], False)

        # 1. Control phrases (fixed cues only — orchestration, not NLG).
        idle_like = self._sm.state in (State.IDLE, State.SLEEP)
        ctrl_turn = (
            self._control_when_dormant(text)
            if idle_like
            else self._control_when_active(text)
        )
        if ctrl_turn is not None:
            return ctrl_turn

        if idle_like:
            return TurnResult([], False)

        # 2–5. ACTIVE developer path — intent → context(scope=transcript) → Ollama (sync inference)
        intent = classify_intent(text)
        task_label, constraint = _TASK_SPECS.get(intent, _TASK_SPECS["generic_code"])
        payload = load_context_bundle(transcript=text)

        cues = ["Thinking...", "Responding..."]
        user_msg = _build_user_message_for_ollama(
            task_label,
            constraint,
            payload.raw_content,
            payload.source_descriptor,
            text,
        )
        messages = [
            {"role": "system", "content": OLLAMA_SYSTEM_POLICY},
            {"role": "user", "content": user_msg},
        ]
        try:
            reply = infer_messages(messages)
        except Exception as e:
            reply = f"I could not complete that: {e}"

        chunks = _split_reply_for_tts_dispatch(reply)
        return TurnResult(cues + chunks, False)

    def _control_when_dormant(self, text: str) -> TurnResult | None:
        ctrl = extract_control_command(text)
        if ctrl == "shutdown":
            self._sm.shutdown()
            return TurnResult(["Shutting down"], True)
        if ctrl == "wake":
            self._sm.apply_control_command("wake")
            return TurnResult(["Atlas is now active"], False)
        return None

    def _control_when_active(self, text: str) -> TurnResult | None:
        ctrl = extract_control_command(text)
        if ctrl == "sleep":
            self._sm.apply_control_command("sleep")
            return TurnResult(["Going to sleep"], False)
        if ctrl == "shutdown":
            self._sm.shutdown()
            return TurnResult(["Shutting down"], True)
        if ctrl == "wake":
            return TurnResult(["Atlas is now active"], False)
        return None
