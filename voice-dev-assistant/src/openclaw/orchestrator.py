"""OpenClaw: orchestration only — intent, state, routing, prompt assembly, final speech plan."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from atlas_context.reader import load_context_bundle
from llm.ollama_client import infer_messages
from openclaw.intents import classify_intent
from state_machine import State, VoiceStateMachine, extract_control_command

# Policy text sent to Ollama as system message (orchestration-owned policy, not raw files).
OLLAMA_SYSTEM_POLICY = """You are Atlas, a voice developer assistant.
Rules:
- Answer only from the provided code context. If missing, say what is missing briefly.
- Be concise. Prefer short bullet points. No long preambles.
- Do not suggest shell commands, rm, format disk, or editing files unless the user explicitly asks to plan edits — and never destructive actions.
- Do not access paths outside the project."""

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


@dataclass(frozen=True, slots=True)
class TurnResult:
    """What to speak (in order) and whether the process should exit."""

    speech_sequence: list[str]
    exit_process: bool


class OpenClawOrchestrator:
    """Single entry for transcript handling after STT; owns state machine."""

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
        text = transcript.strip()
        if not text:
            return TurnResult([], False)

        ctrl = extract_control_command(text)

        st = self._sm.state
        idle_like = st in (State.IDLE, State.SLEEP)

        if idle_like:
            if ctrl == "shutdown":
                self._sm.shutdown()
                return TurnResult(["Shutting down"], True)
            if ctrl == "wake":
                self._sm.apply_control_command("wake")
                return TurnResult(["Atlas is now active"], False)
            return TurnResult([], False)

        # ACTIVE
        if ctrl == "sleep":
            self._sm.apply_control_command("sleep")
            return TurnResult(["Going to sleep"], False)
        if ctrl == "shutdown":
            self._sm.shutdown()
            return TurnResult(["Shutting down"], True)
        if ctrl == "wake":
            return TurnResult(["Atlas is now active"], False)

        # Developer tools path: Context → Ollama → assemble ordered speech (fixed cues + model text only).
        intent = classify_intent(text)
        task_label, constraint = _TASK_SPECS.get(intent, _TASK_SPECS["generic_code"])

        payload = load_context_bundle()
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

        return TurnResult(
            ["Thinking...", "Responding...", reply],
            False,
        )
