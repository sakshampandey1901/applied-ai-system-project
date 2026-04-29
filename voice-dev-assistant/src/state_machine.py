"""Strict voice-controlled state machine for Atlas."""

from __future__ import annotations

from enum import Enum, auto
from typing import Callable


class State(Enum):
    IDLE = auto()
    ACTIVE = auto()
    SLEEP = auto()
    SHUTDOWN = auto()


# Phrases that override all other logic (normalized lowercase, no punctuation)
WAKE_PHRASES = ("atlas wake up",)
SLEEP_PHRASES = ("atlas go to sleep",)
SHUTDOWN_PHRASES = ("atlas shut down", "atlas close agent")


def _normalize(text: str) -> str:
    chars: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch.isspace():
            chars.append(ch if ch.isalnum() else " ")
    return " ".join("".join(chars).split())


def extract_control_command(text: str) -> str | None:
    """Return 'wake', 'sleep', 'shutdown', or None."""
    n = _normalize(text)

    # Order: shutdown → sleep → wake (most specific overrides)
    for p in SHUTDOWN_PHRASES:
        if _normalize(p) in n:
            return "shutdown"

    for p in SLEEP_PHRASES:
        if _normalize(p) in n:
            return "sleep"

    for p in WAKE_PHRASES:
        if _normalize(p) in n:
            return "wake"

    return None


class VoiceStateMachine:
    """Validated transitions only."""

    __slots__ = ("_state", "_on_transition")

    def __init__(
        self,
        initial: State = State.IDLE,
        on_transition: Callable[[State, State], None] | None = None,
    ) -> None:
        self._state = initial
        self._on_transition = on_transition

    @property
    def state(self) -> State:
        return self._state

    def _set(self, new: State) -> None:
        old = self._state
        if old == new:
            return
        self._state = new
        if self._on_transition:
            self._on_transition(old, new)

    def wake(self) -> None:
        if self._state in (State.IDLE, State.SLEEP):
            self._set(State.ACTIVE)

    def sleep(self) -> None:
        if self._state == State.ACTIVE:
            self._set(State.SLEEP)

    def shutdown(self) -> None:
        if self._state == State.SHUTDOWN:
            return
        self._set(State.SHUTDOWN)

    def idle(self) -> None:
        """Reset to idle without exit (recovery)."""
        if self._state != State.SHUTDOWN:
            self._set(State.IDLE)

    def apply_control_command(self, cmd: str | None) -> None:
        if cmd == "wake":
            self.wake()
        elif cmd == "sleep":
            self.sleep()
        elif cmd == "shutdown":
            self.shutdown()
