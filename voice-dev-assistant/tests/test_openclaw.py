"""OpenClaw pipeline tests — mock Ollama inference only."""

from pathlib import Path

import pytest

import openclaw.orchestrator as oc_orch
from openclaw.orchestrator import OpenClawOrchestrator
from state_machine import State


def test_intent_labels():
    from openclaw.intents import classify_intent

    assert classify_intent("explain this function") == "explain_function"


def test_openclaw_developer_turn_uses_llm_layer(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "app.py").write_text("def f(): return 1\n")

    captured_msgs = []

    def fake_infer(messages: list[dict[str, str]], **_kwargs: object) -> str:
        captured_msgs.append(messages)
        return "MODEL OUTPUT"

    monkeypatch.setattr(oc_orch, "infer_messages", fake_infer)

    orch = OpenClawOrchestrator(initial=State.ACTIVE)
    turn = orch.handle_transcript("explain this function")
    assert turn.speech_sequence == ["Thinking...", "Responding...", "MODEL OUTPUT"]
    assert captured_msgs
    user_prompt = captured_msgs[0][1]["content"]
    assert "Source:" in user_prompt
    assert "concrete" in user_prompt.lower()
    assert "If the request says 'this'" in user_prompt


def test_developer_turn_chunks_long_llm_reply(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "f.py").write_text("x = 1\n")

    filler = "Hi. " * 120

    monkeypatch.setattr(oc_orch, "infer_messages", lambda _m: filler)

    orch = OpenClawOrchestrator(initial=State.ACTIVE)
    turn = orch.handle_transcript("explain this code")
    assert turn.speech_sequence[:2] == ["Thinking...", "Responding..."]
    assert len(turn.speech_sequence) >= 4


def test_openclaw_wake_from_idle(monkeypatch: pytest.MonkeyPatch):
    orch = OpenClawOrchestrator(initial=State.IDLE)
    turn = orch.handle_transcript("Atlas wake up")
    assert orch.state == State.ACTIVE
    assert turn.speech_sequence == ["Atlas is now active"]
    assert not turn.exit_process


def test_openclaw_sleep_active(monkeypatch: pytest.MonkeyPatch):
    orch = OpenClawOrchestrator(initial=State.ACTIVE)
    turn = orch.handle_transcript("Atlas go to sleep")
    assert orch.state == State.SLEEP
    assert turn.speech_sequence == ["Going to sleep"]


def test_openclaw_shutdown_idle(monkeypatch: pytest.MonkeyPatch):
    orch = OpenClawOrchestrator(initial=State.IDLE)
    turn = orch.handle_transcript("Atlas shut down")
    assert orch.state == State.SHUTDOWN
    assert turn.exit_process


def test_context_layer_not_called_for_control(monkeypatch: pytest.MonkeyPatch):
    orch = OpenClawOrchestrator(initial=State.ACTIVE)

    def boom() -> None:
        raise AssertionError("context must not load for pure control intents")

    monkeypatch.setattr("openclaw.orchestrator.load_context_bundle", boom)
    orch.handle_transcript("Atlas go to sleep")
