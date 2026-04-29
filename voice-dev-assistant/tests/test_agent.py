"""Agent routing tests (mocked LLM)."""

import pytest
from pathlib import Path

import agent.router as ar
from agent.router import classify_intent, run_intent


def test_classify_intent_buckets():
    assert classify_intent("explain this function for me") == "explain_function"
    assert classify_intent("Summarize this file") == "summarize_file"
    assert classify_intent("please fix this error") == "fix_error"
    assert classify_intent("can you refactor") == "refactor"
    assert classify_intent("what does this code do?") == "what_code"


def test_run_intent_uses_mocked_llm(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "demo.py").write_text("def f():\n    return 42\n")

    captured: dict[str, str] = {}

    def fake_structured_answer(system_rules: str, user_instruction: str, code_context: str, *, task_label: str) -> str:
        captured["ctx"] = code_context.strip()
        return "mock reply"

    monkeypatch.setattr(ar, "structured_answer", fake_structured_answer)

    ans = run_intent("explain_function", "explain this")
    assert ans == "mock reply"
    assert "def f" in captured["ctx"]
