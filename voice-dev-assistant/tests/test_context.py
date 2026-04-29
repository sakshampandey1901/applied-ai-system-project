"""Context layer tests — transcript-derived file scope."""

import pytest

from atlas_context.reader import load_context_bundle, transcript_path_hint


def test_transcript_hints_file_over_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "a.py").write_text("ALPHA")
    (tmp_path / "b.py").write_text("BETA")

    monkeypatch.setenv("ATLAS_CURRENT_FILE", "a.py")
    bundle = load_context_bundle(transcript="please summarize b.py briefly")
    assert "BETA" in bundle.raw_content
    assert "b.py" in bundle.source_descriptor


def test_transcript_path_hint_returns_none_when_no_match(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "x.py").write_text("1")
    assert transcript_path_hint("no filenames here") is None


def test_transcript_traversal_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "safe.py").write_text("ok")
    assert transcript_path_hint("../../../etc/passwd") is None


def test_fallback_context_recurses_into_src(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ATLAS_PROJECT_ROOT", str(tmp_path))
    (tmp_path / "README.md").write_text("top level docs")
    src = tmp_path / "src"
    src.mkdir()
    (src / "main.py").write_text("def concrete_answer():\n    return 42\n")

    bundle = load_context_bundle()

    assert "src/main.py" in bundle.source_descriptor
    assert "concrete_answer" in bundle.raw_content


@pytest.fixture(autouse=True)
def clear_selection_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ATLAS_SELECTED_CODE", raising=False)
