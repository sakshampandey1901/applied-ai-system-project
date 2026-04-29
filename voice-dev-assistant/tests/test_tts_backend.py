"""TTS backend configuration tests."""

from pathlib import Path

from tts.backend import PiperCliTTS, PiperTTS, resolve_piper_model_path


def test_piper_tts_alias_preserves_cli_backend() -> None:
    assert PiperTTS is PiperCliTTS


def test_resolve_piper_model_absolute(monkeypatch) -> None:
    monkeypatch.delenv("ATLAS_PIPER_PROJECT", raising=False)
    p = Path("/tmp/models/en_US-joe-medium.onnx")

    assert resolve_piper_model_path(p) == p


def test_resolve_piper_model_relative_to_project(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ATLAS_PIPER_PROJECT", str(tmp_path))
    monkeypatch.setenv("ATLAS_PIPER_MODEL", "models/en_US-joe-medium.onnx")

    assert resolve_piper_model_path() == tmp_path / "models/en_US-joe-medium.onnx"
