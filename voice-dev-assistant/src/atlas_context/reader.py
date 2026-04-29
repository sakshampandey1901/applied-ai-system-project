"""Safe local filesystem access within project scope — data only, no reasoning."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

SENSITIVE_NAME_PARTS = (
    ".ssh",
    ".gnupg",
    "credentials",
)


def _canonical(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path.absolute()


def project_root() -> Path:
    return _canonical(Path(os.environ.get("ATLAS_PROJECT_ROOT", os.getcwd())))


def is_safe_path(path: Path, root: Path | None = None) -> bool:
    root = root or project_root()
    try:
        rc = _canonical(path)
        rr = _canonical(root)
        rc.relative_to(rr)
    except ValueError:
        return False
    parts_lower = [p.lower() for p in rc.parts]
    for bad in SENSITIVE_NAME_PARTS:
        if any(bad in p for p in parts_lower):
            return False
    return True


def read_file_safe(rel_or_abs: str, root: Path | None = None) -> str:
    root = root or project_root()
    p = Path(rel_or_abs)
    if not p.is_absolute():
        p = root / p
    p = _canonical(p)
    if not is_safe_path(p, root):
        raise PermissionError(f"Access denied outside project or blocked path: {p}")
    if not p.is_file():
        raise FileNotFoundError(f"Not a file: {p}")
    return p.read_text(encoding="utf-8", errors="replace")


def infer_current_file(root: Path | None = None) -> Path | None:
    env = os.environ.get("ATLAS_CURRENT_FILE") or ""
    if not env.strip():
        return None
    p = Path(env.strip())
    root = root or project_root()
    if not p.is_absolute():
        p = root / p
    return p if is_safe_path(p, root) and p.is_file() else None


def read_selected_snippet(max_chars: int = 16000, root: Path | None = None) -> tuple[str | None, str | None]:
    """Returns (snippet, label) from env ATLAS_SELECTED_CODE or .atlas/selection.txt."""
    root = root or project_root()
    raw = os.environ.get("ATLAS_SELECTED_CODE")
    if raw is not None and raw.strip():
        text = raw.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + "\n... [truncated]"
        return text, "selected snippet (env)"

    sel = root / ".atlas" / "selection.txt"
    if sel.is_file() and is_safe_path(sel, root):
        text = sel.read_text(encoding="utf-8", errors="replace").strip()
        if text:
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated]"
            return text, "selected snippet (file)"
    return None, None


@dataclass(frozen=True, slots=True)
class ContextPayload:
    """Raw project content for upstream orchestration (no interpretation)."""

    raw_content: str
    source_descriptor: str


_PATH_IN_TRANSCRIPT = re.compile(
    r"\b([\w\-][\w\-./]*\.(?:py|md|txt|toml|ya?ml|json|tsx?|jsx?|rs|go|java|kt|swift|c|h|cpp|hpp))\b",
    re.IGNORECASE,
)


def transcript_path_hint(transcript: str, root: Path | None = None) -> Path | None:
    """First filesystem path token in `transcript` that resolves to a file under `root`."""
    root = root or project_root()
    rr = _canonical(root)
    for m in _PATH_IN_TRANSCRIPT.finditer(transcript):
        token = m.group(1).strip().lstrip("./")
        if not token or ".." in Path(token).parts:
            continue
        p = _canonical(rr / token)
        if is_safe_path(p, rr) and p.is_file():
            return p
    return None


def load_context_bundle(root: Path | None = None, transcript: str | None = None) -> ContextPayload:
    root = root or project_root()
    snippet, label = read_selected_snippet(root=root)
    if snippet:
        return ContextPayload(raw_content=snippet, source_descriptor=(label or "selected code"))

    if transcript and transcript.strip():
        hinted = transcript_path_hint(transcript.strip(), root)
        if hinted is not None:
            text = read_file_safe(str(hinted), root)
            if len(text) > 24000:
                text = text[:24000] + "\n... [file truncated]"
            return ContextPayload(
                raw_content=text,
                source_descriptor=str(hinted.relative_to(root)),
            )

    cur = infer_current_file(root)
    if cur:
        text = read_file_safe(str(cur), root)
        if len(text) > 24000:
            text = text[:24000] + "\n... [file truncated]"
        return ContextPayload(
            raw_content=text,
            source_descriptor=str(cur.relative_to(root)),
        )

    py = sorted(root.glob("*.py"))
    py.extend(sorted(root.glob("*.md")))
    py = py[:40]
    for p in py:
        try:
            text = read_file_safe(str(p), root)
            return ContextPayload(
                raw_content=text[:12000],
                source_descriptor=str(p.relative_to(root)),
            )
        except OSError:
            continue
    return ContextPayload(
        raw_content="(no readable file — set ATLAS_CURRENT_FILE or add files)",
        source_descriptor="none",
    )
