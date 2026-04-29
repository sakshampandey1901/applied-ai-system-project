"""Intent labels from user text — routing only (no LLM, no side effects)."""

from __future__ import annotations

import re


def _norm(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def classify_intent(text: str) -> str:
    n = _norm(text)
    if "summarize" in n and "file" in n:
        return "summarize_file"
    if "explain" in n and "function" in n:
        return "explain_function"
    if "fix" in n and "error" in n:
        return "fix_error"
    if "refactor" in n:
        return "refactor"
    if "what" in n and "code" in n and "do" in n:
        return "what_code"
    if "what does this code" in n:
        return "what_code"
    return "generic_code"
