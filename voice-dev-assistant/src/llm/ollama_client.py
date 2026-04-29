"""Ollama local LLM client (structured minimal prompts; stdlib urllib only)."""

from __future__ import annotations

import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_MODEL = os.environ.get("ATLAS_OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")


def chat(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float = 0.2,
    timeout_sec: float = 120.0,
) -> str:
    model = model or DEFAULT_MODEL
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        f"{OLLAMA_URL}/api/chat",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=timeout_sec) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"Ollama HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(
            "Cannot reach Ollama. Ensure it is running: ollama serve"
        ) from e

    msg = (body or {}).get("message") or {}
    content = msg.get("content") or ""
    return content.strip()


def structured_answer(
    system_rules: str,
    user_instruction: str,
    code_context: str,
    *,
    task_label: str,
) -> str:
    messages = [
        {"role": "system", "content": system_rules.strip()},
        {
            "role": "user",
            "content": f"{task_label}\n\n--- Context ---\n{code_context}\n--- End ---\n\nInstruction:\n{user_instruction}".strip(),
        },
    ]
    return chat(messages)
