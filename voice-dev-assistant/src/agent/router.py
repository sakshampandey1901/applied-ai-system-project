"""Route natural-language intents to structured LLM tasks.

Control phrases are handled in `state_machine.extract_control_command` first.
"""

from __future__ import annotations

import re

from atlas_context.reader import read_context_for_command
from llm.ollama_client import structured_answer


SYSTEM = """You are Atlas, a voice developer assistant.
Rules:
- Answer only from the provided code context. If missing, say what is missing briefly.
- Be concise. Prefer short bullet points. No long preambles.
- Do not suggest shell commands, rm, format disk, or editing files unless the user explicitly asks to plan edits — and never destructive actions.
- Do not access paths outside the project."""


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


def run_intent(intent: str, original_text: str) -> str:
    ctx, src = read_context_for_command()
    tasks = {
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
    task_label, constraint = tasks.get(intent, tasks["generic_code"])
    user_inst = f"User said: {original_text.strip()}\n{constraint}"
    return structured_answer(SYSTEM, user_inst, ctx, task_label=task_label)
