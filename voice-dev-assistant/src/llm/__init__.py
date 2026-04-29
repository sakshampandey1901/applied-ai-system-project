"""Ollama language model — inference only."""

from llm.ollama_client import DEFAULT_MODEL, OLLAMA_URL, infer_messages

__all__ = ["DEFAULT_MODEL", "OLLAMA_URL", "infer_messages"]
