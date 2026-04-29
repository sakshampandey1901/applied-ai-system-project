"""Local read-only project data (`atlas_context` avoids importing stdlib `context`)."""

from atlas_context.reader import (
    ContextPayload,
    infer_current_file,
    load_context_bundle,
    project_root,
    read_file_safe,
    read_selected_snippet,
)

__all__ = [
    "ContextPayload",
    "infer_current_file",
    "load_context_bundle",
    "project_root",
    "read_file_safe",
    "read_selected_snippet",
]
