"""Local read-only context for the agent."""

from atlas_context.reader import (
    infer_current_file,
    project_root,
    read_context_for_command,
    read_file_safe,
    read_selected_snippet,
)

__all__ = [
    "infer_current_file",
    "project_root",
    "read_context_for_command",
    "read_file_safe",
    "read_selected_snippet",
]
