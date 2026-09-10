"""
nanomind/prompt/config.py — Prompt configuration dataclass.
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class PromptConfig:
    """
    Configuration for prompt template usage.

    Attributes:
        template:           Template format name (see :func:`get_template`).
        system_prompt:      Default system prompt prepended to all conversations.
        max_history_turns:  Maximum number of past turns to include (None = all).
        add_generation_prompt: Append the assistant generation prefix.
        truncate_system:    Truncate system prompt if too long.
    """

    template:              str        = "chatml"
    system_prompt:         str        = "You are a helpful AI assistant."
    max_history_turns:     int | None = None
    add_generation_prompt: bool       = True
    truncate_system:       bool       = False

    def __post_init__(self) -> None:
        assert self.template in ("chatml", "llama2", "llama3", "alpaca", "completion")
        if self.max_history_turns is not None:
            assert self.max_history_turns >= 1
