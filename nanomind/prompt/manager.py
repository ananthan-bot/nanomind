"""
nanomind/prompt/manager.py — PromptManager: stateful conversation + rendering.
"""

from __future__ import annotations

from nanomind.prompt.config import PromptConfig
from nanomind.prompt.registry import get_template
from nanomind.prompt.types import Conversation, Message, Role


class PromptManager:
    """
    Stateful prompt manager: maintains conversation history and renders prompts.

    Combines a :class:`Conversation`, a :class:`PromptTemplate`, and
    a :class:`PromptConfig` into a single easy-to-use interface for
    building multi-turn conversations.

    Args:
        cfg: Prompt configuration.

    Example::

        pm = PromptManager(PromptConfig(template="chatml",
                                        system_prompt="You are an AI tutor."))
        prompt = pm.build("What is a transformer?")
        # ... feed to model, get response ...
        pm.add_assistant("A transformer is ...")
        prompt2 = pm.build("Can you elaborate?")
    """

    def __init__(self, cfg: PromptConfig | None = None) -> None:
        self.cfg      = cfg or PromptConfig()
        self.template = get_template(self.cfg.template,
                                     **({"add_generation_prompt": self.cfg.add_generation_prompt}
                                        if self.cfg.template == "chatml" else {}))
        self.conv     = Conversation(system_prompt=self.cfg.system_prompt)
        # Prepend system message
        if self.cfg.system_prompt:
            self.conv.add(Role.SYSTEM, self.cfg.system_prompt)

    def build(self, user_message: str) -> str:
        """
        Add a user message and return the full rendered prompt.

        Args:
            user_message: The user's input text.

        Returns:
            Formatted prompt string ready for model input.
        """
        self.conv.add(Role.USER, user_message)
        return self._render()

    def add_assistant(self, response: str) -> None:
        """Record an assistant response in conversation history."""
        self.conv.add(Role.ASSISTANT, response)

    def _render(self) -> str:
        """Render the conversation, respecting max_history_turns."""
        messages = list(self.conv)
        if self.cfg.max_history_turns is not None:
            # Always keep system message + last N turns
            sys_msgs  = [m for m in messages if m.role == Role.SYSTEM]
            other     = [m for m in messages if m.role != Role.SYSTEM]
            max_other = self.cfg.max_history_turns * 2
            messages  = sys_msgs + other[-max_other:]

        # Temporary conversation for rendering
        tmp = Conversation(system_prompt=self.conv.system_prompt)
        tmp.messages = messages
        return self.template.render(tmp)

    def reset(self) -> None:
        """Clear conversation history (keep system prompt)."""
        self.conv = Conversation(system_prompt=self.cfg.system_prompt)
        if self.cfg.system_prompt:
            self.conv.add(Role.SYSTEM, self.cfg.system_prompt)

    @property
    def n_turns(self) -> int:
        """Number of completed user-assistant turn pairs."""
        return sum(1 for m in self.conv if m.role == Role.USER)

    def __repr__(self) -> str:
        return (f"PromptManager(template={self.cfg.template!r}, "
                f"turns={self.n_turns})")
