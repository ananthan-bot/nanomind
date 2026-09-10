"""
nanomind/prompt/template.py — Base prompt template class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nanomind.prompt.types import Conversation, Message, Role


class PromptTemplate(ABC):
    """
    Abstract base class for prompt templates.

    A prompt template converts a :class:`Conversation` (list of messages)
    into a single text string that a language model can process.

    Subclass this to implement new chat formats.

    Example::

        template = ChatMLTemplate()
        conv     = Conversation()
        conv.user("What is 2+2?")
        prompt   = template.render(conv)
        # → "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this template format."""

    @abstractmethod
    def render(self, conversation: Conversation) -> str:
        """
        Render a conversation to a prompt string.

        Args:
            conversation: Multi-turn conversation.

        Returns:
            Formatted prompt string ready for the model.
        """

    @abstractmethod
    def render_message(self, message: Message) -> str:
        """Render a single message to a string."""

    def render_messages(self, messages: list[Message]) -> str:
        """Render a list of messages."""
        return "".join(self.render_message(m) for m in messages)

    def apply(
        self,
        system:    str | None = None,
        user:      str | None = None,
        assistant: str | None = None,
    ) -> str:
        """
        Convenience: build a simple single-turn prompt from parts.

        Args:
            system:    System message content.
            user:      User message content.
            assistant: Optional assistant prefix.

        Returns:
            Formatted prompt string.
        """
        conv = Conversation(system_prompt=system)
        if system:
            conv.add(Role.SYSTEM, system)
        if user:
            conv.add(Role.USER, user)
        if assistant:
            conv.add(Role.ASSISTANT, assistant)
        return self.render(conv)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"
