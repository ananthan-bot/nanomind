"""
nanomind/prompt/alpaca.py — Alpaca and plain completion prompt formats.

Alpaca format (Stanford, 2023):
  Instruction fine-tuning format used to train Alpaca (LLaMA fine-tune).

  ### Instruction:
  {instruction}

  ### Input:
  {input}   ← optional context/input

  ### Response:
  {response}

Plain completion: No special format — raw text continuation.
Used for base (non-instruction-tuned) models.
"""

from __future__ import annotations

from nanomind.prompt.template import PromptTemplate
from nanomind.prompt.types import Conversation, Message, Role


class AlpacaTemplate(PromptTemplate):
    """
    Alpaca instruction-following format (Stanford, 2023).

    Maps conversation roles:
      system    → preamble (prepended before ### Instruction)
      user      → ### Instruction / ### Input
      assistant → ### Response
    """

    @property
    def name(self) -> str:
        return "alpaca"

    def render_message(self, message: Message) -> str:
        if message.role == Role.SYSTEM:
            return f"{message.content}

"
        elif message.role == Role.USER:
            return f"### Instruction:
{message.content}

### Response:
"
        elif message.role == Role.ASSISTANT:
            return f"{message.content}

"
        return message.content

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages
        return self.render_messages(messages)

    def format_instruction(
        self,
        instruction: str,
        input_text:  str | None = None,
        response:    str | None = None,
    ) -> str:
        """
        Format a standalone instruction (with optional input and response).

        Args:
            instruction: The task instruction.
            input_text:  Optional input context.
            response:    Optional response (for training data).

        Returns:
            Formatted Alpaca prompt string.
        """
        parts = [f"### Instruction:
{instruction}
"]
        if input_text:
            parts.append(f"
### Input:
{input_text}
")
        parts.append("
### Response:
")
        if response:
            parts.append(response)
        return "".join(parts)


class CompletionTemplate(PromptTemplate):
    """
    Plain text completion format — no special tokens.

    Concatenates all messages with configurable separators.
    Useful for base models (not instruction-tuned).

    Args:
        sep: Separator between messages (default: newline).
    """

    def __init__(self, sep: str = "
") -> None:
        self.sep = sep

    @property
    def name(self) -> str:
        return "completion"

    def render_message(self, message: Message) -> str:
        prefix = {"system": "", "user": "Human: ", "assistant": "Assistant: "}
        return prefix.get(message.role.value, "") + message.content

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages
        return self.sep.join(self.render_message(m) for m in messages)
