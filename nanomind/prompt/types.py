"""
nanomind/prompt/types.py — Core types for chat messages and roles.

## Chat Format History

GPT-3 (2020): Raw text completion — prompt engineering was just string manipulation.

InstructGPT (2022): Introduced instruction fine-tuning.
  Format: "Human: {question}\nAssistant: {answer}"

ChatML (OpenAI, 2023): Structured multi-turn format used by GPT-3.5/GPT-4:
  <|im_start|>system
  You are a helpful assistant.
  <|im_end|>
  <|im_start|>user
  Hello!
  <|im_end|>
  <|im_start|>assistant

LLaMA Chat (Meta, 2023): Llama 2's conversation format:
  [INST] <<SYS>> system prompt <</SYS>> user message [/INST] response

Alpaca (Stanford, 2023): Instruction-following format:
  ### Instruction: {instruction}
  ### Input: {input}
  ### Response: {response}

NanoMind supports all of these via the PromptTemplate system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum


class Role(str, Enum):
    """Chat message roles."""
    SYSTEM    = "system"
    USER      = "user"
    ASSISTANT = "assistant"
    FUNCTION  = "function"   # tool call responses (GPT-4 function calling)


@dataclass
class Message:
    """
    A single chat message.

    Attributes:
        role:    Speaker role (system/user/assistant/function).
        content: Text content of the message.
        name:    Optional name (for multi-agent or function responses).
    """
    role:    Role
    content: str
    name:    str | None = None

    def to_dict(self) -> dict:
        d = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(Role.SYSTEM, content)

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(Role.USER, content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        return cls(Role.ASSISTANT, content)


@dataclass
class Conversation:
    """
    A multi-turn conversation as a list of messages.

    Attributes:
        messages:      Ordered list of :class:`Message` objects.
        system_prompt: Default system prompt (prepended if no system message).
    """
    messages:      list[Message] = field(default_factory=list)
    system_prompt: str | None    = None

    def add(self, role: Role | str, content: str) -> "Conversation":
        if isinstance(role, str):
            role = Role(role)
        self.messages.append(Message(role, content))
        return self

    def user(self, content: str) -> "Conversation":
        return self.add(Role.USER, content)

    def assistant(self, content: str) -> "Conversation":
        return self.add(Role.ASSISTANT, content)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    def to_list(self) -> list[dict]:
        return [m.to_dict() for m in self.messages]

    def last_user_message(self) -> Message | None:
        for m in reversed(self.messages):
            if m.role == Role.USER:
                return m
        return None
