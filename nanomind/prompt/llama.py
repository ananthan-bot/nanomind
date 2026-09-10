"""
nanomind/prompt/llama.py — LLaMA 2 / LLaMA 3 prompt formats.

LLaMA 2 Chat format (Meta, 2023):
  [INST] <<SYS>>
  {system_prompt}
  <</SYS>>

  {user_message_1} [/INST] {assistant_response_1} </s><s> [INST] {user_message_2} [/INST]

LLaMA 3 format (Meta, 2024) — uses special tokens like ChatML:
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>
  {system}<|eot_id|>
  <|start_header_id|>user<|end_header_id|>
  {user}<|eot_id|>
  <|start_header_id|>assistant<|end_header_id|>
"""

from __future__ import annotations

from nanomind.prompt.template import PromptTemplate
from nanomind.prompt.types import Conversation, Message, Role


class LLaMA2Template(PromptTemplate):
    """
    LLaMA 2 Chat prompt format (Meta, 2023).

    Wraps user messages in ``[INST]...[/INST]`` and system prompts
    in ``<<SYS>>...</SYS>>``.
    """

    @property
    def name(self) -> str:
        return "llama2"

    def render_message(self, message: Message) -> str:
        if message.role == Role.SYSTEM:
            return f"<<SYS>>
{message.content}
<</SYS>>

"
        elif message.role == Role.USER:
            return f"[INST] {message.content} [/INST]"
        elif message.role == Role.ASSISTANT:
            return f" {message.content} </s><s>"
        return message.content

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages

        parts: list[str] = ["<s>"]
        sys_text = ""
        for msg in messages:
            if msg.role == Role.SYSTEM:
                sys_text = f"<<SYS>>
{msg.content}
<</SYS>>

"
            elif msg.role == Role.USER:
                parts.append(f"[INST] {sys_text}{msg.content} [/INST]")
                sys_text = ""
            elif msg.role == Role.ASSISTANT:
                parts.append(f" {msg.content} </s><s>")
        return "".join(parts)


class LLaMA3Template(PromptTemplate):
    """
    LLaMA 3 prompt format (Meta, 2024).

    Uses ``<|start_header_id|>`` / ``<|end_header_id|>`` / ``<|eot_id|>`` tokens.
    """

    @property
    def name(self) -> str:
        return "llama3"

    def render_message(self, message: Message) -> str:
        return (
            f"<|start_header_id|>{message.role.value}<|end_header_id|>

"
            f"{message.content}<|eot_id|>
"
        )

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages
        prompt = "<|begin_of_text|>" + self.render_messages(messages)
        prompt += "<|start_header_id|>assistant<|end_header_id|>

"
        return prompt
