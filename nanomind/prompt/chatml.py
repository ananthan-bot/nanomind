"""
nanomind/prompt/chatml.py — ChatML prompt format (OpenAI GPT-3.5/GPT-4).

ChatML (Chat Markup Language) is OpenAI's structured conversation format,
used by GPT-3.5-turbo and GPT-4. It uses special tokens to delimit turns:

  <|im_start|>system
  You are a helpful assistant.
  <|im_end|>
  <|im_start|>user
  What is the capital of France?
  <|im_end|>
  <|im_start|>assistant
  The capital of France is Paris.
  <|im_end|>

The ``<|im_start|>`` and ``<|im_end|>`` tokens are special vocabulary tokens
added to the tokenizer. The assistant turn ends without ``<|im_end|>`` when
generating (the model generates until it produces that token).

Reference: https://github.com/openai/openai-python/blob/main/chatml.md
"""

from __future__ import annotations

from nanomind.prompt.template import PromptTemplate
from nanomind.prompt.types import Conversation, Message, Role

IM_START = "<|im_start|>"
IM_END   = "<|im_end|>"


class ChatMLTemplate(PromptTemplate):
    """
    ChatML prompt format — used by GPT-3.5/GPT-4 and many open-source models.

    Args:
        add_generation_prompt: If True, append ``<|im_start|>assistant\n``
                               at the end to prompt the model to generate.
    """

    def __init__(self, add_generation_prompt: bool = True) -> None:
        self.add_generation_prompt = add_generation_prompt

    @property
    def name(self) -> str:
        return "chatml"

    def render_message(self, message: Message) -> str:
        return f"{IM_START}{message.role.value}
{message.content}{IM_END}
"

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)

        # Prepend system prompt if not already present
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages

        prompt = self.render_messages(messages)
        if self.add_generation_prompt:
            prompt += f"{IM_START}assistant
"
        return prompt

    @property
    def bos_token(self) -> str:
        return IM_START

    @property
    def eos_token(self) -> str:
        return IM_END
