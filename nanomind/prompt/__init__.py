"""NanoMind Prompt sub-package — prompt templates and chat formatting.

Supports all major LLM conversation formats:
  - ChatML     (GPT-3.5, GPT-4, Mistral)
  - LLaMA 2    ([INST] format)
  - LLaMA 3    (<|start_header_id|> format)
  - Alpaca     (### Instruction / ### Response)
  - Completion (plain Human:/Assistant:)

Primary exports:
    - :class:`PromptManager`     — stateful conversation + rendering
    - :class:`PromptConfig`      — template, system_prompt, max_history_turns
    - :class:`ChatMLTemplate`    — OpenAI ChatML format
    - :class:`LLaMA2Template`    — Meta LLaMA 2 chat format
    - :class:`LLaMA3Template`    — Meta LLaMA 3 format
    - :class:`AlpacaTemplate`    — Stanford Alpaca instruction format
    - :class:`CompletionTemplate`— Plain text completion
    - :class:`FewShotBuilder`    — k-shot prompt construction
    - :class:`FewShotExample`    — (input, output) example pair
    - :class:`Conversation`      — multi-turn message list
    - :class:`Message`           — single chat message
    - :class:`Role`              — system / user / assistant enum
    - :func:`get_template`       — look up template by name
    - :func:`list_templates`     — list available template names
    - :func:`register_template`  — register custom template
"""

from nanomind.prompt.types import Role, Message, Conversation
from nanomind.prompt.template import PromptTemplate
from nanomind.prompt.chatml import ChatMLTemplate
from nanomind.prompt.llama import LLaMA2Template, LLaMA3Template
from nanomind.prompt.alpaca import AlpacaTemplate, CompletionTemplate
from nanomind.prompt.few_shot import FewShotBuilder, FewShotExample
from nanomind.prompt.registry import get_template, list_templates, register_template
from nanomind.prompt.config import PromptConfig
from nanomind.prompt.manager import PromptManager

__all__ = [
    "Role", "Message", "Conversation",
    "PromptTemplate",
    "ChatMLTemplate", "LLaMA2Template", "LLaMA3Template",
    "AlpacaTemplate", "CompletionTemplate",
    "FewShotBuilder", "FewShotExample",
    "get_template", "list_templates", "register_template",
    "PromptConfig", "PromptManager",
]
