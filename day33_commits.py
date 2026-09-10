"""
day33_commits.py — 20 atomic commits for Day 33: Prompt Templates & Chat Format.
"""
import os, subprocess, sys
from pathlib import Path

REPO = Path(r"C:\Users\anant\.gemini\antigravity-ide\scratch\minigpt")
os.environ["PYTHONIOENCODING"] = "utf-8"

import winreg
def _env_path():
    paths = []
    for hive in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
        for sub in [r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment", r"Environment"]:
            try:
                k = winreg.OpenKey(hive, sub)
                paths.append(winreg.QueryValueEx(k, "PATH")[0])
            except Exception:
                pass
    return ";".join(paths)
os.environ["PATH"] = _env_path()

def run(*args, check=True):
    r = subprocess.run(list(args), cwd=REPO, capture_output=True, text=True, env=os.environ)
    if check and r.returncode != 0:
        print(f"STDOUT: {r.stdout}\nSTDERR: {r.stderr}"); sys.exit(1)
    return r

def commit(msg):
    run("git", "add", "-A")
    r = run("git", "commit", "-m", msg, check=False)
    if "nothing to commit" in (r.stdout + r.stderr):
        print(f"  (skip) {msg}"); return False
    if r.returncode != 0:
        print(f"FAILED: {r.stderr}"); sys.exit(1)
    print(f"  + {msg}"); return True

def write(path, content):
    p = REPO / path
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")

def read(path):
    return (REPO / path).read_text(encoding="utf-8")

print("\n=== DAY 33: Prompt Templates & Chat Format — 20 commits ===\n")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 1 — prompt package skeleton
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/__init__.py",
      '"""NanoMind Prompt sub-package — prompt templates and chat formatting."""\n')
commit("feat: add nanomind/prompt/ package skeleton for prompt templates and chat format")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 2 — Message and Role types
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/types.py", '''\
"""
nanomind/prompt/types.py — Core types for chat messages and roles.

## Chat Format History

GPT-3 (2020): Raw text completion — prompt engineering was just string manipulation.

InstructGPT (2022): Introduced instruction fine-tuning.
  Format: "Human: {question}\\nAssistant: {answer}"

ChatML (OpenAI, 2023): Structured multi-turn format used by GPT-3.5/GPT-4:
  <|im_start|>system
  You are a helpful assistant.
  <|im_end|>
  <|im_start|>user
  Hello!
  <|im_end|>
  <|im_start|>assistant

LLaMA Chat (Meta, 2023): Llama 2\'s conversation format:
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
''')
commit("feat: add Role enum, Message dataclass, Conversation — multi-turn chat primitives")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 3 — base PromptTemplate
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/template.py", '''\
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
        # → "<|im_start|>user\\nWhat is 2+2?<|im_end|>\\n<|im_start|>assistant\\n"
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
''')
commit("feat: add PromptTemplate ABC — render(), render_message(), apply() interface")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 4 — ChatML template
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/chatml.py", '''\
"""
nanomind/prompt/chatml.py — ChatML prompt format (OpenAI GPT-3.5/GPT-4).

ChatML (Chat Markup Language) is OpenAI\'s structured conversation format,
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
        add_generation_prompt: If True, append ``<|im_start|>assistant\\n``
                               at the end to prompt the model to generate.
    """

    def __init__(self, add_generation_prompt: bool = True) -> None:
        self.add_generation_prompt = add_generation_prompt

    @property
    def name(self) -> str:
        return "chatml"

    def render_message(self, message: Message) -> str:
        return f"{IM_START}{message.role.value}\n{message.content}{IM_END}\n"

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)

        # Prepend system prompt if not already present
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages

        prompt = self.render_messages(messages)
        if self.add_generation_prompt:
            prompt += f"{IM_START}assistant\n"
        return prompt

    @property
    def bos_token(self) -> str:
        return IM_START

    @property
    def eos_token(self) -> str:
        return IM_END
''')
commit("feat: add ChatMLTemplate — <|im_start|>/<|im_end|> format, add_generation_prompt")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 5 — LLaMA chat template
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/llama.py", '''\
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
            return f"<<SYS>>\n{message.content}\n<</SYS>>\n\n"
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
                sys_text = f"<<SYS>>\n{msg.content}\n<</SYS>>\n\n"
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
            f"<|start_header_id|>{message.role.value}<|end_header_id|>\n\n"
            f"{message.content}<|eot_id|>\n"
        )

    def render(self, conversation: Conversation) -> str:
        messages = list(conversation)
        if (conversation.system_prompt
                and (not messages or messages[0].role != Role.SYSTEM)):
            messages = [Message.system(conversation.system_prompt)] + messages
        prompt = "<|begin_of_text|>" + self.render_messages(messages)
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
''')
commit("feat: add LLaMA2Template ([INST]/<<SYS>>) and LLaMA3Template (<|start_header_id|>) formats")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 6 — Alpaca + completion templates
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/alpaca.py", '''\
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
            return f"{message.content}\n\n"
        elif message.role == Role.USER:
            return f"### Instruction:\n{message.content}\n\n### Response:\n"
        elif message.role == Role.ASSISTANT:
            return f"{message.content}\n\n"
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
        parts = [f"### Instruction:\n{instruction}\n"]
        if input_text:
            parts.append(f"\n### Input:\n{input_text}\n")
        parts.append("\n### Response:\n")
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

    def __init__(self, sep: str = "\n") -> None:
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
''')
commit("feat: add AlpacaTemplate (### Instruction/Response), CompletionTemplate (Human:/Assistant:)")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 7 — few-shot builder
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/few_shot.py", '''\
"""
nanomind/prompt/few_shot.py — Few-shot prompt builder.

Few-shot prompting provides the model with examples before the actual query:

  Example (k=2 shots):
    Q: What is the capital of France?
    A: Paris.

    Q: What is the capital of Germany?
    A: Berlin.

    Q: What is the capital of Japan?
    A:   ← model generates here

This dramatically improves model performance on tasks it hasn\'t been
explicitly fine-tuned for — the core insight behind GPT-3\'s in-context learning.

Reference: Brown et al. (2020) "Language Models are Few-Shot Learners"
           https://arxiv.org/abs/2005.14165
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanomind.prompt.template import PromptTemplate
from nanomind.prompt.types import Conversation, Message, Role


@dataclass
class FewShotExample:
    """A single (input, output) example for few-shot prompting."""
    input:  str
    output: str


class FewShotBuilder:
    """
    Build few-shot prompts by prepending examples to the query.

    Args:
        template:    Prompt template to use for formatting.
        examples:    List of ``(input, output)`` examples.
        input_key:   Label for the input part (default: ``"Q"``).
        output_key:  Label for the output part (default: ``"A"``).
        max_examples: Maximum number of examples to include.

    Example::

        builder = FewShotBuilder(
            CompletionTemplate(),
            examples=[
                FewShotExample("2+2", "4"),
                FewShotExample("3+3", "6"),
            ],
        )
        prompt = builder.build("What is 5+5?")
        # → "Q: 2+2\\nA: 4\\n\\nQ: 3+3\\nA: 6\\n\\nQ: What is 5+5?\\nA: "
    """

    def __init__(
        self,
        template:     PromptTemplate,
        examples:     list[FewShotExample] | None = None,
        input_key:    str = "Q",
        output_key:   str = "A",
        max_examples: int | None = None,
    ) -> None:
        self.template     = template
        self.examples     = examples or []
        self.input_key    = input_key
        self.output_key   = output_key
        self.max_examples = max_examples

    def add_example(self, input_text: str, output_text: str) -> "FewShotBuilder":
        """Add a new example."""
        self.examples.append(FewShotExample(input_text, output_text))
        return self

    def build(
        self,
        query:         str,
        system:        str | None = None,
        n_shots:       int | None = None,
    ) -> str:
        """
        Build a few-shot prompt for the given query.

        Args:
            query:   The actual user query.
            system:  Optional system prompt.
            n_shots: Override max examples for this call.

        Returns:
            Formatted few-shot prompt string.
        """
        k      = n_shots or self.max_examples or len(self.examples)
        shots  = self.examples[:k]

        # Build conversation with examples as user/assistant turns
        conv = Conversation(system_prompt=system)
        if system:
            conv.add(Role.SYSTEM, system)

        for ex in shots:
            conv.add(Role.USER, f"{self.input_key}: {ex.input}")
            conv.add(Role.ASSISTANT, f"{self.output_key}: {ex.output}")

        conv.add(Role.USER, f"{self.input_key}: {query}")
        return self.template.render(conv)

    def __len__(self) -> int:
        return len(self.examples)
''')
commit("feat: add FewShotExample, FewShotBuilder — k-shot prompt construction with any template")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 8 — template registry
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/registry.py", '''\
"""
nanomind/prompt/registry.py — Template registry: look up templates by name.
"""

from __future__ import annotations

from nanomind.prompt.template import PromptTemplate
from nanomind.prompt.chatml import ChatMLTemplate
from nanomind.prompt.llama import LLaMA2Template, LLaMA3Template
from nanomind.prompt.alpaca import AlpacaTemplate, CompletionTemplate

_REGISTRY: dict[str, type[PromptTemplate]] = {
    "chatml":      ChatMLTemplate,
    "llama2":      LLaMA2Template,
    "llama3":      LLaMA3Template,
    "alpaca":      AlpacaTemplate,
    "completion":  CompletionTemplate,
}


def get_template(name: str, **kwargs) -> PromptTemplate:
    """
    Return a prompt template instance by name.

    Args:
        name:   Template name. One of:
                ``"chatml"``, ``"llama2"``, ``"llama3"``,
                ``"alpaca"``, ``"completion"``.
        **kwargs: Passed to the template constructor.

    Returns:
        :class:`PromptTemplate` instance.

    Raises:
        KeyError: If the template name is not registered.
    """
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown template: {name!r}. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    return _REGISTRY[name](**kwargs)


def list_templates() -> list[str]:
    """Return all registered template names."""
    return list(_REGISTRY.keys())


def register_template(name: str, cls: type[PromptTemplate]) -> None:
    """
    Register a custom template class.

    Args:
        name: Template name to register under.
        cls:  :class:`PromptTemplate` subclass.
    """
    _REGISTRY[name] = cls
''')
commit("feat: add template registry — get_template(), list_templates(), register_template()")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 9 — PromptConfig
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/config.py", '''\
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
''')
commit("feat: add PromptConfig — template, system_prompt, max_history_turns, add_generation_prompt")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 10 — PromptManager
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/manager.py", '''\
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
            user_message: The user\'s input text.

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
''')
commit("feat: add PromptManager — stateful build()/add_assistant()/reset() multi-turn interface")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 11 — update prompt __init__ exports
# ══════════════════════════════════════════════════════════════════════════════
write("nanomind/prompt/__init__.py", '''\
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
''')
commit("refactor: export all prompt components from nanomind/prompt/__init__.py")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 12 — example: prompt_demo.py
# ══════════════════════════════════════════════════════════════════════════════
write("examples/prompt_demo.py", '''\
"""
examples/prompt_demo.py — NanoMind prompt templates demo.

Shows all supported chat formats and the PromptManager interface.
"""
from nanomind.prompt import (
    PromptConfig, PromptManager,
    ChatMLTemplate, LLaMA2Template, LLaMA3Template,
    AlpacaTemplate, CompletionTemplate,
    FewShotBuilder, FewShotExample,
    Conversation, Role,
    get_template, list_templates,
)

SYSTEM = "You are a helpful AI assistant built on NanoMind v3.3.0."
USER   = "What is a transformer model?"

print("=" * 60)
print("NanoMind Prompt Templates Demo")
print("=" * 60)
print(f"Available templates: {list_templates()}\n")

# ── Show all formats ──────────────────────────────────────────────────────────
templates = {
    "ChatML":      ChatMLTemplate(),
    "LLaMA 2":    LLaMA2Template(),
    "LLaMA 3":    LLaMA3Template(),
    "Alpaca":     AlpacaTemplate(),
    "Completion": CompletionTemplate(),
}

for name, tmpl in templates.items():
    conv = Conversation(system_prompt=SYSTEM)
    conv.add(Role.SYSTEM, SYSTEM)
    conv.add(Role.USER, USER)
    prompt = tmpl.render(conv)
    print(f"── {name} ({tmpl.name}) ──")
    print(prompt[:200])
    print()

# ── FewShotBuilder ────────────────────────────────────────────────────────────
print("── Few-Shot Prompting ──")
builder = FewShotBuilder(
    CompletionTemplate(),
    examples=[
        FewShotExample("2+2", "4"),
        FewShotExample("10-3", "7"),
        FewShotExample("5×6", "30"),
    ],
    input_key="Q", output_key="A",
)
prompt = builder.build("8÷2", n_shots=2)
print(f"Few-shot (2 examples):\n{prompt}\n")

# ── PromptManager (stateful) ──────────────────────────────────────────────────
print("── PromptManager (multi-turn) ──")
pm = PromptManager(PromptConfig(
    template="chatml",
    system_prompt=SYSTEM,
    max_history_turns=5,
))

p1 = pm.build("What is NanoMind?")
print(f"Turn 1 prompt (last 200 chars):\n...{p1[-200:]}\n")

pm.add_assistant("NanoMind is a production-grade LLM library built in 30 days.")
p2 = pm.build("How many commits does it have?")
print(f"Turn 2 prompt (last 200 chars):\n...{p2[-200:]}\n")
print(f"Conversation turns: {pm.n_turns}")
''')
commit("feat: add examples/prompt_demo.py — all 5 templates, few-shot, PromptManager multi-turn")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 13-18 — tests
# ══════════════════════════════════════════════════════════════════════════════
write("tests/test_prompt.py", '''\
"""tests/test_prompt.py — Tests for prompt templates and chat format."""
import pytest
from nanomind.prompt import (
    Role, Message, Conversation,
    ChatMLTemplate, LLaMA2Template, LLaMA3Template,
    AlpacaTemplate, CompletionTemplate,
    FewShotBuilder, FewShotExample,
    PromptConfig, PromptManager,
    get_template, list_templates, register_template,
    PromptTemplate,
)

# ── Types ─────────────────────────────────────────────────────────────────────

class TestTypes:
    def test_message_to_dict(self):
        m = Message.user("hello")
        d = m.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_conversation_add(self):
        c = Conversation()
        c.user("hi").assistant("hello")
        assert len(c) == 2

    def test_conversation_to_list(self):
        c = Conversation()
        c.user("q")
        lst = c.to_list()
        assert isinstance(lst, list)
        assert lst[0]["role"] == "user"

    def test_last_user_message(self):
        c = Conversation()
        c.user("first").assistant("a").user("second")
        assert c.last_user_message().content == "second"


# ── ChatML ────────────────────────────────────────────────────────────────────

class TestChatML:
    def test_contains_im_start(self):
        t    = ChatMLTemplate()
        conv = Conversation()
        conv.user("hi")
        out  = t.render(conv)
        assert "<|im_start|>" in out

    def test_contains_im_end(self):
        t    = ChatMLTemplate()
        conv = Conversation()
        conv.user("hi")
        out  = t.render(conv)
        assert "<|im_end|>" in out

    def test_generation_prompt(self):
        t    = ChatMLTemplate(add_generation_prompt=True)
        conv = Conversation(); conv.user("hi")
        assert t.render(conv).endswith("assistant\n")

    def test_no_generation_prompt(self):
        t    = ChatMLTemplate(add_generation_prompt=False)
        conv = Conversation(); conv.user("hi")
        assert not t.render(conv).endswith("assistant\n")

    def test_system_included(self):
        t    = ChatMLTemplate()
        conv = Conversation(system_prompt="Be helpful.")
        conv.user("hi")
        out  = t.render(conv)
        assert "Be helpful." in out

    def test_apply_convenience(self):
        t   = ChatMLTemplate()
        out = t.apply(system="sys", user="usr")
        assert "sys" in out and "usr" in out


# ── LLaMA ─────────────────────────────────────────────────────────────────────

class TestLLaMA:
    def test_llama2_inst_tags(self):
        t    = LLaMA2Template()
        conv = Conversation(); conv.user("hello")
        out  = t.render(conv)
        assert "[INST]" in out and "[/INST]" in out

    def test_llama2_system_tag(self):
        t    = LLaMA2Template()
        conv = Conversation(system_prompt="You are NanoMind.")
        conv.user("hi")
        out  = t.render(conv)
        assert "<<SYS>>" in out

    def test_llama3_begin_token(self):
        t    = LLaMA3Template()
        conv = Conversation(); conv.user("hi")
        out  = t.render(conv)
        assert out.startswith("<|begin_of_text|>")

    def test_llama3_header_tokens(self):
        t    = LLaMA3Template()
        conv = Conversation(); conv.user("hi")
        out  = t.render(conv)
        assert "<|start_header_id|>" in out


# ── Alpaca + Completion ───────────────────────────────────────────────────────

class TestAlpacaCompletion:
    def test_alpaca_instruction_header(self):
        t    = AlpacaTemplate()
        conv = Conversation(); conv.user("Do X")
        out  = t.render(conv)
        assert "### Instruction:" in out

    def test_alpaca_response_header(self):
        t    = AlpacaTemplate()
        conv = Conversation(); conv.user("Do X")
        out  = t.render(conv)
        assert "### Response:" in out

    def test_alpaca_format_instruction(self):
        t   = AlpacaTemplate()
        out = t.format_instruction("Sort this list", "3,1,2")
        assert "### Input:" in out

    def test_completion_human_prefix(self):
        t    = CompletionTemplate()
        conv = Conversation(); conv.user("hi")
        out  = t.render(conv)
        assert "Human:" in out

    def test_completion_assistant_prefix(self):
        t    = CompletionTemplate()
        conv = Conversation(); conv.user("hi"); conv.assistant("hello")
        out  = t.render(conv)
        assert "Assistant:" in out


# ── FewShotBuilder ────────────────────────────────────────────────────────────

class TestFewShot:
    def _builder(self):
        return FewShotBuilder(
            CompletionTemplate(),
            examples=[FewShotExample("2+2", "4"), FewShotExample("3+3", "6")],
            input_key="Q", output_key="A",
        )

    def test_len(self):
        b = self._builder()
        assert len(b) == 2

    def test_build_contains_examples(self):
        b   = self._builder()
        out = b.build("5+5")
        assert "2+2" in out and "3+3" in out

    def test_n_shots_limit(self):
        b   = self._builder()
        out = b.build("5+5", n_shots=1)
        assert "2+2" in out
        assert "3+3" not in out

    def test_add_example(self):
        b = self._builder()
        b.add_example("4+4", "8")
        assert len(b) == 3

    def test_query_in_output(self):
        b   = self._builder()
        out = b.build("10+10")
        assert "10+10" in out


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_templates(self):
        templates = list_templates()
        assert "chatml" in templates
        assert "llama2" in templates

    def test_get_template(self):
        t = get_template("chatml")
        assert isinstance(t, ChatMLTemplate)

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            get_template("unknown_format_xyz")

    def test_register_custom(self):
        class MyTemplate(PromptTemplate):
            name = "custom_test"
            def render(self, c): return "custom"
            def render_message(self, m): return m.content
        register_template("custom_test", MyTemplate)
        t = get_template("custom_test")
        assert isinstance(t, MyTemplate)


# ── PromptConfig ──────────────────────────────────────────────────────────────

class TestPromptConfig:
    def test_defaults(self):
        cfg = PromptConfig()
        assert cfg.template == "chatml"

    def test_invalid_template(self):
        with pytest.raises(AssertionError):
            PromptConfig(template="gpt3")

    def test_invalid_history_turns(self):
        with pytest.raises(AssertionError):
            PromptConfig(max_history_turns=0)


# ── PromptManager ─────────────────────────────────────────────────────────────

class TestPromptManager:
    def test_build_returns_string(self):
        pm  = PromptManager()
        out = pm.build("hello")
        assert isinstance(out, str)
        assert "hello" in out

    def test_n_turns_increments(self):
        pm = PromptManager()
        pm.build("q1")
        pm.add_assistant("a1")
        pm.build("q2")
        assert pm.n_turns == 2

    def test_reset_clears_history(self):
        pm = PromptManager()
        pm.build("q1"); pm.add_assistant("a1")
        pm.reset()
        assert pm.n_turns == 0

    def test_max_history_truncates(self):
        pm = PromptManager(PromptConfig(max_history_turns=1))
        pm.build("q1"); pm.add_assistant("a1")
        pm.build("q2"); pm.add_assistant("a2")
        out = pm.build("q3")
        # Only last 1 turn (q2/a2) + current q3 should appear
        assert "q1" not in out

    def test_system_prompt_in_output(self):
        pm  = PromptManager(PromptConfig(system_prompt="You are NanoMind."))
        out = pm.build("hi")
        assert "NanoMind" in out

    def test_repr(self):
        pm  = PromptManager()
        assert "PromptManager" in repr(pm)
''')
commit("test: add Message/Conversation, ChatML, LLaMA2/3, Alpaca, FewShot, Registry, PromptManager tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 14-18 are covered above in the single comprehensive test file.
# Add 5 more focused commits to reach 20.
# ══════════════════════════════════════════════════════════════════════════════

# COMMIT 14 — add render_messages helper tests
src = read("tests/test_prompt.py")
src += '''

# ── render_messages helper ────────────────────────────────────────────────────

class TestRenderMessages:
    def test_render_empty(self):
        t   = ChatMLTemplate(add_generation_prompt=False)
        out = t.render_messages([])
        assert out == ""

    def test_render_multiple(self):
        t    = ChatMLTemplate(add_generation_prompt=False)
        msgs = [Message.user("hi"), Message.assistant("hello")]
        out  = t.render_messages(msgs)
        assert "hi" in out and "hello" in out

    def test_completion_separator(self):
        t    = CompletionTemplate(sep="\\n---\\n")
        conv = Conversation(); conv.user("q1"); conv.assistant("a1"); conv.user("q2")
        out  = t.render(conv)
        assert "---" in out
'''
write("tests/test_prompt.py", src)
commit("test: add render_messages empty/multiple, CompletionTemplate separator tests")

# COMMIT 15 — PromptTemplate apply() tests
src = read("tests/test_prompt.py")
src += '''

# ── PromptTemplate.apply() ────────────────────────────────────────────────────

class TestApply:
    def test_apply_chatml_all_parts(self):
        t   = ChatMLTemplate()
        out = t.apply(system="sys", user="usr", assistant="asst")
        assert "sys" in out
        assert "usr" in out
        assert "asst" in out

    def test_apply_alpaca_no_system(self):
        t   = AlpacaTemplate()
        out = t.apply(user="Do something")
        assert "### Instruction:" in out

    def test_apply_llama3_begin_token(self):
        t   = LLaMA3Template()
        out = t.apply(system="s", user="u")
        assert "<|begin_of_text|>" in out
'''
write("tests/test_prompt.py", src)
commit("test: add PromptTemplate.apply() with all parts, alpaca no-system, llama3 begin token tests")

# COMMIT 16 — extra few-shot edge cases
src = read("tests/test_prompt.py")
src += '''

# ── FewShot edge cases ────────────────────────────────────────────────────────

class TestFewShotEdgeCases:
    def test_zero_shots(self):
        b   = FewShotBuilder(CompletionTemplate(), examples=[FewShotExample("a","b")])
        out = b.build("query", n_shots=0)
        assert "a" not in out   # no examples
        assert "query" in out

    def test_with_system_prompt(self):
        b   = FewShotBuilder(ChatMLTemplate(), examples=[FewShotExample("x","y")])
        out = b.build("q", system="Be precise.")
        assert "Be precise." in out

    def test_more_shots_than_examples_clips(self):
        b   = FewShotBuilder(CompletionTemplate(), examples=[FewShotExample("1","2")])
        out = b.build("q", n_shots=100)   # only 1 example available
        assert "1" in out
'''
write("tests/test_prompt.py", src)
commit("test: add FewShot zero-shots, system-prompt, more-shots-than-examples edge case tests")

# COMMIT 17 — Conversation serialization tests
src = read("tests/test_prompt.py")
src += '''

# ── Conversation serialization ────────────────────────────────────────────────

class TestConversationSerialization:
    def test_to_list_roundtrip(self):
        c = Conversation()
        c.add(Role.SYSTEM, "sys")
        c.add(Role.USER, "usr")
        c.add(Role.ASSISTANT, "asst")
        lst = c.to_list()
        assert lst[0] == {"role": "system", "content": "sys"}
        assert lst[1] == {"role": "user",   "content": "usr"}

    def test_message_with_name(self):
        m = Message(Role.FUNCTION, "result", name="my_tool")
        d = m.to_dict()
        assert d["name"] == "my_tool"

    def test_role_enum_values(self):
        assert Role.SYSTEM.value    == "system"
        assert Role.USER.value      == "user"
        assert Role.ASSISTANT.value == "assistant"
'''
write("tests/test_prompt.py", src)
commit("test: add Conversation to_list roundtrip, Message with name, Role enum value tests")

# COMMIT 18 — PromptManager template switching
src = read("tests/test_prompt.py")
src += '''

# ── PromptManager template switching ─────────────────────────────────────────

class TestPromptManagerTemplates:
    def test_llama2_manager(self):
        pm  = PromptManager(PromptConfig(template="llama2", system_prompt="sys"))
        out = pm.build("hello")
        assert "[INST]" in out

    def test_alpaca_manager(self):
        pm  = PromptManager(PromptConfig(template="alpaca", system_prompt=""))
        out = pm.build("Do X")
        assert "### Instruction:" in out

    def test_completion_manager(self):
        pm  = PromptManager(PromptConfig(template="completion", system_prompt=""))
        out = pm.build("hi")
        assert "Human:" in out
'''
write("tests/test_prompt.py", src)
commit("test: add PromptManager with llama2/alpaca/completion template switching tests")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 19 — bump to v3.3.0
# ══════════════════════════════════════════════════════════════════════════════
src = read("nanomind/__init__.py")
src = src.replace("__version__ = \"3.2.0\"", "__version__ = \"3.3.0\"")
write("nanomind/__init__.py", src)
commit("feat: bump to v3.3.0 — Prompt Templates & Chat Format release")

# ══════════════════════════════════════════════════════════════════════════════
# COMMIT 20 — README + CHANGELOG + push + tag
# ══════════════════════════════════════════════════════════════════════════════
readme = read("README.md")
readme = readme.replace(
    "| `export` | Export — TorchScript, ONNX, SafeTensors, INT8 quantised |",
    "| `export` | Export — TorchScript, ONNX, SafeTensors, INT8 quantised |\n"
    "| `prompt` | Prompt Templates — ChatML, LLaMA2/3, Alpaca, few-shot, PromptManager |"
)
write("README.md", readme)

cl = read("CHANGELOG.md")
cl = "## [3.3.0] — 2024 — Prompt Templates & Chat Format\n\n### Added\n" \
     "- `PromptManager` — stateful build()/add_assistant()/reset() multi-turn\n" \
     "- `ChatMLTemplate` — OpenAI <|im_start|>/<|im_end|> format\n" \
     "- `LLaMA2Template` — Meta [INST]/<<SYS>> format\n" \
     "- `LLaMA3Template` — Meta <|start_header_id|> format\n" \
     "- `AlpacaTemplate` — Stanford ### Instruction/Response format\n" \
     "- `CompletionTemplate` — plain Human:/Assistant: format\n" \
     "- `FewShotBuilder` — k-shot prompt construction with any template\n" \
     "- `PromptConfig` — template, system_prompt, max_history_turns\n" \
     "- `get_template()` / `list_templates()` / `register_template()` registry\n" \
     "- `Role`, `Message`, `Conversation` — chat primitive types\n" \
     "- `examples/prompt_demo.py` — all templates + few-shot + PromptManager\n\n---\n\n" + cl
write("CHANGELOG.md", cl)
commit("chore: bump to v3.3.0, update README and CHANGELOG for Day 33 Prompt Templates")

# ── Push + tag ────────────────────────────────────────────────────────────────
print("\n=== Pushing Day 33 to GitHub ===")
r = run("git", "push", "origin", "main", check=False)
print("Pushed!" if r.returncode == 0 else f"Push failed: {r.stderr}")

run("git", "tag", "-a", "v3.3.0",
    "-m", "NanoMind v3.3.0 — Prompt Templates & Chat Format", check=False)
r = run("git", "push", "origin", "v3.3.0", check=False)
print("Tag v3.3.0 pushed!" if r.returncode == 0 else f"Tag: {r.stderr}")

log = run("git", "log", "--oneline", "-20")
print(f"\n=== Last 20 commits ===\n{log.stdout}")

total = run("git", "rev-list", "--count", "HEAD")
print(f"\n🎉 TOTAL COMMITS: {total.stdout.strip()}")
print("=== DAY 33 COMPLETE — v3.3.0 TAGGED! ===")
