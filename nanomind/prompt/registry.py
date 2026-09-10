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
