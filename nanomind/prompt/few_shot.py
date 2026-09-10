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

This dramatically improves model performance on tasks it hasn't been
explicitly fine-tuned for — the core insight behind GPT-3's in-context learning.

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
        # → "Q: 2+2\nA: 4\n\nQ: 3+3\nA: 6\n\nQ: What is 5+5?\nA: "
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
