"""
nanomind/codegen/generator.py — Code generation prompt builder and output parser.

## Code Generation Prompting

Different prompt styles for code generation:

1. Direct completion (Codex/StarCoder style):
   def add(a: int, b: int) -> int:
       """Return sum of a and b."""
       [MODEL COMPLETES HERE]

2. Instruction following (CodeLlama Instruct):
   [INST] Write a Python function that returns the sum of two integers [/INST]
   def add(a: int, b: int) -> int:

3. Fill-in-the-middle (FIM):
   <PRE> def add( <SUF> ) -> int:
    return a + b <MID>

4. Chain-of-thought:
   "Let me think step by step...
    First, I need to...
    The implementation is:
    ```python
    def add(...): ...
    ```"

## Code Extraction

LLMs often wrap code in markdown blocks:
   ```python
   def add(a, b):
       return a + b
   ```

The parser extracts the code from these blocks.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from nanomind.codegen.problem import CodeProblem, ProgrammingLanguage


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    max_tokens:     int   = 512
    temperature:    float = 0.2   # low temperature for code
    top_p:          float = 0.95
    n_samples:      int   = 1     # for pass@k evaluation
    stop_sequences: list  = None  # e.g., ["

", "def "]
    prompt_style:   str   = "completion"   # "completion" | "instruct" | "fim"

    def __post_init__(self):
        if self.stop_sequences is None:
            self.stop_sequences = []


class CodePromptBuilder:
    """
    Build prompts for code generation.

    Args:
        style: Prompt style — "completion", "instruct", or "fim".

    Example::

        builder = CodePromptBuilder(style="completion")
        prompt  = builder.build(problem)
        print(prompt)   # → function signature + docstring ready for completion
    """

    INSTRUCT_TEMPLATE = (
        "Write a Python function to solve the following problem:

"
        "{description}

"
        "The function signature is:
"
        "```python
{signature}
```

"
        "Provide only the complete function implementation."
    )

    COT_TEMPLATE = (
        "Solve this programming problem step by step:

"
        "{description}

"
        "Function signature: {signature}

"
        "Think through the solution:
"
        "1. Understand the problem
"
        "2. Consider edge cases
"
        "3. Write the implementation

"
        "```python
"
    )

    def __init__(self, style: str = "completion") -> None:
        assert style in ("completion", "instruct", "fim", "cot")
        self.style = style

    def build(self, problem: CodeProblem) -> str:
        """Build generation prompt for a code problem."""
        if self.style == "completion":
            return problem.prompt

        elif self.style == "instruct":
            return self.INSTRUCT_TEMPLATE.format(
                description = problem.description,
                signature   = problem.signature,
            )

        elif self.style == "fim":
            # Fill-in-the-middle format (StarCoder)
            prefix = f"{problem.signature}
    "
            suffix = ""
            return f"<fim_prefix>{prefix}<fim_suffix>{suffix}<fim_middle>"

        elif self.style == "cot":
            return self.COT_TEMPLATE.format(
                description = problem.description,
                signature   = problem.signature,
            )
        return problem.prompt

    def add_examples(self, prompt: str, examples: list[str]) -> str:
        """Add few-shot examples to a prompt."""
        ex_block = "

".join(examples)
        return f"# Examples:
{ex_block}

{prompt}"


class CodeOutputParser:
    """
    Parse LLM output to extract clean Python code.

    Handles markdown code blocks, raw code, and mixed output.

    Example::

        parser = CodeOutputParser()
        raw    = "Here is the solution:\n```python\ndef add(a,b):\n    return a+b\n```"
        code   = parser.parse(raw, entry_point="add")
        print(code)   # → "def add(a,b):\n    return a+b"
    """

    _MD_BLOCK  = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
    _FUNC_DEF  = re.compile(r"(def\s+\w+\s*\(.*?)(?=
def |\Z)", re.DOTALL)

    def parse(self, raw: str, entry_point: str = "") -> str:
        """
        Extract code from LLM output.

        Args:
            raw:         Raw LLM output string.
            entry_point: Function name to look for.

        Returns:
            Extracted code string.
        """
        # Try markdown code block first
        md = self._MD_BLOCK.search(raw)
        if md:
            return md.group(1).strip()

        # Try to find function definition
        if entry_point:
            fn_pat = re.compile(
                rf"(def\s+{re.escape(entry_point)}\s*\(.*?)(?=
def |\Z)",
                re.DOTALL
            )
            m = fn_pat.search(raw)
            if m:
                return m.group(1).strip()

        # Fall back: any function definition
        m = self._FUNC_DEF.search(raw)
        if m:
            return m.group(1).strip()

        # Last resort: return raw output stripped
        return raw.strip()

    def extract_all_functions(self, raw: str) -> list[str]:
        """Extract all function definitions from output."""
        return [m.group(1).strip() for m in self._FUNC_DEF.finditer(raw)]

    def has_code(self, raw: str) -> bool:
        """Check if output contains any code."""
        return bool(self._MD_BLOCK.search(raw) or self._FUNC_DEF.search(raw))
