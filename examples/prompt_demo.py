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
print(f"Available templates: {list_templates()}
")

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
print(f"Few-shot (2 examples):
{prompt}
")

# ── PromptManager (stateful) ──────────────────────────────────────────────────
print("── PromptManager (multi-turn) ──")
pm = PromptManager(PromptConfig(
    template="chatml",
    system_prompt=SYSTEM,
    max_history_turns=5,
))

p1 = pm.build("What is NanoMind?")
print(f"Turn 1 prompt (last 200 chars):
...{p1[-200:]}
")

pm.add_assistant("NanoMind is a production-grade LLM library built in 30 days.")
p2 = pm.build("How many commits does it have?")
print(f"Turn 2 prompt (last 200 chars):
...{p2[-200:]}
")
print(f"Conversation turns: {pm.n_turns}")
