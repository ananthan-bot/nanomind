"""
nanomind/safety/taxonomy.py — Safety taxonomy categories, threat vectors, and policy mappings.
"""
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


class SafetyCategory(str, Enum):
    """Standard safety taxonomy categories (Llama Guard / MLCommons compatible)."""
    SAFE = "safe"
    HATE_SPEECH = "hate_speech"
    HARASSMENT = "harassment"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    CYBERATTACKS = "cyberattacks"
    CBRN = "cbrn"  # Chemical, Biological, Radiological, Nuclear
    PROMPT_INJECTION = "prompt_injection"
    PII_LEAKAGE = "pii_leakage"


@dataclass
class SafetyPolicy:
    """Policy rules governing threshold sensitivity and refusal actions per category."""
    thresholds: Dict[SafetyCategory, float] = None
    default_refusal_template: str = "I am unable to assist with {category} content."

    def __post_init__(self):
        if self.thresholds is None:
            self.thresholds = {
                SafetyCategory.HATE_SPEECH: 0.5,
                SafetyCategory.HARASSMENT: 0.5,
                SafetyCategory.VIOLENCE: 0.4,
                SafetyCategory.SELF_HARM: 0.2,  # Strictest threshold
                SafetyCategory.SEXUAL_CONTENT: 0.5,
                SafetyCategory.CYBERATTACKS: 0.4,
                SafetyCategory.CBRN: 0.1,        # Zero-tolerance
                SafetyCategory.PROMPT_INJECTION: 0.6,
                SafetyCategory.PII_LEAKAGE: 0.5,
            }

    def is_violation(self, category: SafetyCategory, score: float) -> bool:
        """True if score exceeds safety threshold for category."""
        thresh = self.thresholds.get(category, 0.5)
        return score >= thresh

    def format_refusal(self, category: SafetyCategory) -> str:
        cat_human = category.value.replace("_", " ")
        return self.default_refusal_template.format(category=cat_human)
