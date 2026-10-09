"""
nanomind/safety/refusal.py — Refusal mechanisms and neutral tone calibration.
Ensures refusals are direct, polite, helpful, and never judgmental or preachy.
"""
from typing import Dict, Any, Optional
import re


class RefusalToneCalibrator:
    """
    Strips preachy, lecturing, or patronizing prefixes from safety refusal messages.
    """

    PREACHY_PATTERNS = [
        r"^as an ai language model,?\s*",
        r"^i must warn you that\s*",
        r"^it is unethical and wrong to\s*",
        r"^i strongly advise against\s*",
        r"^you should not be asking for\s*",
    ]

    @classmethod
    def clean_refusal(cls, refusal_text: str) -> str:
        """Normalize refusal message to concise, neutral tone."""
        cleaned = refusal_text.strip()
        for p in cls.PREACHY_PATTERNS:
            cleaned = re.sub(p, "", cleaned, flags=re.IGNORECASE)
        # Capitalize first letter
        if cleaned:
            cleaned = cleaned[0].upper() + cleaned[1:]
        return cleaned


class RefusalHandler:
    """
    Standardized refusal generator with optional benign pivot.
    """

    def __init__(self, default_refusal: str = "I cannot fulfill this request as it violates safety guidelines."):
        self.default_refusal = default_refusal

    def generate_refusal(self, reason: str = "", benign_alternative: Optional[str] = None) -> str:
        msg = f"I cannot assist with this request. {reason}".strip() if reason else self.default_refusal
        msg = RefusalToneCalibrator.clean_refusal(msg)
        if benign_alternative:
            msg += f" However, I can help you with {benign_alternative}."
        return msg
