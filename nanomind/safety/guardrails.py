"""
nanomind/safety/guardrails.py — Input and Output Guardrails, PII redaction, and Prompt Injection detection.
"""
import re
from typing import Dict, List, Tuple, Optional, Any
from nanomind.safety.config import GuardrailConfig
from nanomind.safety.taxonomy import SafetyCategory


class PIIRedactor:
    """
    Scans and redacts Personally Identifiable Information (PII) using regular expressions.
    Redacts: Emails, Phone Numbers, Credit Cards, IPv4 Addresses, SSNs.
    """

    PATTERNS = {
        "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
        "PHONE": r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}",
        "CREDIT_CARD": r"(?:\d{4}[-\s]?){3}\d{4}",
        "IPV4": r"(?:[0-9]{1,3}\.){3}[0-9]{1,3}",
        "SSN": r"\d{3}-\d{2}-\d{4}",
    }

    def redact(self, text: str, placeholder_style: str = "tag") -> Tuple[str, List[Dict[str, str]]]:
        """
        Returns redacted text and list of found PII entities.
        """
        redacted = text
        findings = []

        for pii_type, pattern in self.PATTERNS.items():
            for match in re.finditer(pattern, text):
                val = match.group(0)
                tag = f"<{pii_type}_REDACTED>" if placeholder_style == "tag" else "[REDACTED]"
                redacted = redacted.replace(val, tag)
                findings.append({"type": pii_type, "value": val, "replacement": tag})

        return redacted, findings


class PromptInjectionDetector:
    """
    Detects adversarial prompt injections, jailbreaks, and system prompt override attempts.
    """

    INJECTION_TRIGGERS = [
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"system\s*override",
        r"you\s+are\s+now\s+in\s+dan\s+mode",
        r"do\s+anything\s+now",
        r"disregard\s+the\s+above",
        r"reveal\s+your\s+system\s+prompt",
        r"base64\s+decode\s+and\s+execute",
        r"roleplay\s+as\s+an\s+unfiltered",
        r"new\s+system\s+instruction:",
    ]

    def detect(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Scans text for injection signatures.
        Returns: (is_injection, confidence_score, matched_patterns)
        """
        matched = []
        lower = text.lower()

        for pattern in self.INJECTION_TRIGGERS:
            if re.search(pattern, lower):
                matched.append(pattern)

        score = min(1.0, len(matched) * 0.5)
        return len(matched) > 0, score, matched
