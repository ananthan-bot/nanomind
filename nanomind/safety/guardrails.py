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


class InputGuardrail:
    """Pre-execution input filter preventing adversarial injection and PII leakage."""

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self.pii_redactor = PIIRedactor()
        self.injection_detector = PromptInjectionDetector()

    def process(self, prompt: str) -> Dict[str, Any]:
        """
        Evaluates prompt against input safety filters.
        """
        is_blocked = False
        reasons = []
        cleaned_prompt = prompt

        # 1. Prompt Injection
        if self.config.enable_prompt_injection_detection:
            is_inj, score, triggers = self.injection_detector.detect(prompt)
            if is_inj and score >= self.config.max_toxicity_threshold:
                is_blocked = True
                reasons.append(f"Prompt injection detected ({len(triggers)} triggers)")

        # 2. PII Redaction
        if self.config.enable_pii_redaction:
            cleaned_prompt, pii_found = self.pii_redactor.redact(cleaned_prompt)
            if pii_found and self.config.action_on_violation == "block":
                is_blocked = True
                reasons.append("Unpermitted PII detected in prompt")

        return {
            "is_blocked": is_blocked,
            "processed_text": cleaned_prompt,
            "reasons": reasons,
            "refusal": self.config.refusal_message if is_blocked else None,
        }


class OutputGuardrail:
    """Post-execution output filter verifying response safety and redaction."""

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self.pii_redactor = PIIRedactor()

    def process(self, response: str) -> Dict[str, Any]:
        is_blocked = False
        reasons = []
        cleaned_response = response

        # Ensure model did not emit PII in its generation
        if self.config.enable_pii_redaction:
            cleaned_response, pii_found = self.pii_redactor.redact(cleaned_response)

        return {
            "is_blocked": is_blocked,
            "processed_text": cleaned_response,
            "reasons": reasons,
        }


class GuardrailPipeline:
    """Wraps full LLM call lifecycle with input screening and output validation."""

    def __init__(self, config: Optional[GuardrailConfig] = None):
        self.config = config or GuardrailConfig()
        self.input_rail = InputGuardrail(self.config)
        self.output_rail = OutputGuardrail(self.config)

    def screen_input(self, text: str) -> Dict[str, Any]:
        return self.input_rail.process(text)

    def screen_output(self, text: str) -> Dict[str, Any]:
        return self.output_rail.process(text)
