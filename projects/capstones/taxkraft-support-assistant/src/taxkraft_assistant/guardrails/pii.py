from __future__ import annotations

import re

from ..schemas import GuardResult, GuardStatus, GuardType

# Ordered patterns. Order matters: check most specific first. Patterns are grouped
# so a single match returns a readable reason.
PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    (
        "aadhaar",
        re.compile(
            r"(?<!\d)(?:\d{4}[\s-]?){2}\d{4}(?!\d)"
            r"|(?<!\d)\d{12}(?!\d)",
        ),
    ),
    (
        "pan",
        re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    ),
    (
        "gstin",
        re.compile(r"\b\d{2}[A-Z]{5}\d{4}[A-Z]\d[Z][A-Z\d]\b"),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{2}[\s-]?\d{3}[\s-]?\d{3}(?!\d)"
        ),
    ),
    (
        "email",
        re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    ),
    (
        "upi_id",
        re.compile(r"\b[\w.-]{2,}@(?:upi|ybl|oksbi|axl|ibl|paytm|apl|abfsp|sbi|ibl|okicici|okhdfcbank|okaxis)\b"),
    ),
    (
        "bank_account",
        re.compile(r"(?<![\dA-Za-z])(?:[0-9]{9,18})(?![\dA-Za-z])"),
    ),
    (
        "passport",
        re.compile(r"\b[A-Z][1-9]\d{6}\b"),
    ),
    (
        "driver_license",
        re.compile(r"\b\d{2}\s?[A-Z]{2}\s?\d{6,10}\b"),
    ),
]

# Phrases that reveal intent to *provide* PII even without the literal value.
SENSITIVE_INTENT = [
    "my aadhaar",
    "my pan card",
    "my pan number",
    "my gstin",
    "my phone number",
    "my mobile number",
    "my email",
    "my upi",
    "my bank account",
    "share my aadhaar",
    "share my pan",
    "upload my aadhaar",
    "my passport",
]


class PIIGuard:
    """Stops personal/sensitive data from entering the chatbot and never echoes it back."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def check(self, text: str) -> GuardResult:
        if not self.enabled:
            return GuardResult(
                name=GuardType.PII,
                status=GuardStatus.NOT_RUN,
                passed=True,
                score=0.0,
                reason="disabled",
            )
        hits: list[str] = []
        lower = text.lower()
        for label, pattern in PII_PATTERNS:
            if pattern.search(text):
                hits.append(label)
        for phrase in SENSITIVE_INTENT:
            if phrase in lower:
                hits.append(phrase)
        passing = not hits
        return GuardResult(
            name=GuardType.PII,
            status=GuardStatus.PASSED if passing else GuardStatus.FAILED,
            passed=passing,
            score=0.0 if passing else 1.0,
            reason="",
            detail="; ".join(sorted(set(hits))) if hits else None,
        )