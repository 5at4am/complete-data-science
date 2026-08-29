from __future__ import annotations

import re

from ..schemas import GuardResult, GuardStatus, GuardType

INJECTION_PATTERNS = [
    # direct instruction-override
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)?\s*(instructions|prompt|rules|messages)",
    r"disregard\s+(the\s+)?(previous|above)?\s*(instructions|prompt|system|rules)",
    r"forget\s+(all\s+)?(your\s+)?(instructions|prompt|rules|guidelines|persona)",
    r"(stop|cease)\s+(following|obeying|trying|to\s+follow)",
    r"do\s+not\s+follow\s+(your|the)\s+(instructions|prompt|rules)",
    r"override\s+(your|the|system)",
    r"release\s+(your|the)\s*(system\s*)?prompt",
    r"reveal\s+(your|the)\s*(system\s*)?prompt",
    r"print\s+(your|the)\s*(system\s*)?prompt",
    r"show\s+(me\s+)?(your|the)\s*(system\s*)?prompt",
    r"say\s+(your|the)\s*(system\s*)?prompt",
    r"what\s+are\s+your\s*(system\s*)?(instructions|prompt|rules)",
    r"system\s*prompt",
    r"hidden\s*(instructions?|prompt|preferences?)",
    r"developer\s*message",
    r"jailbreak|jail\s*break",
    r"do\s+anything\s+now|\bdan\b",
    r"you\s+are\s+now\s+(\w+\s+){0,3}(unfiltered|unconstrained|free)",
    r"act\s+as\s+if\s+you\s+have\s+no\s*(rules|restrictions|limits)",
    r"no\s+(restrictions?|rules|filters?|guardrails?)",
    r"no\s*(guardrail|filter|policy)",
    r"pretend\s+you\s+(don't|do\s+not)\s+have",
    r"override\s+filters?",
    r"simulate\s+unrestricted",
    r"unfiltered\s+mode|developer\s+mode|god\s+mode|sudo\s+mode",
    r"you\s+must\s+now\s+\w+",
    r"new\s+instructions\s*:",
    r"instructions\s+override",
    r"start\s+with\s+(certainly|of\s+course|sure)[\s,]+i",
    r"tell\s+me\s+your\s+persona",
    r"from\s+now\s+on\s+you\s+(are|will)",
    r"\bignore\b.{0,20}\b(user\s*input|above|prompt)",
    r"translate\s+to\s+english[\s\S]{0,20}ignore",
]


class PromptInjectionGuard:
    """Detects jailbreak / instruction-override / prompt-leak attempts."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._compiled = [re.compile(p, flags=re.IGNORECASE) for p in INJECTION_PATTERNS]

    def check(self, text: str) -> GuardResult:
        if not self.enabled:
            return GuardResult(
                name=GuardType.PROMPT_INJECTION,
                status=GuardStatus.NOT_RUN,
                passed=True,
                score=0.0,
                reason="disabled",
            )
        hits = [p.pattern for p in self._compiled if p.search(text)]
        passing = not hits
        return GuardResult(
            name=GuardType.PROMPT_INJECTION,
            status=GuardStatus.PASSED if passing else GuardStatus.FAILED,
            passed=passing,
            score=0.0 if passing else 1.0,
            reason="",
            detail="; ".join(hits[:3]) if hits else None,
        )