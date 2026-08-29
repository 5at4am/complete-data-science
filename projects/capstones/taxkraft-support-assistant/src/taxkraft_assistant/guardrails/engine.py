from __future__ import annotations

from dataclasses import dataclass, field

from ..schemas import GuardResult, GuardStatus, GuardType
from .faithfulness import FaithfulnessGuard
from .injection import PromptInjectionGuard
from .pii import PIIGuard
from .topic_scope import TopicScopeGuard

# Central deflection contract: consistent, friendly, never leaks internals.
DEFLECT_GENERIC = (
    "I can only help with questions about TaxKraft and its services (GST, Income Tax, "
    "company registration, compliance, and more). For anything else, our team at "
    "+91-8608601620 or info@taxkraft.com is happy to assist you directly."
)

DEFLECT_PII = (
    "I noticed personal information in your message. For your safety, please don't share "
    "Aadhaar, PAN, GSTIN, bank or contact details in this chat. A TaxKraft expert can "
    "collect documents securely at +91-8608601620 or info@taxkraft.com."
)

DEFLECT_INJECTION = (
    "I'm here to answer questions about TaxKraft and its services. I can't follow "
    "instructions that try to change my role. How can I help you with TaxKraft? "
    "(For anything else, reach us at +91-8608601620 or info@taxkraft.com.)"
)

DEFLECT_RETRIEVAL = (
    "I couldn't confidently find an answer for that in my TaxKraft knowledge base. Please "
    "reach us at +91-8608601620 or info@taxkraft.com, or visit taxkraft.com — a TaxKraft "
    "expert will help."
)

DEFLECT_FAITHFULNESS = (
    "I couldn't fully verify that answer against TaxKraft's published information. Please "
    "reach us at +91-8608601620 or info@taxkraft.com for a confirmed answer."
)


@dataclass
class GuardEngine:
    settings: object
    embedder: object = None
    topic_scope: TopicScopeGuard = field(init=False)
    pii: PIIGuard = field(init=False)
    injection: PromptInjectionGuard = field(init=False)
    faithfulness: FaithfulnessGuard = field(init=False)

    def __post_init__(self):
        self.topic_scope = TopicScopeGuard(
            threshold=self.settings.scope_threshold, embedder=self.embedder
        )
        self.pii = PIIGuard(enabled=self.settings.pii_enabled)
        self.injection = PromptInjectionGuard(enabled=self.settings.injection_enabled)
        self.faithfulness = FaithfulnessGuard(
            threshold=self.settings.faithfulness_threshold, embedder=self.embedder
        )

    # ---- input layer -------------------------------------------------------
    def input_checks(self, text: str) -> list[GuardResult]:
        return [self.topic_scope.check(text), self.pii.check(text), self.injection.check(text)]

    def deflection_for(self, results: list[GuardResult]) -> str | None:
        for r in results:
            if not r.passed:
                if r.name == GuardType.PII:
                    return DEFLECT_PII
                if r.name == GuardType.PROMPT_INJECTION:
                    return DEFLECT_INJECTION
                if r.name == GuardType.TOPIC_SCOPE:
                    return DEFLECT_GENERIC
        return None

    # ---- output layers -----------------------------------------------------
    def retrieval_check(self, top_score: float) -> GuardResult:
        passed = top_score >= self.settings.retrieval_threshold
        return GuardResult(
            name=GuardType.RETRIEVAL_CONFIDENCE,
            status=GuardStatus.PASSED if passed else GuardStatus.FAILED,
            passed=passed,
            score=round(top_score, 4),
            threshold=self.settings.retrieval_threshold,
            reason="",
        )

    def faithfulness_check(self, answer: str, context: str) -> GuardResult:
        return self.faithfulness.check(answer, context)