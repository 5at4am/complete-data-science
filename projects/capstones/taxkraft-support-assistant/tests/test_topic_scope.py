from __future__ import annotations

from taxkraft_assistant.guardrails.topic_scope import TopicScopeGuard
from taxkraft_assistant.schemas import GuardType


def test_in_scope_queries_pass():
    g = TopicScopeGuard(threshold=0.1)
    for q in [
        "Does TaxKraft offer GST registration?",
        "How do I contact TaxKraft?",
        "What is the TaxKraft phone number?",
        "Can TaxKraft register my private limited company?",
        "Does TaxKraft file income tax returns?",
    ]:
        assert g.check(q).passed, f"expected in-scope: {q!r}"


def test_obviously_out_of_scope_fail():
    g = TopicScopeGuard(threshold=0.7)
    for q in [
        "What is the weather in Delhi?",
        "Tell me a joke",
        "Explain the plot of Oppenheimer",
        "How do I cook pasta?",
    ]:
        assert not g.check(q).passed, f"expected deflected: {q!r}"


def test_competitor_comparison_is_deflected():
    g = TopicScopeGuard(threshold=0.1)
    r = g.check("Compare ClearTax with TaxKraft for GST filing")
    assert not r.passed, "competitor comparison must be deflected"


def test_keyword_score_bounds():
    g = TopicScopeGuard()
    assert 0.0 <= g.keyword_score("random non tax stuff") <= 1.0
    assert g.keyword_score("TaxKraft GST registration for my business") > 0.0


def test_result_schema():
    r = TopicScopeGuard(threshold=0.9).check("Who is TaxKraft?")
    assert r.name == GuardType.TOPIC_SCOPE
    assert hasattr(r, "score") and hasattr(r, "threshold")