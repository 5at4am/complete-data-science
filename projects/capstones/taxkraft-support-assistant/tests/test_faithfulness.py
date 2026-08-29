from __future__ import annotations

from taxkraft_assistant.guardrails.faithfulness import FaithfulnessGuard


def test_grounded_answer_passes():
    g = FaithfulnessGuard(threshold=0.30)
    context = (
        "TaxKraft is a CA firm in India. It files GST returns monthly and quarterly. "
        "Contact TaxKraft at +91-8608601620 or info@taxkraft.com."
    )
    answer = (
        "TaxKraft is a CA firm in India that files GST returns monthly and quarterly. "
        "You can contact TaxKraft at +91-8608601620."
    )
    r = g.check(answer, context)
    assert r.passed, f"score={r.score}"


def test_hallucinated_answer_fails():
    g = FaithfulnessGuard(threshold=0.60)
    context = "TaxKraft files GST returns in India."
    answer = "The Bharat moon mission reached Mars and spicy noodles cure colds."
    r = g.check(answer, context)
    assert not r.passed, f"score={r.score}"


def test_empty_answer_fails():
    g = FaithfulnessGuard()
    assert not g.check("", "some context here for a grounding test").passed