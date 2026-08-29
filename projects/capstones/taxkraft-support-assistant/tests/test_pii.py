from __future__ import annotations

from taxkraft_assistant.guardrails.pii import PIIGuard


def test_detects_aadhaar():
    g = PIIGuard()
    assert not g.check("my aadhaar is 2341 5678 9012 for registration").passed


def test_detects_pan():
    g = PIIGuard()
    assert not g.check("gstin documents with PAN ABCDE1234F attached").passed


def test_detects_phone_and_email():
    g = PIIGuard()
    assert not g.check("call me at 9876543210").passed
    assert not g.check("send the notice to user@example.com").passed


def test_detects_upi():
    g = PIIGuard()
    assert not g.check("my UPI is raj@ybl").passed


def test_clean_question_passes():
    g = PIIGuard()
    assert g.check("What is the GST registration process?").passed


def test_disabled_guard_is_not_run():
    g = PIIGuard(enabled=False)
    r = g.check("my aadhaar is 2341 5678 9012")
    assert r.passed and r.status.value == "not_run"