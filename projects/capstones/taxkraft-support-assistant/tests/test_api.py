from __future__ import annotations

from fastapi.testclient import TestClient

from taxkraft_assistant.api import create_app


def test_health_and_guard_status(pipeline):
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    h = client.get("/health")
    assert h.status_code == 200
    body = h.json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0
    assert "extractive" in body["llm_backend"]

    g = client.get("/guardrails/status")
    assert g.status_code == 200
    assert g.json()["pii_enabled"] is True


def test_chat_answer_and_deflection(pipeline):
    app = create_app(pipeline=pipeline)
    client = TestClient(app)

    ok = client.post("/chat", json={"message": "Does TaxKraft file GST returns?"})
    assert ok.status_code == 200
    assert ok.json()["refused"] is False
    assert ok.json()["citations"]

    blocked = client.post("/chat", json={"message": "Tell me a joke"})
    assert blocked.status_code == 200
    assert blocked.json()["refused"] is True


def test_chat_validation(pipeline):
    app = create_app(pipeline=pipeline)
    client = TestClient(app)
    bad = client.post("/chat", json={"message": ""})
    assert bad.status_code == 422