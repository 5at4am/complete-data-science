from __future__ import annotations

from ..guardrails.engine import (
    DEFLECT_GENERIC,
    DEFLECT_PII,
    DEFLECT_RETRIEVAL,
    DEFLECT_FAITHFULNESS,
    DEFLECT_INJECTION,
)


def answer_metrics(pipeline, in_scope: list[dict]) -> dict:
    """Full-pipeline answer quality on in-scope questions (offline extractive mode)."""
    rows = []
    for item in in_scope:
        out = pipeline.chat_full(item["query"])
        answer = out["answer"]
        # one of the canonical deflections counted as "not answerable"
        deflected = any(d in answer for d in (DEFLECT_GENERIC, DEFLECT_PII, DEFLECT_RETRIEVAL, DEFLECT_FAITHFULNESS, DEFLECT_INJECTION))
        context = "\n\n".join(h.text for h in out["citations"][:5])
        fb_score = pipeline.engine.faithfulness.score(answer, context) if answer and not deflected else 0.0
        rows.append(
            {
                "query": item["query"],
                "refused": out["refused"],
                "answerable": bool(answer) and not deflected and not out["refused"],
                "faithfulness": round(fb_score, 4),
                "n_citations": len(out["citations"]),
                "latency_ms": out["latency_ms"],
                "answer_excerpt": answer[:140],
            }
        )

    n = len(rows) or 1
    answerable = sum(1 for r in rows if r["answerable"])
    return {
        "n_queries": len(rows),
        "answerable_rate": round(answerable / n, 4),
        "mean_faithfulness": round(sum(r["faithfulness"] for r in rows) / n, 4),
        "citation_coverage": round(
            sum(1 for r in rows if r["n_citations"] > 0) / n, 4
        ),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in rows) / n, 1),
        "rows": rows,
    }