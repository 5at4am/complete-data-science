from __future__ import annotations


def retrieval_metrics(retriever, golden: list[dict], top_k: int = 5) -> dict:
    """Recall@1/3/5 + MRR@5 — does the expected topic surface in the top hits?"""
    rows = []
    for item in golden:
        expected = item["expected_topic"]
        hits = retriever.retrieve(item["query"], top_k=top_k)
        topics = [h["topic"] for h in hits]
        ranks = [i + 1 for i, t in enumerate(topics) if t == expected]
        rows.append(
            {
                "query": item["query"],
                "expected_topic": expected,
                "topics": topics,
                "first_rank": ranks[0] if ranks else None,
            }
        )

    def recall(at: int) -> float:
        hits = sum(1 for r in rows if r["first_rank"] is not None and r["first_rank"] <= at)
        return round(hits / len(rows), 4) if rows else 0.0

    mrr = sum(1.0 / r["first_rank"] for r in rows if r["first_rank"] is not None)
    mrr = round(mrr / len(rows), 4) if rows else 0.0
    return {
        "n_queries": len(rows),
        "recall@1": recall(1),
        "recall@3": recall(3),
        "recall@5": recall(5),
        "mrr@5": mrr,
        "rows": rows,
    }