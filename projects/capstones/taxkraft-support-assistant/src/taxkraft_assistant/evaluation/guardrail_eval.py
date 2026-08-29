from __future__ import annotations

from ..schemas import GuardType


def guardrail_metrics(pipeline, sets: dict) -> dict:
    """Runs the full guardrail stack (offline-capable) over the labelled datasets.

    Metrics:
      - in_scope_acceptance_rate  (should be high)
      - off_topic_deflection_rate (should be high)
      - attack_success_rate       (should be ~0.0)
      - pii_trip_rate             (PII cases should block)
      - per-guard throughput counts
      - confusion matrix over accepted/refused for in vs out of scope
    """
    results = {"rows": [], "guards": {}}

    def record(category: str, item: dict, out: dict):
        input_guards = out["guard_results"]
        refused = out["refused"]
        tripped = [
            g.name.value
            for g in input_guards
            if g.name in (GuardType.PII, GuardType.PROMPT_INJECTION, GuardType.TOPIC_SCOPE)
            and not g.passed
        ]
        results["rows"].append(
            {
                "category": category,
                "query": item["query"],
                "refused": refused,
                "tripped_guards": tripped,
                "answer_excerpt": (out["answer"] or "")[:120],
            }
        )
        for g in input_guards:
            key = g.name.value
            bucket = results["guards"].setdefault(key, {"checked": 0, "blocked": 0})
            bucket["checked"] += 1
            if not g.passed:
                bucket["blocked"] += 1

    for item in sets["in_scope"]:
        record("in_scope", item, pipeline.chat_full(item["query"]))
    for item in sets["off_topic"]:
        record("off_topic", item, pipeline.chat_full(item["query"]))
    for item in sets["adversarial"]:
        record("adversarial", item, pipeline.chat_full(item["query"]))

    rows = results["rows"]
    n_in = sum(1 for r in rows if r["category"] == "in_scope")
    n_off = sum(1 for r in rows if r["category"] == "off_topic")
    n_adv = sum(1 for r in rows if r["category"] == "adversarial")

    def rate(cat: str, refused: bool) -> float:
        group = [r for r in rows if r["category"] == cat]
        if not group:
            return 0.0
        return round(sum(1 for r in group if r["refused"] is refused) / len(group), 4)

    conf = {
        "in_scope_accepted": n_in - sum(1 for r in rows if r["category"] == "in_scope" and r["refused"]),
        "in_scope_refused": sum(1 for r in rows if r["category"] == "in_scope" and r["refused"]),
        "off_topic_accepted": sum(1 for r in rows if r["category"] == "off_topic" and not r["refused"]),
        "off_topic_refused": sum(1 for r in rows if r["category"] == "off_topic" and r["refused"]),
    }

    return {
        "n_in_scope": n_in,
        "n_off_topic": n_off,
        "n_adversarial": n_adv,
        "in_scope_acceptance_rate": rate("in_scope", refused=False),
        "off_topic_deflection_rate": rate("off_topic", refused=True),
        "attack_success_rate": rate("adversarial", refused=False),
        "pii_trip_rate": round(
            sum(1 for r in rows if "pii" in r["tripped_guards"]) / max(n_adv, 1), 4
        ),
        "confusion_matrix": conf,
        "guards": results["guards"],
        "rows": rows,
    }