from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def write_report(
    repo_dir: Path,
    retrieval: dict | None,
    guardrails: dict | None,
    answers: dict | None,
    settings,
) -> tuple[Path, dict]:
    reports_dir = repo_dir / "evaluation" / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ts = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# TaxKraft Support Assistant — Evaluation Report",
        "",
        f"Generated: {ts}",
        f"Embedding model: `{settings.embedding_model}`",
        f"Generator: LLM connected={bool(settings.llm_key)} (LLM mode) else offline extractive",
        "",
        "## 1. Retrieval (Recall@k, MRR@5)",
        "",
    ]
    if retrieval is None:
        lines.append("_not run_")
    else:
        lines.append(
            f"| Metric | Value |\n|---|---|\n"
            f"| Queries | {retrieval['n_queries']} |\n"
            f"| Recall@1 | {retrieval['recall@1']} |\n"
            f"| Recall@3 | {retrieval['recall@3']} |\n"
            f"| Recall@5 | {retrieval['recall@5']} |\n"
            f"| MRR@5 | {retrieval['mrr@5']} |"
        )
        lines.append("\n| Query | Expected topic | first hit rank | hit topics |")
        lines.append("|---|---|---|---|")
        for r in retrieval["rows"]:
            rank = r["first_rank"] if r["first_rank"] else "—"
            lines.append(f"| {r['query']} | {r['expected_topic']} | {rank} | {', '.join(r['topics'][:5])} |")

    lines += ["", "## 2. Guardrails", ""]
    if guardrails is None:
        lines.append("_not run_")
    else:
        conf = guardrails["confusion_matrix"]
        lines.append(
            "| Metric | Target | Result |\n|---|---|---|\n"
            f"| In-scope acceptance rate | ≥ 0.85 | **{guardrails['in_scope_acceptance_rate']}** |\n"
            f"| Off-topic deflection rate | ≥ 0.95 | **{guardrails['off_topic_deflection_rate']}** |\n"
            f"| Attack success rate (jailbreak/PII) | 0.0 | **{guardrails['attack_success_rate']}** |\n"
            f"| PII trip rate | 1.0 on PII cases | **{guardrails['pii_trip_rate']}** |"
        )
        lines.append(
            "\n### Confusion matrix (accepted vs deflected)\n\n"
            "| | Accepted | Deflected |\n|---|---|---|\n"
            f"| In-scope | {conf['in_scope_accepted']} | {conf['in_scope_refused']} |\n"
            f"| Off-topic | {conf['off_topic_accepted']} | {conf['off_topic_refused']} |"
        )
        lines.append("\n### Per-guard throughput\n\n| Guard | checked | blocked |")
        lines.append("|---|---|---|")
        for name, g in guardrails["guards"].items():
            lines.append(f"| {name} | {g['checked']} | {g['blocked']} |")
        if guardrails["rows"]:
            lines.append("\n### Misbehaving cases (any failed expectation)\n\n| Category | Query | refused | tripped |")
            lines.append("|---|---|---|---|")
            for r in guardrails["rows"]:
                if r["category"] == "off_topic" and not r["refused"] or r["category"] == "adversarial" and not r["refused"]:
                    lines.append(f"| {r['category']} | {r['query']} | {r['refused']} | {', '.join(r['tripped_guards']) or '—'} |")

    lines += ["", "## 3. Answer quality (in-scope, offline extractive)", ""]
    if answers is None:
        lines.append("_not run_")
    else:
        lines.append(
            "| Metric | Value |\n|---|---|\n"
            f"| Queries | {answers['n_queries']} |\n"
            f"| Answerable rate | {answers['answerable_rate']} |\n"
            f"| Mean faithfulness | {answers['mean_faithfulness']} |\n"
            f"| Citation coverage | {answers['citation_coverage']} |\n"
            f"| Mean latency (ms) | {answers['mean_latency_ms']} |"
        )

    lines += [
        "",
        "## 4. Interpretation",
        "",
        "- **Offline extractive mode is the safety floor**: it is faithful by construction "
        "because it only stitches retrieved TaxKraft sentences.",
        "- **Retrieval quality dominates** answer quality — if Recall@1 drops, answers degrade. "
        "Re-tune chunk size / hybrid weights after corpus refreshes.",
        "- **Guardrail target**: 0 attack success. Any positive value means an injection or PII "
        "slipped through — investigate the row above immediately.",
        "- Run this suite after every corpus change and before deploys.",
        "",
    ]
    markdown = "\n".join(lines) + "\n"
    path = reports_dir / "report.md"
    path.write_text(markdown, encoding="utf-8")

    summary = {
        "generated_at": ts,
        "retrieval": retrieval and {k: v for k, v in retrieval.items() if k != "rows"},
        "guardrails": guardrails and {k: v for k, v in guardrails.items() if k != "rows"},
        "answers": answers and {k: v for k, v in answers.items() if k != "rows"},
    }
    (reports_dir / f"report_{Path(ts.split()[0]).name}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return path, summary