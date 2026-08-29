from __future__ import annotations

import time

from ..config import PROJECT_DIR
from .answer_eval import answer_metrics
from .datasets import load_sets
from .guardrail_eval import guardrail_metrics
from .report import write_report
from .retrieval_eval import retrieval_metrics


def run_full_eval(pipeline, top_k: int = 5) -> tuple:
    sets = load_sets(PROJECT_DIR / "evaluation" / "datasets")
    t0 = time.perf_counter()

    print("· retrieval metrics ...")
    retrieval = retrieval_metrics(pipeline.retriever, sets["retrieval"], top_k=top_k)

    print("· guardrail metrics ...")
    guardrails = guardrail_metrics(pipeline, sets)

    print("· answer metrics (offline extractive) ...")
    answers = answer_metrics(pipeline, sets["in_scope"])

    print("· writing report ...")
    path, summary = write_report(
        PROJECT_DIR, retrieval, guardrails, answers, pipeline.settings
    )
    summary["elapsed_sec"] = round(time.perf_counter() - t0, 1)
    print(
        f"done in {summary['elapsed_sec']}s -> {path.resolve()}"
        f"  [attack_success={guardrails['attack_success_rate']}]"
    )
    return path, summary