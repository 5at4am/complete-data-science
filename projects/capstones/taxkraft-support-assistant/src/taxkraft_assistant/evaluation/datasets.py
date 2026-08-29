from __future__ import annotations

import json
from pathlib import Path


def load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sets(datasets_dir: Path) -> dict[str, list[dict]]:
    return {
        "retrieval": load_json(datasets_dir / "retrieval_golden.json"),
        "in_scope": load_json(datasets_dir / "in_scope_questions.json"),
        "off_topic": load_json(datasets_dir / "off_topic_questions.json"),
        "adversarial": load_json(datasets_dir / "adversarial_attacks.json"),
    }