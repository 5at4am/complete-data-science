"""Verify that no committed notebook contains error outputs (or invalid JSON).

This is the automated guard behind the claim "every notebook executes with 0
error cells". It does not re-execute notebooks; it scans the committed state so
a notebook with a red cell (e.g. a page accidentally saved after a crash) fails
CI instead of silently regressing.

Usage:
    python scripts/verify_notebooks.py [path/to/notebooks]

Exit code 0 = all clean; 1 = at least one notebook has error outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def check_notebook(path: Path) -> tuple[bool, list[str]]:
    """Return (ok, problems) for a single .ipynb file."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        return False, [f"INVALID JSON: {exc}"]

    problems: list[str] = []
    empty: list[str] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs") or []
        if any(out.get("output_type") == "error" for out in outputs):
            for out in outputs:
                if out.get("output_type") == "error":
                    ename = out.get("ename", "?")
                    problems.append(f"cell {idx}: error output ({ename})")
        if not outputs:
            empty.append(str(idx))

    ok = not problems
    if not ok:
        lines = [f"{path.relative_to(Path.cwd())}: {p}" for p in problems]
        if empty:
            lines.append(
                f"{path.relative_to(Path.cwd())}: no-output code cells: {', '.join(empty)}"
            )
        return False, lines
    return True, []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default="notebooks", help="directory tree to scan")
    parser.add_argument(
        "--warn-only", action="store_true", help="report but exit 0 for empty-cell warnings"
    )
    args = parser.parse_args()

    root = Path(args.root)
    notebooks = sorted(root.rglob("*.ipynb"))
    failures: list[str] = []
    checked = 0

    for nb in notebooks:
        if ".ipynb_checkpoints" in nb.parts:
            continue
        checked += 1
        problems = check_notebook(nb)[1]
        if problems and (not args.warn_only or "error output" in " ".join(problems)):
            failures.extend(problems)

    print(f"Checked {checked} notebooks ({root}).")
    if failures:
        print("\n".join(failures))
        print(f"\nFAIL: {len(failures)} problem(s) found.")
        return 1
    print("OK: no error outputs in any notebook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
