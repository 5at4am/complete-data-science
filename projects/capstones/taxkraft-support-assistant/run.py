from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from taxkraft_assistant.config import load_settings  # noqa: E402
from taxkraft_assistant.pipeline import Pipeline, default_llm_client  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(prog="taxkraft-assistant", description="TaxKraft Support Assistant CLI")
    parser.add_argument("command", choices=["ingest", "chat", "eval", "serve", "test"], help="what to run")
    args = parser.parse_args()

    if args.command == "ingest":
        pipeline = Pipeline.build(settings=load_settings(), connect=False)
        pipeline.ingest(reset=True)

    elif args.command == "chat":
        pipeline = Pipeline.build(llm_client=default_llm_client(load_settings()))
        print("TaxKraft Support Assistant (extractive unless a GROQ/OPENAI key is set). Type 'exit' to quit.")
        while True:
            try:
                q = input("\nYou > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in {"exit", "quit"}:
                break
            out = pipeline.chat_full(q)
            print(f"\nBot  > {out['answer']}")
            label = "DEFLECTED" if out["refused"] else "ANSWERED"
            guards = ", ".join(
                f"{g.name.value}={g.status.value}"
                for g in out["guard_results"]
                if g.status.value != "not_run"
            )
            print(f"      [{label}] [{guards}] ({out['generator']})")

    elif args.command == "eval":
        pipeline = Pipeline.build(llm_client=default_llm_client(load_settings()))
        from taxkraft_assistant.evaluation.run_eval import run_full_eval

        path, _ = run_full_eval(pipeline)
        print(f"[OK] report written -> {path}")

    elif args.command == "serve":
        import uvicorn

        from taxkraft_assistant.api import create_app

        uvicorn.run(create_app(), host="127.0.0.1", port=8000, log_level="info")

    elif args.command == "test":
        import subprocess

        subprocess.run([sys.executable, "-m", "pytest", "-q"], check=True)


if __name__ == "__main__":
    main()