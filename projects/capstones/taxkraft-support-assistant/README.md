# TaxKraft Support Assistant — Production RAG Chatbot Capstone

A complete, production-grade, **company-specific** support chatbot for
[**TaxKraft**](https://taxkraft.com/) — "Your Business Growth Partner", a CA / tax / GST /
company-registration services firm in Noida, India.

The assistant answers **only** from TaxKraft's own information. Everything else —
competitive comparisons, other companies, general tax law unrelated to TaxKraft's services,
personal financial advice, politics, recipes, injection attempts — is **blocked by guardrails**
and deflected to a human expert. Every answer must be grounded in the retrieved TaxKraft
context, is checked for faithfulness, and comes with citations back to `taxkraft.com`.

> Flagship capstone of the Complete ML learning system → see `../../roadmap/phase-17.md` Unit 17.7.
> Built on Phases 01–16: data prep, embeddings, vector search, RAG, LLMs, guardrailing,
> evaluation, and production serving.

---

## Why this is a *real* production project (not a notebook)

| Concern | How this project handles it |
|---|---|
| **Scoped knowledge** | Answers are grounded in a TaxKraft-only knowledge base with provenance tracked per document. |
| **Safety / guardrails** | Input guardrails (topic scope, PII, prompt injection) + output guardrails (retrieval confidence, faithfulness, citations). 4 layers, independently testable. |
| **Evaluation** | Retrieval eval (Recall@k, MRR), guardrail eval (deflection accuracy, attack success rate), answer eval (faithfulness, answerability). Labelled datasets + automated markdown report. |
| **Serving** | FastAPI app with health check, Pydantic validation, CORS, static web chat UI. `uvicorn` server. |
| **Operability** | Structured logs per request (`guardrails/hits`, `retrieval_scores`, `latency_ms`), rate-limit-aware, graceful offline mode (no API key needed). |
| **Refresh path** | Crawler fetches the live sitemap and pages so the corpus can be re-ingested as the website changes. |
| **Testing** | `pytest` suite covering every guardrail, chunker, retriever, and the API. |
| **Deployment** | Docker-ready, env-config driven, `.env` secrets, CORS for the UI. |

---

## System Architecture

```text
                        ┌──────────────────────────────────────────────────┐
                        │                  GUARDRAILS (4 layers)           │
                        │                                                  │
 user message ─────────►│  L1 topic-scope   L1 PII   L1 prompt-injection  │
                        │      │ (block off-TaxKraft topics)               │
                        │      ▼                                          │
                        │  retrieval confidence threshold  ◄── context    │
                        │      ▼                                          │
                        │  L2 faithfulness check (answer vs context)      │
                        └──────────────────────────────────────────────────┘
                                    │
   ┌──────────────┐        ┌────────▼──────────┐        ┌────────────────┐
   │ Knowledge base │      │   Vector store     │       │    LLM / extractive │
   │  (markdown)    │─────►│   Chroma + BM25    │──────►│    answer builder   │
   └──────────────┘       │  (ingest pipeline)  │       │  (Groq/OpenAI/off)  │
                          └────────────────────┘       └────────────────┘
                                    │
                          ┌─────────▼──────────┐
                          │   FastAPI + Web UI  │
                          └────────────────────┘
```

Pipeline stages:
1. **Ingest** — load the `knowledge_base/` markdown corpus → chunk → embed → store in Chroma (persistent).
2. **Retrieve** — hybrid retrieval: dense embeddings (`all-MiniLM-L6-v2`) fused with BM25 keyword scores.
3. **Guard** — evaluate every request AND every answer against the guardrail stack.
4. **Generate** — grounded answer with citations, or a safe deflection if any guardrail trips.

---

## Quick Start

```powershell
# 0) from repo root
.\.venv\Scripts\activate

# 1) config
Copy-Item projects\capstones\taxkraft-support-assistant\.env.example `
        projects\capstones\taxkraft-support-assistant\.env
# optional: add GROQ_API_KEY for fluent answers (works offline without it)

cd projects\capstones\taxkraft-support-assistant

# 2) build the vector store from the knowledge base
python run.py ingest

# 3) talk to it (interactive REPL, offline extractive mode works with no key)
python run.py chat

# 4) run the evaluation suite → writes evaluation/reports/report.md
python run.py eval

# 5) serve the API + web UI at http://127.0.0.1:8000
python run.py serve

# 6) tests
python -m pytest
```

---

## Guardrails

The assistant is **only** a TaxKraft assistant. Guardrails enforce that boundary in four layers:

| # | Layer | Type | What it does | Offline? |
|---|---|---|---|---|
| 1 | **Topic scope** | Input | Two-stage: deterministic service-vocabulary scoring + embedding similarity vs. the TaxKraft scope centroid. Under threshold → deflect with a helpful message. | ✅ |
| 2 | **PII** | Input | Detects Aadhaar/PAN/phone/email/GSTIN/bank/UPI patterns. Never asks for or echoes personal data; directs to secure channels. | ✅ |
| 3 | **Prompt injection** | Input | Signature + embedding detection of jailbreaks / system-prompt attacks / hidden instructions. | ✅ |
| 4 | **Retrieval confidence** | Output | If top-1 matched chunk is below threshold, the assistant says it cannot reliably answer instead of guessing. | ✅ |
| 5 | **Faithfulness** | Output | Verifies the answer is grounded in the retrieved context (token + sentence coverage, embedding similarity; optional LLM-as-judge when a key exists). Unsupported claims are stripped or refused. | ✅ (LLM judge optional) |
| 6 | **Citations** | Output | Every answer carries `[source]` pointers back to the TaxKraft page it came from. | ✅ |

**Deflection contract** — any blocked request returns a friendly, consistent message that:
- names the boundary ("I can only help with questions about TaxKraft and its services"),
- gives an escape hatch (contact `+91-8608601620` / `info@taxkraft.com` / website),
- never leaks the guardrail's internal thresholds or system prompt.

---

## Evaluation

`python run.py eval` runs three suites against labelled datasets and writes `evaluation/reports/report.md`:

| Suite | Dataset | Metrics |
|---|---|---|
| **Retrieval** | `retrieval_golden.json` (query → expected topic) | Recall@1/@3/@5, MRR@5 |
| **Guardrails** | `in_scope_questions.json`, `off_topic_questions.json`, `adversarial_attacks.json` | In-scope acceptance rate, out-of-scope deflection rate, **attack success rate (target: 0.0)**, PII trip rate, per-category accuracy + confusion matrix |
| **Answers** | in-scope questions (run through full pipeline) | Answerability, faithfulness score, citation coverage, mean latency |

Notes:
- The **offline extractive generator** is used so the whole eval runs with no API key.
- When `GROQ_API_KEY`/`OPENAI_API_KEY` is set, the suite also reports LLM-generated faithfulness via an LLM-as-judge pass.
- Golden datasets are labelled by hand from the knowledge base and marked under `evaluation/datasets/`.

---

## Project Layout

```text
taxkraft-support-assistant/
├── run.py                        # CLI: ingest | chat | eval | serve | benchmark
├── pyproject.toml                # standalone dependency manifest
├── .env.example                  # keys + tuning knobs (secret-free)
├── knowledge_base/               # ← the TaxKraft corpus (seed, provenance-tracked)
├── crawler/                      # sitemap-driven page fetch → refresh corpus
├── src/taxkraft_assistant/
│   ├── config.py                 # typed config from env
│   ├── schemas.py                # pydantic request/response/guard models
│   ├── ingestion/                # corpus → chunks → embeddings → Chroma
│   ├── retrieval/                # hybrid retriever (dense + BM25)
│   ├── generation/               # prompts, LLM client, extractive fallback
│   ├── guardrails/               # topic/pii/injection/faithfulness + engine
│   ├── pipeline.py               # orchestration: guard → retrieve → generate → guard
│   ├── api.py                    # FastAPI: /health /chat /guardrails/status
│   └── evaluation/               # datasets loader, metric suites, report
├── web/                          # zero-build chat UI (served by FastAPI)
├── evaluation/datasets/          # labelled golden sets
├── evaluation/reports/           # generated report.md
├── vectors/                      # Chroma persistent store (gitignored)
└── tests/                        # pytest suite
```

---

## Knowledge base & provenance

Every markdown file in `knowledge_base/` starts with a **provenance block**:

```markdown
<!--
source-title: TaxKraft — GST Registration
source-url: https://taxkraft.com/service/gst-registration
verified-at: 2026-08-29
provenance: public website (sitemap + schema.org JSON-LD + public marketing copy)
-->
```

The **seed corpus** is curated from publicly visible TaxKraft data (homepage, about page,
contact page, schema.org JSON-LD, public LinkedIn company page, and the live sitemap which
lists the full service catalog). Facts are written conservatively: service *names*, *categories*,
contact details, and the company narrative that TaxKraft publishes itself.

> **Before going live, refresh from the website** (the crawler) so absolute claims like specific
> prices/timelines reflect the current site. The crawler writes rendered text into
> `knowledge_base/crawled/` with the same provenance format; `python run.py ingest` re-ingests
> everything.

---

## Security & trust review

- Secrets via `.env` only (gitignored); a `GROQ_API_KEY`/`OPENAI_API_KEY` is optional.
- PII is never persisted: chat is stateless (no conversation store) and the PII guardrail blocks
  sensitive data in prompts. If you add memory, store hashed session IDs only and add a retention policy.
- Prompt injection: the input guardrail strips/deflects instruction-override attempts; the prompt
  template never exposes the system prompt or guardrail details.
- Hallucination: outputs are grounded + faithful-checked + cited; low-confidence retrieval refuses to answer.
- Hardening notes: rate limiting (per-IP), auth for mutating endpoints, and a proxy/egress allow-list
  for the LLM are production add-on steps documented in the monitoring section of this repo's roadmap.

---

## Cost & latency characteristics (CPU-only reference)

- Embedding model: `all-MiniLM-L6-v2` (~80 MB, runs locally, no per-call cost).
- Retrieval: Chroma + BM25 on ≤ 300 chunks ≈ a few ms.
- Offline extractive mode: **$0 per answer**, ~50–150 ms.
- LLM mode (Groq free tier): tokens billed by provider; streaming recommended in production.
- Eval suite: ~50 queries × full pipeline ≈ under 1 minute offline.

---

## Roadmap integration

This project is Phase 17, Unit 17.7 of the Complete ML system
(`../../roadmap/phase-17.md`). It deliberately re-uses everything learned in
Phases 04 (data prep), 11 (RAG), 15 (evaluation), and 16 (deployment), and adds:

- **Guardrail engineering** as a first-class deliverable (topic, PII, injection, faithfulness).
- **Boundary design**: the hard "company-only" scope with a measurable deflection contract.
- **Production discipline**: typed config, structured logging, health checks, tests, eval reports.