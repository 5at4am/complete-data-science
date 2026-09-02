# Complete ML — AI/ML Engineering Learning System

A **complete, deeply connected, implementation-first learning system** that takes a learner
from fundamentals all the way to advanced Data Science, Machine Learning, Deep Learning,
NLP, Generative AI, LLMs, RAG, LangChain/LangGraph, AI agents, evaluation, deployment, and
production ML/AI engineering.

> **This is not a collection of tutorials.**
> It is a complete learning-and-building system designed so that by the final capstone you can
> independently design and build a new AI/ML system that was never explicitly taught in the curriculum.

| | |
|---|---|
| **Width** | 18 phases × 204 units across the full AI/ML stack |
| **Depth** | 204 implementation-first Jupyter notebooks — all executed, 0 error cells |
| **Style** | Understand → Build → Debug → Compare → Explain → Solve independently |
| **Currently** | All 18 phases ✅ complete — every notebook executed & verified from a clean kernel |

---

## Highlights

- **Dependency-aware roadmap** — phases are ordered so you never hit a concept you can't use yet.
  See [`ROADMAP.md`](ROADMAP.md), [`roadmap/dependency-graph.md`](roadmap/dependency-graph.md).
- **Implementation-first** — build it manually first, adopt libraries after you understand the problem.
- **Mistakes are curriculum** — common failure modes and bad approaches are taught deliberately.
- **Decision skills, not just tools** — every major topic includes "when to choose X vs Y" comparisons.
- **Progressive independence** — mini → intermediate → advanced projects → portfolio capstones.
- **Evaluation is first-class** — from early ML metrics through LLM/RAG/agent evaluation.
- **Retention is engineered** — review notebooks, mastery levels, and a tracking system.

## Quick Start

> **Completely new to programming?** Please start with
> [`docs/getting-started.md`](docs/getting-started.md) — a plain-language,
> step-by-step guide for complete beginners (first session, troubleshooting, first week).
> Then read [`docs/prerequisites.md`](docs/prerequisites.md) and
> [`docs/glossary.md`](docs/glossary.md).
> **Returning learners** can follow the commands below directly.

```bash
# 1. Clone the repository
git clone https://github.com/5at4am/complete-data-science.git
cd complete-data-science

# 2. Set up the environment (uv recommended)
uv sync

# 3. Activate the environment
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux

# 4. Verify your environment
#    Open and run: notebooks/00_environment/00_environment_verification.ipynb
jupyter lab

# 5. Run the sanity tests
pytest
```

Stuck? See [`docs/troubleshooting.md`](docs/troubleshooting.md) (written for beginners).

Requires **Python 3.13+**. `requirements.txt` and `pyproject.toml` (`uv.lock`) pin the stack:
NumPy, pandas, SciPy, scikit-learn, matplotlib, seaborn, PyTorch, Hugging Face
(transformers/datasets/sentence-transformers), FAISS, ChromaDB, OpenAI, LangChain, LangGraph,
XGBoost, LightGBM, FastAPI, Flask, Jupyter, pydantic, pytest.

## Start Learning

1. Begin at **Phase 00** and move forward in order.
2. For every unit: `Understand → Apply → Practice → Build → Debug → Compare → Optimize → Explain → Solve independently`.
3. Don't skip prerequisites — check the [dependency graph](roadmap/dependency-graph.md) if something feels early.
4. Track progress in [`tracking/`](tracking/) (progress, mistakes, experiments, project log, review log).

## The Full Progression

```text
Phase 00  Environment Setup                 Phase 09  Generative AI Foundations
Phase 01  Python Foundations                Phase 10  Applied LLM Engineering
Phase 02  Mathematics for ML                Phase 11  RAG Systems
Phase 03  Statistics & Probability          Phase 12  LangChain / Framework Abstractions
Phase 04  Data Analysis & Preparation       Phase 13  LangGraph / Stateful Workflows
Phase 05  Machine Learning                  Phase 14  AI Agents
Phase 06  Deep Learning                     Phase 15  Evaluation & Experimentation
Phase 07  NLP                               Phase 16  Deployment & MLOps
Phase 08  Transformers                      Phase 17  Capstone Engineering
```

> Learning order is intentional: **manual concepts before framework abstractions**.
> Learn the problem first, then the tool that abstracts it.

### Phase Status

| Phase | Topic | Status | Est. Time | Building Blocks |
|---|---|---|---|---|
| 00 | Environment Setup | ✅ Complete | 2 h | Jupyter, Git, uv/venv, package verification |
| 01 | Python Foundations | ✅ Complete | 20 h | syntax, data structures, functions, OOP, NumPy, pandas, matplotlib |
| 02 | Mathematics for ML | ✅ Complete | 25 h | linear algebra, calculus, gradient descent, probability, information theory, eigenvalues |
| 03 | Statistics & Probability | ✅ Complete | 20 h | distributions, inference, hypothesis testing, correlation, Bayesian thinking |
| 04 | Data Analysis & Preparation | ✅ Complete | 15 h | EDA, cleaning, missing values, outliers, scaling/encoding, splits, leakage |
| 05 | Machine Learning | ✅ Complete (33 units) | 45 h | linear/logistic regression, trees, forests, boosting, KNN, naive Bayes, SVM, clustering, PCA, feature engineering, imbalanced learning, tuning (Optuna), interpretability (SHAP), ensembling, capstone synthesis |
| 06 | Deep Learning | ✅ Complete | 30 h | perceptron, loss functions, backprop, MLP in PyTorch, regularization, optimizers, dataloaders, training loops, CNNs, RNN/LSTM/GRU, attention |
| 07 | NLP | ✅ Complete | 15 h | preprocessing, BoW/TF-IDF, n-grams, text classification, word embeddings, sequence models, attention, evaluation |
| 08 | Transformers | ✅ Complete | 15 h | transformer architecture, self-attention, positional encoding, encoder-decoder, BERT-style, GPT-style |
| 09 | Generative AI | ✅ Complete | 20 h | LLM anatomy, tokenization, embeddings, pretraining, decoding, sampling, instruction alignment, LLM APIs, prompting, structured output, tool calling |
| 10 | Applied LLM Engineering | ✅ Complete (8 units) | 15 h | model landscape & selection, context windows, fine-tuning concepts, RAG vs fine-tuning vs long context, cost/latency, LLM security, LLM evaluation, synthesis |
| 11 | RAG Systems | ✅ Complete | 20 h | TF-IDF/BM25 from scratch, embeddings, vector similarity, ChromaDB/FAISS, ingestion, chunking, retrieval, reranking, grounding, evaluation, naive/advanced/agentic RAG |
| 12 | LangChain / Framework Abstractions | ✅ Complete | 10 h | raw LLM pipeline, prompts, output parsers, LCEL chains, memory, tools, RAG with LangChain, framework-vs-manual |
| 13 | LangGraph / Stateful Workflows | ✅ Complete | 10 h | manual state machines, StateGraph, conditional routing, loops, memory, human approval, failure handling |
| 14 | AI Agents | ✅ Complete (13 units) | 15 h | tool calling, planning, execution, reflection, memory, multi-agent, failure modes, security, evaluation |
| 15 | Evaluation & Experimentation | ✅ Complete | 10 h | ML/LLM/RAG/agent evaluation, evaluation datasets, experiment tracking |
| 16 | Deployment & MLOps | ✅ Complete | 15 h | API design, model serving, Docker, CI/CD, monitoring, LLMOps, security |
| 17 | Capstone Engineering | ✅ Complete | 25 h | ML, DL, NLP, RAG, and agent capstones + final portfolio capstone |

> **Total:** ~325 hours of guided work across 204 notebooks (rough rule of thumb: 1.5–2 h per notebook,
> more for the capstone phases). Actual pace depends on how deep you go on the exercises.

Full per-unit status lives in [`BUILD_STATUS.md`](BUILD_STATUS.md). All notebooks live in
`notebooks/NN_phase/`, one folder per phase.

## Repository Structure

```text
├── notebooks/          # 18 phase directories (00–17), one notebook per unit
├── roadmap/            # Dependency graph + per-phase plans (phase-00.md … phase-17.md)
├── docs/               # Reference docs: math, statistics, ML, DL, NLP, GenAI, engineering, troubleshooting
├── projects/           # mini / intermediate / advanced / capstone project ladders
├── src/                # Reusable code: data, features, models, evaluation, pipelines, RAG, agents
├── templates/          # Notebook templates (concept, implementation, experiment, review)
├── datasets/           # raw/ and processed/ datasets
├── tracking/           # progress, mistakes, experiments, project log, review log
├── tests/              # pytest suite
├── ROADMAP.md          # The complete learning progression
├── BUILD_STATUS.md     # What is built & verified (single source of truth)
├── LEARNING_SYSTEM.md  # How the learning system works
└── pyproject.toml      # Python 3.13+ / uv-managed project config
```

## Project Ladder

Projects appear throughout the roadmap to prevent "tutorial hell" — independence increases at every rung.

| Milestone | Project Type | Independence Level |
|---|---|---|
| Phase 01 | Python data utility | Follow instructions |
| Phase 04 | EDA + data-cleaning report | Partial guide |
| Phase 05 | End-to-end classical ML project | Solve with hints |
| Phase 06 | Neural-network training project | Debug with guidance |
| Phase 07–08 | NLP / transformer project | Make modeling decisions |
| Phase 10–11 | LLM / RAG application | Design retrieval & evaluation |
| Phase 14 | Tool-using agent | Handle failures & security constraints |
| Phase 16 | Deployed ML/AI service | Operate & monitor a system |
| Phase 17 | Portfolio capstones | Design independently |

## Decision Comparisons

A core goal is learning *when* to reach for each tool, not just *how* to use it:

| Comparison | Phase |
|---|---|
| Linear model vs tree model | 05 |
| Random Forest vs XGBoost/LightGBM | 05 |
| Accuracy vs F1 / ROC-AUC / PR-AUC | 05 |
| CNN vs Transformer | 06/08 |
| Fine-tuning vs RAG | 10/11 |
| LangChain vs direct API implementation | 12 |
| LangGraph vs simple chains | 13 |
| Single agent vs workflow | 14 |

## Mastery Levels

```text
LEVEL 0 — Exposure        I have seen it.
LEVEL 1 — Recognition     I can identify it.
LEVEL 2 — Guided impl     I can implement it with help.
LEVEL 3 — Independent     I can implement it without instructions.
LEVEL 4 — Debugging       I can diagnose failures.
LEVEL 5 — Decision making I can choose when/why to use it.
LEVEL 6 — Teaching        I can explain it clearly to another person.
```

## Documentation Map

| Document | Purpose |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | **Start here if you're new** — plain-language setup, first session, first week |
| [docs/glossary.md](docs/glossary.md) | Plain-English dictionary of every term used in the course |
| [docs/prerequisites.md](docs/prerequisites.md) | What you need before starting (and what you don't) |
| [ROADMAP.md](ROADMAP.md) | The complete learning progression |
| [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md) | How the learning system works |
| [BUILD_STATUS.md](BUILD_STATUS.md) | What is built & verified |
| [PROJECT_STATE.md](PROJECT_STATE.md) | Full project state snapshot |
| [CHANGELOG.md](CHANGELOG.md) | Change log |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/](docs/) | Reference documentation |
| [roadmap/](roadmap/) | Phase-by-phase plans |
| [tracking/](tracking/) | Progress, mistakes, concepts, experiments |

## Contributing

This is an educational project. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

## License

Released under the [MIT License](LICENSE). See [`CONTRIBUTING.md`](CONTRIBUTING.md) for
contribution guidelines.