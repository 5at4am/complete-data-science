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
| **Width** | 18 phases × ~177 units across the full AI/ML stack |
| **Depth** | ~55 implementation-first Jupyter notebooks (growing) |
| **Style** | Understand → Build → Debug → Compare → Explain → Solve independently |
| **Currently** | Phases 00–04 ✅ complete · Phase 05 (Machine Learning) ⏳ in progress |

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

| Phase | Topic | Status | Building Blocks |
|---|---|---|---|
| 00 | Environment Setup | ✅ Complete | Jupyter, Git, uv/venv, package verification |
| 01 | Python Foundations | ✅ Complete | syntax, data structures, functions, OOP, NumPy, pandas, matplotlib |
| 02 | Mathematics for ML | ✅ Complete | linear algebra, calculus, gradient descent, probability, information theory, eigenvalues |
| 03 | Statistics & Probability | ✅ Complete | distributions, inference, hypothesis testing, correlation, Bayesian thinking |
| 04 | Data Analysis & Preparation | ✅ Complete | EDA, cleaning, missing values, outliers, scaling/encoding, splits, leakage |
| 05 | Machine Learning | ⏳ In progress (18 units) | linear/logistic regression, trees, forests, boosting, KNN, naive Bayes, SVM, clustering, PCA, interpretability |
| 06–08 | Deep Learning · NLP · Transformers | ⏳ Not started | — |
| 09–11 | GenAI · LLMs · RAG | ⏳ Not started | — |
| 12–14 | LangChain · LangGraph · Agents | ⏳ Not started | — |
| 15–17 | Evaluation · Deployment · Capstone | ⏳ Not started | — |

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

Educational use. See [`CONTRIBUTING.md`](CONTRIBUTING.md).