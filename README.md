# AI/ML Engineering Learning System

A **complete, deeply connected, implementation-first learning system** that takes a learner
from fundamentals to advanced Data Science, Machine Learning, Artificial Intelligence,
Deep Learning, NLP, Generative AI, LLMs, RAG, LangChain, LangGraph, AI agents, evaluation,
deployment, and production ML/AI engineering.

> **This is not a collection of tutorials.**
> It is a complete learning-and-building system designed so that you can leave the final
> capstone and independently design and build a new AI/ML system that was never explicitly
> taught in the curriculum.

---

## Quick Start

### 1. Set up the environment

```bash
# Using uv (recommended)
uv sync

# Activate the environment
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

### 2. Verify your environment

Open and run:
```
notebooks/00_environment/00_environment_verification.ipynb
```

### 3. Start learning

Follow the roadmap in `ROADMAP.md` and the phase files in `roadmap/`.

---

## Key Documents

| Document | Purpose |
|----------|---------|
| [ROADMAP.md](ROADMAP.md) | The complete learning progression |
| [LEARNING_SYSTEM.md](LEARNING_SYSTEM.md) | How the learning system works |
| [BUILD_STATUS.md](BUILD_STATUS.md) | What has been built and verified |
| [PROJECT_STATE.md](PROJECT_STATE.md) | Full project state snapshot |
| [CHANGELOG.md](CHANGELOG.md) | Change log |
| [CONTRIBUTING.md](CONTRIBUTING.md) | How to contribute |
| [docs/](docs/) | Reference documentation |
| [roadmap/](roadmap/) | Phase-by-phase roadmap |
| [tracking/](tracking/) | Progress, mistakes, concepts, experiments |

---

## Project Structure

```
AI-ML-Learning-System/
├── README.md
├── ROADMAP.md
├── LEARNING_SYSTEM.md
├── BUILD_STATUS.md
├── PROJECT_STATE.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── .env.example
│
├── docs/          # Reference documentation
├── roadmap/       # Dependency-aware learning roadmap
├── notebooks/     # 18 phase directories (00-17)
├── projects/      # 4 project levels
├── datasets/      # raw/ and processed/
├── src/           # Reusable source code
├── tests/         # Test suite
├── templates/     # Notebook templates
└── tracking/      # Progress, mistakes, concepts, experiments
```

---

## The Learning Progression

```
Computer/Python Foundations
        ↓
Python for Data Work
        ↓
Mathematics
        ↓
Statistics & Probability
        ↓
Data Analysis
        ↓
Data Preparation
        ↓
Machine Learning Foundations
        ↓
Supervised Learning
        ↓
Unsupervised Learning
        ↓
Model Evaluation
        ↓
Feature Engineering
        ↓
ML Engineering
        ↓
Deep Learning
        ↓
Computer Vision
        ↓
NLP
        ↓
Transformers
        ↓
LLMs
        ↓
Generative AI
        ↓
Embeddings
        ↓
Vector Search
        ↓
RAG
        ↓
LangChain
        ↓
LangGraph
        ↓
Agents
        ↓
Evaluation
        ↓
Deployment
        ↓
MLOps / LLMOps
        ↓
Production Systems
        ↓
Capstone Engineering
```

---

## Learning Philosophy

- **Understanding before implementation** — know *what* and *why* before *how*.
- **Implementation-first** — build from scratch, then use libraries.
- **Why, not just how** — understand the problem each tool solves.
- **Mistakes are learning material** — bad approaches are taught deliberately.
- **Evaluation is first-class** — throughout, not at the end.
- **Retention is engineered** — recall, review, and reuse.
- **Progressive independence** — from guided to independent engineering.
- **Depth over breadth** — one concept at a time, done well.

See [docs/learning-philosophy.md](docs/learning-philosophy.md) for details.

---

## Mastery Levels

```
LEVEL 0 — Exposure        I have seen it.
LEVEL 1 — Recognition     I can identify it.
LEVEL 2 — Guided impl     I can implement it with help.
LEVEL 3 — Independent     I can implement it without instructions.
LEVEL 4 — Debugging       I can diagnose failures.
LEVEL 5 — Decision making I can choose when/why to use it.
LEVEL 6 — Teaching        I can explain it clearly to another person.
```

---

## License

This is an educational project. See `CONTRIBUTING.md` for contribution guidelines.
