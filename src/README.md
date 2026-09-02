# src

> **Reusable production code for the learning system.**
> Importable as a normal package from scripts, tests, and notebooks run from the
> repository root (`import src.data.loader`). Heavy dependencies (torch,
> langchain, chromadb, sentence-transformers, openai) are imported lazily inside
> functions so importing these modules stays cheap.

---

## Structure

```
src/
├── data/          # data/loader.py      dataset loaders + 3-way splits
├── features/      # features/engineering.py  correlation / MI selection, interactions
├── models/        # models/linear.py    from-scratch linear & logistic regression
├── evaluation/    # evaluation/metrics.py     single-call metric bundles
├── pipelines/     # pipelines/build.py  ColumnTransformer + model pipelines
├── rag/           # rag/chunker.py, rag/retriever.py  chunking + TF-IDF/embedding retrieval
├── agents/        # agents/tool.py, agents/loop.py   tool schemas + minimal agent loop
└── utils/         # utils/common.py     seeds, timing, joblib persistence, project root
```

---

## Usage

```python
from src.data.loader import load_sklearn
from src.models.linear import LogisticRegressionGD
from src.evaluation.metrics import evaluate_classification
from src.rag.chunker import chunk_text

X, y = load_sklearn("wine")
```

---

## Guidelines

- Move reusable code from notebooks into `src/` when it's used in multiple places.
- Keep notebooks focused on learning; keep `src/` focused on reusable code.
- Add tests for `src/` code in `tests/`.
- Keep new modules dependency-light: import heavy packages lazily inside functions.
- Document each module with docstrings.