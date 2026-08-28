# src

> **Reusable production code for the learning system.**
> Move reusable code from notebooks into these packages when appropriate.

---

## Structure

```
src/
├── data/          # Data loading and processing
├── features/      # Feature engineering
├── models/        # Model definitions
├── evaluation/    # Evaluation code
├── pipelines/     # ML pipelines
├── rag/           # RAG components
├── agents/        # Agent components
└── utils/         # Utilities
```

---

## Usage

Import from the package:

```python
from src.data.loader import load_dataset
from src.models.linear import LinearRegression
from src.evaluation.metrics import evaluate_classification
```

---

## Guidelines

- Move reusable code from notebooks into `src/` when it's used in multiple places.
- Keep notebooks focused on learning; keep `src/` focused on reusable code.
- Add tests for `src/` code in `tests/`.
- Document each module with docstrings.
