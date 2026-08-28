# Tests

> **Test suite for the learning system's reusable code.**
> Tests live here for `src/` code.

---

## Running Tests

```bash
# From the project root
uv run pytest
# or
.\.venv\Scripts\python.exe -m pytest
```

---

## Structure

```
tests/
├── test_data/       # Tests for src/data
├── test_features/   # Tests for src/features
├── test_models/     # Tests for src/models
├── test_evaluation/ # Tests for src/evaluation
└── test_utils/      # Tests for src/utils
```

---

## Guidelines

- Test reusable code in `src/`.
- Use pytest.
- Aim for meaningful coverage of core logic.
- Keep tests fast and deterministic.
