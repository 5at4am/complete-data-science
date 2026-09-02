# Tests

> **Test suite for the learning system's reusable code and infrastructure.**

---

## Running Tests

```bash
# From the project root
uv run pytest
# or
.\.venv\Scripts\python.exe -m pytest
```

The root `conftest.py` adds the repository root to `sys.path`, so `import src.*`
works with plain `pytest` as well as `python -m pytest`.

---

## Structure

```
tests/
├── conftest.py (root)  # makes src/ importable
├── test_sanity.py            # infrastructure smoke test
├── test_src_utils.py         # src/utils
├── test_src_data.py          # src/data + src/features
├── test_src_models.py        # src/models + src/evaluation
└── test_src_rag_agents.py    # src/rag + src/agents
```

---

## Guidelines

- Test reusable code in `src/`.
- Use pytest.
- Aim for meaningful coverage of core logic.
- Keep tests fast and deterministic (no GPU, no network, no heavy training).