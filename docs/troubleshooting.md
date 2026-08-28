# Troubleshooting

> **Common problems and their solutions.**

---

## Environment Issues

### `ModuleNotFoundError: No module named 'X'`

The package is not installed.

```bash
# With uv
uv add X

# With pip
pip install X
```

### Jupyter kernel doesn't see packages

The kernel is using a different Python than your environment.

```bash
# Register the correct kernel
python -m ipykernel install --user --name=complete-ml
```

### `pip` not found in venv

uv-managed venvs don't include pip by default. Use `uv` commands instead:

```bash
uv add <package>
uv run python script.py
```

---

## Notebook Issues

### Notebook won't open

The `.ipynb` file may be corrupted. Check it's valid JSON:

```bash
python -c "import json; json.load(open('file.ipynb')); print('valid')"
```

### Notebook depends on hidden state

Always run notebooks from a **clean kernel** (Kernel → Restart & Run All).
Notebooks in this curriculum are designed to be self-contained.

### `Kernel died` / out of memory

Reduce batch sizes, use smaller datasets, or close other applications.

---

## Data Issues

### Dataset download fails

- Check internet connection
- Some datasets require Kaggle credentials (`kagglehub` needs `KAGGLE_USERNAME` and `KAGGLE_KEY`)
- Use the local/synthetic fallback datasets provided

### Missing values / unexpected data

Refer to the data-cleaning units in Phase 04 (Data Analysis).

---

## GPU Issues

### `torch.cuda.is_available()` returns False

- You don't have a GPU, or
- PyTorch is CPU-only build

This curriculum works on CPU. No action needed.

---

## API / LLM Issues

### `OpenAIError: The api_key client option must be set`

You need an API key in your `.env` file:

```bash
OPENAI_API_KEY=your_key_here
```

### Rate limit errors

- Add retries with backoff
- Reduce request frequency
- Use a local model as fallback

---

## Git Issues

### `fatal: not a git repository`

```bash
git init
```

### Committing secrets

Never commit `.env` files. They are in `.gitignore`.

---

## Still Stuck?

1. Read the error message carefully — it usually tells you the fix.
2. Search the error text online.
3. Check the relevant reference doc in `docs/`.
4. Record the mistake in `tracking/mistakes.md` — it's learning material.
