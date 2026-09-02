# Environment Setup

> **How to set up the environment for this learning system.**
> Total beginner? Read [`getting-started.md`](getting-started.md) *before* this page —
> it walks you through the same steps with no jargon and pre-empts the common first-hour errors.

---

## Recommended: uv (used in this project)

This project uses **uv** for environment management.

> **Which is right for you?**
> - **You downloaded this repo** → skip to step 2 and run `uv sync` (installs exactly
>   what the project needs from `pyproject.toml`/`uv.lock`). `uv init` / `uv add`
>   (below) are only for **creating a brand-new project**, not for using this one.
> - **Creating your own new project** → `uv init` then `uv add <packages>`.

### 1. Install uv

```bash
# Windows (PowerShell)
pip install uv
# or
winget install astral-sh.uv
```

### 2. Set up the environment

```bash
# In this repository (after git clone + cd into the folder):
uv sync
```

### 3. Activate the environment

```bash
# Windows (PowerShell)
.\.venv\Scripts\activate
```

### 4. Run Jupyter

```bash
jupyter lab
# or
jupyter notebook
```

---

## Alternative: pip + venv

```bash
# Create a virtual environment
python -m venv .venv

# Activate (Windows)
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Alternative: conda

```bash
conda create -n complete-ml python=3.13
conda activate complete-ml
pip install -r requirements.txt
```

---

## Required Packages

See `requirements.txt` for the full list. Core packages:

- `numpy`, `pandas`, `scipy` — numerical computing
- `scikit-learn` — classical ML
- `matplotlib`, `seaborn`, `plotly` — visualization
- `torch` — deep learning
- `transformers`, `datasets`, `tokenizers` — Hugging Face ecosystem
- `sentence-transformers` — embeddings
- `faiss-cpu` — vector search
- `langchain`, `langchain-openai`, `langchain-community` — LLM framework
- `langgraph` — agent orchestration
- `chromadb` — vector database
- `xgboost`, `lightgbm` — gradient boosting
- `fastapi`, `uvicorn` — API serving
- `pydantic` — data validation
- `python-dotenv` — environment variables
- `requests`, `beautifulsoup4` — web/data fetching
- `pymupdf`, `pypdf` — PDF processing
- `tiktoken` — tokenization
- `jupyterlab`, `notebook`, `nbconvert` — notebooks
- `kagglehub`, `opendatasets` — dataset access

---

## Environment Variables

Create a `.env` file in the project root (never commit it):

```bash
# .env
OPENAI_API_KEY=your_key_here
# Add other API keys as needed
```

See `.env.example` for the template.

---

## GPU Setup (Optional)

This curriculum works on CPU. If you have a GPU:

```bash
# Install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

---

## Verify Your Environment

Run the environment-verification notebook:

```
notebooks/00_environment/00_environment_verification.ipynb
```

It checks Python version, all packages, GPU, and internet connectivity.
