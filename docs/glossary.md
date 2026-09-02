# Glossary

> A plain-language dictionary for the words used in this course.
> Skim this once now; **come back whenever you meet a word you don't know.**

---

## Getting Started

| Term | Plain meaning |
|------|---------------|
| **Terminal / PowerShell / Command Line / CLI** | A window where you type commands instead of clicking. Windows calls it *PowerShell*, Mac calls it *Terminal*. |
| **Command** | Something you type and press Enter on, e.g. `uv sync`. |
| **Directory** | A folder. `cd` changes directory. |
| **`cd`** | "Change directory" — move into a folder. |
| **`git clone`** | Download a copy of a repository from the internet. |
| **PATH** | The list of places your computer searches for programs. Programs not "on PATH" can't be run by name. |
| **Keyboard shortcut** | A combo like **Shift + Enter** (run a Jupyter cell) that does an action instantly. |

## Python & Environment

| Term | Plain meaning |
|------|---------------|
| **Python** | The programming language used throughout this course. |
| **Script** | A plain-text file of Python code (`.py`) you run whole. |
| **Interpreter** | The program that reads and runs your Python code. |
| **Package / library** | Ready-made code others wrote (NumPy for math, pandas for tables). |
| **Dependency** | Any package your project needs to run. |
| **Virtual environment (`.venv`)** | A private folder inside a project holding its Python + packages, so they don't clash with other projects. |
| **`uv` / `pip`** | Tools that install packages. `uv` is faster; both do the same core job. |
| **`uv sync`** | "Install exactly the packages this project needs" — reads `pyproject.toml`/`uv.lock`. |
| **Activate** | Turn on the virtual environment (`source .venv/bin/activate`). Your prompt shows `(.venv)`. |
| **`pyproject.toml`** | The project's config file: name, Python version, dependency list. |
| **`requirements.txt`** | An older-style dependency list (readable by `pip too`). |

## Jupyter & Notebooks

| Term | Plain meaning |
|------|---------------|
| **Notebook (`.ipynb`)** | A document mixing readable text (markdown) with runnable Python code (code cells). |
| **Cell** | One block in a notebook: either code or text. |
| **Run a cell** | Execute that block with **Shift + Enter**. |
| **Kernel** | The Python program inside Jupyter that runs your cells. |
| **Restart kernel** | Reboot that Python program (clears variables/memory). *Always do Kernel → Restart & Run All before trusting a notebook.* |
| **Markdown** | Plain-text formatting (headings, lists) rendered nicely in notebooks. |
| **Output** | Whatever a code cell printed/plotted after running. |
| **Error cell** | A cell that failed — red `Traceback`. Reading it is a learned skill. |

## Git

| Term | Plain meaning |
|------|---------------|
| **Git** | A tool that saves snapshots of your files so you can undo mistakes. |
| **Repository / repo** | The project folder plus Git's history of it. |
| **`git add`** | Pick files to include in the next snapshot. |
| **`git commit`** | Take the snapshot (a checkpoint you can return to). |
| **`git push` / `git pull`** | Send / receive snapshots to a remote (e.g. GitHub). |
| **Branch** | A separate line of work; great for experiments that shouldn't touch the main code. |
| **`.gitignore`** | A file listing what Git should *not* track (secrets, big files, caches). |

## Data & ML (you'll meet these from Phase 04)

| Term | Plain meaning |
|------|---------------|
| **Dataset** | A collection of data used for learning, e.g. a CSV table. |
| **CSV** | A text file of rows/columns, the most common data format. |
| **Dataframe** | Pandas' way of storing a table of data. |
| **Feature** | A column of input the model sees (e.g. age, income). |
| **Label / target** | The thing the model is trying to predict (e.g. "will churn?"). |
| **Training** | Showing a model data while adjusting it to make better predictions. |
| **Inference / prediction** | Using a trained model on new data to get an answer. |
| **Model** | The mathematical result that maps features → answers. |
| **Metrics** | Numbers that say how good a model is (accuracy, F1, ROC-AUC...). |
| **Overfitting** | Model memorized the training set but can't handle new data. |
| **Pipeline** | A chained sequence of steps (clean → transform → model → predict). |
| **MLOps** | The practice of keeping models working reliably in production. |

## AI / LLMs (you'll meet these from Phase 09)

| Term | Plain meaning |
|------|---------------|
| **AI (Artificial Intelligence)** | Machines doing tasks that need human-like judgment. |
| **ML (Machine Learning)** | AI approach: learn patterns from data instead of being coded rules. |
| **DL (Deep Learning)** | ML using neural networks — big layered function approximators. |
| **Neural network** | A web of connected "neurons" that learns by adjusting weights. |
| **Model (LLM)** | An AI trained on huge text to predict/continue/generate text. |
| **LLM** | *Large Language Model* — GPT, Claude, Gemini. |
| **Token** | The small text pieces an LLM reasons in (~1 word ≈ 1–2 tokens). |
| **Prompt** | The text you send an LLM to ask it to do something. |
| **Context window** | Max tokens an LLM can "see" at once (memory while answering). |
| **Fine-tuning** | Extra-training a model on your data to specialize it. |
| **Embedding** | Turning text into a list of numbers capturing its meaning, for search/math. |
| **Vector database** | A database built to search by meaning (embeddings) instead of exact text. |
| **RAG** | *Retrieval-Augmented Generation* — fetch relevant documents, then answer with them. |
| **Agent** | An LLM that uses tools/action loops to complete multi-step tasks. |
| **Guardrails** | Safety filters: block injections, PII leaks, out-of-scope questions. |
| **Evals / Evaluation dataset** | A fixed set of cases used to prove a model's quality. |
| **API key** | A secret code authorizing you to call an online AI service. |

## Course Structure

| Term | Plain meaning |
|------|---------------|
| **Phase** | One major topic block (e.g. Phase 05 = Machine Learning). 18 in total. |
| **Unit** | One notebook inside a phase (e.g. `05.07 Gradient Boosting`). |
| **Capstone** | A final portfolio project you design yourself. |
| **Mastery level** | 0 = seen it → 6 = can teach it. Used in `tracking/progress.md`. |
| **Exit criteria** | A checklist telling you when you're ready for the next unit. |

---

**Tip:** if a notebook or doc uses a word not here, search this file — and if it's still missing,
add it. This glossary is meant to grow as you do.