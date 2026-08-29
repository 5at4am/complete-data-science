# Phase 00 — Environment Setup

> **Goal:** Get a working, verified, reproducible environment so all future learning is frictionless.

**Difficulty:** 🟢 Beginner  
**Priority:** Essential  
**Prerequisites:** None  
**Duration:** Short setup phase  
**Mastery target:** Level 3 — independent environment setup and troubleshooting

---

## Why This Phase Exists

Before learning ML, the learner needs a workspace that can run code, save experiments, recover from mistakes, and reproduce results. Environment problems are not side issues; in real projects, broken dependencies, wrong kernels, missing packages, and untracked changes waste enormous time.

### Phase Mental Model

Your environment is the workshop:

```text
Operating system → Python installation → virtual environment → packages → notebook/kernel → Git history
```

If one layer is wrong, code may fail even when the logic is correct.

### What This Phase Prepares For

- running notebooks in later phases
- installing ML/DL libraries
- tracking experiments safely
- recovering from broken code
- sharing work with another person or machine

---

## Units

### Unit 00.1 — Environment Verification

**What is it?**  
Checking that Python, packages, notebooks, GPU/CPU, file paths, and internet access work correctly.

**Why does it matter?**  
If the environment is broken, every later error becomes confusing: is the model wrong, or is the setup wrong?

**Why learn it here?**  
It is the foundation for every future notebook and project.

**Prerequisites:** None.

**Core concepts:**

- Python version
- package import
- kernel selection
- current working directory
- CPU/GPU availability
- internet/API availability
- reproducibility check

**Implementation:** Run the environment-verification notebook.

**Notebook:** `notebooks/00_environment/00_environment_verification.ipynb`

**Simple example:**

```python
import sys
import numpy as np

print(sys.version)
print(np.__version__)
```

**Common mistakes:**

- running a notebook with the wrong kernel
- installing a package into one environment but running another
- assuming a path works on every machine
- ignoring warnings during installation

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| `ModuleNotFoundError` | Package installed in the wrong environment | Check `sys.executable` | Install using the same interpreter/kernel |
| Notebook cannot find a file | Wrong working directory | Print `Path.cwd()` | Use project-relative paths |
| GPU not detected | Missing driver/library or CPU-only package | Run framework device check | Use CPU for now or install correct GPU stack later |

**Hands-On Practice:**

1. Basic: print Python version and current working directory.
2. Guided: import NumPy, Pandas, Matplotlib, and scikit-learn.
3. Independent: create a short script that checks the environment and reports pass/fail.
4. Realistic: intentionally select the wrong kernel, observe the error, then fix it.

**Knowledge Check:**

- Why can a package be installed but still not import inside a notebook?
- Why should paths be relative to the project root?
- What is the difference between Python version and package version?

**Exit Criteria:**

- You can verify Python and key packages.
- You can identify the active interpreter/kernel.
- You can explain whether an error is environment-related or code-related.

**Next Step:** Learn the notebook workflow so verified code can be explored, documented, and rerun.

---

### Unit 00.2 — Jupyter Workflow

**What is it?**  
Using notebooks to mix explanation, code, outputs, plots, and experiments.

**Why does it matter?**  
ML learning often involves inspecting intermediate data, visualizing results, and experimenting quickly.

**Prerequisites:** Unit 00.1.

**Core concepts:**

- cells and execution order
- markdown explanations
- kernels
- restarting and running all cells
- outputs and hidden state
- notebooks vs scripts

**How it works:**  
A notebook stores code cells, text cells, and outputs. The kernel stores runtime memory. If cells are run out of order, the notebook can lie about whether it works from a clean start.

**Simple example:**

```python
x = 10
x * 2
```

Then restart the kernel and run only the second cell to see why execution order matters.

**Common mistakes:**

- relying on variables created in old cells
- not restarting before submitting work
- leaving huge outputs in notebooks
- mixing too much production code into notebooks

**Best Practices:**

- Use markdown to explain intent before code.
- Restart kernel and run all before calling a notebook complete.
- Move reusable logic into `.py` files as projects grow.
- Keep notebooks readable: one idea per section.

**Hands-On Practice:**

1. Basic: create markdown and code cells.
2. Guided: run a simple data summary and plot.
3. Independent: restart the kernel and make the notebook run top-to-bottom.
4. Challenge: break execution order intentionally, diagnose the hidden-state problem, and fix it.

**Exit Criteria:**

- You can create, run, restart, and validate a notebook.
- You can explain why “Restart & Run All” matters.

**Next Step:** Learn Git so notebook and code changes can be tracked safely.

---

### Unit 00.3 — Git & Version Control

**What is it?**  
Git is a version history system for code and documents.

**Why does it matter?**  
It lets you recover from mistakes, compare changes, work on branches, and share work professionally.

**Why learn it here?**  
Every later exercise and project should be version-controlled.

**Mental model:**  
A Git branch is a separate development path. Commits are save points on that path.

**Core concepts:**

- repository
- working tree
- staging area
- commit
- branch
- merge
- remote
- pull request
- conflict

**Implementation:** Initialize or use a repo, check status, commit, branch, and merge.

**Syntax / Commands:**

```bash
git status
git add <file>
git commit -m "describe change"
git switch -c feature/name
git log --oneline
```

**What happens internally?**

- `git add` selects changes for the next snapshot.
- `git commit` creates a saved snapshot with a message.
- a branch points to a commit and moves as new commits are added.
- merging combines histories; conflicts happen when Git cannot safely combine edits.

**Common mistakes:**

- committing generated files, secrets, or huge data
- developing directly on `main` for risky work
- using vague commit messages
- ignoring `git status`

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Wrong files about to commit | Staged too much | `git status` | Unstage unwanted files before committing |
| Merge conflict | Same lines changed in two branches | Open conflicted file | Choose correct content and commit resolution |
| Lost change | Change overwritten or branch changed | `git log`, `git reflog` | Restore from commit/reflog if available |

**Hands-On Practice:**

1. Basic: create a file and commit it.
2. Guided: create a branch, edit a file, and merge it.
3. Independent: make two small commits with clear messages.
4. Realistic: create a conflict intentionally and resolve it.

**Exit Criteria:**

- You can check what changed before committing.
- You can create and switch branches.
- You can explain why teams avoid direct risky work on `main`.
- You can recover from simple mistakes.

**Next Step:** Learn environments and packages so projects remain reproducible.

---

### Unit 00.4 — Python Environments & Packages

**What is it?**  
An environment is an isolated Python workspace with its own packages.

**Why does it matter?**  
Different projects need different package versions. Isolation prevents one project from breaking another.

**Mental model:**  
An environment is a toolbox for one project. Do not mix every project’s tools into one global toolbox.

**Core concepts:**

- Python interpreter
- virtual environment
- dependency
- version pinning
- lock file
- requirements file
- reproducible install

**Implementation:** Create/use a virtual environment and install packages.

**Common commands:**

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m pip freeze
```

If the project uses `uv`, prefer the project’s documented `uv` workflow.

**Common mistakes:**

- installing globally instead of inside the project environment
- not recording dependencies
- mixing conda/pip/uv randomly without understanding where packages go
- ignoring version conflicts

**Decision Guidance:**

| Tool | Use When | Avoid When |
|---|---|---|
| `venv` + `pip` | Simple Python projects | You need advanced dependency locking/workspaces |
| `uv` | Fast reproducible Python dependency management | Team/project does not use it and no lockfile exists |
| Conda | Complex scientific/GPU stacks | You only need simple pure-Python dependencies |

**Hands-On Practice:**

1. Basic: activate the project environment and print `sys.executable`.
2. Guided: install a small package and import it.
3. Independent: create a fresh temporary environment and reproduce a simple script.
4. Challenge: diagnose a version conflict from an error message.

**Exit Criteria:**

- You can explain which environment is active.
- You can install dependencies reproducibly.
- You can diagnose common package/import problems.

**Next Step:** Phase 01 uses this environment to build real Python fluency.

---

## Mini Project — Reproducible Starter Workspace

**Objective:** Prove that the learning environment is reproducible and safe to use.

**Problem Statement:** A new learner should be able to clone/open the project, install dependencies, run a verification notebook, and commit a small change without confusion.

**Requirements:**

- verify Python and package imports
- create a notebook that runs top-to-bottom
- create a small Python script that prints environment info
- make a Git branch and commit the work
- document any setup issue and how it was fixed

**Expected Output:**

- working verification notebook
- short setup note or checklist
- clean Git status after commit

**Evaluation Criteria:**

- environment is reproducible
- notebook runs from a clean kernel
- dependencies are documented
- no secrets or large generated files committed

---

## Phase Review Checklist

- [ ] Environment verification passes.
- [ ] Jupyter workflow is comfortable.
- [ ] Git basics are understood and practiced.
- [ ] Environment/package management is understood.
- [ ] At least one setup failure was diagnosed and documented.
- [ ] Starter workspace mini project completed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Verify your environment is working.
2. Create a new notebook and run it from a clean kernel.
3. Initialize or use a Git repo, create a branch, and commit work.
4. Set up a Python environment from scratch.
5. Explain the difference between a code bug and an environment bug.

## Interview / Explain-Back Questions

- Why do Python projects use virtual environments?
- What does a notebook kernel do?
- What is the difference between `git add` and `git commit`?
- Why should secrets not be committed?
- How would you debug `ModuleNotFoundError`?

## Exit Criteria

Move to Phase 01 only when you can run code, manage dependencies, use notebooks, and commit work without needing step-by-step help.
