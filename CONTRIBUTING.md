# CONTRIBUTING

> **Guidelines for contributing to this learning system.**

---

## Principles

- **Better understanding over more content.**
- **Better implementation over more notebooks.**
- **Working system over finishing faster.**
- **Teaching the mistake over hiding it.**

---

## How to Contribute

### 1. Follow the Build Process

```
STEP A — PLAN       Create a short unit plan
STEP B — IMPLEMENT  Create the required notebook(s)
STEP C — VERIFY     Execute them from a clean kernel
STEP D — REVIEW     Check correctness, progression, quality
STEP E — UPDATE     Update BUILD_STATUS, PROJECT_STATE, tracking
STEP F — COMMIT     Commit the completed unit
STEP G — STOP       Report and continue
```

### 2. Follow the Notebook Design Standard

See `LEARNING_SYSTEM.md` for the full standard.

### 3. Verify Before Committing

Never mark a notebook complete merely because it was written. Execute it from a
clean kernel, fix errors, and confirm outputs.

### 4. Never Fabricate Execution

If execution cannot happen (missing package, no GPU, no internet, missing dataset,
missing API key), mark it `BLOCKED` and explain the exact reason.

### 5. Never Commit Secrets

API keys go in `.env` (gitignored), never in notebooks or committed files.

### 6. Use Meaningful Commits

```
phase-00-environment-unit-01-complete
phase-01-foundations-unit-03-complete
phase-02-statistics-unit-05-complete
```

---

## Code Quality Rules

Every notebook must:
- run from top to bottom
- use deterministic seeds where appropriate
- avoid hidden state
- avoid unexplained magic numbers
- use meaningful variable names
- explain non-obvious code
- handle errors appropriately
- avoid unnecessary duplication
- separate configuration from logic where practical

---

## Documentation

Update these when contributing:
- `BUILD_STATUS.md` — what was built and verified
- `PROJECT_STATE.md` — full project state
- `CHANGELOG.md` — change log
- `tracking/progress.md` — unit completion
- `tracking/concepts.md` — concept graph
- `tracking/experiments.md` — experiment log
- `tracking/mistakes.md` — mistake ledger
