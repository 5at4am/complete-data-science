# Mistake Ledger

> **Every mistake is learning material.**
> Record mistakes honestly. Review them regularly. Do not hide errors.

---

## How to Use

For every meaningful mistake, add an entry with the following structure:

```text
### Mistake #N

- **Date:** YYYY-MM-DD
- **Unit:** e.g., 05.2 Linear Regression
- **Mistake:** What did I do wrong?
- **What I expected:** What did I think would happen?
- **What actually happened:** What actually happened?
- **Root cause:** Why did it happen?
- **Correct understanding:** What is the right way to think about it?
- **How I detected it:** How did I find the error?
- **How to prevent it:** What will I do differently next time?
- **Related concept:** What concept does this relate to?
- **Review date:** When will I review this?
```

---

## Mistakes

### Mistake #1

- **Date:** 2026-08-28
- **Unit:** 01.4 OOP
- **Mistake:** In the `SimpleLinearModel.fit` method, I wrote `pred = self.slope * X + self.intercept` and `error * X` assuming list arithmetic works like NumPy arrays.
- **What I expected:** Element-wise multiplication and addition on Python lists.
- **What actually happened:** `TypeError: can't multiply sequence by non-int of type 'float'` — Python lists don't support element-wise arithmetic.
- **Root cause:** Confusing Python lists with NumPy arrays. Lists only support `*` with an integer (repetition), not element-wise operations.
- **Correct understanding:** Python lists are not vectorized. Use list comprehensions (or NumPy arrays) for element-wise math.
- **How I detected it:** Notebook execution failed with a TypeError during verification.
- **How to prevent it:** Remember lists ≠ arrays. For element-wise math, use list comprehensions or NumPy.
- **Related concept:** Data structures, NumPy vectorization.
- **Review date:** 2026-09-04

---

## Common Mistake Patterns (to watch for)

### Data
- Scaling before splitting (leakage)
- Target leakage
- Ignoring class imbalance
- Using test set multiple times

### Modeling
- Overfitting (memorizing training data)
- Underfitting (too simple)
- Wrong metric for the problem
- Improper cross-validation

### Deep Learning
- Vanishing/exploding gradients
- Wrong learning rate
- Not normalizing inputs
- Overfitting without regularization

### LLM/RAG
- Poor chunking
- Embedding mismatch
- Context pollution
- Prompt injection
- Hallucination
- Unbounded agent loops
