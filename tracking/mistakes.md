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

<!-- Add new mistakes below this line -->

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
