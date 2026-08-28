# LEARNING_SYSTEM

> **How this learning system works.**
> The complete specification of the learning-and-building system.

---

## What This Is

A **complete, deeply connected, implementation-first learning system** that takes a
learner from fundamentals to production ML/AI engineering.

It is not a list of courses, a collection of tutorials, or a topic checklist.
It is a system designed so that the learner can:

- understand what they are doing
- understand why they are doing it
- implement it themselves
- recognize when to use it
- recognize when NOT to use it
- understand the mathematics behind important methods
- debug failures
- compare alternative approaches
- evaluate models properly
- explain their decisions
- build projects independently
- retain knowledge over time
- progressively move from guided implementation to independent engineering

---

## The Learning Structure

```
Phase → Module → Unit → Concept → Implementation → Experiment → Review → Project
```

A **unit** exists because it represents a meaningful learning boundary.

### Example Unit

```
Phase 03 — Machine Learning

Module 03.2 — Regression

Unit 03.2.1 — Linear Regression

Concepts:
- hypothesis, parameters, loss, MSE, gradient, optimization

Implementation:
- from-scratch linear regression
- NumPy implementation
- sklearn implementation

Experiments:
- learning rate, feature scaling, noise, outliers

Evaluation:
- MAE, MSE, RMSE, R²

Failure analysis:
- underfitting, overfitting, unstable optimization

Review:
- closed-book questions, explain-back, rebuild from scratch
```

---

## Every Unit Has a Learning Purpose

Before creating a unit, this design must exist:

```text
WHY: Why does this concept exist?
PREREQUISITES: What must the learner already understand?
LEARNING OUTCOME: What should the learner be able to do afterward?
MENTAL MODEL: What should the learner visualize/understand?
MATHEMATICS: What mathematics is actually necessary?
IMPLEMENTATION: What will be built?
EXPERIMENTS: What will be changed and observed?
FAILURES: What commonly goes wrong?
COMPARISONS: What alternatives should be compared?
REAL-WORLD USE: Where is this actually useful?
ANTI-USE: When should this NOT be used?
RETENTION: How will this knowledge be recalled later?
MASTERY: How will we know the learner actually understands it?
```

---

## Notebook Design Standard

Every serious notebook follows this progression:

```text
1. Title
2. What are we solving?
3. Why does this matter?
4. Prerequisites
5. Learning objectives
6. Mental model
7. Mathematical foundation
8. Small intuitive example
9. First implementation
10. Inspect the implementation
11. Experiment
12. Visualization
13. Failure case
14. Debugging
15. Alternative implementation
16. Library implementation
17. Compare implementations
18. Evaluation
19. Real-world considerations
20. Common mistakes
21. When NOT to use it
22. Challenge
23. Closed-book recall
24. Teach-back questions
25. Summary
26. Further experiment
```

Not every notebook requires every section literally, but the underlying
progression must exist.

---

## Implementation-First Principle

For important algorithms, implement simplified versions from scratch first:

### Linear Regression
```
Python intuition → vectorized NumPy → loss → gradient → gradient descent
→ training loop → prediction → evaluation → sklearn
```

### Logistic Regression
```
sigmoid → probability → log loss → gradient → optimization → classification
```

### Decision Tree
```
splitting → impurity → entropy → Gini → information gain → recursive partitioning
→ simplified tree → sklearn
```

### Neural Networks
```
neuron → activation → forward pass → loss → backpropagation → gradient descent
→ multilayer network → PyTorch
```

The amount of from-scratch implementation decreases as abstraction level increases.

---

## Teach Why, Not Just How

For every important tool/library:

```text
What is it?
Why was it created?
What problem does it solve?
What existed before it?
What does it abstract?
When should I use it?
When should I avoid it?
What are its limitations?
What are common alternatives?
What trade-offs exist?
```

---

## Comparison Notebooks

Learning becomes stronger when alternatives are compared:

```text
Linear Regression vs Random Forest
Random Forest vs XGBoost
Decision Tree vs Gradient Boosting
Bagging vs Boosting
Standardization vs Normalization
PCA vs Feature Selection
CNN vs Vision Transformer
RNN vs Transformer
Fine-tuning vs RAG
RAG vs long-context prompting
LangChain vs direct API implementation
LangGraph vs simple agent loop
Vector database vs keyword search
```

---

## Bad Approaches Are Taught Deliberately

For important topics, controlled failure examples:

```text
Data leakage
Incorrect train/test split
Scaling before splitting
Target leakage
Class imbalance ignored
Wrong metric
Overfitting
Underfitting
Improper cross-validation
Bad feature engineering
Prompt injection
Hallucination
Poor chunking
Bad retrieval
Embedding mismatch
Context pollution
Agent loops
Unbounded tool calls
Weak evaluation
```

Each shows: wrong approach → why it looks reasonable → why it fails →
how to detect it → correct approach → when the wrong approach might be acceptable.

---

## Retention System

### Immediate Recall
At the end of each notebook:
1. Explain the concept.
2. Write the key equation.
3. Explain the algorithm.
4. Explain one failure case.
5. Explain when to use it.

### Delayed Recall
Review notebooks after several units.

### Cumulative Review
Later notebooks deliberately reuse earlier concepts.

---

## Mastery Gates

```
LEVEL 0 — Exposure        I have seen it.
LEVEL 1 — Recognition     I can identify it.
LEVEL 2 — Guided impl     I can implement it with help.
LEVEL 3 — Independent     I can implement it without instructions.
LEVEL 4 — Debugging       I can diagnose failures.
LEVEL 5 — Decision making I can choose when/why to use it.
LEVEL 6 — Teaching        I can explain it clearly to another person.
```

- Important concepts → at least Level 4
- Core concepts → Level 5
- Foundational concepts → Level 6

---

## Tracking Systems

- `tracking/progress.md` — unit completion and mastery
- `tracking/mistakes.md` — mistake ledger
- `tracking/concepts.md` — concept graph
- `tracking/experiments.md` — experiment log
- `tracking/review-log.md` — review schedule
- `tracking/project-log.md` — project outcomes

---

## Project Progression

| Level | Description |
|-------|-------------|
| 1 | Guided — everything specified |
| 2 | Partially guided — dataset + objective, decisions left to learner |
| 3 | Open-ended — only problem statement |
| 4 | Engineering challenge — requirements + constraints + evaluation |
| 5 | Production-style capstone — full system design |

---

## Build Process

```
STEP A — PLAN       Create a short unit plan
STEP B — IMPLEMENT  Create the required notebook(s)
STEP C — VERIFY     Execute them from a clean kernel
STEP D — REVIEW     Check correctness, progression, quality
STEP E — UPDATE     Update BUILD_STATUS, PROJECT_STATE, tracking
STEP F — COMMIT     Commit the completed unit
STEP G — STOP       Report and continue
```

---

## Definition of "Complete"

The project is complete when:

- [ ] Roadmap exists
- [ ] Dependencies are mapped
- [ ] Notebooks are implemented
- [ ] Notebooks execute successfully
- [ ] Concepts are connected
- [ ] Mistakes are documented
- [ ] Reviews exist
- [ ] Projects exist
- [ ] Evaluation is included
- [ ] Engineering practices are included
- [ ] Security is included
- [ ] Deployment is included
- [ ] Capstones exist
- [ ] Progress is tracked
- [ ] Git history is maintained

**Most importantly:** the learner can leave the final capstone and independently
design and build a new AI/ML system never explicitly taught in the curriculum.
