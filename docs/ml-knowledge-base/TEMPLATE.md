# ALGORITHM NOTE TEMPLATE (SHARED)

> This is the master template every algorithm note in this knowledge base MUST follow.
> Copy the section structure below and fill it with the specific algorithm's content.

## Reference: the 53-section master structure

Every algorithm note must contain these sections IN THIS ORDER. Sections may be omitted ONLY when genuinely irrelevant to the algorithm (e.g., "Feature Scaling" for a tree that doesn't need it).

```markdown
# [Sequence]. [Algorithm Name]

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐ (adjust stars)
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview
(table: Algorithm name, Category, Type, Parametric/Non-parametric, Generative/Discriminative, Main objective, Input, Output, Core idea, Typical use cases)

## 02. One-Line Definition
### Beginner Definition
### Technical Definition

## 03. Intuition
(no math, real-life analogy, simple example, step-by-step reasoning)

## 04. Problem It Solves
(what problem existed, data, what we want, why useful, small example)

## 05. Where It Fits in Machine Learning
(ASCII tree showing position in ML ecosystem)

## 06. Important Terminology
(table: Term | Simple Meaning | Technical Meaning)

## 07. Input and Output
(input data, features, target, labels, parameters, hyperparameters, output, prediction)

## 08. Mathematical Foundation
(basic idea, notation, core equation, symbol explanations, interpretation, required math concepts)

## 09. Core Formula
For EVERY formula:
```text
Formula
```
### Meaning
### Symbols
### Intuition
### Example (tiny dataset, calculated)

## 10. Derivation
(start from basic equation, explain every transformation, or state "optional" + give important result)

## 11. How the Algorithm Works
```text
Input ↓ Preprocessing ↓ Initialization ↓ Prediction ↓ Loss ↓ Optimization ↓ Convergence ↓ Final Model ↓ Prediction
```

## 12. Training Process
(pre-training, during, what's learned, changes per iteration, stopping, final model contents)

## 13. Objective Function / Loss Function
(what's optimized, loss, cost, objective, why chosen, high/low loss meaning)

## 14. Optimization
(definition, why, method, gradient, learning rate, update, convergence, local/global optimum)
```text
Current params ↓ Prediction ↓ Loss ↓ Gradient ↓ Update ↓ New params ↓ Repeat
```

## 15. Complete Numerical Example
(2–5 observations, manual step-by-step: input, params, prediction, error, loss, update, final)
**VERIFIED EXAMPLE** (state it was hand-verified)

## 16. Visual Explanation
(ASCII diagrams: regression line, decision boundary, tree, clusters, hyperplane, network)

## 17. Algorithm / Pseudocode
(clean numbered pseudocode)

## 18. From-Scratch Implementation
(beginner-readable Python, comments, follows math)

## 19. Code Explanation
```text
Code ↓ What does it do? ↓ Why required? ↓ Mathematical concept?
```

## 20. Library Implementation
(scikit-learn / numpy / pandas / scipy; imports, model, params, fit, predict, evaluate)

## 21. Hyperparameters
(table: Hyperparameter | Meaning | Effect | Typical Consideration; too high / too low / how to tune)

## 22. Parameters vs Hyperparameters
### Parameters (learned)
### Hyperparameters (chosen)

## 23. Assumptions
(each: what, why, how to check, what if violated, solution)

## 24. Data Requirements
(data type, numerical/categorical, missing values, outliers, scaling, feature engineering, dataset size, class imbalance)

## 25. Feature Scaling
(Required / Recommended / Optional / Unnecessary + why + methods)

## 26. Evaluation Metrics
(each: definition, formula, interpretation, when to use, when NOT to use; separate Training Objective ≠ Evaluation Metric)

## 27. Advantages
(each with why it matters)

## 28. Disadvantages
(each with practical consequences)

## 29. When to Use
(✓ checklist)

## 30. When NOT to Use
(✗ checklist)

## 31. Real-World Applications
(each: Problem ↓ Input ↓ Algorithm ↓ Output)

## 32. Failure Cases
(data, mathematical, optimization, generalization, practical)

## 33. Overfitting and Underfitting

## 34. Bias-Variance Perspective

## 35. Comparison With Similar Algorithms
(table: Algorithm | Main Idea | Strength | Weakness | Best Use)

## 36. Algorithm Selection Guide
(decision tree / pseudocode)

## 37. Common Mistakes
```text
❌ Mistake
Why wrong: ...
Correct: ...
```

## 38. Interview Questions
### Beginner (with answers)
### Intermediate (with answers)
### Advanced (with answers)

## 39. GATE / Exam Perspective
(formulas, concepts, traps, patterns; NO invented PYQs — verify before including)

## 40. Coding Practice
Level 1 Basic → Level 7 Real-world case study

## 41. Practical ML Workflow
(problem → data → EDA → cleaning → feature eng → split → preprocess → train → tune → evaluate → error analysis → deploy → monitor)

## 42. Complexity
(training time, prediction time, space, scaling with samples/features/complexity)

## 43. Advanced Concepts
(regularization, kernels, convexity, probabilistic interpretation, etc. — only genuinely relevant ones)

## 44. Connections to Other Algorithms
(knowledge-graph tree)

## 45. If You Remember Only 5 Things
(EXACTLY 5 numbered points)

## 46. Cheat Sheet
(compact: Algorithm, Category, Goal, Input, Output, Core Formula, Loss, Optimization, Parameters, Hyperparameters, Assumptions, Advantages, Disadvantages, Use When, Avoid When, Related, Key Exam Points, Key Interview Points)

## 47. Final Mental Model
(ASCII end-to-end flow)

## 48. Knowledge Check
### Recall (5)
### Understanding (5)
### Application (5)
### Mathematical (5)
### Interview (5)
### Problem Solving (5)
## Answers (explained)

## 49. Final Learning Checklist
(bullet checkboxes, 20+ items)

## 50. Quality Control Note
(self-review against accuracy, beginner-friendliness, math depth, practical depth, exam depth, structure)
```

---

## RULES FOR AGENTS GENERATING FILES

1. Follow the template section order EXACTLY.
2. Define every term before using it.
3. Explain every formula (meaning, symbols, intuition, example).
4. Use 2–5 row datasets for numerical examples, and hand-verify them.
5. Write from-scratch code BEFORE library code.
6. Never invent GATE previous-year questions — omit or mark clearly as a representative pattern question, NOT a real PYQ.
7. Keep paragraphs short. Use tables where useful.
8. Do NOT add comments to code unless genuinely clarifying.
9. Mark level at top with star ratings.
10. Save each file with the exact numbered filename given.
