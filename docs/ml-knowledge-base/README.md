# 🧠 MACHINE LEARNING KNOWLEDGE BASE

> A complete, beginner-friendly, interview-ready, implementation-oriented Machine Learning knowledge system with every file numbered step-by-step for guided learning.

---

## 📚 HOW TO USE THIS KNOWLEDGE BASE

Each algorithm note is a **self-contained learning system** that follows the same 53-section structure so that once you learn one, you know where to find everything in every other.

### Reading Order

```text
Complete Beginner
        ↓
Fundamentals
        ↓
Mathematical Understanding
        ↓
Algorithm Understanding
        ↓
Implementation
        ↓
Model Selection
        ↓
Problem Solving
        ↓
Real-World ML
        ↓
Advanced ML
```

### Recommended Path

1. Start with **Supervised Learning → Regression → 01. Linear Regression**
2. Then follow the numbered sequence — every file is pre-numbered in learning order
3. Complete one family before moving to the next
4. Implement every algorithm by hand as you go

---

## 🗺️ KNOWLEDGE BASE MAP

```text
MACHINE LEARNING
│
├── 1. SUPERVISED LEARNING
│   │
│   ├── A. REGRESSION
│   │   ├── 01. Linear Regression
│   │   ├── 02. Polynomial Regression
│   │   ├── 03. Ridge Regression
│   │   ├── 04. Lasso Regression
│   │   ├── 05. Elastic Net
│   │   ├── 06. Bayesian Regression
│   │   ├── 07. Huber Regression
│   │   ├── 08. Quantile Regression
│   │   ├── 09. Support Vector Regression
│   │   ├── 10. Decision Tree Regression
│   │   ├── 11. Random Forest Regression
│   │   ├── 12. Extra Trees Regression
│   │   ├── 13. Gradient Boosting Regression
│   │   ├── 14. AdaBoost Regression
│   │   ├── 15. XGBoost
│   │   ├── 16. LightGBM
│   │   └── 17. CatBoost
│   │
│   └── B. CLASSIFICATION
│       ├── 01. Logistic Regression
│       ├── 02. K-Nearest Neighbors
│       ├── 03. Naive Bayes
│       ├── 04. Decision Tree
│       ├── 05. Random Forest
│       ├── 06. Extra Trees
│       ├── 07. Support Vector Machine
│       ├── 08. Gradient Boosting
│       ├── 09. AdaBoost
│       ├── 10. XGBoost
│       ├── 11. LightGBM
│       ├── 12. CatBoost
│       └── 13. Neural Networks
│
├── 2. UNSUPERVISED LEARNING
│   │
│   ├── A. CLUSTERING
│   │   ├── 01. K-Means
│   │   ├── 02. K-Medoids
│   │   ├── 03. Hierarchical Clustering
│   │   ├── 04. DBSCAN
│   │   ├── 05. HDBSCAN
│   │   ├── 06. Mean Shift
│   │   ├── 07. Gaussian Mixture Model
│   │   └── 08. Spectral Clustering
│   │
│   ├── B. DIMENSIONALITY REDUCTION
│   │   ├── 01. PCA
│   │   ├── 02. Kernel PCA
│   │   ├── 03. LDA
│   │   ├── 04. t-SNE
│   │   ├── 05. UMAP
│   │   ├── 06. SVD
│   │   └── 07. NMF
│   │
│   └── C. ASSOCIATION RULE LEARNING
│       ├── 01. Apriori
│       ├── 02. FP-Growth
│       └── 03. Eclat
│
├── 3. SEMI-SUPERVISED LEARNING
│   ├── 01. Self-Training
│   ├── 02. Label Propagation
│   ├── 03. Label Spreading
│   └── 04. Semi-Supervised SVM
│
├── 4. REINFORCEMENT LEARNING
│   ├── 01. Q-Learning
│   ├── 02. SARSA
│   ├── 03. Monte Carlo Methods
│   ├── 04. Temporal Difference Learning
│   ├── 05. Deep Q-Network
│   ├── 06. Policy Gradient
│   ├── 07. Actor-Critic
│   ├── 08. PPO
│   ├── 09. A2C
│   ├── 10. A3C
│   ├── 11. DDPG
│   ├── 12. TD3
│   └── 13. SAC
│
└── 5. ENSEMBLE LEARNING
    ├── 01. Bagging
    ├── 02. Random Forest
    ├── 03. Extra Trees
    ├── 04. Boosting
    │   ├── 04a. AdaBoost
    │   ├── 04b. Gradient Boosting
    │   ├── 04c. XGBoost
    │   ├── 04d. LightGBM
    │   └── 04e. CatBoost
    └── 05. Stacking
```

---

## 📘 THE 53-SECTION MASTER STRUCTURE

Every algorithm note follows this exact numbered structure:

```text
01. Algorithm Overview          — category, type, objective, I/O, core idea
02. One-Line Definition         — beginner + technical definition
03. Intuition                   — no-math explanation + analogy
04. Problem It Solves           — problem, data, objective, use case
05. Where It Fits in ML         — position in the ML ecosystem tree
06. Important Terminology       — table of terms: simple + technical meaning
07. Input and Output            — features, target, params, hyperparams, prediction
08. Mathematical Foundation     — gradual math intro + notation
09. Core Formula                — every formula: meaning, symbols, intuition, example
10. Derivation                  — from basic equation, every transformation explained
11. How the Algorithm Works     — complete step-by-step process flow
12. Training Process            — pre/during/post training, what is learned
13. Objective / Loss Function   — what is optimized, why
14. Optimization                — gradient, learning rate, update, convergence
15. Complete Numerical Example  — small dataset, manual calculation, verified
16. Visual Explanation          — ASCII diagrams, decision boundaries, etc.
17. Algorithm / Pseudocode      — clean, structured pseudocode
18. From-Scratch Implementation — clean beginner-readable Python
19. Code Explanation            — code ↔ formula connection
20. Library Implementation      — sklearn / numpy / pandas / scipy
21. Hyperparameters             — table: meaning, effect, tuning
22. Parameters vs Hyperparams   — what's learned vs what's chosen
23. Assumptions                 — each: what, why, how to check, violation effects
24. Data Requirements           — type, missing values, outliers, scaling, size
25. Feature Scaling             — required / recommended / optional / unnecessary
26. Evaluation Metrics          — definition, formula, interpretation, when to use
27. Advantages                  — strengths with reasons
28. Disadvantages               — weaknesses with consequences
29. When to Use                 — concrete conditions
30. When NOT to Use             — situations to pick another model
31. Real-World Applications     — problem → input → algorithm → output
32. Failure Cases               — data / mathematical / optimization / generalization failures
33. Overfitting & Underfitting  — bias/variance connection
34. Bias-Variance Perspective   — model complexity vs generalization
35. Comparison With Similar     — comparison table
36. Algorithm Selection Guide   — decision tree
37. Common Mistakes             — ❌ / why / correct approach
38. Interview Questions         — beginner / intermediate / advanced with answers
39. GATE / Exam Perspective     — formulas, traps, patterns (verified only)
40. Coding Practice             — 7 levels of increasing difficulty
41. Practical ML Workflow       — full project pipeline
42. Complexity                  — time / space / scaling
43. Advanced Concepts           — regularization, kernels, convexity, etc.
44. Connections to Other        — knowledge graph
45. If You Remember Only 5      — 5 highest-value facts
46. Cheat Sheet                 — compact revision sheet
47. Final Mental Model          — concise end-to-end mental model
48. Knowledge Check             — recall/understanding/application/math/interview/problem-solving + answers
49. Final Learning Checklist    — checkboxes
50. Quality Control Note        — self-review record
```

---

## 🔢 SEQUENCE NUMBERING LOGIC

Every file is prefixed with a number so you always know **what to study next** and **what it depends on**.

```text
1-supervised-learning/A-regression/01-linear-regression.md
1-supervised-learning/A-regression/02-polynomial-regression.md
...
```

Within the filename:

```text
[category]-[subcategory]/[sequence]-[algorithm-name].md
```

Start with the lowest number in **1-supervised-learning/A-regression** and follow the sequence across families.

---

## 🧪 READ-LEARN-IMPLEMENT CYCLE

Every topic should be processed through:

```text
📖 Learn
   ↓
🧮 Calculate  (work the numerical example by hand)
   ↓
💻 Implement  (from scratch, then via library)
   ↓
🧪 Experiment (change hyperparameters, observe)
   ↓
🔍 Analyze    (error analysis, debugging)
   ↓
⚖️ Compare    (with related algorithms)
   ↓
🧠 Explain    (teach-back without notes)
   ↓
🎯 Apply      (to a real dataset)
```

---

## 💻 ENVIRONMENT SETUP

```powershell
# Activate the venv (Windows PowerShell)
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ;
(& "d:\CODE\complete ml\.venv\Scripts\Activate.ps1")

# Core ML libraries
pip install numpy pandas matplotlib seaborn scikit-learn scipy
pip install xgboost lightgbm catboost
```

---

## 📊 DATA LEAKAGE — CRITICAL RECURRING CONCEPT

> Did information from the test set accidentally enter training?

Always split **BEFORE** any preprocessing that uses data statistics (scaling, imputation, encoding fitted on full data).

---

## ✅ MASTERY DEFINITION

A topic is **Mastered** only when you can:

1. Define it
2. Explain it to a beginner
3. Explain the intuition
4. Explain why it exists
5. Explain the mathematics
6. Derive important formulas
7. Perform a small numerical calculation
8. Implement the core algorithm
9. Use a standard ML library
10. Explain training
11. Explain hyperparameters
12. Evaluate the model
13. Diagnose failures
14. Explain assumptions
15. Explain limitations
16. Compare with alternatives
17. Select it for a real problem
18. Explain when NOT to use it
19. Solve interview/exam questions
20. Apply it to a real dataset

---

> Start with **`1-supervised-learning/A-regression/01-linear-regression.md`**
