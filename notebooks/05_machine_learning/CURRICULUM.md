# Machine Learning — Complete Curriculum

**Phase 05 | 33 Notebooks | ~900 Code Cells | Beginner → Production**

---

## Learning Philosophy

This curriculum is built on three principles:

1. **Implementation-first**: Every algorithm is built from scratch in NumPy before using sklearn
2. **Real datasets**: After each concept is introduced with synthetic data, we apply it to real-world datasets (Iris, Titanic, Housing, Wine, Breast Cancer, Diabetes, MNIST)
3. **Active learning**: Every notebook includes coding exercises, not just demonstrations

---

## Prerequisites

- **Phase 00**: Environment setup (Python, Jupyter, Git)
- **Phase 01**: Python fundamentals (data structures, functions, OOP, pandas, NumPy, matplotlib)
- **Phase 02**: Mathematics for ML (linear algebra, calculus, probability)
- **Phase 03**: Statistics & probability
- **Phase 04**: Data analysis & preparation (EDA, cleaning, missing values, encoding)

---

## Curriculum Map

### PART A: Foundations (Notebooks 01–08)
> Build intuition. Learn the math. Implement from scratch.

| # | Title | From Scratch | Real Data | Exercises | Cells |
|---|-------|:---:|:---:|:---:|:---:|
| 01 | ML Fundamentals & Problem Formulation | — | ✓ | ✓ | 30 |
| 02 | Linear Regression from Scratch | ✓ | ✓ | ✓ | 35 |
| 03 | Logistic Regression from Scratch | ✓ | ✓ | ✓ | 32 |
| 04 | Model Evaluation & Metrics | — | ✓ | ✓ | 30 |
| 05 | Decision Trees from Scratch | ✓ | ✓ | ✓ | 35 |
| 06 | Random Forests & Bagging | ✓ | ✓ | ✓ | 32 |
| 07 | Gradient Boosting from Scratch | ✓ | ✓ | ✓ | 35 |
| 08 | K-Nearest Neighbors from Scratch | ✓ | ✓ | ✓ | 28 |

### PART B: Core Algorithms (Notebooks 09–15)
> Master the remaining core algorithms. Each with from-scratch + sklearn.

| # | Title | From Scratch | Real Data | Exercises | Cells |
|---|-------|:---:|:---:|:---:|:---:|
| 09 | Naive Bayes from Scratch | ✓ | ✓ | ✓ | 28 |
| 10 | Support Vector Machines | ✓ | ✓ | ✓ | 30 |
| 11 | Clustering: K-Means, Hierarchical, DBSCAN | ✓ | ✓ | ✓ | 32 |
| 12 | PCA & Dimensionality Reduction | ✓ | ✓ | ✓ | 30 |
| 13 | Feature Engineering & Selection | — | ✓ | ✓ | 32 |
| 14 | Handling Imbalanced Data | — | ✓ | ✓ | 30 |
| 15 | Cross-Validation & Hyperparameter Tuning | — | ✓ | ✓ | 30 |

### PART C: Production ML (Notebooks 16–22)
> Build real pipelines. Handle real data. Ship to production.

| # | Title | From Scratch | Real Data | Exercises | Cells |
|---|-------|:---:|:---:|:---:|:---:|
| 16 | ML Pipelines & ColumnTransformer | — | ✓ | ✓ | 32 |
| 17 | Model Interpretation & Explainability | — | ✓ | ✓ | 30 |
| 18 | Ensembling: Voting, Stacking, Blending | ✓ | ✓ | ✓ | 32 |
| 19 | Multi-Class Classification | — | ✓ | ✓ | 30 |
| 20 | Regression with Trees & Boosting | — | ✓ | ✓ | 30 |
| 21 | Time-Series Splitting & Validation | — | ✓ | ✓ | 30 |
| 22 | Model Saving, Loading & Deployment | — | ✓ | ✓ | 28 |

### PART D: Advanced Topics (Notebooks 23–30)
> Go deeper. Bayesian optimization, neural networks, advanced interpretation.

| # | Title | From Scratch | Real Data | Exercises | Cells |
|---|-------|:---:|:---:|:---:|:---:|
| 23 | Bayesian Optimization with Optuna | — | ✓ | ✓ | 28 |
| 24 | Neural Networks from Scratch | ✓ | ✓ | ✓ | 40 |
| 25 | MLPClassifier & sklearn Neural Nets | — | ✓ | ✓ | 30 |
| 26 | SHAP & Advanced Interpretation | — | ✓ | ✓ | 30 |
| 27 | Advanced Ensembling & Stacking | ✓ | ✓ | ✓ | 32 |
| 28 | Data Leakage & Anti-Patterns | — | ✓ | ✓ | 28 |
| 29 | Learning Curves & Diagnostic Plots | — | ✓ | ✓ | 28 |
| 30 | Model Selection & Comparison Framework | — | ✓ | ✓ | 30 |

### PART E: Capstone & Synthesis (Notebooks 31–33)
> Apply everything. End-to-end projects.

| # | Title | From Scratch | Real Data | Exercises | Cells |
|---|-------|:---:|:---:|:---:|:---:|
| 31 | End-to-End ML Project: Classification | — | ✓ | ✓ | 40 |
| 32 | End-to-End ML Project: Regression | — | ✓ | ✓ | 40 |
| 33 | ML Synthesis & Concept Map | — | ✓ | ✓ | 30 |

---

## Datasets Used

| Dataset | Source | Notebooks | Task |
|---------|--------|-----------|------|
| Iris | sklearn | 01, 08, 11, 19 | Classification (multi-class) |
| Titanic | seaborn | 01, 16, 22, 31 | Classification (binary) |
| Breast Cancer | sklearn | 02, 03, 04, 10, 14, 17, 26, 30 | Classification (binary) |
| Diabetes | sklearn | 02, 20, 29 | Regression |
| Wine | sklearn | 05, 06, 09, 19 | Classification (multi-class) |
| Housing (California) | sklearn | 07, 13, 23, 32 | Regression |
| MNIST (digits) | sklearn | 07, 24, 25, 27 | Classification (multi-class) |
| Synthetic datasets | Generated | All | Concept illustration |

---

## Learning Path (Beginner Guide)

### Week 1: Foundations
- **Day 1**: 05_01 (ML Fundamentals) — understand the landscape
- **Day 2-3**: 05_02 (Linear Regression) — implement from scratch, understand gradient descent
- **Day 4**: 05_03 (Logistic Regression) — classification from scratch
- **Day 5**: 05_04 (Model Evaluation) — metrics matter more than models

### Week 2: Tree-Based Methods
- **Day 1-2**: 05_05 (Decision Trees) — understand splits, entropy, Gini
- **Day 3**: 05_06 (Random Forests) — bagging reduces variance
- **Day 4-5**: 05_07 (Gradient Boosting) — boosting reduces bias

### Week 3: Classic Algorithms
- **Day 1**: 05_08 (KNN) — instance-based learning
- **Day 2**: 05_09 (Naive Bayes) — probabilistic classification
- **Day 3-4**: 05_10 (SVM) — maximum margin classifier
- **Day 5**: 05_11 (Clustering) — unsupervised discovery

### Week 4: Feature Engineering & Tuning
- **Day 1**: 05_12 (PCA) — reduce dimensions
- **Day 2-3**: 05_13 (Feature Engineering) — transform features
- **Day 4**: 05_14 (Imbalanced Data) — handle class imbalance
- **Day 5**: 05_15 (CV & Tuning) — find the best model

### Week 5: Production ML
- **Day 1-2**: 05_16 (Pipelines) — production-ready workflows
- **Day 3**: 05_17 (Interpretation) — explain predictions
- **Day 4**: 05_18 (Ensembling) — combine models
- **Day 5**: 05_19 (Multi-Class) — beyond binary

### Week 6: Advanced Topics
- **Day 1-2**: 05_20-21 (Regression, Time-Series) — specialized domains
- **Day 3**: 05_22 (Model Saving) — deployment basics
- **Day 4-5**: 05_23-25 (Optuna, Neural Nets, MLP) — optimization & deep learning bridge

### Week 7: Advanced Interpretation & Debugging
- **Day 1-2**: 05_26-27 (SHAP, Advanced Ensembling) — interpret & combine
- **Day 3**: 05_28 (Data Leakage) — avoid the #1 mistake
- **Day 4**: 05_29 (Learning Curves) — diagnose bias/variance
- **Day 5**: 05_30 (Model Comparison) — systematic evaluation

### Week 8: Capstone Projects
- **Day 1-3**: 05_31 (Classification Project) — end-to-end Titanic pipeline
- **Day 4-5**: 05_32 (Regression Project) — end-to-end Housing pipeline
- **Day 5**: 05_33 (Synthesis) — connect everything

---

## What Each Notebook Contains

Every notebook follows this structure:

1. **What Are We Solving?** — the problem in plain English
2. **Why Does This Matter?** — real-world relevance
3. **Prerequisites** — what to review first
4. **Learning Objectives** — exact skills you'll gain
5. **Mental Model** — intuitive understanding before math
6. **Mathematical Foundation** — equations, notation, derivation
7. **From-Scratch Implementation** — NumPy code, line by line
8. **sklearn Implementation** — the production version
9. **Real-World Application** — apply to a real dataset
10. **Hyperparameter Deep Dive** — what each parameter does
11. **Comparison & Benchmarking** — vs other algorithms
12. **Common Mistakes** — what beginners get wrong
13. **Coding Exercises** — hands-on practice (fill-in-the-blank + open-ended)
14. **Closed-Book Recall** — 4 questions, no peeking
15. **Teach-Back Questions** — explain to another person
16. **Summary** — key takeaways
17. **Further Experiment** — 3 ideas for self-study
18. **Verification Status** — execution proof

---

## Total Count

| Metric | Count |
|--------|-------|
| Total notebooks | 33 |
| Total code cells | ~900 |
| From-scratch implementations | 12 |
| Real datasets used | 7 |
| Coding exercises | 66+ (2 per notebook) |
| Estimated study time | 8 weeks (part-time) |
