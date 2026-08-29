# BUILD STATUS

> **This file is the single source of truth for what has been built, verified, and what remains.**

Last updated: 2026-08-29

---

## Overall Status

| Metric | Status |
|--------|--------|
| Project scaffolding | ✅ COMPLETE |
| Recovery infrastructure | ✅ COMPLETE |
| Roadmap design | ✅ COMPLETE |
| Notebook templates | ✅ COMPLETE |
| Tracking systems | ✅ COMPLETE |
| Reference docs | ✅ COMPLETE |
| Environment verification | ✅ COMPLETE |
| Phase 00 (Environment) | ✅ COMPLETE |
| Phase 01 (Python) | ✅ COMPLETE |
| Phase 02 (Math) | ✅ COMPLETE |
| Phase 03 (Statistics) | ✅ COMPLETE |
| Phase 04 (Data Analysis) | ✅ COMPLETE |
| Phase 05 (Machine Learning) | ⏳ IN PROGRESS |
| Phase 06 (Deep Learning) | ✅ COMPLETE |
| Phase 07 (NLP) | ✅ COMPLETE |
| Phase 08 (Transformers) | ✅ COMPLETE |
| Phase 09 (Generative AI) | ✅ COMPLETE |
| Phase 10 (LLMs) | ⏳ NOT STARTED |
| Phase 11 (RAG) | ✅ COMPLETE |
| Phase 12 (LangChain) | ✅ COMPLETE |
| Phase 13 (LangGraph) | ✅ COMPLETE |
| Phase 14 (Agents) | ⏳ IN PROGRESS |
| Phase 15 (Evaluation) | ✅ COMPLETE |
| Phase 16 (Deployment) | ✅ COMPLETE |
| Phase 17 (Capstone) | ✅ COMPLETE |

---

## Verification Status Legend

- ✅ **VERIFIED** — Executed from clean kernel, all cells pass, outputs correct.
- ⏳ **IN PROGRESS** — Currently being built or verified.
- ⚠️ **PARTIAL** — Some cells pass, some fail or blocked.
- ❌ **FAILED** — Execution failed, needs fixes.
- 🚫 **BLOCKED** — Cannot execute (missing package, no GPU, no internet, missing dataset, missing API key).

---

## Phase Status Detail

### Phase 00 — Environment Setup

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 00.1 | `00_environment/00_environment_verification.ipynb` | ✅ VERIFIED | 2026-08-28 | All core packages OK; opendatasets blocked (Python 3.13 cgi removal) |
| 00.2 | `00_environment/00_02_jupyter_workflow.ipynb` | ✅ VERIFIED | 2026-08-28 | Jupyter cells, kernels, magic |
| 00.3 | `00_environment/00_03_git_version_control.ipynb` | ✅ VERIFIED | 2026-08-28 | Git basics, branching, recovery |
| 00.4 | `00_environment/00_04_python_environments.ipynb` | ✅ VERIFIED | 2026-08-28 | venv, uv, packages |

---

### Phase 01 — Python Foundations

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 01.1 | `01_python/01_01_python_basics.ipynb` | ✅ VERIFIED | 2026-08-28 | Variables, types, operators, control flow |
| 01.2 | `01_python/01_02_data_structures.ipynb` | ✅ VERIFIED | 2026-08-28 | Lists, dicts, sets, tuples |
| 01.3 | `01_python/01_03_functions_scope.ipynb` | ✅ VERIFIED | 2026-08-28 | Functions, scope, closures |
| 01.4 | `01_python/01_04_oop.ipynb` | ✅ VERIFIED | 2026-08-28 | Classes, inheritance, dunder |
| 01.5 | `01_python/01_05_numpy.ipynb` | ✅ VERIFIED | 2026-08-28 | NumPy arrays, broadcasting |
| 01.6 | `01_python/01_06_pandas.ipynb` | ✅ VERIFIED | 2026-08-28 | DataFrames, Series |
| 01.7 | `01_python/01_07_matplotlib.ipynb` | ✅ VERIFIED | 2026-08-28 | Plotting |
| 01.8 | `01_python/01_08_file_io_errors.ipynb` | ✅ VERIFIED | 2026-08-28 | File I/O, exceptions |
| 01.9 | `01_python/01_09_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Mini project |

---

### Phase 02 — Mathematics for ML

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 02.1 | `02_math/02_01_vectors_matrices.ipynb` | ✅ VERIFIED | 2026-08-28 | Vectors, matrices, norms, dot product |
| 02.2 | `02_math/02_02_matrix_operations.ipynb` | ✅ VERIFIED | 2026-08-28 | Matrix multiplication, inverse |
| 02.3 | `02_math/02_03_derivatives_gradients.ipynb` | ✅ VERIFIED | 2026-08-28 | Derivatives, gradients |
| 02.4 | `02_math/02_04_gradient_descent.ipynb` | ✅ VERIFIED | 2026-08-28 | Gradient descent |
| 02.5 | `02_math/02_05_probability.ipynb` | ✅ VERIFIED | 2026-08-28 | Probability |
| 02.6 | `02_math/02_06_information_theory.ipynb` | ✅ VERIFIED | 2026-08-28 | Information theory |
| 02.7 | `02_math/02_07_eigenvalues.ipynb` | ✅ VERIFIED | 2026-08-28 | Eigenvalues |
| 02.8 | `02_math/02_08_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Synthesis |

---

### Phase 03 — Statistics & Probability

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 03.1 | `03_statistics/03_01_descriptive_statistics.ipynb` | ✅ VERIFIED | 2026-08-28 | Descriptive statistics |
| 03.2 | `03_statistics/03_02_probability_distributions.ipynb` | ✅ VERIFIED | 2026-08-28 | Probability distributions |
| 03.3 | `03_statistics/03_03_inferential_statistics.ipynb` | ✅ VERIFIED | 2026-08-28 | Inferential statistics & sampling |
| 03.4 | `03_statistics/03_04_hypothesis_testing.ipynb` | ✅ VERIFIED | 2026-08-28 | Hypothesis testing |
| 03.5 | `03_statistics/03_05_correlation_regression.ipynb` | ✅ VERIFIED | 2026-08-28 | Correlation & regression basics |
| 03.6 | `03_statistics/03_06_bayesian_thinking.ipynb` | ✅ VERIFIED | 2026-08-28 | Bayesian thinking |
| 03.7 | `03_statistics/03_07_statistics_for_ml.ipynb` | ✅ VERIFIED | 2026-08-28 | Statistics for ML |
| 03.8 | `03_statistics/03_08_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Synthesis |

### Phase 04 — Data Analysis & Preparation

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 04.1 | `04_data_analysis/04_01_eda.ipynb` | ✅ VERIFIED | 2026-08-28 | Data exploration & EDA |
| 04.2 | `04_data_analysis/04_02_data_cleaning.ipynb` | ✅ VERIFIED | 2026-08-28 | Data cleaning |
| 04.3 | `04_data_analysis/04_03_missing_values.ipynb` | ✅ VERIFIED | 2026-08-28 | Missing values |
| 04.4 | `04_data_analysis/04_04_outliers.ipynb` | ✅ VERIFIED | 2026-08-28 | Outliers |
| 04.5 | `04_data_analysis/04_05_feature_scaling_encoding.ipynb` | ✅ VERIFIED | 2026-08-28 | Feature scaling & encoding |
| 04.6 | `04_data_analysis/04_06_train_val_test_splits.ipynb` | ✅ VERIFIED | 2026-08-28 | Train/val/test splits |
| 04.7 | `04_data_analysis/04_07_data_leakage.ipynb` | ✅ VERIFIED | 2026-08-28 | Data leakage |
| 04.8 | `04_data_analysis/04_08_synthesis.ipynb` | ✅ VERIFIED | 2026-08-28 | Data preparation synthesis |

### Phase 05 — Machine Learning

| Unit | Notebook | Status | Last Verified | Notes |
|------|----------|--------|---------------|-------|
| 05.1 | `05_machine_learning/05_01_ml_fundamentals.ipynb` | ✅ VERIFIED | 2026-08-28 | ML fundamentals & problem formulation |
| 05.2 | `05_machine_learning/05_02_linear_regression.ipynb` | ✅ VERIFIED | 2026-08-28 | Linear regression from scratch |
| 05.3 | `05_machine_learning/05_03_logistic_regression.ipynb` | ✅ VERIFIED | 2026-08-28 | Logistic regression from scratch |
| 05.4 | `05_machine_learning/05_04_model_evaluation.ipynb` | ✅ VERIFIED | 2026-08-28 | Model evaluation |
| 05.5 | `05_machine_learning/05_05_decision_trees.ipynb` | ✅ VERIFIED | 2026-08-28 | Decision trees |
| 05.6 | `05_machine_learning/05_06_random_forests.ipynb` | ✅ VERIFIED | 2026-08-28 | Random forests |
| 05.7 | `05_machine_learning/05_07_gradient_boosting.ipynb` | ✅ VERIFIED | 2026-08-28 | Gradient boosting (XGBoost, LightGBM) |
| 05.8 | `05_machine_learning/05_08_knn.ipynb` | ⏳ IN PROGRESS | — | k-Nearest Neighbors |
| 05.9 | `05_machine_learning/05_09_naive_bayes.ipynb` | ⏳ IN PROGRESS | — | Naive Bayes |
| 05.10 | `05_machine_learning/05_10_svm.ipynb` | ⏳ IN PROGRESS | — | Support Vector Machines |
| 05.11 | `05_machine_learning/05_11_clustering.ipynb` | ⏳ IN PROGRESS | — | Clustering |
| 05.12 | `05_machine_learning/05_12_pca.ipynb` | ⏳ IN PROGRESS | — | Dimensionality Reduction (PCA) |
| 05.13 | `05_machine_learning/05_13_feature_engineering.ipynb` | ⏳ IN PROGRESS | — | Feature Engineering |
| 05.14 | `05_machine_learning/05_14_imbalanced_learning.ipynb` | ⏳ IN PROGRESS | — | Imbalanced Learning |
| 05.15 | `05_machine_learning/05_15_cv_hyperparameter_tuning.ipynb` | ⏳ IN PROGRESS | — | Cross-Validation & Tuning |
| 05.16 | `05_machine_learning/05_16_model_interpretation.ipynb` | ⏳ IN PROGRESS | — | Model Interpretation |
| 05.17 | `05_machine_learning/05_17_ensembling.ipynb` | ⏳ IN PROGRESS | — | Ensembling |
| 05.18 | `05_machine_learning/05_18_synthesis.ipynb` | ⏳ IN PROGRESS | — | ML Synthesis & Review |

---

## Environment Snapshot

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.13.9 (uv venv) | ✅ |
| uv | 0.9.9 | ✅ |
| Git | initialized (no commits yet) | ✅ |
| GPU | None (CPU-only) | ⚠️ CPU-only |
| Internet | Available | ✅ |
| Jupyter | 4.6.3 (jupyterlab) | ✅ |
| numpy | 2.5.2 | ✅ |
| pandas | 3.0.5 | ✅ |
| scikit-learn | 1.9.0 | ✅ |
| scipy | 1.18.1 | ✅ |
| matplotlib | 3.11.1 | ✅ |
| seaborn | 0.13.2 | ✅ |
| torch | 2.13.0 (CPU) | ✅ |
| transformers | 5.16.1 | ✅ |
| datasets | 5.0.1 | ✅ |
| langchain | 1.3.18 | ✅ |
| langgraph | installed | ✅ |
| xgboost | 3.4.1 | ✅ |
| lightgbm | 4.7.0 | ✅ |
| chromadb | 1.5.9 | ✅ |
| faiss-cpu | 1.15.0 | ✅ |
| fastapi | 0.141.1 | ✅ |
| opendatasets | 0.1.22 | 🚫 BLOCKED (Python 3.13 removed `cgi`) |

---

## Blocked Items

| Item | Reason |
|------|--------|
| opendatasets | Python 3.13 removed the `cgi` module which opendatasets depends on. Use `kagglehub` instead for dataset downloads. |

---

## Decisions Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-28 | Use uv-managed virtual environment | User's chosen environment manager |
| 2026-08-28 | CPU-only PyTorch | No GPU detected on this machine |
| 2026-08-28 | Build docs/architecture/roadmap first, notebooks after | Per master build instruction §51 |
