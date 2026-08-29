# Phase 05 — Machine Learning

> **Goal:** Master classical machine learning — from-scratch implementations, evaluation, and engineering.

**Prerequisites:** Phase 02, Phase 03, Phase 04
**Mastery target:** Level 5 (core concepts)

---

## Units

> **Learning pattern for this phase:** baseline first, then model, then evaluation, then error analysis, then improvement. Do not start by tuning advanced models.

### Phase Mental Model

Machine learning is not "pick an algorithm." It is a loop:

```text
Problem → Data → Baseline → Model → Metric → Error analysis → Improvement → Validation
```

Every model in this phase should be learned through what problem it solves, what assumptions it makes, how it fails, and how it compares to simpler alternatives.

### Unit 05.1 — ML Fundamentals & Problem Formulation
- What is ML, types of learning
- The ML pipeline
- **Why now:** Data is already clean enough to model after Phase 04.
- **Mental model:** ML learns patterns from examples instead of being manually programmed.
- **Prerequisites:** Data splits, leakage, basic statistics, Python/Pandas.
- **Implementation:** Formulate a problem
- **Practice:** Convert 3 real-world questions into supervised/unsupervised/recommendation-style ML tasks.
- **Outcome:** Frame problems correctly

### Unit 05.2 — Linear Regression (from scratch)
- Hypothesis, loss (MSE), gradient, gradient descent
- **Why it matters:** It is the simplest model for understanding parameters, loss, gradients, and residuals.
- **Mental model:** Find the weighted line/plane that minimizes prediction error.
- **Implementation:** NumPy from scratch → sklearn
- **Experiments:** learning rate, scaling, noise, outliers
- **Evaluation:** MAE, MSE, RMSE, R²
- **Debugging:** If loss explodes, check feature scaling, learning rate, and gradient signs.

### Unit 05.3 — Logistic Regression (from scratch)
- Sigmoid, probability, log loss, gradient
- **Why it matters:** It introduces probabilistic classification and decision thresholds.
- **Mental model:** Linear regression passed through a probability-squashing function.
- **Implementation:** NumPy from scratch → sklearn
- **Evaluation:** accuracy, precision, recall, F1, ROC-AUC
- **Common mistake:** Treating 0.5 as always the correct threshold.

### Unit 05.4 — Model Evaluation
- Confusion matrix, all classification metrics
- Regression metrics
- **Implementation:** Evaluate models properly
- **Outcome:** Choose the right metric
- **Decision guidance:** Use accuracy only when classes and error costs are balanced. Use F1/PR-AUC when positives are rare. Use recall when missing positives is expensive. Use precision when false positives are expensive.

### Unit 05.5 — Decision Trees
- Splitting, impurity (entropy, Gini), information gain
- **Implementation:** Simplified tree → sklearn
- **Outcome:** Understand tree-based models

### Unit 05.6 — Random Forests
- Bagging, random feature selection
- **Implementation:** Build and evaluate
- **Outcome:** Reduce variance

### Unit 05.7 — Gradient Boosting
- Boosting concept, gradient boosting
- XGBoost, LightGBM
- **Implementation:** Train and compare
- **Outcome:** State-of-the-art tabular models
- **Decision guidance:** Prefer Random Forest for a strong stable baseline. Prefer XGBoost/LightGBM when performance matters and you can tune carefully.

### Unit 05.8 — k-Nearest Neighbors
- Distance, k selection
- **Implementation:** Build and evaluate
- **Outcome:** Understand lazy learning

### Unit 05.9 — Naive Bayes
- Bayes' theorem, independence assumption
- **Implementation:** Build and evaluate
- **Outcome:** Fast probabilistic classifier

### Unit 05.10 — Support Vector Machines
- Margin, hyperplane, kernel trick
- **Implementation:** Build and evaluate
- **Outcome:** Understand SVM

### Unit 05.11 — Clustering
- K-Means, Hierarchical, DBSCAN
- **Implementation:** Cluster data
- **Outcome:** Unsupervised learning

### Unit 05.12 — Dimensionality Reduction (PCA)
- Eigenvectors, variance, projection
- **Implementation:** PCA from scratch → sklearn
- **Outcome:** Reduce dimensions

### Unit 05.13 — Feature Engineering
- Feature creation, selection, transformation
- **Implementation:** Engineer features
- **Outcome:** Improve model performance

### Unit 05.14 — Imbalanced Learning
- Problems, resampling, class weights, metrics
- **Implementation:** Handle imbalance
- **Outcome:** Deal with imbalanced data

### Unit 05.15 — Cross-Validation & Hyperparameter Tuning
- k-fold CV, grid/random/Bayesian search
- **Implementation:** Tune models
- **Outcome:** Optimize hyperparameters
- **Common mistake:** Tuning on the test set. The test set is the final exam, not coaching feedback.

### Unit 05.16 — Model Interpretation
- Feature importance, SHAP, partial dependence
- **Implementation:** Interpret models
- **Outcome:** Explain predictions

### Unit 05.17 — Ensembling
- Bagging, boosting, stacking, voting
- **Implementation:** Combine models
- **Outcome:** Improve performance

### Unit 05.18 — ML Synthesis & Review
- Cumulative review
- Comparison notebooks (RF vs XGBoost, etc.)
- Mini project: end-to-end ML
- **Outcome:** Apply ML independently

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Linear Regression vs Random Forest | You need interpretability and a simple baseline | You need nonlinear interactions | Simplicity vs flexibility |
| Random Forest vs XGBoost/LightGBM | You need a stable low-tuning model | You need stronger tabular performance | Robustness vs tuning complexity |
| KNN vs trained model | You need a simple distance-based baseline | You need scalable inference/generalization | No training vs slow prediction |
| PCA vs feature selection | You can accept transformed features | You need original feature meaning | Compression vs interpretability |
| Accuracy vs F1/PR-AUC | Balanced classes | Imbalanced or high-cost positives | Simplicity vs task relevance |

---

## How to Improve an ML Model

```text
Check data quality
    ↓
Check leakage and split strategy
    ↓
Build a dumb baseline
    ↓
Choose the right metric
    ↓
Train a simple model
    ↓
Analyze errors by segment
    ↓
Improve features/preprocessing
    ↓
Tune hyperparameters
    ↓
Try justified alternative models
    ↓
Validate once on the test set
```

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Very high validation score but bad real-world performance | Leakage | Inspect features and split timing | Rebuild pipeline with split before preprocessing |
| Accuracy is high but minority class fails | Wrong metric/class imbalance | Confusion matrix and PR curve | Use recall/F1/PR-AUC, class weights, resampling |
| Linear/logistic model unstable | Scaling or learning-rate issue | Plot loss curve | Standardize features, lower learning rate |
| Tree model overfits | Too deep, too few samples per leaf | Train vs validation score | Limit depth, min samples, cross-validate |
| Tuned model does worse on test | Over-tuned validation | Compare against untouched test once | Use nested CV or holdout discipline |

---

## Mini Project — End-to-End Classical ML System

**Objective:** Build a complete predictive pipeline on a real tabular dataset.

**Requirements:** define the problem and target; perform EDA and cleaning; create a baseline; train at least 3 model families; choose metrics based on problem cost; tune one model; perform error analysis; interpret the final model; document limitations.

**Expected output:** notebook/script, saved model artifact, evaluation report, and README.

**Evaluation criteria:** correct split, no leakage, justified metric, baseline comparison, reproducible results, clear explanation of trade-offs.

**Advanced extensions:** model card, data card, simple API endpoint, experiment tracking log.

---

## Interview Questions

- Why is leakage dangerous, and how do you detect it?
- When is accuracy misleading?
- Explain bias vs variance using a model example.
- When would you choose a linear model over XGBoost?
- How would you improve a model that performs badly on one user segment?
- What is the difference between validation and test data?

---

## Phase Review

- [ ] All units complete
- [ ] Comparison experiments done
- [ ] Mini project done with README, evaluation report, and error analysis
- [ ] At least one failure case intentionally reproduced and fixed
- [ ] At least three model families compared against a baseline
- [ ] Cumulative review passed

## Mastery Check

At the end of this phase, you should be able to:
1. Implement linear and logistic regression from scratch.
2. Choose the right model for a problem.
3. Evaluate models with the right metrics.
4. Handle imbalanced data and leakage.
5. Tune hyperparameters properly.
6. Interpret model predictions.
7. Build an end-to-end ML pipeline.
