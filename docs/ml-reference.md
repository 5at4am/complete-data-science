# Machine Learning Reference

> **A quick-reference guide to classical machine learning.**

---

## ML Fundamentals

### What is ML?
A program that improves its performance on a task (T) with experience (E),
measured by a performance metric (P).

### Types of Learning
- **Supervised**: labeled data (X → y)
- **Unsupervised**: unlabeled data (find structure)
- **Semi-supervised**: mix of labeled and unlabeled
- **Reinforcement**: learn from rewards/penalties

### The ML Pipeline
1. Problem formulation
2. Data collection
3. Data cleaning
4. Feature engineering
5. Train/validation/test split
6. Model selection
7. Training
8. Evaluation
9. Hyperparameter tuning
10. Deployment
11. Monitoring

---

## Data Splitting

### Train/Validation/Test
- **Train**: fit the model
- **Validation**: tune hyperparameters, select model
- **Test**: final, unbiased evaluation (used ONCE)

### Cross-Validation
- k-fold: split into k folds, train on k-1, validate on 1, rotate
- More reliable than single split

### Data Leakage (CRITICAL)
- When information from the test set leaks into training
- Examples: scaling before splitting, target leakage, duplicate rows
- **Always split BEFORE any preprocessing that uses data statistics**

---

## Regression

### Linear Regression
- `y = w·x + b`
- Minimize MSE: `(1/n) Σ (yᵢ - ŷᵢ)²`
- Closed-form: `w = (XᵀX)⁻¹Xᵀy`
- Or gradient descent

### Regularized Regression
- **Ridge (L2)**: adds `λ||w||²` penalty
- **Lasso (L1)**: adds `λ||w||₁` penalty (sparse)
- **Elastic Net**: combination of both

### Metrics
- MAE, MSE, RMSE, R²

---

## Classification

### Logistic Regression
- `P(y=1|x) = σ(w·x + b)` where σ is sigmoid
- Minimize log loss (cross-entropy)

### Decision Trees
- Recursively split data by feature thresholds
- Use impurity measures: Gini, entropy
- Interpretable, but prone to overfitting

### Random Forest
- Bagging + random feature selection
- Many trees, average predictions
- Reduces variance vs single tree

### Gradient Boosting
- Sequentially add trees that correct previous errors
- XGBoost, LightGBM, CatBoost are optimized implementations

### Support Vector Machines (SVM)
- Find hyperplane that maximizes margin
- Kernel trick for non-linear boundaries

### k-Nearest Neighbors (kNN)
- Predict based on k nearest training points
- Lazy learner, no training phase

### Naive Bayes
- Bayes' theorem with independence assumption
- Fast, works well for text

### Metrics
- Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix

---

## Unsupervised Learning

### Clustering
- **K-Means**: partition into k clusters by centroid distance
- **Hierarchical**: build a tree of clusters
- **DBSCAN**: density-based, handles arbitrary shapes and noise

### Dimensionality Reduction
- **PCA**: project onto directions of maximum variance
- **t-SNE**: non-linear, for visualization
- **UMAP**: non-linear, faster than t-SNE

---

## Model Evaluation

### Regression Metrics
- **MAE**: `(1/n) Σ |yᵢ - ŷᵢ|`
- **MSE**: `(1/n) Σ (yᵢ - ŷᵢ)²`
- **RMSE**: `√MSE`
- **R²**: proportion of variance explained

### Classification Metrics
- **Accuracy**: correct / total
- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: harmonic mean of precision and recall
- **ROC-AUC**: area under ROC curve
- **PR-AUC**: area under precision-recall curve

### When to Use Which Metric
- **Imbalanced classes**: use precision/recall/F1, NOT accuracy
- **False positives costly**: optimize precision
- **False negatives costly**: optimize recall
- **Ranking**: use ROC-AUC

---

## Feature Engineering

### Feature Scaling
- **Standardization**: `(x - μ) / σ` — for models assuming normal distribution
- **Normalization**: `(x - min) / (max - min)` — for bounded features

### Encoding Categorical
- **One-hot**: binary columns per category
- **Label encoding**: integer per category (ordinal)
- **Target encoding**: mean target per category

### Feature Selection
- Filter (correlation, mutual information)
- Wrapper (recursive feature elimination)
- Embedded (L1 regularization, tree importance)

---

## Imbalanced Learning

### Problems
- Accuracy is misleading
- Model ignores minority class

### Solutions
- Resampling: oversample minority, undersample majority, SMOTE
- Class weights
- Different metrics (precision/recall/F1)
- Ensemble methods

---

## Hyperparameter Tuning

### Grid Search
- Exhaustive search over parameter grid

### Random Search
- Random sampling of parameters

### Bayesian Optimization
- Model-based, more efficient

---

## Model Selection Guide

| Model | When to Use | When NOT to Use |
|-------|-------------|-----------------|
| Linear Regression | Linear relationships, interpretability | Non-linear, high-dim |
| Logistic Regression | Binary classification, baseline | Complex non-linear boundaries |
| Decision Tree | Interpretability | High variance, complex data |
| Random Forest | Tabular data, robust baseline | Very high-dim, text |
| Gradient Boosting | Tabular data, state-of-the-art | Very large data, interpretability |
| SVM | Small-medium data, clear margin | Large data, high-dim |
| kNN | Simple, small data | Large data, high-dim |
| Neural Networks | Complex patterns, images, text | Small data, interpretability |
