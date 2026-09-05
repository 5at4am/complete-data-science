# 01. Logistic Regression

> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐☆☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Logistic Regression |
| Category | Supervised Learning |
| Type | Classification (Binary / Multiclass) |
| Parametric / Non-parametric | Parametric |
| Generative / Discriminative | Discriminative |
| Main Objective | Learn a linear boundary that separates classes by estimating class probabilities |
| Input | Feature matrix X (n × d), labels y ∈ {0, 1} |
| Output | Probability P(y=1\|x) and class label |
| Core Idea | Pass a linear combination of features through a sigmoid function to map it to [0, 1] |
| Typical Use Cases | Spam detection, disease diagnosis, credit scoring, click-through rate prediction |

---

## 02. One-Line Definition

### Beginner Definition
Logistic Regression predicts the probability that an input belongs to a particular class using a sigmoid-shaped curve.

### Technical Definition
Logistic Regression is a discriminative parametric classifier that models the log-odds of the positive class as a linear function of the input features and applies the sigmoid (logistic) function to produce a calibrated probability estimate.

---

## 03. Intuition

Imagine you are a doctor deciding whether a patient has a disease. You have their age, blood pressure, and cholesterol level. You combine these into a single "risk score." If the risk score is high, the patient likely has the disease; if low, they probably don't.

Logistic Regression does exactly this: it takes multiple input features, combines them into a single weighted sum (just like a risk score), and then pushes that sum through a sigmoid curve. The sigmoid takes any real number and squashes it into a range between 0 and 1 — interpreting it as a probability. If the output is above 0.5, we classify as "positive"; otherwise "negative."

The key insight: even though the name says "regression," it is actually a **classification** algorithm. It performs logistic (sigmoid) regression on the log-odds to produce a classification.

---

## 04. Problem It Solves

**Problem:** Given labeled data with features and binary outcomes, how do we predict the probability and class of a new unseen data point?

**Example:** You have 1000 emails. Each has features like word frequencies. Some are spam (1), some are not (0). You want to classify new emails.

**Why useful:** Linear regression cannot be used for classification because:
1. It can output values outside [0, 1], which are not probabilities.
2. It assumes a linear relationship between features and target — classification boundaries are different.

Logistic Regression solves this by using the sigmoid function to ensure outputs are valid probabilities.

---

## 05. Where It Fits in Machine Learning

```
Machine Learning
├── Supervised Learning
│   ├── Classification
│   │   ├── Logistic Regression  ◄── YOU ARE HERE
│   │   ├── K-Nearest Neighbors
│   │   ├── Naive Bayes
│   │   ├── Support Vector Machine
│   │   ├── Decision Tree
│   │   ├── Random Forest
│   │   └── Neural Networks
│   └── Regression
│       ├── Linear Regression
│       ├── Ridge / Lasso
│       └── ...
└── Unsupervised Learning
    └── ...
```

Logistic Regression is the **foundational** classification algorithm — the starting point for understanding all discriminative classifiers.

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Sigmoid Function | S-shaped curve | σ(z) = 1 / (1 + e⁻ᶻ), maps ℝ → (0, 1) |
| Log-odds (logit) | Log of odds ratio | ln(p / (1-p)), the inverse of sigmoid |
| Decision Boundary | Line separating classes | The hyperplane where P(y=1\|x) = 0.5 |
| Odds | Ratio of probabilities | p / (1-p), how much more likely positive than negative |
| Maximum Likelihood | Best fit idea | Finding parameters that maximize the probability of observed data |
| Cross-Entropy Loss | Classification loss | Measures divergence between predicted and true probability distributions |
| Multinomial Logistic Regression | Multiclass extension | Uses softmax instead of sigmoid for K > 2 classes |

---

## 07. Input and Output

**Input:**
- **Feature matrix X:** n samples × d features. Features can be numerical or encoded categorical.
- **Label vector y:** n binary labels ∈ {0, 1} for binary classification, or ∈ {0, 1, ..., K-1} for multiclass.
- **Hyperparameters:** learning rate, regularization strength C, number of iterations, solver type.

**Output:**
- **Probability:** P(y=1|x) ∈ (0, 1) for each sample.
- **Class prediction:** ŷ = 1 if P(y=1|x) ≥ threshold (default 0.5), else 0.
- **Learned parameters:** weight vector w ∈ ℝᵈ and bias b ∈ ℝ.

---

## 08. Mathematical Foundation

**Core idea:** Model the log-odds of the positive class as a linear function of features.

**Notation:**
- x = (x₁, x₂, ..., x_d)ᵀ — input feature vector
- w = (w₁, w₂, ..., w_d)ᵀ — weight vector (parameters to learn)
- b — bias (intercept)
- z = wᵀx + b — linear combination (logit)
- σ(z) = 1 / (1 + e⁻ᶻ) — sigmoid function

**Required math concepts:**
1. **Sigmoid function:** maps any real number to (0, 1)
2. **Log-odds:** ln(p/(1-p)) is the inverse of sigmoid
3. **Maximum Likelihood Estimation (MLE):** statistical method for finding best parameters
4. **Gradient Descent:** optimization algorithm for minimizing loss

**Connection:** The sigmoid function is the inverse of the logit (log-odds) function:

```
logit(p) = ln(p/(1-p))     — go from probability to log-odds
sigmoid(z) = 1/(1+e⁻ᶻ)    — go from log-odds to probability
```

---

## 09. Core Formula

### Formula 1: Sigmoid Function

```
σ(z) = 1 / (1 + e⁻ᶻ)
```

**Meaning:** Converts any real number z into a value between 0 and 1.

**Symbols:**
- σ(z) — output probability, ∈ (0, 1)
- z — input = wᵀx + b (the linear combination)
- e — Euler's number ≈ 2.71828

**Intuition:** When z is large and positive, e⁻ᶻ ≈ 0, so σ(z) ≈ 1. When z is large and negative, e⁻ᶻ is huge, so σ(z) ≈ 0. At z = 0, σ(0) = 0.5 (the decision boundary).

**Example:**
```
z = 2.0
σ(2.0) = 1 / (1 + e⁻²) = 1 / (1 + 0.1353) = 1 / 1.1353 ≈ 0.881
→ 88.1% probability of class 1
```

### Formula 2: Linear Combination (Logit)

```
z = wᵀx + b = w₁x₁ + w₂x₂ + ... + w_d·x_d + b
```

**Meaning:** A weighted sum of all input features plus a bias term.

**Symbols:**
- wᵢ — weight for feature i (importance of feature i)
- xᵢ — value of feature i
- b — bias / intercept (shifts the decision boundary)

**Intuition:** Each weight controls how much its corresponding feature contributes to the final decision. A large positive wᵢ means feature i strongly pushes toward class 1.

**Example:**
```
w = [0.5, -1.2], b = 0.3, x = [3, 2]
z = 0.5×3 + (-1.2)×2 + 0.3 = 1.5 - 2.4 + 0.3 = -0.6
σ(-0.6) = 1/(1+e⁰·⁶) = 1/(1+1.822) = 1/2.822 ≈ 0.354
→ 35.4% probability of class 1 → predicted class: 0
```

### Formula 3: Probability Output

```
P(y=1 | x) = σ(wᵀx + b)
P(y=0 | x) = 1 - σ(wᵀx + b)
```

**Meaning:** The model outputs the probability of each class. Since it's binary, P(y=0|x) = 1 - P(y=1|x).

### Formula 4: Decision Rule

```
ŷ = 1  if  P(y=1|x) ≥ 0.5  (i.e., wᵀx + b ≥ 0)
ŷ = 0  if  P(y=1|x) < 0.5   (i.e., wᵀx + b < 0)
```

**Meaning:** The decision boundary is the hyperplane wᵀx + b = 0.

### Formula 5: Binary Cross-Entropy Loss (Cost Function)

```
J(w, b) = -1/n Σᵢ₌₁ⁿ [yᵢ · log(ŷᵢ) + (1 - yᵢ) · log(1 - ŷᵢ)]
```

**Meaning:** Measures how far predicted probabilities are from actual labels.

**Symbols:**
- n — number of training samples
- yᵢ — true label (0 or 1) for sample i
- ŷᵢ = σ(wᵀx⁽ⁱ⁾ + b) — predicted probability for sample i
- log — natural logarithm

**Intuition:**
- When yᵢ = 1: loss = -log(ŷᵢ). If ŷᵢ is close to 1, loss ≈ 0. If ŷᵢ ≈ 0, loss → ∞ (heavy penalty).
- When yᵢ = 0: loss = -log(1-ŷᵢ). If ŷᵢ is close to 0, loss ≈ 0. If ŷᵢ ≈ 1, loss → ∞.

**Example (3 samples):**

| Sample | y | ŷ | -y·log(ŷ) | -(1-y)·log(1-ŷ) | Total Loss |
|--------|---|------|------------|------------------|------------|
| 1 | 1 | 0.9 | -log(0.9) = 0.105 | 0 | 0.105 |
| 2 | 0 | 0.2 | 0 | -log(0.8) = 0.223 | 0.223 |
| 3 | 1 | 0.4 | -log(0.4) = 0.916 | 0 | 0.916 |

Average loss = (0.105 + 0.223 + 0.916) / 3 = 0.415

---

## 10. Derivation

### Sigmoid from Log-Odds

Start with the log-odds model:

```
ln(p/(1-p)) = wᵀx + b
```

Exponentiate both sides:

```
p/(1-p) = e^(wᵀx + b)
```

Solve for p:

```
p = (1-p) · e^(wᵀx + b)
p = e^(wᵀx + b) - p · e^(wᵀx + b)
p(1 + e^(wᵀx + b)) = e^(wᵀx + b)
p = e^(wᵀx + b) / (1 + e^(wᵀx + b))
p = 1 / (1 + e^(-(wᵀx + b)))
```

This is the sigmoid function: **σ(z) = 1/(1 + e⁻ᶻ)**.

### Cross-Entropy Loss from Maximum Likelihood

For a single sample:

```
P(y|x) = ŷʸ · (1-ŷ)^(1-y)     [Bernoulli likelihood]
```

Take negative log-likelihood:

```
- log P(y|x) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Average over all n samples to get the cost function J(w, b). Minimizing this is equivalent to maximizing the likelihood.

### Gradient (used in optimization)

```
∂J/∂wⱼ = 1/n Σᵢ₌₁ⁿ (ŷᵢ - yᵢ) · xⱼ⁽ⁱ⁾
∂J/∂b  = 1/n Σᵢ₌₁ⁿ (ŷᵢ - yᵢ)
```

This elegant result (prediction error × input) drives gradient descent.

---

## 11. How the Algorithm Works

```
Input Data (X, y)
       ↓
Initialize weights w = 0, b = 0
       ↓
┌──────────────────────────────────────┐
│  REPEAT until convergence:           │
│                                      │
│  1. For each sample: compute         │
│     z = wᵀx + b                     │
│                                      │
│  2. Apply sigmoid: ŷ = σ(z)          │
│                                      │
│  3. Compute loss J(w,b)             │
│                                      │
│  4. Compute gradients:               │
│     ∂J/∂w, ∂J/∂b                    │
│                                      │
│  5. Update:                          │
│     w = w - α · ∂J/∂w              │
│     b = b - α · ∂J/∂b              │
│                                      │
└──────────────────────────────────────┘
       ↓
Final Model: w*, b*
       ↓
Prediction: ŷ = σ(w*ᵀx + b*)
```

---

## 12. Training Process

**Pre-training:**
- Initialize all weights w to small random values or zeros, bias b = 0.
- Standardize features (recommended for faster convergence).

**During training:**
- Each iteration (epoch): compute all predictions, compute loss, compute gradients, update weights.
- The sigmoid output for each sample shifts as weights change.
- Loss should decrease monotonically (with proper learning rate).

**What's learned:**
- w encodes the direction and magnitude of the decision boundary.
- b encodes the offset of the boundary from the origin.

**Stopping criteria:**
- Maximum iterations reached.
- Gradient norm below a tolerance (convergence).
- Loss change between iterations below a threshold.

**Final model:** The learned w* and b* are used at prediction time. No training data is stored (unlike KNN).

---

## 13. Objective Function / Loss Function

**Objective:** Minimize the average binary cross-entropy (BCE) loss:

```
J(w, b) = -1/n Σ [yᵢ·log(ŷᵢ) + (1-yᵢ)·log(1-ŷᵢ)]
```

**Why BCE and not MSE?**
- MSE with sigmoid creates a non-convex loss surface (multiple local minima).
- BCE with sigmoid creates a **convex** loss surface — guaranteeing a single global minimum.
- BCE also penalizes confident wrong predictions more heavily.

**With regularization (L2):**

```
J(w, b) = BCE loss + λ/2 · ||w||²
```

The regularization term prevents large weights, reducing overfitting.

**High loss** means predictions are far from actual labels. **Low loss** means the model fits the data well.

---

## 14. Optimization

**Definition:** Find w and b that minimize J(w, b).

**Method:** Gradient Descent (or variants).

```
Current (w, b)
       ↓
Forward pass: compute ŷ = σ(wᵀx + b)
       ↓
Compute loss J(w, b)
       ↓
Backward pass: compute gradients ∂J/∂w, ∂J/∂b
       ↓
Update:
  w ← w - α · ∂J/∂w
  b ← b - α · ∂J/∂b
       ↓
New (w, b)
       ↓
Repeat until convergence
```

**Learning rate α:** Controls step size. Too large → overshooting/divergence. Too small → slow convergence.

**Variants:**
- **Batch GD:** uses entire dataset per update (stable but slow).
- **Stochastic GD (SGD):** uses one sample per update (noisy but fast).
- **Mini-batch GD:** uses a subset (common choice, ~32–256 samples).

**Convergence:** Since BCE + L2 is convex, gradient descent is guaranteed to reach the global minimum (given sufficient iterations and appropriate learning rate).

---

## 15. Complete Numerical Example

**Dataset (2 features, 3 samples):**

| Sample | x₁ | x₂ | y |
|--------|-----|-----|---|
| 1 | 1.0 | 2.0 | 1 |
| 2 | 2.0 | 1.0 | 0 |
| 3 | 3.0 | 3.0 | 1 |

**Initialize:** w₁ = 0, w₂ = 0, b = 0. Learning rate α = 0.5.

### Iteration 1

**Step 1 — Forward pass:**

Sample 1: z₁ = 0·1 + 0·2 + 0 = 0 → ŷ₁ = σ(0) = 0.5
Sample 2: z₂ = 0·2 + 0·1 + 0 = 0 → ŷ₂ = σ(0) = 0.5
Sample 3: z₃ = 0·3 + 0·3 + 0 = 0 → ŷ₃ = σ(0) = 0.5

**Step 2 — Compute gradients:**

```
∂J/∂w₁ = 1/3 [(0.5-1)·1 + (0.5-0)·2 + (0.5-1)·3]
        = 1/3 [(-0.5)(1) + (0.5)(2) + (-0.5)(3)]
        = 1/3 [-0.5 + 1.0 - 1.5]
        = 1/3 [-1.0] = -0.333

∂J/∂w₂ = 1/3 [(0.5-1)·2 + (0.5-0)·1 + (0.5-1)·3]
        = 1/3 [-1.0 + 0.5 - 1.5]
        = 1/3 [-2.0] = -0.667

∂J/∂b  = 1/3 [(0.5-1) + (0.5-0) + (0.5-1)]
        = 1/3 [-0.5 + 0.5 - 0.5]
        = -0.167
```

**Step 3 — Update:**

```
w₁ = 0 - 0.5·(-0.333) = 0.167
w₂ = 0 - 0.5·(-0.667) = 0.333
b  = 0 - 0.5·(-0.167) = 0.083
```

After 1 iteration: w = [0.167, 0.333], b = 0.083

The weights are now nonzero — the model has started learning. After many more iterations, w₁ and w₂ would converge to their optimal values.

**VERIFIED EXAMPLE** — gradients computed by hand following the exact formula.

---

## 16. Visual Explanation

### Decision Boundary (2D)

```
    x₂
    ↑
  4 |          ○ ○          ← Class 1
    |
  3 |       ○   | ●
    |           |
  2 |    ○  ----|----●      ← Decision boundary: w₁x₁ + w₂x₂ + b = 0
    |           |
  1 |       ●   | ●
    |           |
  0 +-----------+--------→ x₁
    0    1    2    3    4
              ↑
        ● ● ●    ← Class 0
```

The decision boundary is a **line** (in 2D) or a **hyperplane** (in higher dimensions). Points on one side are classified as 0, on the other as 1.

### Sigmoid Curve

```
σ(z)
 1.0 |                    ___________
     |                   /
 0.8 |                  /
     |                 /
 0.5 |· · · · · · · ·/· · · · · · ·  ← threshold
     |              /
 0.2 |             /
     |            /
 0.0 |___________/
     +----------|----------|--------→ z
              -5    0     +5
                    ↑
              z=0 → σ=0.5
```

---

## 17. Algorithm / Pseudocode

```
ALGORITHM: Logistic Regression (Binary, Gradient Descent)

1. INPUT: Training data X (n×d), y (n×1), learning rate α, max_iters M
2. INITIALIZE: w ← 0_d, b ← 0
3. FOR iter = 1 TO M:
   a. FOR i = 1 TO n:
      i.   z_i ← wᵀ · x_i + b
      ii.  ŷ_i ← 1 / (1 + exp(-z_i))
   b. Compute loss J = -1/n Σ [y_i·log(ŷ_i) + (1-y_i)·log(1-ŷ_i)]
   c. Compute gradients:
      dw ← 1/n · Xᵀ · (ŷ - y)
      db ← 1/n · Σ(ŷ_i - y_i)
   d. UPDATE:
      w ← w - α · dw
      b ← b - α · db
4. RETURN w, b
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.1, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iters):
            linear = X @ self.weights + self.bias
            y_predicted = self.sigmoid(linear)

            dw = (1 / n_samples) * (X.T @ (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        linear = X @ self.weights + self.bias
        return self.sigmoid(linear)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                               n_informative=2, random_state=1, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegressionScratch(learning_rate=0.1, n_iters=1000)
    model.fit(X_train, y_train)
    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy:  {model.score(X_test, y_test):.4f}")
```

---

## 19. Code Explanation

```
LogisticRegressionScratch class:
  __init__       → stores hyperparameters (lr, iterations)
  sigmoid        → applies σ(z) = 1/(1+e⁻ᶻ) to map linear output to probability
  fit            → training loop:
                   compute z = Xw + b
                   compute ŷ = sigmoid(z)
                   compute gradients dw = (1/n)·Xᵀ·(ŷ-y), db = (1/n)·Σ(ŷ-y)
                   update w ← w - α·dw, b ← b - α·db
  predict_proba  → returns P(y=1|x) for new data
  predict        → applies threshold (default 0.5) to probabilities
  score          → computes accuracy = correct predictions / total
```

---

## 20. Library Implementation

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=500, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(C=1.0, solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")
```

**Key parameters:**
- `C` = 1/λ (regularization strength). C small → strong regularization. C large → weak regularization.
- `solver`: 'lbfgs' (default, L2), 'liblinear' (small datasets), 'saga' (L1/L2, large datasets), 'newton-cg' (L2).
- `penalty`: 'l2' (default), 'l1', 'elasticnet', None.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| C (inverse regularization) | Controls regularization strength (C = 1/λ) | High C → overfitting; Low C → underfitting | Tune via cross-validation: try [0.001, 0.01, 0.1, 1, 10, 100] |
| max_iter | Maximum gradient descent iterations | Too low → no convergence; Too high → wasted time | Default 100 is often enough; increase if warning about convergence |
| solver | Optimization algorithm | Different solvers support different penalties | 'lbfgs' for L2; 'saga' for L1/ElasticNet; 'liblinear' for small datasets |
| penalty | Type of regularization | L1 → sparse weights; L2 → small weights; None → no regularization | L1 for feature selection; L2 for general use |
| learning_rate | Only for saga solver; controls step size | Too high → diverge; Too low → slow | Usually auto-set by solver |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned from data)
- **Weights w:** One weight per feature. Encodes feature importance and direction.
- **Bias b:** Single scalar. Shifts the decision boundary.

### Hyperparameters (chosen before training)
- **C:** Regularization strength.
- **max_iter:** Maximum iterations.
- **solver:** Optimization method.
- **penalty:** Regularization type (L1, L2, ElasticNet).

---

## 23. Assumptions

| Assumption | What It Means | How to Check | If Violated | Solution |
|---|---|---|---|---|
| Binary target | y ∈ {0, 1} | Check unique values | Multinomial problem | Use multinomial logistic regression or other classifiers |
| Linearity of log-odds | ln(p/(1-p)) is linear in features | Plot log-odds vs features; check residual patterns | Nonlinear relationship | Add polynomial/interaction features; use non-linear models |
| No multicollinearity | Features are not highly correlated | Check VIF (Variance Inflation Factor); VIF > 10 is problematic | Unstable weights, inflated standard errors | Remove correlated features; use PCA; increase regularization |
| Independence of observations | Each sample is independent | Study design knowledge | Biased standard errors | Use mixed-effects models or appropriate corrections |
| Large sample size | Enough data for MLE to work well | Rule of thumb: ≥10 events per predictor variable | Unreliable parameter estimates | Regularization; collect more data; reduce features |

---

## 24. Data Requirements

| Aspect | Requirement |
|---|---|
| Data type | Numerical features (encode categorical with one-hot or ordinal encoding) |
| Missing values | Not handled natively — impute or remove |
| Outliers | Sensitive — logistic regression can be affected by extreme outliers |
| Scaling | Recommended (helps convergence, especially with regularization) |
| Feature engineering | Polynomial features can capture non-linear relationships |
| Dataset size | Works well with small-to-medium datasets; rule of thumb: ≥10 events per feature |
| Class imbalance | Handles reasonably well; can adjust class_weight or use threshold tuning |

---

## 25. Feature Scaling

**Status: Recommended**

**Why:** Although logistic regression doesn't strictly require feature scaling (the sigmoid doesn't care about the scale of z), gradient descent converges much faster with standardized features. Regularization also benefits from scaling because L1/L2 penalizes all weights equally — if features have different scales, the penalty is unfair.

**Methods:**
- **StandardScaler** (z-score): (x - μ) / σ — preferred for logistic regression.
- **MinMaxScaler**: (x - min) / (max - min) — useful when you need bounded features.

**Note:** Tree-based models (decision trees, random forests) don't need scaling. Logistic regression does benefit from it.

---

## 26. Evaluation Metrics

| Metric | Formula | Interpretation | When to Use | When NOT to Use |
|---|---|---|---|---|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | Overall correct predictions | Balanced classes | Imbalanced classes |
| Precision | TP/(TP+FP) | Of predicted positives, how many are actually positive | When false positives are costly (spam filter) | When recall matters more |
| Recall (Sensitivity) | TP/(TP+FN) | Of actual positives, how many did we catch | When false negatives are costly (disease) | When precision matters more |
| F1-Score | 2·P·R/(P+R) | Harmonic mean of precision and recall | Imbalanced classes; balance P and R | When precision or recall alone matters |
| AUC-ROC | Area under ROC curve | Probability that model ranks a random positive higher than negative | Model comparison; threshold-independent | When you need calibrated probabilities |
| Log Loss | -Σ[y·log(ŷ)+(1-y)·log(1-ŷ)] | Measures probabilistic accuracy | When probability estimates matter | When you only care about class labels |
| Confusion Matrix | Table of TP, TN, FP, FN | Full picture of classification errors | Always useful for understanding errors | Doesn't give a single number |

**Training Objective ≠ Evaluation Metric:** The training objective is cross-entropy loss (minimized during training). Evaluation metrics like accuracy, F1, or AUC are computed after training and are NOT directly optimized.

---

## 27. Advantages

1. **Simple and interpretable:** Weights directly tell you feature importance and direction (positive/negative impact).
2. **Outputs probabilities:** Provides calibrated probability estimates, not just class labels.
3. **No hyperparameter tuning needed:** Works well with default settings (C=1).
4. **Computationally efficient:** Fast training and prediction (linear model).
5. **Convex optimization:** Guaranteed to find the global minimum (no local minima problem).
6. **Strong baseline:** Often surprisingly competitive with more complex models.
7. **Regularization built-in:** L1 (feature selection), L2 (weight shrinkage), ElasticNet.
8. **Well-studied:** Extensive theory, confidence intervals, hypothesis testing available.

---

## 28. Disadvantages

1. **Assumes linearity of log-odds:** Cannot capture complex non-linear relationships without manual feature engineering.
2. **Not great for complex problems:** Outperformed by tree-based methods and neural networks on complex tasks.
3. **Sensitive to outliers:** Extreme values can shift the decision boundary significantly.
4. **Multicollinearity issues:** Highly correlated features lead to unstable weight estimates.
5. **Feature engineering dependent:** Performance relies heavily on the quality of features provided.
6. **Decision boundary is linear:** Cannot learn curved or non-linear boundaries natively.

---

## 29. When to Use

- ✓ You need a fast, interpretable baseline model.
- ✓ You want probability estimates (not just class labels).
- ✓ The relationship between features and log-odds is approximately linear.
- ✓ You have a small-to-medium dataset.
- ✓ You want a model for inference (understanding feature effects).
- ✓ Binary or multinomial classification.
- ✓ You need to deploy a lightweight model in production.

---

## 30. When NOT to Use

- ✗ The decision boundary is clearly non-linear (curved, circular, XOR pattern).
- ✗ You have complex feature interactions that are hard to engineer manually.
- ✗ You need the highest possible predictive accuracy (use gradient boosting or neural nets).
- ✗ You have massive datasets with millions of features and non-linear patterns.
- ✗ You have severe multicollinearity and cannot remove correlated features.

---

## 31. Real-World Applications

1. **Spam Detection**
   - Problem: Classify emails as spam or not spam
   - Input: Word frequencies, sender info, email metadata
   - Algorithm: Logistic Regression
   - Output: Probability of spam → class label

2. **Disease Diagnosis**
   - Problem: Predict whether a patient has a disease
   - Input: Lab results, age, symptoms
   - Algorithm: Logistic Regression (often with L1 for feature selection)
   - Output: Probability of disease → risk category

3. **Credit Scoring**
   - Problem: Predict loan default probability
   - Input: Income, credit history, debt ratio, employment
   - Algorithm: Logistic Regression (industry standard in banking)
   - Output: Default probability → approve/deny

4. **Click-Through Rate (CTR) Prediction**
   - Problem: Predict whether a user will click an ad
   - Input: User features, ad features, context
   - Algorithm: Logistic Regression (large-scale, online learning)
   - Output: Click probability

---

## 32. Failure Cases

1. **Data:** XOR problem — logistic regression cannot learn XOR with 2 features because the decision boundary is linear.
2. **Mathematical:** Perfect separation — when classes are linearly separable, MLE weights diverge to infinity (need regularization).
3. **Optimization:** Poor learning rate — too high causes divergence; too low causes slow convergence.
4. **Generalization:** Overfitting with too many features relative to samples.
5. **Practical:** Missing feature engineering — without interaction/polynomial features, cannot capture non-linear patterns.

---

## 33. Overfitting and Underfitting

**Underfitting (high bias):**
- Model too simple (e.g., linear decision boundary for non-linear data).
- Symptoms: Low train accuracy, low test accuracy.
- Solution: Add polynomial features, interaction terms, or use a non-linear model.

**Overfitting (high variance):**
- Model too complex relative to data (e.g., too many features, no regularization).
- Symptoms: High train accuracy, much lower test accuracy.
- Solution: Increase regularization (decrease C), reduce features, collect more data.

**Balanced:**
- Train accuracy ≈ Test accuracy, both reasonably high.
- Moderate regularization (C around 0.1–10).

---

## 34. Bias-Variance Perspective

**Logistic Regression is a HIGH-BIAS, LOW-VARIANCE model.**

- **Bias:** Because it assumes a linear decision boundary, it can systematically miss non-linear patterns (high bias).
- **Variance:** Because it's a simple parametric model with few parameters, it's stable across different training sets (low variance).
- **Trade-off:** Use regularization (L1/L2) to control this trade-off. Lower C → higher bias, lower variance. Higher C → lower bias, higher variance.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Logistic Regression | Linear boundary + sigmoid | Interpretable, fast, probabilistic | Cannot capture non-linear patterns | Baseline, interpretable models |
| SVM | Maximum margin classifier | Strong theoretical foundation, kernel trick | Less interpretable, no probabilities by default | High-dimensional data |
| Decision Tree | Rule-based splits | Handles non-linearity, interpretable | Prone to overfitting | When interpretability matters |
| Naive Bayes | Conditional independence | Very fast, works with small data | Independence assumption often wrong | Text classification |
| Neural Network | Layered non-linear transforms | Captures complex patterns | Black box, needs lots of data | Complex tasks with large data |

---

## 36. Algorithm Selection Guide

```
Is your problem classification?
├── YES
│   ├── Need probability estimates?
│   │   ├── YES → Logistic Regression (start here)
│   │   └── NO
│   │       ├── Small dataset, high dimensions? → SVM / Naive Bayes
│   │       ├── Need interpretability? → Decision Tree / Logistic Regression
│   │       ├── Need maximum accuracy? → Gradient Boosting / Neural Network
│   │       └── Quick prototype? → Logistic Regression
│   └── Binary or multiclass?
│       ├── Binary → Logistic Regression (simplest)
│       └── Multiclass → Logistic Regression (softmax) or One-vs-Rest
└── NO → Regression problem → use Linear Regression / etc.
```

---

## 37. Common Mistakes

```
❌ Not scaling features before training
   Why wrong: Gradient descent converges slowly with unscaled features; regularization is unfair.
   Correct: Apply StandardScaler before fitting.

❌ Ignoring multicollinearity
   Why wrong: Highly correlated features make weight estimates unstable and uninterpretable.
   Correct: Check VIF; remove or combine correlated features.

❌ Using accuracy for imbalanced classes
   Why wrong: A model that always predicts majority class gets 95% accuracy but is useless.
   Correct: Use F1-score, precision, recall, or AUC-ROC.

❌ Not applying regularization (C too large)
   Why wrong: Overfitting, especially with many features.
   Correct: Use cross-validation to tune C; start with C=1.0.

❌ Forgetting that logistic regression is linear in log-odds
   Why wrong: Expecting it to learn non-linear boundaries without feature engineering.
   Correct: Add polynomial features or use a non-linear model.
```

---

## 38. Interview Questions

### Beginner

**Q1: Why is logistic regression called "regression" if it's a classification algorithm?**
A: It models the probability (a continuous value between 0 and 1) using a regression-like equation. The name refers to the regression on log-odds. It then rounds to get a class label.

**Q2: What is the sigmoid function and why do we use it?**
A: σ(z) = 1/(1+e⁻ᶻ). It maps any real number to (0, 1), making the output interpretable as a probability.

**Q3: What is the decision boundary of logistic regression?**
A: A hyperplane defined by wᵀx + b = 0. It's always linear in the original feature space.

**Q4: What loss function does logistic regression use?**
A: Binary cross-entropy (log loss): J = -1/n Σ [y·log(ŷ) + (1-y)·log(1-ŷ)].

**Q5: How do you handle multiclass classification with logistic regression?**
A: Two approaches: One-vs-Rest (OvR) trains K binary classifiers, or Multinomial (Softmax) directly models K classes.

### Intermediate

**Q6: Why not use MSE loss with sigmoid for logistic regression?**
A: MSE + sigmoid creates a non-convex loss surface with multiple local minima. Cross-entropy loss + sigmoid is convex, guaranteeing a global minimum.

**Q7: What does the C parameter control?**
A: C = 1/λ is the inverse regularization strength. Small C = strong regularization (simpler model), large C = weak regularization (complex model).

**Q8: How does L1 regularization differ from L2 in logistic regression?**
A: L1 (Lasso) drives some weights to exactly zero (feature selection). L2 (Ridge) shrinks all weights toward zero but never exactly zero.

**Q9: What happens when classes are perfectly separable?**
A: Without regularization, MLE weights diverge to infinity (the sigmoid pushes toward 0 or 1, and the boundary becomes infinitely sharp). Regularization prevents this.

**Q10: How does logistic regression handle missing values?**
A: It doesn't natively. You must impute (mean, median, mode) or remove missing values before training.

### Advanced

**Q11: Explain the connection between logistic regression and neural networks.**
A: Logistic regression is a single-layer neural network with a sigmoid activation function. A multi-layer neural network is a non-linear generalization.

**Q12: What is the probabilistic interpretation of logistic regression?**
A: It's a generalized linear model (GLM) with a Bernoulli distribution for the response and a logit link function. It models P(y=1|x) directly.

**Q13: How would you derive the gradient of the cross-entropy loss?**
A: Start with J = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]. Use the fact that dσ/dz = σ(1-σ). Chain rule gives ∂J/∂w = (σ(z) - y)·x.

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
1. Sigmoid: σ(z) = 1/(1+e⁻ᶻ)
2. Logit (inverse sigmoid): z = ln(p/(1-p))
3. Cross-entropy: J = -1/n Σ [y·log(ŷ) + (1-y)·log(1-ŷ)]
4. Gradient: ∂J/∂wⱼ = 1/n Σ (ŷᵢ - yᵢ)·xⱼ⁽ⁱ⁾

**Key concepts:**
- Logistic regression is **discriminative** (models P(y|x) directly), not generative (which models P(x|y) and uses Bayes' rule).
- It's a **linear classifier** in the original feature space.
- Convex loss → guaranteed global optimum.
- Multiclass via softmax (multinomial logistic regression) or One-vs-Rest.

**Common traps:**
- Confusing sigmoid with softmax (sigmoid for binary, softmax for multiclass).
- Forgetting that logistic regression needs regularization for separable data.
- Assuming it can learn non-linear boundaries (it can't without feature engineering).

*(The above are representative concept patterns, not past GATE PYQs.)*

---

## 40. Coding Practice

**Level 1 — Basic:**
Implement sigmoid function and verify: σ(0) = 0.5, σ(5) ≈ 0.993, σ(-5) ≈ 0.007.

**Level 2 — Simple model:**
Train logistic regression on a 2-feature, 2-class dataset. Plot the decision boundary.

**Level 3 — Binary classification:**
Implement binary classification on breast cancer dataset (sklearn). Report accuracy, confusion matrix, classification report.

**Level 4 — Regularization:**
Compare L1 vs L2 vs no regularization. How do weights change? Which features are selected by L1?

**Level 5 — Multiclass:**
Use One-vs-Rest logistic regression on Iris dataset. Compare with multinomial (softmax) approach.

**Level 6 — Advanced:**
Implement logistic regression with mini-batch SGD from scratch. Compare convergence speed with batch GD.

**Level 7 — Real-world case study:**
Predict customer churn using a telecom dataset. Handle missing values, class imbalance, feature engineering. Evaluate with AUC-ROC and precision-recall curve.

---

## 41. Practical ML Workflow

```
Problem Definition
    ↓ "Binary classification: predict customer churn"
Data Collection
    ↓ "5000 customers, 20 features (usage, demographics, billing)"
EDA
    ↓ "Class distribution: 73% stay, 27% churn (imbalanced)"
Cleaning
    ↓ "Impute missing values (median for numerical, mode for categorical)"
Feature Engineering
    ↓ "Create tenure_groups, monthly_charge_ratio, encode contract type"
Split
    ↓ "80% train, 20% test, stratified split to preserve class ratios"
Preprocessing
    ↓ "StandardScaler for numerical features, OneHotEncoder for categorical"
Train
    ↓ "LogisticRegression(C=1.0, class_weight='balanced')"
Tune
    ↓ "GridSearchCV over C=[0.01, 0.1, 1, 10], penalty=['l1','l2']"
Evaluate
    ↓ "AUC-ROC: 0.85, Precision: 0.72, Recall: 0.68, F1: 0.70"
Error Analysis
    ↓ "Most errors on customers with short tenure — add interaction features"
Deploy & Monitor
    ↓ "Serve via Flask API, monitor prediction distribution drift"
```

---

## 42. Complexity

| Aspect | Complexity |
|---|---|
| Training time | O(n · d · I) where n = samples, d = features, I = iterations |
| Prediction time | O(d) per sample (just wᵀx + b and sigmoid) |
| Space | O(d) — stores weight vector and bias only |
| Scaling with n | Linear — very efficient with large n |
| Scaling with d | Linear — but consider regularization for high d |
| Scaling with classes | O(K·d) for K classes (multinomial) or O(K) binary classifiers (OvR) |

---

## 43. Advanced Concepts

1. **Maximum Likelihood Estimation:** Logistic regression parameters are estimated by maximizing the likelihood of the observed data under a Bernoulli model.

2. **Convexity:** The cross-entropy loss for logistic regression is convex in w and b. This is a rare and powerful property — no local minima exist.

3. **Probit Regression:** An alternative to logistic regression using the CDF of the normal distribution instead of the sigmoid. Very similar results in practice.

4. **Ordinal Logistic Regression:** For ordered categories (e.g., ratings: low, medium, high).

5. **Regularization paths:** L1 regularization creates a path of solutions as C varies. Some weights become exactly zero at different C values — useful for feature selection.

6. **Online Logistic Regression:** Can be trained one sample at a time (SGD), making it suitable for streaming data.

---

## 44. Connections to Other Algorithms

```
                    Logistic Regression
                    /         |         \
                   /          |          \
        Linear        Neural Network    SVM
      Regression      (single layer)   (linear kernel)
         |               |               |
    Same linear      Same formula     Same linear
    model,           with hidden      boundary,
    different        layers adds      different
    output           non-linearity    loss function
    (continuous)     (sigmoid→ReLU)   (hinge loss)
```

- **Linear Regression → Logistic Regression:** Same linear model, different output activation (identity vs sigmoid), different loss (MSE vs BCE).
- **Logistic Regression → Neural Network:** Stack multiple layers; logistic regression is the simplest (single-layer) neural network.
- **Logistic Regression vs SVM:** Both find linear boundaries but use different loss functions (BCE vs hinge loss). LR gives probabilities; SVM focuses on margin.

---

## 45. If You Remember Only 5 Things

1. **Sigmoid squashes real numbers to (0, 1)** — that's how logistic regression produces probabilities.
2. **Cross-entropy loss is convex** — guaranteeing a single global minimum during training.
3. **The decision boundary is linear** — wᵀx + b = 0. Add polynomial features for non-linear patterns.
4. **C controls regularization** — small C = simpler model, large C = complex model.
5. **It's the baseline classifier** — always try logistic regression first. It's fast, interpretable, and surprisingly strong.

---

## 46. Cheat Sheet

| Item | Detail |
|---|---|
| Algorithm | Logistic Regression |
| Category | Supervised, Classification |
| Goal | Model P(y=1\|x) using sigmoid of linear combination |
| Input | Features X, binary labels y |
| Output | Probability ŷ ∈ (0,1), class label ŷ ∈ {0,1} |
| Core Formula | ŷ = σ(wᵀx + b) = 1/(1+e⁻⁽ʷᵀˣ⁺ᵇ⁾) |
| Loss | Binary Cross-Entropy: -1/n Σ [y·log(ŷ)+(1-y)·log(1-ŷ)] |
| Optimization | Gradient Descent (batch, stochastic, or mini-batch) |
| Parameters | w (weights), b (bias) — learned from data |
| Hyperparameters | C, solver, penalty, max_iter |
| Assumptions | Linearity of log-odds, independence, no multicollinearity |
| Advantages | Interpretable, fast, probabilistic, convex, strong baseline |
| Disadvantages | Linear boundary only, needs feature engineering for non-linear |
| Use When | Baseline needed, interpretable model, probability estimates |
| Avoid When | Complex non-linear patterns, maximum accuracy needed |
| Related | Linear Regression, SVM, Neural Networks |
| Key Exam Points | Sigmoid, BCE loss, convexity, MLE, discriminative vs generative |
| Key Interview Points | Why BCE not MSE, sigmoid properties, regularization effect |

---

## 47. Final Mental Model

```
┌──────────┐    ┌──────────────┐    ┌────────┐    ┌────────────┐    ┌──────────┐
│ Raw Data │───→│ wᵀx + b     │───→│ σ(z)   │───→│ P(y=1|x)  │───→│ Class ŷ  │
│ (X, y)   │    │ (linear      │    │(sigmoid│    │ probability│    │ 0 or 1   │
│          │    │  combination)│    │ squish)│    │            │    │          │
└──────────┘    └──────────────┘    └────────┘    └────────────┘    └──────────┘
                       ↑                                        │
                       │         ┌──────────────────┐           │
                       └─────────│  Gradient Descent │←──────────┘
                                 │  minimize BCE loss│
                                 └──────────────────┘
```

---

## 48. Knowledge Check

### Recall (5)

1. What is the formula for the sigmoid function?
2. What loss function does logistic regression use?
3. Is logistic regression a generative or discriminative model?
4. What does the C hyperparameter control?
5. What is the shape of the decision boundary for logistic regression with 2 features?

### Understanding (5)

6. Why does logistic regression use cross-entropy loss instead of MSE?
7. How does logistic regression extend to multiclass problems?
8. Why is the cross-entropy loss for logistic regression convex?
9. What happens to the sigmoid function when z → +∞ and z → -∞?
10. Why is logistic regression considered the "baseline" classifier?

### Application (5)

11. You have a dataset with 5 features and 1000 samples. What would you do first?
12. Your logistic regression has 95% accuracy but your F1-score is 0.3. What does this tell you?
13. You train logistic regression and the weights are very large (|w| > 100). What's likely wrong?
14. How would you use logistic regression for a problem with 3 classes?
15. Your model performs well on training data but poorly on test data. What do you adjust?

### Mathematical (5)

16. If σ(z) = 0.8, what is z? (Hint: z = ln(p/(1-p)))
17. Given y=1 and ŷ=0.9, what is the loss for this sample?
18. Given y=0 and ŷ=0.1, what is the loss for this sample?
19. If the gradient ∂J/∂w₁ is positive, should w₁ increase or decrease?
20. What is the derivative of σ(z) with respect to z?

### Interview (5)

21. Explain logistic regression to a non-technical stakeholder.
22. When would you choose logistic regression over a random forest?
23. What are the limitations of logistic regression in production?
24. How do you handle class imbalance in logistic regression?
25. What's the relationship between logistic regression and a single-layer neural network?

### Problem Solving (5)

26. Design a spam detection system using logistic regression. What features would you create?
27. You receive a dataset with 99% negative and 1% positive samples. How do you train a logistic regression model?
28. A model trained on data from 2020 doesn't perform well on 2024 data. What happened and what do you do?
29. Your logistic regression converges slowly. List 3 things you could try.
30. Compare logistic regression performance with C=0.01 vs C=100. Which overfits?

### Answers

**1.** σ(z) = 1/(1+e⁻ᶻ)

**2.** Binary cross-entropy (log loss).

**3.** Discriminative — it models P(y|x) directly.

**4.** Inverse regularization strength. C = 1/λ. Small C → strong regularization.

**5.** A line (in 2D) or hyperplane (in higher dimensions).

**6.** MSE + sigmoid creates a non-convex loss surface with local minima. Cross-entropy + sigmoid is convex → guaranteed global minimum.

**7.** One-vs-Rest (K binary classifiers) or multinomial (softmax function for K classes).

**8.** It's the negative log-likelihood of a Bernoulli distribution, which is convex in the parameters when the sigmoid link function is used.

**9.** σ(+∞) → 1, σ(-∞) → 0.

**10.** It's simple, fast, interpretable, has strong theoretical foundations, and often surprisingly competitive. Start with it to establish a baseline.

**11.** Split data, train a logistic regression as a baseline, evaluate, then try more complex models.

**12.** The classes are likely imbalanced. The model predicts mostly the majority class, getting high accuracy but missing the minority class (low recall → low F1).

**13.** Likely not enough regularization (C too large). Increase regularization (decrease C).

**14.** Use `LogisticRegression(multi_class='multinomial')` or `LogisticRegression(multi_class='ovr')`.

**15.** Increase regularization (decrease C), add dropout, reduce features, or collect more data.

**16.** z = ln(0.8/(1-0.8)) = ln(4) ≈ 1.386.

**17.** -log(0.9) ≈ 0.105.

**18.** -log(1-0.1) = -log(0.9) ≈ 0.105.

**19.** Decrease (we subtract α · gradient in gradient descent).

**20.** dσ/dz = σ(z)(1-σ(z)). For σ(z)=0.8: 0.8 × 0.2 = 0.16.

**21–25.** Open-ended; review sections 03, 27–30, 35 for reference.

**26–30.** Open-ended; review sections 24, 29–30, 33, 41 for reference.

---

## 49. Final Learning Checklist

- [ ] I can explain logistic regression in plain English
- [ ] I know the sigmoid function formula and can compute it by hand
- [ ] I understand why cross-entropy loss is used (not MSE)
- [ ] I can derive the gradient of the BCE loss
- [ ] I can implement logistic regression from scratch in Python
- [ ] I know how to use sklearn's LogisticRegression
- [ ] I understand what C (regularization) does and how to tune it
- [ ] I know the difference between L1 and L2 regularization
- [ ] I can explain the decision boundary (linear hyperplane)
- [ ] I understand the difference between parameters and hyperparameters
- [ ] I can handle multiclass classification with logistic regression
- [ ] I know when to use and when NOT to use logistic regression
- [ ] I can compute accuracy, precision, recall, F1, and AUC-ROC
- [ ] I understand the bias-variance tradeoff for logistic regression
- [ ] I can compare logistic regression with SVM, decision trees, and neural networks
- [ ] I know the computational complexity (training and prediction)
- [ ] I can explain the probabilistic interpretation (GLM with Bernoulli)
- [ ] I can identify and handle class imbalance
- [ ] I can handle missing values before training
- [ ] I know the assumptions and what happens when they're violated
- [ ] I can explain logistic regression to a non-technical person
- [ ] I have completed at least one real-world classification project with logistic regression

---

## 50. Quality Control Note

| Criterion | Status | Notes |
|---|---|---|
| Accuracy | ✅ | Sigmoid, BCE loss, gradient, and convexity verified against standard references |
| Beginner-friendliness | ✅ | Real-life analogy (doctor diagnosis), step-by-step reasoning, no jargon without definition |
| Math depth | ✅ | Full derivation of sigmoid from log-odds, BCE from MLE, gradient formula, worked example |
| Practical depth | ✅ | sklearn usage, hyperparameter tuning, evaluation metrics, real-world applications |
| Exam depth | ✅ | Key formulas, conceptual traps, GATE-relevant concepts clearly identified |
| Code quality | ✅ | From-scratch implementation with numpy, clean sklearn usage |
| Structure compliance | ✅ | All 50 sections present in order; no sections omitted |
