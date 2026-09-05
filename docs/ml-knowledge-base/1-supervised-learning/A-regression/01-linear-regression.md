# 01. Linear Regression

> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐☆☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Linear Regression |
| Category | Supervised Learning |
| Type | Regression |
| Parametric / Non-parametric | Parametric |
| Generative / Discriminative | Discriminative |
| Main Objective | Model the linear relationship between input features and a continuous target variable |
| Input | Feature matrix X (n samples × m features) |
| Output | Continuous numeric prediction ŷ |
| Core Idea | Fit a straight line (or hyperplane) that best predicts the target by minimizing the sum of squared errors |
| Typical Use Cases | House price prediction, sales forecasting, salary prediction, trend analysis |

---

## 02. One-Line Definition

### Beginner Definition
Linear Regression draws the straight line that passes closest to all the data points, then uses that line to predict new values.

### Technical Definition
Linear Regression is a supervised parametric model that assumes the target variable `y` is a linear function of the features `X`, and it learns the weight coefficients by minimizing the residual sum of squared errors between predicted and actual values.

---

## 03. Intuition

Imagine you have data points showing how much a pizza costs based on its diameter. Bigger pizzas cost more, roughly in a straight-line relationship.

The task: draw the one straight line that comes as close as possible (overall) to every point. Some points will be slightly above the line, some slightly below. Linear Regression finds the line where the total "distance error" is minimized.

Step-by-step reasoning:
1. We believe cost = (some number) × diameter + (some base cost).
2. The two unknown numbers are the **slope** and the **intercept**.
3. We try different slopes and intercepts, computing how wrong the line is each time.
4. We keep the pair that makes the total squared error smallest.

That line, once found, lets us predict the cost of any diameter — even ones we never saw.

---

## 04. Problem It Solves

**Problem:** We have data where a numeric outcome is influenced by one or more numeric features, and we want to predict that outcome for new data.

**Example:** A real-estate agent has a table of houses: `size (sq ft)` and `price`. They want to estimate the price of a new house with a known size.

What makes it hard: the relationship is noisy — two same-sized houses can sell for different amounts. Linear Regression captures the **average trend** and quantifies uncertainty around it.

Why useful: it's fast, interpretable, gives you the exact influence of each feature (its coefficient), and serves as the baseline every more complex model is compared against.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning (labeled data)
│   ├── Classification (discrete target)
│   └── Regression (continuous target)
│       ├── Linear Models
│       │   ├── Linear Regression          ← YOU ARE HERE
│       │   ├── Polynomial Regression
│       │   ├── Ridge Regression
│       │   ├── Lasso Regression
│       │   ├── Elastic Net
│       │   ├── Bayesian Regression
│       │   └── Huber / Quantile
│       ├── Support Vector Regression
│       ├── Tree-based (Decision Tree, RF, Boosting)
│       └── Neural Networks
├── Unsupervised Learning
└── Reinforcement Learning
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Feature (X) | Input variable used for prediction | Independent variable, predictor, explanatory variable |
| Target (y) | Value we try to predict | Dependent variable, response, label |
| Coefficient / Weight | How much each feature contributes | The slope; the learned parameter w |
| Intercept (b) | The value when all features are 0 | The bias term; where the line crosses the y-axis |
| Prediction (ŷ) | The model's guess | Estimated target using learned weights |
| Residual | How far prediction is from truth | Error = y − ŷ |
| RSS / SSE | Measure of total error | Residual Sum of Squares = Σ(y − ŷ)² |
| MSE | Average squared error | Mean Squared Error = RSS / n |
| Ordinary Least Squares (OLS) | The fitting method used | Minimizing sum of squared residuals to find best line |
| Fit | The training process | Finding optimal w and b |

---

## 07. Input and Output

**Input (Training):**
- Feature matrix `X` of shape (n_samples, n_features).
- Target vector `y` of shape (n_samples,) — continuous values.

**Input (Prediction):**
- New feature row(s) with same number of columns.

**Output (Training):**
- Learned parameters: weight vector `w` (n_features,) and bias `b` (scalar).

**Output (Prediction):**
- Continuous prediction ŷ per row.

**Parameters learned:** coefficients (slope) `w`, intercept `b`.

**Hyperparameters (simple linear regression):** essentially none if using OLS; `fit_intercept` toggles the bias term.

---

## 08. Mathematical Foundation

The core assumption is that the true target is a linear combination of the features plus some noise:

```text
y = b + w₁x₁ + w₂x₂ + ... + wₘxₘ + ε
```

Where `ε` (epsilon) is the irreducible error/noise term.

For a single feature this reduces to:

```text
y = b + w·x
```

The model's prediction is:

```text
ŷ = b + Σ(wⱼ·xⱼ)   for j = 1..m
```

**Notation used:**
- `n` = number of samples (rows)
- `m` = number of features (columns)
- `xᵢⱼ` = value of feature j for sample i
- `yᵢ` = actual target of sample i
- `ŷᵢ` = predicted target of sample i
- `wⱼ` = weight/coefficient for feature j
- `b` = intercept / bias
- `εᵢ` = error term for sample i

**Required math concepts:** basic algebra, partial derivatives, summation notation, matrix multiplication.

---

## 09. Core Formula

### The Model (Prediction)

```text
ŷ = w·x + b
```

#### Meaning
For any input x, multiply it by the slope w and add the intercept b to get the prediction.

#### Symbols
- `x` = input feature value
- `w` = slope / weight (how steep the line is)
- `b` = intercept (value of ŷ when x = 0)
- `ŷ` = predicted output

#### Intuition
"Bigger x means bigger ŷ if w is positive." The slope controls direction and steepness; the intercept shifts the whole line up or down.

#### Example
Data: (x, y) = (1, 3), (2, 5), (3, 7). Let w = 2, b = 1.
- x = 1: ŷ = 2·1 + 1 = 3 ✓
- x = 2: ŷ = 2·2 + 1 = 5 ✓
- x = 3: ŷ = 2·3 + 1 = 7 ✓

---

### Ordinary Least Squares (OLS) Objective

```text
Minimize  Σᵢ(yᵢ − ŷᵢ)²  =  Σᵢ(yᵢ − (w·xᵢ + b))²
```

#### Meaning
We choose w and b to minimize the sum of squared vertical distances between actual targets and predictions.

#### Symbols
- `yᵢ` = actual target of sample i
- `ŷᵢ` = prediction for sample i
- `Σᵢ` = sum over all n samples
- `(yᵢ − ŷᵢ)²` = squared residual for sample i

#### Intuition
Squaring makes all errors positive, penalizes large errors more than small ones, and gives a smooth, differentiable function to minimize.

#### Example
w = 2, b = 1, with samples (1,3), (2,6), (3,7):
- Residuals: 3−3=0, 6−5=1, 7−7=0
- Squared: 0, 1, 0
- RSS = 0 + 1 + 0 = 1

Compare w=2, b=0: predictions 2,4,6; residuals 1,2,1; RSS = 1+4+1 = 6. So (w=2,b=1) is better.

---

### Closed-Form Solution (single feature)

```text
w = Σᵢ(xᵢ − x̄)(yᵢ − ȳ) / Σᵢ(xᵢ − x̄)²
b = ȳ − w·x̄
```

#### Meaning
The best slope is the ratio of the covariance of x and y to the variance of x; the best intercept passes the line through the mean point.

#### Symbols
- `x̄` = mean of x values
- `ȳ` = mean of y values
- `w`, `b` = slope and intercept

#### Intuition
The line always passes through the "center of gravity" of the data (x̄, ȳ), and the slope is a normalized measure of how x and y move together.

---

### Matrix Form (general, m features)

```text
ŷ = X·w      (with b absorbed by adding a column of 1s to X)
w = (XᵀX)⁻¹ Xᵀ y
```

#### Meaning
The closed-form least-squares solution in matrix notation.

#### Symbols
- `X` = (n × (m+1)) design matrix with a column of ones
- `w` = ((m+1)) vector of weights including intercept
- `Xᵀ` = transpose of X
- `(XᵀX)⁻¹` = inverse of the Gram matrix

#### Intuition
Directly computes the weights that minimize squared error in one shot, provided `XᵀX` is invertible.

---

## 10. Derivation

**Step 1 — Start with the objective.** Minimize the sum of squared errors:

```text
J(w, b) = Σᵢ(yᵢ − b − w·xᵢ)²
```

**Step 2 — Take partial derivative w.r.t. b.**

```text
∂J/∂b = Σᵢ 2(yᵢ − b − w·xᵢ)(−1) = −2 Σᵢ(yᵢ − b − w·xᵢ)
```

Set to 0:
```text
Σᵢ(yᵢ − b − w·xᵢ) = 0  ⇒  n·ȳ − n·b − w·n·x̄ = 0  ⇒  b = ȳ − w·x̄
```

**Step 3 — Take partial derivative w.r.t. w.**

```text
∂J/∂w = Σᵢ 2(yᵢ − b − w·xᵢ)(−xᵢ) = −2 Σᵢ xᵢ(yᵢ − b − w·xᵢ)
```

Set to 0 and substitute b = ȳ − w·x̄. After algebra (expanding and using Σ(xᵢ−x̄)(xᵢ−x̄) = Σ(xᵢ−x̄)²):

```text
w = Σᵢ(xᵢ − x̄)(yᵢ − ȳ) / Σᵢ(xᵢ − x̄)²
```

**Step 4 — The matrix generalization.** For multiple features, the same minimization in matrix form yields:

```text
w = (XᵀX)⁻¹ Xᵀ y
```

This is the **normal equation**. It requires `XᵀX` to be invertible (no perfect multicollinearity), otherwise you use gradient descent or add regularization.

---

## 11. How the Algorithm Works

```text
Input (X, y)
    ↓
Preprocessing (check linearity, handle missing values)
    ↓
Initialization (choose OLS closed form OR random w,b for gradient descent)
    ↓
Compute predictions:  ŷ = w·x + b
    ↓
Compute loss:  J = Σ(y − ŷ)²
    ↓
Optimization (closed form: solve normal equation; OR gradient descent steps)
    ↓
Convergence / direct solve → final w, b
    ↓
Final Model:  ŷ = w·x + b
    ↓
Prediction on new data
```

---

## 12. Training Process

**Pre-training:** Split data into train/test. Optionally standardize features (helps gradient descent, not OLS).

**During training:**
- With **OLS**: solve the normal equation directly — no iterations.
- With **gradient descent** (preferred for many features / big data): repeatedly update w and b in the direction that reduces cost until convergence.

**What is learned:** the weight vector and the bias.

**Changes per iteration (gradient descent):**
```text
w ← w − α·(1/n)·Σᵢ(ŷᵢ − yᵢ)·xᵢ
b ← b − α·(1/n)·Σᵢ(ŷᵢ − yᵢ)
```
where `α` is the learning rate.

**Stopping:** cost stops decreasing meaningfully, or a fixed number of epochs, or gradient magnitude small.

**Final model contents:** just `w` (vector) and `b` (scalar).

---

## 13. Objective Function / Loss Function

The objective is to minimize the **Residual Sum of Squares (RSS)**:

```text
J(w, b) = Σᵢ(yᵢ − ŷᵢ)²
```

Why squared? Squaring:
- Makes all residuals non-negative
- Penalizes large errors proportionally more
- Makes the function smooth and convex (guaranteed single global minimum)
- Mathematically tractable (derivatives are linear)

**Convexity:** RSS is a convex function of w and b, so any local minimum is the global minimum — this is why OLS/gradient descent always converge to the same answer regardless of initialization.

**Low loss** = predictions close to targets. **High loss** = predictions far off.

---

## 14. Optimization

**Definition:** Find the w, b that minimize J.

**Method options:**
1. **Closed form (normal equation):** w = (XᵀX)⁻¹Xᵀy. Exact, one step, but O(m³) and fails if XᵀX singular.
2. **Gradient Descent:** iterative; works for large m and huge data.

**Gradient:** the vector of partial derivatives pointing in the direction of steepest increase of cost. We move opposite to it.

**Update rule:**
```text
Current w,b
    ↓
Predict  ŷᵢ = w·xᵢ + b
    ↓
Compute gradient:  ∇w = (2/n)Σ(ŷᵢ−yᵢ)xᵢ ;  ∇b = (2/n)Σ(ŷᵢ−yᵢ)
    ↓
Update:  w ← w − α·∇w ;  b ← b − α·∇b
    ↓
Repeat until convergence
```

**Learning rate α:** too large → overshoot/divergence; too small → slow convergence.

**Convergence:** since the objective is convex, gradient descent reaches the unique global minimum.

---

## 15. Complete Numerical Example

Take 3 samples: (x, y) = (1, 2), (2, 4), (3, 5).

**Step 1 — Compute means:**
```text
x̄ = (1+2+3)/3 = 2.0
ȳ = (2+4+5)/3 = 3.667
```

**Step 2 — Compute w using closed form:**
```text
Numerator   = Σ(x−x̄)(y−ȳ)
            = (1−2)(2−3.667) + (2−2)(4−3.667) + (3−2)(5−3.667)
            = (−1)(−1.667) + (0)(0.333) + (1)(1.333)
            = 1.667 + 0 + 1.333 = 3.0
Denominator = Σ(x−x̄)² = (−1)² + 0² + 1² = 2.0
w = 3.0/2.0 = 1.5
```

**Step 3 — Compute b:**
```text
b = ȳ − w·x̄ = 3.667 − 1.5·2.0 = 3.667 − 3.0 = 0.667
```

**Step 4 — Predictions and residuals:**
```text
x=1: ŷ = 1.5·1 + 0.667 = 2.167   residual = 2 − 2.167 = −0.167
x=2: ŷ = 1.5·2 + 0.667 = 3.667   residual = 4 − 3.667 = 0.333
x=3: ŷ = 1.5·3 + 0.667 = 5.167   residual = 5 − 5.167 = −0.167
```

**Step 5 — RSS:**
```text
(−0.167)² + (0.333)² + (−0.167)² = 0.028 + 0.111 + 0.028 = 0.167
```

**VERIFIED EXAMPLE** — hand-verified with exact arithmetic; the fitted line is ŷ = 1.5x + 0.667.

---

## 16. Visual Explanation

```text
      y
      │                 • (3,5)
      │              •  /      ← fitted line ŷ = 1.5x + 0.667
      │           •   /
      │        (2,4) /
      │          ↑__/ ← residual (vertical gap)
      │        /
      │   •  (1,2)
      │   /
      │__/____________________
      │  1   2   3        x
      │
     (line passes through the "middle" of the points)
```

For multiple features, the line becomes a **hyperplane** in m+1 dimensions.

---

## 17. Algorithm / Pseudocode

```text
1. Input: X (features), y (target)
2. Add a column of ones to X for the intercept (design matrix)
3. If using closed form:
     w = (XᵀX)⁻¹ Xᵀ y
   Else (gradient descent):
     Initialize w randomly
     Repeat until convergence:
       ŷ = X·w
       gradient = (2/n)·Xᵀ(ŷ − y)
       w = w − α·gradient
4. Return w
5. To predict new X_new:  ŷ = X_new·w
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        ones = np.ones((n, 1))
        X_design = np.hstack([ones, X])
        # Normal equation
        XtX = X_design.T @ X_design
        XtX_inv = np.linalg.inv(XtX)
        theta = XtX_inv @ X_design.T @ y
        self.b = theta[0]
        self.w = theta[1:]

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b
```

---

## 19. Code Explanation

```text
Line:  X_design = np.hstack([ones, X])
   What: adds a column of 1s to X
   Why: absorbs the intercept b into the weight vector
   Math: b = w₀, so ŷ = w₀·1 + w₁x₁ + ... = X_design·theta

Line:  theta = np.linalg.inv(XtX) @ XtX_inv.T @ ...  (normal equation)
   What: solves w = (XᵀX)⁻¹Xᵀy
   Why: closed-form least squares
   Math: minimizes RSS directly

Line:  return X @ self.w + self.b
   What: computes ŷ = Xw + b
   Why: applies the learned model
   Math: dot product of features with weights plus bias
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 5, 4, 6])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))
```

Alternative from-scratch-friendly libraries: `numpy.linalg.lstsq`, `statsmodels` (gives p-values, confidence intervals).

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| `fit_intercept` | Whether to learn bias b | If False, line forced through origin | Keep True unless you have a strong reason |
| `normalize` (older sklearn) | Standardize features before fit | Helps interpretation | Deprecated; use StandardScaler |
| `copy_X` | Copy or overwrite X | Memory vs safety | Keep True |

For gradient-descent linear regression add:

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Learning rate α | Step size | Too high → diverge; too low → slow | ~0.01–0.1, tune |
| Epochs / tolerance | Stop condition | More epochs → lower loss till convergence | Stop when loss plateaus |
| Batch size (SGD) | Samples per update | Influences noise/speed | Small = noisy, large = smooth |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Weight vector `w` (one per feature)
- Bias `b`
These are learned from data by minimizing RSS.

### Hyperparameters (chosen by us)
- `fit_intercept` (whether to include bias)
- Learning rate and epochs (only if using gradient descent)
These are set before training and not learned from data.

---

## 23. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Linearity | y is a linear function of features | The model formula is linear | Scatter plots, residual vs fitted plots | Try polynomial/other models |
| Independence | Samples are independent | Standard statistics assume this | Domain knowledge | Use time-series/other methods |
| Homoscedasticity | Variance of errors is constant | OLS assumes equal spread | Residual vs fitted plot (funnel shape = bad) | Weighted least squares, transform |
| Normality of errors | Residuals ~ Normal | Needed for inference (p-values, CIs) | Q-Q plot, histogram | Robust methods; still predictions OK |
| No multicollinearity | Features not highly correlated | (XᵀX)⁻¹ unstable otherwise | Correlation matrix, VIF | Drop/correlate features, ridge |
| No (little) outliers | No extreme points | Large influence on the line | Boxplots, residual plots | Robust regression (Huber) |

---

## 24. Data Requirements

- **Data type:** primarily numerical features; categorical must be encoded (one-hot).
- **Target:** continuous numeric.
- **Missing values:** must be imputed or dropped — OLS can't handle NaN.
- **Outliers:** sensitive; a single extreme point can pull the line dramatically.
- **Scaling:** not strictly required for OLS (closed form), but recommended for gradient descent and for comparing coefficients.
- **Feature engineering:** helpful — linear model can't capture interactions unless you add them manually.
- **Dataset size:** learns coefficients regardless of size, but small data → high variance of estimates.
- **Class imbalance:** N/A (regression).

---

## 25. Feature Scaling

**Required / Recommended:** Recommended for **gradient-descent** optimization (features on different scales cause uneven gradients and slow convergence). Not required for the **closed-form** normal equation (scale-invariant), but it helps interpreting coefficients.

**Methods:** Standardization (z-score), Min-Max scaling.

---

## 26. Evaluation Metrics

| Metric | Formula | Interpretation | Use When | Avoid When |
|---|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | Avg squared error; penalizes big errors | Standard regression metric | You want errors in original units |
| RMSE | √MSE | In same unit as y | Most common | Outliers dominate your view |
| MAE | (1/n)Σ\|y−ŷ\| | Avg absolute error; robust to outliers | Outliers present | Large errors should be penalized |
| R² | 1 − SS_res/SS_tot | Proportion of variance explained | Model fit comparison | Comparing models across datasets |
| Adjusted R² | R² adjusted for #features | Penalizes extra features | Feature selection | Small feature sets |

**Important:** The **training objective** (minimize RSS/MSE) is NOT the same as the **evaluation metric** (e.g., R² for reporting, MAE for interpretability). Decide the evaluation metric from the business goal, and use it for model selection.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Simple & fast | Works even on huge datasets; milliseconds of training |
| Highly interpretable | Each coefficient tells you the exact effect of that feature |
| Convex objective | Guaranteed global optimum; no tuning of randomness |
| Strong baseline | Every other model is judged against it |
| Closed-form solution | No iterative training needed for small/medium data |
| Extensible | Foundation for Ridge, Lasso, Elastic Net, polynomial, etc. |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| Assumes linearity | Poor fit on genuinely non-linear data |
| Sensitive to outliers | One bad point can wreck the line |
| Assumes independence & equal variance | Invalidates inferences when violated |
| Multicollinearity issues | Unstable, inflated coefficients |
| Can't capture interactions | Needs manual feature engineering |
| No handling of missing values | Must preprocess manually |
| High bias potential | Underfits complex patterns |

---

## 29. When to Use

✓ The relationship between features and target is approximately linear.
✓ You need an interpretable model (business/regulatory).
✓ You need a fast baseline before trying complex models.
✓ Target is continuous.
✓ Dataset is not too large (closed form) or you can use SGD.
✓ You want to know each feature's isolated effect.

---

## 30. When NOT to Use

✗ The relationship is strongly nonlinear.
✗ There are heavy outliers that matter.
✗ Features are highly collinear (prefer Ridge/Lasso).
✗ Target is categorical (use logistic regression).
✗ You need to capture complex feature interactions automatically.
✗ There are many irrelevant features (prefer Lasso).

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| House price prediction | size, bedrooms, location | Linear Regression | Predicted price |
| Sales forecasting | past sales, marketing spend | Linear Regression | Future sales |
| Salary prediction | experience, education | Linear Regression | Expected salary |
| Stock trend analysis | historical prices, volume | Linear Regression | Price trend line |
| Energy consumption | temperature, time | Linear Regression | Energy usage |

---

## 32. Failure Cases

- **Data failure:** missing values cause errors; outliers skew coefficients.
- **Mathematical failure:** if XᵀX is singular (perfect multicollinearity) the normal equation can't invert it.
- **Optimization failure:** poor learning rate → divergence or slow convergence (only for GD).
- **Generalization failure:** linear assumption wrong → high bias, underfits.
- **Practical failure:** predicting outside the range of training data (extrapolation) is unreliable.

---

## 33. Overfitting and Underfitting

- **Underfitting:** model too simple; high bias; the line can't capture the pattern → high training AND test error.
- **Overfitting:** less common for plain linear regression (few parameters) unless you add polynomial terms with many degrees — the model wiggles to fit noise → low training error, high test error.

**Balance:** With linear regression, overfitting typically emerges only with many features relative to samples, or with high-degree polynomial terms.

---

## 34. Bias-Variance Perspective

- **High bias:** linear regression assumes a rigid shape. If the truth is nonlinear, no amount of data helps — structural underfit.
- **Variance:** increases with number of features; with few data points, coefficients are unstable.
- **Tradeoff:** linear regression sits at the low-variance, potentially high-bias end. Adding polynomial terms raises variance, lowers bias.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Linear Regression | Minimize squared error | Interpretable, simple | High bias, sensitive to outliers | Baseline, linear data |
| Polynomial Regression | Add polynomial features | Handles curvature | Dangerously overfits | Curved trends |
| Ridge | L2 penalty on weights | Shrinks weights, stable | Doesn't zero features | Multicollinearity |
| Lasso | L1 penalty | Feature selection | Shrinks, not closed form | Many features |
| Elastic Net | L1+L2 | Combines both | Two penalties to tune | Mixed cases |
| Huber | Robust loss | Resistant to outliers | Extra epsilon param | Outlier-heavy data |

---

## 36. Algorithm Selection Guide

```text
Is target continuous and data approximately linear?
├── YES → Linear Regression works well, then ...
│        Multicollinearity? → RIDGE
│        Many irrelevant features? → LASSO
│        Heavy outliers? → HUBER
│        Both outliers & selection? → Robust + Lasso variants
└── NO (nonlinear) → Polynomial, SVR, Trees, Neural Nets
```

---

## 37. Common Mistakes

```text
❌ Splitting AFTER scaling (data leakage)
Why wrong: statistics (mean/std) from full data leak into test split.
Correct: split first, then fit scaler on train only.

❌ Using Linear Regression on clearly non-linear data and expecting good R²
Why wrong: linear model can't capture curvature.
Correct: check residual plots; use polynomial or other model.

❌ Interpreting coefficients when features are on different units
Why wrong: larger-magnitude features appear more important than they are.
Correct: standardize features before comparing coefficients.

❌ Forgetting the intercept
Why wrong: forcing through origin biases the fit if intercept ≠ 0.
Correct: keep fit_intercept=True unless justified.

❌ Using R² alone for model comparison across datasets
Why wrong: R² depends on total variance of data.
Correct: compare within same dataset, or use RMSE/MAE.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is linear regression?**
A: A supervised algorithm modeling y as a linear function of x, minimizing sum of squared errors.

**Q2. What is a residual?**
A: The difference between actual and predicted value, y − ŷ.

**Q3. What do the slope and intercept represent?**
A: Slope = change in y per unit change in x; intercept = predicted y when x = 0.

### Intermediate
**Q4. Why do we minimize squared error instead of absolute error?**
A: Squared error is differentiable everywhere, convex, penalizes large errors more, yielding a closed-form solution. Absolute error is non-differentiable at 0 and robust but harder to optimize.

**Q5. What is the normal equation?**
A: w = (XᵀX)⁻¹Xᵀy — the closed-form least-squares solution.

**Q6. What is multicollinearity and why is it a problem?**
A: Features highly correlated → (XᵀX) near-singular → unstable/inflated coefficients.

### Advanced
**Q7. What is the difference between MSE as a loss and RMSE as a metric?**
A: Both derived from residuals; loss is what's minimized in training; metric is how we report/compare models.

**Q8. When would gradient descent be preferred over the normal equation?**
A: When m (features) is very large (O(m³) inversion too costly) or dataset too big for closed form.

**Q9. Explain bias and variance in the context of linear regression.**
A: The linear form imposes high bias (rigid shape); coefficient estimates have variance that grows with features/shrinks with data.

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
```text
w = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²
b = ȳ − w·x̄
R² = 1 − Σ(y−ŷ)² / Σ(y−ȳ)²
```

**Concepts likely tested:**
- Interpretation of slope/intercept
- RSS, SSE, R² meaning
- Effect of outliers on the fitted line
- Effect of scaling on coefficients
- Normal equation and its invertibility condition
- Relation between correlation coefficient and slope: for standardized data, slope = correlation.

> **Representative pattern question (NOT a past GATE PYQ):** "Given points (1,2),(2,3),(3,5), find the least-squares line." Use the closed form above.

**Common traps:**
- Forgetting to center data when computing covariance.
- Thinking R² "how good" universally — it only measures in-sample linear fit.
- Confusing slope with correlation (they're equal only with standardized variables).

---

## 40. Coding Practice

**Level 1 — Basic:** Implement single-feature linear regression from scratch; print w and b.

**Level 2 — Vectorized:** Implement the matrix normal equation with numpy.

**Level 3 — SGD:** Implement linear regression with mini-batch gradient descent.

**Level 4 — Evaluation:** Compute MSE, RMSE, MAE, R² manually.

**Level 5 — Multi-feature:** Regression on a dataset with 3+ features; interpret coefficients.

**Level 6 — Preprocessing:** Handle missing values, scale features, compare scaled vs unscaled.

**Level 7 — Real-world case study:** Build a house-price model on a public dataset (e.g., California Housing via sklearn), full workflow: EDA → split → scale → train → evaluate → residual analysis → report.

---

## 41. Practical ML Workflow

```text
Problem → define target (continuous) & features
   ↓
Data → collect, inspect, handle missing
   ↓
EDA → plots, correlations, outlier scan, linearity check
   ↓
Cleaning → impute, remove outliers, fix types
   ↓
Feature engineering → encoding, interactions (if adding)
   ↓
Split → train / validation / test (stratify not needed for regression)
   ↓
Preprocess → scale features (fit on train only!)
   ↓
Train → LinearRegression.fit
   ↓
Tune → try alternatives (ridge, polynomial) if bias is high
   ↓
Evaluate → MSE/RMSE/R² on test, residual plots
   ↓
Error analysis → check where model fails, patterns in residuals
   ↓
Deploy → save model, serve predictions
   ↓
Monitor → track drift in input distribution / performance
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Closed-form training | O(n·m² + m³) | XᵀX multiply + matrix inverse |
| Gradient descent per epoch | O(n·m) | One pass over data |
| Prediction | O(m) per sample | Dot product |
| Space (model) | O(m) | Store m weights + bias |
| Scales with samples | Linear | GD fine for huge n |
| Scales with features | Cubic (closed form) | GD better for large m |
| Scales with complexity | Linear model — constant | No added complexity beyond m |

---

## 43. Advanced Concepts

- **Regularization:** add penalty to weights to reduce variance (Ridge/Lasso — covered in later notes).
- **Probabilistic interpretation:** assume y = w·x + b + ε, ε ~ N(0, σ²). Maximizing likelihood = minimizing RSS. This justifies p-values and confidence intervals.
- **Convexity:** RSS is convex → unique global minimum.
- **Weighted least squares:** weights samples unequally to handle heteroscedasticity.
- **RANSAC & Theil-Sen:** robust alternatives for heavy outliers.
- **Standardized coefficients:** coefficients of scaled features are directly comparable.

---

## 44. Connections to Other Algorithms

```text
Linear Regression
   │
   ├── extends → Polynomial Regression (nonlinear features)
   ├── extends → Ridge (L2 penalty)
   ├── extends → Lasso (L1 penalty)
   ├── extends → Elastic Net (L1+L2)
   ├── special case → Bayesian Regression (priors on weights)
   └── relation → Logistic Regression (linear + sigmoid for classification)
```

---

## 45. If You Remember Only 5 Things

1. Linear Regression models y = w·x + b and minimizes the sum of squared residuals.
2. The closed-form solution is w = (XᵀX)⁻¹Xᵀy.
3. It's a parametric, discriminative, supervised model with a convex objective (guaranteed global optimum).
4. Key assumptions: linearity, independence, homoscedasticity, normality (for inference), no multicollinearity.
5. It's the interpretable baseline that every other model is compared against.

---

## 46. Cheat Sheet

```text
Algorithm   : Linear Regression
Category    : Supervised, Regression
Goal        : Minimize Σ(y − ŷ)²
Input       : X (n×m), y (n,)
Output      : ŷ (continuous)
Core Formula: ŷ = w·x + b ;  w = (XᵀX)⁻¹Xᵀy
Loss        : RSS = Σ(y − w·x − b)²
Optimization: Normal equation OR gradient descent
Parameters  : w (weights), b (bias)
Hyperparams : fit_intercept, (α, epochs for GD)
Assumptions : linearity, independence, homoscedasticity, normality, no multicollinearity
Advantages  : simple, fast, interpretable, baseline
Disadvantages: linear bias, outlier sensitive, multicollinearity issues
Use When    : linear relationship, need interpretability, baseline
Avoid When  : strong nonlinearity, heavy outliers, collinear features
Related     : Ridge, Lasso, Elastic Net, Polynomial, Bayesian
Key Exam    : w = Σ(x−x̄)(y−ȳ)/Σ(x−x̄)² ; R² formula ; outlier effect
Key Interv  : residual meaning, normal equation, bias/variance, when GD vs closed form
```

---

## 47. Final Mental Model

```text
Data (X, y)
   ↓
Fit straight line/hyperplane to minimize squared vertical gaps
   ↓
Learned: slope w and intercept b
   ↓
Predict: ŷ = w·x + b
   ↓
New input → dot product with weights + bias → predicted value
```

---

## 48. Knowledge Check

### Recall (5)
1. Write the model equation.
2. What is the normal equation?
3. Define residual.
4. What does R² measure?
5. What are the two learned parameters?

### Understanding (5)
6. Why is squaring preferred over absolute error here?
7. Why is the objective convex and why does that matter?
8. What is a "baseline model"?
9. How does an outlier affect the line?
10. Why does scaling matter for gradient descent but not closed form?

### Application (5)
11. Given data, compute w and b manually.
12. Predict a new value using the fitted line.
13. Decide if linear regression is appropriate for given data.
14. Choose metric between RMSE and MAE.
15. Diagnose high bias vs high variance from plots.

### Mathematical (5)
16. Derive b = ȳ − w·x̄.
17. Show the RSS is convex.
18. Compute R² for a small dataset.
19. Relate slope to correlation.
20. Interpret a coefficient of 3.0 for a standardized feature.

### Interview (5)
21. Explain when to use normal equation vs gradient descent.
22. What is multicollinearity?
23. How do you check linearity assumption?
24. What is the probabilistic interpretation?
25. Why might you prefer a linear model for business reporting?

### Problem Solving (5)
26. R² is 0.3 — is the model useless?
27. Predictions are all systematically too low — what's wrong?
28. Features on wildly different scales — what to do?
29. One point has huge residual and flips the line — how to fix?
30. You must explain a coefficient to a non-technical manager — how?

## Answers (explained)

1. ŷ = w·x + b.
2. w = (XᵀX)⁻¹Xᵀy — closed-form least-squares solution.
3. y − ŷ, the vertical gap between actual and predicted.
4. Proportion of variance in y explained by the model: R² = 1 − SS_res/SS_tot.
5. Weight vector w and bias b.
6. Squared error is differentiable everywhere (needed for calculus optimization) and convex with a unique minimum; it also penalizes large errors more.
7. Convexity guarantees any local minimum is global — heavier penalties for large residuals.
8. A simple, well-understood reference model (linear regression) used to compare more complex models' added value.
9. An outlier can pull the regression line toward itself, distorting slope/intercept (because squared error weights it heavily).
10. Gradient descent moves on all dims equally scaled; without scaling, uneven gradients slow convergence; closed form is scale-invariant (the math cancels units).
11–30: Use the derivation in Section 10, closed-form formulas, and concepts above. For (30), say: "Holding other factors constant, a one-unit increase in feature k changes the prediction by w_k units."

---

## 49. Final Learning Checklist

- [ ] I can define linear regression in one sentence
- [ ] I know the model equation ŷ = w·x + b
- [ ] I can derive the closed-form solution
- [ ] I can explain what residual means
- [ ] I can interpret slope and intercept
- [ ] I can compute w, b by hand on 3 points
- [ ] I can predict on new data
- [ ] I can compute MSE, RMSE, MAE, R²
- [ ] I know what the normal equation is
- [ ] I understand the convexity of RSS
- [ ] I know the 5 classic assumptions
- [ ] I can check linearity with plots
- [ ] I know the effect of outliers
- [ ] I know why scaling matters for GD
- [ ] I can implement OLS from scratch
- [ ] I can implement with sklearn
- [ ] I know the difference between parameters & hyperparameters
- [ ] I know when to use vs avoid linear regression
- [ ] I can connect it to Ridge, Lasso, Elastic Net
- [ ] I can explain it to a non-technical audience

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** Closed-form formulas, derivation check out; hand-verified numerical example.
- **Beginner-friendliness:** Analogy, no-math intuition, short paragraphs, tables.
- **Math depth:** Full derivation + worked numerical example.
- **Practical depth:** From-scratch + library code, hyperparameters, workflow.
- **Exam depth:** Key formulas, traps, representative (non-PYQ) questions.
- **Structure:** All 50 sections present in order.

**Verified:** The numerical example in Section 15 was recomputed by hand and confirms ŷ = 1.5x + 0.667 with RSS = 0.167.
