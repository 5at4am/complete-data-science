# 04. Lasso Regression

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Lasso Regression (Least Absolute Shrinkage and Selection Operator) |
| Category | Supervised Learning |
| Type | Regression |
| Parametric / Non-parametric | Parametric |
| Generative / Discriminative | Discriminative |
| Main Objective | Fit a linear model with an L1 penalty that shrinks coefficients toward zero AND drives some to exactly zero (feature selection) |
| Input | Feature matrix X (n×m), target y (continuous) |
| Output | Continuous prediction ŷ; sparse coefficient vector (some weights = 0) |
| Core Idea | Add L1 penalty (sum of absolute weights) to least squares; the kink geometry forces irrelevant coefficients to exactly zero |
| Typical Use Cases | Feature selection in high dimensions, sparse models, interpretable models with many variables |

---

## 02. One-Line Definition

### Beginner Definition
Lasso is like a regression line that not only draws the best fit but also "turns off" unimportant features completely (sets their weight to zero), leaving only the useful ones.

### Technical Definition
Lasso minimizes the residual sum of squares plus an L1 penalty, λ·Σ|wⱼ|, which produces a sparse solution where many coefficients become exactly zero, performing simultaneous shrinkage and feature selection.

---

## 03. Intuition

Suppose you have 50 measurements and only 3 actually matter. Linear regression uses all 50, Ridge shrinks all 50 (but keeps all), and clutter stays.

Lasso's penalty is different: instead of discouraging *large* coefficients by their square, it charges by their *absolute* size. Because the penalty's shape has "sharp corners" (think of a diamond), the optimal solution often lands exactly on a corner — which means a coefficient gets set to 0.

Effect: Lasso automatically **selects** the most important features and drops the rest. This is huge for interpretability and for fighting the curse of dimensionality.

Step-by-step:
1. Start with all features.
2. Apply L1 penalty with strength λ.
3. As λ grows, weak features' coefficients hit exactly 0 one by one.
4. The remaining nonzero coefficients are your "selected" features.

---

## 04. Problem It Solves

**Problem:** When there are many features (especially more features than samples), you need:
1. A solvable model (OLS fails when p > n).
2. Feature selection (which features actually matter?).
3. Interpretability (a model you can explain).

**Example:** Predicting disease from 1000s of genes — you can't use all; you want the few genes truly associated, with others set to 0.

Why useful: Lasso returns a clean, sparse, interpretable model — it tells you *which variables matter* while Ridge cannot.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning
│   └── Regression
│       ├── Linear Models
│       │   ├── Linear Regression
│       │   ├── Ridge (L2)
│       │   ├── Lasso (L1)             ← YOU ARE HERE
│       │   ├── Elastic Net (L1+L2)
│       │   └── Bayesian / Huber / Quantile
├── (Lasso also = embedded feature-selection method)
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| L1 norm | Size via sum of absolute values | ‖w‖₁ = Σ\|wⱼ\| |
| Sparse solution | Many coefficients exactly zero | Only subset of features used |
| Feature selection | Choosing which features matter | Automatic via L1 zeroing |
| Shrinkage | Pulling coefficients toward 0 | All coefficients reduced |
| Soft-thresholding | The L1 solution operator | Shrinks value and zeros small ones |
| Subgradient | Generalization of gradient | Needed because |w| isn't differentiable at 0 |

---

## 07. Input and Output

**Input:** X (n×m) numeric, y continuous.
**Output:** prediction ŷ; sparse coefficient vector (many = 0).

**Parameters learned:** w (sparse weights), b (intercept).

**Hyperparameters:** α (λ) — L1 penalty strength (main). Possibly `max_iter`, `tol`.

---

## 08. Mathematical Foundation

Lasso objective:

```text
Minimize  J(w) = Σᵢ(yᵢ − ŷᵢ)² + λ·Σⱼ |wⱼ|
```

The L1 term is non-differentiable at 0, which is why Lasso needs specialized solvers (coordinate descent, LARS) rather than plain gradient descent, and why it can produce exact zeros.

**Notation:**
- `λ ≥ 0` = regularization strength
- `|wⱼ|` = absolute value of coefficient j
- `n` = samples, `m` = features
- `w` = coefficient vector

**Required math:** OLS, absolute value, subgradients, coordinate descent (for the algorithm).

---

## 09. Core Formula

### Lasso Objective

```text
J = RSS + λ·Σⱼ |wⱼ| = Σᵢ(yᵢ − ŷᵢ)² + λ·Σⱼ |wⱼ|
```

#### Meaning
Minimize errors while discouraging coefficients — and forcing some to zero.

#### Symbols
- `RSS` = sum of squared residuals
- `λ` = penalty strength
- `Σⱼ|wⱼ|` = L1 norm of coefficients
- `wⱼ` = j-th coefficient

#### Intuition
λ=0 → OLS. As λ↑, coefficients shrink; the smallest become exactly 0 (unlike Ridge). Larger λ = sparser model.

#### Example
w = [3, 0.5, 0]. RSS = 5, λ = 1:
- Penalty = 1·(|3|+|0.5|+|0|) = 3.5
- Objective = 5 + 3.5 = 8.5

If you set w₂ to 0: RSS may rise to 5.4, penalty = 1·(3+0+0)=3 → objective = 8.4 (better!). The 0.5 coefficient wasn't worth keeping — Lasso drops it.

---

### Soft-Thresholding (single-coordinate update)

```text
wⱼ ← sign(zⱼ)·max(0, |zⱼ| − λ)
```
where `zⱼ` is the OLS-style solution for coordinate j given others fixed.

#### Meaning
Each coefficient update: shrink by λ; if it would cross zero, set to exactly 0.

#### Symbols
- `zⱼ` = current unpenalized value for coordinate j
- `sign(zⱼ)` = +1 or −1
- `|zⱼ|` = absolute value
- `max(0, …)` = clamp at 0
- `λ` = penalty

#### Intuition
This is why Lasso zeros features: `max(0, |z|−λ)` returns exactly 0 whenever |z| ≤ λ.

#### Example
z = 0.4, λ = 1: |z|−λ = −0.6 → max(0,−0.6)=0 → w = 0 (feature dropped). z = 2.5, λ = 1: |z|−λ=1.5 → sign=+ → w = +1.5 (kept but shrunk).

---

## 10. Derivation

**Step 1 — Lasso objective (assume centered data, excluding intercept):**

```text
minimize  (1/2)Σᵢ(yᵢ − Σⱼwⱼxᵢⱼ)² + λΣⱼ|wⱼ|
```

(The 1/2 is a convenience) may be used.

**Step 2 — Consider a single coordinate wⱼ**, holding all others (call residual rᵢ, ignoring j's contribution) fixed:

```text
J = (1/2)Σᵢ(yᵢ − wⱼxᵢⱼ − (others))² + λ|wⱼ| + (const)
```

**Step 3 — Let zⱼ = (Σxᵢⱼ(yᵢ−others))/Σxᵢⱼ²** be the OLS value for wⱼ ignoring penalty. The subgradient condition gives:

```text
zⱼ − wⱼ + λ·sign(wⱼ) = 0   (for wⱼ>0)
zⱼ − wⱼ − λ·sign(wⱼ) = 0   (for wⱼ<0)
|zⱼ| ≤ λ  ⇒  wⱼ = 0
```

**Step 4 — Solve to get soft-thresholding:**

```text
wⱼ = sign(zⱼ)(|zⱼ| − λ)₊   where (u)₊ = max(u, 0)
```

**Step 5 — Coordinate descent** sweeps all coordinates applying this update until convergence. This is the standard Lasso algorithm (sklearn `lasso_path`, etc.).

> (Optional deeper result: the L1 kink at 0 — unlike L2's smooth parabola — is what allows exact zeros.)

---

## 11. How the Algorithm Works

```text
Input (X, y), choose λ
    ↓
Center/scale data
    ↓
Initialize w (e.g., all zeros)
    ↓
Coordinate descent loop:
    for each coordinate j:
        compute zⱼ (OLS value for j, others fixed)
        wⱼ = soft-threshold(zⱼ, λ)
    ↓
Repeat until convergence (coefficients stop changing)
    ↓
Final sparse model
    ↓
Predict ŷ = Xw + b
```

---

## 12. Training Process

**Pre-training:** choose λ (tune by CV); standardize features.

**During training:** iterate coordinate descent; each step solves one coordinate given others, applying soft-thresholding (which zeros small coefficients).

**What is learned:** a sparse weight vector — most zero, few nonzero.

**Stopping:** coefficients converge (change below tolerance).

**Final model:** the sparse coefficient set (the "selected" features) and intercept.

---

## 13. Objective Function / Loss Function

```text
Objective = RSS + λ·‖w‖₁
```

Why L1? Because its non-smooth geometry produces exact zeros → feature selection. This is the key difference from Ridge's L2.

- λ=0 → pure OLS.
- λ large → sparse, all/almost all zero.
- Training objective includes penalty; evaluation uses plain R²/MSE (no penalty).

---

## 14. Optimization

**Method:** coordinate descent (standard) or LARS; NOT plain gradient descent (non-differentiable at 0).

**Subgradient** of |wⱼ| is sign(wⱼ) (a set for wⱼ=0: any value in [−1,1]).

**Coordinate descent update (soft-threshold):**
```text
wⱼ = sign(zⱼ)·max(0, |zⱼ| − λ)
```

**Convergence:** convex objective → global minimum for fixed λ, though path may depend slightly on coordinate order (still converges to global optimum).

**Feature selection emerges:** as iterations proceed, coefficients whose magnitude ≤ λ get set to exactly 0.

---

## 15. Complete Numerical Example

Data: 2 samples, 2 features.
- Sample 1: x = [2, 0], y = 4
- Sample 2: x = [0, 2], y = 4

**Step 1 — Compute OLS-style coordinates (features orthogonal, so each is independent).**

For feature 1 (others fixed):
```text
z₁ = Σ x₁·y / Σ x₁² = (2·4 + 0·4) / (2² + 0²) = 8/4 = 2.0
```
For feature 2:
```text
z₂ = Σ x₂·y / Σ x₂² = (0·4 + 2·4) / (0² + 2²) = 8/4 = 2.0
```

**Step 2 — Apply soft-threshold with λ = 1:**
```text
w₁ = sign(2)·max(0, 2 − 1) = 1·1 = 1.0
w₂ = sign(2)·max(0, 2 − 1) = 1·1 = 1.0
```

Both kept (shrunk from 2 → 1).

**Step 3 — Try larger λ = 3:**
```text
w₁ = sign(2)·max(0, 2−3) = 1·max(0,−1) = 0   ← feature 1 dropped
w₂ = sign(2)·max(0, 2−3) = 0                   ← feature 2 dropped
```

Both dropped at λ=3 — data had both contributing, but penalty dominates.

**Step 4 — Mixed case: say z₁=3, z₂=1, λ=2:**
```text
w₁ = max(0, 3−2) = 1   (kept)
w₂ = max(0, 1−2) = 0   (dropped — too weak)
```

**VERIFIED EXAMPLE** — hand-verified with soft-thresholding. Shows exactly how Lasso zeros weak coefficients.

---

## 16. Visual Explanation

```text
L1 constraint (diamond):                      w₂
   |w₁| + |w₂| ≤ t                             │
   has sharp CORNERS on the axes               │
                                               ● (corner → w₂=0)
                                               │
                                ───────●───────│────  w₁
Diamond touches least-squares     corners     │
contour at a corner → w₂ set to 0             │

L2 constraint (circle): no corners
→ solution rarely on an axis → Ridge keeps all
```

```text
Compare:
Ridge boundary: ○ circle      → shrink only
Lasso boundary: ◇ diamond     → shrink + zero corners
```

---

## 17. Algorithm / Pseudocode

```text
1. Input: X, y, λ
2. Center & scale X, center y
3. Initialize w = 0
4. Repeat until convergence:
     for j in 1..m:
       r = y - (X·w excluding column j)
       zⱼ = (X[:,j]ᵀ·r) / (X[:,j]ᵀ·X[:,j])
       wⱼ = sign(zⱼ)·max(0, |zⱼ| − λ)
5. Return sparse w, intercept b
6. Predict: ŷ = Xw + b
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

class LassoRegression:
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
        n, m = Xc.shape
        w = np.zeros(m)
        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(m):
                residual = yc - Xc @ w + w[j] * Xc[:, j]
                zj = (Xc[:, j] @ residual) / (Xc[:, j] @ Xc[:, j])
                w[j] = np.sign(zj) * max(0.0, abs(zj) - self.alpha)
            if np.max(np.abs(w - w_old)) < self.tol:
                break
        self.w = w
        self.b = y_mean - X_mean @ w

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return X @ self.w + self.b
```

---

## 19. Code Explanation

```text
Line:  residual = yc - Xc@w + w[j]*Xc[:,j]
   What: removes j's contribution to get residual w/o feature j
   Why: coordinate descent needs residual "as if wⱼ were free"
   Math: standard form for zⱼ

Line:  zj = (Xc[:,j]@residual)/(Xc[:,j]@Xc[:,j])
   What: OLS value for coordinate j
   Why: base before soft-threshold
   Math: zⱼ = Σxᵢⱼrᵢ/Σxᵢⱼ²

Line:  w[j] = np.sign(zj)*max(0.0, abs(zj)-self.alpha)
   What: soft-thresholding
   Why: THE Lasso step — shrinks and zeros
   Math: sign(z)(|z|−λ)₊
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = np.random.RandomState(42).randn(100, 20)
w_true = np.zeros(20)
w_true[[0, 3]] = [2.5, -1.5]
y = X @ w_true + np.random.RandomState(0).randn(100) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

model = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("Coefficients:", model.named_steps['lasso'].coef_)
print("MSE:", mean_squared_error(y_test, y_pred))

params = {'lasso__alpha': np.logspace(-3, 1, 50)}
grid = GridSearchCV(Lasso(), params, cv=5)
grid.fit(X_train, y_train)
print("Best alpha:", grid.best_params_)
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| α (λ) | L1 penalty strength | Higher → sparser | Tune via CV (log-spaced) |
| `max_iter` | Max coordinate-descent passes | Convergence | Increase if warning |
| `tol` | Convergence tolerance | Precision | Default |

**Too low α:** close to OLS — no sparsity, possible overfit. **Too high α:** too sparse — drops useful features. **Tune:** CV over log-spaced α.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Sparse weight vector w (many exactly 0)
- Intercept b

### Hyperparameters (chosen)
- α (λ) — determines how much sparsity
- `max_iter`, `tol`

---

## 23. Assumptions

| Assumption | What | Why | Check | If violated |
|---|---|---|---|---|
| Linear relationship | Linear in features | Model form | Residual plots | Polynomial / other |
| Independence | Samples independent | Statistics | Domain | Time-series |
| Homoscedasticity | Constant variance | Stable | Residual plot | Weighted LS |
| Feature scale comparable | Fair penalty | L1 treats all equal magnitude | — | Standardize |
| Sparsity ground truth | Few features matter | Lasso is good when true model is sparse | Domain knowledge | Elastic Net better if many correlated |

---

## 24. Data Requirements

- **Type:** numeric features; categorical encoded.
- **Missing:** impute/remove.
- **Outliers:** squared-loss sensitive; consider robust variant.
- **Scaling:** required (fair L1 penalty).
- **Dataset size:** works with p > n (sparse assumption helps).
- **High-dim:** a primary use case (e.g., genomics).

---

## 25. Feature Scaling

**Required:** Yes — the L1 penalty sums absolute weights regardless of feature scale, so features with larger magnitudes get unfairly penalized. Standardize all features first.

---

## 26. Evaluation Metrics

Same as linear regression (MSE, RMSE, MAE, R²).

**Training vs evaluation:** training minimizes RSS + λ‖w‖₁; evaluation uses unpenalized test metrics. Report plain RMSE/R² on held-out data when comparing.

---

## 27. Advantages

| Advantage | Why matters |
|---|---|
| Feature selection | Automatically zeros irrelevant features |
| Interpretable | Sparse model — only important variables |
| Handles p > n | Solvable and sparse in high dimensions |
| Reduces variance | Removes noise features |
| Computational (coordinate descent) | Fast even with many features |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| Arbitrarily picks among correlated features | With a group of correlated features, picks one, unstable |
| Not stable with collinear groups | Elastic Net handles this better |
| Sensitive to λ choice | Needs careful CV |
| Shrinks selected coefficients too | Nonzero coefficients are also biased downward |
| Can't handle p>n well when true model not sparse | Poor performance |

---

## 29. When to Use

✓ Many features, few truly matter (sparse truth).
✓ Need automatic feature selection.
✓ High-dimensional data (p>n).
✓ Interpretability/explainability required.
✓ You want a sparse, simple model.

---

## 30. When NOT to Use

✗ Features are highly correlated in groups (use Elastic Net).
✗ You need to keep all features (Ridge).
✗ True signal is dense (many features each small effect).
✗ Small p with clear interpretable linear model (plain OLS).
✗ Heavy outliers.

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Genomic biomarker discovery | thousands of genes | Lasso | Key genes selected |
| Credit scoring | many financial features | Lasso | Sparse risk model |
| Marketing attribution | many campaign features | Lasso | Which campaigns matter |
| Image feature selection | many pixels/features | Lasso | Important features |
| Text classification | many word features | Lasso | Key words/features |

---

## 32. Failure Cases

- **Correlated group failure:** among two near-identical features, Lasso keeps one arbitrarily.
- **Non-sparse truth:** if all features matter a little, Lasso drops most and underfits.
- **λ mis-tuned:** too small → overfit, too big → drops everything.
- **Large-magnitude feature dominance:** without scaling, one feature dominates.

---

## 33. Overfitting and Underfitting

- **Overfitting:** λ too small — no sparsity, fits noise on irrelevant features.
- **Underfitting:** λ too large — drops too many features, high bias.
- **Lasso's role:** sparsity acts as a powerful guard against overfitting (fewer used features = less variance). Tune λ to balance.

---

## 34. Bias-Variance Perspective

- L1 penalty trades bias (shrunk/dropped coefficients) for variance (fewer features, more stability).
- **Large λ:** high bias, low variance (few features).
- **Small λ:** low bias, high variance (many features).
- Optimal λ minimizes total error; effective model complexity = number of nonzero coefficients.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Ridge | L2, shrink only | Stable, keeps all | No selection | Collinear/wide |
| Lasso | L1, shrink + zero | Feature selection | Unstable in groups | Sparse selection |
| Elastic Net | L1+L2 | Groups + selection | 2 params | Correlated sparse |
| Linear Regression | No penalty | Unbiased | p>n fails | Clean data |

---

## 36. Algorithm Selection Guide

```text
Need feature selection / sparsity?
├── YES, features independent-ish → LASSO
├── YES, but correlated groups → ELASTIC NET
├── NO, keep all, handle collinearity → RIDGE
└── No penalty needed → LINEAR REGRESSION
```

---

## 37. Common Mistakes

```text
❌ Not scaling features before Lasso
Why wrong: unfair penalty; large-magnitude features dominate.
Correct: standardize first.

❌ Expecting stable selection among correlated features
Why wrong: Lasso picks arbitrarily within a correlated group.
Correct: use Elastic Net.

❌ Using Lasso when true model is dense
Why wrong: drops many small-but-real effects → underfit.
Correct: Ridge if effects are dense/small.

❌ Forgetting that selected coefficients are biased downward
Why wrong: post-selection, nonzero weights are shrunk.
Correct: optionally refit OLS on selected features.

❌ Tuning λ on training error
Why wrong: always picks λ→0.
Correct: tune via CV.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is Lasso?**
A: Linear regression with L1 penalty that shrinks coefficients and sets some to exactly 0 (feature selection).

**Q2. What is the L1 penalty?**
A: λ·Σ|wⱼ| — the sum of absolute coefficient magnitudes.

**Q3. What's the key advantage over ridge?**
A: Feature selection — it zeroes out unimportant features.

### Intermediate
**Q4. Why does Lasso zero coefficients but Ridge doesn't?**
A: The L1 constraint (diamond) has sharp corners on axes where a coefficient can be exactly 0; L2 (circle) has no corners.

**Q5. How is Lasso solved (vs gradient descent)?**
A: Usually coordinate descent with soft-thresholding because |w| isn't differentiable at 0.

**Q6. Lasso vs Elastic Net?**
A: Elastic Net adds L2 too, stabilizing selection among correlated features.

### Advanced
**Q7. What's soft-thresholding?**
A: w = sign(z)·max(0,|z|−λ) — the coordinate-wise Lasso update; zeros coefficients with |z|≤λ.

**Q8. Why can Lasso behave arbitrarily with correlated features?**
A: L1 selects one from a correlated group arbitrarily, giving unstable selection; Elastic Net averages over the group.

**Q9. What's the Bayesian view of Lasso?**
A: MAP estimate with a Laplace (double-exponential) prior on coefficients — its peak at 0 drives sparsity.

---

## 39. GATE / Exam Perspective

**Key formulas:**
```text
Objective: Σ(y − ŷ)² + λ Σ|wⱼ|
Soft-threshold: wⱼ = sign(zⱼ)·max(0, |zⱼ| − λ)
```

**Concepts tested:**
- L1 vs L2 penalty behavior (zero vs shrink).
- Lasso can set coefficients to exactly zero; Ridge cannot.
- Feature selection property.
- Solution methods (coordinate descent).

> **Representative pattern question (NOT a past GATE PYQ):** "Given soft-threshold rule and z=1.5, λ=2, find w." Answer: 0 (since |z|−λ negative → clamped to 0).

**Traps:**
- Confusing L1/L2 (Lasso zeros, Ridge shrinks).
- Forgetting Lasso handles p > n.
- Thinking all nonzero Lasso coefficients are "correct" — they're biased downward.

---

## 40. Coding Practice

**Level 1:** Implement soft-thresholding function.
**Level 2:** Implement coordinate-descent Lasso.
**Level 3:** Verify on data with known sparse truth (recover the nonzero features).
**Level 4:** Tune α via CV.
**Level 5:** Compare Lasso vs Ridge on correlated features (observe instability).
**Level 6:** Preprocess (scale) and observe coefficient fairness.
**Level 7:** Case study — gene-selection style (e.g., p>n synthetic), build sparse model, report selected features & performance.

---

## 41. Practical ML Workflow

```text
Problem → many features, want selection
   ↓
EDA → correlations, density of signal
   ↓
Clean → impute, handle outliers
   ↓
Encode categoricals
   ↓
Split → train/val/test
   ↓
Scale → StandardScaler
   ↓
Train → Lasso over α grid
   ↓
Tune → CV choose α (sparsity vs accuracy)
   ↓
Evaluate → RMSE/R² on test, check selected features
   ↓
Error analysis → stability of selection (repeat with different seeds)
   ↓
Deploy → save scaler + sparse model
   ↓
Monitor
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Coordinate descent/epoch | O(n·m) per pass | Each coordinate O(n) |
| Convergence passes | Depends; often ~few tens | Iterate to tolerance |
| Prediction | O(k) | Only k nonzero features matter |
| Space | O(m) | Sparse: store nonzero only |
| Scaling with m | Linear+ | Good for high-dim |

---

## 43. Advanced Concepts

- **LARS (Least Angle Regression):** efficient Lasso path algorithm.
- **Group Lasso:** selects groups of features rather than individual ones.
- **Bayesian view:** Laplace prior → Laplace MAP = Lasso.
- **Post-lasso re-estimation:** refit OLS on selected features to reduce bias.
- **Stability selection:** repeat Lasso on subsamples to identify stable features.

---

## 44. Connections to Other Algorithms

```text
Linear Regression
   └── Lasso (L1 penalty → sparsity)
        ├── Elastic Net (L1 + L2)
        ├── Group Lasso
        ├── Bayesian Regression (Laplace prior)
        └── LARS (efficient path solver)
```

---

## 45. If You Remember Only 5 Things

1. Lasso = linear regression + L1 penalty λ·Σ|wⱼ|.
2. L1's diamond "corners" set coefficients to exactly 0 → feature selection.
3. Solved by coordinate descent + soft-thresholding.
4. Great for high-dimensional, sparse-truth data; unstable for correlated groups.
5. Always scale features; tune λ by CV.

---

## 46. Cheat Sheet

```text
Algorithm   : Lasso Regression
Category    : Supervised, Regression, regularized linear
Goal        : Sparse, selectable model
Input       : X (n×m), y; λ
Output      : ŷ; sparse w
Core Formula: minimize RSS + λΣ|wⱼ|
Loss        : RSS + λ‖w‖₁
Optimization: coordinate descent + soft-threshold
Parameters  : sparse w, b
Hyperparams : α(λ), max_iter, tol
Assumptions : linear, indep, homosced, scaling, sparsity
Advantages  : selection, interpretable, p>n, low variance
Disadvantages: correlated-group instability, shrinks selected, dense-truth fails
Use When    : sparse high-dim, need selection
Avoid When  : correlated groups, dense truth
Related     : Ridge, Elastic Net, Group Lasso, LARS
Key Exam    : soft-threshold; L1 zeros vs L2 shrinks
Key Interv  : why zeros, coordinate descent, Bayesian Laplace, Elastic Net
```

---

## 47. Final Mental Model

```text
Data + λ
   ↓
Coordinate descent with soft-thresholding
   ↓
wⱼ = sign(zⱼ)·max(0, |zⱼ|−λ)
   ↓
Weak features → exactly 0 (dropped)
   ↓
Sparse model: only important features
   ↓
Predict ŷ = Xw + b
```

---

## 48. Knowledge Check

### Recall (5)
1. Write Lasso objective.
2. Write soft-threshold update.
3. What's the L1 norm?
4. Does Lasso zero coefficients?
5. What solver is typically used?

### Understanding (5)
6. Why L1 produces zeros (geometry)?
7. Why scale features?
8. Lasso vs Ridge selection?
9. What's sparse solution?
10. Why pick Lasso for p>n?

### Application (5)
11. Apply soft-threshold to given z,λ.
12. Choose λ via CV.
13. Decide Lasso vs Elastic Net.
14. Interpret a sparse model's nonzero coefficients.
15. Detect correlated-group instability.

### Mathematical (5)
16. Explain subgradient of |w|.
17. Derive soft-threshold.
18. What's Laplace prior → Lasso?
19. Why LARS is efficient?
20. How does Elastic Net fix group issue?

### Interview (5)
21. "Why L1 not L2 for selection?"
22. "What's coordinate descent?"
23. "When does Lasso fail?"
24. "Post-lasso re-estimation — why?"
25. "How do you tune α?"

### Problem Solving (5)
26. Coefficients unstable across runs — why?
27. Model chose only 1 of 2 correlated features — step?
28. Want both sparsity and stability — which model?
29. R² low, many features dropped — diagnose.
30. Need to explain selected genes — how?

## Answers (explained)
1. Σ(y−ŷ)² + λΣ|wⱼ|.
2. wⱼ = sign(zⱼ)·max(0,|zⱼ|−λ).
3. Sum of absolute values of weights.
4. Yes — exactly to 0 via soft-threshold.
5. Coordinate descent / LARS.
6. L1 constraint boundary is a diamond with corners on axes; solution touches a corner ⇒ coefficient 0.
7. Fair penalty; otherwise large-magnitude features dominate.
8. Lasso selects (zeros); Ridge shrinks (keeps all).
9. A model where most coefficients are exactly 0 (few used features).
10. It's solvable and produces a sparse, interpretable model despite more features than samples.
11–30: apply soft-threshold and concepts above. For (27): test Elastic Net. For (29): λ too big — reduce or reconsider sparsity assumption.

---

## 49. Final Learning Checklist

- [ ] I can write Lasso objective
- [ ] I understand L1 geometry (diamond vs circle)
- [ ] I know why it zeros coefficients
- [ ] I can apply soft-thresholding
- [ ] I can implement coordinate descent
- [ ] I know why to scale features
- [ ] I can tune λ via CV
- [ ] I understand the sparse-truth assumption
- [ ] I know when it's unstable (correlated groups)
- [ ] I can compare with Ridge & Elastic Net
- [ ] I know the Bayesian (Laplace) view
- [ ] I can recognize over/under-fitting from λ
- [ ] I can handle p>n
- [ ] I know about post-lasso refinement
- [ ] I can use sklearn Lasso
- [ ] I understand coefficient bias (shrunk downward)
- [ ] I know LARS & Group Lasso ecosystems
- [ ] I can interpret a sparse model
- [ ] I can apply in a full workflow
- [ ] I know when NOT to use Lasso

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** Soft-thresholding, objective, geometry verified; worked example recomputed by hand.
- **Beginner-friendliness:** Analogy, diamond/circle ASCII, short paragraphs, tables.
- **Math depth:** Derivation, subgradient, coordinate descent.
- **Practical depth:** From-scratch + sklearn, hyperparameters, workflow, sparsity handling.
- **Exam depth:** L1 vs L2, soft-threshold, non-PYQ representative questions.
- **Structure:** All 50 sections in order.

**Verified:** Section 15 worked example recomputed by hand.
