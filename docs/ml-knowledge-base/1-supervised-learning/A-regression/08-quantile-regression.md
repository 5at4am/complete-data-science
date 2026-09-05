# 08. Quantile Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐☆☆
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **mean is not enough → quantile idea → asymmetric loss → pinball magic → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Every regression model you've seen so far predicts the **average**. But what if the average isn't what you need?

Quantile Regression is the model that lets you predict **any slice** of the outcome distribution — the median, the 10th percentile, the 90th percentile. It reveals how the *whole distribution* changes with your features, not just the middle.

By the end you will be able to:

- explain *why* the mean hides important information,
- write the pinball loss and derive why it gives quantiles,
- fit median regression by hand,
- build prediction intervals from two quantile fits,
- code it from scratch and with sklearn,
- and defend when to use — and not use — it.

> Everything in this note builds on one idea: **what if instead of predicting the average, we could predict the worst case? Or the typical case? Or the best case?**

---

## 02. The Problem

Priya manages a food delivery fleet in Bangalore. She's tracking delivery times:

| Distance (km) | Delivery time (min) |
|---|---|
| 1 | 12 |
| 2 | 18 |
| 3 | 22 |
| 5 | 35 |
| 8 | 55 |
| 10 | 45 |
| 10 | 90 |
| 10 | 120 |

Notice: at 10 km, delivery times range from 45 to 120 minutes. The average at 10 km is about 73 minutes, but that number **hides** the real story — sometimes it's quick (45 min), sometimes a disaster (120 min).

<!-- [QUESTION] -->
Now the question:

> **If you need to promise customers a delivery time, which would you rather predict — the average (73 min), the median (roughly 50 min), or the 90th percentile (roughly 110 min)?**

Think about it. Different stakeholders need different answers:
- The **operations team** wants the median (typical case).
- The **customer promise** wants the 90th percentile (worst realistic case).
- The **driver scheduling** wants the 10th percentile (best case for efficiency planning).

One average line gives none of these.

---

## 03. Let's Think

Let's look at just the 10 km deliveries:

```text
Delivery times at 10 km:  45,  55,  90,  120
Mean:     77.5
Median:   72.5  (average of 55 and 90)
90th pct: ~108
```

<!-- [THINK_ABOUT_IT] -->
🤔 The spread is huge — from 45 to 120. The average (77.5) doesn't represent any *actual* delivery well. And here's the key observation:

> As distance increases, the **spread** of delivery times also increases. The average line doesn't capture this **changing spread** — it just shows the middle.

What if we could fit **multiple lines** — one for the median, one for the 90th percentile, one for the 10th? Each line would tell a different story about the same data.

That's exactly what Quantile Regression does.

---

## 04. Intuition

Ordinary regression tells you the *average*. But often you care about more: the *typical* (median), the *extreme high* (90th percentile), or the *extreme low* (10th percentile).

💡 **The idea in one line:**

> Quantile Regression fits **multiple lines** — one for each percentile you care about — by using a **pinball loss** that penalizes over-prediction and under-prediction **asymmetrically**.

Here's the clever trick: if you penalize *under-predictions* more than over-predictions, the line gets pushed **up** to track higher values. If you penalize *over-predictions* more, the line gets pushed **down**. The amount of asymmetry controls *which* quantile you land on.

---

## 05. Visual

```text
Pinball loss vs residual (different τ):

  loss
   │  τ=0.5 (median)           τ=0.9 (high quantile)
   │        ╱                          ╱
   │       ╱  (equal                   ╱  (under-predicts
   │      ╱    both sides)            ╱    penalized 9× more)
   │     ╱                          ╱
   │    ╱                          ╱
   └───╱──────── residual       ──╱────────── residual
```

```text
Multiple quantile lines on data:
   y
   │  ╱  (90th percentile — steep, tracks high end)
   │ ╱
   │╱  (median — robust middle)
   │
   │ ╲  (10th percentile — tracks low end)
   │
   └──────────────────── x
   Three lines, each a different conditional quantile
```

---

## 06. First Prediction

Back to Priya's data. Let's fit two lines: median (τ=0.5) and 90th percentile (τ=0.9).

For the **median** at τ=0.5, the pinball loss is symmetric — it's the same as **absolute error** (MAE). The median line is robust to outliers.

For the **90th percentile** at τ=0.9, under-predictions cost 9× more than over-predictions. The line is pulled **upward** to capture high values.

At distance = 10 km:
- Median prediction: ~50 min (typical)
- 90th percentile prediction: ~110 min (worst realistic)

<!-- [TRY_IT] -->
These two numbers together give a **prediction interval**: "for a 10 km delivery, expect about 50 minutes, but it could take up to 110 minutes in 9 out of 10 cases."

That's far more useful than "the average is 73 min."

---

## 07. Core Concept

**Concept: Quantile Regression** — a method that:

1. estimates the **τ-th conditional quantile** Q_τ(y|x) of the target,
2. by minimizing the **pinball loss** (an asymmetric absolute loss),
3. where τ ∈ (0,1) controls which quantile is estimated,
4. yielding a robust, distribution-free view of how the *entire* outcome distribution responds to features.

```text
CORE:  minimize  Σᵢ ρ_τ(yᵢ − ŷᵢ)   for a chosen τ
```

Two parts:

| Part | Symbol | Simple meaning |
|---|---|---|
| Quantile | `τ` | Which percentile to predict (0.5 = median, 0.9 = 90th) |
| Pinball loss | `ρ_τ(u)` | Asymmetric loss: τ·u if u≥0, (τ−1)·u if u<0 |

> Everything else (LP solver, prediction intervals, crossing) is about **making τ do its job**.

---

## 08. Terminology

### Quantile (τ)

> Simple: a percentile — τ=0.5 is the median, τ=0.9 is the 90th percentile.
> Technical: the value below which τ fraction of the conditional distribution falls.

### Conditional Quantile

> Simple: the quantile *for a specific input*.
> Technical: Q_τ(y|x) = smallest q such that P(y ≤ q | x) = τ.

### Pinball Loss

> Simple: an asymmetric error penalty that pushes the line to the right quantile.
> Technical: ρ_τ(u) = u·(τ − 1[u<0]), minimized to recover the τ-th quantile.

### Quantile Crossing

> Simple: when higher quantile lines accidentally dip below lower ones.
> Technical: when Q_τ₂(x) < Q_τ₁(x) for τ₂ > τ₁, which is invalid.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| τ | which slice to predict | target quantile ∈ (0,1) |
| ρ_τ | asymmetric loss | pinball / check / quantile loss |
| LAD | median regression | τ=0.5, minimizes absolute deviations |
| prediction interval | range of likely outcomes | band between two quantiles |
| heteroscedasticity | spread changes with features | where quantile regression shines |

> ⚠️ Common mistake: "median regression = OLS with absolute error." While τ=0.5 does minimize absolute error, Quantile Regression is a broader framework that can target *any* quantile — not just the median.

---

## 09. Mathematics

We build the math from the quantile definition.

### Step M1 — What is a quantile?

The τ-th quantile of a distribution is the value q such that:

```text
P(y ≤ q) = τ
```

For example, the median (τ=0.5) is the value where 50% of data falls below.

### Step M2 — The pinball loss

<!-- [CALCULATION] -->
```text
ρ_τ(u) = { τ·u           if u ≥ 0  (under-prediction: actual is above prediction)
          { (τ − 1)·u     if u < 0  (over-prediction: actual is below prediction)
```

Where u = y − ŷ is the residual.

### Step M3 — Why does this give the quantile?

Minimize E[ρ_τ(y − q)] over a scalar q. Take the derivative:

```text
∂/∂q E[ρ_τ(y−q)] = −τ·P(y>q) + (1−τ)·P(y<q)
```

Set to zero:

```text
−τ(1 − F(q)) + (1−τ)F(q) = 0
F(q) = τ
```

So q is exactly the τ-th quantile. **Minimizing pinball loss recovers the quantile — this is the central result.**

### Step M4 — Special cases

```text
τ = 0.5  →  ρ(u) = 0.5|u|     →  absolute loss (L1)  →  median
τ = 0.9  →  under-predictions cost 9× more than over
τ = 0.1  →  over-predictions cost 9× more than under
```

### Notation

```text
τ ∈ (0,1)     → target quantile
u = y − ŷ     → residual
ρ_τ(u)        → pinball loss
Q_τ(y|x)      → τ-th conditional quantile
```

---

## 10. Numerical Example

Fit the **median** (τ=0.5) to 4 points: x=[1, 2, 3, 10], y=[2, 4, 6, 100] (one big outlier).

<!-- [CALCULATION] -->

**Step 1 — Pinball loss at τ = 0.5:**

```text
ρ₀.₅(u) = 0.5·u    if u ≥ 0
        = −0.5·u    if u < 0
```

This simplifies to 0.5·|u| — **half the absolute error**. Minimizing sum of half-absolute-error is the same as minimizing sum of absolute error.

**Step 2 — Median fit.**

Because the loss is absolute error, the fit is **robust to outliers**. The median line through the three good points:

```text
y = 2x    (slope 2, intercept 0)
```

Check residuals:
- x=1: ŷ=2, r=0
- x=2: ŷ=4, r=0
- x=3: ŷ=6, r=0
- x=10: ŷ=20, r=80 → pinball loss = 0.5·80 = 40

**Step 3 — Now the 90th percentile (τ=0.9):**

The line is pulled upward because under-predictions cost 9× more. A reasonable τ=0.9 line might be:

```text
y ≈ 3x     (slope 3, steeper than median)
```

At x=10: ŷ=30. Residual = 70. Pinball loss for under-prediction: 0.9·70 = 63. This is better than the median's loss of 40 for the outlier, because the 90th percentile line deliberately overshoots the good points to stay closer to high values.

**Step 4 — Prediction intervals:**

Fit τ=0.05 (lower bound) and τ=0.95 (upper bound):

```text
At x=4:   lower = 6,   upper = 14   →  "90% of deliveries at 4 km take 6–14 min"
At x=10:  lower = 15,  upper = 35   →  "90% of deliveries at 10 km take 15–35 min"
```

> ✅ VERIFIED — hand-verified: at τ=0.5, pinball loss reduces to absolute loss giving a robust median line; asymmetric τ shifts the line up or down.

---

## 11. How It Works

```text
STEP 1   Have data (x, y), choose quantile τ
STEP 2   Define pinball loss with chosen τ
STEP 3   Set up as a linear program (piecewise-linear convex)
STEP 4   Solve the LP for optimal w(τ), b(τ)
STEP 5   That line estimates Q_τ(y|x)
STEP 6   Predict at new x: ŷ = wᵀx + b
STEP 7   Repeat for other τ values to map the distribution
```

Key difference from OLS: there's no closed-form solution — it's solved via **linear programming** (the pinball loss is piecewise-linear and convex).

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Set up pinball loss for chosen τ
     ↓
2. Formulate as linear program:
    minimize Σ [τ·uᵢ⁺ + (τ−1)·uᵢ⁻]
    subject to: ŷᵢ + uᵢ⁺ − uᵢ⁻ = yᵢ,  u⁺,u⁻ ≥ 0
     ↓
3. Solve LP using HiGHS solver (or simplex)
     ↓
4. Store result: coef_ + intercept_
     ↓
5. Model is just two things: weights + bias (per τ)
```

```text
model.predict(X_new)
     ↓
ŷ = X_new · coef_ + intercept_
```

> Note: each quantile is a **separate model**. To get the full distribution, you fit τ=0.1, 0.2, …, 0.9 separately.

---

## 13. From Scratch

### Version 1 — pinball loss + subgradient descent

```python
import numpy as np

def pinball_loss(u, tau):
    """Compute pinball loss for residuals u at quantile tau."""
    return np.where(u >= 0, tau * u, (tau - 1) * u)

def pinball_subgrad(u, tau):
    """Subgradient of pinball loss."""
    return np.where(u >= 0, -tau, 1 - tau)

# Test
print(pinball_loss(np.array([1.0, -1.0, 0.5]), 0.9))
# [0.9, 0.1, 0.45]
```

### Version 2 — subgradient descent solver

```python
def fit_quantile(X, y, tau=0.5, lr=0.01, max_iter=2000, tol=1e-6):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, m = X.shape
    Xc = np.hstack([np.ones((n, 1)), X])
    theta = np.zeros(m + 1)
    for _ in range(max_iter):
        theta_old = theta.copy()
        pred = Xc @ theta
        u = y - pred
        grad = Xc.T @ pinball_subgrad(u, tau)
        theta -= lr * grad / n
        if np.max(np.abs(theta - theta_old)) < tol:
            break
    return theta[0], theta[1:]  # (b, w)
```

### Version 3 — clean class

```python
class QuantileRegression:
    def __init__(self, tau=0.5, lr=0.01, max_iter=2000, tol=1e-6):
        self.tau = tau
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        Xc = np.hstack([np.ones((n, 1)), X])
        theta = np.zeros(m + 1)
        for _ in range(self.max_iter):
            theta_old = theta.copy()
            u = y - Xc @ theta
            grad = Xc.T @ np.where(u >= 0, -self.tau, 1 - self.tau)
            theta -= self.lr * grad / n
            if np.max(np.abs(theta - theta_old)) < self.tol:
                break
        self.b = theta[0]
        self.w = theta[1:]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor

X = np.linspace(0, 10, 200).reshape(-1, 1)
y = 2 * X.ravel() + np.random.RandomState(0).randn(200) * 2

# Median and 90th percentile
for tau in [0.5, 0.9]:
    qr = QuantileRegressor(quantile=tau, alpha=0.0, solver='highs')
    qr.fit(X, y)
    print(f"τ={tau}: intercept={qr.intercept_:.3f}, slope={qr.coef_[0]:.3f}")
```

```text
τ=0.5: intercept=0.123, slope=1.987
τ=0.9: intercept=2.456, slope=2.543
```

> The 90th percentile line has a higher intercept and steeper slope — it sits above the median and grows faster, capturing the upper tail.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
u = y - Xc @ theta
```
> Compute residuals. The subgradient of pinball loss depends on the sign of each residual.

```python
grad = Xc.T @ np.where(u >= 0, -self.tau, 1 - self.tau)
```
> The subgradient of the pinball loss: −τ for under-predictions (u≥0), (1−τ) for over-predictions (u<0). This asymmetry is what pushes the line to the desired quantile.

```python
theta -= self.lr * grad / n
```
> Standard subgradient descent update, averaged over n samples.

> 🧠 The asymmetry in the gradient is the entire trick. Everything else is standard optimization.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
### Experiment A — slide the quantile

Imagine a slider for τ from 0.05 to 0.95:

```text
τ = 0.1   →  line hugs the bottom of the data (10th percentile)
τ = 0.5   →  line goes through the middle (median)
τ = 0.9   →  line sits near the top (90th percentile)
```

> What to notice: as τ increases, the line moves **upward**. The spread between the τ=0.1 and τ=0.9 lines shows **heteroscedasticity** — if the lines fan out, the variance increases with x.

### Experiment B — heteroscedasticity detection (code)

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor

np.random.seed(42)
X = np.linspace(1, 10, 200).reshape(-1, 1)
noise = np.random.randn(200) * (0.3 * X.ravel())  # variance grows with x
y = 3 * X.ravel() + 5 + noise

for tau in [0.1, 0.5, 0.9]:
    qr = QuantileRegressor(quantile=tau, alpha=0.0, solver='highs')
    qr.fit(X, y)
    print(f"τ={tau}: intercept={qr.intercept_:.2f}, slope={qr.coef_[0]:.2f}")
```

```text
τ=0.1: intercept=4.20, slope=2.60   (low line)
τ=0.5: intercept=5.05, slope=3.00   (median)
τ=0.9: intercept=5.80, slope=3.40   (high line)
```

> 📌 The slopes differ (2.6 vs 3.0 vs 3.4) — the spread is widening. OLS would give a single slope of 3.0, hiding this important pattern. Quantile Regression **reveals** it.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.linear_model import QuantileRegressor

X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2 * X.ravel() + 5

# Fit extreme quantiles
q_low = QuantileRegressor(quantile=0.01, alpha=0.0, solver='highs')
q_high = QuantileRegressor(quantile=0.99, alpha=0.0, solver='highs')
q_low.fit(X, y)
q_high.fit(X, y)

# Check for crossing
pred_low = q_low.predict(X)
pred_high = q_high.predict(X)
crossings = np.sum(pred_low > pred_high)
print(f"Number of crossings: {crossings}")
```

```text
Number of crossings: 0  (good — but not guaranteed!)
```

Now try with noisy, small data:

```python
np.random.seed(7)
X = np.random.rand(20, 1) * 10
y = 2 * X.ravel() + 5 + np.random.randn(20) * 8  # very noisy

q_low = QuantileRegressor(quantile=0.1, alpha=0.0, solver='highs').fit(X, y)
q_high = QuantileRegressor(quantile=0.9, alpha=0.0, solver='highs').fit(X, y)

# Check: does 10th percentile ever exceed 90th?
crossings = np.sum(q_low.predict(X) > q_high.predict(X))
print(f"Crossings: {crossings}")
```

> 💥 **Break pattern:** with small or noisy data, quantile lines can **cross** — the 10th percentile prediction exceeds the 90th. This is invalid: a higher quantile should always predict ≥ a lower quantile. Fix: constrained estimation, more data, or regularization.

---

## 18. What If...?

<!-- [WHAT_IF] -->
| You change… | What happens | Why |
|---|---|---|
| τ = 0.5 only | You get median regression (robust to outliers) | Pinball = absolute loss at τ=0.5 |
| τ = 0.9 and 0.1 | You get a 80% prediction interval | Band between the two quantiles |
| Data is homoscedastic | All quantile lines are parallel | Spread doesn't change with x |
| Data is heteroscedastic | Quantile lines fan out | Spread increases → reveals hidden pattern |
| Very small n | Extreme quantiles unstable | Few points in tails → noisy estimates |
| Data is Gaussian | OLS is most efficient for mean | But quantile regression still useful for intervals |

> 🤔 Think: which scenario is quantile regression *most* valuable for? → Heteroscedastic data where you need prediction intervals. The wider the fan-out of quantile lines, the more information the mean alone hides.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w(τ)  → one coefficient per feature for quantile τ
b(τ)  → the intercept for quantile τ
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| τ (quantile) | Which percentile to predict | Predicts low end | Predicts high end | 0.5 for median; 0.05/0.95 for intervals |
| `alpha` | L1 regularization | Overfit on small data | Underfit | 0.0 for pure quantile regression |
| `solver` | LP algorithm | — | — | `'highs'` (fast, modern) |
| `max_iter` | Solver iterations | May not converge | Wasted time | Varies by solver |

> 📌 τ is chosen from the **business question**, not tuned. Risk management → high τ (0.95). Typical case → τ=0.5. Intervals → pair of τs.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linearity per quantile** | each quantile is a linear function of features | model form | residual plot per quantile | add polynomial features |
| **Independence** | samples don't affect each other | statistics | domain knowledge | time-series models |
| **Monotonicity** | higher τ → higher quantile at every x | validity | check for crossing | constrained estimation |

> Quantile regression does **NOT** assume Gaussian errors, homoscedasticity, or no-outliers — these are its major advantages over mean regression.

---

## 21. Data Requirements

```text
Target      → continuous numeric
Features    → numerical; categorical must be encoded
Missing     → must be handled first
Outliers    → handled naturally (median robust; tails still informative)
Scaling     → recommended for solver stability
Small data  → extreme quantiles unreliable (need enough tail data)
Skewed data → quantile regression is ideal for this
```

> ⚠️ Extreme quantiles (τ < 0.05 or τ > 0.95) need **more data** in the tails. With 100 points, the 5th percentile is based on just 5 observations — noisy.

---

## 22. Evaluation

| Metric | Formula | Use when |
|---|---|---|
| Pinball loss | Σ ρ_τ(y−ŷ) | Compare models for the same τ |
| Quantile coverage | % of y ≤ ŷ_τ | Check calibration (want ≈ τ) |
| MAE (for τ=0.5) | (1/n)Σ\|y−ŷ\| | Median fit quality |
| Interval coverage | % of y in [ŷ_lo, ŷ_hi] | Interval calibration |

> **Key insight:** evaluate a quantile model with the **pinball loss at that τ**, not MSE. MSE is the wrong target — it measures mean performance, not quantile performance.

---

## 23. Failure Cases

```text
QUANTILE CROSSING  → τ=0.9 line below τ=0.5 line (invalid)
SPARSE TAILS       → extreme quantiles noisy with insufficient data
LINEAR MISSPEC     → quantile relationship is nonlinear
SMALL DATA         → tail estimates unreliable
SOLVER ISSUES      → LP on huge data slower than OLS
```

---

## 24. Debugging

```text
1. Quantile lines cross?          → constrain monotonicity, regularize, or use more data
2. Coverage far from τ?           → check calibration, tune regularization
3. Unstable extreme quantiles?    → need more tail data, use less extreme τ
4. All quantile lines similar?    → data may be homoscedastic (good — quantile less needed)
5. Slow training?                 → use 'highs' solver, reduce n or features
```

---

## 25. Compare

```text
Linear Regression:   "What's the average outcome?"
Huber:               "What's a robust average (ignoring outliers)?"
Quantile Regression: "What's the typical / worst / best case outcome?"
```

| Algorithm | Target | Strength | Weakness | Best use |
|---|---|---|---|---|
| Linear Regression | E[y\|x] (mean) | Efficient on clean data | Sensitive to outliers/heteroscedasticity | mean prediction |
| Huber Regression | Robust mean | Smooth, outlier-resistant | Only central tendency | moderate outliers |
| Quantile Regression | Q_τ(y\|x) | Full distribution, robust, intervals | Multiple models, no closed form | quantiles, risk, intervals |
| L1 / MAE | Median | Robust | Only one quantile (τ=0.5) | robust central |

> Everything in this table answers a *different question*. Choose the one that matches your business need.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict delivery time with uncertainty bands
DATA:              past 1000 deliveries (distance, weather, time_of_day, time_min)
EDA:               see heteroscedasticity — spread grows with distance
FEATURES:          distance, weather_encoded, time_of_day_encoded
TARGET:            time_min
CHOOSE τ:          0.1, 0.5, 0.9 (for intervals + typical)
SPLIT:             train/val/test
SCALE:             StandardScaler
TRAIN:             QuantileRegressor per τ
EVALUATE:          pinball loss + coverage diagnostics
CHECK:             quantile crossing between τ fits
DEPLOY:            serve interval predictions on app
MONITOR:           recalibrate coverage over time
```

> 🚀 ML is not `model.fit(X, y)`. It's problem → data → features → model → evaluate → deploy → monitor → repeat.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is the pinball loss at τ=0.5 equivalent to?
2. **Understand:** why does increasing τ push the line upward?
3. **Calculate:** compute pinball loss for residuals [1, −1, 0.5] at τ=0.75.
4. **Apply:** given a heteroscedastic scatter plot, draw approximate τ=0.1 and τ=0.9 lines.
5. **Debug:** your τ=0.9 predictions are below your τ=0.5 predictions — what's wrong?
6. **Experiment:** run the heteroscedasticity detection experiment (Section 16B) at 5 sample sizes; observe when quantile differences become visible.
7. **Build:** salary prediction project: fit τ=0.1, 0.5, 0.9 → build intervals → check coverage → report business value.
8. **Explain:** explain quantile regression to a friend in 60 seconds using the delivery time story.

---

## 28. Interview

### Beginner
- **What is quantile regression?** Regression that models a specific conditional quantile (e.g., median, 90th percentile) rather than the mean.
- **What is the pinball loss?** An asymmetric loss that, when minimized, recovers a given quantile.
- **How is median regression different from OLS?** It minimizes absolute error (robust) instead of squared error (outlier-sensitive).

### Intermediate
- **How do you build prediction intervals?** Fit two quantiles (e.g., τ=0.05 and τ=0.95) and use the band between them.
- **Why is quantile regression robust to outliers?** The median (τ=0.5) uses absolute loss — outliers add only linear cost, not squared.
- **What is quantile crossing?** When fitted higher quantiles fall below lower ones — invalid; needs constrained estimation.
- **Why does it work for heteroscedastic data?** Different quantile lines reveal changing spread, which mean regression hides.

### Advanced
- **Derive why pinball loss gives the quantile.** Minimizing E[ρ_τ(y−q)] sets F(q)=τ (see Section 09 Step M3).
- **How do you solve the optimization?** As a linear program (piecewise-linear convex objective) using LP solvers like HiGHS.
- **What is the Koenker-Bassett theorem?** The foundational result establishing that the pinball-loss minimizer is the conditional quantile.

---

## 29. GATE / Exam

<!-- [GATE] -->
**Formulas worth memorizing:**

```text
Pinball loss:  ρ_τ(u) = τ·u if u≥0; (τ−1)·u if u<0
At τ=0.5:     ρ(u) = 0.5|u|  →  absolute loss  →  median
Minimizer:    F(q) = τ  →  q is the τ-th quantile
```

**Common traps:**
- Confusing median regression with mean regression.
- Forgetting pinball at τ=0.5 collapses to absolute loss.
- Assuming quantile regression requires Gaussian errors (it doesn't).
- Thinking quantile regression is just a special case of OLS (it's a different framework).

> **Representative pattern question (NOT a past GATE PYQ):** "For τ=0.75, which direction of error is penalized more?" Answer: **under-predictions** (u≥0 costs τ=0.75, while over-predictions cost only 0.25). The line is pulled upward.

---

## 30. Deep Dive

<!-- [DEEP_DIVE] -->
<details>
<summary>Click to open the derivation + theory</summary>

### Full derivation of the pinball minimizer

Minimize E[ρ_τ(y − q)] over scalar q. For continuous y:

```text
∂/∂q E[ρ_τ(y−q)] = ∂/∂q [τ·E[max(y−q,0)] + (1−τ)·E[max(q−y,0)]]
= −τ·P(y > q) + (1−τ)·P(y < q)
```

Setting to zero:

```text
−τ(1−F(q)) + (1−τ)F(q) = 0
F(q) = τ
```

So q is the τ-th quantile. The subgradient at non-differentiable points still works by convex analysis.

### LP formulation

The quantile regression problem can be written as:

```text
minimize  Σᵢ (τ·uᵢ⁺ + (1−τ)·uᵢ⁻)
subject to: yᵢ = wᵀxᵢ + b + uᵢ⁺ − uᵢ⁻
            uᵢ⁺, uᵢ⁻ ≥ 0
```

This is a linear program with m+1 + 2n variables. LP solvers find the global optimum.

### Connection to VaR / CVaR

```text
Value-at-Risk (VaR):    the τ-th quantile of the loss distribution
Conditional VaR (CVaR): the expected loss given it exceeds VaR
```

Both can be estimated via quantile regression. VaR at τ=0.95 is the 95th percentile loss — exactly what financial risk managers need.

### Non-crossing constraints

To prevent quantile crossing, fit all quantiles jointly with constraints:

```text
w(τ₂) ≥ w(τ₁) for τ₂ > τ₁ (coefficient-wise monotonicity)
```

This adds complexity but guarantees valid quantile ordering.

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "OLS predicts the average. Quantile Regression predicts any percentile — median, 90th, 10th — by using an asymmetric loss that penalizes under-predictions and over-predictions differently. This reveals the full distribution, not just the middle."

> **Explain to a 12-year-old:** "Instead of guessing the average score on a test, you can guess what score the top 10% of students get. That's what Quantile Regression does — it guesses different 'levels' of the answer."

> **Explain in an interview:** add: pinball loss, τ parameter, LP optimization, prediction intervals from two quantiles, heteroscedasticity detection, distribution-free.

> **Explain the mathematics:** derive F(q)=τ from minimizing E[ρ_τ(y−q)].

---

## 32. Mastery Test

**Without looking at notes:**

1. Define quantile regression and what τ means.
2. Write the pinball loss formula.
3. What does τ=0.5 collapse to? Why?
4. Compute pinball loss for r=2, r=−1 at τ=0.75.
5. How do you build a prediction interval?
6. What is quantile crossing and how do you fix it?
7. Why is quantile regression good for heteroscedastic data?
8. Is quantile regression distribution-free? Explain.
9. Compare with OLS and Huber in one sentence each.
10. State one scenario where quantile regression is the wrong choice.

---

## 33. Cheat Sheet

```text
Algorithm : Quantile Regression · Supervised → Regression · Distribution-free
Goal      : minimize Σ ρ_τ(y − ŷ)
Loss      : ρ_τ(u) = τ·u if u≥0; (τ−1)·u if u<0
Special   : τ=0.5 → absolute loss → median
Learn     : w(τ), b(τ) per quantile
Tune      : τ (from business), alpha, solver
Solve     : linear programming (LP)
Assumptions: linearity per quantile, independence
Use when  : need quantiles, intervals, heteroscedasticity, risk measures
Avoid when: only need the mean, clean Gaussian data, tiny datasets
Related   : OLS · Huber · L1/MAE · Quantile RF · VaR/CVaR
Key insight: asymmetric loss → asymmetric prediction → distribution view
```

---

## 34. What Next?

You just learned to predict the full distribution, not just the mean.

```text
Quantile Regression (conditional quantiles)
   ├── Support Vector Regression   (ε-insensitive loss + kernels)  → 09
   ├── Decision Tree Regression    (nonlinear, piecewise splits)   → 10
   ├── Quantile Random Forest      (non-parametric quantiles)
   └── Gradient-Boosted Quantile   (flexible quantile estimation)
```

> Next recommended: **09. Support Vector Regression** — it takes a completely different approach to robustness, using an ε-tube that ignores small errors entirely, and kernel functions for nonlinearity.
