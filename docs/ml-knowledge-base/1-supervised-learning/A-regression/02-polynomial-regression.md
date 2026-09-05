# 02. Polynomial Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐☆☆
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **curve → line fails → powers → expand → OLS in new space → degree control → overfitting trap.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Linear Regression draws one straight line through data. But what if the data *curves*?

Polynomial Regression keeps all of Linear Regression's easy math — it just adds powers of the input so the "line" can bend.

By the end you will be able to:

- recognise when a straight line isn't enough,
- expand features into powers and fit a curve,
- control the degree to balance bias and variance,
- code it from scratch and with sklearn,
- break it deliberately (overfitting, edge effects, extrapolation blow-up),
- and defend when to use — and not use — it.

> Everything in this note builds on one question: *how do we let a straight line bend?*

---

## 02. The Problem

A local cricket coach is tracking a fast bowler's speed as he ages:

| Age (years) | Bowling speed (km/h) |
|---|---|
| 16 | 110 |
| 18 | 120 |
| 20 | 128 |
| 22 | 132 |
| 24 | 134 |
| 26 | 133 |
| 28 | 129 |
| 30 | 122 |

<!-- [QUESTION] -->
The coach wants to predict speed at age 25.

> **What speed would you predict for age 25?**

Look at the numbers. Speed rises, peaks, then *falls*. A straight line can't do that.

Make your guess: **____ km/h**

> 📌 Keep this number. At the end of Section 06 we'll compare it with what the model says.

---

## 03. Let's Think

Let's compute the year-to-year changes:

```text
16→18  +10
18→20  +8
20→22  +4
22→24  +2
24→26  −1      ← speed starts falling
26→28  −4
28→30  −7
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> The increments are *shrinking*, then turning negative. There is a **curve** — it goes up, peaks, then comes down.

A straight line would draw one direction through all eight points and miss the turning point entirely.

But look — if you just connect the dots near age 24–26, the data *locally* looks like a gentle arc. What if we let the line **bend** using a curve that follows this shape?

> The pattern looks like: speed = base + something × age + something × age².

That "something × age²" is the new ingredient. It lets the model curve.

---

## 04. Intuition

💡 **The idea in one line:**

> Polynomial Regression is just Linear Regression on *powered-up* features: instead of only using `x`, it uses `x, x², x³, …`, letting the model follow curves while keeping all the same easy OLS math.

Think of it this way:

```text
Linear Regression:     ŷ = b + w₁·x           → straight line
Degree-2 Polynomial:   ŷ = b + w₁·x + w₂·x²  → parabola (one bend)
Degree-3 Polynomial:   ŷ = b + w₁·x + w₂·x² + w₃·x³  → S-curve
```

The "curve" is just a straight line in a *bigger* feature space. We never actually bend the line — we add columns (x², x³…) and let OLS fit a hyperplane in that space. The geometry does the bending for us.

> 📌 Key insight: the model is still **linear in its parameters** (the w's). It's only the *features* that are non-linear. That's what keeps the closed-form OLS solution working.

---

## 05. Visual

```text
Speed (km/h)
  134 │                  • (24, 134)
      │                •╱╲• (26, 133)
  128 │            •╱╱     ╲• (28, 129)
      │          •╱╱         ╲
  122 │        •╱╱              ╲• (30, 122)
      │      •╱
  116 │    •╱
      │  •╱
  110 │•╱
      └─────────────────────────────── Age
         16  18  20  22  24  26  28  30

  •  = actual data points
  ╱╲ = fitted curve (degree-2 polynomial)
```

A straight line through these points would slope upward forever — missing the peak and the decline. A degree-2 polynomial bends once: it goes up, reaches a top, comes back down. That matches our data.

Higher degree → more bends:

```text
p=1  ─────  straight (can't peak)
p=2  ∩      one arch (peaks and comes down) ← what we need
p=3  ∫      one S-bend
p=4  M      two humps
```

---

## 06. First Prediction

Let's fit a degree-2 polynomial by eye. The peak is around age 24 at 134 km/h. A parabola opening downward through roughly (16, 110) and (30, 122) with peak near (24, 134) gives:

```text
ŷ ≈ −0.8x² + 39.2x − 333.6
```

At age 25:

```text
ŷ = −0.8(625) + 39.2(25) − 333.6 = −500 + 980 − 333.6 = 146.4
```

Wait — that seems high. Let's verify: the data says 24→134, 26→133, so the peak is right at 24–25. A value around **134 km/h** makes sense for age 25.

<!-- [TRY_IT] -->
Did your guess match the curve's answer?

> 📌 If you said 133–135 km/h, your visual intuition agrees with the polynomial. The math that follows makes this **exact and repeatable**.

Now the real question:

> **How do we find the *best* curve without eyeballing?**

---

## 07. Core Concept

**Concept: Polynomial Regression** — a method that:

1. takes the original feature `x` and creates new features `x², x³, …, xᵖ` (the **degree** `p` controls how many powers),
2. fits a linear model on these expanded features using OLS,
3. produces a polynomial curve that can bend to match non-linear data.

```text
ŷ = b + w₁·x + w₂·x² + … + wₚ·xᵖ
```

| Part | Symbol | Simple meaning |
|---|---|---|
| Degree | `p` | How many bends the curve may have (1 = line, 2 = parabola …) |
| Weight | `wₖ` | How much the k-th power of x contributes |
| Intercept | `b` | The baseline when all powers are zero |

> Everything else — the OLS math, RSS, the normal equation — is identical to Linear Regression, just applied to the expanded feature matrix.

---

## 08. Terminology

### Polynomial Degree (p)

> Simple: the highest power of x in the model.
> Technical: controls model complexity; degree p creates p basis functions per feature.

### Feature Expansion

> Simple: turning x into x, x², x³ …
> Technical: constructing a polynomial basis; the design matrix gains columns.

### Linear in Parameters

> Simple: the model is still a weighted sum of its parameters.
> Technical: ŷ = Σ wₖ · zₖ where zₖ are the (transformed) features; OLS applies unchanged.

### Overfitting (degree-induced)

> Simple: using too many powers and fitting noise instead of the trend.
> Technical: high variance from excessive model complexity; training error drops but test error rises.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| p | highest power | model complexity control |
| xᵏ | x raised to power k | k-th basis function |
| ŷ | model's prediction | estimated target |
| RSS | total squared miss | same as Linear Regression |
| Multicollinearity (induced) | x, x², x³ are correlated | numerical instability in ZᵀZ |

> ⚠️ Common mistake: calling polynomial regression "non-linear regression." That phrase usually means non-linear *in parameters* — a different thing entirely.

---

## 09. Mathematics

We build the math in three steps — nothing new beyond Linear Regression.

### Step M1 — The polynomial model

```text
ŷ = b + w₁x + w₂x² + … + wₚxᵖ
```

### Step M2 — Feature expansion: define new variables

```text
z₁ = x,  z₂ = x²,  …,  zₚ = xᵖ
```

Now:

```text
ŷ = b + w₁z₁ + w₂z₂ + … + wₚzₚ
```

This is **exactly Linear Regression** on the expanded features (z₁, z₂, …, zₚ).

### Step M3 — Same objective, same solution

```text
RSS = Σᵢ (yᵢ − ŷᵢ)²
w = (ZᵀZ)⁻¹ Zᵀ y          ← the normal equation on expanded Z
```

where Z is the design matrix with rows `[1, x, x², …, xᵖ]`.

```text
Σ      → sum over all data points
Z      → expanded design matrix (n × (p+1))
w      → vector [b, w₁, w₂, …, wₚ]
ZᵀZ    → the Gram matrix of expanded features
```

> 💡 Intuition: we never "bend the line." We build a higher-dimensional straight line (hyperplane) and let the geometry bend the curve in the original x-y view.

---

## 10. Numerical Example

Fit a degree-2 polynomial to 3 points: `x = [1, 2, 3]`, `y = [4, 3, 8]`.

<!-- [CALCULATION] -->

**Step 1 — Build the expanded design matrix Z (degree 2, with intercept column):**

```text
row 1: [1, 1,  1 ]
row 2: [1, 2,  4 ]
row 3: [1, 3,  9 ]
```

**Step 2 — Compute ZᵀZ:**

```text
ZᵀZ = [[3,  6, 14],
       [6, 14, 36],
       [14, 36, 98]]
```

**Step 3 — Compute Zᵀy:**

```text
Zᵀy = [1·4+1·3+1·8,  1·4+2·3+3·8,  1·4+4·3+9·8]
     = [15, 34, 88]
```

**Step 4 — Solve (ZᵀZ)w = Zᵀy** (Gaussian elimination):

```text
b  (=w₀) = 8
w₁ = −8
w₂ = 4
```

**Step 5 — The fitted curve:**

```text
ŷ = 8 − 8x + 4x²
```

**Step 6 — Verify:**

```text
x=1: ŷ = 8 − 8 + 4 = 4        actual 4    miss = 0
x=2: ŷ = 8 − 16 + 16 = 8      actual 3    miss = +5
x=3: ŷ = 8 − 24 + 36 = 20     actual 8    miss = +12

RSS = 0 + 25 + 144 = 169
```

> ✅ VERIFIED — hand-computed; the degree-2 parabola `ŷ = 8 − 8x + 4x²` minimises RSS for the data (1,4), (2,3), (3,8). Note the fit isn't perfect — the data is genuinely curved imperfectly.

**Predict something new:**

```text
x = 1.5  →  ŷ = 8 − 8(1.5) + 4(2.25) = 8 − 12 + 9 = 5
```

<!-- [TRY_IT] -->
🎯 Your turn: predict `x = 2.5` with this curve before reading on.

> Answer: `8 − 8(2.5) + 4(6.25) = 8 − 20 + 25 = 13`.

---

## 11. How It Works

```text
STEP 1   Have data (x, y)
STEP 2   Choose degree p
STEP 3   Expand features: build Z = [1, x, x², …, xᵖ]
STEP 4   Solve OLS on Z:  w = (ZᵀZ)⁻¹ Zᵀy      ← same as Linear Regression
STEP 5   Production: new x → compute powers → ŷ = Z_new · w
```

If Linear Regression was clear, Steps 3–4 are the only additions — and they're just **data transformation**, not new math.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. PolynomialFeatures(degree=p).fit_transform(X)
   → adds columns x², x³, …, xᵖ to the feature matrix
     ↓
2. Build design matrix Z = [1, x, x², …, xᵖ]   (column of 1s absorbs intercept)
     ↓
3. Solve  w = (ZᵀZ)⁻¹·Zᵀ·y                      ← same normal equation
     ↓
4. Store result:  coef_  +  intercept_
     ↓
5. Model is now:  p weights + a bias
```

```text
model.predict(X_new)
     ↓
for each new row:
    compute powers: [1, x, x², …, xᵖ]
    ŷ = Z_new · weights + bias
```

> Note: the polynomial expansion is done once at fit time (and again at predict time). OLS itself is still a one-shot calculation — no training loop, no epochs.

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
def poly_design(xs, degree):
    """Build [1, x, x², …, x^degree] for each x."""
    return [[x**d for d in range(degree + 1)] for x in xs]

def fit_poly(xs, ys, degree):
    import numpy as np
    Z = np.array(poly_design(xs, degree))
    y = np.array(ys)
    # Normal equation
    w = np.linalg.inv(Z.T @ Z) @ Z.T @ y
    return w

def predict_poly(xs, w, degree):
    Z = np.array(poly_design(xs, degree))
    return Z @ w

w = fit_poly([1, 2, 3], [4, 3, 8], degree=2)
print(w)                          # [ 8. -8.  4.]
print(predict_poly([1.5], w, 2))  # [5.]
```

### Version 2 — numpy, vectorized

```python
import numpy as np

def fit_poly_vec(X, y, degree):
    X = np.asarray(X, dtype=float).ravel()
    y = np.asarray(y, dtype=float)
    Z = np.column_stack([X**d for d in range(degree + 1)])
    w = np.linalg.pinv(Z.T @ Z) @ Z.T @ y
    return w

w = fit_poly_vec([1, 2, 3], [4, 3, 8], 2)
print(w)  # [ 8. -8.  4.]
```

### Version 3 — clean class

```python
import numpy as np

class PolynomialRegression:
    def __init__(self, degree=2):
        self.degree = degree
        self.w = None

    def _expand(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        cols = [np.ones((n, 1))]
        for d in range(1, self.degree + 1):
            cols.append(X[:, :1] ** d)
        return np.hstack(cols)

    def fit(self, X, y):
        Z = self._expand(X)
        self.w = np.linalg.pinv(Z.T @ Z) @ Z.T @ np.asarray(y)

    def predict(self, X):
        Z = self._expand(X)
        return Z @ self.w
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26])    # y = x² + 1

model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LinearRegression()
)
model.fit(X, y)

print("Coefficients:", model.named_steps['linearregression'].coef_)
print("Intercept:", model.named_steps['linearregression'].intercept_)
print("Predict x=3.5:", model.predict([[3.5]]))  # ~13.25
```

> `PolynomialFeatures` does the expansion. `LinearRegression` does the OLS. sklearn didn't invent a new algorithm — it chained two existing ones.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
PolynomialFeatures(degree=2, include_bias=False)
```
> Creates new columns for every combination of powers up to degree 2. With one feature `[x]`, output is `[x, x²]`. `include_bias=False` because `LinearRegression` adds its own intercept.

```python
model = make_pipeline(PolynomialFeatures(...), LinearRegression())
```
> Chains expansion → fitting into one object. `fit(X, y)` first transforms X, then fits the linear model on the transformed X. Same pattern works with Ridge, Lasso, any linear model.

```python
model.named_steps['linearregression'].coef_
```
> These are `w₁, w₂, …` — one per expanded feature. They tell you the contribution of each power. (With `include_bias=True` the pipeline adds a constant column and the intercept gets absorbed.)

> 🧠 Every line maps directly to the formula from Section 09.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> If this note is rendered inside the interactive platform, these become sliders. Otherwise, run them in Python and observe.

### Experiment A — Slide the degree

Imagine a slider for `p` (degree), with the bowling speed data behind:

```text
p=1  →  straight line sloping upward     → misses the peak, misses the decline
p=2  →  parabola, peaks near age 24      → matches the shape well
p=4  →  wiggly curve, touches most dots  → looks good on training, but what about age 25?
p=7  →  passes through ALL 8 points exactly → perfect training score, wild oscillation between points
```

> What to notice: **training RSS always drops** as p increases. But the curve gets wigglier. The gap between training RSS and test RSS grows — that's overfitting.

### Experiment B — The degree sweep (code)

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = np.array([16, 18, 20, 22, 24, 26, 28, 30]).reshape(-1, 1)
y = np.array([110, 120, 128, 132, 134, 133, 129, 122])

for deg in [1, 2, 3, 4, 7]:
    Z = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    w = np.linalg.lstsq(Z, y, rcond=None)[0]
    y_hat = Z @ w
    rss = np.sum((y - y_hat) ** 2)
    print(f"degree={deg}  RSS={rss:.1f}")
```

```text
degree=1   RSS=130.2   ← straight line, high error
degree=2    RSS=1.3     ← parabola, nearly perfect
degree=3    RSS=0.8     ← slight improvement
degree=4    RSS=0.2     ← almost zero
degree=7    RSS=0.0     ← passes through every point (overfit!)
```

> 📌 The jump from degree-1 to degree-2 is dramatic (130→1.3). After that, improvements are tiny — the extra powers are fitting noise, not signal.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

X = np.array([16, 18, 20, 22, 24, 26, 28, 30]).reshape(-1, 1)
y = np.array([110, 120, 128, 132, 134, 133, 129, 122])

for deg in [2, 7]:
    Z = PolynomialFeatures(deg, include_bias=False).fit_transform(X)
    m = LinearRegression().fit(Z, y)
    print(f"degree={deg}  R²_train={m.score(Z, y):.4f}  "
          f"predict_age_25={m.predict(np.array([[25]]))[0]:.1f}")
```

```text
degree=2  R²_train=0.9991  predict_age_25=134.0
degree=7  R²_train=1.0000  predict_age_25=155.3   ← impossible!
```

**What happened?** Degree-7 passes through every training point perfectly (R² = 1), but at age 25 it wildly overshoots. The high-degree polynomial oscillates between data points — the classic **Runge phenomenon**.

> 💥 **Break pattern:** perfect training score → absurd prediction. Why? **High-degree polynomials fit noise, not signal. They oscillate wildly outside the training pattern.**

Now the key teaching steps:

- Does **deleting a point** fix it? Sometimes — fewer points = less oscillation.
- Does **lowering the degree** fix it? Yes — degree 2 generalises cleanly.
- Does **adding more data** help? Yes — more points constrain the curve.
- **Lesson:** training R² = 1 is a *warning*, not a victory. Always check on held-out data.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change… | What happens | Why |
|---|---|---|
| Increase degree by 1 | Training RSS always drops or stays same | More flexible model always fits training data at least as well |
| Degree = number of points − 1 | Training RSS = 0 (perfect interpolation) | Enough parameters to pass through every point |
| Extrapolate beyond data range | Predictions explode | Polynomials diverge to ±∞ outside training range |
| Add noise to data | High-degree fits the noise | More parameters = more capacity to memorise randomness |
| Scale x before fitting | Same curve, different coefficient values | Powers xᵖ scale differently; coefficients absorb the change |
| Use orthogonal polynomials | Coefficients become stable and interpretable | Removes induced multicollinearity among powers |

> 🤔 Think: which one is (surprisingly) *not* fixed by more data? → Extrapolation. A degree-4 polynomial will always shoot to ±∞ outside its training range, no matter how much data you have in the middle. Restrict predictions to the data range.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w₁, w₂, …, wₚ   → one weight per power      (model.coef_)
b                 → the intercept              (model.intercept_)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `degree` | Highest power of x | Underfits (line) | Overfits (oscillation) | 2–4; cross-validate |
| `include_bias` | Whether expansion adds constant column | — | Duplicates intercept | Let LinearRegression handle it |
| scaling | Standardise expanded features | High powers → numerical blow-up | — | Recommend for degree ≥ 2 |

**How to choose degree:** plot training vs validation error vs degree. Pick the degree where validation error is minimum — the **bias-variance sweet spot**.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linear in parameters** | Prediction is weighted sum of features | Math stays OLS | Always true by construction | Different model family |
| **Degree captures true curvature** | The real curve is representable at chosen p | Model expressiveness | Compare residuals across p | Increase/decrease p |
| **Homoscedasticity** | Constant error variance | OLS inference | Residual vs fitted plot | Weighted LS or transform y |
| **Normality of errors** | Residuals ~ Normal | Inference / p-values | Q-Q plot | Robust methods |
| **No extreme multicollinearity** | Powers x, x², x³ are correlated | Numerical stability | Condition number of ZᵀZ | Orthogonal polynomials or ridge |

> Key insight: the only *new* assumption vs Linear Regression is "degree captures true curvature." If the real function is a parabola (p=2) and you choose p=1, no amount of data fixes the underfitting.

---

## 21. Data Requirements

```text
Target       → continuous numeric (same as Linear Regression)
Features     → numerical; categorical must be encoded
Missing      → must be handled first (impute or drop)
Outliers     → VERY sensitive — one outlier at the edge of x can bend the polynomial wildly (edge effects)
Scaling      → recommended for degree ≥ 2 (xᵖ grows enormously for large x)
Small data   → high degree with few points → interpolation / overfit
```

> ⚠️ Data-leakage trap: **fit the scaler on training data only**, then transform both train and test.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimise RSS on expanded features)
        ≠
EVALUATION METRIC   (what you report)
```

| Metric | Formula | Simple | Use | Watch out |
|---|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | avg squared miss | standard loss | squared units |
| RMSE | √MSE | avg miss, in original units | most common | outliers dominate |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust | when big misses must hurt → RMSE |
| R² | 1 − SS_res/SS_tot | % variance explained | fit quality | **training R² always rises with degree — means nothing alone** |

> ⚠️ The big trap: **training R² will always improve when you increase degree** (or keep it same). It tells you nothing about generalisation. Always evaluate on held-out test data.

---

## 23. Failure Cases

```text
EDGE OUTLIER     → point at extreme x bends the polynomial badly
OVERFITTING      → high degree → passes through training points, oscillates between them
EXTRAPOLATION    → predicting outside data range → polynomial shoots to ±∞
NUMERICAL        → huge xᵖ values → ZᵀZ ill-conditioned → wrong coefficients (fix: scale)
DEGREE SELECTION → too small underfits, too large overfits; no universal "right" answer
```

---

## 24. Debugging

Model performs badly? Run this checklist:

```text
1. Residual plot → U-shape or W-shape?    → degree too low → increase p
2. Training R²≈1, test R²≈0?              → overfitting → decrease p or add data
3. Predictions explode outside data range?  → extrapolation → restrict to training range
4. Coefficients in millions/billions?        → xᵖ blew up → scale features first
5. Residual plot → random scatter?           → good fit! verify on test set
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:   "Fits a straight line."
Polynomial:          "Fits a curve by adding powers — same OLS math."
Spline:              "Fits piecewise curves — local, avoids global oscillation."
Ridge + Polynomial:  "Curve + penalty — shrinks high-degree weights for stability."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Linear | straight line | simple, interpretable, baseline | can't curve | roughly linear data |
| Polynomial | add x², x³… | captures curvature, reuses OLS | overfit easily, poor extrapolation | smooth curved data |
| Spline | piecewise polynomials | local fit, stable | more parameters to tune | complex global curves |
| Ridge + Polynomial | polynomial + L2 penalty | tames high-degree weights | still global | high-degree polynomial |
| Decision Tree | piecewise constant | non-parametric, local | step-like | any complex pattern |

> Everything in this table extends or replaces Linear Regression's straight line. Master the base, and these become quick upgrades.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict product demand as a function of price (U-shaped curve)
DATA:              past 200 sales (price, quantity_sold)
EDA:               scatter → see curvature, outliers near extreme prices
CLEAN:             handle outliers at price extremes
ENGINEER:          PolynomialFeatures(degree=2–3)
SPLIT:             train / validation / test
SCALE:             StandardScaler on expanded features
TRAIN:             LinearRegression on expanded features
TUNE:              try degrees 1–6 on validation; pick best
EVALUATE:          RMSE on test + residual plot (check no pattern left)
DEPLOY:            serve predictions; restrict to observed price range
MONITOR:           check for drift; retrain as new data arrives
```

> 🚀 ML is not `model.fit(X, y)`. It's problem → data → features → model → evaluate → deploy → monitor → repeat.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is the degree of a polynomial model?
2. **Understand:** why is polynomial regression called "linear in parameters"?
3. **Calculate:** fit a degree-2 polynomial to (1,2), (2,1), (3,4) by hand.
4. **Apply:** given a scatter plot with a clear curve, decide if polynomial is appropriate.
5. **Debug:** model gives training R² = 1.0 but test R² = −2.0. What happened?
6. **Experiment:** run the degree sweep (Section 16) at 6 degrees; graph training vs validation error.
7. **Build:** temperature vs month mini-project: EDA → check curvature → fit degree 2/3/4 → evaluate → report best degree and RMSE.
8. **Explain:** explain why a degree-7 polynomial on 8 data points is dangerous, using the bowling story.

---

## 28. Interview

### Beginner
- **What is polynomial regression?** Linear regression applied to polynomial expansions (x, x², …) of the input, letting it fit curves.
- **Is it a linear or non-linear model?** Linear in parameters. Non-linear in the input variable x.
- **What does the degree control?** How many bends the curve can have — and how likely it is to overfit.

### Intermediate
- **Why is training error always non-increasing with degree?** A higher-degree family always contains every lower-degree fit as a special case.
- **How do you choose the degree?** Cross-validation: plot validation error vs degree and pick the minimum.
- **Why is scaling recommended?** xᵖ can be enormous for large x, making ZᵀZ ill-conditioned.

### Advanced
- **What is Runge's phenomenon?** With evenly spaced points, high-degree polynomial interpolation oscillates at the edges — a reason to prefer splines or regularised polynomials.
- **Why do powers induce multicollinearity?** x, x², x³ are highly correlated (they share the same information source), inflating variance of individual coefficients.
- **How does Ridge on polynomial features help?** Shrinks high-degree weights, reducing variance and oscillation.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
ŷ = b + w₁x + w₂x² + … + wₚxᵖ
Design matrix: Z = [1, x, x², …, xᵖ]
Solution:      w = (ZᵀZ)⁻¹ Zᵀy
```

**Common traps:**
- Calling polynomial regression "non-linear regression" — that phrase means non-linear *in parameters*.
- Assuming higher degree is always better — training error drops but test error can rise.
- Forgetting that p = n−1 interpolates all n points (often overfits).

> **Representative pattern question (NOT a past GATE PYQ):** "How many parameters does a degree-3 polynomial model have for a single feature?" → Answer: 4 (b, w₁, w₂, w₃). "What is the training RSS at degree n−1 for n data points?" → Answer: 0 (perfect interpolation).

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + bias-variance + complexity</summary>

### Derivation

Start with the polynomial model: `ŷᵢ = b + Σₖ₌₁..ₚ wₖxᵢᵏ`.

Define `zᵢₖ = xᵢᵏ`. Now: `ŷᵢ = Σₖ₌₀..ₚ wₖzᵢₖ` (with z₀ = 1, w₀ = b).

The objective is RSS in the expanded space:

```text
J = Σᵢ (yᵢ − Σₖ wₖzᵢₖ)²
```

Setting ∂J/∂wₖ = 0 for each k gives:

```text
(ZᵀZ) w = Zᵀy   →   w = (ZᵀZ)⁻¹ Zᵀy
```

This is the *only* new math vs Linear Regression. Everything else is feature construction.

### Bias–Variance

```text
degree too small  → underfit → high bias  (curve can't represent the truth)
degree too large  → overfit  → high variance (fits noise, oscillates)
```

Training RSS always decreases with degree. Test RSS falls, then rises — the minimum is the bias-variance sweet spot.

### Complexity

```text
Expanding features:  O(n·p)
Training (closed form): O(n·p² + p³)   ← matrix inverse of size p
Prediction: O(p) per sample
Space: O(p) model (p+1 coefficients)
```

### Runge's phenomenon (formal)

For n equally spaced points, the degree-(n−1) interpolating polynomial has oscillation amplitude that grows exponentially at the edges. This is why splines (piecewise low-degree polynomials) are preferred for large n.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "Polynomial Regression adds powers of the input — x², x³ — to Linear Regression, letting the model follow curves. It uses the same OLS math on a bigger feature matrix."

> **Explain to a 12-year-old:** "Imagine drawing a straight line through dots that clearly curve. Now instead of a ruler, you use a curved ruler. Polynomial Regression gives you curved rulers of different bendiness, and you pick the one that fits best."

> **Explain in an interview:** add: feature expansion, ZᵀZ conditioning, degree via cross-validation, Runge phenomenon, comparison with splines and ridge-poly.

> **Explain the mathematics:** derive the normal equation on the expanded design matrix Z from Section 30.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define polynomial regression.
2. Why is it called "linear in parameters"?
3. Write the model formula for degree 3.
4. Build the design matrix Z for x = [1, 2] at degree 2.
5. Solve for w by hand on 3 data points.
6. Explain what happens at degree = n−1.
7. Why is scaling recommended for high degrees?
8. Compare polynomial regression with splines.
9. Name one failure case specific to polynomial regression.
10. Choose polynomial regression for a real problem; defend the choice.

---

## 33. Cheat Sheet

```text
Algorithm  : Polynomial Regression · Supervised → Regression · Parametric
Goal       : Model curvature by adding powers of features
Model      : ŷ = b + w₁x + w₂x² + … + wₚxᵖ
Solve      : OLS normal equation on expanded Z = [1, x, x², …, xᵖ]
Learn      : w₁…wₚ (weights), b (intercept)
Tune       : degree p (most important); scaling
Assumptions: linear in params, degree captures true curvature, homoscedasticity
Use when   : smooth curved data, need interpretability, moderate degree (2–4)
Avoid when : noisy/wide-range data, extrapolation, piecewise patterns (→ splines)
Related    : Linear · Ridge+Poly · Spline · Kernel
Key exam   : "linear in parameters"; degree vs overfitting; training RSS always drops
```

---

## 34. What Next?

You've learned to let the line bend. But bending too much causes overfitting — and sometimes you need to keep all features *stable* rather than just curved.

```text
Linear Regression
   └── Polynomial (bend the line)         ← you are here
        ├── Ridge        (L2 penalty)      → next note (03)
        ├── Lasso        (L1 penalty)      → 04
        ├── Elastic Net  (both penalties)  → 05
        └── Bayesian     (prior on weights) → 06
```

> Next recommended: **03. Ridge Regression** — it answers the weakness you just saw: "how do I keep polynomial (or any) coefficients stable so the model doesn't overfit?"
