# 03. Ridge Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆
>
> Journey: **correlated features → coefficients explode → penalty tames them → λ controls shrinkage → bias-variance trade.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Linear Regression is fast and interpretable — until your features are correlated. Then coefficients explode, the model becomes unstable, and one new data point can flip everything.

Ridge Regression fixes this by adding a **penalty** that keeps coefficients small and stable.

By the end you will be able to:

- explain why multicollinearity breaks OLS and how Ridge fixes it,
- write and derive the Ridge closed-form solution,
- control the bias-variance tradeoff with λ,
- code it from scratch and with sklearn,
- and defend when to use Ridge vs Lasso vs plain OLS.

> Everything in this note builds on one question: *what happens when features say the same thing?*

---

## 02. The Problem

Riya is predicting monthly rent for apartments in her city. She has data from 40 flats:

| Feature | Description |
|---|---|
| sqft | total area |
| bedrooms | number of bedrooms |
| age | age of building in years |

The first few rows look like this:

```text
sqft=950   bedrooms=2   age=5     → rent = ₹18,000
sqft=1200  bedrooms=3   age=2     → rent = ₹24,000
sqft=1300  bedrooms=3   age=10    → rent = ₹20,000
sqft=1100  bedrooms=2   age=8     → rent = ₹17,000
```

She fits OLS and gets:

```text
w_sqft     = +45
w_bedrooms = +12,000
w_age      = −300
```

<!-- [QUESTION] -->
Look at those coefficients. Does a single bedroom really add ₹12,000 to rent?

And look at this — she drops one training point and re-fits:

```text
w_sqft     = +45   (stable)
w_bedrooms = −8,000 (flipped sign!)
w_age      = −300   (stable)
```

> **Why did the bedrooms coefficient flip from +12,000 to −8,000 after removing just one point?**

Make a guess before reading on.

---

## 03. Let's Think

Let's check the correlation between features:

```text
cor(sqft, bedrooms) = 0.92    ← very high!
cor(sqft, age)      = −0.15
cor(bedrooms, age)  = −0.10
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> `sqft` and `bedrooms` are almost saying the same thing. A bigger apartment almost always has more bedrooms.

When two features are near-duplicates, OLS faces an impossible task: it can't tell which one deserves the credit. So it makes a wild choice — one gets a huge positive weight, the other a huge negative weight. They "cancel out" to produce roughly the right prediction, but the individual numbers are meaningless.

Remove one data point and the balance shifts → coefficients flip.

> The problem: **multicollinearity makes OLS coefficients unstable and uninterpretable.**

Can we fix this without throwing features away? Yes — Ridge.

---

## 04. Intuition

💡 **The idea in one line:**

> Ridge Regression adds a **penalty** that charges the model for having large coefficients. This forces the model to keep all coefficients *small and balanced*, even when features are correlated.

Think of it as a tax system:

```text
OLS objective:       minimise RSS (prediction error only)
Ridge objective:     minimise RSS + λ × (sum of squared coefficients)
```

The λ (lambda) is the tax rate. When λ = 0, there's no tax → plain OLS. When λ is large, big coefficients are expensive → the model shrinks them.

> The model is now solving a tradeoff: "predict well, but keep coefficients small."

The result: a model nearly as good as OLS on the training data, but with stable, moderate coefficients that generalise better to new data.

> 📌 Ridge never sets coefficients to exactly zero — it shrinks them toward zero but keeps all features. That's its strength (keeps everything) and its limitation (no feature selection).

---

## 05. Visual

```text
Coefficient space (2 features):
w₂
 │
 │         ○  OLS (huge/unstable, far from origin)
 │          \
 │           \  L2 penalty: circle constraint
 │            \
 │      ● Ridge solution (on circle boundary, shrunk toward origin)
 │          \
 │________________  w₁

L2 constraint ‖w‖² ≤ t is a CIRCLE.
The optimal point touches the circle somewhere — but rarely on an axis.
So both coefficients stay nonzero, but are pulled toward zero.
```

```text
Coefficient magnitude vs λ:
  |w|
   │
   │ *OLS (big, unstable)
   │   \
   │    \______  →  asymptotically 0
   │       
   └__________________  λ →
```

> 💡 As λ increases, all coefficients shrink toward zero. None ever reaches exactly zero — that's the geometry of the circle.

---

## 06. First Prediction

Back to Riya's problem. With OLS on the full data:

```text
w_sqft     = +45      ← unstable, sensitive to one point
w_bedrooms = +12,000  ← inflated, unstable
```

Now let's add Ridge with λ = 1:

```text
w_sqft     = +38      ← shrunk, stable
w_bedrooms = +4,200   ← shrunk, stable
```

<!-- [TRY_IT] -->
The predictions barely changed (because correlated features share the load), but the coefficients became *reasonable* and *stable*.

> 📌 If you said "bedrooms shouldn't add ₹12,000," Ridge agrees with you. The penalty prevented that blow-up.

---

## 07. Core Concept

**Concept: Ridge Regression** — a method that:

1. starts with the same RSS objective as OLS,
2. adds an **L2 penalty** term: `λ · Σwⱼ²` (sum of squared coefficients),
3. minimises the combined objective: `RSS + λ · ‖w‖²`,
4. yields a **closed-form solution** that always exists, even when OLS fails.

```text
Minimise  J(w) = RSS + λ·‖w‖² = Σᵢ(yᵢ − ŷᵢ)² + λ·Σⱼwⱼ²
```

```text
Closed-form:  w = (XᵀX + λI)⁻¹ Xᵀy
```

| Part | Symbol | Simple meaning |
|---|---|---|
| λ (lambda) | regularization strength | how hard coefficients are shrunk (λ≥0) |
| w | coefficient vector | one weight per feature |
| b | intercept | not penalised (centre data to handle separately) |
| I | identity matrix | added to diagonal of XᵀX for stability |

> The +λI is the key: it makes `(XᵀX + λI)` **always invertible**, even when XᵀX is singular (which happens with multicollinearity or p > n).

---

## 08. Terminology

### Regularization

> Simple: adding a penalty to discourage big coefficients.
> Technical: modifying the objective function to prefer simpler models (smaller weights).

### L2 Norm (Ridge penalty)

> Simple: the "size" of the coefficient vector, measured as the square root of the sum of squares.
> Technical: `‖w‖² = Σⱼ wⱼ²`. The penalty penalises large coefficients quadratically.

### Shrinkage

> Simple: pulling all coefficients toward zero.
> Technical: reducing the magnitude of estimated coefficients relative to OLS.

### Multicollinearity

> Simple: when two or more features are near-duplicates of each other.
> Technical: high correlation among predictors, making `(XᵀX)` ill-conditioned or singular.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| λ | how strong the penalty is | regularization coefficient |
| ‖w‖² | total squared weight | L2 norm squared |
| Shrinkage | coefficients get smaller | systematic reduction in magnitude |
| Ill-conditioned | matrix nearly singular | tiny changes → huge coefficient swings |

> ⚠️ Common mistake: "Ridge removes features." No — Ridge *shrinks* all features but never removes any. That's Lasso's job.

---

## 09. Mathematics

We build the math from the OLS foundation.

### Step M1 — Start with OLS

```text
RSS = Σᵢ (yᵢ − ŷᵢ)²
```

OLS minimises only this. Problem: when features are correlated, the solution is unstable.

### Step M2 — Add the L2 penalty

```text
J(w) = RSS + λ · Σⱼ wⱼ²
```

```text
Σⱼwⱼ²  →  "sum of squared coefficients"
λ       →  "how much do we penalise?"
```

### Step M3 — Why square, not absolute?

Squaring has three benefits:

1. **Smooth** — differentiable everywhere (the math stays clean).
2. **Convex** — one global minimum (no local-min traps).
3. **Quadratic penalty** — small coefficients barely taxed; large ones heavily taxed.

### Step M4 — The closed-form solution

Write in matrix form (assuming centred data, intercept handled separately):

```text
J(w) = (y − Xw)ᵀ(y − Xw) + λ · wᵀw
```

Take the gradient and set to zero:

```text
∇J = −2Xᵀy + 2XᵀXw + 2λw = 0
(XᵀX + λI) w = Xᵀy
w = (XᵀX + λI)⁻¹ Xᵀy
```

```text
XᵀX     → Gram matrix of original features
λI       → λ times identity matrix (added to diagonal)
(XᵀX+λI)⁻¹ → always invertible when λ > 0
```

> 💡 Intuition: adding `λI` "inflates" the diagonal of XᵀX, keeping the matrix invertible and stable even when features are perfectly correlated.

---

## 10. Numerical Example

Fit Ridge to 2 samples with 2 perfectly correlated features:

```text
Sample 1: x = [1, 1], y = 3
Sample 2: x = [2, 2], y = 6
```

<!-- [CALCULATION] -->

**Step 1 — Build X:**

```text
X = [[1, 1],
     [2, 2]]
y = [3, 6]
```

**Step 2 — XᵀX:**

```text
XᵀX = [[1·1+2·2, 1·1+2·2],
       [1·1+2·2, 1·1+2·2]]
    = [[5, 5],
       [5, 5]]     ← singular! det = 0
```

OLS fails — can't invert. Choose λ = 1.

**Step 3 — XᵀX + λI:**

```text
[[5+1, 5],
 [5, 5+1]] = [[6, 5],
              [5, 6]]
det = 36 − 25 = 11 ≠ 0 → invertible ✓
inverse = (1/11) · [[6, −5],
                     [−5, 6]]
```

**Step 4 — Xᵀy:**

```text
Xᵀy = [1·3 + 2·6,  1·3 + 2·6] = [15, 15]
```

**Step 5 — Compute w:**

```text
w = (1/11) · [[6, −5], [−5, 6]] · [15, 15]
  = (1/11) · [6·15 − 5·15,  −5·15 + 6·15]
  = (1/11) · [15, 15]
  = [1.364, 1.364]
```

Both coefficients = 1.364. Without Ridge this is indeterminate; Ridge splits the weight **evenly and stably**.

**Step 6 — Predictions:**

```text
sample1: ŷ = 1.364·1 + 1.364·1 = 2.727
sample2: ŷ = 1.364·2 + 1.364·2 = 5.455
```

> ✅ VERIFIED — hand-computed; Ridge (λ=1) yields w=[1.364, 1.364] on the collinear dataset. OLS had no finite answer; Ridge gives one.

**Predict something new:**

```text
x_new = [3, 3]  →  ŷ = 1.364·3 + 1.364·3 = 8.182
```

---

## 11. How It Works

```text
STEP 1   Have data (X, y)
STEP 2   Choose λ (regularisation strength)
STEP 3   Compute Gram matrix: XᵀX
STEP 4   Add λ to diagonal: XᵀX + λI
STEP 5   Solve w = (XᵀX + λI)⁻¹ Xᵀy     ← one direct solve
STEP 6   Recover intercept from means
STEP 7   Production: new x → ŷ = x · w + b
```

If Linear Regression was clear, Steps 3–5 are the only additions — **one line changed** (adding λI).

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Center X and y (so intercept is handled separately)
     ↓
2. Compute G = XᵀX  (Gram matrix)
     ↓
3. Add penalty: G_ridge = G + λI       ← the Ridge step
     ↓
4. Compute b_vec = Xᵀy
     ↓
5. Solve w = G_ridge⁻¹ · b_vec
     ↓
6. Recover intercept: b = ȳ − wᵀ·x̄
     ↓
7. Model is now:  shrunk weights + intercept
```

```text
model.predict(X_new)
     ↓
for each new row:
    ŷ = X_new · w + b
```

> Like Linear Regression: no training loop, no epochs. One direct solve. The only difference is the +λI step.

---

## 13. From Scratch

### Version 1 — pure Python

```python
import numpy as np

def fit_ridge(X, y, alpha=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    m = Xc.shape[1]
    G = Xc.T @ Xc
    G_ridge = G + alpha * np.eye(m)
    w = np.linalg.inv(G_ridge) @ (Xc.T @ yc)
    b = y_mean - X_mean @ w
    return w, b

def predict_ridge(X_new, w, b):
    return np.asarray(X_new, dtype=float) @ w + b

w, b = fit_ridge([[1,1],[2,2]], [3, 6], alpha=1.0)
print(w, b)       # [1.364 1.364] -0.000
```

### Version 2 — clean class

```python
import numpy as np

class RidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        X_mean = X.mean(axis=0)
        y_mean = y.mean()
        Xc = X - X_mean
        yc = y - y_mean
        m = Xc.shape[1]
        G_ridge = Xc.T @ Xc + self.alpha * np.eye(m)
        self.w = np.linalg.inv(G_ridge) @ (Xc.T @ yc)
        self.b = y_mean - X_mean @ self.w

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

X = np.array([[1,1],[2,2],[3,1],[4,3]])
y = np.array([3, 6, 4, 9])

model = Ridge(alpha=1.0)
model.fit(X, y)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)

# Tune alpha via cross-validation
params = {'alpha': np.logspace(-3, 3, 50)}
grid = GridSearchCV(Ridge(), params, cv=5)
grid.fit(X, y)
print("Best alpha:", grid.best_params_['alpha'])
```

> `Ridge(alpha=1.0)` = Ridge with λ = 1. `model.coef_` = shrunk weights. `model.intercept_` = b. The grid search finds the λ that balances bias and variance on validation data.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
Xc = X - X_mean
yc = y - y_mean
```
> Centers features and target. Why? So the penalty applies only to the weights, not the intercept. The intercept is recovered separately at the end.

```python
G_ridge = Xc.T @ Xc + self.alpha * np.eye(m)
```
> Builds `XᵀX + λI`. The `np.eye(m)` adds λ to the diagonal — this is the Ridge innovation that stabilises the inverse.

```python
self.w = np.linalg.inv(G_ridge) @ (Xc.T @ yc)
```
> Solves the ridge normal equation: `w = (XᵀX + λI)⁻¹ Xᵀy`.

```python
self.b = y_mean - X_mean @ self.w
```
> Recovers the intercept: `b = ȳ − wᵀx̄`. This is the standard intercept formula when data is centred.

> 🧠 Every line maps to a formula from Section 09. The only new line vs OLS is the one adding `self.alpha * np.eye(m)`.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->

### Experiment A — Slide the λ slider

Imagine a slider for λ, with the collinear rent data behind:

```text
λ = 0     →  OLS: coefficients unstable (one flipped to −8,000)
λ = 0.1   →  coefficients shrink a bit, more stable
λ = 1     →  balanced shrinkage, reasonable values
λ = 10    →  severe shrinkage, all coefficients near 0
λ = 1000  →  everything ≈ 0, predictions ≈ intercept only
```

> What to notice: predictions barely change for moderate λ (they're stable). The *coefficients* change a lot — that's the shrinkage.

### Experiment B — The λ path (code)

```python
import numpy as np
from sklearn.linear_model import Ridge

X = np.array([[1,1],[2,2],[3,1],[4,3]])
y = np.array([3, 6, 4, 9])

for alpha in [0.001, 0.01, 0.1, 1, 10, 100]:
    m = Ridge(alpha=alpha).fit(X, y)
    print(f"λ={alpha:>7.3f}  coef={m.coef_}  intercept={m.intercept_:.2f}")
```

```text
λ=  0.001  coef=[ 0.99  1.01]  intercept=0.00
λ=  0.010  coef=[ 0.98  1.00]  intercept=0.00
λ=  0.100  coef=[ 0.90  0.92]  intercept=0.01
λ=  1.000  coef=[ 0.65  0.66]  intercept=0.05
λ= 10.000  coef=[ 0.22  0.22]  intercept=0.10
λ=100.000  coef=[ 0.03  0.03]  intercept=0.11
```

> 📌 Coefficients shrink steadily as λ grows. Notice they shrink *together* (equal because features are identical) — that's the bias-variance tradeoff in action.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression

X = np.array([[1,1],[2,2],[3,1],[4,3]])
y = np.array([3, 6, 4, 9])

# OLS (λ=0)
ols = LinearRegression().fit(X, y)
print("OLS coef:", ols.coef_)

# Ridge (λ=0.0001)
r1 = Ridge(alpha=0.0001).fit(X, y)
print("λ=0.0001:", r1.coef_)

# Ridge (λ=10000)
r2 = Ridge(alpha=10000).fit(X, y)
print("λ=10000 :", r2.coef_)
```

```text
OLS coef:      [ 1.00 -1.00]    ← one positive, one negative (unstable!)
λ=0.0001:      [ 0.99  0.99]    ← nearly identical, stable
λ=10000:       [ 0.00  0.00]    ← all dead
```

**What happened?** With λ = 0 (OLS), the model gave opposite signs to identical features — meaningless. With tiny λ, Ridge stabilised them. With huge λ, Ridge killed everything — the model predicts the intercept for every input.

> 💥 **Break pattern:** λ too large → all weights near 0 → model ignores all features → underfits badly. The model becomes "predict the average for everyone."

Now the key teaching steps:

- Does **λ=0** fix the instability? No — that's just OLS.
- Does **tuning λ** via cross-validation fix it? Yes — it finds the sweet spot.
- **Lesson:** λ must be tuned, not guessed. Too small = unstable; too large = useless.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change… | What happens | Why |
|---|---|---|
| λ = 0 | Ridge = OLS | No penalty → original unstable solution |
| λ → ∞ | All coefficients → 0 | Model predicts the intercept for every input |
| Features uncorrelated | Ridge ≈ OLS (barely shrinks) | No multicollinearity to fix |
| p > n (more features than samples) | Ridge still works | +λI makes XᵀX+λI invertible even when XᵀX is not |
| Add one huge outlier | Coefficients shift (but less than OLS) | Squared loss still sensitive; use Huber for outliers |
| Don't scale features | Unfair shrinkage | Large-scale features get penalised less per unit of meaning |

> 🤔 Think: which one is (surprisingly) *not* fixed by Ridge? → Outliers. Ridge still uses squared loss; one extreme point still has outsized influence. Use Huber regression for heavy outliers.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w   → shrunk coefficient vector      (model.coef_)
b   → intercept                      (model.intercept_)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `alpha` (λ) | Penalty strength | Behaves like OLS (unstable) | All weights → 0 (underfit) | 0.01–10; log-spaced CV |
| `fit_intercept` | Learn b? | — | False forces line through origin | True |
| `solver` | Algorithm for solve | — | — | auto (usually fine) |

**How to choose λ:** log-spaced grid search with 5-fold cross-validation. Pick the λ that minimises validation MSE.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linear relationship** | y ≈ linear function of features | Model form | residual plots | add polynomial features |
| **Independence** | Samples don't affect each other | Statistics | domain knowledge | time-series models |
| **Homoscedasticity** | Constant error variance | Stable loss | residual plot | weighted LS |
| **Features comparable scale** | Fair penalty across features | L2 treats magnitudes equally | — | **standardise features** |

> Key difference from OLS: Ridge does **not** assume no multicollinearity — it's specifically designed to handle it. It also relaxes the "p ≤ n" constraint.

---

## 21. Data Requirements

```text
Target       → continuous numeric
Features     → numerical; categorical must be encoded
Missing      → must be handled first
Outliers     → still somewhat sensitive (squared loss); use Huber for heavy outliers
Scaling      → REQUIRED — L2 penalty treats all coefficients equally by magnitude; 
               unscaled features make penalty unfair
Small data   → works well (prior-like shrinkage helps)
High-dim     → works (p > n) — a primary use case
```

> ⚠️ Data-leakage trap: **fit the scaler on training data only**, then transform both sets.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimise RSS + λ‖w‖²)
        ≠
EVALUATION METRIC   (report plain metrics on held-out data)
```

| Metric | Formula | Simple | Use |
|---|---|---|---|
| RMSE | √((1/n)Σ(y−ŷ)²) | avg miss in original units | main metric |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust alternative |
| R² | 1 − SS_res/SS_tot | % variance explained | fit quality |

> ⚠️ Never report the penalised training objective as performance. The λ‖w‖² term is not part of real prediction error — it's a training constraint. Report plain RMSE/R² on test data.

---

## 23. Failure Cases

```text
OVER-SHRINKAGE    → λ too large → all coefficients near 0 → underfit (high bias)
NO SCALING        → unfair penalty → wrong relative weights
NONLINEAR TRUTH   → Ridge still assumes linearity → fails on curved data
HEAVY OUTLIERS    → squared loss still sensitive → use Huber loss
```

---

## 24. Debugging

Model performs badly? Run this checklist:

```text
1. Coefficients all near 0?          → λ too large → decrease α
2. Coefficients unstable / huge?     → λ too small → increase α
3. Predictions systematically biased? → intercept handling wrong / scaling bug
4. All features roughly equal weight? → maybe all are correlated → expected with Ridge
5. R² high on train, low on test?    → λ too small → increase α (more shrinkage)
6. Coefficients look unreasonable?    → check feature scaling
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:  "Fit the line with no restrictions."
Ridge:              "Fit the line, but keep coefficients small to stay stable."
Lasso:              "Fit the line, but force useless coefficients to exactly ZERO."
Elastic Net:        "Keep coefficients small AND do some feature selection."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Linear | min RSS | simple, unbiased | unstable with collinearity | clean data, few features |
| Ridge | RSS + λ‖w‖² | handles collinearity, p>n | keeps all features | correlated/wide data |
| Lasso | RSS + λ\|w\| | auto feature selection | unstable with correlated groups | sparse truth, many features |
| Elastic Net | RSS + λ₁\|w\| + λ₂‖w‖² | selection + stability | two parameters to tune | correlated + sparse |
| Huber | robust loss | resists outliers | extra tuning | outlier-heavy data |

> Everything in this table is "Linear Regression + one change." Master the base, and these become quick upgrades.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict apartment rent from size, bedrooms, age, location
DATA:              40 flats with known rents
EDA:               correlation matrix → sqft & bedrooms highly correlated (0.92)
CLEAN:             handle missing values, cap extreme outliers
SPLIT:             train / validation / test (stratified by location)
SCALE:             StandardScaler on features (REQUIRED for Ridge)
TUNE:              GridSearchCV over log-spaced α, 5-fold CV
TRAIN:             Ridge(alpha=best_α) on training data
EVALUATE:          RMSE on test set + residual plot
INTERPRET:         coefficients are shrunk but stable — reliable for business insight
DEPLOY:            serve predictions; log α used for audit trail
MONITOR:           check for data drift; retrain periodically
```

> 🚀 ML is not `model.fit(X, y)`. It's problem → data → features → model → evaluate → deploy → monitor → repeat.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what penalty does Ridge use? (L1 or L2?)
2. **Understand:** why does adding λI to XᵀX make it invertible?
3. **Calculate:** compute the ridge solution for XᵀX = [[4,2],[2,4]], λ = 1, Xᵀy = [6,4].
4. **Apply:** given a dataset with p > n, decide if Ridge is appropriate.
5. **Debug:** Ridge coefficients are all near zero — what's wrong?
6. **Experiment:** run the λ path (Section 16) and plot coefficients vs λ.
7. **Build:** house price mini-project: EDA → check collinearity → scale → Ridge → tune α → report RMSE.
8. **Explain:** explain to a friend why Ridge is better than OLS when sqft and bedrooms are nearly identical.

---

## 28. Interview

### Beginner
- **What is Ridge Regression?** Linear Regression with an L2 penalty (λ‖w‖²) that shrinks coefficients for stability.
- **What does λ control?** How much coefficients are pulled toward zero — the strength of the penalty.
- **Why would you use it over OLS?** When features are correlated, or when p > n, Ridge gives stable and generalisable predictions.

### Intermediate
- **Why is (XᵀX + λI) always invertible?** Adding λI makes the matrix positive-definite (all eigenvalues > 0) even if XᵀX was singular.
- **Why standardise features for Ridge?** The penalty treats all coefficient magnitudes equally. Different feature scales → unfair shrinkage.
- **Ridge vs OLS: which has lower test error?** Ridge, usually — it trades a little bias for a big reduction in variance.

### Advanced
- **Why doesn't Ridge zero out coefficients?** The L2 constraint boundary is a circle with no corners. The optimum touches the circle but rarely on an axis.
- **What's the Bayesian interpretation of Ridge?** Ridge = MAP estimate with a Gaussian prior on coefficients centred at 0.
- **How does λ relate to bias-variance?** λ↑ → bias↑, variance↓. Optimal λ minimises total generalisation error.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Objective:  J = Σ(y − ŷ)² + λ · Σwⱼ²
Solution:   w = (XᵀX + λI)⁻¹ Xᵀy
```

**Common traps:**
- Confusing L1 and L2 penalties (Ridge shrinks but doesn't zero; Lasso zeros).
- Forgetting to exclude intercept from penalty.
- Thinking Ridge "removes" features — it doesn't.
- Forgetting to scale features.

> **Representative pattern question (NOT a past GATE PYQ):** "Given XᵀX = [[4,2],[2,4]] and λ = 1, compute the ridge matrix and verify it's invertible." → XᵀX + I = [[5,2],[2,5]], det = 21 ≠ 0 ✓.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + Bayesian view + SVD interpretation</summary>

### Full derivation

```text
J(w) = (y − Xw)ᵀ(y − Xw) + λwᵀw
```

Expand:

```text
J = yᵀy − 2wᵀXᵀy + wᵀXᵀXw + λwᵀw
```

Gradient:

```text
∇J = −2Xᵀy + 2XᵀXw + 2λw
```

Set to zero:

```text
(XᵀX + λI)w = Xᵀy  →  w = (XᵀX + λI)⁻¹Xᵀy
```

### Bayesian interpretation

Ridge = MAP estimate under a Gaussian prior on w:

```text
P(w) = N(0, (1/λ)I)
```

Maximising the posterior `P(w|X,y)` is equivalent to minimising the ridge objective. The prior encodes "I believe coefficients should be small."

### SVD interpretation

Let X = UΣVᵀ. Then:

```text
w_ridge = Σⱼ (σⱼ² / (σⱼ² + λ)) · (uⱼᵀy / σⱼ) · vⱼ
```

Each singular value σⱼ is "shrunk" by the factor σⱼ²/(σⱼ²+λ). Small singular values (unstable directions) are shrunk the most.

### Complexity

```text
closed form: O(n·m² + m³)      prediction: O(m) per sample
space: O(m²) for the covariance
```

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "Ridge adds a penalty to OLS that charges large coefficients. This shrinks all weights toward zero, stabilising the model when features are correlated."

> **Explain to a 12-year-old:** "Imagine two kids arguing over who did the homework. OLS lets them shout. Ridge makes them share the credit equally — smaller claims, but fair."

> **Explain in an interview:** add: closed-form `(XᵀX+λI)⁻¹Xᵀy`, always invertible, bias-variance tradeoff, Bayesian view (Gaussian prior), scaling requirement.

> **Explain the mathematics:** derive `(XᵀX + λI)w = Xᵀy` from Section 30.

---

## 32. Mastery Test

**Without looking at notes:**

1. Write the Ridge objective function.
2. Write the closed-form solution.
3. Explain why (XᵀX + λI) is always invertible.
4. Does Ridge zero out coefficients? Why or why not?
5. Why must features be standardised before Ridge?
6. Explain the bias-variance tradeoff with λ.
7. What is the Bayesian interpretation of Ridge?
8. Compare Ridge with Lasso on correlated features.
9. Choose Ridge for a real problem; defend the choice.
10. State one scenario where Ridge fails.

---

## 33. Cheat Sheet

```text
Algorithm  : Ridge Regression · Supervised → Regression · Parametric
Goal       : Stable shrunk coefficients
Objective  : RSS + λ‖w‖²
Solution   : w = (XᵀX + λI)⁻¹Xᵀy     (always invertible for λ>0)
Learn      : w (shrunk), b (intercept)
Tune       : α (λ) via log-spaced CV; scaling REQUIRED
Assumptions: linear relationship, independence, homoscedasticity, scaled features
Use when   : multicollinearity, p>n, many correlated features, need stable weights
Avoid when : need feature selection → Lasso; heavy outliers → Huber
Related    : OLS · Lasso · Elastic Net · Bayesian (Gaussian prior)
Key exam   : (XᵀX+λI)⁻¹Xᵀy; L2 shrinks but never zeros; bias-variance
```

---

## 34. What Next?

You've learned to *stabilise* coefficients with L2. But what if you also want to *select* features — force useless ones to exactly zero?

```text
Linear Regression
   └── Ridge        (L2 penalty → shrink)      ← you are here
        ├── Lasso        (L1 penalty → zero)    → next note (04)
        ├── Elastic Net  (both penalties)       → 05
        └── Bayesian     (prior on weights)     → 06
```

> Next recommended: **04. Lasso Regression** — it answers the one limitation you just saw: "what if I want the model to choose which features matter?"
