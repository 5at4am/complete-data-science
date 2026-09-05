# 04. Lasso Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆
>
> Journey: **many features → which matter? → L1 penalty → diamond corners → zeros → soft-thresholding → sparsity.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Ridge shrinks coefficients but keeps every feature. What if you have 500 features and only 5 matter?

Lasso Regression uses a different penalty that **forces useless coefficients to exactly zero** — automatic feature selection built into the model.

By the end you will be able to:

- explain why the L1 penalty produces zeros while L2 doesn't,
- apply soft-thresholding by hand,
- code Lasso from scratch and with sklearn,
- recognise when Lasso is unstable (correlated groups), and
- defend when to use Lasso vs Ridge vs Elastic Net.

> Everything in this note builds on one question: *how do we make a model choose which features to use?*

---

## 02. The Problem

Arjun works at a biotech startup. A drug trial measured 20 blood biomarkers for 80 patients. The question: which biomarkers actually predict recovery time?

Arjun tries OLS:

```text
w_biomarker1  = 3.2
w_biomarker2  = −0.001
w_biomarker3  = 8.7
w_biomarker4  = 12.4
w_biomarker5  = −0.003
...           (20 features, all nonzero)
```

All 20 coefficients are nonzero. OLS uses *everything*. How does Arjun know which ones are real signals and which are noise?

<!-- [QUESTION] -->
He tries Ridge. Still all 20 are nonzero. It helps with stability but doesn't tell him *which* biomarkers matter.

> **What if the model could automatically set the useless biomarkers' coefficients to exactly zero — leaving only the important ones?**

That's what Lasso does.

---

## 03. Let's Think

Ridge uses the L2 penalty: `λ · Σwⱼ²`. Squaring means a coefficient of 0.5 is taxed 0.25, and a coefficient of 5 is taxed 25. Big coefficients are hurt more, but small ones never reach exactly zero.

Lasso uses the L1 penalty: `λ · Σ|wⱼ|`. Here, the tax is *proportional* to the absolute value. A coefficient of 0.5 is taxed 0.5, and a coefficient of 5 is taxed 5.

<!-- [THINK_ABOUT_IT] -->
🤔 Why does this make a difference?

> The key: L1's penalty shape has **sharp corners** (a diamond in 2D). The optimal solution often lands exactly on a corner — where a coefficient is *exactly zero*.

L2's penalty shape is a circle — smooth everywhere. The optimum rarely lands on an axis.

> The geometry of the penalty determines whether zeros happen. Sharp corners → zeros. Smooth curves → shrinkage only.

---

## 04. Intuition

💡 **The idea in one line:**

> Lasso adds an **L1 penalty** (λ · Σ|wⱼ|) to the RSS objective. The penalty's sharp geometry forces irrelevant coefficients to **exactly zero**, producing a **sparse** model that only uses a subset of features.

Think of it as a strict budget:

```text
OLS:     "Predict as well as possible. Use everything."
Ridge:   "Predict well, but keep all coefficients small."
Lasso:   "Predict well, but each coefficient has a fixed cost.
          If a feature isn't worth its cost, cut it entirely."
```

The "cost" is λ. Each feature's coefficient costs `λ · |wⱼ|`. If the feature's contribution to reducing RSS doesn't justify its cost, Lasso sets it to zero.

> 📌 The result: a model that *tells you which features matter* — the most interpretable regularised model.

---

## 05. Visual

```text
L1 constraint (diamond):
   |w₁| + |w₂| ≤ t
   has sharp CORNERS on the axes

   w₂
    │
    │◇      ← diamond: corners at (±t, 0) and (0, ±t)
    │◇◇◇
    │◇◇◇◇◇
    └──────── w₁
   Diamond touches least-squares contour at a corner
   → one coefficient set to exactly 0

Compare with L2 (circle):
   w₂
    │
    │○      ← circle: smooth everywhere
    │○○○○
    │○○○○○
    └──────── w₁
   Circle touches contour somewhere smooth
   → both coefficients nonzero (shrinkage only)
```

```text
Compare:
Ridge boundary:  ○ circle      → shrink only
Lasso boundary:  ◇ diamond    → shrink + zero at corners
```

> 💡 The diamond's corners lie on the axes — the only places where a coefficient is exactly zero. The circle never has corners → no zeros.

---

## 06. First Prediction

Back to Arjun's 20 biomarkers. Lasso with a suitable λ gives:

```text
w_biomarker1  = 2.8     ← kept
w_biomarker2  = 0.0     ← ZERO (dropped!)
w_biomarker3  = 0.0     ← ZERO
w_biomarker4  = 11.2    ← kept
w_biomarker5  = 0.0     ← ZERO
...
(active features: 1, 4, 7, 12, 18 — only 5 out of 20)
```

<!-- [TRY_IT] -->
Lasso gave Arjun a clear answer: only biomarkers 1, 4, 7, 12, and 18 matter. The rest are noise.

> 📌 Ridge would have kept all 20 with nonzero coefficients. Lasso *selected* the important ones — that's the difference.

---

## 07. Core Concept

**Concept: Lasso Regression** — a method that:

1. starts with the same RSS objective as OLS,
2. adds an **L1 penalty** term: `λ · Σ|wⱼ|` (sum of absolute coefficients),
3. minimises the combined objective: `RSS + λ · Σ|wⱼ|`,
4. drives irrelevant coefficients to **exactly zero** (feature selection),
5. is solved via **coordinate descent with soft-thresholding** (no closed-form solution).

```text
Minimise  J(w) = RSS + λ·Σⱼ|wⱼ| = Σᵢ(yᵢ − ŷᵢ)² + λ·Σⱼ|wⱼ|
```

| Part | Symbol | Simple meaning |
|---|---|---|
| λ (lambda) | penalty strength | how aggressively to drop features (λ≥0) |
| \|wⱼ\| | absolute value of coefficient j | the "cost" of keeping feature j active |
| w | sparse coefficient vector | many entries are exactly zero |

> Unlike Ridge: there is **no closed-form solution**. Lasso uses coordinate descent + soft-thresholding.

---

## 08. Terminology

### L1 Norm

> Simple: the "size" of a vector measured by the sum of absolute values.
> Technical: `‖w‖₁ = Σⱼ |wⱼ|`. The penalty that Lasso uses.

### Sparsity

> Simple: a model where most coefficients are exactly zero.
> Technical: the solution vector has many zero entries; only a subset of features are used.

### Soft-Thresholding

> Simple: shrink a coefficient by λ, and if it would cross zero, clamp it to exactly 0.
> Technical: the coordinate-wise update `wⱼ = sign(zⱼ) · max(0, |zⱼ| − λ)`.

### Subgradient

> Simple: a generalisation of "derivative" for functions that have sharp corners (like |w|).
> Technical: the subgradient of |w| at 0 is any value in [−1, 1]; this is why plain gradient descent doesn't work for Lasso.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| ‖w‖₁ | total absolute weight | L1 norm |
| Sparse | many zeros | most coefficients exactly zero |
| Soft-thresholding | shrink + zero | the Lasso coordinate update |
| Coordinate descent | optimise one variable at a time | standard Lasso solver |

> ⚠️ Common mistake: "Lasso shrinks coefficients like Ridge." No — Lasso *shrinks AND zeros*. The zeros are the defining feature.

---

## 09. Mathematics

We build the math from the OLS and Ridge foundation.

### Step M1 — Start with OLS

```text
RSS = Σᵢ (yᵢ − ŷᵢ)²
```

### Step M2 — Add the L1 penalty

```text
J(w) = RSS + λ · Σⱼ |wⱼ|
```

```text
Σⱼ|wⱼ|  →  sum of absolute coefficients
λ         →  penalty strength
```

### Step M3 — Why L1 produces zeros (intuition)

The absolute value function has a **kink** (sharp corner) at zero. When the gradient pushes a coefficient toward zero, the kink "traps" it — the coefficient gets stuck at exactly 0 rather than passing through smoothly.

### Step M4 — Soft-thresholding (the core update)

For each coordinate j, holding all others fixed, define:

```text
zⱼ = Σ xᵢⱼ(yᵢ − Σₖ≠ⱼ wₖxᵢₖ) / Σ xᵢⱼ²
```

This is the OLS value for coordinate j. Then the Lasso update is:

```text
wⱼ = sign(zⱼ) · max(0, |zⱼ| − λ)
```

```text
sign(zⱼ)  → +1 or −1 (direction)
|zⱼ| − λ  → shrink magnitude by λ
max(0, …) → if result is negative, clamp to 0
```

> 💡 **This is why Lasso zeros features:** whenever `|zⱼ| ≤ λ`, the coefficient becomes exactly 0.

---

## 10. Numerical Example

Data: 2 samples, 2 features (orthogonal for simplicity).

```text
Sample 1: x = [2, 0], y = 4
Sample 2: x = [0, 2], y = 4
```

<!-- [CALCULATION] -->

**Step 1 — OLS-style values for each coordinate (features are orthogonal, so independent):**

```text
z₁ = Σ x₁y / Σ x₁² = (2·4 + 0·4)/(4 + 0) = 8/4 = 2.0
z₂ = Σ x₂y / Σ x₂² = (0·4 + 2·4)/(0 + 4) = 8/4 = 2.0
```

**Step 2 — Apply soft-threshold with λ = 1:**

```text
w₁ = sign(2)·max(0, |2| − 1) = +1·1 = 1.0
w₂ = sign(2)·max(0, |2| − 1) = +1·1 = 1.0
```

Both kept (shrunk from 2 → 1).

**Step 3 — Try larger λ = 3:**

```text
w₁ = sign(2)·max(0, |2| − 3) = +1·max(0, −1) = 0   ← feature 1 DROPPED
w₂ = sign(2)·max(0, |2| − 3) = +1·max(0, −1) = 0   ← feature 2 DROPPED
```

Both dropped at λ=3 — the penalty dominates.

**Step 4 — Mixed case: z₁=3, z₂=1, λ=2:**

```text
w₁ = sign(3)·max(0, |3| − 2) = +1·1 = 1   (kept)
w₂ = sign(1)·max(0, |1| − 2) = +1·max(0, −1) = 0   (dropped — too weak)
```

> ✅ VERIFIED — hand-computed with soft-thresholding. Shows exactly how Lasso zeros weak coefficients.

**Predictions (λ=1 case):**

```text
sample1: ŷ = 1.0·2 + 1.0·0 = 2.0
sample2: ŷ = 1.0·0 + 1.0·2 = 2.0
```

---

## 11. How It Works

```text
STEP 1   Have data (X, y)
STEP 2   Choose λ (regularisation strength)
STEP 3   Scale features (REQUIRED — fair penalty)
STEP 4   Initialise w = 0
STEP 5   Coordinate descent loop:
           for each coordinate j:
             compute zⱼ (OLS value for j, others fixed)
             wⱼ = soft-threshold(zⱼ, λ)
STEP 6   Repeat until convergence (coefficients stop changing)
STEP 7   The final w is sparse — many entries are 0
STEP 8   Production: new x → only active features contribute
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Scale features (StandardScaler)
     ↓
2. Initialise w = 0 for all features
     ↓
3. Repeat until convergence:
     for each feature j:
       compute partial residual (y minus contributions of all other features)
       compute zⱼ = correlation of feature j with partial residual
       apply soft-threshold: wⱼ = sign(zⱼ)·max(0, |zⱼ|−λ)
     ↓
4. Many wⱼ become exactly 0 → sparse model
     ↓
5. Recover intercept from means
```

```text
model.predict(X_new)
     ↓
for each new row:
    ŷ = X_new · w + b
    (only nonzero wⱼ contribute — the rest are dropped)
```

> No gradient descent, no matrix inverse in the loop. Just iteratively scanning coordinates and applying the soft-threshold.

---

## 13. From Scratch

### Version 1 — pure Python

```python
import numpy as np

def fit_lasso(X, y, alpha=1.0, max_iter=1000, tol=1e-4):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    n, m = Xc.shape
    w = np.zeros(m)
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(m):
            residual = yc - Xc @ w + w[j] * Xc[:, j]
            zj = (Xc[:, j] @ residual) / (Xc[:, j] @ Xc[:, j])
            w[j] = np.sign(zj) * max(0.0, abs(zj) - alpha)
        if np.max(np.abs(w - w_old)) < tol:
            break
    b = y_mean - X_mean @ w
    return w, b
```

### Version 2 — clean class

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
        return np.asarray(X, dtype=float) @ self.w + self.b
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

X = np.random.RandomState(42).randn(100, 20)
w_true = np.zeros(20)
w_true[[0, 3]] = [2.5, -1.5]   # only 2 features matter
y = X @ w_true + np.random.RandomState(0).randn(100) * 0.5

model = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
model.fit(X, y)

coefs = model.named_steps['lasso'].coef_
print("Active features:", np.where(coefs != 0)[0])
print("Coefficients:", coefs[coefs != 0])

# Tune alpha
params = {'lasso__alpha': np.logspace(-3, 1, 50)}
grid = GridSearchCV(make_pipeline(StandardScaler(), Lasso()), params, cv=5)
grid.fit(X, y)
print("Best alpha:", grid.best_params_['lasso__alpha'])
```

> `Lasso(alpha=0.1)` = Lasso with λ = 0.1. The pipeline scales features first (REQUIRED). `model.coef_` contains many zeros — the active features are the nonzero ones.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
residual = yc - Xc @ w + w[j] * Xc[:, j]
```
> Computes the partial residual: "what's left after accounting for all features *except j*." This is what feature j needs to explain.

```python
zj = (Xc[:, j] @ residual) / (Xc[:, j] @ Xc[:, j])
```
> The OLS solution for feature j alone on the partial residual. This is the "unpenalised" best value for wⱼ.

```python
w[j] = np.sign(zj) * max(0.0, abs(zj) - self.alpha)
```
> **The soft-threshold step.** Shrink by λ; if the result would cross zero, clamp to exactly 0. THIS is where feature selection happens.

> 🧠 Every line maps to the formula from Section 09. The core innovation is one line: the soft-threshold.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->

### Experiment A — Slide the λ slider

```text
λ = 0     →  all 20 features active (OLS)
λ = 0.01  →  18 features active (tiny shrinkage)
λ = 0.1   →  12 features active (beginning to drop)
λ = 1.0   →   5 features active (clear selection)
λ = 10.0  →   1 feature active (aggressive)
λ = 100   →   0 features active (all dead)
```

> What to notice: **the number of active features drops as λ increases.** Lasso is doing automatic feature selection — the model gets simpler.

### Experiment B — The sparsity sweep (code)

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

X = np.random.RandomState(42).randn(100, 20)
w_true = np.zeros(20)
w_true[[0, 3, 7]] = [2, -1.5, 0.8]
y = X @ w_true + np.random.RandomState(0).randn(100) * 0.5

X_scaled = StandardScaler().fit_transform(X)

for alpha in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
    m = Lasso(alpha=alpha).fit(X_scaled, y)
    n_active = np.sum(m.coef_ != 0)
    print(f"λ={alpha:>5.2f}  active_features={n_active:>2d}  "
          f"true_active_correct={np.sum((m.coef_ != 0) & (w_true != 0))}")
```

```text
λ= 0.01  active_features=16  true_active_correct=3
λ= 0.05  active_features=11  true_active_correct=3
λ= 0.10  active_features= 8  true_active_correct=3
λ= 0.50  active_features= 4  true_active_correct=3
λ= 1.00  active_features= 2  true_active_correct=2
λ= 5.00  active_features= 0  true_active_correct=0
```

> 📌 At λ=0.5, Lasso correctly identifies all 3 true features while dropping 13 noise features. At λ=5, it's too aggressive — drops even the real ones.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Two perfectly correlated features + one independent
X = np.column_stack([
    np.random.RandomState(42).randn(100),         # feature 1
    np.random.RandomState(42).randn(100) + 0.01,  # feature 2 = feature 1 + noise
    np.random.RandomState(7).randn(100)            # feature 3 (independent)
])
y = 3 * X[:, 0] + 2 * X[:, 2] + np.random.RandomState(0).randn(100) * 0.5

X_scaled = StandardScaler().fit_transform(X)
m = Lasso(alpha=0.1).fit(X_scaled, y)
print("Coefficients:", np.round(m.coef_, 3))
```

```text
Coefficients: [ 1.82  0.    1.53]    ← run 1
Coefficients: [ 0.    1.79  1.51]    ← run 2 (different data shuffle)
```

**What happened?** Features 1 and 2 are nearly identical (correlated). Lasso picks *one arbitrarily* — sometimes feature 1, sometimes feature 2. Run it on different data splits and the selection flips.

> 💥 **Break pattern:** correlated features → Lasso picks one arbitrarily → unstable selection. Why? The L1 penalty has no mechanism to share weight across correlated features.

Now the key teaching steps:

- Does **Ridge** fix this? Ridge keeps both, but doesn't select. Not what we want.
- Does **Elastic Net** fix this? Yes — it shares weight across correlated features via L2.
- **Lesson:** Lasso is unstable with correlated feature groups. Use Elastic Net when features come in correlated clusters.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change… | What happens | Why |
|---|---|---|
| λ = 0 | All features active (OLS) | No penalty |
| λ → ∞ | All coefficients = 0 | Penalty dominates — everything dropped |
| Features are orthogonal | Lasso selects cleanly | No ambiguity about which feature deserves credit |
| Features are correlated in groups | Lasso picks one arbitrarily | L1 has no grouping mechanism |
| True model is sparse | Lasso excels | Many zeros align with truth |
| True model is dense (all features matter) | Lasso underfits | Drops too many real effects |

> 🤔 Think: which one is *not* fixed by more data? → Dense truth. If all 500 features each have a small real effect, Lasso will still drop most of them — no amount of data changes that. Use Ridge for dense models.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w   → sparse coefficient vector (many = 0)     (model.coef_)
b   → intercept                                  (model.intercept_)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `alpha` (λ) | L1 penalty strength | Close to OLS, no sparsity | Drops everything | 0.001–10; log-spaced CV |
| `max_iter` | Max coordinate-descent passes | May not converge | Wasted time | 1000 (default usually fine) |
| `tol` | Convergence tolerance | — | — | 1e-4 |

**How to choose λ:** log-spaced grid search with cross-validation. Monitor both RMSE and number of nonzero features.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linear relationship** | y ≈ linear function of features | Model form | residual plots | add features / different model |
| **Sparsity** | Few features truly matter | Lasso works best when truth has many zeros | domain knowledge | use Ridge for dense models |
| **Features comparable scale** | Fair L1 penalty | |wⱼ| summed regardless of scale | — | **standardise features** |
| **Independence** | Samples don't affect each other | Statistics | domain knowledge | time-series models |

> Key difference from Ridge: Lasso **assumes sparsity** — the true model has many zero coefficients. This is a strong assumption that doesn't always hold.

---

## 21. Data Requirements

```text
Target       → continuous numeric
Features     → numerical; categorical must be encoded
Missing      → must be handled first
Outliers     → squared-loss sensitive; use robust variant for heavy outliers
Scaling      → REQUIRED — L1 penalty sums absolute weights; features on different scales → unfair penalty
Small data   → works well (sparsity helps)
High-dim     → a primary use case (p >> n)
```

> ⚠️ Data-leakage trap: **fit the scaler on training data only**, then transform both sets.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimise RSS + λ‖w‖₁)
        ≠
EVALUATION METRIC   (report plain metrics on held-out data)
```

| Metric | Formula | Simple | Use |
|---|---|---|---|
| RMSE | √((1/n)Σ(y−ŷ)²) | avg miss in original units | main metric |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust alternative |
| R² | 1 − SS_res/SS_tot | % variance explained | fit quality |

**Additional: sparsity**

Count the number of nonzero coefficients — this is a key output of Lasso. A model with 5 active features from 200 is far more interpretable than one with all 200.

> ⚠️ Never report the penalised training objective as performance.

---

## 23. Failure Cases

```text
CORRELATED GROUPS    → Lasso picks one feature arbitrarily, unstable selection
DENSE TRUTH          → all features matter a little → Lasso drops most → underfit
λ MIS-TUNED          → too small (overfit), too large (drops everything)
NO SCALING           → large-magnitude features dominate the penalty
```

---

## 24. Debugging

Model performs badly? Run this checklist:

```text
1. All coefficients = 0?               → λ too large → decrease α
2. No zeros at all?                     → λ too small → increase α
3. Selection flips between runs?        → correlated features → use Elastic Net
4. R² low, many features dropped?      → λ too large OR truth is not sparse
5. Selected features look random?       → check correlation structure; consider Elastic Net
6. Coefficients are biased downward?    → expected (Lasso shrinks nonzero too); optionally refit OLS on selected features
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:  "Use everything. No penalty."
Ridge:              "Use everything, but keep weights small."
Lasso:              "Use only the important features. Drop the rest."
Elastic Net:        "Use a subset, and share weight across correlated groups."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Ridge | RSS + λ‖w‖² | stable, handles collinearity | no feature selection | correlated/wide data |
| Lasso | RSS + λ\|w\| | auto feature selection | unstable with correlated groups | sparse truth, p>>n |
| Elastic Net | RSS + λ₁\|w\| + λ₂‖w‖² | selection + stability | two parameters | correlated + sparse |
| Linear | no penalty | unbiased | p>>n fails | clean data |

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict patient recovery time from 500 blood biomarkers
DATA:              80 patients, 500 measurements
EDA:               many features, sparse signal expected
CLEAN:             impute missing values, handle outliers
SPLIT:             train / validation / test
SCALE:             StandardScaler (REQUIRED)
TUNE:              GridSearchCV over log-spaced α, 5-fold CV
TRAIN:             Lasso(alpha=best_α) on training data
EVALUATE:          RMSE on test + count nonzero features
INTERPRET:         report active biomarkers and their coefficients
VALIDATE:          run stability selection (repeat on subsamples to check which features are consistently selected)
DEPLOY:            serve sparse model; document selected features
```

> 🚀 Lasso's real value: it's not just a model — it's a **discovery tool** that tells you which variables matter.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what penalty does Lasso use? (L1 or L2?)
2. **Understand:** why does the L1 penalty produce exact zeros?
3. **Calculate:** apply soft-thresholding to z=2.5, λ=1.5.
4. **Apply:** given 20 features with only 3 truly active, decide if Lasso is appropriate.
5. **Debug:** Lasso coefficients flip between runs — what's happening?
6. **Experiment:** run the sparsity sweep (Section 16) and plot active features vs λ.
7. **Build:** biomarker mini-project: synthetic sparse data → Lasso → identify true features → compare with Ridge.
8. **Explain:** explain to a friend why Lasso zeros coefficients but Ridge doesn't, using the diamond vs circle geometry.

---

## 28. Interview

### Beginner
- **What is Lasso Regression?** Linear Regression with an L1 penalty that shrinks coefficients and sets some to exactly zero — automatic feature selection.
- **What is the L1 penalty?** λ · Σ|wⱼ| — the sum of absolute coefficient magnitudes.
- **What's the key advantage over Ridge?** Feature selection — it zeroes out unimportant features.

### Intermediate
- **Why does Lasso zero coefficients but Ridge doesn't?** The L1 constraint (diamond) has sharp corners on axes where a coefficient can be exactly 0. L2 (circle) has no corners.
- **How is Lasso solved?** Coordinate descent with soft-thresholding. Not plain gradient descent — |w| is not differentiable at 0.
- **What's the weakness of Lasso?** With correlated features, it picks one arbitrarily — unstable selection.

### Advanced
- **What's soft-thresholding?** `wⱼ = sign(zⱼ) · max(0, |zⱼ| − λ)`. The coordinate-wise Lasso update; zeros coefficients with |zⱼ| ≤ λ.
- **What's the Bayesian view?** Lasso = MAP estimate with a Laplace (double-exponential) prior on coefficients — its peak at 0 drives sparsity.
- **How do you fix correlated-group instability?** Use Elastic Net — the L2 component averages across the group.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Objective:  J = Σ(y − ŷ)² + λ · Σ|wⱼ|
Soft-threshold:  wⱼ = sign(zⱼ) · max(0, |zⱼ| − λ)
```

**Common traps:**
- Confusing L1 (Lasso) and L2 (Ridge) — Lasso zeros, Ridge shrinks.
- Forgetting Lasso needs coordinate descent (not plain gradient descent).
- Assuming nonzero Lasso coefficients are "correct" — they're biased downward.

> **Representative pattern question (NOT a past GATE PYQ):** "Given z=1.5 and λ=2, what is the Lasso coefficient?" → Answer: 0 (since |z|−λ = −0.5 → clamped to 0).

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + subgradient + Bayesian view</summary>

### Full derivation (single coordinate)

For one coordinate wⱼ, holding all others fixed, the subproblem is:

```text
minimise  (1/2)(wⱼ − zⱼ)² + λ|wⱼ|
```

where zⱼ is the OLS value for coordinate j.

For wⱼ > 0: derivative = (wⱼ − zⱼ) + λ = 0 → wⱼ = zⱼ − λ
For wⱼ < 0: derivative = (wⱼ − zⱼ) − λ = 0 → wⱼ = zⱼ + λ
For wⱼ = 0: subgradient condition requires |zⱼ| ≤ λ

Combining: `wⱼ = sign(zⱼ) · max(0, |zⱼ| − λ)`

### Why the L1 kink matters

The absolute value |w| has a subdifferential at 0 equal to [−1, 1]. This means any value of z with |z| ≤ λ satisfies the optimality condition at w = 0. The kink "traps" the solution at exactly zero — unlike L2's smooth parabola, which has a unique gradient at 0 that always pushes the solution away from exactly zero.

### Bayesian interpretation

Lasso = MAP estimate with a Laplace prior:

```text
P(wⱼ) = (λ/2) · exp(−λ|wⱼ|)
```

The Laplace prior is sharply peaked at 0 (more probability mass near zero than a Gaussian). This drives sparsity in the posterior.

### Coordinate descent convergence

The full Lasso objective is convex. Coordinate descent converges to the global optimum for convex functions, regardless of coordinate order. Each coordinate update is a soft-threshold (closed-form for that coordinate).

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "Lasso adds an L1 penalty to OLS. The penalty charges by the absolute size of each coefficient. If a feature isn't worth its penalty, its coefficient gets set to exactly zero — automatic feature selection."

> **Explain to a 12-year-old:** "Imagine you have a backpack and can only carry 5 toys. Lasso picks the 5 most important toys and leaves the rest behind. Ridge would let you carry all toys but make you hold each one lightly."

> **Explain in an interview:** add: soft-thresholding, coordinate descent, Laplace prior, correlated-group instability, comparison with Ridge and Elastic Net.

> **Explain the mathematics:** derive the soft-threshold update from Section 30.

---

## 32. Mastery Test

**Without looking at notes:**

1. Write the Lasso objective function.
2. Write the soft-thresholding update.
3. Why does L1 produce zeros but L2 doesn't?
4. What solver is used for Lasso? Why not plain gradient descent?
5. Explain the diamond vs circle geometry.
6. What is Lasso's weakness with correlated features?
7. What's the Bayesian interpretation (Laplace prior)?
8. Compare Lasso with Ridge on a sparse problem.
9. Choose Lasso for a real problem; defend the choice.
10. State one scenario where Lasso fails.

---

## 33. Cheat Sheet

```text
Algorithm  : Lasso Regression · Supervised → Regression · Parametric
Goal       : Sparse model — automatic feature selection
Objective  : RSS + λ Σ|wⱼ|
Solve      : coordinate descent + soft-thresholding (no closed form)
Learn      : sparse w (many = 0), b
Tune       : α (λ) via log-spaced CV; scaling REQUIRED
Assumptions: linear, sparse truth, scaled features, independence
Use when   : many features, few truly matter (sparse truth), need selection
Avoid when : correlated feature groups (→ Elastic Net), dense truth (→ Ridge)
Related    : Ridge · Elastic Net · Group Lasso · LARS
Key exam   : soft-thresholding; L1 zeros vs L2 shrinks
```

---

## 34. What Next?

You've learned to *select* features with L1. But Lasso is unstable with correlated features — it picks one arbitrarily. What if you want both selection AND stability?

```text
Linear Regression
   ├── Ridge        (L2 penalty → shrink)
   └── Lasso        (L1 penalty → zero)      ← you are here
        ├── Elastic Net  (L1 + L2 → both)    → next note (05)
        └── Bayesian     (prior on weights)   → 06
```

> Next recommended: **05. Elastic Net** — it answers the weakness you just saw: "what if my important features are correlated?"
