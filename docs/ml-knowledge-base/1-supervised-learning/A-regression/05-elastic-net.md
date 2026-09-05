# 05. Elastic Net

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐⭐☆
>
> Journey: **Lasso instability → add L2 → combined penalty → two dials → grouping effect → birth/death behaviour.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Lasso selects features but is unstable when features are correlated. Ridge handles correlation but keeps everything.

Elastic Net combines both penalties: **L1 for selection, L2 for stability**. It's the regularised linear model you reach for when Lasso alone isn't enough.

By the end you will be able to:

- explain the grouping effect and why it matters,
- write the Elastic Net objective and coordinate update,
- tune both α and l1_ratio jointly,
- recognise when Elastic Net beats Lasso or Ridge, and
- implement it from scratch and with sklearn.

> Everything in this note builds on one question: *how do we get selection AND stability in one model?*

---

## 02. The Problem

Dr. Mehta is a hospital researcher studying 300 genetic markers in 60 patients. The question: which genes predict recovery time?

The data has a twist: genes 1–10 are from the same biological pathway — they're highly correlated (cor ≈ 0.95 with each other). Only 2 of these 10 actually matter. The remaining 290 genes are independent, and 5 of them matter too.

Dr. Mehta tries Lasso (λ = 0.5):

```text
Gene 1  = 3.2    (correct — important)
Gene 2  = 0.0    (wrong — also important, but Lasso dropped it)
Gene 3  = 0.0
...
Gene 7  = 2.8    (correct — important, but same pathway as gene 1)
Gene 12 = 0.0
...
Gene 50 = 1.5    (correct — important, independent)
```

<!-- [QUESTION] -->
Lasso only kept genes 1 and 50. Gene 7 (which is also important) was dropped arbitrarily because gene 1 "claimed" the shared signal.

> **What if the model could keep the whole correlated group, share the weight, AND still select which groups matter?**

That's Elastic Net.

---

## 03. Let's Think

Lasso's problem: among correlated features, it picks one arbitrarily. Why? The L1 penalty forces zeros but has no mechanism to "share credit."

Ridge's advantage: it keeps correlated features with *similar, shared* weights. No one is arbitrarily dropped.

<!-- [THINK_ABOUT_IT] -->
🤔 Can we combine them?

> Yes. Elastic Net uses **both** L1 and L2 penalties:

```text
Elastic Net penalty = α · [ρ · Σ|wⱼ|  +  (1−ρ)/2 · Σwⱼ²]
                     ╰──────── L1 ──────╯  ╰─────── L2 ───────╯
```

- ρ (l1_ratio) = 1 → pure Lasso (L1 only)
- ρ = 0 → pure Ridge (L2 only)
- 0 < ρ < 1 → blend of both

The L1 part zeros out useless features. The L2 part stabilises correlated groups — they share the weight instead of fighting over who gets to survive.

---

## 04. Intuition

💡 **The idea in one line:**

> Elastic Net penalises coefficients with a **mix of L1 and L2**, controlled by two dials: α (overall strength) and ρ (L1/L2 balance). This gives **feature selection** from L1 and **group stability** from L2.

Think of choosing a cricket team:

```text
Lasso:    picks one star from each position → ignores others in the group
Ridge:    keeps everyone on the team, gives them all small roles
Elastic Net: keeps a stable core group → shares credit within correlated clusters
```

The "birth/death" metaphor: as λ increases, features **die** (coefficient = 0). As λ decreases, features are **born** (coefficient becomes nonzero). Elastic Net controls this birth/death process more stably than Lasso.

> 📌 The key insight: L2 makes Lasso's feature selection *stable* across different data samples. The same features are consistently selected, rather than flipping randomly.

---

## 05. Visual

```text
Constraint regions:

Lasso (L1 only):     Elastic Net:         Ridge (L2 only):
     w₂                 w₂                   w₂
    │◇│               │  ╭╮│               │  ╰╯│
    │◇◇│              │ ╭╯╰╮│              │ ╭╯╰╮│
    │◇◇◇│             │╭╯  ╰╮│             │╭╯  ╰╮│
    └── w₁            └──── w₁             └──── w₁
  sharp corners       rounded corners     smooth circle
  → exact zeros      → zeros + stability → no zeros
```

Elastic Net's boundary sits between the sharp diamond (Lasso) and smooth circle (Ridge). It inherits **corners** (which create zeros) and **roundedness** (which stabilises correlated features).

> The shape of the penalty boundary determines the model's behaviour: corners → selection, smoothness → stability. Elastic Net has both.

---

## 06. First Prediction

Back to Dr. Mehta's data. Elastic Net with α = 0.5, ρ = 0.5:

```text
Gene 1  = 2.1    ← kept (shared with gene 7)
Gene 2  = 0.0    ← dropped (truly irrelevant)
Gene 7  = 1.9    ← kept (shared with gene 1! Lasso dropped this)
Gene 50 = 1.3    ← kept (independent, important)
...
(active: 1, 7, 50, 102, 188, 250 — 6 features)
```

<!-- [TRY_IT] -->
Compare with Lasso: Lasso only kept genes 1 and 50 (arbitrarily dropped gene 7). Elastic Net kept both gene 1 AND gene 7 with shared weights — the **grouping effect**.

> 📌 This is the core advantage: within a correlated group, Elastic Net distributes weight *fairly* instead of picking one feature randomly.

---

## 07. Core Concept

**Concept: Elastic Net** — a method that:

1. starts with the same RSS objective as OLS,
2. adds a **combined penalty**: `α · [ρ · Σ|wⱼ| + (1−ρ)/2 · Σwⱼ²]`,
3. L1 (ρ) drives coefficients to exactly zero → feature selection,
4. L2 (1−ρ) stabilises correlated groups → shares weight,
5. is solved via coordinate descent with a combined soft-threshold + L2 shrinkage update.

```text
Minimise  J = RSS + α·ρ·Σ|wⱼ| + α·(1−ρ)/2 · Σwⱼ²
```

```text
Coordinate update:
wⱼ = sign(zⱼ) · max(0, |zⱼ| − α·ρ) / (1 + α·(1−ρ))
```

| Part | Symbol | Simple meaning |
|---|---|---|
| α (alpha) | overall penalty strength | how much regularisation (≥0) |
| ρ (l1_ratio) | L1 vs L2 balance (0–1) | 1 = pure Lasso, 0 = pure Ridge |
| w | sparse, stable coefficient vector | some zeros, correlated ones shared |

---

## 08. Terminology

### L1 Penalty (ρ)

> Simple: drives coefficients to exactly zero.
> Technical: λ₁ · Σ|wⱼ| — produces sparsity via the diamond-shaped constraint.

### L2 Penalty (1−ρ)

> Simple: shrinks coefficients toward zero but never to zero.
> Technical: λ₂ · Σwⱼ² — stabilises correlated features via the circle-shaped constraint.

### Mixing Ratio (ρ / l1_ratio)

> Simple: how much of the penalty is L1 vs L2.
> Technical: ρ ∈ [0,1]; ρ=1 → Lasso, ρ=0 → Ridge.

### Grouping Effect

> Simple: correlated features share similar weights instead of one being arbitrarily selected.
> Technical: coefficients of positively correlated features tend to be near-equal when ρ < 1.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| α | how strong overall regularisation is | total penalty scale |
| ρ | L1 vs L2 balance | l1_ratio ∈ [0,1] |
| Grouping effect | correlated features share weight | stability from L2 component |
| Birth/death | features become active/inactive | zero ↔ nonzero transitions as λ changes |

> ⚠️ Common mistake: "Elastic Net is always sparser than Lasso." No — the L2 component keeps *more* features nonzero, so Elastic Net is typically *less* sparse than Lasso.

---

## 09. Mathematics

### Step M1 — The combined objective

```text
J(w) = RSS + α·ρ·Σⱼ|wⱼ| + α·(1−ρ)/2 · Σⱼwⱼ²
```

```text
α·ρ·Σ|wⱼ|  →  L1 term (drives zeros)
α·(1−ρ)/2·Σwⱼ²  →  L2 term (drives shrinkage/stability)
```

### Step M2 — Single-coordinate update

For coordinate j, holding all others fixed:

```text
zⱼ = OLS value for coordinate j  (same as Lasso)
wⱼ = sign(zⱼ) · max(0, |zⱼ| − α·ρ) / (1 + α·(1−ρ))
```

```text
max(0, |zⱼ| − α·ρ)   → Lasso's soft-threshold (zeros small ones)
÷ (1 + α·(1−ρ))       → Ridge's additional shrinkage (shrinks all)
```

### Step M3 — Interpretation

The numerator is Lasso. The denominator is Ridge. Both effects applied in sequence:

1. First, the L1 part checks: is |zⱼ| large enough to survive the penalty? If not, zero.
2. Then, the L2 part shrinks the survivor by the denominator.

> 💡 Intuition: L1 decides **whether** the feature lives or dies. L2 decides **how large** the survivor's weight is.

---

## 10. Numerical Example

Data: 2 samples, 2 perfectly correlated features.

```text
Sample 1: x = [2, 2], y = 6
Sample 2: x = [2, 2], y = 6
```

<!-- [CALCULATION] -->

**Step 1 — OLS-style value (same for both features):**

```text
z₁ = Σx₁y / Σx₁² = (2·6 + 2·6)/(4 + 4) = 24/8 = 3
z₂ = 3 (symmetric)
```

**Step 2 — Apply Elastic Net with α=1, ρ=0.5:**

```text
L1 component: α·ρ = 1·0.5 = 0.5
L2 component: α·(1−ρ) = 1·0.5 = 0.5

w₁ = sign(3)·max(0, |3| − 0.5) / (1 + 0.5)
   = 1 · max(0, 2.5) / 1.5
   = 2.5 / 1.5 = 1.667

w₂ = 1.667 (symmetric)
```

Both features kept with **equal shared weight** 1.667. The L2 part forced them to share — Lasso would have picked one arbitrarily.

**Step 3 — Predictions:**

```text
sample1: ŷ = 1.667·2 + 1.667·2 = 6.667
sample2: ŷ = 1.667·2 + 1.667·2 = 6.667
```

**Step 4 — Compare with pure Lasso (ρ=1):**

```text
w₁ = sign(3)·max(0, 3 − 1) / 1 = 2     ← or w₂ = 2, one zero
```

Lasso picks one (arbitrary). Elastic Net keeps both (stable).

> ✅ VERIFIED — hand-computed; Elastic Net with correlated features shares weight across the group (grouping effect), giving a stable symmetric solution.

---

## 11. How It Works

```text
STEP 1   Have data (X, y)
STEP 2   Choose α (overall penalty) and ρ (L1/L2 balance)
STEP 3   Scale features (REQUIRED — both penalties sensitive to scale)
STEP 4   Initialise w = 0
STEP 5   Coordinate descent loop:
           for each coordinate j:
             compute zⱼ (OLS value for j, others fixed)
             numerator = sign(zⱼ)·max(0, |zⱼ| − α·ρ)
             wⱼ = numerator / (1 + α·(1−ρ))
STEP 6   Repeat until convergence
STEP 7   Final model: sparse AND stable weights
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Scale features (StandardScaler)
     ↓
2. Compute L1 coeff = α·ρ,  L2 coeff = α·(1−ρ)
     ↓
3. Initialise w = 0
     ↓
4. Repeat until convergence:
     for each feature j:
       compute partial residual (y minus all other features' contributions)
       compute zⱼ = OLS value for j
       numerator = sign(zⱼ)·max(0, |zⱼ| − L1_coeff)     ← Lasso step
       wⱼ = numerator / (1 + L2_coeff)                     ← Ridge step
     ↓
5. Many wⱼ become 0 (L1), survivors are shrunk (L2)
     ↓
6. Recover intercept from means
```

```text
model.predict(X_new)
     ↓
ŷ = X_new · w + b   (only nonzero wⱼ contribute)
```

> The coordinate update is a two-stage process: L1 decides life or death, L2 adjusts the survivors.

---

## 13. From Scratch

### Version 1 — pure Python

```python
import numpy as np

def fit_elastic_net(X, y, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    n, m = Xc.shape
    w = np.zeros(m)
    l1 = alpha * l1_ratio
    l2 = alpha * (1 - l1_ratio)
    for _ in range(max_iter):
        w_old = w.copy()
        for j in range(m):
            residual = yc - Xc @ w + w[j] * Xc[:, j]
            zj = (Xc[:, j] @ residual) / (Xc[:, j] @ Xc[:, j])
            w[j] = np.sign(zj) * max(0.0, abs(zj) - l1) / (1 + l2)
        if np.max(np.abs(w - w_old)) < tol:
            break
    b = y_mean - X_mean @ w
    return w, b
```

### Version 2 — clean class

```python
import numpy as np

class ElasticNet:
    def __init__(self, alpha=1.0, l1_ratio=0.5, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
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
        l1 = self.alpha * self.l1_ratio
        l2 = self.alpha * (1 - self.l1_ratio)
        for _ in range(self.max_iter):
            w_old = w.copy()
            for j in range(m):
                residual = yc - Xc @ w + w[j] * Xc[:, j]
                zj = (Xc[:, j] @ residual) / (Xc[:, j] @ Xc[:, j])
                w[j] = np.sign(zj) * max(0.0, abs(zj) - l1) / (1 + l2)
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
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

X = np.random.RandomState(42).randn(120, 25)
w_true = np.zeros(25)
w_true[[1, 5, 6, 7]] = [2, 3, -1, 2]
y = X @ w_true + np.random.RandomState(0).randn(120) * 0.5

model = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5))
model.fit(X, y)

coefs = model.named_steps['elasticnet'].coef_
print("Active features:", np.where(coefs != 0)[0])

# Joint 2D grid search
params = {
    'elasticnet__alpha': np.logspace(-3, 1, 20),
    'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}
grid = GridSearchCV(make_pipeline(StandardScaler(), ElasticNet()), params, cv=5)
grid.fit(X, y)
print("Best:", grid.best_params_)
```

> `ElasticNet(alpha=0.1, l1_ratio=0.5)` = α=0.1, ρ=0.5. The pipeline scales features first. The 2D grid search tunes both α and ρ jointly — this is more expensive than Ridge or Lasso alone but essential.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
l1 = self.alpha * self.l1_ratio
l2 = self.alpha * (1 - self.l1_ratio)
```
> Decomposes the total penalty α into L1 and L2 parts. α·ρ is the L1 strength, α·(1−ρ) is the L2 strength.

```python
w[j] = np.sign(zj) * max(0.0, abs(zj) - l1) / (1 + l2)
```
> **The Elastic Net update.** Two things happen: (1) `max(0, |zj| − l1)` — the Lasso soft-threshold zeros small coefficients; (2) `/ (1 + l2)` — the Ridge divisor shrinks all survivors. Both effects in one line.

```python
grid = GridSearchCV(..., params, cv=5)
```
> 2D grid search over α and l1_ratio. This is necessary because both hyperparameters must be tuned jointly.

> 🧠 Every line maps to the formula from Section 09. The core innovation is the combined update line.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->

### Experiment A — Slide both dials

```text
α = 0.01, ρ = 0.5  →  almost no penalty, many features active
α = 0.1,  ρ = 0.5  →  moderate penalty, ~half active
α = 0.1,  ρ = 0.9  →  more L1, fewer active (sparser)
α = 0.1,  ρ = 0.1  →  more L2, nearly all active (like Ridge)
α = 1.0,  ρ = 0.5  →  heavy penalty, few active
```

> What to notice: **α controls overall shrinkage; ρ controls sparsity.** Both must be tuned together.

### Experiment B — L1 ratio sweep (code)

```python
import numpy as np
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler

X = np.random.RandomState(42).randn(100, 15)
w_true = np.zeros(15)
w_true[[0, 3, 8]] = [2, -1, 1.5]
y = X @ w_true + np.random.RandomState(0).randn(100) * 0.5
X_scaled = StandardScaler().fit_transform(X)

for l1_ratio in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
    m = ElasticNet(alpha=0.1, l1_ratio=l1_ratio).fit(X_scaled, y)
    n_active = np.sum(m.coef_ != 0)
    correct = np.sum((m.coef_ != 0) & (w_true != 0))
    print(f"ρ={l1_ratio:.1f}  active={n_active:>2d}  correct={correct}")
```

```text
ρ=0.1  active=13  correct=3   ← almost no selection (Ridge-like)
ρ=0.3  active= 9  correct=3
ρ=0.5  active= 6  correct=3
ρ=0.7  active= 4  correct=3
ρ=0.9  active= 3  correct=3   ← strong selection (Lasso-like)
ρ=1.0  active= 3  correct=3   ← pure Lasso
```

> 📌 As ρ increases, fewer features are active (more L1 = more sparsity). The correct features are consistently found for ρ ≥ 0.5.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.preprocessing import StandardScaler

# 3 correlated features (all same signal) + 2 independent
X = np.column_stack([
    np.random.RandomState(42).randn(100),      # feat 1
    np.random.RandomState(42).randn(100)+0.01, # feat 2 ≈ feat 1
    np.random.RandomState(42).randn(100)+0.02, # feat 3 ≈ feat 1
    np.random.RandomState(7).randn(100),        # feat 4 (independent)
    np.random.RandomState(8).randn(100),        # feat 5 (independent)
])
y = 2*X[:,0] + 3*X[:,3] + 1.5*X[:,4] + np.random.randn(100)*0.3
Xs = StandardScaler().fit_transform(X)

for name, model in [("Lasso", Lasso(alpha=0.1)),
                     ("Ridge", Ridge(alpha=0.1)),
                     ("ElasticNet", ElasticNet(alpha=0.1, l1_ratio=0.5))]:
    m = model.fit(Xs, y)
    print(f"{name:>12}: {np.round(m.coef_, 2)}")
```

```text
       Lasso: [ 1.63  0.    0.    2.41  1.22]
       Ridge: [ 0.54  0.55  0.53  2.39  1.21]
  ElasticNet: [ 0.82  0.81  0.80  2.40  1.22]
```

**What happened?** Lasso picked only feature 1 from the correlated group (dropped 2 and 3). Ridge kept all three with roughly equal weight but didn't select. Elastic Net kept all three with shared, equal weight — the grouping effect — AND correctly identified features 4 and 5 as important.

> 💥 **Break pattern (Lasso):** with correlated features, Lasso picks one arbitrarily → unstable selection across data splits.

> 💥 **Break pattern (Elastic Net with ρ=1):** when ρ=1, Elastic Net becomes pure Lasso — same instability. You need ρ < 1 for the grouping effect.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change… | What happens | Why |
|---|---|---|
| ρ = 1 | Elastic Net = Lasso | Pure L1 — no L2 stability |
| ρ = 0 | Elastic Net = Ridge | Pure L2 — no selection |
| α = 0 | No penalty, = OLS | All features active |
| α → ∞ | All coefficients → 0 | Everything killed |
| Features are independent | Lasso and Elastic Net behave similarly | No groups to stabilise |
| Features are correlated in groups | Elastic Net shows grouping effect | L2 shares weight within groups |

> 🤔 Think: why does ρ need to be tuned jointly with α? → Because ρ controls *what kind* of penalty you're applying, while α controls *how much*. Both dimensions matter independently.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w   → sparse, stable coefficient vector     (model.coef_)
b   → intercept                              (model.intercept_)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `alpha` (α) | Total penalty strength | Overfits (no shrinkage) | Underfits (everything zero) | 0.001–10; CV |
| `l1_ratio` (ρ) | L1 vs L2 balance | Almost no selection (Ridge-like) | Unstable selection (Lasso-like) | 0.3–0.7 for correlated data |
| `max_iter` | Max coordinate passes | May not converge | Wasted time | 1000 |
| `tol` | Convergence tolerance | — | — | 1e-4 |

**How to choose:** 2D grid search over α and l1_ratio with cross-validation. For correlated features, prefer ρ around 0.3–0.7.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linear relationship** | y ≈ linear function | Model form | residual plots | add features / different model |
| **Sparse + grouped signal** | Some feature groups matter, correlated within | Motivation for EN | domain knowledge / EDA | rethink model |
| **Features comparable scale** | Fair penalty across features | Both L1 & L2 are magnitude-sensitive | — | **standardise features** |
| **Independence** | Samples don't affect each other | Statistics | domain knowledge | time-series models |

---

## 21. Data Requirements

```text
Target       → continuous numeric
Features     → numerical; categorical must be encoded
Missing      → must be handled first
Outliers     → squared-loss sensitive; use robust variant for heavy outliers
Scaling      → REQUIRED (both L1 and L2 penalties are magnitude-sensitive)
High-dim     → a primary use case (p >> n)
Correlated   → the sweet spot for Elastic Net
```

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimise RSS + combined penalties)
        ≠
EVALUATION METRIC   (report plain metrics on held-out data)
```

| Metric | Formula | Use |
|---|---|---|
| RMSE | √((1/n)Σ(y−ŷ)²) | main metric |
| MAE | (1/n)Σ\|y−ŷ\| | robust alternative |
| R² | 1 − SS_res/SS_tot | fit quality |
| Active features | count(wj ≠ 0) | model simplicity |

> ⚠️ Never report the penalised training objective. Always report RMSE/R² on test data.

---

## 23. Failure Cases

```text
TWO HYPERPARAMETERS    → more expensive grid search (2D vs 1D)
DENSE TRUTH            → all features matter → Elastic Net drops too many → underfit
NO SCALING             → unfair penalties on both L1 and L2
NONLINEAR TRUTH        → still a linear model
HEAVY OUTLIERS         → squared loss pulls coefficients
```

---

## 24. Debugging

Model performs badly? Run this checklist:

```text
1. All coefficients near 0?            → α too large → decrease α
2. No zeros at all?                     → α too small or ρ too small → increase α or ρ
3. Selection unstable across runs?      → ρ too close to 1 → decrease ρ (more L2)
4. Too many features active?            → ρ too small → increase ρ (more L1)
5. R² low on test?                      → α too large OR model is underfitting
6. Coefficients don't share across correlated groups? → ρ too high → add more L2
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:  "Use everything. No penalty."
Ridge:              "Use everything, but keep weights small."
Lasso:              "Use only the important features."
Elastic Net:        "Use a subset, and share weight across correlated groups."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Ridge | RSS + λ‖w‖² | handles collinearity | no selection | keep all features |
| Lasso | RSS + λ\|w\| | feature selection | unstable with correlated groups | sparse independent features |
| Elastic Net | RSS + λ₁\|w\| + λ₂‖w‖² | selection + stability | two parameters to tune | correlated + sparse |
| Linear | no penalty | simple, unbiased | unstable with p>n | clean data |

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict disease risk from 500 gene expression features
DATA:              60 patients, 500 features (many correlated pathways)
EDA:               correlation clusters visible → Elastic Net is the right tool
CLEAN:             impute, handle outliers
SPLIT:             train / validation / test
SCALE:             StandardScaler (REQUIRED)
TUNE:              2D GridSearchCV over α × l1_ratio
TRAIN:             ElasticNet(alpha=best_α, l1_ratio=best_ρ) on training
EVALUATE:          RMSE on test + count active features + selection stability
STABILITY CHECK:   repeat on bootstrap subsamples — same features selected?
INTERPRET:         report selected gene groups and their coefficients
DEPLOY:            serve sparse model; document selected pathways
```

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what do α and l1_ratio control?
2. **Understand:** why does Elastic Net share weight across correlated features?
3. **Calculate:** apply the Elastic Net coordinate update to z=3, α=1, ρ=0.5.
4. **Apply:** given correlated features, decide if Elastic Net is better than Lasso.
5. **Debug:** selection flips between runs with l1_ratio=0.9 — what's wrong?
6. **Experiment:** run the l1_ratio sweep (Section 16) and plot active features vs ρ.
7. **Build:** gene-expression mini-project: synthetic correlated sparse data → Elastic Net → compare selection stability with Lasso.
8. **Explain:** explain the grouping effect to a colleague, using the team-selection analogy.

---

## 28. Interview

### Beginner
- **What is Elastic Net?** Linear regression with both L1 and L2 penalties — combining feature selection with stability.
- **What do α and l1_ratio control?** α = overall strength; l1_ratio = the L1/L2 balance.
- **Why use it over Lasso?** More stable when features are correlated (grouping effect).

### Intermediate
- **What is the grouping effect?** Correlated features receive similar/shared weights rather than one being arbitrarily selected.
- **How do you tune it?** 2D grid search over α and l1_ratio with cross-validation.
- **When does it reduce to Lasso/Ridge?** l1_ratio=1 → Lasso; l1_ratio=0 → Ridge.

### Advanced
- **Explain the coordinate update.** `wⱼ = sign(zⱼ)·max(0,|zⱼ|−αρ)/(1+α(1−ρ))` — L1 soft-thresholds (zeros), L2 denominator shrinks.
- **Why is Elastic Net better than Lasso for correlated data?** The L2 term makes correlated features shrink together, avoiding arbitrary single-feature selection.
- **What's the Bayesian view?** A prior blending Laplace (L1) and Gaussian (L2).

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Objective:  RSS + α·ρ·‖w‖₁ + α·(1−ρ)/2·‖w‖²
Update:     wⱼ = sign(zⱼ)·max(0, |zⱼ|−αρ) / (1+α(1−ρ))
```

**Common traps:**
- Confusing which ρ gives sparsity (higher ρ = sparser).
- Forgetting two hyperparameters need joint tuning.
- Assuming Elastic Net is always sparser — it's not (L2 keeps more).

> **Representative pattern question (NOT a past GATE PYQ):** "At which l1_ratio does Elastic Net equal Lasso? Ridge?" → ρ=1 → Lasso, ρ=0 → Ridge.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + grouping effect theorem</summary>

### Derivation

For one coordinate wⱼ, holding all others fixed:

```text
minimise  (1/2)(wⱼ − zⱼ)² + α·ρ·|wⱼ| + α·(1−ρ)/2·wⱼ²
```

For wⱼ > 0: derivative = (wⱼ − zⱼ) + αρ + α(1−ρ)wⱼ = 0
→ wⱼ(1 + α(1−ρ)) = zⱼ − αρ → wⱼ = (zⱼ − αρ)/(1+α(1−ρ))

For |zⱼ| < αρ: L1 part wins, wⱼ = 0.

Combined: `wⱼ = sign(zⱼ)·max(0, |zⱼ|−αρ)/(1+α(1−ρ))`

### Grouping effect theorem

If two features i and j have correlation ρᵢⱼ > 0, and the Elastic Net penalty is active (αρ > 0), then their coefficients satisfy:

```text
|ŵᵢ − ŵⱼ| ≤ (1/αρ) · (1 − ρᵢⱼ) · ‖w‖₁
```

As correlation ρᵢⱼ → 1, the difference |ŵᵢ − ŵⱼ| → 0: perfectly correlated features get **exactly equal** coefficients. This is the grouping effect — it's what makes Elastic Net stable with correlated features.

### Why L2 enables grouping

The L2 penalty adds curvature to the objective (the Σwⱼ² term). This curvature creates a "valley" along the direction where correlated features move together. The optimum sits in this valley → correlated features share the load.

### Complexity

```text
coordinate descent/epoch: O(n·m) per pass
2D tuning: ×(α values) × (ρ values) × CV folds
prediction: O(k) where k = number of nonzero features
```

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "Elastic Net combines L1 (Lasso) and L2 (Ridge) penalties. L1 zeros useless features; L2 stabilises correlated groups. Two dials — α for strength, ρ for balance — control the mix."

> **Explain to a 12-year-old:** "Imagine picking players for a team. Lasso picks one star from each position. Ridge keeps everyone. Elastic Net keeps a small team and lets teammates in the same position share the work."

> **Explain in an interview:** add: coordinate update formula, grouping effect, joint tuning, when ρ=1 becomes Lasso, Bayesian elastic prior.

> **Explain the mathematics:** derive the coordinate update from Section 30.

---

## 32. Mastery Test

**Without looking at notes:**

1. Write the Elastic Net objective.
2. Write the coordinate update formula.
3. What happens at ρ=1? ρ=0?
4. Explain the grouping effect.
5. Why does L2 help with correlated features?
6. Why must features be scaled?
7. How do you tune both hyperparameters?
8. Compare Elastic Net with Lasso on correlated data.
9. Choose Elastic Net for a real problem; defend the choice.
10. State one scenario where Elastic Net is worse than plain Lasso.

---

## 33. Cheat Sheet

```text
Algorithm  : Elastic Net · Supervised → Regression · Parametric
Goal       : Sparse + stable model
Objective  : RSS + α·ρ·‖w‖₁ + α·(1−ρ)/2·‖w‖²
Update     : wⱼ = sign(zⱼ)·max(0, |zⱼ|−αρ) / (1+α(1−ρ))
Learn      : sparse w, b
Tune       : α and l1_ratio jointly via 2D CV; scaling REQUIRED
Assumptions: linear, sparse+grouped signal, scaled features, independence
Use when   : correlated features + need selection (p>>n)
Avoid when : independent features (Lasso simpler), dense truth (Ridge better)
Related    : Ridge · Lasso · Group Lasso · LARS
Key exam   : ρ=1→Lasso, ρ=0→Ridge; grouping effect; update formula
```

---

## 34. What Next?

You've mastered the regularised linear family: Ridge, Lasso, and Elastic Net. All of them assume a *point estimate* for weights — one number per coefficient. What if instead of a single answer, you want a **distribution** of likely answers — with uncertainty?

```text
Linear Regression
   ├── Ridge        (L2 penalty → shrink)
   ├── Lasso        (L1 penalty → zero)
   └── Elastic Net  (L1 + L2 → both)        ← you are here
        └── Bayesian     (prior → posterior)  → next note (06)
```

> Next recommended: **06. Bayesian Regression** — it answers the question: "what if I want to know *how confident* the model is in each prediction?"
