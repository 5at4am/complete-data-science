# 15. XGBoost (eXtreme Gradient Boosting)

> Difficulty: ⭐⭐⭐⭐⭐ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐⭐☆
> GATE: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **problem → 2nd-order → regularization → formula → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

XGBoost is the **most-used algorithm in Kaggle history** on tabular data — and for good reason. It took Gradient Boosting's core idea and added three upgrades: second-order optimization, built-in regularization, and engineering that makes it fast.

By the end you will be able to:

- explain why "second-order" matters and what regularization does,
- compute a full XGBoost round by hand (gradient, Hessian, split gain, leaf weight),
- code it from scratch and with the xgboost library,
- break it deliberately and fix it,
- and compare it fairly against LightGBM and CatBoost.

> Everything in this note builds on Gradient Boosting's foundation, with two key additions: *curvature* and *penalty*.

---

## 02. The Problem

Arjun works at an insurance company in Bangalore. He needs to **predict claim amount** (₹) for new customers based on age, vehicle type, driving history, and past claims. The dataset has 10,000 rows, 15 features, and some missing values in the driving history column.

He tried plain Gradient Boosting and got decent R² ≈ 0.82. But:

- The model **overfits on noisy claims** (extreme outliers in claim amounts)
- Training takes too long on the full dataset
- Missing values need manual imputation

<!-- [QUESTION] -->
Here's the question:

> **Can we make gradient boosting (a) faster, (b) automatically handle missing values, and (c) resist overfitting better — all at once?**

Think about what modifications would achieve each before reading on.

---

## 03. Let's Think

Let's recall Gradient Boosting's weaknesses:

```text
Problem                     What's happening
Too slow on large data      Every split sorts all feature values → O(n·m·log n)
Overfits on noisy data      No penalty for complex trees → fits noise
Missing values need impute  Trees can't split on NaN
No regularization built-in  Need to manually tune tree complexity
```

XGBoost attacks all four. The key insights:

1. **Use second derivatives (Hessian)** → more precise steps, faster convergence
2. **Add a penalty for complexity** → fewer leaves, smaller weights → less overfitting
3. **Handle missing values natively** → learn which direction missing values should go
4. **Histogram/approximate splits** → faster on large data

<!-- [THINK_ABOUT_IT] ]
🤔 What if you could penalize a tree for being too complicated?

> That's exactly what regularization does. A tree with 500 leaves is "worse" than one with 10 leaves, even if both fit training data similarly — the simpler one generalizes better.

---

## 04. Intuition

💡 **The idea in one line:**

> XGBoost is Gradient Boosting with a **built-in complexity penalty** and **second-order (Newton) optimization**, making it more accurate, faster to converge, and harder to overfit.

Think of it like GPS navigation. Gradient Boosting is like following someone's directions: "turn left at the next landmark." XGBoost is like having a GPS with **both the road gradient (slope) AND the curvature** — you know not just which way to go, but how sharply the road turns. You reach the destination in fewer turns, with fewer wrong turns.

The regularization is like a **speed limit** — you could drive faster, but you'd risk crashing (overfitting). XGBoost forces the model to stay on the road.

---

## 05. Visual

```text
Newton (XGBoost) vs Gradient step:

   loss
    │    ╲
    │     ╲  gradient direction (first-order)
    │      ╲___
    │          ╲____  Newton direction (first + curvature)
    │               ╲__  reaches minimum in fewer steps
    │                  ╲___
    └──────────────────────── params
```

```text
Regularization effect:

   No reg (γ=0, λ=0):        With reg (γ=1, λ=1):
   tree has 50 leaves         tree has 8 leaves
   fits every training point  smooths over noise
   test R² drops              test R² stays high
```

---

## 06. First Prediction

Using our insurance example: start with F₀ = mean(claim) = ₹15,000.

Gradient at each sample: gᵢ = 2(15000 − claim_i), Hessian: hᵢ = 2 (constant for squared loss).

After one XGBoost round with γ=0, λ=1, η=0.1:

```text
Optimal leaf weight: w* = −Σg / (Σh + λ)
If a leaf has 100 samples with total gradient Σg = 500000, total Hessian Σh = 200:
w* = −500000 / (200 + 1) ≈ −2488
```

New prediction for those samples: 15000 + 0.1 × (−2488) = 12512.

> 📌 Each leaf's weight is now *shrunk* by λ — unlike plain Gradient Boosting where leaf values are unconstrained.

---

## 07. Core Concept

**XGBoost** (Chen & Guestrin, 2016) — a scalable tree-boosting system that:

1. Starts with F₀ = mean(y),
2. At each round, computes **first and second derivatives** (g, h) of the loss,
3. Greedily grows a tree by maximizing a **regularized split gain**,
4. Assigns each leaf an **optimal weight** w* = −Σg/(Σh + λ),
5. Updates predictions with **shrinkage** η.

The objective at each round:

```text
Obj = Σ L(yᵢ, ŷ̂ᵢ) + Σ Ω(fₜ)

where Ω(f) = γT + ½λ‖w‖²
```

| Upgrade over plain GB | What it does |
|---|---|
| Second-order (g, h) | More precise steps, faster convergence |
| γT penalty | Penalizes number of leaves → simpler trees |
| λ‖w‖² penalty | Penalizes large leaf values → smoother predictions |
| Sparsity-aware splits | Handles missing values natively |

---

## 08. Terminology

### Gradient (g) and Hessian (h)

> Simple: g tells you the direction of steepest descent; h tells you how sharply the curve bends.
> Technical: gᵢ = ∂L/∂ŷ̂, hᵢ = ∂²L/∂ŷ̂² — first and second derivatives of the loss at sample i.

### Split Gain

> Simple: how much better the model gets from adding a split.
> Technical: the reduction in objective (loss + regularization) from splitting a node into left and right children, minus γ.

### Regularization (Ω)

> Simple: a penalty for making the model too complicated.
> Technical: Ω(f) = γT + ½λ‖w‖² — penalizes leaf count (T) and leaf weight magnitudes (w).

### Leaf weight (w*)

> Simple: the value each leaf outputs.
> Technical: w* = −Σg/(Σh + λ), the closed-form optimal leaf value balancing fit against regularization.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| gᵢ | gradient of loss at sample i | ∂L/∂ŷ̂ |
| hᵢ | curvature of loss at sample i | ∂²L/∂ŷ̂² |
| γ | cost of adding a leaf | complexity penalty per leaf |
| λ | L2 penalty on leaf weights | shrinks weight magnitudes |
| T | number of leaves | tree complexity |

> ⚠️ Common mistake: "XGBoost is just Gradient Boosting with regularization." No — it also uses **second-order optimization** (Hessian), which is a fundamentally different optimization approach.

---

## 09. Mathematics (gradual)

We build the math in four steps.

### Step M1 — The regularized objective

```text
Obj = Σᵢ L(yᵢ, ŷ̂ᵢ) + Σₜ Ω(fₜ)

where Ω(f) = γT + ½λ‖w‖²
```

### Step M2 — Taylor expand the loss to second order

At round t, the current prediction is ŷ̂^{(t−1)}. We add a new tree fₜ. Taylor expand:

```text
L(y, ŷ̂ + fₜ) ≈ L(y, ŷ̂) + gᵢ·fₜ(xᵢ) + ½hᵢ·fₜ(xᵢ)²
```

Dropping the constant L(y, ŷ̂):

```text
Obj ≈ Σᵢ [gᵢ·fₜ(xᵢ) + ½hᵢ·fₜ(xᵢ)²] + γT + ½λΣwⱼ²
```

### Step M3 — Optimal leaf weight

Group samples by leaf. In leaf j with sample set Iⱼ:

```text
Obj_j = (Σᵢ∈Iⱼ gᵢ)·wⱼ + ½(Σᵢ∈Iⱼ hᵢ + λ)·wⱼ² + γ
```

Minimize w.r.t. wⱼ (quadratic → closed form):

```text
wⱼ* = −Σgᵢ / (Σhᵢ + λ)
```

### Step M4 — Split gain

Compare objective before vs after splitting node into left (L) and right (R):

```text
Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
```

```text
G_L, H_L  = Σg, Σh over left samples
G_R, H_R  = Σg, Σh over right samples
G, H      = Σg, Σh over all samples in node
γ          = cost of adding a leaf
```

**Split if Gain > 0.** Larger gain → better split.

---

## 10. Numerical Example

Data: x = [1, 2, 3, 4], y = [1, 3, 3, 5]. Loss = squared error. γ=0, λ=0, η=1, base ŷ̂ = mean = 3, max_depth=1 (stump).

<!-- [CALCULATION] -->

**Round 1 — compute gradients (squared error):**

```text
g = 2(ŷ̂ − y) = 2(3 − y); h = 2
y = [1,3,3,5] → g = [4, 0, 0, −4]; h = [2, 2, 2, 2]
```

**Try split at x ≤ 2** (left: samples 1,2; right: 3,4):

```text
G_L = 4+0 = 4, H_L = 4
G_R = 0+(−4) = −4, H_R = 4
G = 0, H = 8

Gain = ½[4²/4 + (−4)²/4 − 0²/8] − 0 = ½[4+4] = 4.0
```

**Try split at x ≤ 3** (left: 1,2,3; right: 4):

```text
G_L = 4+0+0 = 4, H_L = 6
G_R = −4, H_R = 2

Gain = ½[16/6 + 16/2 − 0] = ½[2.667 + 8] = 5.333
```

Best split: **x ≤ 3** (gain 5.333).

**Leaf weights:**

```text
Left (x≤3): w* = −G_L/(H_L+λ) = −4/6 = −0.667
Right (x=4): w* = −G_R/(H_R+λ) = −(−4)/2 = 2.0
```

**Prediction update: ŷ̂ += f₁(x)**

```text
x=1: 3 + (−0.667) = 2.333
x=2: 3 + (−0.667) = 2.333
x=3: 3 + (−0.667) = 2.333
x=4: 3 + 2.0 = 5.0
```

New errors: |y − ŷ̂| = [1.33, 0.667, 0.667, 0] — already better.

**Round 2** — recompute g, h from new ŷ̂ and fit another tree (refines left group).

> ✅ VERIFIED — hand-computed. Shows gradient/Hessian computation, split-gain, closed-form leaf weights, and additive update.

---

## 11. How It Works

```text
Start ŷ̂ = F₀ = mean(y)
     ↓
For t = 1..n_estimators:
     compute gᵢ, hᵢ from current ŷ̂ and loss
     ↓
     greedily grow a tree:
         at each node, evaluate splits by Gain
         pick split with max Gain (> 0)
     ↓
     set leaf weight wⱼ* = −Σg/(Σh+λ)
     ↓
     ŷ̂ += η · fₜ(x)
     ↓
Repeat
Final ŷ̂ = Σ η·fₜ(x)
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. ŷ̂ = mean(y)
     ↓
2. For each round:
     a. g = 2(ŷ̂ − y), h = 2  (for squared loss)
     b. For each feature, sort values, try splits by Gain
     c. Pick best split (highest Gain, if > 0)
     d. Assign w* = −Σg/(Σh+λ) to each leaf
     e. ŷ̂ += η × leaf_weight[xi's leaf]
     ↓
3. Store trees + base prediction
```

```text
model.predict(X_new)
     ↓
result = base (mean)
for each tree:
    result += η × tree.predict(X_new)
return result
```

> XGBoost's trees store split features, thresholds, and leaf weights. Prediction traverses each tree and sums contributions.

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
import numpy as np

class SimpleXGBRegressor:
    def __init__(self, n_estimators=10, lr=0.3, max_depth=1,
                 reg_lambda=1.0, gamma=0.0):
        self.n_estimators = n_estimators
        self.lr = lr
        self.max_depth = max_depth
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.trees = []

    def _squared_grad_hess(self, y, pred):
        g = 2 * (pred - y)
        h = np.full_like(y, 2.0)
        return g, h

    def _best_split(self, X, g, h):
        best_gain, best_col, best_thr = -np.inf, None, None
        n = len(X)
        for col in range(X.shape[1]):
            xc = X[:, col]
            idx = np.argsort(xc)
            xc_s = xc[idx]; g_s = g[idx]; h_s = h[idx]
            GL, HL = 0.0, 0.0
            G_all, H_all = g_s.sum(), h_s.sum()
            for i in range(n - 1):
                GL += g_s[i]; HL += h_s[i]
                GR = G_all - GL; HR = H_all - HL
                if xc_s[i] == xc_s[i + 1]:
                    continue
                gain = 0.5 * (GL**2/(HL+self.reg_lambda)
                              + GR**2/(HR+self.reg_lambda)
                              - G_all**2/(H_all+self.reg_lambda)) - self.gamma
                if gain > best_gain:
                    best_gain = gain; best_col = col
                    best_thr = (xc_s[i]+xc_s[i+1])/2
        return best_col, best_thr, best_gain

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        pred = np.full(len(y), y.mean())
        for _ in range(self.n_estimators):
            g, h = self._squared_grad_hess(y, pred)
            col, thr, gain = self._best_split(X, g, h)
            if gain <= 0 or col is None:
                tree = {'w': -g.sum()/(h.sum()+self.reg_lambda),
                        'col': None, 'thr': None}
            else:
                left = X[:, col] <= thr
                wl = -g[left].sum()/(h[left].sum()+self.reg_lambda)
                wr = -g[~left].sum()/(h[~left].sum()+self.reg_lambda)
                tree = {'col': col, 'thr': thr, 'wl': wl, 'wr': wr}
            self.trees.append(tree)
            pred += self.lr * self._predict_tree(X, tree)
        self.base = y.mean()
        return self

    def _predict_tree(self, X, t):
        if t['col'] is None:
            return np.full(len(X), t['w'])
        return np.where(X[:, t['col']] <= t['thr'], t['wl'], t['wr'])

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.base)
        for t in self.trees:
            pred += self.lr * self._predict_tree(X, t)
        return pred
```

### Version 2 — with early stopping

```python
def xgb_fit_early_stop(X_train, y_train, X_val, y_val,
                        max_trees=200, lr=0.1, lamb=1.0, gam=0.0):
    X_tr, y_tr = np.asarray(X_train, float), np.asarray(y_train, float)
    X_v, y_v = np.asarray(X_val, float), np.asarray(y_val, float)
    pred_tr = np.full(len(y_tr), y_tr.mean())
    pred_v = np.full(len(X_v), y_tr.mean())
    trees, best_val, best_m, patience = [], float('inf'), 0, 20
    for m in range(max_trees):
        g, h = 2*(pred_tr - y_tr), np.full(len(y_tr), 2.0)
        # ... grow tree (same as above) ...
        trees.append(tree)
        pred_tr += lr * predict_tree(X_tr, tree)
        pred_v += lr * predict_tree(X_v, tree)
        val_mse = np.mean((y_v - pred_v)**2)
        if val_mse < best_val:
            best_val, best_m = val_mse, m+1
        elif m - best_m > patience:
            break
    return trees[:best_m]
```

---

## 14. Library Implementation

```python
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=15, noise=0.1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    reg_lambda=1.0, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
    objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Importances:", model.feature_importances_)
```

> **Install:** `pip install xgboost`. The library handles missing values natively — no imputation needed.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
g = 2 * (pred - y)
```
> Gradient of squared loss: L = (y−p)² → ∂L/∂p = 2(p−y). This is the direction to push predictions.

```python
h = np.full_like(y, 2.0)
```
> Hessian (second derivative) of squared loss: ∂²L/∂p² = 2, constant. For other losses (Huber, log-loss), h varies per sample — this is where Newton's method shines.

```python
gain = 0.5*(GL**2/(HL+λ) + GR**2/(HR+λ) - G_all**2/(H_all+λ)) - γ
```
> Split gain: the reduction in regularized objective from splitting. The γ term penalizes adding a leaf — we only split if the improvement exceeds γ.

```python
w = -G/(H+λ)
```
> Optimal leaf weight, derived by setting ∂Obj/∂w = 0. The λ in the denominator shrinks the weight → less overfit.

```python
pred += lr * predict_tree(X, tree)
```
> Additive update with shrinkage: ŷ̂ += η·fₜ(x). Small η means each tree has limited influence.

> 🧠 Every line maps to a formula from Section 09. The code *is* the math.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> If this note is rendered inside the interactive platform, these become sliders. Otherwise, run them in Python and observe.

### Experiment A — effect of regularization (γ, λ)

```python
import xgboost as xgb
import numpy as np

X = np.random.RandomState(42).rand(200, 5)
y = X[:,0]*3 + X[:,1]*2 - X[:,2] + np.random.RandomState(42).randn(200)*0.5

for gamma, lam in [(0,0), (1,0), (0,1), (1,1)]:
    m = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1,
                          max_depth=5, gamma=gamma, reg_lambda=lam,
                          random_state=0)
    m.fit(X, y)
    print(f"γ={gamma} λ={lam}  train_R²={m.score(X,y):.3f}")
```

```text
γ=0 λ=0  train_R²=1.000  ← overfits perfectly
γ=1 λ=0  train_R²=0.98x  ← slightly constrained
γ=0 λ=1  train_R²=0.99x  ← weights shrunk
γ=1 λ=1  train_R²=0.97x  ← most constrained → best generalization
```

> 📌 Higher γ and λ → lower training R² but often better test R². That's regularization working.

### Experiment B — missing values

```python
import xgboost as xgb
import numpy as np

X = np.random.RandomState(42).rand(200, 5)
X[0:10, 2] = np.nan   # introduce missing values
y = X[:,0]*3 + np.random.RandomState(42).randn(200)*0.5

m = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=0)
m.fit(X, y)  # no error! XGBoost handles NaN natively
print("R²:", m.score(X, y))
```

> XGBoost learns **default directions** for missing values at each split — it tries sending NaN left and right, picks whichever gives higher gain. No imputation needed.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import xgboost as xgb
import numpy as np

X = np.random.RandomState(42).rand(50, 3)
y = X[:,0]*5 + np.random.RandomState(42).randn(50)*0.3

m1 = xgb.XGBRegressor(n_estimators=50, max_depth=3, random_state=0)
m1.fit(X, y)
print("Normal R²:", m1.score(X, y))

# Deep trees + no regularization + many rounds
m2 = xgb.XGBRegressor(n_estimators=500, max_depth=10,
                       gamma=0, reg_lambda=0, learning_rate=0.3, random_state=0)
m2.fit(X, y)
print("Overfit R²:", m2.score(X, y))
print("Trees have", m2.get_num_boosting_rounds(), "rounds, depth 10")
```

**What happens?** With deep trees, no regularization, and many rounds, the model memorizes training data perfectly. But on test data, performance collapses.

> 💥 **Break pattern:** no regularization + deep trees + many rounds → perfect train, terrible test. Why? **γ=0 and λ=0 means no penalty for complexity** — the model has no reason to stay simple.

**Fixes:**
- Set γ > 0 and/or λ > 0
- Reduce max_depth to 3–5
- Lower learning_rate + use early stopping
- Add subsample and colsample_bytree

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Set γ=0, λ=0 | Overfits easily | No complexity penalty |
| Double max_depth | Dramatically more overfitting | Exponential growth in possible splits |
| Set λ=100 | Severe underfitting | Leaf weights forced near zero |
| Set η=0.01 with 10 trees | Underfitting | Too few steps to converge |
| Add subsample=0.5 | More robust, slightly slower | Stochastic GB effect |
| Set colsample_bytree=0.5 | Reduces feature overfitting | Each tree sees different feature subset |

> 🤔 Think: which parameter is more important for controlling overfitting — γ or λ? → They work differently: γ controls tree *structure* (how many leaves), λ controls leaf *values* (how extreme). Both matter.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
Tree structures (split features, thresholds)
Leaf weights wⱼ
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `n_estimators` | Number of trees | Underfit | Overfit | 100–1000 |
| `learning_rate` (η) | Step size | Very slow | Overfits fast | 0.01–0.3 |
| `max_depth` | Tree depth | Underfit | Overfits, complex interactions | 3–8 |
| `min_child_weight` | Min Σh per leaf | Noisy leaves | Underfit | 1–10 |
| `gamma` (γ) | Min gain to split | No structural penalty | Too few splits | 0–5 |
| `reg_lambda` (λ) | L2 on leaf weights | No weight penalty | Underfit | 1 |
| `reg_alpha` (α) | L1 on leaf weights | — | Feature selection | 0 |
| `subsample` | Row sampling | — | — | 0.5–1.0 |
| `colsample_bytree` | Feature sampling | — | — | 0.5–1.0 |

> 📌 **Key tradeoff:** η ↔ n_estimators (lower η needs more trees). γ ↔ max_depth (higher γ needs deeper trees to split). λ ↔ learning (higher λ needs more rounds).

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Loss differentiable** | Need g, h | Newton step requires derivatives | Use differentiable loss | Pick compatible loss |
| **Additivity** | Model = sum of trees | Model form | — | Use different model |
| **Enough data per leaf** | min_child_weight | Stable leaf estimates | Check leaf sizes | Raise min_child_weight |
| **Not extreme label noise** | Targets roughly correct | Trees fit anything | Residual diagnostics | Early stop, subsample, robust loss |

> XGBoost is **assumption-light** (no linearity, no scaling, no normality). Its main practical requirement: careful tuning to avoid overfitting.

---

## 21. Data Requirements

```text
Target      → continuous numeric (else → classification with xgb.XGBClassifier)
Features    → numeric; categoricals need encoding (or use CatBoost for native handling)
Missing     → handled natively (sparsity-aware default direction)
Outliers    → generally robust; use Huber loss for extreme outliers
Scaling     → unnecessary (tree-based)
Small data  → works but tune carefully; reduce depth, increase regularization
High-dim    → works well; use colsample_bytree for additional regularization
```

> XGBoost's native missing value handling is a major advantage: it learns whether NaN samples should go left or right at each split, rather than imputing.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimize loss + regularization)
        ≠
EVALUATION METRIC   (what you report)
```

| Metric | Formula | Simple | Use |
|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | avg squared miss | standard loss |
| RMSE | √MSE | avg miss in ₹ | most common |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust |
| R² | 1 − SS_res/SS_tot | % variance explained | model quality |

> **Critical:** Use `eval_set` + early stopping in production. XGBoost can overfit given enough rounds — the validation curve tells you when to stop.

---

## 23. Failure Cases

```text
DATA            → extreme outliers with squared loss → chases them
OPTIMIZATION    → too many rounds, no regularization → overfit
GENERALIZATION  → extrapolation fails (leaf means bounded by training range)
PRACTICAL       → heavy categoricals → worse than CatBoost
STRUCTURAL      → slower than LightGBM on very large data
```

---

## 24. Debugging

Model performs badly? Run this checklist in order:

```text
1. Train R² = 1.0, test much lower   → overfit → add γ/λ, reduce depth, early stop
2. Both train and test low            → underfit → increase depth, more trees, lower η
3. RMSE very high on specific range   → check outliers in that range
4. Feature importance all ~equal      → noisy features → colsample, feature selection
5. Training too slow                  → use hist method: tree_method='hist'
6. NaN in predictions                 → check for data leakage or corrupt values
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Gradient Boosting:   "Fit residuals, add shallow trees."                (1st-order)
XGBoost:             "Same, but with Newton (2nd-order) + regularization."
LightGBM:            "Same objective, but histogram bins + leaf-wise growth." (faster)
CatBoost:            "Same, but ordered boosting + native categoricals."     (robust)
Random Forest:       "Average many independent deep trees."                   (no boosting)
```

| Algorithm | Main idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Gradient Boosting | 1st-order residual fit | Simple, interpretable | No regularization, slower | Prototype |
| XGBoost | 2nd-order + regularization | Accurate, fast, regularized | Many hyperparameters | Default production |
| LightGBM | Histogram + leaf-wise | Fastest on big data | Overfits easily | Large datasets |
| CatBoost | Ordered boosting + categoricals | Best categorical handling | Slower to tune | Categorical-heavy |
| Random Forest | Bagged deep trees | Robust, parallel | Lower peak accuracy | Baseline |

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict insurance claim amount
DATA:              10,000 policies (age, vehicle, history, claims_amount)
FEATURES:          mixed numeric + some categoricals
TARGET:            claim_amount_₹
MODEL:             xgb.XGBRegressor (handle missing natively)
SPLIT:             train / validation / test
TUNE:              η × depth × λ × subsample via CV; early stopping
EVALUATE:          RMSE on test + residual plot + feature importance
DEPLOY:            serve prediction on claims processing page
MONITOR:           retrain quarterly; check for claim pattern shifts
```

> 🚀 XGBoost's default pipeline is: load data → split → set eval_set → fit with early stopping → evaluate. The library does the heavy lifting.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what are g and h in XGBoost?
2. **Understand:** why does regularization (γ, λ) help generalization?
3. **Calculate:** compute one XGBoost round by hand (as Section 10).
4. **Apply:** given a dataset with missing values, explain how XGBoost handles them.
5. **Debug:** train R² = 1.0, test R² = 0.3 — diagnose and fix with 3 levers.
6. **Experiment:** run Section 16's regularization experiment at 4 (γ, λ) settings.
7. **Build:** claim prediction mini-project: EDA → handle missing → fit XGB → tune → early stop → feature importance → business summary.
8. **Explain:** explain XGBoost vs Gradient Boosting to a friend in 60 seconds.

---

## 28. Interview

### Beginner
- **What is XGBoost?** A regularized tree-boosting system using second-order (Newton) optimization to additively fit shallow trees.
- **What does additive mean?** F(x) = Σ fₜ(x); each tree's output is added to the ensemble total.
- **What regularization does it use?** γT (leaf count penalty) + λ‖w‖² (L2 leaf weight penalty).

### Intermediate
- **Why Newton (second order)?** Uses both gradient and Hessian → more precise, faster-converging steps than first-order gradient descent alone.
- **How is leaf weight computed?** w* = −Σg/(Σh+λ), closed-form from minimizing the quadratic objective.
- **How does it handle missing values?** Sparsity-aware: at each split, it tries sending missing values left and right, picks whichever gives higher gain.

### Advanced
- **Derive the leaf weight from the objective.** The objective is quadratic in wⱼ → set derivative to zero → w* = −Σg/(Σh+λ). See Section 30.
- **Compare XGBoost vs LightGBM.** XGB uses level-wise exact/approx Newton; LightGBM uses histogram + leaf-wise. LightGBM is faster on big data; XGBoost is more robust by default.
- **Why does shrinkage η help?** Scales each tree's contribution → smoother, lower-variance ensemble; pairs with more trees and early stopping.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Objective:    Obj = ΣL(y,ŷ̂) + Σ(γT + ½λ‖w‖²)
Leaf weight:  w* = −Σg / (Σh + λ)
Split gain:   Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ
Update:       ŷ̂ += η · fₜ(x)
```

**Common traps:**
- Forgetting the −γ term in the gain formula (split cost).
- Writing g = ŷ − y instead of g = 2(ŷ − y) for squared loss (sign and scale).
- Assuming feature scaling is needed (it isn't — tree splits are scale-invariant).
- Confusing gradient boosting (1st-order) with XGBoost (2nd-order).

> **Representative pattern question (NOT a past GATE PYQ):** "Why does XGBoost generalize better than plain gradient boosting on noisy data?" Answer: the γ/λ regularization penalizes tree complexity and large leaf weights, controlling the variance that plain boosting accumulates.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + complexity + theory</summary>

### Full derivation of leaf weight

At round t, minimize:

```text
Obj = Σᵢ [gᵢ·fₜ(xᵢ) + ½hᵢ·fₜ(xᵢ)²] + γT + ½λΣwⱼ²
```

Each sample goes to exactly one leaf. Group by leaf j (sample set Iⱼ):

```text
Obj = Σⱼ [ (Σᵢ∈Iⱼ gᵢ)·wⱼ + ½(Σᵢ∈Iⱼ hᵢ + λ)·wⱼ² + γ ]
```

Minimize w.r.t. wⱼ (take derivative, set to zero):

```text
∂Obj/∂wⱼ = Gⱼ + (Hⱼ + λ)·wⱼ = 0
wⱼ* = −Gⱼ / (Hⱼ + λ)
```

Plug back:

```text
Obj* = −½ Σⱼ Gⱼ²/(Hⱼ+λ) + γT
```

### Split gain derivation

Before split: one node with G, H. After split: left (G_L, H_L) and right (G_R, H_R).

```text
Gain = [Obj_before] − [Obj_after]
     = ½G²/(H+λ) − ½[G_L²/(H_L+λ) + G_R²/(H_R+λ)] + γ
     = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)] − γ
```

Gain > 0 → the split reduces the objective → do it.

### Why second-order matters

For squared loss, h = 2 (constant) — second order doesn't help much. But for other losses:

| Loss | g | h |
|---|---|---|
| Squared | 2(ŷ−y) | 2 |
| Logistic | ŷ−y | ŷ(1−ŷ) |
| Huber | clipped gradient | depends |

When h varies per sample, the Newton step weights samples differently — samples where the loss curve is flat (large h) get smaller steps, samples where it's steep get larger steps. This is **more precise** than using only the gradient.

### Complexity

```text
Split finding (exact):   O(d · n · log n)   per node
Split finding (hist):    O(d · n)           approximate, much faster
Overall training:        O(rounds · tree_cost)
Prediction:              O(depth · rounds)   per sample
Space:                   O(rounds · tree_size)
```

The exact split finder sorts each feature → O(n log n). The histogram approximation bins features → O(n). Use `tree_method='hist'` for large datasets.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "XGBoost builds trees one at a time, each fixing the previous model's errors. But unlike plain gradient boosting, it uses both the slope and curvature of the loss for precise steps, and penalizes complex trees to prevent overfitting."

> **Explain to a 12-year-old:** "It's like playing a game where you guess numbers. Each round you learn from your mistakes, but there's a rule: you can't make your answers too complicated. That rule keeps you from memorizing instead of learning."

> **Explain in an interview:** add: second-order Taylor, γT + λ‖w‖² regularization, closed-form leaf weights, split gain, sparsity-aware missing handling, shrinkage, early stopping.

> **Explain the mathematics:** derive w* = −Σg/(Σh+λ) from the quadratic objective, then show the split gain formula.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define XGBoost in one sentence.
2. What is the objective function? Write it in full.
3. Compute leaf weight given Σg=4, Σh=6, λ=2.
4. Compute split gain for a simple example.
5. How does XGBoost handle missing values?
6. What's the difference between XGBoost and plain Gradient Boosting?
7. Name 3 ways to prevent overfitting in XGBoost.
8. Why is second-order optimization better than first-order?
9. When would you choose LightGBM over XGBoost?
10. State one scenario where you would NOT use XGBoost.

---

## 33. Cheat Sheet

```text
Algorithm : XGBoost (eXtreme Gradient Boosting) · Supervised → Regression/Classification · Ensemble
Goal      : regularized, second-order additive tree boosting
Model     : F_M = Σ η·fₜ(x); Obj = ΣL + Σ(γT + ½λ‖w‖²)
Learn     : tree structures + leaf weights w* = −Σg/(Σh+λ)
Tune      : n_estimators, learning_rate, max_depth, γ, λ, subsample, colsample
Key trick : 2nd-order (g,h) + regularization + sparsity-aware
Use when  : tabular data, competitions, need accuracy + speed
Avoid when: huge categorical (→ CatBoost), images/text (→ deep learning)
Related   : Gradient Boosting · LightGBM · CatBoost · AdaBoost · Random Forest
Classification counterpart → B-classification/11 (XGBClassifier)
```

---

## 34. What Next?

You just learned the most popular tabular algorithm.

```text
XGBoost
   ├── LightGBM   (histogram + leaf-wise — faster on big data)  → next note (16)
   ├── CatBoost   (ordered boosting + categoricals)             → 17
   └── Random Forest (bagging — different philosophy)            → 09
```

> Next recommended: **16. LightGBM** — takes XGBoost's objective and makes it *fast* with histogram binning and leaf-wise growth. If XGBoost is the BMW, LightGBM is the Tesla — same destination, much faster ride.
