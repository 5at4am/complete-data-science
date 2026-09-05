# 15. XGBoost (eXtreme Gradient Boosting)

> Difficulty: ⭐⭐⭐⭐⭐ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | XGBoost (eXtreme Gradient Boosting) |
| Category | Supervised Learning (Ensemble — Gradient Boosting) |
| Type | Regression (also classification) |
| Parametric / Non-parametric | Non-parametric (additive trees) |
| Generative / Discriminative | Discriminative |
| Main Objective | Additively fit shallow trees to the negative gradient of a loss with a regularized objective, using second-order (Newton) approximations |
| Input | Feature matrix X (n×m), target y (continuous) |
| Output | Sum of all trees' leaf scores |
| Core Idea | Stage-wise additive model; each tree optimizes loss + γT + λ‖w‖²; uses first & second derivatives for fast, regularized fitting |
| Typical Use Cases | Tabular regression/classification, competitions, structured data |

---

## 02. One-Line Definition

### Beginner Definition
XGBoost builds many small decision trees, each fixing the mistakes of the previous ones, and adds them all together — with extra safety measures (regularization) so it doesn't memorize noise.

### Technical Definition
XGBoost is a scalable tree-boosting system: it sequentially adds shallow CART trees, each fitted to minimize a regularized objective `loss + γT + λ‖w‖²` using exact/approx second-order (Newton) optimization over a sparsity-aware gradient histogram.

---

## 03. Intuition

Imagine a group of friends estimating a house price. The first (simple) estimate is off. The next friend looks at the *remaining error* and corrects it. Each new friend focuses only on what's still wrong. Summing everyone's small corrections gives a great total estimate.

XGBoost is that, done with trees, but it adds two clever tricks:

1. **It penalizes complexity** (regularization) — trees that are too complicated or leaves with huge weights are discouraged. This prevents overfitting.
2. **It pays attention to not just how wrong but how quickly wrongness changes** (second derivatives / Newton method) — so it steps more precisely toward the optimum.

The "add all the corrections" structure is why XGBoost is so accurate on tabular data: shallow trees + careful learning + regularization.

---

## 04. Problem It Solves

**Problem:** Gradient boosting is powerful but needs to be (a) fast, (b) well-regularized, (c) handle sparse/missing data, and (d) excellent on structured data.

**Example:** Predicting insurance claim costs from thousands of tabular features. XGBoost handles mixed features, missing values, and non-linear interactions far better than linear models, and is more accurate/regularized than plain gradient boosting.

**Why useful:** It became the de-facto competitor winner on Kaggle because it squeezes accuracy + speed + regularization from tree boosting.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
└── Supervised Learning
    └── Regression
        └── Ensembles
            └── Boosting
                ├── AdaBoost
                ├── Gradient Boosting (basic)
                ├── XGBOOST            ← YOU ARE HERE
                ├── LightGBM
                └── CatBoost
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Additive model | Sum of small models | F(x) = Σ fₜ(x), each a tree |
| Leaf weight w | Value a leaf outputs | Optimized scalar per leaf |
| Tree complexity γT | Penalty for many leaves | T = leaves, γ = penalty per leaf |
| L2 penalty λ‖w‖² | Punish large leaf values | Ridge-like on weights |
| First derivative g | Gradient of loss | How wrong, direction |
| Second derivative h | Hessian of loss | How sharply it changes |
| Shrinkage η | Learning rate step | Multiply each tree's output |
| Column subsampling | Use subset of features/tree | Reduces variance, faster |
| Approximate histogram | Bin data for speed | Approx. split finding |

---

## 07. Input and Output

**Input:** X (n×m), y (continuous).
**Output:** ŷ = Σₜ fₜ(x), sum of tree leaf scores.

**Parameters learned:** structure and leaf weights of each tree.

**Hyperparameters:** n_estimators, max_depth, learning_rate (eta), gamma, lambda, alpha, subsample, colsample_bytree, min_child_weight.

---

## 08. Mathematical Foundation

XGBoost minimizes an additive, regularized objective at each step t:

```text
Obj(θ) = Σᵢ L(yᵢ, ŷ̂ᵢ^{(t)}) + Σ_t Ω(f_t)

where Ω(f) = γT + ½λ‖w‖²  (plus optional L1 α|w|)
```

**Notation:**
- `L` = differentiable loss (e.g., squared error)
- `ŷ̂ᵢ^{(t)}` = prediction at step t
- `f_t` = t-th tree (maps x to leaf weight w)
- `T` = number of leaves in a tree
- `γ` = complexity penalty per leaf
- `λ` = L2 weight penalty, `α` = L1 penalty
- `w` = vector of leaf weights

**Required math:** Taylor expansion (Newton), gradient & Hessian of loss, greedy tree splitting with a split-gain formula.

---

## 09. Core Formula

### Objective (with Taylor approx. to second order)

```text
Obj ≈ Σᵢ [ gᵢ · f(xᵢ) + ½ hᵢ · f(xᵢ)² ] + γT + ½λ Σ wⱼ²

gᵢ = ∂L(yᵢ, ŷ̂)/∂ŷ̂   (gradient)
hᵢ = ∂²L(yᵢ, ŷ̂)/∂ŷ̂²  (hessian)
```

#### Meaning
Drop the constant loss term; only gradient and Hessian matter. The objective becomes quadratic in the leaf weights, so it has a closed-form optimum.

#### Symbols
- `gᵢ,hᵢ` = first, second derivative of loss for sample i
- `f(xᵢ)` = tree output for sample i
- `wⱼ` = j-th leaf weight
- `T` = leaves, `γ,λ` = penalties

#### Intuition
Newton (second-order) method reaches the optimum faster and more precisely than first-order gradient descent — this is why XGBoost is accurate and converges well.

---

### Optimal Leaf Weight

```text
wⱼ* = − (Σ gᵢ) / (Σ hᵢ + λ)
```
(sum over samples i in leaf j)

#### Meaning
The best value each leaf should output, balancing fit (gradients) against the λ regularization.

#### Symbols
- `Σgᵢ` = summed gradient in leaf
- `Σhᵢ` = summed Hessian in leaf
- `λ` = L2 penalty

#### Intuition
A leaf outputs the negative mean gradient divided by the mean Hessian (plus a damping λ). λ shrinks weights → less overfit.

#### Example
A leaf with Σg=4, Σh=6, λ=2 → w* = −4/(6+2) = −0.5. If λ were 0 → −4/6 = −0.667 (larger magnitude without regularization).

---

### Split Gain

```text
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
```
Gain > 0 → add the split; larger = better.

#### Meaning
Gain = improvement after splitting a node into left (L) and right (R) minus the γ cost of one more leaf.

#### Symbols
- `G,H` = summed g,h in node and children
- `γ` = cost of adding a leaf

#### Intuition
We split only if the fit improvement (left+right gains) exceeds the complexity penalty γ. Precise, regularized splitting.

---

## 10. Derivation

**Step 1 — Additive objective at step t:** minimize
```text
Σᵢ L(yᵢ, ŷ̂ᵢ^{(t-1)} + f_t(xᵢ)) + Ω(f_t)
```

**Step 2 — Taylor expand** the loss around ŷᵢ^{(t-1)} to second order:
```text
L(y, ŷ̂+Δ) ≈ L(y, ŷ̂) + gᵢ·Δ + ½ hᵢ·Δ²
```
Dropping the constant, with Δ = f_t(xᵢ):
```text
Obj ≈ Σᵢ [ gᵢ f_t(xᵢ) + ½ hᵢ f_t(xᵢ)² ] + γT + ½λΣwⱼ²
```

**Step 3 — Group by leaves:** each sample goes to one leaf. In leaf j with sample set Iⱼ:
```text
Obj = Σⱼ [ (Σᵢ∈Iⱼ gᵢ) wⱼ + ½ (Σᵢ∈Iⱼ hᵢ + λ) wⱼ² ] + γT
```

**Step 4 — Minimize wⱼ** (quadratic):
```text
wⱼ* = −Gⱼ/(Hⱼ + λ),  Gⱼ=Σg, Hⱼ=Σh
```

**Step 5 — Optimal objective for a given leaf:** plug back:
```text
Obj* = −½ Σⱼ Gⱼ²/(Hⱼ+λ) + γT
```

**Step 6 — Split gain:** compare objective before vs after split (node → L,R):
```text
Gain = ½[ G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ) ] − γ
```

---

## 11. How the Algorithm Works

```text
Start ŷ̂ = base prediction
    ↓
For t = 1..n_estimators:
    compute gᵢ, hᵢ from current ŷ̂ and loss
    ↓
    greedily grow a tree:
        at each node, try splits (exact or histogram)
        pick split with max Gain (>0)
    ↓
    set each leaf weight wⱼ* = −Σg/(Σh+λ)
    ↓
    ŷ̂ ← ŷ̂ + η·f_t(x)       (η = learning_rate)
    ↓
Repeat
Final ŷ̂ = Σ η·f_t(x)
```

---

## 12. Training Process

- Compute g,h iteratively (Newton step).
- Grow each tree greedily, choosing best (regularized) split.
- Assign optimal leaf weights.
- Apply shrinkage η.
- (Optional) column & row subsampling per tree.

**What is learned:** tree structures + leaf weights.

**Stopping:** fixed rounds or early stopping on validation loss.

---

## 13. Objective Function / Loss Function

The objective is the sum of a **loss** (any differentiable) + **regularization**:

```text
Obj = Σ L(yᵢ, ŷ̂ᵢ) + Σ Ω(fₜ)
```
For regression commonly `L = squared error`, giving gᵢ = 2(ŷ̂−yᵢ), hᵢ = 2 (constant). The regularization Ω = γT + ½λ‖w‖² (+ α‖w‖ L1) is what makes XGBoost concrete-control overfitting.

---

## 14. Optimization

- **Second-order (Newton) descent:** uses gradient + Hessian → faster, more exact steps.
- **Greedy tree structure search:** at each split maximize Gain (a per-split closed form).
- **Histogram approximation** for large data: bin features, approximate gains.
- **Sparsity-aware splits:** handle missing values by learning default directions.
- **Shrinkage η:** scale each tree → slower, smoother, less overfit.
- **Early stopping** on validation loss.

This combination (exact/approx Newton + regularization + shrinkage) is why XGBoost is both accurate and fast.

---

## 15. Complete Numerical Example

Data: x = [1,2,3,4], y = [1,3,3,5]. Loss = squared error. No regularization (λ=0, γ=0), η=1 for clarity, base ŷ̂=mean = 3, max_depth=1 (stump per round).

**Round 1 — compute gradients (squared error):**
```text
L = (y − ŷ̂)²; g = 2(ŷ̂ − y) = 2(3 − y); h = 2
y = [1,3,3,5] → g = [4, 0, 0, −4]
```

**Try split at x ≤ 2** (left: samples 1,2; right: 3,4):
```text
G_L = 4+0 = 4, H_L = 2+2 = 4
G_R = 0+(−4) = −4, H_R = 4
Gain = ½[4²/4 + (−4)²/4 − 0²/8] − 0   (=½[4+4])= 4
```

**Try split at x ≤ 3** (left: 1,2,3; right: 4):
```text
G_L = 4+0+0=4, H_L=6; G_R=−4, H_R=2
Gain = ½[16/6 + 16/2 − 0] = ½[2.667+8] = 5.33
```
Round-1 best split: x ≤ 3 (gain 5.33).

**Leaf weights:**
```text
left (x≤3): w = −G_L/H_L = −4/6 = −0.667
right (x=4): w = −G_R/H_R = 4/2 = 2.0
```

**Prediction update:** ŷ̂ᵢ += f₁(xᵢ)
```text
x=1: 3 + (−0.667) = 2.333
x=2: 3 + (−0.667) = 2.333
x=3: 3 + (−0.667) = 2.333
x=4: 3 + 2.0 = 5.0
```
New errors: |y−ŷ̂| = [1.33, 0.667, 0.667, 0] — already better.

**Round 2 — recompute g, h from new ŷ̂ and fit another tree** (refines left group toward 1,3,3).

**VERIFIED EXAMPLE** — hand-verified. Shows gradient computation, split-gain, closed-form leaf weights, and additive update.

---

## 16. Visual Explanation

```text
XGBoost = additive sum of shallow trees

f₁:  x≤3 → −0.667 ; else → 2.0
f₂:  (refines remaining error)
f₃:  ...
F(x) = η(f₁ + f₂ + f₃ + ...)

Regularization:
   fewer leaves (γ)  → simpler trees
   small weights (λ) → shrink leaf values → robust
```

```text
Newton vs Gradient step:
   gradient  : move based on slope (slow, crudely)
   Newton    : move using slope + curvature (precise)
```

---

## 17. Algorithm / Pseudocode

```text
Input: X, y, objective(L), rounds, γ, λ, η, max_depth, subsamples
ŷ̂ = base_prediction (mean)
for t = 1..rounds:
    gₜ = ∂L(y, ŷ̂)/∂ŷ̂ ; hₜ = ∂²L(y, ŷ̂)/∂ŷ̂²
    build tree:
        for each node, evaluate splits:
            Gain = ½[Σg_L²/(Σh_L+λ)+Σg_R²/(Σh_R+λ)−Σg²/(Σh+λ)] − γ
            pick max-gain split if >0
        assign leaf weight wⱼ = −Σgⱼ/(Σhⱼ+λ)
    ŷ̂ = ŷ̂ + η·f_t(x)
end
return ŷ̂
```

---

## 18. From-Scratch Implementation

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
                    best_gain = gain; best_col = col; best_thr = (xc_s[i]+xc_s[i+1])/2
        return best_col, best_thr, best_gain

    def fit(self, X, y):
        X = np.asarray(X, float); y = np.asarray(y, float)
        pred = np.full(len(y), y.mean())
        for _ in range(self.n_estimators):
            g, h = self._squared_grad_hess(y, pred)
            tree = {}
            col, thr, gain = self._best_split(X, g, h)
            if gain <= 0 or col is None:
                tree['w'] = -g.sum()/(h.sum()+self.reg_lambda)
                tree['col'] = tree['thr'] = None
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

---

## 19. Code Explanation

```text
Line:  g = 2*(pred-y); h = 2
   What: gradient & Hessian for squared loss
   Why: Newton step needs both derivatives
   Math: L=(y-p)² → g=2(p-y), h=2

Line:  gain = 0.5*(GL²/(HL+λ)+GR²/(HR+λ)−G²/(H+λ)) − γ
   What: regularized split-gain
   Why: pick best split, discount complexity
   Math: from derivation §10

Line:  w = -G/(H+λ)
   What: optimal leaf weight
   Why: closed-form Newton optimum
   Math: −Σg/(Σh+λ)

Line:  pred += lr*_predict_tree(...)
   What: shrinkage additive update
   Why: smaller steps → less overfit
   Math: ŷ̂ += η·f
```

---

## 20. Library Implementation

```python
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=500, n_features=15, noise=0.1,
                       random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)

model = xgb.XGBRegressor(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    reg_lambda=1.0, gamma=0.0, subsample=0.8, colsample_bytree=0.8,
    objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train,
          eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

params = {'n_estimators':[100,200], 'max_depth':[3,5],
          'learning_rate':[0.05,0.1], 'reg_lambda':[0.0,1.0,10.0]}
grid = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror',
                     random_state=42), params, cv=5)
grid.fit(X_train, y_train)
print("Best:", grid.best_params_)
print("Feature importance:", model.feature_importances_)
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical |
|---|---|---|---|
| n_estimators | Number of trees | More → bias ↓, risk overfit | 100–1000 |
| learning_rate (eta) | Shrinkage per tree | Lower → more robust | 0.01–0.3 |
| max_depth | Tree depth | Higher → complex interactions | 3–8 |
| min_child_weight | Min Σh per leaf | Higher → simpler | 1–10 |
| gamma | Min gain to split | Higher → fewer splits | 0–5 |
| reg_lambda (λ) | L2 on leaf weights | Higher → shrink | 1 |
| reg_alpha (α) | L1 on leaf weights | Sparsity/feature selection | 0 |
| subsample | Row sampling | Variance ↓ | 0.5–1.0 |
| colsample_bytree | Feature sampling | Variance ↓, speed | 0.5–1.0 |

**Too deep/many trees** → overfit (mitigate with eta, λ, γ, subsample). **Too small** → underfit.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Tree structures (split features/thresholds)
- Leaf weights wⱼ

### Hyperparameters (chosen)
- n_estimators, max_depth, learning_rate, gamma, lambda, alpha, min_child_weight, subsamples, colsample

---

## 23. Assumptions

| Assumption | What | Why | Check | If violated |
|---|---|---|---|---|
| Loss differentiable | Need g, h | Newton step | Use diff'able loss | Pick compatible loss |
| Additivity | Sum of trees | Model form | — | Use other model |
| Enough data per leaf | min_child_weight | Stability | Leaf sample check | Raise min_child_weight |
| No extreme label noise | Trees can fit noise | Bias/Variance | Residual/CV | Early stop, subsample |

Generally assumption-light (no linearity/scaling/normality). Main practical caution: overfitting on small/noisy data.

---

## 24. Data Requirements

- **Type:** numeric (can handle categorical via encoding; native categorical in newer versions/encoders).
- **Missing:** XGBoost handles natively (sparsity-aware default direction).
- **Outliers:** generally robust; extreme outliers handled by loss choice/anchor.
- **Scaling:** unnecessary (trees) — scale-invariant to monotone transforms.
- **Dataset size:** works small→large; histogram mode for big.
- **Label noise:** mitigate via subsample, η, early stop, gamma/lambda.

---

## 25. Feature Scaling

**Unnecessary** — tree thresholds are scale-invariant to per-feature monotone transforms. Do not standardize features for XGBoost. (Only matters if you also use distance-based components, which you typically don't.)

---

## 26. Evaluation Metrics

(Same regression family: MSE, RMSE, MAE, R², and objective-specific custom objectives.)

**Training vs evaluation:** use validation/eval-set metrics with early stopping — XGBoost can overfit given many rounds; monitor validation loss and stop when it starts rising, and compare train vs test for overfitting signal.

---

## 27. Advantages

| Advantage | Why matters |
|---|---|
| Very accurate | Best-in-class tabular results |
| Regularization (γ, λ, L1/L2) | Strong overfitting control |
| Newton second-order | Fast, precise convergence |
| Handles missing values | Sparsity-aware default paths |
| Feature importance | Interpretability |
| Column/row subsampling | Variance control, fast |
| Highly parallelizable | Scales to big data |
| Wide software support | xgboost libraries in many langs |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| Many hyperparameters | Requires careful tuning |
| Slower than LightGBM/histogram | On very large datasets |
| Less interpretable than single tree | Ensemble of many trees |
| Memory on exact splits | Histogram mode mitigates |
| Risk of overfit if untuned | Needs early stopping |
| Categorical handling weaker than CatBoost | Encoding needed typically |

---

## 29. When to Use

✓ Tabular/structured regression & classification.
✓ Competitions (Kaggle default favorite).
✓ Mixes of numeric features, some missing values.
✓ Need feature importance.
✓ Both small & large datasets.
✓ Looking for top accuracy with reasonable speed.

---

## 30. When NOT to Use

✗ Image/text/audio (deep learning better).
✗ Tiny datasets than can overfit badly.
✗ Extreme latency/memory constraints (consider LightGBM).
✗ Heavily categorical (consider CatBoost).
✗ Need full interpretability (single tree/linear).

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| House/bike price prediction | tabular | XGBoost | Price |
| Insurance claim severity | policy features | XGBoost | Claim amount |
| Sales/demand forecasting | time features | XGBoost | Quantity |
| Ad CTR/value | user features | XGBoost | Metric value |
| Recommendation value | interaction features | XGBoost | Score |

---

## 32. Failure Cases

- **Untuned overfitting:** too deep/many trees without η/λ → memorize train.
- **Very small data:** powerful trees overfit → use simpler/regularized.
- **Extreme categorical cardinality:** poor default handling → use CatBoost or target-encoding.
- **Massive sparse data:** memory growth → histogram/approx or LightGBM.
- **Non-differentiable loss chosen incorrectly:** g,h undefined → pick appropriate objective.

---

## 33. Overfitting and Underfitting

- **Underfitting:** too few/short trees, η too small, too much regularization.
- **Overfitting:** too many/deep trees, small η not applied, low λ, many rounds without early stop.
- **Balance:** tune η↔n_estimators; use γ, λ, subsample; monitor validation with early stopping; keep max_depth modest.

---

## 34. Bias-Variance Perspective

- Boosting is **bias-reducing**: additive trees correct residual error.
- XGBoost controls **variance** via shrinkage (η), regularization (γ, λ), subsampling — this is exactly why it beats naive gradient boosting on noisy data.
- The tradeoff: more rounds lower bias but raise variance; η, λ, subsample lower variance at some bias cost.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Gradient Boosting | Fit residuals/gradient | Simple | No reg., slower | Prototype |
| XGBoost | + Newton + regularization | Accurate, reg., parallel | Tunable/complex | Competitions/production |
| LightGBM | Histogram + leaf-wise | Faster, memory-light | Overfit w/ leaf-wise | Large datasets |
| CatBoost | Ordered boosting, categoricals | Best categorical, robust | Slower to tune | Heavy categorical |
| Random Forest | Bagged deep trees | Robust, parallel | Lower peak accuracy | Baseline |

---

## 36. Algorithm Selection Guide

```text
Tabular supervised?
├── Heavy categorical → CATBOOST
├── Very large data / speed → LIGHTGBM
├── Default powerful → XGBOOST
├── Need interpretable → SINGLE TREE / RANDOM FOREST
└── Small quick baseline → RANDOM FOREST / REGULARIZED LR
```

---

## 37. Common Mistakes

```text
❌ Forgetting regularization (λ, γ)
Fix: use them — that's XGBoost's edge.

❌ Very small η with way too many trees / no early stop
Fix: set eval set + early stopping.

❌ Not tuning max_depth (too deep default)
Fix: keep 3–8 and tune.

❌ No subsampling → overfit
Fix: subsample, colsample_bytree.

❌ Scaling features (unnecessary for trees)
Fix: skip — monotone-invariant.

❌ Treating categorical by one-hot explosion
Fix: consider CatBoost/target-encoding/native categorical.

❌ Misusing custom objective (wrong g,h)
Fix: verify gradient/Hessian via numeric check.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is XGBoost?**
A: Regularized tree boosting using Newton (second-order) steps to additively fit shallow trees minimizing loss + complexity penalty.

**Q2. What does additive mean here?**
A: F(x) = Σ fₜ(x); each tree's output is added to the ensemble total.

**Q3. What regularization does it use?**
A: γT (leaf-count penalty) + λ‖w‖² (L2 leaf weights), optional α‖w‖ L1.

### Intermediate
**Q4. Why Newton (second order)?**
A: Uses gradient + Hessian → more precise, faster-converging steps than first-order.

**Q5. How is leaf weight computed?**
A: w* = −Σg/(Σh+λ), using summed gradient & Hessian.

**Q6. What is the split gain?**
A: ½[GL²/(HL+λ)+GR²/(HR+λ)−G²/(H+λ)] − γ; split if >0.

### Advanced
**Q7. Compare XGBoost and LightGBM.**
A: XGB uses level-wise exact/approx Newton; LightGBM uses histogram + leaf-wise, faster on large data but risks overfit.

**Q8. How does it handle missing values?**
A: Sparsity-aware: learns default direction for missing values at each node.

**Q9. Derive the leaf weight from the objective.**
A: Quadratic in w → w*=−G/(H+λ); see §10.

**Q10. Why shrinkage η matters.**
A: Scales each tree's contribution → smoother, robust, lower variance; pairs with more trees.

---

## 39. GATE / Exam Perspective

**Key formulas:**
```text
Obj = ΣL(y,ŷ̂) + Σ(γT + ½λ‖w‖²)
wⱼ* = −Σg/(Σh+λ)
Gain = ½[GL²/(HL+λ)+GR²/(HR+λ)−G²/(H+λ)] − γ
ŷ̂ += η·f(x)
```

**Concepts:** additive trees, gradient+Hessian, regularization, leaf weight closed form, shrinkage, split gain.

> **Representative pattern question (NOT a past GATE PYQ):** "Why does XGBoost generalize better than plain gradient boosting on noisy data?" Answer: the γ/λ regularization and shrinkage control the variance that plain boosting adds.

**Traps:**
- Forgetting the `−γ` and the `λ` denominators in gain/weight.
- Confusing gradient boosting (first-order) with XGBoost (second-order).
- Assuming scaling is needed (it isn't for trees).
- Writing g wrongly for squared loss (g=2(ŷ̂−y), h=2).

---

## 40. Coding Practice

**L1:** Compute g,h for squared loss.
**L2:** Find best split by gain.
**L3:** Implement closed-form leaf weights.
**L4:** Full simple XGBoost (as §18).
**L5:** sklearn/XGBoost library usage with early stopping.
**L6:** Tune η, depth, λ, subsample via CV.
**L7:** Case study — predict on a tabular regression dataset; compare XGBoost vs LightGBM vs RF on RMSE + runtime; report feature importance.

---

## 41. Practical ML Workflow

```text
Problem → tabular regression
   ↓
EDA → inspect features, missing, outliers, distribution
   ↓
Clean → encode/aggregate; let XGBoost handle missing (or impute)
   ↓
Split → train/val/test
   ↓
No scaling (trees)
   ↓
Baseline → simple model for reference
   ↓
Train → XGBRegressor(+ eval set)
   ↓
Tune → η & n_estimators & depth & λ & subsample via CV
   ↓
Early stop → on validation
   ↓
Evaluate → RMSE/R² on test
   ↓
Compare → LightGBM/CatBoost/RF
   ↓
Deploy → best
   ↓
Monitor → drift
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Split finding (exact) | O(d · n log n) | per node |
| Split finding (histogram) | O(d · n) approx | large data |
| Overall training | O(rounds · trees) | parallelizable |
| Prediction | O(depth · rounds) | per sample |
| Space | O(rounds · tree size) | fit memory |

---

## 43. Advanced Concepts

- **Newton boosting / second-order approximation** — the core upgrade.
- **Histogram (approx) split finding** for huge datasets (`hist`, `hist`/`gpu_hist`).
- **Sparsity-aware default direction** for missing values.
- **Column/row subsampling** (`colsample_by*`, `subsample`).
- **Custom objectives / evaluation metrics** (define g, h yourself).
- **Monotonic constraints**, interaction constraints for specialized models.
- **Boosting with ranker/rank objectives** (for ranking).

---

## 44. Connections to Other Algorithms

```text
XGBoost
   ├── Gradient Boosting (1st-order) → XGBoost adds 2nd-order + reg.
   ├── LightGBM (histogram, leaf-wise) — friendly rival
   ├── CatBoost (ordered boosting, categoricals)
   ├── AdaBoost (sample reweighting)
   ├── Random Forest (bagging, no boosting)
   └── Regularized linear models (share λ L2 idea)
```

---

## 45. If You Remember Only 5 Things

1. XGBoost adds shallow trees additively, each fitting the residual of the sum.
2. It uses **Newton (gradient + Hessian)** for precise fast steps.
3. Objective = loss + **γT + λ‖w‖²** regularization (its overfitting control).
4. Leaf weight is closed-form: `−Σg/(Σh+λ)`; split by the gain formula.
5. It's the best default for tabular data; tune η, depth, λ, subsample, early stop.

---

## 46. Cheat Sheet

```text
Algorithm   : XGBoost (eXtreme Gradient Boosting)
Category    : Supervised, Regression (also classification), boosting
Goal        : Additively fit regularized shallow trees via Newton steps
Input       : X (n×m) numeric, y
Output      : ŷ = Σ η·fₜ(x)
Core Formula: Obj=ΣL+Σ(γT+½λ‖w‖²); w=−Σg/(Σh+λ); Gain formula
Optimization: 2nd-order greedy tree boosting + shrinkage + histograms
Parameters  : tree structures + leaf weights
Hyperparams : n_estimators, max_depth, eta, γ, λ, α, subsample, colsample, min_child_weight
Loss        : differentiable (default squared)
Assumptions : loss differentiable; enough data/leaf
Advantages  : very accurate, regularized, parallel, feature importance
Disadvantages: many hyperparams; slower than LightGBM; fewer native categoricals
Use When    : tabular, competitions, top accuracy
Avoid When  : images/text, extreme categorical, tiny data (careful)
Related     : GB, LightGBM, CatBoost, RF
Key Exam    : Newton, leaf weight, gain, regularization
Key Interv  : vs LightGBM, derive leaf weight, missing handling, why reg
```

---

## 47. Final Mental Model

```text
Start at base ŷ̂
   ↓ each round
compute g,h from current loss
   ↓
grow tree maximizing regularized gain
   ↓
set leaf weights −Σg/(Σh+λ)
   ↓
shrink & add: ŷ̂ += η·tree
   ↓
repeat
   ↓
regularization (γ,λ) + shrinkage keep it from overfitting
```

---

## 48. Knowledge Check

### Recall (5)
1. Write the objective in full.
2. What are g and h?
3. Give leaf weight formula.
4. Give the split gain.
5. What does η do?

### Understanding (5)
6. Why Newton (2nd order)?
7. Why regularization (γ, λ)?
8. Why does shrinkage help generalization?
9. Why handle missing natively?
10. Why level-wise default split/why boost additively?

### Application (5)
11. Run one round of §15 by hand.
12. Choose η & n_estimators for a noisy dataset.
13. Handle missing values.
14. Tune subsample/colsample.
15. Early-stopping strategy.

### Mathematical (5)
16. Derive leaf weight from objective.
17. Derive the split gain.
18. Compute g,h for squared & Huber-like loss.
19. Explain how λ changes leaf weights.
20. Derive why Newton step is accurate.

### Interview (5)
21. "XGBoost vs LightGBM?"
22. "Why is XGBoost better than plain gradient boosting?"
23. "How do you prevent overfitting?"
24. "How are missing values handled?"
25. "What's a custom objective / how to specify g,h?"

### Problem Solving (5)
26. Very large tabular data — approach?
27. Heavy categorical — pick CatBoost vs XGBoost?
28. Overfitting — which knobs?
29. Need both accuracy and speed — how to tune.
30. Extreme outliers — loss choice?

## Answers (explained)
1. Obj=ΣₜL(y,ŷ̂ₜ)+Σₜ(γT+½λ‖w‖²).
2. g=gradient ∂L/∂ŷ̂; h=Hessian ∂²L/∂ŷ̂².
3. −Σg/(Σh+λ).
4. ½[GL²/(HL+λ)+GR²/(HR+λ)−G²/(H+λ)]−γ.
5. η shrinks each tree's contribution (smoother, less overfit).
6–30: refer to §10–14, §23–33. For (27): use CatBoost. For (28): raise η? No — lower η, add λ/γ, lower depth, subsample, early stop. For (30): robust loss (Huber) or winsorize, and check residuals.

---

## 49. Final Learning Checklist

- [ ] I can define additive tree boosting
- [ ] I understand Newton (g, h) steps
- [ ] I can write the objective + regularization
- [ ] I can compute leaf weight
- [ ] I can compute split gain
- [ ] I understand shrinkage η
- [ ] I understand missing-value handling
- [ ] I know how to prevent overfitting
- [ ] I can implement from scratch
- [ ] I can use xgboost library
- [ ] I can set up early stopping
- [ ] I can tune hyperparameters
- [ ] I can compare with LightGBM/CatBoost/RF
- [ ] I understand feature importance
- [ ] I can define a custom objective
- [ ] I know when to use/avoid
- [ ] I understand histogram/approx mode
- [ ] I can apply it in a workflow
- [ ] I know its bias-variance role
- [ ] I can reason about its real-world success

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** Objective, leaf weight, split-gain formulas and worked example verified (hand-computed gradient, gain, leaf weights).
- **Beginner-friendliness:** Friends-correcting-analogy, additive & regularization ASCII, short paragraphs, tables.
- **Math depth:** Full derivation of leaf weight + gain; Newton justification.
- **Practical depth:** From-scratch Newton tree, library use, tuning, workflow, comparison, early stopping.
- **Exam depth:** Gain/weight formulas, non-PYQ representative questions.
- **Structure:** All 50 sections in order.

**Verified:** Section 15 worked example hand-verified.
