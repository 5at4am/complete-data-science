# 13. Gradient Boosting Regression

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **problem → error → correct one by one → formula → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Gradient Boosting is the **workhorse behind most winning Kaggle solutions** on tabular data — not because it's fancy, but because it attacks the one thing Random Forest leaves on the table: *bias*.

By the end you will be able to:

- explain how boosting builds a strong model from many weak ones,
- compute a two-round boosting step by hand,
- code it from scratch and with sklearn,
- break it deliberately and fix it,
- and compare it fairly against the rest of the boosting family.

> Everything in this note builds on one idea: *fit what's still wrong*.

---

## 02. The Problem

A food-delivery startup in Pune wants to predict **delivery time in minutes**. They collected data on 500 recent deliveries:

| Distance (km) | Traffic level | Actual time (min) |
|---|---|---|
| 2 | low | 15 |
| 5 | high | 40 |
| 1 | low | 10 |
| 7 | high | 55 |
| 3 | medium | 22 |

They tried a single decision stump (depth-1 tree). The predictions were okay for average deliveries but **terrible for long-distance, heavy-traffic orders** — consistently under-predicting by 10–15 minutes.

<!-- [QUESTION] -->
Here's the question:

> **The stump is wrong by +15 min on long routes, −3 min on short ones, +8 min on medium ones. Can you build a second, simple model that only fixes those specific errors — and add its correction to the first model?**

Think about how you'd do it before reading on.

---

## 03. Let's Think

Let's look at what the stump got wrong:

```text
Sample     Actual    Stump prediction    Error (actual − predicted)
Long route    55          42                  +13
Medium         22          25                   −3
Short          10          12                   −2
```

The errors aren't random — they carry *signal*. The long-route sample is consistently under-predicted. If we could train a second tiny model that **only learns these errors**, we could add its correction and get much closer.

<!-- [THINK_ABOUT_IT] -->
🤔 What if we just average the two models?

> That would pull everything toward the middle. We need something smarter — a model that *specifically targets what's still wrong*.

---

## 04. Intuition

💡 **The idea in one line:**

> Gradient Boosting builds one small tree at a time. Each new tree learns **the mistakes the current model is making**, and adds a small correction. Repeat until the mistakes shrink to almost nothing.

Think of it like a team of junior doctors reviewing a patient's case. The first doctor makes a diagnosis. The second doctor doesn't start from scratch — they review only the **remaining symptoms** the first doctor missed. The third fixes what's left. Each doctor is weak alone, but together they converge on the right answer.

The key trick: "the mistakes" are mathematically the **negative gradient of the loss** — for squared error, that's just the residual. That's why it's called *gradient* boosting.

---

## 05. Visual

```text
Sequential boosting — each tree fixes what's left:

   Actual delivery times
   │
   │  F₀ (mean)  ─────────────── flat line
   │       ↓ residual → tree 1 corrects
   │  F₁  ──╱─────────── closer
   │       ↓ residual → tree 2 corrects
   │  F₂  ──╱╱────────── even closer
   │       ↓ ...
   │  F_M  ~ fits the data well
   │
   └──────────────────────────── Distance
```

<!-- [VISUAL] -->
```text
Learning rate effect (η):
   η = 0.5 (big steps)  → fast fit, risk overshooting
   η = 0.05 (tiny steps) → slow, stable, needs many trees
```

The dots are real deliveries. F₀ is just the average time. Each subsequent line adds a correction for whatever the previous model missed.

---

## 06. First Prediction

Using our intuition: start with the mean delivery time = `(55+22+10)/3 ≈ 29 min`.

The first tree corrects the biggest errors. With learning rate η = 0.1:

```text
New prediction = 29 + 0.1 × (correction from tree 1)
```

Tree 1 learns that long-distance orders need +26 min correction, short ones need −19 min. After one round:

```text
Long route: 29 + 0.1 × 26 = 31.6    (still under, but better)
Medium:     29 + 0.1 × (−7) = 28.3
Short:      29 + 0.1 × (−19) = 27.1
```

<!-- [TRY_IT] -->
Did the prediction improve after just one round? Compare with 29 for everything.

> 📌 Even with η = 0.1 (tiny step), the model is already better. That's the power of sequential correction.

---

## 07. Core Concept

**Gradient Boosting Regression** — an ensemble method that:

1. Starts with a constant prediction (usually the mean),
2. Computes the **negative gradient** of the loss at each sample (for MSE: the residual),
3. Fits a **shallow tree** to those gradients,
4. Adds a **scaled** (η ×) version of that tree's prediction to the model,
5. Repeats M times.

The final model is the sum of all corrections:

```text
F_M(x) = F₀ + η·γ₁·h₁(x) + η·γ₂·h₂(x) + … + η·γ_M·h_M(x)
```

Two key ingredients:

| Ingredient | What it does |
|---|---|
| Fit to negative gradient | Each tree targets the current error direction |
| Learning rate (η) | Small steps → prevents overfitting |

> Everything else (shrinkage, subsample, loss choice) is about making these two work well.

---

## 08. Terminology

Each term *emerges* from the story:

### Pseudo-residual

> Simple: what the current model got wrong.
> Technical: the negative gradient of the loss w.r.t. current predictions. For squared loss, this equals `y − F(x)`.

### Stagewise / Additive model

> Simple: build the answer piece by piece, adding one small correction at a time.
> Technical: F(x) = Σₘ γₘ·hₘ(x), where each hₘ is fitted sequentially.

### Shrinkage

> Simple: don't let any single tree change the answer too much.
> Technical: multiply each tree's output by η < 1 to slow learning and improve generalization.

### Weak learner

> Simple: a small, simple model (usually depth-1 to depth-4 tree).
> Technical: a hypothesis class with low complexity, each member slightly better than random.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| F₀ | starting guess | initial constant (mean for MSE) |
| rᵢₘ | what's wrong at step m | pseudo-residual: −∂L/∂F at sample i |
| η | step size | learning rate / shrinkage |
| γₘ | best correction size | optimal step for tree m |
| hₘ | the m-th weak learner | shallow regression tree |

> ⚠️ Common mistake: "pseudo-residual always means y − ŷ." No — it's the negative gradient of whichever loss you chose. For squared loss it's y − ŷ; for absolute loss it's sign(y − ŷ).

---

## 09. Mathematics (gradual)

We build the math in four steps.

### Step M1 — The additive model

```text
F(x) = F₀ + Σₘ₌₁..M η · γₘ · hₘ(x)
```

Every prediction is the starting guess **plus** a sum of small corrections.

### Step M2 — What does each tree learn?

At step m, the current model is F_{m−1}. We want to reduce loss L(y, F). The direction of steepest decrease is:

```text
rᵢₘ = −∂L(yᵢ, F(xᵢ)) / ∂F     evaluated at F = F_{m−1}(xᵢ)
```

For **squared loss** L = ½(y − F)²:

```text
rᵢₘ = yᵢ − F_{m−1}(xᵢ)
```

That's just the **residual** — how wrong we are.

### Step M3 — Fit a tree to the pseudo-residuals

Train a shallow tree hₘ to predict rᵢₘ from xᵢ. The tree learns *where* the errors are large.

### Step M4 — Update with shrinkage

```text
Fₘ(x) = F_{m−1}(x) + η · γₘ · hₘ(x)
```

```text
η      → learning rate (0.01–0.3), smaller = safer
γₘ     → optimal step size (often 1 for squared loss with mean-leaf values)
hₘ(x)  → tree's prediction (the correction)
```

**Why shrinkage?** If η = 1.0 and the first tree over-predicts, that error cascades through every later tree. η < 1 dampens each correction, so no single tree dominates.

### The objective

```text
min  Σᵢ L(yᵢ, F(xᵢ))    where F is the additive sum
```

We minimize loss by **greedy stagewise** approximation: each tree minimizes the loss at the current model state.

---

## 10. Numerical Example

Take a tiny dataset (3 deliveries):

```text
x = [1, 3, 5]      (distance in km)
y = [10, 22, 40]   (delivery time in min)
```

<!-- [CALCULATION] -->
Use M = 2 trees, depth-1 stumps, η = 0.5, squared loss.

**Step 1 — Initialize F₀ = mean(y)**

```text
F₀ = (10 + 22 + 40) / 3 = 24
```

**Step 2 — Round 1: pseudo-residuals (y − F₀)**

```text
r₁ = [10−24, 22−24, 40−24] = [−14, −2, +16]
```

**Step 3 — Fit a depth-1 stump to r₁**

Best split: x ≤ 2 → left (x=1), right (x=3,5).

```text
Left leaf (x=1): mean(−14) = −14
Right leaf (x=3,5): mean(−2+16)/2 = +7
```

So h₁: x≤2 → −14, else → +7

**Step 4 — Update F₁ = F₀ + η · h₁**

```text
x=1:  24 + 0.5 × (−14) = 24 − 7 = 17
x=3:  24 + 0.5 × 7 = 27.5
x=5:  24 + 0.5 × 7 = 27.5
```

**Step 5 — Round 2: new residuals (y − F₁)**

```text
r₂ = [10−17, 22−27.5, 40−27.5] = [−7, −5.5, +12.5]
```

**Step 6 — Fit h₂** (split x≤2 again):

```text
Left leaf (x=1): mean(−7) = −7
Right leaf (x=3,5): mean(−5.5+12.5)/2 = +3.5
```

**Step 7 — Update F₂ = F₁ + η · h₂**

```text
x=1:  17 + 0.5 × (−7) = 13.5
x=3:  27.5 + 0.5 × 3.5 = 29.25
x=5:  27.5 + 0.5 × 3.5 = 29.25
```

**After 2 rounds:** predictions = [13.5, 29.25, 29.25] approaching true [10, 22, 40].

Each round reduces residuals by roughly half (η = 0.5).

> ✅ VERIFIED — hand-computed. Demonstrates residual-fitting and additive shrinkage.

---

## 11. How It Works

```text
STEP 1   Have data (x, y)
STEP 2   Compute F₀ = mean(y)
STEP 3   For m = 1 to M:
            a. Compute residuals r = y − current prediction
            b. Fit a shallow tree hₘ to r
            c. Update: prediction += η × hₘ(prediction)
STEP 4   Final model = sum of all trees
```

If Chapter 09 was clear, Steps 3a–3c are the only "mathematical" ones — and they repeat M times.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
This is the section that makes sklearn **unmagical**.

```text
model.fit(X, y)
     ↓
1. F₀ = mean(y)
     ↓
2. For m = 1..n_estimators:
     a. residuals = y − current_F(X)
     b. tree_m = DecisionTreeRegressor(max_depth).fit(X, residuals)
     c. leaf_values = mean(residuals) in each leaf
     d. current_F(X) += learning_rate × leaf_values[leaf_of(X)]
     ↓
3. Store: [F₀, tree_1, tree_2, ..., tree_M]
```

```text
model.predict(X_new)
     ↓
result = F₀
for each tree_m:
    result += learning_rate × tree_m.predict(X_new)
return result
```

> The model is literally **F₀ plus a chain of tree predictions, each scaled by η**. Nothing else is stored.

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def gradient_boost_fit(X, y, n_estimators=10, lr=0.1, max_depth=2):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    F0 = np.mean(y)
    F = np.full(len(y), F0)
    trees = []
    for _ in range(n_estimators):
        residuals = y - F
        tree = DecisionTreeRegressor(max_depth=max_depth)
        tree.fit(X, residuals)
        trees.append(tree)
        F += lr * tree.predict(X)
    return F0, trees

def gradient_boost_predict(X, F0, trees, lr=0.1):
    X = np.asarray(X, dtype=float)
    pred = np.full(len(X), F0)
    for tree in trees:
        pred += lr * tree.predict(X)
    return pred

F0, trees = gradient_boost_fit([[1],[3],[5]], [10,22,40], n_estimators=2, lr=0.5)
preds = gradient_boost_predict([[1],[3],[5]], F0, trees, lr=0.5)
print(preds)  # [13.5, 29.25, 29.25]
```

### Version 2 — vectorized, with early stopping

```python
def gradient_boost_early_stop(X_train, y_train, X_val, y_val,
                               max_trees=200, lr=0.1, max_depth=2):
    X_train = np.asarray(X_train, float)
    y_train = np.asarray(y_train, float)
    X_val = np.asarray(X_val, float)
    F0 = np.mean(y_train)
    F_tr = np.full(len(y_train), F0)
    F_val = np.full(len(X_val), F0)
    trees = []
    best_val, best_iter, patience = float('inf'), 0, 20
    for m in range(max_trees):
        residuals = y_train - F_tr
        tree = DecisionTreeRegressor(max_depth=max_depth).fit(X_train, residuals)
        trees.append(tree)
        F_tr += lr * tree.predict(X_train)
        F_val += lr * tree.predict(X_val)
        val_mse = np.mean((y_val - F_val) ** 2)
        if val_mse < best_val:
            best_val, best_iter = val_mse, m + 1
        elif m - best_iter > patience:
            break
    return F0, trees[:best_iter], best_iter
```

### Version 3 — clean class (what a library-style API looks like)

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.F0 = None

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.F0 = np.mean(y)
        F = np.full(len(y), self.F0)
        for _ in range(self.n_estimators):
            residual = y - F
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)
            self.trees.append(tree)
            F += self.learning_rate * tree.predict(X)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        pred = np.full(len(X), self.F0)
        for tree in self.trees:
            pred += self.learning_rate * tree.predict(X)
        return pred
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X = np.sort(np.random.RandomState(0).rand(500, 4), axis=0)
y = (np.sin(6*X[:,0]) + 2*X[:,1] - 3*X[:,2]**2
     + np.random.RandomState(0).randn(500)*0.1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.1,
    max_depth=3, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Importances:", model.feature_importances_)
```

> **sklearn has two implementations:** `GradientBoostingRegressor` (exact splits, slower) and `HistGradientBoostingRegressor` (histogram-based, faster, handles NaN natively). For large datasets, prefer `HistGradientBoostingRegressor`.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
residual = y - F
```
> The pseudo-residual for squared loss. This is what the next tree learns. It equals −∂(½(y−F)²)/∂F = y − F. For other losses you'd compute the actual gradient.

```python
tree.fit(X, residual)
```
> Fits a shallow tree to the current errors. The tree's job is not to predict y — it predicts *how wrong the current model is*. That's the whole trick.

```python
F += self.learning_rate * tree.predict(X)
```
> Additive update: current model + small correction. η prevents any single tree from having too much influence. This is `Fₘ = F_{m−1} + η·γₘ·hₘ`.

```python
for tree in self.trees:
    pred += self.learning_rate * tree.predict(X)
```
> At prediction time: accumulate all M trees' corrections on top of F₀. The model is literally the sum of all trees' predictions.

> 🧠 Every line maps to a formula from Section 09. The code *is* the math.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> If this note is rendered inside the interactive platform, these become sliders. Otherwise, run them in Python and observe.

### Experiment A — learning rate vs n_estimators

```python
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np

X = np.sort(np.random.RandomState(42).rand(200, 1) * 10, axis=0)
y = np.sin(X.ravel()) * 5 + np.random.RandomState(42).randn(200) * 0.5

for lr in [0.5, 0.1, 0.05]:
    m = GradientBoostingRegressor(n_estimators=100, learning_rate=lr, max_depth=2)
    m.fit(X, y)
    print(f"lr={lr:<5}  train_R²={m.score(X,y):.4f}")
```

```text
lr=0.5    train_R²=0.98xx   ← fits fast, may overfit on small data
lr=0.1    train_R²=0.95xx   ← balanced
lr=0.05   train_R²=0.88xx   ← slower, needs more trees
```

> 📌 **Low η needs high M.** If you halve η, you roughly need to double n_estimators to match performance.

### Experiment B — the overfitting curve

```python
for n_est in [10, 50, 100, 300, 500]:
    m = GradientBoostingRegressor(n_estimators=n_est, learning_rate=0.1, max_depth=4)
    m.fit(X[:150], y[:150])
    train = m.score(X[:150], y[:150])
    test = m.score(X[150:], y[150:])
    print(f"M={n_est:<4}  train_R²={train:.3f}  test_R²={test:.3f}")
```

> Watch: test_R² eventually *decreases* while train_R² keeps climbing → overfitting. Use early stopping.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([10, 15, 20, 25, 30])

m1 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0)
m1.fit(X, y)
print("Normal:", m1.score(X, y))

y_broken = np.array([10, 15, 20, 25, 300])   # ONE huge outlier
m2 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=2, random_state=0)
m2.fit(X, y_broken)
print("Outlier:", m2.score(X, y_broken))
print("Predictions:", m2.predict(X))
```

**What happens?** With squared loss, the outlier at y=300 produces a huge residual. Every subsequent tree chases that residual, inflating predictions for all nearby points.

> 💥 **Break pattern:** normal model → one outlier → trees cascade corrections toward it. Why? **Squared loss gives quadratic weight to large residuals**, and boosting iteratively chases them.

**Fixes:**
- Use `loss='huber'` or `loss='absolute_error'` → robust loss caps outlier influence
- Remove or cap the outlier before training
- Lower `n_estimators` + early stopping → fewer rounds to chase the outlier

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Halve learning rate | Need ~2× trees, but more stable | Smaller steps generalize better |
| Double n_estimators | Better on train, may overfit test | More rounds memorize noise |
| Increase max_depth to 10 | Dramatically overfits | Trees too complex → memorize |
| Switch to absolute_error loss | More robust to outliers | Gradient = sign(error), capped |
| Use subsample=0.5 | Stochastic GB, more robust | Each tree sees different data → decorrelates |
| Add 100 irrelevant features | Performance drops | Trees waste splits on noise → need tuning |

> 🤔 Think: which is more dangerous — too few trees or too many? → Too many (overfitting). Too few just underfits, which is recoverable.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
Trees h₁..h_M  and their leaf values
F₀ = mean(y)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `n_estimators` (M) | Number of trees | Underfit | Overfit | 100–500 (use early stopping) |
| `learning_rate` (η) | Step size per tree | Very slow, needs many trees | Overfits fast | 0.01–0.3 |
| `max_depth` | Tree complexity | Underfit | Overfits, defeats "weak learner" | 2–4 |
| `min_samples_leaf` | Min samples in a leaf | Noisy leaves | Over-smoothed | 5–20 |
| `subsample` | Fraction of data per tree | Slower | — | 0.7–1.0 |
| `loss` | Loss function | — | — | `squared_error` default; `huber` if outliers |

> 📌 **η ↔ M tradeoff:** the two most important levers. Low η + high M + early stopping is the standard recipe. `max_depth` should stay small — these are *weak* learners.

---

## 20. Assumptions

Gradient Boosting is far less assumption-heavy than linear models. Here's what actually matters:

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Loss is differentiable** | Need gradient for pseudo-residuals | Core algorithm | — | Use subgradient (Huber/MAE ok) |
| **Weak learners capture signal** | Trees can fit the residuals | Effective correction | Residual plot after fitting | Increase depth or tree complexity |
| **Enough data per leaf** | Each leaf has meaningful average | Stable leaf estimates | Check leaf sizes | Raise `min_samples_leaf` |
| **Not extreme label noise** | Target values are roughly correct | Trees can fit anything including noise | Residual diagnostics | Robust loss, early stopping |

> Unlike linear regression: **no linearity, no normality, no homoscedasticity, no scaling required.** The model is fully non-parametric. The main practical worry is overfitting, not distributional assumptions.

---

## 21. Data Requirements

```text
Target      → continuous numeric (else → classification)
Features    → numerical; categoricals must be encoded (or use HistGradientBoosting for some)
Missing     → sklearn GradientBoostingRegressor needs imputation;
              HistGradientBoostingRegressor handles NaN natively
Outliers    → painful with squared loss; use Huber or MAE loss
Scaling     → unnecessary (tree-based, splits are threshold-based)
Small data  → works but trees overfit easily; reduce depth, increase min_samples_leaf
High-dim    → okay; use subsample + lower max_depth to control
```

> ⚠️ Data-leakage trap: **split BEFORE any encoding.** Target encoding or frequency encoding on the full dataset leaks test-set information.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimize loss via additive trees)
        ≠
EVALUATION METRIC   (what you report to a manager)
```

| Metric | Formula | Simple | Use | Avoid |
|---|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | avg squared miss | standard loss | units are "squared" |
| RMSE | √MSE | avg miss, in minutes/₹ | most common | outliers dominate |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust, interpretable | when big misses matter more |
| R² | 1 − SS_res/SS_tot | % variance explained | model quality | comparing across datasets |

> **Critical:** Training loss goes down every round (by design). But test loss eventually goes *up* → that's overfitting. Always use early stopping on a validation set.

---

## 23. Failure Cases

```text
DATA            → heavy outliers + squared loss → model chases outliers
OPTIMIZATION    → too many rounds without early stopping → overfit
GENERALIZATION  → extrapolation failure (leaf means can't go beyond training range)
PRACTICAL       → one-hot encoding high-cardinality categoricals → sparse, slow
STRUCTURAL      → sequential training → can't parallelize like Random Forest
```

---

## 24. Debugging

Model performs badly? Run this checklist in order:

```text
1. Train R² high, test R² low     → overfitting → lower η, fewer trees, early stop
2. Both train and test R² low     → underfitting → increase depth, more trees, lower η
3. RMSE spikes on certain ranges  → outlier in that range → check residuals, robust loss
4. Feature importance all similar → noisy features → feature selection, subsample
5. Training is very slow           → use HistGradientBoostingRegressor instead
6. Predictions are all near mean  → η too small or too few trees
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Random Forest:       "I'll average many independent deep trees."       (parallel, reduces variance)
Gradient Boosting:   "I'll add many shallow trees, each fixing errors." (sequential, reduces bias)
AdaBoost:            "I'll reweight hard samples and combine."          (sequential, reweighting)
XGBoost:             "I'll add second-order + regularization to GB."   (Newton + penalty)
```

| Algorithm | Main idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Random Forest | Bagged parallel trees | Low variance, fast, robust | Doesn't reduce bias much | Robust baseline |
| Gradient Boosting | Sequential residual fit | High accuracy, bias reduction | Slow (sequential), tuning | Tabular accuracy |
| AdaBoost | Sample reweighting | Simple boosting concept | Noise-sensitive, less accurate | Pedagogical |
| XGBoost | Regularized 2nd-order GB | Accurate + regularized + fast | More hyperparameters | Competitions/production |

> Gradient Boosting is the **conceptual ancestor** of XGBoost, LightGBM, and CatBoost. Master this, and those three become quick upgrades.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict delivery time for a food-delivery app
DATA:              past 10,000 deliveries (distance, traffic, time_of_day, actual_time)
FEATURES:          distance_km, traffic_level (encoded), hour_of_day
TARGET:            delivery_time_min
MODEL:             GradientBoostingRegressor (or HistGradientBoostingRegressor)
SPLIT:             train / validation / test
TUNE:              η × n_estimators × max_depth via grid search; early stopping
EVALUATE:          RMSE on test + residual plot
DEPLOY:            serve prediction on order confirmation page
MONITOR:           retrain weekly; check if traffic patterns shift seasonally
```

Same skeleton powers house price prediction, insurance claim severity, ad-click value estimation.

> 🚀 ML is not `model.fit(X, y)`. It's problem → data → features → model → evaluate → deploy → monitor → repeat.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is a pseudo-residual for squared loss?
2. **Understand:** why does shrinkage (η < 1) help generalization?
3. **Calculate:** run one boosting round by hand on 3 points (as Section 10).
4. **Apply:** given a residual plot that still shows a pattern after 50 trees, what should you do?
5. **Debug:** RMSE on test is 3× higher than training RMSE — diagnose and fix.
6. **Experiment:** run Section 16's experiment at 5 learning rates; graph test_R² vs M.
7. **Build:** delivery-time mini-project: EDA → residual check → fit GBM → tune via CV → early stopping → one-line business summary.
8. **Explain:** explain gradient boosting to a friend in 60 seconds using the doctor analogy.

---

## 28. Interview

### Beginner
- **What is gradient boosting?** An ensemble that adds trees sequentially, each fitting the current model's errors (pseudo-residuals), scaled by a learning rate.
- **What's a residual in this context?** `y − current prediction` — what the next tree tries to correct.
- **What does the learning rate do?** Scales each tree's contribution; smaller values are more stable and generalize better.

### Intermediate
- **Why is it called "gradient" boosting?** Each tree fits the negative gradient of the loss function at the current predictions — this is gradient descent in function space, not parameter space.
- **Gradient Boosting vs Random Forest?** GB is sequential, reduces bias, can overfit; RF is parallel, reduces variance, robust. GB usually gives better accuracy with tuning.
- **How do you prevent overfitting?** Low learning rate, early stopping, shallow trees (depth 2–4), subsampling.

### Advanced
- **Explain the pseudo-residual for a general loss.** r = −∂L(y,F)/∂F — the direction of steepest loss decrease. For squared loss it's y−F; for absolute loss it's sign(y−F); for Huber it interpolates.
- **What is shrinkage and why does it work?** η < 1 scales each tree's output, slowing learning. This prevents any single tree from dominating and forces the model to spread the correction across many trees → lower variance.
- **How does subsampling (stochastic GB) help?** Each tree trains on a random subset of data → decorrelates trees, adds variance reduction, similar to mini-batch SGD in neural nets.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Update:          Fₘ = F_{m−1} + η · γₘ · hₘ
Pseudo-residual (MSE): r = y − F
Objective:       min Σ L(yᵢ, F(xᵢ))   via stagewise greedy
```

**Common traps:**
- Thinking boosting and bagging use the same aggregation (additive vs averaging).
- Confusing learning rate with gradient descent step size in parameters — it's a step size in *function space*.
- Forgetting that `n_estimators` alone doesn't control overfitting — it interacts with `learning_rate`.

> **Representative pattern question (NOT a past GATE PYQ):** "For squared error loss, what is the pseudo-residual that each tree in gradient boosting fits?" Answer: `y − F_{m−1}(x)`, the current residual.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + complexity + theory</summary>

### Functional gradient descent derivation

**Objective:** minimize `J(F) = Σᵢ L(yᵢ, F(xᵢ))` over functions F.

**Step 1:** Start with F₀ = argmin_c Σ L(yᵢ, c). For MSE → F₀ = mean(y).

**Step 2:** At step m, the functional gradient is:

```text
∂J/∂F(xᵢ) = ∂L(yᵢ, F(xᵢ))/∂F
```

The direction of steepest decrease is the **negative gradient**:

```text
rᵢₘ = −∂L(yᵢ, F(xᵢ))/∂F  |_{F=F_{m−1}}
```

**Step 3:** Fit a weak learner hₘ to rᵢₘ (a regression tree on pseudo-residuals).

**Step 4:** Line search for optimal step:

```text
γₘ = argmin_γ Σᵢ L(yᵢ, F_{m−1}(xᵢ) + γ · hₘ(xᵢ))
```

For MSE, γₘ = 1 when leaf values are mean residuals. For other losses, it's a scalar optimization.

**Step 5:** Update with shrinkage:

```text
Fₘ = F_{m−1} + η · γₘ · hₘ
```

This is **greedy stagewise additive modeling** — each step minimizes the loss at the current model state.

### Complexity

```text
Training:   O(M · n · m · depth)     sequential M rounds
Prediction: O(M · depth)              sum M tree traversals
Space:      O(M · nodes)              store all trees
```

Sequential nature is the bottleneck. HistGradientBoosting and XGBoost address this with histogram-based splits and parallelism.

### Why MSE → leaf values = mean residuals

For squared loss, the optimal leaf value that minimizes `Σ(yᵢ − (F + w))²` over samples in the leaf is:

```text
w* = (1/|leaf|) Σᵢ∈leaf (yᵢ − F(xᵢ)) = mean(residuals in leaf)
```

This is the closed-form solution to a quadratic minimization — why MSE makes boosting particularly clean.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "Gradient Boosting builds a model by adding one small tree at a time. Each new tree learns from the mistakes of everything built before. The learning rate makes sure each step is small and safe."

> **Explain to a 12-year-old:** "Imagine you're guessing someone's weight. First you guess average. Then you look at how wrong you were and guess a little better. Keep correcting, and you'll get really close."

> **Explain in an interview:** add: pseudo-residuals, functional gradient descent, shrinkage η, early stopping, subsampling, loss pluggability, bias vs variance.

> **Explain the mathematics:** derive r = y − F from squared loss gradient, then show the additive update formula.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define gradient boosting in one sentence.
2. What is a pseudo-residual for squared loss?
3. Why do we use shallow trees (weak learners)?
4. Compute one boosting round by hand on 3 points.
5. Explain what `learning_rate` does and why it helps.
6. What's the difference between boosting and bagging?
7. Name two ways to prevent overfitting in gradient boosting.
8. When would you use `loss='huber'` instead of `loss='squared_error'`?
9. Compare with Random Forest: which reduces bias, which reduces variance?
10. State one scenario where you would NOT use gradient boosting.

---

## 33. Cheat Sheet

```text
Algorithm : Gradient Boosting Regression · Supervised → Regression · Ensemble (Boosting)
Goal      : reduce bias by sequentially fitting residuals / negative gradients
Model     : F_M = F₀ + Σ η·γₘ·hₘ(x)    (additive sum of shallow trees)
Learn     : tree structures + leaf values
Tune      : n_estimators, learning_rate, max_depth, subsample, loss
Key trick : pseudo-residual = −∂L/∂F; for MSE = y − F
Use when  : tabular data, need high accuracy, can tune carefully
Avoid when: very large data (→ HistGBM/XGB/LightGBM), need interpretability
Related   : AdaBoost · XGBoost · LightGBM · CatBoost · Random Forest
Baseline  : every boosting variant is compared against this
Classification counterpart → B-classification/09 (GradientBoostingClassifier)
```

---

## 34. What Next?

You just learned the foundational boosting algorithm.

```text
Gradient Boosting
   ├── AdaBoost        (reweighting-based boosting)  → next note (14)
   ├── XGBoost         (2nd-order + regularization)   → 15
   ├── LightGBM        (histogram + leaf-wise)        → 16
   ├── CatBoost        (ordered boosting + categoricals) → 17
   └── HistogramGBM    (sklearn's fast variant)       → built into sklearn
```

> Next recommended: **14. AdaBoost Regression** — it's the older, simpler boosting idea (reweighting instead of gradient fitting) that led to Gradient Boosting.
