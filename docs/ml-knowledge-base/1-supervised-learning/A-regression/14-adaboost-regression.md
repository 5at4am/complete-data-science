# 14. AdaBoost Regression

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐☆☆
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **problem → reweight → median → formula → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

AdaBoost is the **original boosting algorithm** — the one that started the entire field. It introduced the core idea that every boosting method since has built on: *focus on what's still hard*.

By the end you will be able to:

- explain how reweighting steers later learners toward hard samples,
- compute one AdaBoost reweighting round by hand,
- understand why regression uses a weighted median (not a mean),
- code it from scratch and with sklearn,
- and know exactly when to use it — and when to pick Gradient Boosting instead.

> Everything in this note builds on one idea: *reweight the hard ones*.

---

## 02. The Problem

Riya is an interior designer in Mumbai. She needs to **estimate monthly rent** for apartments based on area (sq ft) and distance from the nearest metro (km). She has data from 6 recent rentals:

| Area (sqft) | Distance to metro (km) | Rent (₹ thousands) |
|---|---|---|
| 400 | 0.5 | 28 |
| 600 | 2.0 | 22 |
| 800 | 0.3 | 35 |
| 500 | 3.0 | 18 |
| 700 | 1.5 | 26 |
| 350 | 4.0 | 14 |

She builds a single decision stump (depth-1 tree). It does okay for most apartments but **consistently under-predicts rent for small, metro-near apartments** (sample 1) and **over-predicts for large, far-from-metro ones** (sample 4).

<!-- [QUESTION] -->
Here's the question:

> **The first stump is worst on samples 1 and 4. Can you give those two samples more importance so the next stump pays extra attention to them?**

Think about how you'd redistribute "importance" before reading on.

---

## 03. Let's Think

Let's look at where the stump was wrong:

```text
Sample    Area   Distance    Actual    Stump    Error
  1       400      0.5        28        22      +6  (under)
  2       600      2.0        22        23      −1
  3       800      0.3        35        33      +2
  4       500      3.0        18        24      −6  (over)
  5       700      1.5        26        27      −1
  6       350      4.0        14        15      −1
```

Samples 1 and 4 have the biggest errors (±6). If we could **tell the next learner "these two matter more"**, it would focus its limited capacity on fixing the hardest cases.

<!-- [THINK_ABOUT_IT] -->
🤔 But how do we actually change "importance"?

> Each sample gets a **weight**. Initially all equal (1/6 each). After the first stump's errors, samples 1 and 4 get *higher* weights. The next stump trains with these weights — it *must* do better on high-weight samples or its weighted error will be terrible.

---

## 04. Intuition

💡 **The idea in one line:**

> AdaBoost trains weak learners one at a time. After each round, it **increases the weights of samples that were hard to predict** and **decreases the weights of easy ones**, so the next learner focuses where it matters.

Think of a cricket team reviewing match footage. After a loss, the coach doesn't watch every ball equally — they **replay the overs where most wickets fell**. The batsmen who got out practice those specific deliveries more. Each practice session targets the weaknesses.

For regression, the final prediction combines all learners using a **weighted median** (not a mean) — making the ensemble robust to extreme predictions from any single weak learner.

---

## 05. Visual

```text
AdaBoost reweighting flow:

   All samples equal weight (1/n)
        ↓ fit learner 1
   some samples wrong → errors found
        ↓ reweight: hard ↑, easy ↓
   Learner 2 focuses on hard samples
        ↓ reweight again
   Learner 3 focuses on remaining hard ones
        ↓
   Combine all via weighted median
```

<!-- [VISUAL] -->
```text
Weighted median vs mean (with outlier prediction):

predictions:  22,  25,  100    (weights: 0.4, 0.4, 0.2)
mean = (22×0.4 + 25×0.4 + 100×0.2) = 46.8    ← pulled by outlier
weighted median = 25                              ← robust center
```

The weighted median is resistant to any single learner making a wild prediction — that's why AdaBoost uses it for regression.

---

## 06. First Prediction

Starting with equal weights D₁ = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6], the first stump predicts roughly:

```text
Small metro-near:  ~22    (actual 28)  → under by 6
Large far:         ~24    (actual 18)  → over by 6
Others:            ~26    (close to actual)
```

The weighted error of stump 1 is about L₁ ≈ 0.4 (40% of weighted predictions are notably wrong). This means the stump is "okay but not great" — and AdaBoost will assign it a moderate trust weight.

After reweighting, samples 1 and 4 gain importance. The second stump, trained on the new weights, will try harder on those two.

> 📌 After combining both stumps via weighted median, the prediction for sample 1 should improve from 22 toward 28.

---

## 07. Core Concept

**AdaBoost Regression** (Drucker, 1997) — an ensemble method that:

1. Starts with equal sample weights D₁,
2. Fits a weak learner to the weighted data,
3. Computes a **weighted error** Lₜ and converts it to a **trust weight** αₜ,
4. **Updates sample weights**: samples with high relative error keep/gain weight, easy ones lose weight,
5. Combines all learners via a **weighted median** weighted by αₜ.

```text
Final prediction: F(x) = weighted_median of {hₜ(x)} with weights {αₜ}
```

| Ingredient | What it does |
|---|---|
| Sample reweighting | Directs later learners to hard cases |
| Trust weights (αₜ) | More accurate learners get more say in the final answer |
| Weighted median | Robust combination — not swayed by outliers |

---

## 08. Terminology

Each term *emerges* from the story:

### Sample weight Dₜ

> Simple: how much each sample "counts" at round t.
> Technical: a probability distribution over training samples, updated each round.

### Relative error eᵢ

> Simple: how wrong the learner was on sample i, normalized.
> Technical: `|yᵢ − hₜ(xᵢ)| / max_j |yⱼ − hₜ(xⱼ)|`, scaled to [0, 1].

### Confidence βₜ

> Simple: "how reliable is this learner?" Small β = reliable.
> Technical: βₜ = Lₜ / (1 − Lₜ), where Lₜ is the weighted loss.

### Trust weight αₜ

> Simple: how much we trust this learner's vote.
> Technical: αₜ = ln(1/βₜ) — large for accurate learners.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Dₜ(i) | weight of sample i at round t | probability distribution over samples |
| eᵢ | relative error of sample i | normalized error in [0, 1] |
| Lₜ | weighted average error | Σ Dₜ(i) · eᵢ |
| βₜ | learner reliability | Lₜ/(1−Lₜ) |
| αₜ | trust in learner t | ln(1/βₜ) |

> ⚠️ Common mistake: "αₜ = accuracy." No — αₜ is derived from the weighted loss Lₜ. A learner with Lₜ = 0.1 gets αₜ = ln(1/0.111) ≈ 2.2. A learner with Lₜ = 0.4 gets αₜ = ln(1/0.667) ≈ 0.405 — much less trust.

---

## 09. Mathematics (gradual)

We build the math in four steps.

### Step M1 — Initialize weights

```text
D₁(i) = 1/n   for all i
```

Every sample starts equally important.

### Step M2 — Fit and compute error

Fit weak learner hₜ to weighted data. Compute per-sample **relative error**:

```text
eᵢ = |yᵢ − hₜ(xᵢ)| / max_j |yⱼ − hₜ(xⱼ)|
```

Then the **weighted loss**:

```text
Lₜ = Σᵢ Dₜ(i) · eᵢ
```

### Step M3 — Convert loss to trust

```text
βₜ = Lₜ / (1 − Lₜ)     (valid when Lₜ < 0.5)
αₜ = ln(1 / βₜ)
```

When Lₜ is small (good learner), βₜ is small and αₜ is large → high trust. When Lₜ ≈ 0.5 (barely better than chance), αₜ ≈ 0 → no trust.

### Step M4 — Update weights

```text
D_{t+1}(i) = Dₜ(i) · βₜ^(1 − eᵢ)     then normalize to sum to 1
```

- **High eᵢ** (hard sample): exponent `(1 − eᵢ)` is small → weight barely shrinks (relatively grows)
- **Low eᵢ** (easy sample): exponent is large → weight shrinks a lot

The net effect: hard samples dominate the next round.

---

## 10. Numerical Example

Data: x = [1, 3, 5], y = [10, 22, 40]. Use 2 stumps (T=2), linear loss.

<!-- [CALCULATION] -->

**Step 1 — Initialize D₁ = [1/3, 1/3, 1/3]**

**Step 2 — Round 1:** Stump h₁ predicts roughly: x=1→15, x=3→25, x=5→35.

```text
Errors: |10−15|=5, |22−25|=3, |40−35|=5
Max error = 5
eᵢ = [5/5, 3/5, 5/5] = [1.0, 0.6, 1.0]
L₁ = (1/3)(1.0) + (1/3)(0.6) + (1/3)(1.0) = 0.867
```

Since L₁ > 0.5, this learner is weak — let's use a better stump. Suppose instead h₁ predicts: x=1→12, x=3→12, x=5→35 (split at x≤4).

```text
Errors: |10−12|=2, |22−12|=10, |40−35|=5
Max error = 10
eᵢ = [0.2, 1.0, 0.5]
L₁ = (1/3)(0.2) + (1/3)(1.0) + (1/3)(0.5) = 0.567
```

Still > 0.5. Let's use an even better split: h₁ predicts x=1→15, x=3→28, x=5→35.

```text
Errors: |10−15|=5, |22−28|=6, |40−35|=5
Max error = 6
eᵢ = [0.833, 1.0, 0.833]
L₁ = (1/3)(0.833 + 1.0 + 0.833) = 0.889
```

OK, let's take a case that works. Suppose stump 1 gives good predictions on samples 2 and 3, worse on 1:

h₁: x=1→18, x=3→22, x=5→38.

```text
Errors: |10−18|=8, |22−22|=0, |40−38|=2
Max error = 8
eᵢ = [1.0, 0.0, 0.25]
L₁ = (1/3)(1.0 + 0.0 + 0.25) = 0.417
```

**Step 3 — Trust:**

```text
β₁ = 0.417 / (1 − 0.417) = 0.417 / 0.583 = 0.715
α₁ = ln(1/0.715) = 0.336
```

**Step 4 — Update weights:** D₂(i) = D₁(i) · β₁^(1 − eᵢ)

```text
sample1 (e=1.0): (1/3) · 0.715^0.0 = 0.333    (unchanged — hard)
sample2 (e=0.0): (1/3) · 0.715^1.0 = 0.238    (shrunk — easy)
sample3 (e=0.25): (1/3) · 0.715^0.75 = 0.258   (slightly shrunk)
Normalize: total = 0.829 → D₂ = [0.402, 0.287, 0.311]
```

Sample 1 (hardest) now has the highest weight → stump 2 will focus on it.

**Step 5 — Round 2** fits stump h₂ paying special attention to sample 1 (weight 0.402), improving its prediction.

**Step 6 — Final prediction:** weighted median of h₁, h₂ by αₜ.

> ✅ VERIFIED — hand-computed. Demonstrates reweighting: hard samples gain weight, steering later learners.

---

## 11. How It Works

```text
STEP 1   Initialize equal weights D₁ = 1/n
STEP 2   For t = 1 to T:
            a. Fit weak learner hₜ with sample weights Dₜ
            b. Compute relative errors eᵢ and weighted loss Lₜ
            c. Compute trust: βₜ = Lₜ/(1−Lₜ), αₜ = ln(1/βₜ)
            d. Update weights: D_{t+1}(i) = Dₜ(i)·βₜ^(1−eᵢ), normalize
STEP 3   Final prediction: weighted median of all hₜ by αₜ
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] ]
This is the section that makes sklearn **unmagical**.

```text
model.fit(X, y)
     ↓
1. D₁ = [1/n, 1/n, ..., 1/n]
     ↓
2. For t = 1..n_estimators:
     a. hₜ = DecisionTreeRegressor(max_depth=1).fit(X, y, sample_weight=Dₜ)
     b. pred = hₜ.predict(X)
     c. eᵢ = |yᵢ − predᵢ| / max|yⱼ − predⱼ|
     d. Lₜ = Σ Dₜ(i) · eᵢ
     e. βₜ = Lₜ/(1−Lₜ); αₜ = ln(1/βₜ)
     f. Dₜ₊₁(i) = Dₜ(i) · βₜ^(1−eᵢ); normalize
     ↓
3. Store [h₁..h_T] and [α₁..α_T]
```

```text
model.predict(X_new)
     ↓
for each hₜ:
    collect hₜ(X_new) with weight αₜ
return weighted median of predictions
```

> The model stores T weak learners and their trust weights. Prediction is a weighted median — robust to any single bad learner.

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def adaboost_fit(X, y, n_estimators=10):
    X, y = np.asarray(X, float), np.asarray(y, float)
    n = len(y)
    D = np.full(n, 1.0 / n)
    learners, alphas = [], []
    for _ in range(n_estimators):
        stump = DecisionTreeRegressor(max_depth=1)
        stump.fit(X, y, sample_weight=D)
        pred = stump.predict(X)
        err = np.abs(y - pred)
        max_err = err.max() if err.max() > 0 else 1.0
        e = err / max_err
        L = np.dot(D, e)
        if L >= 0.5:
            break
        beta = L / (1 - L)
        alpha = np.log(1.0 / beta)
        learners.append(stump)
        alphas.append(alpha)
        D = D * np.power(beta, 1.0 - e)
        D /= D.sum()
    return learners, alphas

def weighted_median predictions, alphas):
    order = np.argsort(predictions)
    cum = np.cumsum(np.array(alphas)[order])
    mid = cum[-1] / 2
    idx = np.searchsorted(cum, mid)
    return predictions[order[idx]]

def adaboost_predict(X, learners, alphas):
    X = np.asarray(X, float)
    preds = np.array([t.predict(X) for t in learners])
    return np.array([weighted_median(preds[:, i], alphas)
                     for i in range(len(X))])
```

### Version 2 — vectorized weighted median

```python
def weighted_median_vec(preds, alphas):
    """preds: (n_learners, n_samples), alphas: (n_learners,)"""
    order = np.argsort(preds, axis=0)
    a_sorted = np.array(alphas)[order]
    cum = np.cumsum(a_sorted, axis=0)
    mid = sum(alphas) / 2
    out = np.zeros(preds.shape[1])
    for i in range(preds.shape[1]):
        idx = np.searchsorted(cum[:, i], mid)
        out[i] = preds[order[idx, i], i]
    return out
```

### Version 3 — clean class

```python
class AdaBoostRegressor:
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.learners, self.alphas = [], []

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(y)
        D = np.full(n, 1.0/n)
        for _ in range(self.n_estimators):
            stump = DecisionTreeRegressor(max_depth=1)
            stump.fit(X, y, sample_weight=D)
            pred = stump.predict(X)
            e = np.abs(y - pred)
            max_e = max(e.max(), 1e-10)
            e_norm = e / max_e
            L = np.dot(D, e_norm)
            if L >= 0.5: break
            beta = L / (1 - L)
            alpha = np.log(1.0 / beta)
            self.learners.append(stump)
            self.alphas.append(alpha)
            D *= np.power(beta, 1 - e_norm)
            D /= D.sum()
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        preds = np.array([t.predict(X) for t in self.learners])
        return weighted_median_vec(preds, self.alphas)
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

X = np.sort(np.random.RandomState(0).rand(300, 1), axis=0)
y = np.sin(6*X).ravel() + np.random.RandomState(0).randn(300)*0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    loss='linear',
    random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
```

> sklearn's `AdaBoostRegressor` uses the Drucker algorithm internally. The `loss` parameter controls how relative error is computed: `'linear'` (default), `'square'`, or `'exponential'`.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
stump.fit(X, y, sample_weight=D)
```
> Fits a weak learner honoring sample weights. sklearn's `DecisionTreeRegressor` accepts `sample_weight` — the splitter criterion weights each sample by D. High-weight samples dominate the split quality score.

```python
e = err / max_err
```
> Relative error per sample, normalized by the worst error. This makes the error scale-invariant — a sample with absolute error 5 in a dataset where worst error is 10 gets e=0.5.

```python
D = D * np.power(beta, 1.0 - e)
```
> The weight update: `D_{t+1}(i) ∝ Dₜ(i) · βₜ^{(1−eᵢ)}`. High e → small exponent → weight barely shrinks. Low e → large exponent → weight shrinks.

```python
alpha = np.log(1.0 / beta)
```
> Trust weight: larger α means more vote in the final prediction. Derived from the loss: better learners get more trust.

> 🧠 Every line maps to a formula from Section 09. The code *is* the math.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] >
> If this note is rendered inside the interactive platform, these become sliders. Otherwise, run them in Python and observe.

### Experiment A — how weights shift

```python
import numpy as np

# After one round with L=0.417
L = 0.417
beta = L / (1 - L)
e = np.array([1.0, 0.0, 0.25])  # relative errors
D_old = np.array([1/3, 1/3, 1/3])
D_new = D_old * np.power(beta, 1 - e)
D_new /= D_new.sum()
print(f"beta={beta:.3f}")
print(f"Old weights: {D_old.round(3)}")
print(f"New weights: {D_new.round(3)}")
```

```text
beta=0.715
Old weights: [0.333 0.333 0.333]
New weights: [0.402 0.287 0.311]
```

> 📌 The hardest sample (e=1.0) gained the most weight. The easiest (e=0.0) lost the most. This is the core reweighting dynamic.

### Experiment B — n_estimators and noise sensitivity

```python
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

rng = np.random.RandomState(42)
X = rng.rand(100, 1) * 10
y = 2*X.ravel() + rng.randn(100) * 0.5
y_noisy = y.copy()
y_noisy[0] = 500  # one extreme outlier

for n_est in [10, 50, 200]:
    m = AdaBoostRegressor(n_estimators=n_est, random_state=0)
    m.fit(X, y_noisy)
    pred = m.predict(X[:5])
    print(f"n_est={n_est:<4} predictions for first 5: {pred.round(1)}")
```

> Watch: more estimators → AdaBoost obsesses over the outlier (sample with y=500) → predictions shift toward it. This is the noise-sensitivity problem.

---

## 17. Break the Model

<!-- [BREAK_IT] ]
Code:

```python
import numpy as np
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([10, 15, 20, 25, 30])

m1 = AdaBoostRegressor(n_estimators=50, random_state=0)
m1.fit(X, y)
print("Normal R²:", m1.score(X, y))

y_noisy = np.array([10, 15, 20, 25, 300])   # ONE extreme outlier
m2 = AdaBoostRegressor(n_estimators=50, random_state=0)
m2.fit(X, y_noisy)
print("Noisy predictions:", m2.predict(X).round(1))
print("Weights on outlier:", m2.estimator_weights_[:5])
```

**What happens?** AdaBoost reweights the outlier sample to extremely high weight. Every subsequent stump tries to fit it. The ensemble becomes dominated by trying to predict that one crazy value.

> 💥 **Break pattern:** normal model → one outlier → weights explode → model pivots to fit the outlier. Why? **AdaBoost's reweighting is aggressive on hard samples — and outliers are the hardest samples of all.**

**Fixes:**
- Use `loss='linear'` (default) instead of `'exponential'` — less aggressive reweighting
- Cap outliers before training
- Limit n_estimators + use validation
- Use Gradient Boosting with Huber loss instead (more robust)

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Use a deep tree (depth=5) as base | Boosting degenerates — no room for improvement | Strong learners don't benefit from reweighting |
| Double n_estimators | May overfit on noisy data | More rounds → more obsession over hard samples |
| Set loss='exponential' | More aggressive reweighting | Exponential error amplifies hard samples |
| Flip one label | AdaBoost obsesses over it | Reweighting amplifies noise |
| Use mean instead of median | Outliers swing the prediction | Mean is not robust; median is |

> 🤔 Think: why can't we use a strong learner (depth=10) as the base? → Because if each learner is already accurate, there's little residual error to reweight. Boosting's power comes from combining *weak* learners.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
Weak learners h₁..h_T
Trust weights α₁..α_T
Final weight distribution D
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `n_estimators` | Number of weak learners | Underfit | Overfit (especially with noise) | 50–200 |
| `learning_rate` | Scales αₜ | Slower convergence | More aggressive | 0.5–1.5 |
| `loss` | How error is measured | — | `'exponential'` = more aggressive | `'linear'` (default) |
| `estimator` | Weak learner type | Too weak → no progress | Too strong → defeats boosting | `DecisionTreeRegressor(max_depth=1)` |

> 📌 **Base learner must be weak.** depth=1 (stump) is the classic choice. depth=2 is sometimes better. Anything deeper risks overfitting and defeats the boosting philosophy.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Weak learners are better than random** | Each hₜ has some signal | Boosting needs progress each round | Lₜ < 0.5 always | Improve base learner |
| **Not extreme label noise** | Targets are roughly correct | AdaBoost amplifies hard samples — noise looks "hard" | Residual diagnostics | Use robust loss, cap outliers |
| **Lₜ < 0.5 each round** | Learner is better than chance | Algorithm requires this for βₜ to be valid | Monitor Lₜ | Early stop |
| **Enough data for stumps** | Each split has meaningful statistics | Stumps are low-capacity but need data | Check leaf sizes | Collect more data |

> AdaBoost is **assumption-light** (no linearity, no scaling, no normality). Its main sensitivity is to **label noise** — mislabeled samples get reweighted aggressively, hijacking the ensemble.

---

## 21. Data Requirements

```text
Target      → continuous numeric (else → classification)
Features    → numerical; categoricals must be encoded
Missing     → must be imputed before training (stumps can't handle NaN)
Outliers    → dangerous! AdaBoost reweights them aggressively
Scaling     → unnecessary (tree-based)
Small data  → works (stumps are low-capacity), but noise sensitivity increases
Label noise → the #1 risk — check carefully
```

> ⚠️ AdaBoost is more sensitive to label noise than Gradient Boosting or XGBoost, because its reweighting mechanism is *designed* to chase hard samples — and noisy labels are the hardest.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimize weighted loss)
        ≠
EVALUATION METRIC   (what you report)
```

| Metric | Formula | Simple | Use |
|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | avg squared miss | standard loss |
| RMSE | √MSE | avg miss in ₹ | most common |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust, interpretable |
| R² | 1 − SS_res/SS_tot | % variance explained | model quality |

> **Watch for:** training error going to zero while test error climbs → overfitting via too many rounds on noisy data.

---

## 23. Failure Cases

```text
DATA            → noisy labels → reweighting amplifies noise
ALGORITHMIC     → Lₜ ≥ 0.5 → learner not better than chance → stops early
GENERALIZATION  → extrapolation fails (tree-based)
PRACTICAL       → often outperformed by Gradient Boosting / XGBoost in accuracy
STRUCTURAL      → sequential → can't parallelize
```

---

## 24. Debugging

Model performs badly? Run this checklist in order:

```text
1. Training R² high, test low    → overfitting → fewer rounds, check for label noise
2. Both train and test low       → base learner too weak or data insufficient
3. One prediction wildly off     → outlier reweighted heavily → check data
4. Lₜ ≥ 0.5 early               → learner can't improve → change base estimator
5. RMSE much higher than RF      → expected; AdaBoost is less accurate in practice
6. Algorithm stops early         → loss ≥ 0.5 threshold hit → tune base estimator
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
AdaBoost:            "I'll reweight hard samples so later learners focus there."
Gradient Boosting:   "I'll fit the negative gradient of the loss at each step."
XGBoost:             "I'll do gradient boosting with regularization + second-order."
Random Forest:       "I'll average many independent deep trees (no reweighting)."
```

| Algorithm | Main idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| AdaBoost | Sample reweighting | Simple, canonical, robust median | Noise-sensitive, less accurate | Pedagogical, historical |
| Gradient Boosting | Gradient fitting | High accuracy, loss-flexible | More complex | Production |
| XGBoost | 2nd-order + reg | Accurate, fast, regularized | Many hyperparameters | Competitions |
| Random Forest | Parallel bagging | Robust, parallel | Higher bias | Baseline |

> AdaBoost is the **conceptual ancestor** of Gradient Boosting. Understanding its reweighting idea makes gradient boosting's "fit to the gradient" feel like a natural upgrade.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  estimate monthly apartment rent
DATA:              past rentals (area, distance, rent)
FEATURES:          area_sqft, distance_metro_km (encoded)
TARGET:            rent_₹_thousands
MODEL:             AdaBoostRegressor with DecisionTreeRegressor(max_depth=1)
SPLIT:             train / validation / test
TUNE:              n_estimators × learning_rate via grid search
EVALUATE:          RMSE on test + check weight distribution
DEPLOY:            serve prediction on rental listing page
MONITOR:           check if weights concentrate on few samples → overfitting
```

> 🚀 AdaBoost's main value today is as a **teaching algorithm** — understanding it is prerequisite for understanding Gradient Boosting and its descendants.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is a sample weight in AdaBoost?
2. **Understand:** why does AdaBoost use a weighted median instead of a weighted mean?
3. **Calculate:** compute one reweighting round by hand (as Section 10).
4. **Apply:** given weighted errors, compute βₜ and αₜ.
5. **Debug:** RMSE spikes after adding more estimators — what's happening?
6. **Experiment:** run Section 16's noise experiment; compare n_estimators=10 vs 200.
7. **Build:** rent prediction mini-project: EDA → fit AdaBoost → tune → compare with GradientBoostingRegressor.
8. **Explain:** explain AdaBoost's reweighting to a friend in 60 seconds using the cricket analogy.

---

## 28. Interview

### Beginner
- **What is AdaBoost regression?** Sequentially trains weak learners, reweighting samples so hard cases get more attention, combining via weighted median.
- **What's a weak learner?** A simple model slightly better than random — typically a depth-1 decision stump.
- **How does the weighted median work?** Order predictions, accumulate αₜ weights, pick the prediction at the 50% cumulative weight mark.

### Intermediate
- **Why reweight hard samples?** So the next learner focuses on the team's weak spots → adaptive improvement each round.
- **What is βₜ and αₜ?** βₜ = L/(1−L) measures learner reliability; αₜ = ln(1/βₜ) is the trust weight used in combining.
- **Why is AdaBoost sensitive to noise?** It repeatedly up-weights mislabeled or outlier samples, overfitting to them.

### Advanced
- **Explain Drucker's AdaBoost.R2 algorithm.** Normalize errors by max, compute weighted loss, map to β, reweight D ~ β^(1−e), combine via weighted median.
- **Why not use a mean for combination?** Weighted mean is pulled by extremes; weighted median is robust — a single learner making a wild prediction doesn't dominate.
- **How does AdaBoost relate to gradient boosting?** Both are boosting; AdaBoost reweights samples, gradient boosting fits gradients. AdaBoost can be seen as a special case of gradient boosting with exponential loss.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
βₜ = Lₜ / (1 − Lₜ)
αₜ = ln(1 / βₜ)
D_{t+1}(i) = Dₜ(i) · βₜ^{(1 − eᵢ)}     (normalized)
Final = weighted median of {hₜ} by {αₜ}
```

**Common traps:**
- AdaBoost **classification** uses weighted *majority vote*; **regression** uses weighted *median* — don't mix up.
- Forgetting that αₜ is log-scaled — not linear.
- Confusing AdaBoost reweighting (sample weights) with gradient boosting's residual fitting (gradient direction).

> **Representative pattern question (NOT a past GATE PYQ):** "Why can't you use a strong base learner in AdaBoost?" Answer: boosting gains come from combining weak learners; if each learner already fits well, reweighting doesn't help — there's no room for adaptive improvement.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + complexity + theory</summary>

### Drucker's AdaBoost.R2 derivation

**Step 1:** Initialize D₁(i) = 1/n.

**Step 2:** At round t, fit hₜ to weighted data. Compute per-sample relative error:

```text
eᵢ = |yᵢ − hₜ(xᵢ)| / max_j |yⱼ − hₜ(xⱼ)|
```

The max normalization makes eᵢ ∈ [0, 1] and ensures the loss Lₜ stays bounded.

**Step 3:** Weighted loss:

```text
Lₜ = Σᵢ Dₜ(i) · eᵢ
```

**Step 4:** Confidence and trust:

```text
βₜ = Lₜ / (1 − Lₜ)     ← valid when Lₜ < 0.5
αₜ = ln(1/βₜ)
```

When Lₜ < 0.5, the learner is better than random → βₜ < 1 → αₜ > 0 (positive trust).

**Step 5:** Weight update:

```text
D_{t+1}(i) ∝ Dₜ(i) · βₜ^{(1 − eᵢ)}
```

**Why this formula?** For high eᵢ (hard), the exponent (1−eᵢ) ≈ 0 → βₜ^0 = 1 → weight unchanged. For low eᵢ (easy), exponent ≈ 1 → weight multiplied by βₜ < 1 → shrunk.

**Step 6:** Final prediction is the weighted median:

```text
F(x) = argmin_p Σₜ αₜ · |hₜ(x) − p|
```

This is equivalent to the weighted median — the value minimizing the weighted sum of absolute deviations.

### Complexity

```text
Training:   O(T · C_stump)     where C_stump = O(n·m·log n)
Prediction: O(T)               median over T learners
Space:      O(T)               store T weak learners + αₜ
```

T sequential rounds; each stump is cheap. Overall fast but less accurate than gradient boosting.

### Connection to exponential loss (classification)

In AdaBoost classification, the weight update uses exp(−αₜ·yᵢ·hₜ(xᵢ)), which minimizes the exponential loss. The regression variant (Drucker) adapts this with the linear/square/exponential relative error and median combination.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "AdaBoost trains simple models one at a time. After each round, it makes the hard-to-predict samples more important. The next model focuses on those. Finally, combine all models, trusting the better ones more."

> **Explain to a 12-year-old:** "You're playing a game where you guess prices. First round, you get some wrong. Second round, you practice hardest on the ones you missed. Third round, practice the still-hard ones. Keep combining what you learned."

> **Explain in an interview:** add: weighted median for regression, Drucker's algorithm, βₜ/αₜ formulas, noise sensitivity, comparison with gradient boosting.

> **Explain the mathematics:** derive βₜ from Lₜ, show the weight update formula, explain why the weighted median minimizes weighted absolute deviations.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define AdaBoost regression in one sentence.
2. What is the weight update formula?
3. Why weighted median instead of weighted mean?
4. Compute βₜ and αₜ given Lₜ = 0.3.
5. Why must the base learner be weak?
6. What happens when Lₜ ≥ 0.5?
7. Name the three loss options for relative error.
8. Why is AdaBoost sensitive to noisy labels?
9. Compare with Gradient Boosting: which is more accurate in practice?
10. State one scenario where you WOULD use AdaBoost over Gradient Boosting.

---

## 33. Cheat Sheet

```text
Algorithm : AdaBoost Regression (Drucker) · Supervised → Regression · Ensemble (Boosting)
Goal      : focus learners on hard samples via reweighting
Model     : weighted median of weak learners, weighted by trust αₜ
Learn     : weak learners hₜ + trust weights αₜ
Tune      : n_estimators, learning_rate, loss, base estimator
Key trick : D_{t+1} ∝ Dₜ · β^{(1−e)}; α = ln(1/β)
Use when  : pedagogical understanding, moderate clean data, baseline boosting
Avoid when: noisy labels, state-of-the-art accuracy needed
Related   : Gradient Boosting · XGBoost · LightGBM · CatBoost · Random Forest
Classification counterpart → B-classification/06 (AdaBoostClassifier)
```

---

## 34. What Next?

You just learned the original boosting algorithm.

```text
AdaBoost
   ├── Gradient Boosting (fit gradients instead of reweighting)  → next note (13 revisited)
   ├── XGBoost         (2nd-order + regularization)              → 15
   ├── LightGBM        (histogram + leaf-wise)                   → 16
   └── CatBoost        (ordered boosting + categoricals)         → 17
```

> Next recommended: **15. XGBoost** — the most important practical upgrade: regularized gradient boosting with second-order optimization. If you understood AdaBoost's reweighting, XGBoost's gradient fitting will feel like a natural generalization.
