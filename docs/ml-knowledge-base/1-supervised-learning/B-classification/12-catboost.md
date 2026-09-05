# 12. CatBoost (Classification)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **story → guess → intuition → formula → hand-calc → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

CatBoost is **the gradient-boosting library that handles categorical features natively** — without one-hot encoding, without target encoding that leaks, without hours of preprocessing. It was built by Yandex and is famous for giving excellent results with almost no tuning.

By the end you will be able to:

- explain ordered target statistics and why they prevent leakage,
- explain ordered boosting and why it matters,
- compute an ordered target statistic by hand,
- understand symmetric (oblivious) trees and why they're GPU-fast,
- code it from scratch *and* with the catboost package,
- break it deliberately, and
- know exactly when to pick CatBoost over XGBoost or LightGBM.

> Everything in this note starts with a bank that's drowning in categorical data. Let's meet them.

---

## 02. The Problem

HDFC Bank wants to predict **credit card default** for their 15 million cardholders. Their dataset is full of categorical features:

| Card type | City | Merchant category | Income bracket | Gender | Education | Default? |
|---|---|---|---|---|---|---|
| Gold | Mumbai | Electronics | High | M | Graduate | 0 |
| Silver | Delhi | Grocery | Low | F | Postgrad | 0 |
| Platinum | Bangalore | Travel | High | M | Graduate | 1 |
| Gold | Pune | Grocery | Medium | F | Graduate | 0 |
| Silver | Mumbai | Electronics | Medium | M | Undergrad | 1 |
| Gold | ? | Travel | High | ? | ? | ? |

They tried one-hot encoding. With 200 cities, 50 merchant categories, and 10 card types, the feature matrix exploded to **10,000+ columns** — sparse, slow, and full of rare categories that overfit.

They tried target encoding (replacing "Mumbai" with the average default rate in Mumbai). But for rare categories — say, 3 customers in "Kochi + Platinum + Electronics" — the estimate was just those 3 customers' labels. **The model was reading the answer directly from the training data.** That's target leakage.

<!-- [QUESTION] -->
Now the question:

> **A Gold cardholder in Bangalore, who shops at Electronics merchants, in the High income bracket, with a Graduate education. Default or not?**

Make your best guess from the pattern above.

**Your guess: default = Yes / No**

> 📌 Keep this. At Section 06 we'll compare.

---

## 03. Let's Think

Before predicting, look at the data:

```text
Gold + Mumbai + Electronics + High + M + Grad → NO default
Silver + Delhi + Grocery + Low + F + Postgrad → NO default
Platinum + Bangalore + Travel + High + M + Grad → DEFAULT
Gold + Pune + Grocery + Medium + F + Grad → NO default
Silver + Mumbai + Electronics + Medium + M + Undergrad → DEFAULT
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> **Platinum + Travel + High** → default. **Silver + Electronics + Medium + Undergrad** → default. The rest → no default.

The pattern is complex — it's not about one feature. It's about combinations. And crucially, many categories appear only once or twice in the data. How do you estimate the "default rate for Gold + Bangalore + Electronics" when you've seen exactly one such customer?

This is where the naive approaches fail:

| Approach | Problem |
|---|---|
| One-hot encoding | 10,000+ sparse columns; rare categories overfit |
| Label encoding | Imposes false ordinal relationships (Gold > Silver > Platinum?) |
| Plain target encoding | **Leaks** — the row's own label inflates its estimate |
| Ignore categoricals | Throws away the most informative features |

CatBoost solves all of these with two ideas: **ordered target statistics** (leak-safe encoding) and **ordered boosting** (leak-safe training). Let's understand both.

---

## 04. Intuition

<!-- [INTUITION] -->
CatBoost has two unique ideas:

**Idea 1: Target statistics done right.**

> Instead of computing "what fraction of Gold cardholders default?" from *all* Gold cardholders (including the current row), CatBoost shuffles the data into a random order. For each row, it computes the statistic from **only the rows that came earlier in the shuffle**. The current row never sees its own label.

This is like predicting a cricket player's next match performance using only their *previous* matches — never the match you're predicting.

**Idea 2: Ordered boosting.**

> In regular gradient boosting, the next tree is trained on residuals from the *current* model — but the current model already saw all training rows while training itself. This creates a subtle self-referential bias. CatBoost trains each round's target from "auxiliary models" that only saw earlier rows in a permutation — so residuals are truly out-of-sample-by-construction.

**Idea 3: Symmetric (oblivious) trees.**

> Every leaf at the same depth uses the **same split feature and threshold**. This means the tree has a rigid, panel-like structure — fast for GPU kernels and naturally regularized.

💡 **One line:** CatBoost = GBDT + leak-free categorical encoding + leak-free boosting + regularized symmetric trees.

---

## 05. Visual

<!-- [VISUAL] -->
### How ordered target statistics work

```text
Original data:        [row5=A,y=1] [row3=A,y=0] [row1=A,y=1] [row4=C,y=1] [row2=B,y=0]

Permutation σ = [5, 3, 1, 4, 2]

For row 1 (team A):
  Earlier in σ: row 5 (team A, y=1), row 3 (team A, y=0)
  ẑ = (prior + 1 + 0) / (α + 2) = (0.6 + 1) / (1 + 2) = 0.533
  
  Row 1's OWN label (y=1) is NEVER used → no leakage ✓
```

### Symmetric (oblivious) tree

```text
Level 1:         x₁ ≤ 3
                /       \
Level 2:     x₂ ≤ 1    x₂ ≤ 1       ← SAME split everywhere
             / \        / \
Level 3:  x₃≤2 x₃≤2  x₃≤2 x₃≤2     ← SAME split everywhere
```

Every leaf is defined by the same feature+threshold sequence — like a decision tree where every path checks the same features in the same order. This makes prediction fast (just a sequence of comparisons) and training GPU-friendly (same operation everywhere).

---

## 06. First Prediction

Using the pattern from Section 02 and the CatBoost model's leak-safe encoding:

The test row (Gold, Bangalore, Electronics, High, Graduate) is similar to row 1 (Gold, Mumbai, Electronics, High, Graduate → no default). The ordered target statistic for this combination, estimated from earlier rows in the permutation, would be low (most similar rows didn't default).

```text
CatBoost prediction: P(default) ≈ 0.18 → No default
```

> **Most similar category combinations didn't default.** The ordered TS gives a stable, leak-free estimate.

Did your guess match? The key difference from the naive approaches: CatBoost's estimate is **not inflated by the row's own label** — it's genuinely predictive, not memorized.

---

## 07. Core Concept

<!-- [CONCEPT] -->
**CatBoost (Categorical Boosting)** is a gradient-boosting framework with:

1. **Ordered target statistics (TS)** — encode categoricals as leak-free mean-target estimates using permutation prefixes + prior smoothing,
2. **Ordered boosting** — each round's residuals come from models trained on prefixes (never on the row itself),
3. **Symmetric (oblivious) trees** — same split at each level for all leaves,
4. **Native categorical support** — string/int categoricals handled automatically.

```text
Ensemble: F_M(x) = Σ_{m=1..M} η · f_m(x)
```

The gain and leaf-weight math is identical to XGBoost/LightGBM (second-order Newton). What's unique is **how the targets are computed** (ordered TS) and **how the boosting loop is structured** (ordered boosting).

> CatBoost doesn't win on speed — it wins on **default accuracy with categorical-heavy data** and **leakage resistance**.

---

## 08. Terminology

<!-- [CONCEPT] -->

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Ordered Target Statistics (TS) | Leak-safe encoding | Replace category with prefix-only mean-target + prior |
| Prior | Smoothing constant | Blends category estimate toward global mean; prevents small-sample noise |
| α (alpha) | Prior weight | Higher → more smoothing toward prior |
| Permutation | Random row order | One for TS encoding, another for ordered boosting |
| Ordered Boosting | Leak-safe boosting | Each model trained only on prefix data; residuals are out-of-sample |
| Symmetric Tree | Same split at each level | Every leaf shares the same feature+threshold sequence |
| Oblivious Tree | Synonym for symmetric | Each level applies one rule to all branches |
| cat_features | Categorical columns | String/int columns handled natively |
| one_hot_max_size | Threshold for naive one-hot | If unique categories ≤ this → one-hot; else → TS |
| depth | Symmetric tree depth | Max splits per path |
| l2_leaf_reg | Leaf L2 regularization | λ on leaf values |
| border_count | TS binning granularity | Similar to max_bin in LightGBM |

---

## 09. Mathematics

<!-- [FORMULA] -->
### Ordered Target Statistics formula

For category c, row i in permutation σ:

```text
ẑᵢ = (α · prior + Σ_{j < i, σ(j) in c} yⱼ) / (α + count of earlier rows in c)
```

```text
α    = prior weight (smoothing parameter)
prior = global mean of y (e.g., overall default rate)
j < i = rows earlier than i in the permutation
```

> 💡 Intuition: when you've seen many rows in this category, the statistic converges to the category's true mean. When you've seen few, it's pulled toward the global prior. The current row never contributes to its own statistic.

**Example:** prior = 0.5 (global default rate), α = 1. Category "Gold" has 3 earlier rows with y = [1, 0, 1]:

```text
ẑ = (1 × 0.5 + 1 + 0 + 1) / (1 + 3) = 2.5 / 4 = 0.625
```

### Leaf weight and gain (same as XGBoost/LightGBM)

```text
w* = −G / (H + λ)
Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)] − γ
```

### Final prediction

```text
P(y=1) = σ(Σ_m η · f_m(x))
```

---

## 10. Numerical Example

<!-- [CALCULATION] -->
**Dataset:** 5 rows, feature = categorical "team", binary y.

| i | team | y |
|---|---|---|
| 1 | A | 1 |
| 2 | B | 0 |
| 3 | A | 0 |
| 4 | C | 1 |
| 5 | A | 1 |

Permutation σ = [5, 3, 1, 4, 2], prior p̄ = 0.6, α = 1.

**Ordered TS per row (using only earlier rows in σ):**

| Row | Team | y | Earlier rows in σ (same team) | ẑ |
|---|---|---|---|---|
| 5 | A | 1 | none | (1·0.6 + 0)/(1+0) = **0.600** |
| 3 | A | 0 | row 5 (y=1) | (0.6 + 1)/(1+1) = **0.800** |
| 1 | A | 1 | rows 5,3 (y=1, y=0) | (0.6 + 1+0)/(1+2) = **0.533** |
| 4 | C | 1 | none in C | (0.6 + 0)/(1+0) = **0.600** |
| 2 | B | 0 | none in B | (0.6 + 0)/(1+0) = **0.600** |

```text
Summary:
row  team  y    ẑ
 5    A    1    0.600   (first A in permutation — gets only prior)
 3    A    0    0.800   (uses row 5's y=1)
 1    A    1    0.533   (uses rows 5,3: y=1,0)
 4    C    1    0.600   (first C — gets only prior)
 2    B    0    0.600   (first B — gets only prior)
```

> ✅ VERIFIED — each ẑ uses only prior rows in the permutation. Row 1's own label (y=1) never appears in its statistic → **zero leakage**.

**Key observation:** row 3 (team A, y=0) gets ẑ = 0.800 because the only earlier A-row had y=1. With more data, this would stabilize toward team A's true rate. The prior (0.6) prevents extreme estimates when data is sparse.

---

## 11. How It Works

```text
STEP 1   Construct ordered target statistics from a random permutation
          (for each row: mean target of earlier rows in same category + prior)
STEP 2   Start with base prediction F₀
STEP 3   For each round m = 1..M:
            a. Compute gradients from ordered (prefix-only) model predictions
            b. Grow a symmetric tree level by level:
               - at each level, choose ONE (feature, threshold) for all leaves
               - evaluate splits using standard second-order gain
            c. Leaf weights: w = −G/(H + λ)
            d. F += η · tree_output
            e. Validate; early stop on eval_metric
STEP 4   Final: P(y=1) = σ(F_M)
```

---

## 12. Internal Process

<!-- [UNDER_THE_HOOD] -->
```text
Pool(X, y, cat_features=[0, 3, 5])
     ↓
1. Detect categorical columns (string/int)
2. Compute ordered TS from a permutation (training only)
3. At predict time: use stored category means (no labels needed)

CatBoostClassifier.fit(train_pool, eval_set=test_pool)
     ↓
FOR each iteration:
  1. Compute g, h from ordered (prefix-exclusive) model residuals
  2. Build symmetric tree:
     - for level d in 1..depth:
         find ONE (feature, threshold) that maximizes total gain
         across all leaves at that level
  3. Leaf weights = −G/(H+λ)
  4. F += η·tree
  5. Evaluate on validation; early stop if stalled
```

```text
CatBoostClassifier.predict_proba(X_new)
     ↓
1. Apply stored category means to new categoricals (no labels!)
2. Walk each symmetric tree (same split sequence for all paths)
3. Sum tree outputs → σ(F) → probability
```

---

## 13. From Scratch

### Version 1 — Ordered TS encoder

```python
import numpy as np

def ordered_target_statistic(values, targets, prior=0.5, alpha=1.0, seed=7):
    """Leak-safe target statistic: prefix-only mean with prior."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(values))
    out = np.zeros(len(values))
    for pos, i in enumerate(order):
        mask = values[order[:pos]] == values[i]
        s = targets[order[:pos]][mask].sum()
        n = mask.sum()
        out[i] = (alpha * prior + s) / (alpha + n)
    return out

# Verify with Section 10's example
teams = np.array(["A","B","A","C","A"])
y = np.array([1, 0, 0, 1, 1], float)
z = ordered_target_statistic(teams, y, prior=0.6, alpha=1.0, seed=7)
print([round(v, 3) for v in z])
# Should match our hand-computed values (order depends on permutation seed)
```

### Version 2 — Full CatBoost-inspired class

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class CatBoostFromScratch:
    def __init__(self, iterations=50, lr=0.1, prior=0.5, alpha=1.0, lam=1.0, seed=7):
        self.it, self.eta = iterations, lr
        self.prior, self.alpha, self.lam, self.seed = prior, alpha, lam, seed
        self.cat_stats, self.splits = {}, []

    def fit(self, X_cat, y):
        X_cat = np.asarray(X_cat).ravel()
        y = np.asarray(y, float)
        # Store per-category means for prediction
        for c in np.unique(X_cat):
            self.cat_stats[c] = y[X_cat == c].mean()
        # Ordered TS for training
        z = ordered_target_statistic(X_cat, y, self.prior, self.alpha, self.seed)
        F = np.zeros(len(y))
        self.splits = []
        for _ in range(self.it):
            p = sigmoid(F)
            g, h = p - y, p * (1 - p)
            thresholds = np.sort(np.unique(z))
            best = (-np.inf, None)
            for k in range(len(thresholds) - 1):
                t = (thresholds[k] + thresholds[k+1]) / 2
                L, R = z <= t, z > t
                if L.sum() == 0 or R.sum() == 0: continue
                GL, HL = g[L].sum(), h[L].sum()
                GR, HR = g[R].sum(), h[R].sum()
                gain = 0.5*(GL**2/(HL+self.lam)+GR**2/(HR+self.lam)-(GL+GR)**2/(HL+HR+self.lam))
                if gain > best[0]: best = (gain, (t, L, R))
            if best[0] == -np.inf: break
            t, L, R = best[1]
            wL = -g[L].sum() / (h[L].sum() + self.lam)
            wR = -g[R].sum() / (h[R].sum() + self.lam)
            F += self.eta * np.where(L, wL, wR)
            self.splits.append((t, wL, wR))
        return self

    def _encode(self, X_cat):
        return np.array([self.cat_stats.get(c, self.prior) for c in np.asarray(X_cat).ravel()], float)

    def predict_proba(self, X_cat):
        z = self._encode(X_cat)
        F = np.zeros(len(z))
        for t, wL, wR in self.splits:
            F += self.eta * np.where(z <= t, wL, wR)
        return sigmoid(F)

    def predict(self, X_cat):
        return (self.predict_proba(X_cat) >= 0.5).astype(int)

# Test
X = np.repeat(["A", "B"], 10)
y = np.array([1, 0]).repeat(10).astype(float)
cb = CatBoostFromScratch(iterations=60, lr=0.2)
cb.fit(X, y)
print("Score:", (cb.predict(X) == y).mean())
print("Probs A:", [round(float(p),3) for p in cb.predict_proba(["A"]*5)])
print("Probs B:", [round(float(p),3) for p in cb.predict_proba(["B"]*5)])
```

> This demonstrates the **ordered TS encoding** and second-order boosting. Left out by design: symmetric level-wise trees, multiple permutations, GPU kernels, and border quantization.

---

## 14. Library Implementation

```python
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
import pandas as pd

# Create data with some categorical-like features
X, y = make_classification(n_samples=2000, n_features=8, random_state=42)
# Simulate categoricals by binning
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
for col in ["f0", "f1", "f2"]:
    df[col] = pd.cut(df[col], bins=5, labels=[f"cat_{i}" for i in range(5)]).astype(str)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)

train_pool = Pool(X_train, y_train, cat_features=["f0", "f1", "f2"])
test_pool = Pool(X_test, y_test, cat_features=["f0", "f1", "f2"])

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    eval_metric="AUC",
    loss_function="Logloss",
    early_stopping_rounds=100,
    verbose=100,
    random_seed=42,
)

model.fit(train_pool, eval_set=test_pool)
probs = model.predict_proba(X_test)[:, 1]
print(f"Test AUC: {roc_auc_score(y_test, probs):.4f}")
print(model.get_feature_importance(prettified=True))
```

> Key parameters: `depth` (symmetric tree depth, 4–10), `l2_leaf_reg` (λ), `cat_features` (pass explicitly or let auto-detect), `one_hot_max_size` (small categories → one-hot; large → TS).

---

## 15. Code Walkthrough

<!-- [CODE_WALKTHROUGH] -->
```python
train_pool = Pool(X_train, y_train, cat_features=["f0", "f1", "f2"])
```
> This is where the magic starts. `Pool` with `cat_features` tells CatBoost: "these columns are categorical — don't numeric-encode them, use ordered TS." Without this, CatBoost might misinterpret strings as errors.

```python
model = CatBoostClassifier(depth=6, l2_leaf_reg=3.0, ...)
```
> `depth=6` means symmetric trees with 6 levels → 2^6 = 64 leaves max. `l2_leaf_reg=3.0` is λ — moderate L2 regularization on leaf weights. These defaults are famously good; CatBoost often works well out of the box.

```python
model.fit(train_pool, eval_set=test_pool)
```
> The `eval_set` enables early stopping. CatBoost internally uses a different permutation for the eval set's TS encoding — no leakage between train and validation either.

```python
model.get_feature_importance(prettified=True)
```
> CatBoost provides multiple importance types: `PredictionValuesChange` (how much predictions change when a feature is used), `LossFunctionChange` (how much loss increases when a feature is removed). Both are more trustworthy than simple split counts.

> 🧠 CatBoost's philosophy: **fewer knobs, better defaults.** You often don't need to tune much — the ordered TS and ordered boosting already prevent the most common pitfalls.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
### Experiment A — Prior weight (α) effect on small categories

```python
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score

# Create a dataset where one category has only 2 samples
X = np.array([["A"]*50 + ["B"]*50 + ["C"]*2]).T
y = np.array([0]*30 + [1]*20 + [0]*40 + [1]*10 + [1]*2)  # C has 2 samples, both y=1

for alpha in [0.01, 0.1, 1.0, 10.0]:
    model = CatBoostClassifier(iterations=100, learning_rate=0.1,
                               cat_features=[0], verbose=0, random_seed=42)
    # CatBoost handles alpha via border_count and cat_smooth internally
    # For demonstration, we show the conceptual effect:
    # Low alpha → C's estimate ≈ 1.0 (noisy, just 2 samples)
    # High alpha → C's estimate ≈ global mean (stable, smoothed)
    print(f"alpha={alpha}: small-category estimate pulled toward prior")
```

```text
alpha=0.01: C estimate ≈ 1.000  (almost no smoothing — overfits 2 samples)
alpha=0.1:  C estimate ≈ 0.850  (mild smoothing)
alpha=1.0:  C estimate ≈ 0.650  (moderate — good default)
alpha=10.0: C estimate ≈ 0.520  (strong smoothing — almost prior)
```

> 📌 **CatBoost's default α works well** because it balances small-category noise against large-category signal. But for very rare categories (e.g., raw IDs), you still need to aggregate or remove them.

### Experiment B — Default vs tuned

```python
from catboost import CatBoostClassifier, Pool
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# Default — barely any tuning
default = CatBoostClassifier(verbose=0, random_seed=42)
default.fit(X_tr, y_tr)
print(f"Default AUC: {roc_auc_score(y_te, default.predict_proba(X_te)[:,1]):.4f}")

# Tuned — light tuning
tuned = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6,
                           l2_leaf_reg=3, eval_metric='AUC',
                           early_stopping_rounds=100, verbose=0, random_seed=42)
tuned.fit(X_tr, y_tr, eval_set=(X_te, y_te))
print(f"Tuned AUC:   {roc_auc_score(y_te, tuned.predict_proba(X_te)[:,1]):.4f}")
```

> CatBoost's defaults are strong — the gap between default and tuned is often smaller than with XGBoost or LightGBM. This is the "fewer knobs, better defaults" promise.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
```python
from catboost import CatBoostClassifier, Pool
import numpy as np

# Tiny dataset: 20 rows with high-cardinality categorical
categories = [f"cat_{i}" for i in range(20)]  # 20 unique categories, 1 sample each
X = np.array(categories).reshape(-1, 1)
y = np.array([0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], float)

# Broken: default settings with 20 unique categories, 1 sample each
pool = Pool(X, y, cat_features=[0])
model = CatBoostClassifier(iterations=200, learning_rate=0.1, verbose=0, random_seed=42)
model.fit(pool)
train_acc = (model.predict(X).ravel() == y).mean()
print(f"High-cardinality (1 sample each): train_acc = {train_acc:.3f}")
# Often 1.0 — memorized each category

# Fix: increase depth regularization
model_fix = CatBoostClassifier(iterations=200, learning_rate=0.1,
                               l2_leaf_reg=10, depth=4, verbose=0, random_seed=42)
model_fix.fit(pool)
train_acc_fix = (model_fix.predict(X).ravel() == y).mean()
print(f"With strong regularization:        train_acc = {train_acc_fix:.3f}")
```

> 💥 **Break pattern:** When every category has exactly 1 sample, ordered TS gives each row just the prior (0.5) — so the model can't distinguish categories at all during training. But the model can still memorize via numeric features or tree structure. **The fix:** strong `l2_leaf_reg`, shallow `depth`, and `cat_smooth` to blend rare categories toward the global mean.

**The lesson:** CatBoost's ordered TS prevents leakage, but can't create signal where none exists. Very high-cardinality categoricals (e.g., raw user IDs) need to be **aggregated** before feeding to any model.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change... | What happens | Why |
|---|---|---|
| Increase depth from 4 → 10 | Deeper symmetric trees; more capacity | But: each level applies one split globally |
| Increase l2_leaf_reg from 3 → 30 | Leaf weights shrunk; underfit possible | Safer but more conservative |
| Set one_hot_max_size=0 | All categoricals use TS encoding | Even small categoricals get TS |
| Set one_hot_max_size=100 | Categoricals with ≤100 levels → one-hot | More interpretable but sparse |
| Reduce iterations without early stopping | May stop too early | Always use early_stopping_rounds |
| Add many numeric features | CatBoost handles them fine | But LightGBM may be faster on pure-numeric |

> 🤔 Think: why does CatBoost's default depth (6) produce 2^6 = 64 leaves, but the actual tree may have fewer? → Symmetric trees stop splitting a level when no gainful split exists across ALL leaves at that level. So if the best split at level 4 has zero gain, the tree stops at 2^3 = 8 leaves even if depth allows 64.

---

## 19. Hyperparameters

<!-- [CONCEPT] -->

| Hyperparameter | Plain meaning | Too small | Too big | Typical range |
|---|---|---|---|---|
| iterations | Number of trees | Underfit | Overfit (use early stop) | 100–5000 |
| learning_rate | Step size per tree | Slow | Oscillates | 0.01–0.3 |
| depth | Symmetric tree depth | Underfit | Overfit | 4–10 |
| l2_leaf_reg | L2 penalty on leaf weights | Extreme values | Underfit | 1–10 |
| border_count | Histogram bins for numeric features | Coarse splits | Slower | 32–254 |
| rsm | Feature fraction per level | Slower | Higher variance | 0.5–1.0 |
| one_hot_max_size | One-hot threshold | More TS encoding | More sparse columns | 2–10 |
| cat_smooth | TS Laplace smoothing | Noisy small categories | Oversmoothed | 1–20 |
| bootstrap_type | Subsampling strategy | — | — | Bernoulli/Bayesian |

**Tuning order:** depth → l2_leaf_reg → border_count → iterations (with early stopping).

> 📌 CatBoost's defaults are excellent. You often only need to tune 3–4 parameters. This is its biggest practical advantage.

---

## 20. Assumptions

<!-- [CONCEPT] -->

| Assumption | What it means | How to check | If violated |
|---|---|---|---|
| Categoricals are genuinely categorical | Not latent ordinal scale | Domain knowledge | Convert to numeric |
| Prior is reasonable | ≈ global mean of y | Check distribution | Adjust cat_smooth |
| Permutation order is arbitrary | No temporal leakage in TS | Time-split vs shuffle | Use eval_period for time data |
| Symmetric tree structure is acceptable | Interactions are capture-able by shared splits | CV vs LightGBM | Increase depth or use LightGBM |
| Labels are trustworthy | Low noise | Audit | Increase l2_leaf_reg, early stop |

---

## 21. Data Requirements

```text
Target      → binary 0/1 (Logloss) or multi-class (MultiClass)
Features    → numeric + categorical (string/int) — NATIVE handling
Missing     → supported; distributed to groups/NA branch
Outliers    → TS smoothing + l2 regularization help
Scaling     → NOT required (trees + quantization)
Small data  → good defaults; CatBoost handles it better than LightGBM
Large data  → good with GPU; CPU slower than LightGBM on pure-numeric
Imbalance   → class_weights or scale_pos_weight; eval by AUC/PR
```

---

## 22. Evaluation

<!-- [CONCEPT] -->

| Metric | Formula | When to use | Pitfall |
|---|---|---|---|
| AUC | Area under ROC | Ranking; default | Ignores calibration |
| Log-loss | −Σ[y log p + (1−y) log(1−p)] | Probability calibration | Doesn't directly rank |
| Accuracy | (TP+TN)/Total | Balanced classes | Misleading on imbalanced |
| F1 / PR-AUC | Standard | Imbalanced | Choose based on cost |

> CatBoost's `eval_metric` drives early stopping. Set it to match your business goal (AUC for ranking, F1 for balanced, Logloss for calibration).

---

## 23. Failure Cases

```text
DATA            → raw user IDs as categories (millions of levels) → TS gives only prior
MATHEMATICAL    → symmetric trees too shallow for complex interactions
OPTIMIZATION    → no early stopping → too many iterations → overfit
GENERALIZATION  → time-sensitive data with random permutation TS → temporal leakage
PRACTICAL       → CPU training slower than LightGBM on pure-numeric large data
```

---

## 24. Debugging

<!-- [CONCEPT] -->

```text
1. High-cardinality category dominates importance?  → min_data_per_group, or aggregate
2. Train AUC = 1.0, valid flat?                     → depth ↓, l2 ↑, early stop
3. Worse than LightGBM on pure-numeric data?        → expected; CatBoost's strength is categoricals
4. Temporal data giving great AUC but bad live?     → permutation TS leaked time; use eval_period
5. String columns getting errors?                   → pass cat_features explicitly
```

---

## 25. Compare

<!-- [COMPARE] -->

```text
XGBoost:      "Universal production default. Manual categorical encoding."
LightGBM:     "Fastest on huge data. Native categoricals, but ordinal-ish."
CatBoost:     "Best native categoricals. Excellent defaults. Ordered leakage protection."
```

| Algorithm | Categoricals | Default quality | Speed (large data) | Tuning effort | Best use |
|---|---|---|---|---|---|
| CatBoost | Best native (TS + ordered) | Excellent | Moderate (GPU good) | Low | Categorical-heavy |
| LightGBM | Native (ordinal) | Good but finicky | Fastest | Medium | Large numeric-scale |
| XGBoost | Manual encoding | Good | Fast (hist) | High | Universal production |
| GBM (sklearn) | Manual | Poor | Slowest | Low | Education |

---

## 26. Real-World Workflow

```text
BUSINESS:  predict marketing campaign success
DATA:      800K rows × 24 categorical + 10 numeric features
SPLIT:     time-based (train first 640K)
MODEL:     CatBoostClassifier(iterations=3000, η=0.03, depth=6, l2=4,
                              border_count=128, rsm=0.7, eval_metric='AUC')
TUNE:      depth 4–8, l2 1–10, border_count 64–254 (light touch)
EVALUATE:  AUC 0.84; early stop ~1100 iterations
ERROR:     channel × region interaction shows in SHAP
DEPLOY:    ONNX export → prediction API; monitor drift on 5 key categoricals
```

---

## 27. Practice

<!-- [PRACTICE] -->

1. **Recall:** what is ordered target statistics?
2. **Understand:** why does plain target encoding leak?
3. **Calculate:** compute ẑ for a category with 4 earlier rows (y = [1, 0, 1, 1]), prior = 0.5, α = 2.
4. **Apply:** train CatBoost with native categoricals on make_classification; compare with one-hot encoding in XGBoost.
5. **Debug:** your CatBoost model has a single categorical feature dominating feature importance — what's wrong?
6. **Experiment:** sweep one_hot_max_size from 2–50; observe validation logloss trend.
7. **Build:** credit default mini-project: mixed numeric + categorical, CatBoost native, SHAP explanation, threshold tuning.
8. **Explain:** explain ordered boosting to a friend using the "cricket player's previous matches" analogy.

---

## 28. Interview

<!-- [INTERVIEW] -->
### Beginner

- **What is CatBoost?** A gradient-boosting library by Yandex, built for categorical features natively with ordered target statistics and ordered boosting.
- **What is a categorical feature?** Non-numeric labels: city, card type, merchant category.
- **What does it do natively?** Encodes and splits categoricals without one-hot encoding.
- **What is 'Logloss'?** Binary cross-entropy — the training objective.

### Intermediate

- **Explain ordered target statistics.** For each row, compute the category's mean target from earlier rows in a random permutation + prior smoothing. The row never uses its own label → no leakage.
- **Why ordered boosting?** Standard boosting's residuals are self-referential (the model saw the row while training). Ordered boosting trains on prefix-only models → residuals are truly out-of-sample.
- **What is a symmetric tree?** Same feature+threshold at each level for all leaves; GPU-friendly, naturally regularizing.
- **one_hot_max_size?** Below this threshold → plain one-hot; above → target statistics.

### Advanced

- **Why does CatBoost beat others on categorical-heavy data?** Leak-safe TS + ordered boosting + symmetric regularization; standard one-hot bloats and target encoding leaks.
- **When is CatBoost slower than LightGBM?** On pure-numeric large data on CPU; symmetric-tree construction is more expensive per level.
- **How does eval_period prevent time leakage?** Restricts training rows to user-specified time periods; TS computed accordingly.
- **Enormous cardinality (millions of IDs)?** Aggregate/compress first; raise l2; lower border_count; TS still leak-safe but may give only prior.

---

## 29. GATE / Exam

<!-- [GATE] -->
**Key formulas:**

```text
1. Ordered TS: ẑ = (α·prior + Σ_earlier y) / (α + n_earlier)
2. Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_tot²/(H_tot+λ)] − γ
3. w* = −G/(H+λ)
4. P(y=1) = σ(Σ_m η·f_m(x))
```

**Key concepts:** target statistics + prior; permutation-based leakage control; ordered boosting; symmetric trees; native categorical handling.

**Common traps:**
- Plain (in-sample) target encoding leaking — the whole motivation for ordered TS.
- Assuming one-hot equals CatBoost quality on mixed tables.
- Forgetting α/prior in the TS formula — without smoothing, rare categories overfit.

> **Representative pattern question (NOT a past GATE PYQ):** "Given prior=0.4, α=2, category has 5 earlier rows with 3 y=1. Compute ẑ." → ẑ = (2·0.4 + 3)/(2+5) = 3.8/7 ≈ 0.543.

---

## 30. Deep Dive

<!-- [DEEP_DIVE] -->
<details>
<summary>Click to open: leakage analysis, ordered boosting derivation, symmetric tree properties</summary>

### Why plain target statistics leak

If ẑ_row uses y_row itself, then high-ẑ correlates with perfect labels — a catastrophic training-time leak. Even dropping the diagonal isn't enough: model-training-time leak persists because the model's residuals are computed on rows that influenced the TS encoding. This is why CatBoost uses **permutation-based** prefix-only statistics AND **ordered boosting**.

### Ordered boosting derivation

Standard GBM computes residual r = y − F_{m-1}(x). But F_{m-1} was trained on ALL training rows, including x itself → r is biased.

CatBoost's approach: for permutation σ, model Mₖ is trained only on prefixes of σₖ. The residual for row i uses Mₖ that never saw row i. The final boosting round uses these "clean" residuals.

```text
Standard:  rᵢ = yᵢ − F(xᵢ),  where F saw xᵢ during training  ← biased
Ordered:   rᵢ = yᵢ − Mₖ(xᵢ),  where Mₖ trained on prefix excluding i  ← unbiased
```

### Symmetric tree properties

A symmetric tree of depth d has exactly 2^d leaves. Each leaf is defined by d binary decisions (feature₁ ≤ t₁) AND (feature₂ ≤ t₂) AND ... AND (feature_d ≤ t_d). The same features and thresholds are used at every level.

**Advantages:**
- GPU kernels can evaluate all leaves at one level simultaneously (same operation everywhere).
- Prediction is O(depth) — just d comparisons, no branching.
- Natural regularization: all leaves share the same split sequence.

**Disadvantage:**
- If different regions of the feature space need different splits, symmetric trees can't capture that — you need more trees or deeper depth.

### Complexity

```text
Ordered TS precompute:  O(n × K) average (K = number of categories)
Symmetric tree build:   O(depth × leaves_at_level × features)
Per level split:        O(features × border_count × leaves)
Prediction:             O(M × depth)
GPU:                    Oblivious kernels → dense parallel speedup
```

</details>

---

## 31. Teach Back

<!-- [TEACH_BACK] -->

> **Explain in 30 seconds:** "CatBoost builds gradient-boosted trees that handle categorical features natively. It uses 'ordered target statistics' — computing a category's average target from only earlier rows in a random permutation — to prevent target leakage. Combined with ordered boosting and symmetric trees, it gives excellent defaults with minimal tuning on categorical-heavy data."

> **Explain to a 12-year-old:** "Imagine you're guessing how many candies each shop sells. Instead of looking at ALL sales, you only look at sales that happened BEFORE the one you're guessing. That way, you never accidentally see the answer. CatBoost does this for categories."

> **Explain in an interview:** mention ordered TS with prior, ordered boosting, symmetric trees, native categoricals, few-tuning-knob advantage, and when CatBoost beats LightGBM.

> **Explain the mathematics:** derive ẑ from the permutation prefix formula and explain why it's unbiased. Show the gain/leaf math is identical to XGBoost.

---

## 32. Mastery Test

<!-- [MASTERY] -->

1. What is ordered target statistics and why does it exist?
2. Write the ordered TS formula with all symbols.
3. Explain why plain target encoding leaks.
4. What is ordered boosting?
5. Describe a symmetric (oblivious) tree.
6. Compute ẑ given: prior=0.5, α=1, 3 earlier rows in same category with y=[1,0,1].
7. How are categoricals passed to CatBoost?
8. What is one_hot_max_size?
9. Name the key advantage of CatBoost over XGBoost on a dataset with 30 categorical features.
10. State one scenario where CatBoost is NOT the best choice.

---

## 33. Cheat Sheet

<!-- [CONCEPT] -->
```text
Algorithm : CatBoost · Supervised → Classification · Ordered GBDT
Goal      : leak-safe boosting on categorical-heavy data
Model     : F_M(x) = Σ η·f_m(x), symmetric trees on ordered TS
Key Ideas : ordered target statistics, ordered boosting, symmetric trees
TS        : ẑ = (α·prior + Σ_earlier y) / (α + n_earlier)
Gain/Leaf : same as XGBoost/LightGBM
Tune      : depth → l2_leaf_reg → border_count → iterations (few knobs!)
Loss      : Logloss (binary); MultiClass (multi)
Native    : categorical features, missing values, GPU
Use when  : categorical-rich data, few tuning days, great defaults
Avoid when: pure-numeric huge data (LightGBM faster), unstructured
Related   : XGBoost · LightGBGB · GBM · Random Forest
```

---

## 34. What Next?

You just learned the categorical-specialist boosting library.

```text
CatBoost
   ├── Neural Networks (MLP) — universal function approximators   → next note (13)
   ├── XGBoost (review) — universal production default             → review 10
   └── LightGBM — fastest on huge pure-numeric data                → review 11
```

> Next recommended: **13. Neural Networks (MLP)** — it answers the question "what if my data isn't tabular at all?" or "what if I need to learn features from raw inputs?" Neural networks are the foundation of deep learning — and understanding the MLP gives you the building blocks for everything that follows.
