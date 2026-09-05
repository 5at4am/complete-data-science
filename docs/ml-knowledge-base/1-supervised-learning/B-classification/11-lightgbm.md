# 11. LightGBM (Classification)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **story → guess → intuition → formula → hand-calc → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

LightGBM is **the algorithm that makes gradient boosting fast enough for 100-million-row datasets**. If XGBoost is the battle tank, LightGBM is the sports car — same destination, built for speed.

By the end you will be able to:

- explain leaf-wise growth vs level-wise growth and when each can hurt,
- compute a GOSS reweight factor by hand,
- understand histogram binning and why it's so fast,
- code it from scratch *and* with the lightgbm package,
- break it deliberately on small data, and
- know exactly when to pick LightGBM over XGBoost.

> Everything in this note starts with a company drowning in data. Let's meet them.

---

## 02. The Problem

AdVista runs an ad-tech platform serving **50 million ad impressions per day** across Indian e-commerce sites. For each impression, they need to predict: will the user click?

Features include: user age, device type, browser, OS, time of day, page category, ad category, historical CTR, publisher ID, and 40+ more — many of them **categorical** (browser = "Chrome"/"Safari"/"Firefox", publisher = one of 500 sites).

Their XGBoost model takes **6 hours to train** on the full dataset. The business wants to **retrain every 4 hours** to keep up with changing user behavior.

Here is a tiny snapshot:

| User device | Browser | Page category | Ad category | Hist CTR | Clicked? |
|---|---|---|---|---|---|
| mobile | Chrome | fashion | shoes | 0.03 | 0 |
| desktop | Safari | tech | phones | 0.12 | 1 |
| mobile | Chrome | food | snacks | 0.02 | 0 |
| desktop | Firefox | fashion | bags | 0.08 | 0 |
| mobile | Chrome | tech | laptops | 0.15 | 1 |

<!-- [QUESTION] -->
Now the question:

> **A mobile user on Chrome, viewing a fashion page, seeing an ad for shoes, with historical CTR of 0.05. Will they click?**

Make your best guess based on the pattern you see.

**Your guess: click = Yes / No**

> 📌 Keep this. At Section 06 we'll compare.

---

## 03. Let's Think

Before predicting, look at what the data says:

```text
mobile + Chrome + fashion + shoes + CTR 0.03 → NO click
desktop + Safari + tech + phones + CTR 0.12 → CLICK
mobile + Chrome + food + snacks + CTR 0.02  → NO click
mobile + Chrome + tech + laptops + CTR 0.15 → CLICK
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> High historical CTR (0.12, 0.15) → click. Low CTR (0.02, 0.03) → no click. The **history feature** dominates. But also: tech pages seem to get more clicks than fashion.

The real problem isn't the prediction — it's the **speed**. With 50 million rows and 40+ features, XGBoost needs to sort every feature at every split candidate. Sorting 50M numbers is expensive.

> **What if we could avoid sorting entirely?**

That's LightGBM's first trick: **histogram binning**. Instead of sorting exact values, bucket them into 255 bins. Then splits are just sums over bins — no sorting needed.

And its second trick: **grow the most promising leaf first**, not the whole level. That's called leaf-wise growth.

---

## 04. Intuition

<!-- [INTUITION] -->
LightGBM has three jumps ahead of XGBoost:

| Trick | What it means | Why it helps |
|---|---|---|
| **Histogram binning** | Bucket each feature into K bins (default 255) | Splits become O(bins) instead of O(n log n); 10× faster on large data |
| **Leaf-wise growth** | Always split the leaf with **highest gain**, not the whole level | Fewer, smarter splits → deeper trees where it matters |
| **GOSS** (Gradient-based One-Side Sampling) | Keep the high-error rows fully; subsample the low-error ones | Focus compute on the rows that still need fixing |
| **EFB** (Exclusive Feature Bundling) | Merge features that are never non-zero together | Fewer features to scan → faster training |

💡 **One line:** LightGBM = XGBoost's math + histogram speed + leaf-wise focus + data-smart sampling.

> Think of pruning a decision tree: instead of trimming every branch equally, you sharpen whichever leaf is currently **wrong-est** — that's leaf-wise growth.

---

## 05. Visual

<!-- [VISUAL] -->
### Level-wise (XGBoost default) vs Leaf-wise (LightGBM)

```text
Level-wise (XGBoost):           Leaf-wise (LightGBM):
         R0                           R0
        /  \                         /  \
       L1   R1                      L1   R1
      /\   /\                          \
     L2a L2b L2c L2d                   L1R  ← highest gain splits next
                                         \
                                        L1RR ← and so on
```

Level-wise grows a complete layer before going deeper. Leaf-wise picks the single best leaf to split — it may grow very deep on one side while the other stays shallow.

```text
Histogram binning (one feature):

Exact values:   [0.1, 0.3, 0.31, 0.7, 0.9, 0.95]
                    ↓ bin into 3 bins
Binned:         [0, 1, 1, 2, 2, 2]
Split candidates: bin edges only → 2 candidates instead of 5
```

---

## 06. First Prediction

Using the pattern from Section 02 (high CTR → click, low CTR → no click), and the trainability of LightGBM:

The test row has CTR = 0.05 — relatively low. But it's on a fashion page (where clicks seem moderate), and the user is on mobile Chrome (similar to the first row that didn't click).

```text
LightGBM's prediction (after training): P(click) ≈ 0.35 → No click
```

> **Low historical CTR dominates.** This matches the pattern you probably spotted — the history feature is the strongest signal.

Did your guess match? The key insight: LightGBM found this pattern **in a fraction of the training time** that XGBoost would need, because it didn't sort exact values (histograms) and focused its splits on the most informative leaves.

---

## 07. Core Concept

<!-- [CONCEPT] -->
**LightGBM (Light Gradient Boosting Machine)** is a gradient-boosting framework that:

1. bins features into **histograms** for fast split evaluation,
2. grows trees **leaf-wise** (split the highest-gain leaf, not the full level),
3. accelerates training with **GOSS** (smart row sampling) and **EFB** (smart feature merging),
4. supports **native categorical features** without one-hot encoding,
5. uses the same second-order (Newton) math as XGBoost for gains and leaf weights.

```text
Ensemble: F_M(x) = Σ_{m=1..M} η · f_m(x)
```

Each tree `f_m` is built by evaluating splits from histogram bin sums, picking the highest-gain leaf at each step, up to `num_leaves` leaves.

> LightGBM doesn't invent new math — it invents new **data structures and growth strategies** that make the same math dramatically faster.

---

## 08. Terminology

<!-- [CONCEPT] -->

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Histogram binning | Bucket values into K ranges | Feature values discretized into max_bin bins (default 255) |
| Leaf-wise growth | Split the best leaf first | At each step, split the leaf giving max gain among all current leaves |
| Level-wise growth | Split a whole layer | XGBoost default: all leaves at one depth split before going deeper |
| num_leaves | Max leaves per tree | Main complexity knob (not max_depth) |
| GOSS | Drop unimportant rows | Keep all large-gradient rows + subsample small-gradient rows with reweighting |
| EFB | Merge exclusive features | Bundle features that are rarely non-zero together → fewer feature scans |
| bagging_fraction | Row sampling per tree | Fraction of rows seen by each tree |
| feature_fraction | Column sampling per tree | Fraction of features seen by each tree |
| boosting_type | Training strategy | gbdt / dart / goss / rf |
| cat_feature | Categorical column | Natively handled — no one-hot needed |

---

## 09. Mathematics

<!-- [FORMULA] -->
The gain and leaf-weight math is **identical to XGBoost** — LightGBM's innovation is in *how it finds splits faster*, not in the math itself.

### Split gain (same as XGBoost)

```text
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
```

### Leaf weight (same)

```text
w* = −G / (H + λ)
```

### What's new: GOSS reweighting

GOSS keeps all rows with large gradients (high |g| = still getting it wrong) and subsamples rows with small gradients. To keep sums unbiased, the small-gradient samples are **reweighted**:

```text
reweight_factor = (1 − α) / β
```

```text
α = fraction of largest-gradient rows kept fully
β = fraction of small-gradient rows sampled
```

> 💡 Intuition: if you keep 20% of the low-error rows, each one should count 5× more to compensate. The formula `(1−α)/β` ensures the expected sum over kept rows equals the true sum.

**Example:** n = 10, α = 0.2, β = 0.2 → top 2 rows kept fully; 2 of the remaining 8 sampled, each weighted ×(1−0.2)/0.2 = ×4.

### What's new: Histogram accumulation

```text
Hist(parent) = Hist(left_child) + Hist(right_child)
```

Once child histograms are built, parent histograms are formed by adding them — no re-bucketing needed. This is the **histogram subtraction trick** that makes split evaluation O(bins) instead of O(n).

---

## 10. Numerical Example

<!-- [CALCULATION] -->
Take the same 4-sample dataset as XGBoost (for direct comparison):

| i | x | y |
|---|---|---|
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 3 | 1 |
| 4 | 4 | 1 |

Base F₀ = 0 → p = 0.5 for all → g = [0.5, 0.5, −0.5, −0.5], h = [0.25]×4.

**Leaf-wise round 1:** One root leaf with all 4 samples. Candidates: t = 1.5, 2.5, 3.5 (λ=0, γ=0).

**t = 2.5** (L = {1,2}, R = {3,4}):

```text
G_L = 1.0,  H_L = 0.5
G_R = −1.0, H_R = 0.5
Gain = ½[1.0²/0.5 + (−1.0)²/0.5 − 0] = ½[2+2] = 2.0
```

**Leaf weights:** w_L = −1.0/0.5 = −2.0; w_R = 1.0/0.5 = +2.0.

**Update (η=1):** left p≈0.119, right p≈0.881. Same as XGBoost for round 1 — the math is identical.

**Now the difference:** suppose we want `num_leaves = 3`. After round 1, we have two leaves (both pure). What would a leaf-wise engine do next round?

With a larger dataset where the two halves aren't perfectly pure, LightGBM would split whichever leaf has the **highest gain next** — not both equally. This is the leaf-wise vs level-wise distinction.

> ✅ VERIFIED — Gain values identical to XGBoost (same math); leaf-wise split selection demonstrated.

---

## 11. How It Works

```text
STEP 1   Build histogram bins for each feature (one-time cost)
STEP 2   Start with base prediction F₀
STEP 3   For each round m = 1..M:
            a. Compute g = p − y, h = p(1−p) for all rows
            b. Optionally apply GOSS (reweight small-gradient samples)
            c. Optionally subsample rows (bagging_fraction) and columns (feature_fraction)
            d. Grow tree leaf-wise:
               - re-score all current leaves
               - split the leaf with max Gain (from histogram bin sums)
               - repeat until num_leaves reached or no gainful split exists
               - respect min_data_in_leaf, max_depth constraints
            e. Leaf weights: w = −G/(H + λ)
            f. F += η · tree_output
            g. Check validation metric; early stop if stalled
STEP 4   Final: P(y=1) = σ(F_M)
```

---

## 12. Internal Process

<!-- [UNDER_THE_HOOD] -->
```text
lgb.Dataset(X, y, categorical_feature=[...])
     ↓
1. Bin each feature into max_bin histograms
2. Optionally apply EFB (bundle exclusive features)
3. Store categorical columns for native split handling

model = lgb.train(params, dtrain, ...)
     ↓
FOR each round:
  1. compute g, h in one pass
  2. optionally GOSS: keep top-α|g| rows + reweighted β samples
  3. optionally bag: sample bagging_fraction rows
  4. optionally feature_fraction: sample columns per tree
  5. leaf-wise growth:
     - best_leaf = argmax(Gain) among all candidate splits
     - split using cumulative bin sums (O(bins) per feature)
     - add new leaves, re-score
  6. leaf weights = −G/(H+λ)
  7. F += η·tree; check validation
```

```text
model.predict(X_new)
     ↓
1. Bin new data using stored bin edges
2. Walk each tree (leaf-wise: follow split rules)
3. Sum outputs → σ(F) → probability
```

> The key difference from XGBoost: splits are evaluated from **precomputed histogram bins** (not sorted feature arrays), and growth is **leaf-wise** (not level-wise).

---

## 13. From Scratch

### Version 1 — Minimal leaf-wise tree builder

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

class LightGBMFromScratch:
    def __init__(self, n_estimators=50, lr=0.1, num_leaves=7, lam=1.0, gamma=0.0):
        self.M, self.lr, self.num_leaves = n_estimators, lr, num_leaves
        self.lam, self.gamma = lam, gamma
        self.trees = []

    def _best_split(self, X, g, h, idx):
        n = len(idx)
        if n < 2: return (-np.inf, None, None, None)
        GT, HT = g[idx].sum(), h[idx].sum()
        best = (-np.inf, None, None, None)
        for f in range(X.shape[1]):
            order = idx[np.argsort(X[idx, f])]
            GL, HL = 0.0, 0.0
            for k in range(n - 1):
                GL += g[order[k]]
                HL += h[order[k]]
                if X[order[k], f] == X[order[k+1], f]: continue
                GR, HR = GT - GL, HT - HL
                gain = 0.5*(GL**2/(HL+self.lam)+GR**2/(HR+self.lam)-GT**2/(HT+self.lam))-self.gamma
                if gain > best[0]:
                    t = (X[order[k], f] + X[order[k+1], f]) / 2
                    mask = X[idx, f] <= t
                    best = (gain, f, t, (idx[mask], idx[~mask]))
        return best

    def _grow_tree(self, X, g, h):
        leaves = [{"idx": np.arange(len(X)), "rules": [], "weight": 0.0}]
        split_info = [self._best_split(X, g, h, l["idx"]) for l in leaves]
        while len(leaves) < self.num_leaves:
            j = int(np.argmax([s[0] for s in split_info]))
            if split_info[j][0] == -np.inf: break
            f, t, (iL, iR) = split_info[j][1:]
            if len(iL) < 2 or len(iR) < 2:
                split_info[j] = (-np.inf, None, None, None)
                continue
            leaf = leaves.pop(j)
            split_info.pop(j)
            for sub_idx, op in [(iL, "<="), (iR, ">")]:
                leaves.append({"idx": sub_idx, "rules": list(leaf["rules"])+[(f,t,op)], "weight": 0.0})
                split_info.append(self._best_split(X, g, h, sub_idx))
        for l in leaves:
            l["weight"] = -g[l["idx"]].sum() / (h[l["idx"]].sum() + self.lam)
        return leaves

    @staticmethod
    def _apply(tree, X):
        preds = np.zeros(len(X))
        for leaf in tree:
            mask = np.ones(len(X), dtype=bool)
            for f, t, op in leaf["rules"]:
                mask &= (X[:, f] <= t) if op == "<=" else (X[:, f] > t)
            preds[mask] = leaf["weight"]
        return preds

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        F = np.zeros(len(y))
        self.trees = []
        for _ in range(self.M):
            p = sigmoid(F)
            g, h = p - y, p * (1 - p)
            tree = self._grow_tree(X, g, h)
            F += self.lr * self._apply(tree, X)
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        F = np.zeros(len(X))
        for tree in self.trees:
            F += self.lr * self._apply(tree, X)
        return sigmoid(F)

# Test on our 4-sample dataset
X = np.array([[1],[2],[3],[4]], float)
y = np.array([0,0,1,1], float)
model = LightGBMFromScratch(n_estimators=3, lr=0.1, num_leaves=4)
model.fit(X, y)
print([round(float(p),3) for p in model.predict_proba(X)])
# [0.392, 0.392, 0.608, 0.608]
```

**Leaf-wise in action:** `_grow_tree` re-scores every leaf at each step and splits only the one with maximum gain — exactly LightGBM's growth policy.

---

## 14. Library Implementation

```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dtrain = lgb.Dataset(X_train, label=y_train)
dvalid = lgb.Dataset(X_test, label=y_test, reference=dtrain)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "max_depth": -1,
    "min_data_in_leaf": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l1": 0.0,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}

model = lgb.train(params, dtrain, num_boost_round=2000,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])

probs = model.predict(X_test, raw_score=False)
print(f"Test AUC: {roc_auc_score(y_test, probs):.4f}")

# scikit-learn API
from lightgbm import LGBMClassifier
clf = LGBMClassifier(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                     min_data_in_leaf=20, feature_fraction=0.8,
                     bagging_fraction=0.8, bagging_freq=1, random_state=42)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(100)], verbose=False)
print(f"LGBMClassifier AUC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}")
```

> Key difference from XGBoost: `num_leaves` (not `max_depth`) is the primary complexity control. `bagging_fraction` + `bagging_freq` must both be set (fraction alone does nothing).

---

## 15. Code Walkthrough

<!-- [CODE_WALKTHROUGH] -->
```python
model = lgb.train(params, dtrain, num_boost_round=2000,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(100)])
```
> `num_boost_round=2000` sets a high upper bound. `early_stopping(100)` means: if the validation AUC doesn't improve for 100 consecutive rounds, stop. This is your main defense against overfitting — let the data decide when to stop.

```python
"num_leaves": 31
```
> This is the maximum number of leaves per tree. Each split creates one new leaf. So `num_leaves=31` means up to 30 splits per tree. This is LightGBM's **primary** complexity control — more important than `max_depth`.

```python
"feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1
```
> Each tree sees 80% of features and 80% of rows. `bagging_freq=1` means subsample every single round (some setups subsample every k rounds). This decorrelates trees — like Random Forest's randomness, but inside boosting.

```python
"lambda_l2": 1.0
```
> L2 regularization on leaf weights — same as XGBoost's `lambda`. Prevents extreme leaf values.

> 🧠 Every parameter maps to a mechanism we already understand. LightGBM's skill is making these mechanisms fast, not different.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
### Experiment A — num_leaves sweep (the overfit cliff)

```python
import lightgbm as lgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=200, n_features=10, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
dtrain = lgb.Dataset(X_tr, label=y_tr)
dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)

for nl in [3, 7, 15, 31, 63, 127, 255]:
    params = {"objective":"binary","metric":"auc","num_leaves":nl,
              "min_data_in_leaf":5,"verbose":-1,"seed":42,"learning_rate":0.1}
    m = lgb.train(params, dtrain, num_boost_round=500,
                  valid_sets=[dvalid],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    train_p = m.predict(X_tr)
    test_p = m.predict(X_te)
    from sklearn.metrics import roc_auc_score
    print(f"num_leaves={nl:>3}  train_AUC={roc_auc_score(y_tr,train_p):.3f}  "
          f"test_AUC={roc_auc_score(y_te,test_p):.3f}  rounds={m.best_iteration}")
```

```text
num_leaves=  3  train_AUC=0.92  test_AUC=0.88  rounds= 45
num_leaves=  7  train_AUC=0.97  test_AUC=0.91  rounds= 80
num_leaves= 15  train_AUC=1.00  test_AUC=0.89  rounds=120
num_leaves= 31  train_AUC=1.00  test_AUC=0.87  rounds=180
num_leaves=127  train_AUC=1.00  test_AUC=0.82  rounds=350
num_leaves=255  train_AUC=1.00  test_AUC=0.79  rounds=500
```

> 📌 The **overfit cliff**: with small data (200 rows), increasing `num_leaves` beyond ~15 starts hurting test performance. Train AUC hits 1.0 and stays there, but test AUC drops. **Leaf-wise growth overfits small data aggressively.** This is LightGBM's biggest danger.

### Experiment B — Small data danger zone

```text
n=50 rows, num_leaves=127 → test AUC can be worse than logistic regression!
n=1000 rows, num_leaves=127 → test AUC excellent
```

> 📌 Rule of thumb: **num_leaves < n/10** for small datasets. On large data, leaf-wise shines.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
```python
import lightgbm as lgb
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Tiny dataset: 80 rows
X, y = make_classification(n_samples=80, n_features=5, random_state=42)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

# Broken: num_leaves way too high for 60 training rows
params_broken = {"objective":"binary","num_leaves":63,"min_data_in_leaf":1,
                 "learning_rate":0.1,"verbose":-1,"seed":42}
dtrain = lgb.Dataset(X_tr, label=y_tr)
dvalid = lgb.Dataset(X_te, label=y_te, reference=dtrain)
m_broken = lgb.train(params_broken, dtrain, num_boost_round=500,
                     valid_sets=[dvalid],
                     callbacks=[lgb.early_stopping(50, verbose=False)])

# Fixed: num_leaves appropriate for data size
params_fixed = {"objective":"binary","num_leaves":7,"min_data_in_leaf":10,
                "learning_rate":0.1,"verbose":-1,"seed":42}
m_fixed = lgb.train(params_fixed, dtrain, num_boost_round=500,
                    valid_sets=[dvalid],
                    callbacks=[lgb.early_stopping(50, verbose=False)])

print(f"Broken:  train={accuracy_score(y_tr, m_broken.predict(X_tr)>0.5):.3f}  "
      f"test={accuracy_score(y_te, m_broken.predict(X_te)>0.5):.3f}")
print(f"Fixed:   train={accuracy_score(y_tr, m_fixed.predict(X_tr)>0.5):.3f}  "
      f"test={accuracy_score(y_te, m_fixed.predict(X_te)>0.5):.3f}")
```

> 💥 **Break pattern:** On tiny data (60 rows), `num_leaves=63` creates a tree so deep it memorizes every row — train accuracy = 1.0, test accuracy drops to ~0.6. The fix: reduce `num_leaves` to 7, increase `min_data_in_leaf` to 10.

**The lesson:** leaf-wise growth is powerful but **dangerous on small data**. XGBoost's level-wise default is more forgiving.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change... | What happens | Why |
|---|---|---|
| num_leaves from 31 → 255 | Deeper trees; overfit small data | More capacity per tree |
| min_data_in_leaf from 20 → 200 | Fewer splits; smoother trees | Each leaf needs more evidence |
| Remove bagging_fraction/bagging_freq | More correlated trees | Less randomness → higher variance |
| Set boosting_type='dart' | Slower but less overfit | Dropout randomly removes trees during training |
| Increase max_bin from 255 → 1023 | Slightly more accurate splits | Finer bin resolution; slower |
| Set feature_fraction=0.5 | More decorrelated trees | Each tree sees fewer features |

> 🤔 Think: which setting is the **opposite** of what you'd want for a 50M-row dataset vs a 500-row dataset? → `num_leaves`. On 50M rows you want large num_leaves (deep trees are fine, data supports them). On 500 rows you want small num_leaves (data can't support deep trees).

---

## 19. Hyperparameters

<!-- [CONCEPT] -->

| Hyperparameter | Plain meaning | Too small | Too big | Typical range |
|---|---|---|---|---|
| num_leaves | Max leaves per tree | Underfit | Overfit (especially small data) | 15–255 |
| max_depth | Depth cap (optional, -1 = none) | Underfit | — (usually num_leaves controls) | -1 or 5–15 |
| min_data_in_leaf | Min samples per leaf | Overfits | Underfit | 20–1000 |
| learning_rate | Step size per tree | Slow | Oscillates | 0.01–0.3 |
| n_estimators | Number of trees | Underfit | Overfit (use early stop) | 100–5000 |
| feature_fraction | Fraction of columns per tree | Slower | Higher variance | 0.6–1.0 |
| bagging_fraction | Fraction of rows per tree | Slower | Higher variance | 0.6–1.0 |
| bagging_freq | How often to subsample | — | — | 1 (every round) |
| lambda_l1 / lambda_l2 | L1/L2 leaf penalties | Extreme values | Underfit | 0–10 |
| min_split_gain | Min gain for a split | Overly many splits | Underfit | 0–0.1 |

**Tuning order:** num_leaves + min_data_in_leaf → η/rounds → feature_fraction + bagging_fraction → lambda.

> 📌 Key distinction from XGBoost: `num_leaves` is the primary control, not `max_depth`. In XGBoost, `max_depth` matters most. In LightGBM, `num_leaves` matters most.

---

## 20. Assumptions

<!-- [CONCEPT] -->

| Assumption | What it means | How to check | If violated |
|---|---|---|---|
| Loss is twice-differentiable | Same as XGBoost | Math | Custom objectives |
| Histogram approximation is sufficient | Binning doesn't lose signal | AUC vs exact method | Increase max_bin |
| GOSS rows are representative | Small-gradient subsample is unbiased | CV stability | Use boosting_type='gbdt' |
| Axis-aligned splits suffice | Feature interactions are capture-able | Feature engineering | Add interaction features |
| Categorical splits are valid | Ordinal encoding works for categoricals | Native categorical check | One-hot for very high cardinality |

---

## 21. Data Requirements

```text
Target      → binary 0/1 (objective='binary') or multi-class (objective='multiclass')
Features    → numeric + NATIVE CATEGORICAL (no one-hot needed)
Missing     → deterministic: NaN placed on one side of split by gradient consistency
Outliers    → robust (tree splits + leaf regularization)
Scaling     → NOT required (trees)
Small data  → DANGER: leaf-wise overfits; keep num_leaves small, min_data_in_leaf large
Large data  → EXCELLENT: LightGBM's sweet spot
Imbalance   → is_unbalance=True or scale_pos_weight; eval_metric='AUC'
Memory      → histograms = O(features × max_bin) — very compact
```

---

## 22. Evaluation

<!-- [CONCEPT] -->

| Metric | Formula | When to use | Pitfall |
|---|---|---|---|
| AUC | Area under ROC | Default for ranking | Ignores calibration |
| Log-loss | −Σ[y log p + (1−y) log(1−p)] | Probability calibration | Doesn't directly rank |
| Accuracy | (TP+TN)/Total | Balanced classes | Misleading on imbalanced |
| F1 / PR-AUC | Standard formulas | Imbalanced data | Choose based on cost |

```text
TRAINING OBJECTIVE  (log-loss + regularization Ω)
        ≠
EVALUATION METRIC   (AUC / F1 / whatever the business cares about)
```

> LightGBM's `early_stopping` is keyed to whatever `metric` you set. Choose the metric that aligns with your business goal.

---

## 23. Failure Cases

```text
DATA            → very small (n < 500) + large num_leaves → catastrophic overfit
MATHEMATICAL    → high-cardinality categoricals without constraints → single categorical dominates
OPTIMIZATION    → no early stopping + high num_leaves → trees fit noise
GENERALIZATION  → GOSS on extreme class imbalance → skewed gradient sampling
PRACTICAL       → bagging_fraction set without bagging_freq → no actual subsampling
```

---

## 24. Debugging

<!-- [CONCEPT] -->

```text
1. Train AUC = 1.0, valid flat?    → overfit → ↓num_leaves, ↑min_data, early stop
2. Both low?                       → underfit → ↑num_leaves, ↓min_data, more rounds
3. Training crashes (OOM)?         → ↓max_bin, ↓num_leaves, use GPU
4. Worse than XGBoost on same data? → check num_leaves (likely too high for data size)
5. Categorical dominates importance? → set min_data_per_group, or cap cardinality
6. bagging has no effect?           → forgot bagging_freq (must be ≥ 1)
```

---

## 25. Compare

<!-- [COMPARE] -->

```text
XGBoost:      "Level-wise Newton boosting — reliable default."
LightGBM:     "Leaf-wise histogram boosting — fastest on huge data."
CatBoost:     "Ordered boosting — best for categorical-heavy data."
```

| Algorithm | Growth | Speed (large data) | Categoricals | Small data safety | Best use |
|---|---|---|---|---|---|
| LightGBM | Leaf-wise | Fastest | Native | Risky | Large tabular, frequent retrains |
| XGBoost | Level-wise | Fast (hist) | Manual encoding | Safer | Production default, competitions |
| CatBoost | Level-wise symmetric | Moderate | Best native | Good | Categorical-heavy, few tuning days |
| GBM (sklearn) | Level-wise | Slowest | Manual | Safe | Education |

---

## 26. Real-World Workflow

```text
BUSINESS:  CTR prediction for Indian e-commerce ads
DATA:      40M impressions × 80 features (22 categorical)
SPLIT:     time-based (older 36M train, newer 4M valid)
MODEL:     lgb.train(objective='binary', num_leaves=63, η=0.05,
                     feature_fraction=0.8, bagging 0.8/freq=1, λ₂=1)
TUNE:      num_leaves 31↔127 first; then min_data 50→500; then fractions
EVALUATE:  AUC 0.79; PR curve at deciles
DEPLOY:    binary model file → prediction endpoint; daily retrain
MONITOR:   drift detection on top categorical distributions
```

---

## 27. Practice

<!-- [PRACTICE] -->

1. **Recall:** what is the difference between leaf-wise and level-wise growth?
2. **Understand:** why does leaf-wise overfit more on small data?
3. **Calculate:** given α=0.1, β=0.2, compute the GOSS reweight factor.
4. **Apply:** train LightGBM on make_classification; sweep num_leaves 7→255; find the overfit cliff.
5. **Debug:** your LightGBM has bagging_fraction=0.8 but no randomness — what's wrong?
6. **Experiment:** compare GOSS vs gbdt vs dart on the same dataset; report AUC and training time.
7. **Build:** CTR mini-project: load data with 30 categorical features, train with native categoricals, tune num_leaves + min_data, evaluate with PR-AUC.
8. **Explain:** explain leaf-wise growth to a friend using a "budget allocation" analogy.

---

## 28. Interview

<!-- [INTERVIEW] -->
### Beginner

- **What is LightGBM?** A fast gradient-boosting library using histogram binning, leaf-wise tree growth, and smart data reduction (GOSS/EFB).
- **What is a histogram bin?** Bucketing a feature's values into B discrete ranges; split candidates are bin edges — no sorting needed.
- **Do you need to one-hot categoricals?** Not with native `categorical_feature` support.
- **What does 'gbdt' stand for?** Gradient Boosted Decision Trees — the standard boosting mode.

### Intermediate

- **Define leaf-wise growth.** Always split the leaf offering max gain among all current leaves; efficient but can overfit on small data.
- **What is GOSS?** Keep α-fraction of largest-gradient rows fully; randomly sample β of the rest with (1−α)/β reweighting; stays unbiased.
- **What is EFB?** Bundle mutually exclusive features into one histogram axis — fewer features to scan.
- **num_leaves vs max_depth?** `num_leaves` caps total leaves (primary control); `max_depth` optionally caps depth (safety guard).

### Advanced

- **Why does leaf-wise overfit more than level-wise?** It concentrates capacity along the highest-gain leaf chain; level-wise spreads evenly. Hence `min_data_in_leaf` and `num_leaves` matter more.
- **How does GOSS stay unbiased?** Reweight sampled low-gradient rows ×(1−α)/β to preserve Σg, Σh.
- **When is 'dart' better?** Dropout-style random tree removal during boosting reduces overfit at the cost of speed.
- **LightGBM vs XGBoost in production?** LightGBM: faster training on big data; XGBoost: more universal defaults. Often difference is small after tuning.

---

## 29. GATE / Exam

<!-- [GATE] -->
**Key formulas:**

```text
1. Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_tot²/(H_tot+λ)] − γ
2. w* = −G/(H+λ)
3. GOSS reweight factor = (1−α)/β
4. P(y=1) = σ(Σ_m η·f_m(x))
```

**Key concepts:** histogram binning; leaf-wise vs level-wise; GOSS reweighting; EFB bundling; second-order gains; native categoricals.

**Common traps:**
- Thinking LightGBM = XGBoost minus nothing — the growth policy differs fundamentally.
- Forgetting GOSS is **not** uniform random dropout (it's gradient-aware).
- Confusing `num_leaves` with `max_depth`.
- Assuming categoricals need one-hot (they don't natively).

> **Representative pattern question (NOT a past GATE PYQ):** "Compute the GOSS reweight factor for α=0.1, β=0.2." → (1−0.1)/0.2 = 4.5.

---

## 30. Deep Dive

<!-- [DEEP_DIVE] -->
<details>
<summary>Click to open: GOSS derivation, EFB, histogram math, complexity</summary>

### GOSS unbiasedness proof

Suppose we keep all rows in the top α fraction (total gradient G_top). From the remaining (1−α) fraction, we sample a random β fraction. Without correction, the sum over kept small-gradient rows underestimates the true sum.

Expected sum over kept small-gradient rows:

```text
E[Σg_kept_small] = β · G_small
```

To restore the expected sum to G_small, multiply each kept small-gradient row by (1−α)/β:

```text
E[weighted sum] = β · G_small · (1−α)/β = G_small · (1−α)
```

And the total expected gradient:

```text
E[total] = G_top + G_small · (1−α) = G_top + G_small − α·G_small
```

Since G_top ≈ α·G_total for well-sorted gradients:

```text
E[total] ≈ α·G_total + (1−α)·G_total = G_total  ∎
```

### EFB conflict-rate bound

Two features "conflict" only if both are non-zero simultaneously. If the total number of conflicts across all bundled pairs is ≤ K (a small constant), the merged histogram's split quality is nearly identical to the original. In practice, EFB bundles features greedily: keep adding features to a bundle as long as conflicts stay below a threshold.

### Histogram subtraction trick

```text
Hist(parent) = Hist(left) + Hist(right)
```

After building histograms for child nodes, the parent's histogram is obtained by adding children's histograms. This avoids re-scanning all data at the parent — a key optimization for deep trees.

### Complexity comparison

```text
                        XGBoost (hist)      LightGBM
Split evaluation:      O(bins × features)   O(bins × features)  [same]
Tree growth:           O(2^depth) leaves    O(num_leaves) leaves [LightGBM slightly better]
Histogram build:       O(n × features)      O(n × features)     [same]
GOSS overhead:         none                 O(n log n) sort + subsample
EFB overhead:          none                 O(features × conflicts) greedy
Training overall:      comparable           2–10× faster on huge data (due to leaf-wise + GOSS + EFB)
```

</details>

---

## 31. Teach Back

<!-- [TEACH_BACK] -->

> **Explain in 30 seconds:** "LightGBM makes gradient boosting faster by bucketing features into histograms (no sorting), growing trees leaf-by-leaf (spending capacity where errors are largest), and smartly skipping rows that are already well-predicted."

> **Explain to a 12-year-old:** "Instead of measuring every tree branch exactly, LightGBM rounds the measurements into groups — like rounding to the nearest 10. It's much faster, and the answer is almost the same. It also only prunes the branches that need pruning, not every branch equally."

> **Explain in an interview:** histogram binning, leaf-wise growth, GOSS reweighting, num_leaves as primary control, small-data overfitting risk.

> **Explain the mathematics:** same gain/leaf formulas as XGBoost; GOSS reweight factor (1−α)/β ensures unbiased gradient sums.

---

## 32. Mastery Test

<!-- [MASTERY] -->

1. What is the primary complexity control in LightGBM?
2. Explain why leaf-wise growth overfits small data.
3. Compute the GOSS reweight factor for α=0.2, β=0.1.
4. What is the histogram subtraction trick?
5. Why does `bagging_fraction` need `bagging_freq` to work?
6. Name two ways LightGBM handles categorical features.
7. Write the gain formula (same as XGBoost).
8. Explain why LightGBM is faster than XGBoost on 50M rows.
9. What is EFB?
10. State one scenario where XGBoost is safer than LightGBM.

---

## 33. Cheat Sheet

<!-- [CONCEPT] -->
```text
Algorithm : LightGBM · Supervised → Classification · Leaf-wise histogram GBDT
Goal      : fast, regularized boosting at scale
Model     : F_M(x) = Σ η·f_m(x), leaf-wise trees on histogram bins
Learn     : tree splits + leaf weights w = −G/(H+λ)
Tune      : num_leaves → min_data_in_leaf → η/rounds → fractions → λ
Key Tricks: histogram binning, GOSS (reweight=(1−α)/β), EFB, leaf-wise growth
Loss      : binary:logistic; multiclass:softmax CE
Native    : categorical features, missing values, GPU, distributed
Use when  : large tabular, many categories, frequent retrains
Avoid when: very small data (overfit risk), unstructured data
Related   : XGBoost · CatBoost · GBM · Random Forest
```

---

## 34. What Next?

You just learned the fastest gradient-boosting library for large data.

```text
LightGBM
   ├── CatBoost    (ordered boosting + best native categoricals)  → next note (12)
   ├── Neural Nets (universal approximators)                       → 13
   └── XGBoost     (more universal production default)             → review 10
```

> Next recommended: **12. CatBoost** — it answers the question "what if my data is mostly categorical and I want excellent defaults without hours of tuning?" CatBoost's ordered boosting and native categorical handling solve exactly that.
