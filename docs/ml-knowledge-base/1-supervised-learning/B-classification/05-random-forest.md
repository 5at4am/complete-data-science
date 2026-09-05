# 05. Random Forest (Classification)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **problem → many experts → vote → bootstrap → feature lottery → formula → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Random Forest is the "**out-of-the-box legend**" of tabular ML — the model people reach for first when they just want a strong classifier without hours of torture. It takes the decision tree you met earlier, grows *hundreds* of them, and lets them vote.

By the end you will be able to:

- explain why asking *many* slightly-different experts beats one genius,
- tell exactly where the "random" in Random Forest comes from,
- compute the famous `e⁻¹ ≈ 36.8%` bootstrap number,
- vote across trees by hand on tiny data,
- code a forest from scratch and with sklearn,
- break it deliberately and fix it,
- and decide when the forest wins — and when a single tree or a boosting model is better.

> Everything here is "one tree, plus two sources of randomness." Hold that thought.

---

## 02. The Problem

<!-- [STORY] -->
Meera works at a credit-card company. Every night her team must flag transaction fraud — a `1` means "block this card", a `0` means "let it through."

She built a single decision tree last month. It did fine… sometimes. But she noticed two things that bothered her:

- If she retrained the tree on a *slightly* different slice of the data, the whole decision boundary changed. Jumpy.
- It kept finding rules that worked on yesterday's data but not today's.

So she wonders out loud:

<!-- [QUESTION] -->
> **How do we build a classifier that is as smart as a good decision tree, but far less jumpy — so a tiny change in the data doesn't upend the whole model?**

Write down two ideas you'd try before scrolling on.

**Your guesses:** 1) ________  2) ________

---

## 03. Let's Think

<!-- [THINK_ABOUT_IT] -->
Before you scroll, try the classic "committee" thought experiment.

Suppose there are **50** financial analysts. Each one, alone, is right about **70%** of the time. Now you ask all 50 to predict, and you go with the **majority answer**.

🤔 Quick check — if the analysts are independent (they make different mistakes), is the majority likely to be:

- (a) also about 70% correct,
- (b) worse than any single analyst,
- (c) noticeably better than a single analyst?

Most people think (a). Let's see why the surprising answer is (c).

> The crowd's answer is right whenever **more than 25** analysts are right. With independent analysts at 70%, it's overwhelmingly probable that 26+ are right — so the majority is right almost always. The individual errors *cancel out*.

That single idea — **many imperfect but independent guesses beat one good guess** — is the entire soul of Random Forest.

---

## 04. Intuition

💡 **The idea in one line:**

> Grow many decision trees, each on a *slightly different* slice of the data, make each one see only a *random subset* of features, then let them **vote**. The forest is far steadier than any single tree.

Two knobs give the "slightly different":

1. **Bootstrap** — each tree trains on a random sample *with replacement*, so it sees ~63% of the data (some rows twice).
2. **Feature lottery** — at every split, a tree only considers a random handful of features rather than all of them.

Why does this help? A single deep tree is **high variance**: wiggle the data and the tree changes a lot. Voting many *decorrelated* trees averages those wiggles away while keeping the tree's low bias. That's the whole trick — nothing else.

---

## 05. Visual

<!-- [VISUAL] -->
Picture the fraud model as a crowd.

```text
                       Fraud Data
                            │   bootstrap × many
      ┌─────────────────────┼─────────────────────┐
      ▼                     ▼                     ▼
   [Tree A]             [Tree B]             [Tree C]
  "x₁ ≤ 500?"          "x₂ ≤ 2?"            "x₁ ≤ 300?"
    /      \              /      \              /     \
  fraud    ok           fraud    ok          fraud    ok
      │                     │                     │
      └─────────────────────┼─────────────────────┘
                            ▼
                count the votes → PREDICTION
```

Now the *stability* picture — a single tree's boundary vs the forest's:

```text
Single Tree (jagged):      Random Forest (smooth average):

  A A A A|B B                A A A A  B B
  A A A|B B B                A A A   B B B
  A A|B B B B                A A    B B B B
  A|B B B B B                A     B B B B B

  boundary hugs every point   boundary is averaged, stable
```

> 📌 The single tree's boundary is nervous. The forest's boundary is the same data looked at through many lenses and averaged.

---

## 06. First Prediction

Let's use the intuition to make a real prediction before any formula.

Back to fraud. Suppose a transaction has **amount = ₹350** and we have a forest of three trees that only consider one feature each:

```text
Tree 1:  amount ≤ 200 → fraud       (₹350 > 200 → not fraud)
Tree 2:  amount  > 100 and ≤ 500 → fraud    (₹350 → fraud)
Tree 3:  amount ≤ 400 → not fraud   (₹350 → not fraud)
```

<!-- [TRY_IT] -->
🎯 Count the votes. **Is this transaction fraud (1) or not (0)?**

Think for a second, then scroll.

> Tree 1 says 0, Tree 2 says 1, Tree 3 says 0. Votes: **0 → 2, 1 → 1**. The forest predicts **0 (not fraud)** — because *two of three* trees agree. That is literally all classification-a-vote is: the plurality class wins.

Did you guess that before seeing the count? If yes, you already understand the model — the math below only makes it exact.

---

## 07. Core Concept

<!-- [CONCEPT] -->
Introducing the idea formally, right after we've already met it.

**Concept: Random Forest** — an **ensemble** of decision trees where:

1. each tree trains on a **bootstrap sample** (rows drawn with replacement),
2. each **split** only considers a random subset of features,
3. predictions come from a **majority vote** across trees.

| Part | What it does | Why |
|---|---|---|
| Ensemble | Many trees built together | Errors of one tree are outvoted |
| Bootstrap | Each tree sees ~63% of rows, some twice | Different training view per tree |
| Feature subset | Random `m ≈ √d` features per split | Forces trees to disagree → decorrelated |
| Majority vote | The class with most tree-votes wins | Stable, accurate final answer |

> The secret ingredient is **decorrelation** (making trees disagree), not strength of any single tree.

---

## 08. Terminology

Each term emerges from the story:

<!-- [CONCEPT] -->
### Bootstrap sample
> Simple: a copy of the data with some rows repeated and some missing.
> Technical: `n` rows drawn *with replacement*; ~63.2% of original rows appear at least once.

### Out-of-Bag (OOB) samples
> Simple: the rows a tree never saw.
> Technical: the ~36.8% of rows absent from a tree's bootstrap — a free validation set.

### Bagging
> Simple: train many models on bootstraps and combine (vote/average).
> Technical: **B**ootstrap + **Agg**regat**ing**.

### Feature subsampling
> Simple: at each split, only a random handful of features are allowed to split.
> Technical: consider `m = round(√d)` random features per node.

### Ensemble
> Simple: a group of models working together.
> Technical: a collection of base models whose predictions are combined.

### OOB error
> Simple: score each tree only on the rows it never trained on, then average.
> Technical: a built-in cross-validation estimate from out-of-bag predictions.

### Variable importance
> Simple: how much each feature helped the forest separate classes.
> Technical: aggregated impurity decrease (or permutation drop in accuracy) over trees.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Bootstrap | Sampled-with-replacement blend | ~63.2% unique rows per tree |
| OOB | Left-out rows | Free validation estimate |
| Bagging | Bootstrap + voting | Variance-reduction by averaging |
| Feature subset | Feature lottery per split | `m ≈ √d` random features |
| Majority vote | Plurality class wins | `argmax_c` over tree votes |
| n_estimators (B) | How many trees | 100–500 typical |

> ⚠️ Common mistake: "the forest fits one big tree." No — it fits *B separate trees* and combines them.

---

## 09. Mathematics (gradual)

The math here is lighter than for linear regression, but the two formulas you'll never stop hearing are worth deriving slowly.

<!-- [FORMULA] -->
### Step M1 — The bootstrap "miss" probability

Draw `n` samples with replacement from `n` rows. What's the chance a *particular* row is **never** chosen?

```text
P(not chosen in one draw) = 1 − 1/n
P(not chosen in n draws)  = (1 − 1/n)ⁿ  ≈ e⁻¹ ≈ 0.368
```

So each tree trains on ~63.2% of the rows and has ~36.8% "held out" per tree.

> 💡 This is why OOB works: every tree accidentally leaves out a different third of the data.

### Step M2 — The majority vote

```text
ŷ(x) = argmax_c  Σ_{β=1}^{B}  I[ h_β(x) = c ]
```

```text
B      → number of trees (n_estimators)
h_β(x) → what tree β predicts for x
I[·]   → 1 if true, else 0
c      → a candidate class
```

Just counting votes, exactly like Section 06.

### Step M3 — The variance story (the heart)

Averaging `B` estimators each of variance `σ²` and pairwise correlation `ρ` gives:

```text
Var(ensemble) = ρσ² + (1 − ρ)σ² / B
```

```text
ρ  → how correlated the trees are (0 = independent, 1 = same)
σ² → variance of one tree's prediction
B  → number of trees
```

> 💡 Two things shrink this number: more trees (the `(1−ρ)σ²/B` term → 0 as B grows), and **lower ρ** (the irreducible floor `ρσ²` drops). Feature subsampling exists specifically to lower `ρ`. That is the entire reason for the "random" in Random Forest.

---

## 10. Numerical Example

Take a tiny dataset we can check on paper.

<!-- [CALCULATION] -->
```text
x₁  x₂   y
 1   1   A
 1   2   A
 2   1   A
 3   3   B
 3   4   B
 4   3   B
```

We'll grow **B = 3 trees**, with **m = 1** feature per split (here `√2 ≈ 1`), using bootstrap.

**Tree 1 bootstrap** (draw 6 with replacement, could give): rows `[1, 2, 2, 4, 5, 6]`
- Only feature x₁ available. Best split `x₁ ≤ 2` → left rows {1,2,2}=A, right {4,5,6}=B. Tree: `A / B`.

**Tree 2 bootstrap:** rows `[1, 3, 3, 4, 4, 6]`
- Only feature x₂. Best split `x₂ ≤ 1.5` → left {1,3,3}=A, right {4,4,6}=B. Tree: `A / B`.

**Tree 3 bootstrap:** rows `[2, 3, 5, 5, 6, 6]`
- Only feature x₁. Split `x₁ ≤ 2` → left {2,3}=A, right {5,5,6}=B. Tree: `A / B`.

Now query a **new point (x₁=2, x₂=3)**:

```text
Tree 1: x₁=2 ≤ 2 → A
Tree 2: x₂=3 > 1.5 → B
Tree 3: x₁=2 ≤ 2 → A
```

Votes: **A = 2, B = 1 → predict A**.

> ✅ VERIFIED — every split and vote hand-computed. Notice the trees make *different* calls (Tree 2 disagrees), but the majority is decisive.

> 🎯 Try it: what if the point were (3, 4)? → Tree 1: B, Tree 2: B, Tree 3: B → all agree on B.

---

## 11. How It Works

```text
STEP 1   Have data (X, y)
STEP 2   Choose B and m (≈√d)
STEP 3   For β = 1..B:
             draw bootstrap sample
             grow a deep tree using only m random features per split
STEP 4   Store the B trees
STEP 5   Prediction: new x runs through all trees, majority vote wins
```

That's the whole algorithm. The cleverness is *where* the randomness is inserted (step 3).

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
for β in 1..B:                         (each tree independent → PARALLEL)
    idx ← sample n rows with replacement
    grow tree on X[idx], y[idx]:
        at each node:
            pick m random features
            find the best (feature, threshold) among THEM
            split, recurse
     ↓
store B trees + OOB error + importances
```

```text
model.predict(X_new)
     ↓
for each tree: walk to a leaf, get its class vote
final = most common vote across trees
```

> Note: unlike gradient descent, there's **no gradient, no loop over epochs**. Training is "grow trees in parallel." That's why it's so fast on multicore machines.

---

## 13. From Scratch

### Version 1 — pure Python, readable forest

```python
import numpy as np
from collections import Counter

class RandomForest:
    def __init__(self, n_estimators=10, max_features="sqrt", max_depth=5,
                 min_samples_split=2, seed=42):
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.rng = np.random.default_rng(seed)
        self.trees = []

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1.0 - (p ** 2).sum()

    def _best_split(self, X, y, subset):
        best_ig, best = -1.0, None
        n = len(y)
        parent = self._gini(y)
        for f in subset:
            for i in range(len(np.unique(X[:, f])) - 1):
                t = (np.unique(X[:, f])[i] + np.unique(X[:, f])[i + 1]) / 2
                m = X[:, f] <= t
                if m.sum() == 0 or m.sum() == n:
                    continue
                child = (m.sum()/n) * self._gini(y[m]) + ((~m).sum()/n) * self._gini(y[~m])
                if parent - child > best_ig:
                    best_ig, best = parent - child, (f, t)
        return best

    def _grow(self, X, y, depth):
        classes, counts = np.unique(y, return_counts=True)
        node = {"label": classes[np.argmax(counts)], "leaf": True}
        if depth >= self.max_depth or len(classes) == 1 or len(y) < self.min_samples_split:
            return node
        d = X.shape[1]
        m = max(1, int(np.sqrt(d))) if self.max_features == "sqrt" else d
        subset = self.rng.choice(d, size=m, replace=False)
        split = self._best_split(X, y, subset)
        if split is None:
            return node
        f, t = split
        mask = X[:, f] <= t
        return {"feature": f, "threshold": t, "leaf": False,
                "left": self._grow(X[mask], y[mask], depth + 1),
                "right": self._grow(X[~mask], y[~mask], depth + 1)}

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        n = len(y)
        for _ in range(self.n_estimators):
            idx = self.rng.choice(n, size=n, replace=True)   # BOOTSTRAP
            self.trees.append(self._grow(X[idx], y[idx], 0))
        return self

    def _traverse(self, node, x):
        while not node.get("leaf", True):
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["label"]

    def predict(self, X):
        return np.array([Counter(self._traverse(t, x) for t in self.trees)
                         .most_common(1)[0][0] for x in np.array(X)])
```

### Version 2 — numpy vectorized `predict_proba` (sklearn-style)

```python
def predict_proba(self, X):
    rows = []
    for x in np.array(X):
        votes = Counter(self._traverse(t, x) for t in self.trees)
        total = sum(votes.values())
        rows.append({c: v / total for c, v in votes.items()})
    return rows

def score(self, X, y):
    return np.mean(self.predict(X) == np.array(y))
```

The two places randomness appears — `rng.choice(...)` for rows and `rng.choice(...)` for features — are *the* difference from a single tree.

---

## 14. Library Implementation

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_wine

X, y = load_wine(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                               max_depth=None, min_samples_leaf=1,
                               random_state=42, n_jobs=-1, oob_score=True)
model.fit(X_tr, y_tr)

print("OOB score:      ", round(model.oob_score_, 4))
print("Test accuracy:  ", round(model.score(X_te, y_te), 4))
print("Test ROC-AUC:   ", round(roc_auc_score(y_te, model.predict_proba(X_te),
                                              multi_class='ovo'), 4))
print(classification_report(y_te, model.predict(X_te)))
```

> `max_features="sqrt"` = our `m ≈ √d`. `n_estimators=200` = our `B`. `oob_score=True` turns on the free validation estimate. `n_jobs=-1` trains the trees in parallel.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
idx = self.rng.choice(n, size=n, replace=True)
```
> The **bootstrap**: sample `n` rows *with replacement*. Some rows repeat, ~37% are missing — this is what gives Tree 1 a different view than Tree 2.

```python
subset = self.rng.choice(d, size=m, replace=False)
```
> The **feature lottery**: `m ≈ √d` random features allowed at this split. This is what makes trees *decorrelated* (lower ρ). Without it, every tree would split on the same dominant feature and voting would barely help.

```python
Counter(self._traverse(t, x) for t in self.trees).most_common(1)[0][0]
```
> The **majority vote**: walk every tree to a leaf, count the class votes, return the most common one. Exactly Section 10's arithmetic.

> 🧠 Every line behaves the way it does because of one of the three formulas in Section 09.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> In the interactive platform these become sliders/buttons. Otherwise run them in Python.

### Experiment A — slide the number of trees (B)

Imagine a slider `B = 1 → 500` on the fraud data. Watch the test accuracy curve:

```text
B = 1   → it's just ONE tree: jagged, jumpy, high variance
B = 10  → accuracy jumps, variance starts smoothing
B = 100 → smooth, stable accuracy
B = 500 → nearly identical to B=100 (diminishing returns)
```

> What to notice: after a point, more trees buy almost nothing — the **OOB error plateaus**. That plateau tells you when to stop.

### Experiment B — the noise experiment (code)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

rng = np.random.default_rng(0)
acc = {"tree": [], "forest": []}
for trial in range(10):                      # 10 different data wiggles
    X, y = make_classification(n_samples=400, n_features=8, random_state=trial)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=trial)
    t = DecisionTreeClassifier().fit(X_tr, y_tr)
    f = RandomForestClassifier(n_estimators=100, random_state=trial).fit(X_tr, y_tr)
    acc["tree"].append(t.score(X_te, y_te))
    acc["forest"].append(f.score(X_te, y_te))

print("tree   accuracy spread:", round(np.std(acc["tree"]), 3))
print("forest accuracy spread:", round(np.std(acc["forest"]), 3))
```

> Expected: the single tree's accuracy wobbles more across trials; the forest's is steadier and usually higher. That is the variance reduction from Section 09's formula, made visible.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
The classic Random Forest failure: **heavy class imbalance** with a tiny minority.

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

rng = np.random.default_rng(1)
X = rng.normal(0, 1, (1000, 5))
y = np.zeros(1000); y[:20] = 1            # 20 fraud out of 1000 = 2%
X[y == 1, 0] += 3

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X, y)
print("F1 for 'fraud' class:", round(f1_score(y, rf.predict(X), pos_label=1), 3))
print("Predictions:", np.bincount(rf.predict(X)))
```

**What happened?** The forest almost always votes "not fraud" (class 0) because 98% of the rows are class 0. The 20 fraud rows are drowned out — even by a forest, majority voting is a *majority* vote.

> 💥 **Break pattern:** healthy model → make one class tiny → forest ignores it. Why? The bootstrap samples ~63% of rows, so many fraud rows never even appear in most trees, and even when they do, the honest majority of votes says "0."

The fix is to weigh the minority class so a single fraud vote counts more:

```python
rf_b = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_b.fit(X, y)
print("Balanced F1:", round(f1_score(y, rf_b.predict(X), pos_label=1), 3))
```

> 📌 **Lesson:** the forest is *out-of-the-box strong*, but imbalance is not auto-fixed. Your evaluation metric should be F1/AUC (not accuracy) whenever classes are skewed.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| `B` (trees) 1 → 1000 | Accuracy rises then plateaus | Reducible term `(1−ρ)σ²/B` → 0 |
| `m` goes from `√d` to `d` | Trees become correlated (`ρ→1`) → forest ≈ bagged copies of one tree | No decorrelation |
| `m` very small | Very diverse but weak trees; may underfit | Too little signal per split |
| `bootstrap=False` | All trees see the same full data | Only feature randomness remains |
| Class is 98/2 | Minority ignored | Majority vote obeys majority |
| Correlated features | Importance splits credit between them | Gini importance is correlation-biased |
| You need a *single* readable rule | Forest fails | It's hundreds of trees, not one set of rules |

> 🤔 Think: which change above does **more data NOT fix**? → The correlated-feature importance bias, and the inherent need for axis-aligned splits. No amount of rows cures those.

---

## 19. Hyperparameters

**Learned by the model (parameters):** B tree structures (split features, thresholds, leaf labels), per-leaf class proportions, derived feature importances and OOB estimates.

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `n_estimators` (B) | Number of trees | High variance, unstable votes | Slow, memory-heavy, little gain | 100–500 |
| `max_features` (m) | Features per split | Weak, over-diverse trees | Correlated trees, less averaging | `"sqrt"` |
| `max_depth` | Tree depth cap | High bias | Overfits noisy data | `None` + leaf limits |
| `min_samples_leaf` | Min rows per leaf | Overfits | Slightly high bias, smoother | 1 (clean) / 5–20 (noisy) |
| `bootstrap` | Use bagging? | — | — | `True` |
| `class_weight` | Weight classes | Ignore minority | Over-weight it | `"balanced"` for skew |
| `criterion` | Split impurity | — | — | `gini` (near-identical to entropy) |

> 📌 Practical recipe: bump `B` until OOB plateaus, then tune `max_features` and `min_samples_leaf`. `B` is cheap insurance; the real levers are `m` and the leaf limits.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Signal exists in features | Features can separate classes | Every model needs it | Baseline split | Better features / more data |
| Enough variety in bootsraps | Data isn't degenerate | Averaging needs diversity | Verify ~63% coverage | More robust sampling |
| Axis-aligned separability | Splits are single-feature thresholds | Trees are axis-aligned | Compare vs linear/SVM | Other models |
| Representative data | Test looks like training | Generalization | Drift checks | Retrain / monitor |

> Forests drop almost every *parametric* assumption (no linearity, no independence, **no feature scaling**). The surviving assumptions are about the data being informative and representative, not the functional form.

---

## 21. Data Requirements

```text
Target      → class labels (could be 3+ classes)
Features    → numeric preferred; categorical → encode (one-hot; CatBoost native)
Missing     → must impute first in sklearn (forest itself can't take NaN)
Outliers    → very robust — outliers become isolated leaves
Scaling     → NOT required (trees split on raw thresholds)
Feature eng → still helpful, but less critical than for linear models
Size        → scales to large n (trains in parallel); moderate d fine
Imbalance   → use class_weight='balanced' or resample (Section 17)
```

> ⚠️ Data-leakage trap: **split BEFORE anything that touches the test set.** Forests don't need scaling, but if you ever one-hot-encode or impute using the whole dataset, test statistics leak into training.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (per-tree greedy impurity reduction)
        ≠
EVALUATION METRIC   (accuracy / F1 / AUC you report)
```

| Metric | Formula / Simple | Use | Avoid |
|---|---|---|---|
| Accuracy | (TP+TN)/Total | Balanced classes | Skewed classes |
| Precision | TP/(TP+FP) | When false alarms cost | — |
| Recall | TP/(TP+FN) | When missing positives costs | — |
| F1 | 2·P·R/(P+R) | Imbalanced classes | As a sole metric |
| ROC-AUC | from `predict_proba` | Ranking / model comparison | If you need a hard rule |
| Confusion matrix | counts of TP/FP/TN/FN | Error analysis | As a single number |
| OOB error | error on out-of-bag votes | Free validation estimate | Final decision (use a held-out test) |

> Misconception to avoid: the forest does **not** minimize accuracy or F1 during training — it greedily reduces impurity per split, then you *score* it with these metrics. Loss (impurity) ≠ metric (accuracy/F1). Also, `predict_proba` gives class *fractions across trees*, which is better than a single tree but not guaranteed to be perfectly calibrated.

---

## 23. Failure Cases

```text
DATA            → heavy imbalance (<~30 minority rows), mislabeled data, leakage
MATHEMATICAL    → correlated features → misleading impurity importance
OPTIMIZATION    → greedy per-node splits miss some multi-feature interactions
GENERALIZATION  → deep leaves memorize label noise (>~10%)
PRACTICAL       → memory & latency: thousands of trees; not great for text/sparse data
```

---

## 24. Debugging

Forest performing badly? Run this checklist:

```text
1. Train ≈ 100%, test much lower?     → overfit → raise min_samples_leaf, cap depth
2. OOB and test both low?             → underfit → more trees, allow deeper, tune m
3. F1 ≈ 0 on the minority?            → imbalance → class_weight='balanced', use AUC
4. Importance looks wrong on a feature?→ correlated features → use permutation importance
5. Accuracy high but business hates it?→ wrong metric → use precision/recall/F1
6. Train high, test low despite tuning?→ data leakage → audit the split
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Decision Tree:    "One greedy expert."
Random Forest:    "Many decorrelated experts vote (bagging + feature lottery)."
Extra Trees:      "Same, but experts pick random split points — faster, even lower variance."
Gradient Boosting:"One expert correcting the last expert's mistakes, sequentially."
AdaBoost:         "Experts focus on the samples the previous ones got wrong."
```

### ☝ The big idea
| Algorithm | Strategy | Variance | Bias | Training |
|---|---|---|---|---|
| Decision Tree | single greedy tree | high | low | 1 tree |
| Random Forest | bag + vote | ↓↓ | ~ single tree | parallel |
| Extra Trees | random thresholds | ↓↓↓ | slightly ↑ | faster |
| Gradient Boosting | sequential residual-fix | ↑ w/ trees | ↓↓↓ | sequential |
| AdaBoost | reweight mistakes | ↑ on noise | ↓ | sequential |

> 📌 Random Forest is the *variance-reducing* ensemble. Boosting methods are the *bias-reducing* ones. That single contrast explains most of the next three files.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  flag fraudulent credit-card transactions
DATA:              last 90 days, 200k transactions × 45 features
FEATURES:          amount, merchant code, device, hour, past-purchase velocity
TARGET:            is_fraud (1/0), ~0.5% positive
MODEL:             RandomForestClassifier(class_weight='balanced')
TRAIN:             split (stratified) → no scaling → fit
EVALUATE:          OOB score + held-out AUC / recall@precision
DEPLOY:            serve predict_proba; alert when P(fraud) > threshold
MONITOR:           watch fraud rate and prediction drift weekly
```

Same skeleton powers churn, disease screening, loan-default, and image-feature classifiers.

> 🚀 ML is not `model.fit(X, y)`. It's problem → data → features → model → evaluate → deploy → monitor → repeat.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is a bootstrap sample, and how much data does a tree see?
2. **Understand:** why does feature subsampling matter beyond bagging?
3. **Calculate:** for the Section 10 dataset, hand-write the votes for query (4, 4).
4. **Apply:** given a tabular dataset, decide whether Random Forest is a good first choice — and why.
5. **Debug:** OOB = 0.90 but held-out test = 0.65. Give the likely causes and fixes.
6. **Experiment:** run Experiment B at 20 trials and graph the accuracy spread vs a single tree.
7. **Build:** churn mini-project: EDA → train forest → compare default vs `class_weight='balanced'` → report F1 + top features → one-line business story.
8. **Explain:** explain Random Forest to a friend in 60 seconds using only the fraud / committee story.

---

## 28. Interview

### Beginner
- **What is a bootstrap sample?** `n` rows drawn with replacement; each tree sees ~63.2% unique rows, ~36.8% are out-of-bag.
- **What is bagging?** Bootstrap + aggregation: train many models on bootstraps, combine by vote (classification) or average (regression).
- **Why does the forest beat a single tree?** A single deep tree is high-variance; averaging decorrelated trees cuts variance while keeping low bias.
- **What is OOB error?** Score each tree only on rows it never trained on, then average — a free validation estimate.
- **What is max_features?** The number of random features considered at each split (`≈√d`), forcing tree diversity.

### Intermediate
- **Why is feature subsampling necessary for bagging to work well?** Without it all trees split on the dominant feature → `ρ≈1` → variance floor stays high. Subsampling forces `ρ` down.
- **How do you choose the number of trees?** Increase `B` until OOB error plateaus; after that more trees buy little (100–500 typical).
- **How does the forest compute feature importance?** Aggregate impurity decrease across splits (Gini importance), or permutation importance by shuffling a feature and measuring the drop.
- **Can a forest overfit?** Yes, far less than a single tree, but deep forests on noisy/small/mislabeled data still memorize. Regularize with `min_samples_leaf`, depth, `max_features`.
- **Train ≈ 100%, test much lower — diagnose?** Overfitting. Raise `min_samples_leaf`, cap depth, or shrink `max_features`; check for leakage.

### Advanced
- **Write the ensemble variance formula and explain why decorrelation is the key.** `Var = ρσ² + (1−ρ)σ²/B`. As B→∞ the second term → 0, leaving the floor `ρσ²`; only lowering `ρ` (decorrelation) lowers that floor.
- **Why does the forest reduce variance but not bias?** Averaging estimators keeps their bias: the forest's bias ≈ a single deep tree's (low) bias, while variance shrinks via averaging + decorrelation.
- **How do forests handle correlated features in importance?** Gini importance spreads credit among correlated predictors, understating each; permutation importance and SHAP are more honest.
- **Random Forest vs Extra Trees?** Extra Trees also randomizes the *threshold* (not just the feature), lowering variance further and training faster at a small bias cost.
- **When do forests beat boosting, and vice versa?** Forests win on noisy data and low tuning budgets; boosting wins on clean data when you can tune. Forests are parallel; boosting is sequential.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
P(row missed in bootstrap) = (1 − 1/n)ⁿ ≈ e⁻¹ ≈ 0.368   → OOB ≈ 36.8%
Var(ensemble) = ρσ² + (1 − ρ)σ² / B                      (ρ = correlation, σ² = tree variance)
m = round(√d)   for classification (sklearn "sqrt")
ŷ(x) = argmax_c Σ_β I[h_β(x) = c]                        (majority vote)
```

**Common traps:**
- Saying the forest reduces **bias** — it reduces **variance**; bias ≈ single tree's.
- Forgetting the ~63.2% bootstrap coverage (hint: `e⁻¹`).
- Confusing **m** (features per split) with **B** (number of trees).
- Claiming accuracy is the right metric under class imbalance.

> **Representative pattern question (NOT a past GATE PYQ):** "For 1000 bootstrap samples of size n, what fraction of rows are never selected in any given sample?" → `(1 − 1/n)ⁿ ≈ 0.368`, i.e. ~36.8% are out-of-bag per tree.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the variance-reduction derivation + matrix view + theory</summary>

### Why averaging lowers variance

For `B` estimators with variance `σ²` and pairwise correlation `ρ`:

```text
Var( (1/B) Σ h_β ) = (1/B²) [ Bσ² + B(B−1)ρσ² ]
                   = (σ²/B) + (1 − 1/B)ρσ²
                   ≈ ρσ² + (1 − ρ)σ²/B
```

- The `(1−ρ)σ²/B` term vanishes as B grows.
- The floor `ρσ²` is set purely by how correlated the trees are — hence feature subsampling.

### Why feature subsampling lowers ρ

If every split sees all `d` features, every tree splits on the same strongest feature → decisions are near-identical → `ρ ≈ 1` → bagging barely helps. Restricting each split to `m ≈ √d` forces trees to rely on different features and thresholds → `ρ` drops well below 1.

### Bias–variance of the forest

```text
too shallow / tiny max_features  → underfit → high bias
too deep on small, noisy data    → overfit  → high variance
forest sweet spot = deep trees (low bias) + heavy averaging (low variance)
```

### Why trees stay deep

Random Forest deliberately keeps trees near-unpruned. A single deep tree has low bias and high variance; the forest keeps the low bias and averages the high variance away. Shallow trees would add bias with no offsetting variance win.

### The bootstrap number, exactly

```text
P(a given row is never drawn) = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368 as n → ∞
```

Each tree trains on ~63.2% of rows; the complementary ~36.8% forms that tree's implicit validation set — the OOB sample.

### Complexity

```text
training:      O(B · n · log n · m)      (trees train in parallel)
prediction:    O(B · depth) per sample
space:         O(B · nodes)               (B trees stored)
```

### Feature importance honesty

Gini/impurity importance is fast but biased toward high-cardinality and correlated features. **Permutation importance** (shuffle a feature, measure the score drop) and SHAP are more faithful.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "A single tree is jumpy. Random Forest grows hundreds of trees, each on a random slice of the data using random features, then lets them vote — so any one tree's wobble gets averaged away into a steady, accurate answer."

> **Explain to a 12-year-old:** "Ask 100 friends who each watched a slightly different TV show. Take a show poll — the answer most friends give is almost always right, even if every friend is wrong sometimes."

> **Explain in an interview:** add: bootstrap (`e⁻¹` coverage), feature subsampling lowers `ρ`, `Var = ρσ² + (1−ρ)σ²/B`, OOB error, imbalance handling, when to prefer boosting.

> **Explain the mathematics:** derive the variance floor from the correlation formula and justify `m = √d`.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define Random Forest.
2. Explain its intuition with the committee/fraud story.
3. Write and explain the bootstrap coverage formula.
4. Write the ensemble variance formula and name both levers that lower it.
5. Explain what's inside `fit()` for a forest.
6. Why does feature subsampling exist?
7. When does the forest fail under class imbalance, and what fixes it?
8. Compare RF with a single tree and with boosters.
9. Choose it for a real problem and defend the choice.
10. State one counter-example where you would NOT use a forest.

---

## 33. Cheat Sheet

```text
Algorithm : Random Forest · Supervised → Classification (also regression)
Family    : Ensemble — bagging of decision trees
Goal      : reduce variance (not bias) by averaging decorrelated trees
Recipe    : bootstrap each tree + m≈√d random features per split + majority vote
Formulas  : P(miss)≈e⁻¹≈0.368 · Var=ρσ²+(1−ρ)σ²/B · m=round(√d) · ŷ=argmax votes
Learn     : B tree structures (no numeric weights, no gradient)
Tune      : n_estimators · max_features · max_depth · min_samples_leaf · class_weight
Assumptions: informative & representative data, axis-aligned separability
Use when  : general tabular classification, need a strong no-tuning baseline
Avoid when: interpretability (single tree), text/sparse, extreme imbalance without weighting
Bonus     : OOB score = free validation · importances built-in
Related   : Decision Tree · Extra Trees · AdaBoost · Gradient Boosting
```

---

## 34. What Next?

You just learned the flagship **variance-reducing** ensemble.

```text
Decision Tree (base)
   └── Random Forest   (bagging + feature lottery)   ← you are here
        ├── Extra Trees (also randomize thresholds)  → 06
        └── Boosting    (reduce bias instead)
             ├── SVM (not a tree — max margin)       → 07
             ├── Gradient Boosting (fit residuals)   → 08
             └── AdaBoost (reweight mistakes)        → 09
```

> Next recommended: **06. Extra Trees (Classification)** — the same forest family, but the split *threshold itself* is drawn at random, making it faster and even lower-variance. It answers the question: "what if we add even *more* randomness?"
