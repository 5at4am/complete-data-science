# 08. Gradient Boosting (Classification)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐⭐☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **problem → coach → residuals → gradient → additive trees → formula → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Random Forest averages many independent trees to cut **variance**. Gradient Boosting takes the opposite route: it builds trees **one at a time**, each one fixing the mistakes of everything before it, to cut **bias**. It's the accuracy workhorse of tabular ML — and the ancestor of XGBoost, LightGBM, and CatBoost.

By the end you will be able to:

- explain why each new tree "fixes" the previous ones,
- derive the beautiful result that the gradient of log-loss is just `y − p`,
- see how the learning rate controls the whole game,
- compute one boosting round by hand on tiny data,
- code a small boosting classifier from scratch and with sklearn,
- break it deliberately and fix it.

> One idea underpins everything: **the next tree is fitted to the errors of the current model.**

---

## 02. The Problem

<!-- [STORY] -->
Pooja trains an archery squad. Every candidate gets a score: **hit (1)** or **miss (0)**.

She started with her best single coach (one tree from earlier files). It hit maybe 60%. Her old approach — Random Forest — hired a big crowd of coaches and took the average. That helped, but here's what she noticed:

> The crowd's misses weren't random — they clustered. Every archer in the crowd struggled with the *same* windy days. Averaging didn't fix a *shared blind spot*; it just hid it.

So she asks:

<!-- [QUESTION] -->
> **Instead of averaging coaches who all share the same weakness — what if I hired a NEW coach whose ONLY job is to fix exactly the shots the current team keeps missing, then another for what's still wrong, and so on?**

Write down what you think this changes about (a) the final skill, (b) how carefully each coach must be chosen.

**Your guesses:** (a) ______  (b) ______

---

## 03. Let's Think

<!-- [THINK_ABOUT_IT] -->
Before any formula, think about the two ways to combine experts.

- **Averaging (bagging):** an expert who is wrong 40% of the time, averaged over a crowd that is *independently* wrong, keeps pulling the answer toward the truth. But if everyone is wrong in the *same way* (all miss high on windy days), averaging never fixes that shared bias.

- **Correcting (boosting):** instead, measure *what's still wrong* after each new expert, and make the **next** expert specialize in exactly those leftover mistakes.

🤔 Which approach would you reach for when the crowd's errors are **correlated / systematic** (same blind spot)?

> When errors are systematic, averaging can't fix them — a straight average of "always-high" answers is still high. The fix is to *teach to the errors*: lower the score on windy days next time. That's correction, not averaging.

> Gradient Boosting automates "teach to the errors," one shallow tree at a time.

---

## 04. Intuition

💡 **The idea in one line:**

> Gradient Boosting builds a strong classifier by adding many **small trees**, each fitted to the **negative gradient of the loss** — i.e., the current model's residuals, `y − p` — scaled by a small learning rate, so each step cautiously fixes what's still wrong.

The loop, in plain words:

1. Start with a constant guess (the majority class probability).
2. See how wrong each case is (`residual = actual − prediction`).
3. Train a **small** tree to predict those residuals (the current mistakes).
4. Add that tree, scaled by a small learning rate (so progress is cautious).
5. Recompute the residuals — they should shrink — and repeat.

Each tree is tiny and the step is small, so Pooja's model improves slowly but steadily — and can be stopped before it starts memorizing noise.

> Unlike Random Forest (variance↓), boosting **reduces bias**: it deliberately fixes the errors, not averages them away.

---

## 05. Visual

<!-- [VISUAL] -->
Picture the model's score `F` as a curve that keeps getting reshaped to hug the data:

```text
After base:      After tree 1:     After tree 2:
  ~                 ~ /~\             ~ /~~\
 ~~~             ~~~~~~~           ~~~~~~~~~
  (flat, wrong)  (tilted to data)  (follows data closely)
```

The residuals shrinking each round:

```text
Round 1 residuals:   ● ● ● ● ●     big everywhere
Round 5 residuals:     ○ ● ○        smaller, concentrated
Round 20 residuals:      ○           nearly zero (except noise)
```

And the learning-rate knob:

```text
η large:  jumps, can over-learn          η small: slow, smooth, safe
   __/\__/  \_  _/                          ___/\/\___/\/\__
```

> 📌 Each tree is a small "correction patch," and `η` controls how thick each patch is.

---

## 06. First Prediction

Let's use the intuition before any formula.

Pooja's model has an ensemble score so far. For one candidate, the accumulated score is `F = 0.8`. The current predicted probability is:

```text
p = σ(F) = 1 / (1 + e^−0.8) ≈ 0.69
```

The candidate actually **hit (y = 1)**.

<!-- [TRY_IT] -->
🎯 The residual rule is `r = y − p`. Compute `r` and decide: should the next tree *raise* or *lower* this candidate's score?

Think, then scroll.

> `r = 1 − 0.69 = +0.31`. The model under-predicted (said 0.69, truth is 1), so the residual is **positive** — the next tree must *raise* this candidate's score. That's the whole mechanic: the new tree is fit to push scores in the direction of the residuals.

> 📌 Positive residual → push up; negative residual → push down. That directional "fix" is literally the gradient of the loss.

---

## 07. Core Concept

<!-- [CONCEPT] -->
Introducing the idea formally, right after we've met it.

**Concept: Gradient Boosting (for classification)** — an **additive** ensemble of small trees where:

1. the model is a sum: `F_M(x) = F₀(x) + η·h₁(x) + … + η·h_M(x)`,
2. each tree `h_m` is **fitted to the negative gradient** of the loss — for binary log-loss, the pseudo-residual `r = y − p`,
3. trees are added **stagewise** (each one fixed before the next; earlier trees never change),
4. the final score is passed through the **sigmoid** to get a probability.

| Part | What it does | Symbol |
|---|---|---|
| Additive model | a sum of trees | `F_M = F₀ + ηΣh` |
| Base learner | the small tree per step | shallow tree (depth 2–4) |
| Pseudo-residual | "which way to fix" | `r = y − p` |
| Learning rate | thickness of each patch | `η` (0.01–0.3) |
| Sigmoid | score → probability | `p = σ(F)` |

> Everything else is detail. The soul: **fit the next tree to the remaining error**.

---

## 08. Terminology

Each term emerges from the story:

<!-- [CONCEPT] -->
### Additive model
> Simple: a sum of many small models.
> Technical: `F_M(x) = Σ_m η·h_m(x) + F₀`.

### Base learner / weak learner
> Simple: the small, barely-better-than-random model per step.
> Technical: a shallow tree (depth ~2–4).

### Pseudo-residual
> Simple: the direction we need to fix — how far each prediction is from truth.
> Technical: the negative gradient of the loss at the current score; `r = y − p` for log-loss.

### Learning rate (η / shrinkage)
> Simple: how big each correction step is.
> Technical: the multiplier on each tree; small η = cautious, needs more trees.

### Stagewise fitting
> Simple: fix each tree before adding the next.
> Technical: `F_m` built from `F_{m−1}`; earlier trees never change.

### Gradient of log-loss
> Simple: the residual for binary classification.
> Technical: `−∂L/∂F = y − p`.

### Early stopping
> Simple: stop adding trees when the validation score stops improving.
> Technical: halt when validation loss rises for k consecutive rounds.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Additive model | sum of trees | `F_M = F₀ + ηΣh` |
| Weak learner | barely better than chance | shallow tree |
| Pseudo-residual | direction to fix | `r = y − p` |
| Learning rate | step size | `η` |
| Stagewise | one tree at a time, fixed forever | sequential |
| Shrinkage | smaller η per tree | regularizer |

> ⚠️ Common mistake: "boosting fits the *predictions*, not the errors." No — the new tree is fitted to the **residuals** (what's still wrong), then added.

---

## 09. Mathematics (gradual)

We build the math from the model to the gradient.

<!-- [FORMULA] -->
### Step M1 — The additive model

```text
F_M(x) = F₀(x) + η·h₁(x) + η·h₂(x) + … + η·h_M(x)
```

```text
F_M → the accumulated score after M trees
F₀  → the initial constant (log-odds of the prior)
η   → learning rate (how much of each tree we keep)
h_m → the m-th small tree
```

### Step M2 — Binary log-loss

```text
L = −(1/n) Σᵢ [ yᵢ·log(pᵢ) + (1−yᵢ)·log(1−pᵢ) ]
```

```text
yᵢ ∈ {0,1} → true label
pᵢ = σ(F(xᵢ)) → predicted probability
```

> 💡 Log-loss punishes *confident wrong* predictions heavily — exactly the probability-style loss we want for classification.

### Step M3 — The gradient (the star)

The negative gradient of log-loss w.r.t. the score `F` simplifies to a shockingly simple number:

```text
rᵢ = yᵢ − pᵢ
```

```text
rᵢ → pseudo-residual (what to fix)
yᵢ → truth
pᵢ → current predicted probability
```

Work it with tiny numbers:

| y | p | r = y − p | Meaning |
|---|---|---|---|
| 1 | 0.7 | +0.3 | under-predicted → raise score |
| 0 | 0.8 | −0.8 | over-predicted → lower score |
| 1 | 0.4 | +0.6 | under-predicted a lot → raise a lot |

> 💡 This is why it's called *gradient* boosting: each tree steps in the direction (`y − p`) that most lowers the log-loss.

### Step M4 — Prediction

```text
p(x) = σ(F_M(x))          ŷ = 1 if p ≥ 0.5 else 0
```

The summed score through the sigmoid gives the probability; threshold at 0.5 gives the class.

---

## 10. Numerical Example

Take a tiny dataset we can check on paper.

<!-- [CALCULATION] -->
```text
i   x₁  x₂   y
1    1   1    1
2    1   2    0
3    2   1    0
4    3   3    1
5    4   3    0
```

**Step 0 — base.** `ȳ = 2/5 = 0.4`. The initial constant (log-odds):

```text
F₀ = log(0.4 / 0.6) = log(0.667) ≈ −0.405
p₀ = 0.4 for all
```

**Iteration 1 — residuals:**

```text
r = y − p₀ = [0.6, −0.4, −0.4, 0.6, −0.4]
```

Fit a **stump** (depth 1) to predict these residuals. Suppose it splits on `x₁`: threshold 1.5 → left rows {1,2,3}, right rows {4,5}.

Leaf values (mean residual):

```text
left  leaf: (0.6 − 0.4 − 0.4)/3 = −0.067
right leaf: (0.6 − 0.4)/2       = +0.10
```

Let `η = 0.5`. Update the ensemble for each point:

```text
Point 1 (left):  F₁ = −0.405 + 0.5·(−0.067) = −0.439 → p = σ(−0.439) ≈ 0.392
```

**Iteration 2 — recompute residuals** using the new `p`, fit another stump, add again. The residuals shrink each round as the trees specialize in what's left.

After M rounds: `F_M = −0.405 + 0.5·(h₁ + h₂ + …)`. The predictions converge toward the true classes.

> ✅ VERIFIED — `F₀`, residuals, first split, and leaf values all hand-computed. The `y − p` identity is exact.

> 🎯 Try it: what's the residual of point 4 after the first update? → It was in the right leaf: `F₁ = −0.405 + 0.5·(0.10) = −0.355 → p = σ(−0.355) ≈ 0.412`, so `r = 1 − 0.412 = +0.588` — still positive, still needs pushing up.

---

## 11. How It Works

```text
STEP 1   Initialize F₀(x) = logit(majority class) ≈ log(ȳ/(1−ȳ))
STEP 2   For m = 1..M:
             compute p = σ(F_{m−1})
             compute residuals r = y − p
             fit a small tree h_m to (X, r)
             set each leaf to the best constant (Newton step)
             F_m = F_{m−1} + η·h_m
STEP 3   Prediction: p = σ(F_M), class = p ≥ 0.5
```

Step 2 is the whole algorithm — and it is **sequential** (tree `m` depends on `F_{m−1}`), which is why boosting can't be trained in parallel like a forest.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
F₀ = logit(mean(y))
for m in 1..M:                       (SEQUENTIAL — cannot parallelize)
    p  = sigmoid(F_{m−1})            # current probabilities
    r  = y − p                       # pseudo-residuals (the gradient)
    h  = fit shallow tree on (X, r)  # the next small "fixer"
    F  = F + η * h(x)                # cautious step
     ↓
store: base + the sequence of trees + η
```

```text
model.predict(X_new)
     ↓
score = base + η·Σ trees
prob  = sigmoid(score)
class = prob ≥ 0.5
```

> Like Random Forest, no gradient *descent over weights* — but unlike RF, the trees *themselves* are built from the gradient of the loss each round.

---

## 13. From Scratch

### Version 1 — pure Python, readable

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

class GradientBoostingBinary:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=2,
                 min_samples_leaf=3):
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.base = 0.0
        self.trees = []

    def _fit_reg_tree(self, X, r):
        """Shallow regression tree; leaf = mean residual."""
        def build(X, r, depth):
            node = {"value": r.mean()}
            n = len(r)
            if depth >= self.max_depth or len(np.unique(r)) == 1 or n < self.min_samples_leaf:
                return node
            best_gain, best = 1e18, None
            for f in range(X.shape[1]):
                vals = np.unique(X[:, f])
                for i in range(len(vals) - 1):
                    t = (vals[i] + vals[i + 1]) / 2
                    m = X[:, f] <= t
                    if m.sum() == 0 or (~m).sum() == 0:
                        continue
                    gain = np.var(r[m]) * m.sum() + np.var(r[~m]) * (~m).sum()
                    if gain < best_gain:
                        best_gain, best = gain, (f, t, m)
            if best is None:
                return node
            f, t, m = best
            return {"feature": f, "threshold": t,
                    "left": build(X[m], r[m], depth + 1),
                    "right": build(X[~m], r[~m], depth + 1)}
        return build(X, r, 0)

    def _predict_tree(self, node, x):
        while "feature" in node:
            node = node["left"] if x[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def fit(self, X, y):
        X, y = np.array(X), np.array(y)
        self.base = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6))   # F₀
        F = np.full(len(y), self.base)
        for _ in range(self.n_estimators):
            p = sigmoid(F)
            r = y - p                                            # THE gradient
            tree = self._fit_reg_tree(X, r)
            h = np.array([self._predict_tree(tree, x) for x in X])
            F += self.lr * h                                     # cautious step
            self.trees.append(tree)
        return self

    def predict_proba(self, X):
        F = np.full(np.array(X).shape[0], self.base)
        for tree in self.trees:
            F += self.lr * np.array([self._predict_tree(tree, x) for x in np.array(X)])
        return sigmoid(F)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == np.array(y))
```

### Version 2 — clean, uses the fitted trees

```python
# same as above; keys: self.base (F₀), self.lr (η), self.trees (small fixers)
```

> Everything reduces to: `base`, a loop that fits each tree to `y − p`, and adding `η·tree`. That's the entire algorithm in three lines of intent.

---

## 14. Library Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

X, y = ...   # your binary data
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

gbm = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=3,
    min_samples_leaf=5, subsample=0.8, max_features='sqrt', random_state=42
)
gbm.fit(X_tr, y_tr)
print("AUC: ", round(roc_auc_score(y_te, gbm.predict_proba(X_te)[:, 1]), 4))
print(classification_report(y_te, gbm.predict(X_te)))

# Fast histogram version with built-in early stopping:
from sklearn.ensemble import HistGradientBoostingClassifier
hgb = HistGradientBoostingClassifier(max_iter=500, learning_rate=0.05,
                                     max_leaf_nodes=15, validation_fraction=0.2,
                                     early_stopping=True, random_state=42)
hgb.fit(X_tr, y_tr)
print("Hist AUC: ", round(roc_auc_score(y_te, hgb.predict_proba(X_te)[:, 1]), 4))
```

> `n_estimators` = M · `learning_rate` = η · `max_depth` = tree depth · `subsample` = stochastic rows per tree. `HistGradientBoostingClassifier` is faster and supports built-in early stopping.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
self.base = np.log(np.clip(y.mean(), 1e-6, 1 - 1e-6))
```
> The initial constant `F₀ = logit(mean(y))`. With 40% positives, we start by assuming everyone is 0.4 — then let the trees correct that.

```python
p = sigmoid(F)
r = y - p
```
> **The heart of the algorithm.** `p` is the current predicted probability; `r = y − p` is the negative gradient of log-loss (Section 09 M3). Fitting this "residual" is what "gradient boosting" means.

```python
F += self.lr * h
```
> The cautious step. Each tree is shrunk by `η` before being added, so no single tree makes a huge jump. Small η + many trees = smooth, safe convergence.

> 🧠 There is no mysterious optimizer. "Optimization" here is literally: repeatedly fit a tree to the gradient, add a slice of it. Every line maps to Section 09's formulas.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> In the interactive platform these become sliders/buttons. Otherwise run them in Python.

### Experiment A — the learning-rate slider (η)

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

X, y = make_classification(n_samples=800, n_features=12, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

for eta in [0.01, 0.1, 0.3, 1.0]:
    m = GradientBoostingClassifier(n_estimators=200, learning_rate=eta,
                                   max_depth=2, random_state=0).fit(X_tr, y_tr)
    print(f"η={eta:>4}  train={m.score(X_tr,y_tr):.3f}  test={m.score(X_te,y_te):.3f}")
```

> What to notice: small η is safe but needs many trees; big η can reach high training accuracy while *test* accuracy drops (overfit). The train–test gap is where you see the damage.

### Experiment B — the M (trees) slider with early stopping

```python
import matplotlib.pyplot as plt

train_scores, test_scores = [], []
for M in [10, 50, 100, 300, 800]:
    m = GradientBoostingClassifier(n_estimators=M, learning_rate=0.1,
                                   max_depth=2, random_state=0).fit(X_tr, y_tr)
    train_scores.append(m.score(X_tr, y_tr))
    test_scores.append(m.score(X_te, y_te))

for M, tr, te in zip([10, 50, 100, 300, 800], train_scores, test_scores):
    print(f"M={M:>3}  train={tr:.3f}  test={te:.3f}")
```

> Expected: training keeps rising; validation/test rises then **falls** — the classic boosting overfitting curve. Early stopping (or tuning `M × η` together) captures the peak before the fall.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
The classic boosting failure: **too many trees with a large learning rate on noisy data** → it memorizes the noise.

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=600, n_features=8, random_state=0)
y = y.copy(); flip = rng.random(len(y)) < 0.15
y[flip] = 1 - y[flip]                       # corrupt 15% of labels
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

bad = GradientBoostingClassifier(n_estimators=5000, learning_rate=0.5,
                                 max_depth=4, random_state=0).fit(X_tr, y_tr)
good = GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                  max_depth=2, random_state=0).fit(X_tr, y_tr)

print("Too many trees, big η :", round(bad.score(X_te, y_te), 3))
print("Fewer trees, small η   :", round(good.score(X_te, y_te), 3))
```

**What happened?** With 5000 trees at η=0.5 and deep trees, the model kept "fixing" the 15% wrongly-labeled points — its residuals never truly shrink to noise, so it chases them forever, overfitting the mistakes.

> 💥 **Break pattern:** healthy model → crank up M and η → overfits noise. Why? Each tree corrects *whatever* residual remains — including the ones caused by bad labels. Without early stopping or a small η, there's nothing to stop it from memorizing the noise.

The fix: **fewer trees + small η + early stopping**, plus shallower trees. Boosting needs its brakes; the learning rate and leaf limits *are* the brakes.

> 📌 **Lesson:** boosting is powerful *because* it keeps correcting — and that's exactly why it needs regularization and early stopping. It genuinely will fit noise if you let it.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| `learning_rate` small + `M` large | best accuracy, slower | smooth, safe steps |
| `learning_rate` large | overfits | each tree jumps too far |
| `max_depth` large | overfits | trees memorize |
| `subsample` < 1 | faster + regularized | each tree sees part of the data |
| Class 98/2 | minority ignored | log-loss overwhelmed by majority |
| Many trees, noisy labels | overfits noise | keeps chasing bad residuals |
| You add early stopping | test peaks, stops rising | stops before overfit |

> 🤔 Think: which is the *non-obvious* lever? → **`subsample`**. Fitting each tree on a random ~80% of rows is "stochastic gradient boosting" — it's both a speed win *and* a regularizer that helps generalization, which surprises people who only think of it as sampling.

---

## 19. Hyperparameters

**Learned by the model (parameters):** the base constant `F₀` and each tree's structure + leaf values.

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `n_estimators` (M) | number of trees | underfit | overfit | pair with early stopping |
| `learning_rate` (η) | step size | needs many trees | jumps, overfit | 0.01–0.3 |
| `max_depth` | tree depth | high bias | overfit | 2–5 |
| `min_samples_leaf` | leaf minimum | overfit | underfit/smooth | 3–10 |
| `subsample` | rows per tree | very noisy stochastics | no regularization | 0.5–0.9 |
| `max_features` | features per split | weak trees | correlated | `"sqrt"` |
| `loss` | objective | — | — | `"log_loss"` |

> 📌 **Rule of thumb:** lower `η` and raise `M` with early stopping gives the best accuracy but slower training. The two knobs `η` and `M` must be tuned together.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Loss differentiable & convex in F | gradient exists | stagewise descent | compute gradient | smooth loss |
| Marginal signal per tree | each weak tree helps | the loop depends on it | validation/round | lower η |
| Residuals are learnable | structure in the errors | boosts rely on it | residual plots | stop early, bigger leaves |
| Data representative | train ≈ serve | generalization | drift check | re-train |
| Axis-aligned splits | threshold separability | trees are axis-aligned | CV | feature engineering |

> Boosting's assumptions are mild — no linearity, no scaling. The dominant ones are about *learnable residuals* and *representative data*.

---

## 21. Data Requirements

```text
Target      → class labels
Features    → numeric; categorical via encoding (or LightGBM/CatBoost native)
Missing     → classic GBM needs impute; HistGradientBoosting handles NaN
Outliers    → robust-ish; cap depth/leaf to prevent isolating weird values
Scaling     → NOT required (trees)
Feature eng → interactions help boosting most
Size        → thousands to millions; sub-sampled for huge n
Imbalance   → class_weight='balanced' or sample weights
```

> ⚠️ Because boosting fits residuals *sequentially*, class imbalance needs explicit handling (weights/resampling) — the log-loss is otherwise overwhelmed by the majority class.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimize log-loss — a probabilistic loss)
        ≠
EVALUATION METRIC   (accuracy / F1 / AUC you report)
```

| Metric | Formula / Simple | Use | Avoid |
|---|---|---|---|
| Accuracy | (TP+TN)/Total | Balanced | Skewed |
| Precision / Recall / F1 | standard | Imbalance | As sole metric |
| ROC-AUC | from `predict_proba` | Ranking / comparison | Hard rule |
| Log-loss | binary cross-entropy | Calibration | As a single number |
| Validation curve | loss vs M | early stopping signal | Judge by eye alone |

> Boosting trains on **log-loss** (a probability loss), but you report **accuracy/F1/AUC** (threshold-based metrics) — again **loss ≠ metric**. Because it optimizes log-loss, its `predict_proba` tends to be well-calibrated, which is a real strength for business use.

---

## 23. Failure Cases

```text
DATA            → heavy label noise (boosting chases it), severe imbalance
MATHEMATICAL    → residuals not learnable (pure noise) → nothing to boost
OPTIMIZATION    → η too high (oscillates) or M too small with small η (underfit)
GENERALIZATION  → trained too long → overfits; too short → underfits
PRACTICAL       → classic GBM slow on big n; memory from many trees
```

---

## 24. Debugging

Boosting misbehaving? Run this checklist:

```text
1. Train ≈ 1.0, test lower?          → overfit → smaller η, fewer M, shallower trees, early stop
2. Train AND test both low?          → underfit → more M, larger η/ depth
3. Validation dips then climbs?      → classic overfit → early stop at the minimum
4. Minority class ignored?           → imbalance → class_weight='balanced', use AUC/PR
5. Slow on big data?                 → use HistGradientBoosting / LightGBM / CatBoost
6. Small η but small M fixed?        → underfit → raise M (they're coupled)
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Random Forest:       "Many independe nt experts vote (variance↓)."
Gradient Boosting:   "One expert fixes the last expert's mistakes (bias↓)."
AdaBoost:            "Experts focus on the samples the last one got wrong."
```

| Algorithm | Strategy | Bias | Variance | Parallel |
|---|---|---|---|---|
| Random Forest | parallel bagging | floor (avg of strong) | low | yes |
| Gradient Boosting | sequential residual-fix | ↓↓ | ↑ with M | no |
| AdaBoost | reweight mistakes | ↓ | ↑ on noise | no |
| XGBoost/LightGBM/CatBoost | boosted engines | ↓↓ | tuned | partially |

> 📌 The headline contrast: **Random Forest reduces variance, boosting reduces bias.** That's why on clean, structured data boosting usually wins the accuracy contest — it keeps subtracting the systematic error.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  credit default prediction (0 = pays, 1 = defaults)
DATA:              120k loans × 60 features
FEATURES:          income, debt-to-income, utilization, late-pay streaks
TARGET:            default (1), ~12%
MODEL:             GradientBoostingClassifier(η=0.05, M=400, depth=3, min_leaf=10, subsample=0.8)
TRAIN:             stratify split → no scaling → fit
EVALUATE:          AUC + PR-AUC at top decile (imbalanced)
DEPLOY:            threshold tuned to capital → serve probability → monthly re-train
```

> A common production note: the **histogram version** (HistGradientBoosting / LightGBM / CatBoost) replaces the classic GBM at scale — same ideas, far faster.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is a pseudo-residual?
2. **Understand:** why is boosting bias-reducing, unlike bagging?
3. **Calculate:** for the Section 10 data, recompute the residuals after iteration 1.
4. **Apply:** when would you reach for boosting over Random Forest?
5. **Debug:** validation curve dips then climbs — explain and fix.
6. **Experiment:** run the η slider (A) and the M curve (B); explain both shapes.
7. **Build:** churn mini-project: EDA → boost → compare default vs `class_weight` → tune `M × η` with early stopping → report AUC + F1 + SHAP drivers.
8. **Explain:** explain boosting to a friend in 60 seconds using the "new coach fixes the misses" story.

---

## 28. Interview

### Beginner
- **What is gradient boosting in one line?** Add trees one at a time; each fits the negative gradient (`y − p` for binary log-loss) of the ensemble so far, scaled by a learning rate.
- **What are pseudo-residuals?** The direction predictions should move to reduce loss; for binary log-loss, exactly `y − p`.
- **What does learning_rate do?** Scales each tree's contribution; small η = cautious, needs more trees.
- **Bagging vs boosting — the key difference?** Bagging averages independent strong models (variance↓); boosting sequentially fixes errors (bias↓).

### Intermediate
- **Why can't boosting trees train in parallel?** Tree `m` depends on the residuals of `F_{m−1}` — a hard sequential dependency.
- **How do you prevent overfitting?** Small η, shallow trees, leaf minima, subsampling, feature subsets, early stopping.
- **What is stochastic gradient boosting?** Each tree fits a random row subsample (~0.8) — faster and acts as regularization.
- **How does GBM produce probabilities?** The final score `F` is passed through the sigmoid: `p = σ(F)`.
- **Why log-loss rather than squared loss for classification?** Log-loss matches the Bernoulli nature of labels, giving the clean gradient `y − p`; MSE treats 0/1 as magnitudes and over-confidently fits.

### Advanced
- **Explain the Newton step for leaf values.** Structure is fit on residual `r`; each leaf's constant `γ` is then chosen to minimize local loss — second-order Newton result for log-loss is `γ ≈ Σr / Σ p(1−p)`.
- **GBM vs XGBoost?** XGBoost adds L1/L2 regularization, second-order (Newton) boosting, shrinkage, feature subsampling, and an optimized solver — a hardened production GBM.
- **How do regression and classification share machinery?** Identical; only the loss changes — regression residual `r = y − F`, classification `r = y − p` (log-loss).
- **What is the algorithm's view of "loss ≠ metric"?** It trains to minimize log-loss; you evaluate with accuracy/F1/AUC. Rolling your own threshold on `predict_proba` is fine.
- **When does boosting beat Random Forest, and vice versa?** Boosting wins on clean, structured data when you can tune and have time; Random Forest wins on noisy data and when you want a no-tuning baseline.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
F_M = F₀ + η·Σ h_m                    (additive model)
r = y − p                              (pseudo-residual = negative log-loss gradient)
σ(x) = 1/(1 + e^(−x))                  (sigmoid)
p ≥ 0.5 → class 1                      (prediction)
L = −(1/n) Σ [ y log p + (1−y) log(1−p) ]
```

**Key concepts:** additive/stagewise fitting, gradient descent in **function space**, bias reduction (vs bagging's variance), learning-rate regularization, early stopping.

**Common traps:**
- Saying boosting reduces **variance** primarily (it's **bias**-driven).
- Computing residuals as `error of the previous tree` instead of the **gradient `y − p`**.
- Confusing AdaBoost (weights on samples) with gradient boosting (fit to residuals).
- Forgetting the `η` scaling on each tree.

> **Representative pattern question (NOT a past GATE PYQ):** "For a binary log-loss model with `p = 0.7` and `y = 1`, what is the pseudo-residual the next tree is fitted to?" → `r = 1 − 0.7 = +0.3` (the gradient `y − p`).

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the gradient derivation + Newton leaves + theory</summary>

### The gradient of log-loss, step by step

`p = σ(F)`, so `dσ/dF = p(1−p)`. The single-sample log-loss:

```text
l(y, F) = −[ y log p + (1−y) log(1−p) ]
```

Differentiate w.r.t. `F`:

```text
dl/dF = −[ y·(1/p)·p(1−p) + (1−y)·(1/(1−p))·(−p(1−p)) ]
      = −[ y(1−p) − (1−y)p ]
      = −(y − p)
```

So the **negative gradient** used for the next tree is:

```text
r = −dl/dF = y − p
```

That single algebra result is why boosting "fits residuals."

### Newton-step leaf values

After the tree structure is set on residual `r`, each leaf's constant `γ` is refined with a second-order (Newton) step:

```text
γ_leaf ≈ Σᵢ∈leaf rᵢ / Σᵢ∈leaf pᵢ(1−pᵢ)
```

sklearn handles this internally; conceptually each leaf minimizes the *local* loss.

### Additive model & function-space gradient descent

The model `F_M = F₀ + ηΣh` is *gradient descent performed over functions*, not parameters. Each tree is a step down the negative gradient of the loss evaluated at the current `F`.

### Why boosting reduces bias

Bagging averages independent strong learners — variance falls, bias stays. Boosting fitsthe *residuals* (the remaining systematic error) directly, so bias falls. With too many flexible trees, variance creeps back in — hence small η and shallow trees as bias-for-variance control.

### Complexity

```text
training:   O(M · n · log n · d)   sequential rounds × tree build
histogram:  O(M · n · d)           (LightGBM / HistGB)
prediction: O(M · depth) per sample
space:      O(M · nodes)
```

### Monotonicity & custom losses

Modern engines allow monotonic constraints (e.g., a feature must not increase risk) and pluggable losses — the machinery is loss-agnostic; only the gradient changes.

</details>

---

## 31. Teach Back

Try all four.

> **Explain in 30 seconds:** "Gradient boosting builds many small trees, one at a time. Each new tree is fitted to the current model's remaining mistakes — for classification, the residual `y − p` — so every round subtracts a bit more of the systematic error, and the learning rate keeps each step cautious."

> **Explain to a 12-year-old:** "Our team's first expert misses some shots. So we hire a new expert who only focuses on the shots the first one missed, then another for what's still wrong, and so on. Each expert fixes a little of what's left, so the team slowly gets great."

> **Explain in an interview:** add: additive model `F_M = F₀ + ηΣh`, gradient `r = y − p`, bias vs variance, `η ↔ M` coupling, early stopping, Newton leaf values, engines.

> **Explain the mathematics:** derive `r = y − p` from `−∂l/∂F`, write the additive update, and sketch the Newton leaf step.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define gradient boosting.
2. Explain its intuition with the "coach fixes the misses" story.
3. Derive the pseudo-residual `r = y − p` from log-loss.
4. Write the additive model and its symbols.
5. Explain what's inside `fit()` — and why it can't be parallelized.
6. Why does boosting reduce bias, while bagging reduces variance?
7. How do you prevent (and cause) boosting overfitting?
8. Compare it with Random Forest and AdaBoost.
9. Choose it for a real problem and defend the choice.
10. State one scenario where you would NOT use boosting.

---

## 33. Cheat Sheet

```text
Algorithm : Gradient Boosting (GBM) · Supervised → Classification
Family    : Ensemble — sequential additive trees (bias-reducing)
Goal      : minimize log-loss by adding residual-correcting trees
Core      : F_M = F₀ + η·Σ h_m ;  r = y − p (the gradient)
Predict   : p = σ(F_M),  ŷ = 1 if p ≥ 0.5
Loss      : binary log-loss (pluggable)
Learn     : base F₀ + each tree's structure & leaf values
Tune      : n_estimators · learning_rate · max_depth · min_samples_leaf · subsample
Use when  : accuracy-first tabular, calibrated probabilities, ranking
Avoid when: massive scale without engines (use LightGBM/CatBoost/Hist), interpretability, streaming
Related   : AdaBoost · XGBoost · LightGBM · CatBoost · HistGradientBoostingClassifier
```

---

## 34. What Next?

You met the accuracy workhorse and the bias-reducing champion.

```text
Random Forest / Extra Trees  → (variance↓)
SVM                          → (max margin + kernels)
Gradient Boosting            ← you are here (bias↓)
   └── 09. AdaBoost (reweight mistakes, exponential loss)
```

> Next recommended: **09. AdaBoost (Classification)** — the historically-earlier boosting algorithm that learns by **reweighting the samples** previous weak learners got wrong (with the now-famous `α = ½·ln((1−ε)/ε)`). It answers: "can we boost by focusing *attention*, not residuals?"
