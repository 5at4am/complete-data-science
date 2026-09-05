# 07. Support Vector Machine (SVM)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐☆☆
>
> Journey: **problem → widest street → margin → the edge points → formula → kernel → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

So far you've met trees and forests. SVM is a **completely different animal** — no cuts, no votes. It thinks in **distances** and draws the *widest possible street* between two classes.

It's the theoretical heavyweight of ML: convex, global optimum, gorgeous math, and a trick — the **kernel trick** — that bends a straight line into a curve with almost no extra cost.

By the end you will be able to:

- explain what a "margin" is and why wider is better,
- spot the few **support vectors** that actually define the model,
- compute a tiny SVM by hand,
- derive the hinge loss and the dual, and explain the kernel trick,
- understand why **scaling is non-negotiable** here,
- break it deliberately and fix it.

> Everything hinges on one picture: **two armies separated by the widest possible no-man's land.**

---

## 02. The Problem

<!-- [STORY] -->
Dr. Nair runs a screening lab. She separates two types of cell samples from microscope images — **benign (◯)** and **malignant (●)** — using two numeric features: cell size and cell texture.

She's tried a decision tree and a forest. They work. But her real worry is different:

> When a *brand-new* sample arrives, her model draws a boundary that just barely separates yesterday's training cells. It "works" on the training set but feels fragile — a slightly-off sample can flip across the boundary.

She shows you a drawing of the two clusters and asks:

<!-- [QUESTION] -->
> **There are infinitely many straight lines that separate the benign cells from the malignant cells. Which single line should I draw so that a slightly-noisy new sample is still classified correctly?**

Write down the principle you'd use before scrolling.

**Your principle:** ________________________________

---

## 03. Let's Think

<!-- [THINK_ABOUT_IT] -->
Before any formula, think like an urban planner drawing a border between two neighborhoods.

Two villages, benign on the left, malignant on the right. You can draw the border line *anywhere* in the gap between them.

🤔 Pick the "best" line. Which of these feels most robust?

- (a) A line that hugs the benign villages, so malignant just barely stays out.
- (b) A line that hugs the malignant villages, so benign just barely stays out.
- (c) A line that sits **as far as possible from both**, in the middle of the empty gap.

Most people pick (c) — and that's exactly SVM's answer.

> The idea: **maximize the distance to the nearest point of each class.** Draw the line as far from the closest villages on both sides as you can. A sample that's a little noisy is far less likely to cross a boundary that has a wide clearance on both sides.

That clearance has a name — the **margin** — and the points that touch the boundary's edge are the **support vectors**.

---

## 04. Intuition

💡 **The idea in one line:**

> SVM finds the separating line with the **widest possible margin** — the largest empty gap between the two classes — and a new point is classified by *which side of that line it falls on*.

Two more ideas come along for free:

1. **Only a few points matter.** The line is determined *only* by the closest points on each side — the **support vectors**. Everything else could be deleted and the line wouldn't move.
2. **If the classes aren't linearly separable**, SVM *lifts* the data into a higher dimension (imagine lifting one class "up" like a bowl) where a flat plane CAN separate them, then projects the decision back down as a curve. That's the **kernel trick**.

> No stopping at "good enough separation." SVM insists on the *safest* separation.

---

## 05. Visual

<!-- [VISUAL] -->
The famous "widest street":

```text
        x₂
         |
     ◯   |     ◯
      ◯  |   ◯          ← benign (+1)
         |
   ●     |          margin edges: w·x+b = ±1
      ●  |  ●                ◯
    ----------------●--------    ← decision hyperplane (w·x+b = 0)
        ●     |
               |
      ←───────┼────────→  width = 2 / ‖w‖
               |
     ●    ◯   |  ●    ◯   ← malignant (−1)
         ◯    |
```

The *dots sitting exactly on the margin edges* are the **support vectors** — they "hold up" the street.

```text
Original (not line-separable):   Lifted into 3D → separable by a plane:
      ◯                            ◯  ◯  ◯          ← plane at height 1
   ● ● ●  ◯           ──→               ● ● ● ● ●   ← plane at height 0
     ◯ ◯ ◯
        ●          (kernel maps back to a CURVED 2D boundary)
```

> 📌 Bigger margin = wider street = more robust. That's the WHOLE priority of SVM: margin size first, perfect fit second.

---

## 06. First Prediction

Let's use the intuition before any formula.

Back to Dr. Nair. A model has already learned the boundary `f(x) = 1.5·size − 2·texture + 5`. For a new cell with **size = 2, texture = 3**:

```text
f = 1.5×2 − 2×3 + 5 = 3 − 6 + 5 = +2
```

<!-- [TRY_IT] -->
🎯 The rule is: **if f > 0 → benign (+1), if f < 0 → malignant (−1).** What's this sample? And *how confident* is the model?

Think, then scroll.

> `f = +2 > 0`, so the sample is **benign (+1)**. The *magnitude* `|2|` tells us confidence: it's `2/‖w‖ = 2/2.5 ≈ 0.8` units away from the boundary — comfortable, but not a max-margin veteran. Positive side + positive distance = benign, with `+2` as the signed distance score.

<!-- [TRY_IT] -->
Did you get "benign" before seeing the sign rule? If yes, you already understand the decision: **the sign of the score decides the class, the size of the score is the confidence.**

---

## 07. Core Concept

<!-- [CONCEPT] -->
Introducing the idea formally, right after we've met it.

**Concept: Support Vector Machine** — a classifier that:

1. labels classes as `+1` and `−1`,
2. finds the hyperplane `w·x + b = 0` that **maximizes the margin** `2/‖w‖` between classes,
3. is defined *only* by its **support vectors** (the points on the margin edge),
4. generalizes to non-linear data via the **kernel trick**,
5. handles messy, overlapping data via a **soft margin** governed by `C`.

| Part | What it does | Symbol |
|---|---|---|
| Hyperplane | the decision boundary | `w·x + b = 0` |
| Margin | width of the empty gap | `2/‖w‖` |
| Support vectors | points on the margin edge | `αᵢ > 0` |
| Kernel | bends the boundary | `k(xᵢ, xⱼ)` |
| C | how hard to punish mistakes | `C` |

> The boundary is "held up" by its closest neighbors — the name *support vectors* comes from them literally supporting the street.

---

## 08. Terminology

Each term emerges from the story:

<!-- [CONCEPT] -->
### Hyperplane
> Simple: the dividing line (or plane / surface in higher dimensions).
> Technical: `w·x + b = 0` in d-dimensional space.

### Margin
> Simple: the width of the empty gap between the classes.
> Technical: `2/‖w‖` — double the distance from the boundary to the nearest support vector.

### Support vectors
> Simple: the edge points that "hold up" the boundary.
> Technical: training points with `αᵢ > 0`, lying on the margin edges `w·x + b = ±1`.

### Hinge loss
> Simple: the penalty for being on the wrong side (or too close to the boundary).
> Technical: `max(0, 1 − y·f(x))`.

### Kernel
> Simple: a "similarity" function that lets the boundary curve.
> Technical: `k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩`, an inner product in a transformed space computed without forming `φ`.

### Soft margin / slack
> Simple: letting a few points trespass, at a price.
> Technical: slack `ξᵢ ≥ 0` and penalty `C·Σ ξᵢ`.

### Hard margin
> Simple: demanding perfect separation.
> Technical: `C → ∞`, no slack allowed; fragile with outliers.

### C
> Simple: how strictly to avoid mistakes vs keeping a wide margin.
> Technical: soft-margin penalty per unit violation.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Hyperplane | dividing surface | `w·x + b = 0` |
| Margin | width of gap | `2/‖w‖` |
| Support vector | boundary-supporting point | `αᵢ > 0` |
| Hinge loss | wrong-side penalty | `max(0, 1 − yf)` |
| Kernel | curves the boundary | `k(xᵢ,xⱼ)` inner product |
| C | mistake tolerance | soft-margin penalty |
| Dual | optimization view | Lagrange on `α` |

> ⚠️ Common mistake: "support vectors are all the training points." No — only the handful on the margin edges matter (most `αᵢ` are exactly 0).

---

## 09. Mathematics (gradual)

We build the math from zero in small steps.

<!-- [FORMULA] -->
### Step M1 — The score

```text
f(x) = w·x + b       ŷ(x) = sign(f(x))
```

```text
w → weight vector, normal to the boundary
b → bias offset
f(x) → signed distance-like score: positive one side, negative the other
```

### Step M2 — The margin

The distance between the two margin edges is:

```text
margin = 2 / ‖w‖
```

```text
‖w‖ = √(w₁² + w₂² + ...)   the length of the weight vector
```

> 💡 Maximizing the margin ⟺ **minimizing ‖w‖**. A smaller `‖w‖` = wider street. That's the entire objective.

### Step M3 — The hinge loss

For a single point with true label `y ∈ {+1, −1}`:

```text
L_hinge = max(0, 1 − y·f(x))
```

Let's compute it with tiny numbers (for `y = +1`):

| f(x) | y·f | Loss `max(0, 1 − yf)` | Meaning |
|---|---|---|---|
| +3 | +3 | 0 | confident, correct — free |
| +0.5 | +0.5 | 0.5 | correct but inside the margin — pay a little |
| −2 | −2 | 3 | misclassified — pay a lot |

> 💡 Correct-but-confident points cost **0**. Only points that are wrong, or too close to the boundary, cost anything. That's why the model is defined by its *edge* points.

### Step M4 — The objective (soft margin)

```text
minimize  (1/2)‖w‖²  +  C · Σᵢ ξᵢ
subject to  yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ,   ξᵢ ≥ 0
```

Equivalently (the regularized hinge form):

```text
minimize  (1/2)‖w‖²  +  C · Σᵢ max(0, 1 − yᵢ(w·xᵢ + b))
```

```text
(1/2)‖w‖²  → keep the margin wide
C          → how hard to punish violations
ξᵢ         → slack: how far point i trespasses
```

> 💡 Two forces fight: a **wide margin** (small `‖w‖`) vs **few mistakes** (small `Σξ`). `C` sets the referee's strictness.

---

## 10. Numerical Example

Take a tiny 1-D dataset we can check on paper.

<!-- [CALCULATION] -->
```text
Point   x    y
 A      1    +1
 B      2    +1
 C      4    −1
```

**Goal:** find `w, b` so both classes satisfy the margin constraint `y(wx + b) ≥ 1`.

Constraint for each point:

```text
A:  w·1 + b ≥ 1       →  w + b ≥ 1
B:  w·2 + b ≥ 1       →  2w + b ≥ 1
C:  (−1)(w·4 + b) ≥ 1 →  −4w − b ≥ 1
```

Constraint B is looser than A (since `2w + b > w + b` for positive w), so the *active* (tightest) ones are **A and C** — those will be the support vectors:

```text
A on margin:  w + b = 1
C on margin:  −4w − b = 1
```

Solve. From the first, `b = 1 − w`. Substitute into the second:

```text
−4w − (1 − w) = 1
−4w − 1 + w = 1
−3w = 2
w = −2/3,   b = 1 − (−2/3) = 5/3
```

**Margin:**

```text
2/‖w‖ = 2/(2/3) = 3
```

**Predict x = 3:**

```text
f(3) = (−2/3)(3) + 5/3 = −2 + 1.667 = −0.333 < 0  →  predict −1
```

**Support vectors:** A and C (both exactly on the margin). B is interior — correctly placed, but not binding.

> ✅ VERIFIED — constraints, `w, b`, margin `3`, and prediction all hand-computed.

> 🎯 Try it: what's `f` at `x = 1.5`? → `(−2/3)(1.5) + 5/3 = −1 + 1.667 = +0.667 > 0` → predict `+1`. Points near A stay positive; points near C go negative — the boundary at `f = 0` sits between them.

---

## 11. How It Works

```text
STEP 1   Standardize the features (distance-based!)
STEP 2   Choose a kernel and C
STEP 3   Solve the convex dual (SMO):
             maximize  Σ α − ½ ΣΣ αα·y·y·k(x, x)
             subject to  Σ α·y = 0,  0 ≤ α ≤ C
STEP 4   Keep only the support vectors (α > 0), compute b
STEP 5   Prediction: ŷ = sign( Σ_{SV} αᵢ yᵢ k(xᵢ, x) + b )
```

Steps 3–4 are where the "learning" happens — a clean, convex optimization with a guaranteed global optimum.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)      (after StandardScaler)
     ↓
1. Build the kernel matrix K[i][j] = k(xᵢ, xⱼ)
2. Solve the convex quadratic program (SMO/LIBLINEAR)
     → returns multipliers α (most are exactly 0)
3. Support vectors = {i : αᵢ > 0}
4. Compute bias b from any free support vector
     ↓
store: support vectors + α + b + kernel params
```

```text
model.predict(X_new)
     ↓
score = Σ_{i in SV} αᵢ yᵢ k(xᵢ, x_new) + b
class = +1 if score ≥ 0 else −1
```

> The model stores **only the support vectors** — typically a small fraction of the training data. That sparsity is a feature: fast, compact predictions.

---

## 13. From Scratch

### Version 1 — pure Python, primal hinge (linear)

```python
import numpy as np

class SimpleSVM:
    """Primal subgradient SVM minimizing regularized hinge loss (linear)."""

    def __init__(self, learning_rate=0.01, n_epochs=100, reg=0.01):
        self.lr, self.n_epochs, self.reg = learning_rate, n_epochs, reg
        self.w, self.b = None, 0.0

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.where(np.array(y) <= 0, -1.0, 1.0)
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.n_epochs):
            for i in range(n):
                hinge = 1 - y[i] * (np.dot(self.w, X[i]) + self.b)
                if hinge > 0:                      # this point violates the margin
                    grad_w = self.reg * self.w - y[i] * X[i]   # + hinge loss grad
                    grad_b = -y[i]
                else:                              # correctly & comfortably placed
                    grad_w = self.reg * self.w     # only the margin term remains
                    grad_b = 0.0
                self.w -= self.lr * grad_w
                self.b -= self.lr * grad_b

    def decision_function(self, X):
        return np.dot(np.array(X), self.w) + self.b

    def predict(self, X):
        return np.where(self.decision_function(X) >= 0, 1, -1)

    def score(self, X, y):
        y_true = np.where(np.array(y) <= 0, -1.0, 1.0)
        return np.mean(self.predict(X) == y_true)
```

### Version 2 — sklearn-style (kernels)

```python
# Kernel SVMs require a proper dual solver (SMO, ~150 lines) in production;
# for learning, the primal above shows the objective. For real kernels:
from sklearn.svm import SVC
```

> The primal above is literal: update `w, b` *only* when a point's hinge is positive (within or across the margin) — matching Section 09's "only edge points matter."

---

## 14. Library Implementation

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

X, y = load_breast_cancer(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                          random_state=42, stratify=y)

pipe = make_pipeline(
    StandardScaler(),                                    # REQUIRED
    SVC(C=1.0, kernel="rbf", gamma="scale", probability=True, random_state=42)
)
pipe.fit(X_tr, y_tr)
print("Accuracy: ", round(accuracy_score(y_te, pipe.predict(X_te)), 4))
print("ROC-AUC:  ", round(roc_auc_score(y_te, pipe.decision_function(X_te)), 4))
print(classification_report(y_te, pipe.predict(X_te)))
```

> `StandardScaler()` inside the pipeline is **non-negotiable** — SVM is distance-based, and without scaling the kernel is dominated by the largest-range feature.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
StandardScaler()
```
> SVM's kernels are functions of Euclidean distance. If one feature spans `[0, 1e6]` and another `[0, 1]`, the first completely dominates every distance. Standardizing puts all features on equal footing. Fit it on TRAIN only.

```python
hinge = 1 - y[i] * (np.dot(self.w, X[i]) + self.b)
if hinge > 0:  grad_w = self.reg * self.w - y[i]*X[i]
else:          grad_w = self.reg * self.w
```
> This is the **hinge loss**, line by line. Points inside/outside the margin (`hinge > 0`) pull the boundary toward them; comfortably-correct points (`hinge ≤ 0`) only pay the `‖w‖²` margin term. Exactly Section 09's table.

```python
SVC(kernel="rbf", gamma="scale", C=1.0)
```
> `kernel="rbf"` uses the Gaussian similarity `exp(−γ‖x−x'‖²)` for curved boundaries. `C` balances wide-margin vs mistakes. `gamma` controls how local each "bump" is.

> 🧠 Every line maps to a formula in Section 09. Nothing arbitrary — the "only support vectors matter" behavior comes from hinge loss being zero for confident points.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> In the interactive platform these become sliders/buttons. Otherwise run them in Python.

### Experiment A — slide C (strictness)

Imagine a slider `C: 0.01 → 1000` on two overlapping clouds:

```text
C = 0.01  → very wide margin, tolerant, many points inside it (underfit-ish)
C = 1.0   → balanced sweet spot
C = 1000  → tiny margin, insists every point is on the right side (overfits outliers)
```

> What to notice: as C grows, the boundary tightens around the outliers and test accuracy can DROP — the model memorizes noise to satisfy a strict referee.

### Experiment B — scaling importance (code)

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=500, n_features=5, random_state=0)
X[:, 0] *= 1000                      # make feature 0 dominate the distance scale
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

no_scale = SVC(kernel='rbf', random_state=0).fit(X_tr, y_tr)
scaled = SVC(kernel='rbf', random_state=0).fit(StandardScaler().fit_transform(X_tr), y_tr)

print("Without scaling:", round(no_scale.score(X_te, y_te), 3))
print("With scaling:   ", round(scaled.score(StandardScaler().fit_transform(X_te), y_te), 3))
```

> Expected: the scaled model scores clearly higher. That is the scaling REQUIRED warning in action.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
The classic SVM break uses **Kevin, a wandering outlier** and overlapping classes.

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                           n_clusters_per_class=1, class_sep=0.6, random_state=0)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)
X_tr = StandardScaler().fit_transform(X_tr)
X_te = StandardScaler().fit_transform(X_te)

m = SVC(kernel='rbf', C=1.0, random_state=0).fit(X_tr, y_tr)
print("Baseline accuracy:", round(m.score(X_te, y_te), 3))

# Kevin: one rogue +1 point deep inside the −1 region
X_sabot = np.vstack([X_tr, [5.0, 5.0]]); y_sabot = np.append(y_tr, 1)
m2 = SVC(kernel='rbf', C=100.0, random_state=0).fit(X_sabot, y_sabot)   # strict C
print("After Kevin + strict C:", round(SVC(kernel='rbf', C=100.0, random_state=0)
      .fit(np.vstack([StandardScaler().fit_transform(X_tr), [6,6]]),
           np.append(y_tr, 1)).score(X_te, y_te), 3))
```

**What happened?** With a large `C`, SVM fights to put Kevin (an outlier) on the correct side, contorting the boundary around him — the margin collapses and test accuracy drops. With overlapping classes, no margin exists at all, and a strict SVM just latches onto noise.

> 💥 **Break pattern:** healthy model → add one outlier + crank C → boundary contorts. Why? `C` punishes *every* violation, so one rogue point drags the whole boundary to please it, squeezing the margin the model was built to maximize.

The fix: lower `C` (tolerate a few violations for a wider margin), or clean the data. A *softer* margin is often the *safer* one.

> 📌 **Lesson:** max-margin is a *regularization* story. If your data overlaps or has outliers, a hard, strict margin is a trap.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| `C` small → large | boundary tightens, may overfit | more punishment per mistake |
| `γ` small → large (RBF) | each point's "bump" shrinks | tight local bubbles → overfit |
| Features unscaled | one feature dominates | distance-based kernel |
| Add one outlier + big C | boundary contorts | strict margin chases the outlier |
| Classes heavily overlap | accuracy hits a ceiling | no useful margin exists |
| Data grows to millions | training blows up | kernel SVM is ~O(n²)–O(n³) |
| You need *probabilities* | need calibration | SVM outputs a score, not a probability |

> 🤔 Think: which is the *surprising* one? → **Unscaled features.** A model that "just ignores units" for trees silently collapses for SVM, because the whole method is distances. Zero extra code until you forget the scaler.

---

## 19. Hyperparameters

**Learned by the model (parameters):** `α` multipliers (mostly 0), bias `b`, and (for linear kernels) the weight vector `w`. The support-vector set is derived, not tuned.

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `C` | mistake penalty | Wide margin, underfit | Tight, overfits | log-grid [0.01..1000] |
| `kernel` | boundary shape | — | — | `rbf` default; `linear` for text |
| `gamma` (RBF) | bump locality | smooth / underfit | tight bubbles / overfit | `"scale"`, [0.01,0.1,1] |
| `degree` (poly) | poly flexibility | simple | overfit | 2–5 |
| `class_weight` | class costs | ignore minority | overweight it | `"balanced"` |

> 📌 The tuning recipe is `C` × `gamma` on a log scale, cross-validated. These two levers control nearly everything about an RBF SVM.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Comparable feature scales | distances are meaningful | kernels use distance | feature ranges | StandardScaler REQUIRED |
| Kernel suits the geometry | boundary shape matches data | kernel = similarity | CV across kernels | try rbf/poly |
| Margin structure usable | classes separable-with-violations | margins need a gap | visualize | soft C + RBF / more features |
| Reliable labels | no gross label noise | a margin chases noise | audit | clean labels, smaller C |
| Not too many samples (kernel) | QP memory grows ~O(n²) | solver cost | n vs resources | LinearSVC / SGD / trees |

> For pure **prediction**, the scaling and kernel-match rows matter most. The rest mostly govern when SVM falls apart.

---

## 21. Data Requirements

```text
Target      → binary (extend to multiclass via one-vs-rest)
Features    → numeric only; categorical must be encoded
Missing     → must impute first
Outliers    → sensitive (outliers contort a strict margin)
Scaling     → REQUIRED (critical for RBF) — fit scaler on TRAIN only
Feature eng → classic high-dim use (text, genes)
Size        → best for small/medium n; kernel SVM ~O(n²)–O(n³)
Imbalance   → class_weight='balanced' or resample
```

> ⚠️ Data-leakage trap: the scaler must learn its mean/scale on the **training split only**, then transform both. Scaling on the whole dataset leaks test statistics into training.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimize regularized hinge loss —— a margin story)
        ≠
EVALUATION METRIC   (accuracy / F1 / AUC you report)
```

| Metric | Formula / Simple | Use | Avoid |
|---|---|---|---|
| Accuracy | (TP+TN)/Total | Balanced | Skewed |
| Precision / Recall / F1 | standard | Imbalance | As sole metric |
| ROC-AUC | from `decision_function` | Ranking | When you need calibrated probs |
| Confusion matrix | counts | Error analysis | Single number |
| SV fraction | #SV / n | Diagnose complexity | — |

> Note: `decision_function` gives a *score*, not a probability. If your business layer needs a true probability, use `probability=True` (Platt scaling) or a separate calibration step. And as always: the *hinge loss* you train ≠ the *accuracy/F1* you report.

---

## 23. Failure Cases

```text
DATA            → huge n (kernel QP explodes); unscaled features; outliers
MATHEMATICAL    → γ too large → each SV is its own island; γ too small → underfit
OPTIMIZATION    → SMO slows with huge C on noisy data
GENERALIZATION  → severe class overlap can't be fixed by any margin
PRACTICAL       → probabilities need calibration; black-box curved boundaries
```

---

## 24. Debugging

SVM misbehaving? Run this checklist:

```text
1. Terrible accuracy on numeric data? → did you forget StandardScaler? (most common!)
2. Train ≈ 100%, test much lower?     → overfit → lower C and/or gamma
3. Train AND test both low?           → underfit → raise C/gamma, try a richer kernel
4. Scores huge/strange?               → scaling broken / γ mis-set
5. Taking forever on big data?        → kernel too expensive → LinearSVC/SGD or trees
6. Need probabilities for business?    → use probability=True or calibrate
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Logistic Regression: "Draw ONE sensible line, output a probability."
SVM:                 "Draw the WIDEST possible line; only edge points matter."
Decision Tree:       "Draw boxy axis-aligned cuts."
Random Forest:       "Many boxy cuts, vote."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Logistic Regression | linear + probability | calibrated, fast | linear only | baseline/probabilities |
| SVM (linear) | max-margin line | great high-dim (text) | needs scaling | text/sparse |
| SVM (RBF) | curved max-margin | great small/med non-linear | slow on big n, tuning | small data, curved |
| Decision Tree | single greedy tree | interpretable | overfits | rules/audit |
| Random Forest | bagged trees | scales, robust | less precise margins | general tabular |

> 📌 SVM vs Logistic Regression is the classic pairing: both are "linear-ish" classifiers, but LR minimizes cross-entropy (probabilities) while SVM minimizes hinge loss (margin). SVM is the *robustness*-focused cousin; LR is the *probabilistic* one.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  flag fraud from engineered transaction features
DATA:              80k transactions × 45 features, ~1.2% fraud
FEATURES:          amount-to-median, velocity counts, hour stability
TARGET:            is_fraud (1/0)
MODEL:             SVC(C=1, kernel='rbf', gamma='scale', class_weight='balanced')
TRAIN:             time-split → StandardScaler (fit on train) → fit
EVALUATE:          PR-AUC (imbalanced) + recall@low-FPR
DEPLOY:            set threshold on decision_function; monitor calibration
```

> A classic use is **high-dimensional, small-n** problems (text with TF-IDF, gene panels) where a linear SVM is both fast and strong.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is a support vector?
2. **Understand:** why does maximizing the margin help generalization?
3. **Calculate:** for the Section 10 SVM, compute `f` and the class at `x = 2.5`.
4. **Apply:** decide if SVM is a good fit for a 1M-row unscaled dataset.
5. **Debug:** train A UC = 0.99, test = 0.70 on an RBF SVM — diagnose and fix.
6. **Experiment:** run Experiment B (scaling) and slide C across classes.
7. **Build:** text-classification mini-project: TF-IDF → LinearSVC → per-class F1 → compare vs Naive Bayes → justify.
8. **Explain:** explain SVM's "widest street" to a friend in 60 seconds.

---

## 28. Interview

### Beginner
- **What is a support vector?** A training point lying exactly on the margin edge; only these define the boundary (`α > 0`).
- **What is the margin?** The distance between the two margin edges; SVM maximizes it (`2/‖w‖`).
- **Linear vs RBF separation?** Linear can't split e.g. circles-in-square; RBF lifts data to a space where a plane works and projects a curved boundary back.
- **What does C do?** Balances wide-margin vs violations: larger C = harder separation, smaller C = wider, softer margin.
- **Why scale features for SVM?** Kernels are distance-based; scale differences make one feature dominate.

### Intermediate
- **Why is the objective convex, and why does that matter?** Convex QP → one global optimum; SMO converges there reliably — no local-minima traps like neural nets.
- **What's the kernel trick exactly?** The dual objective and decision function use only inner products `k(xᵢ,xⱼ)`; we compute those directly, never forming the high-dimensional `φ(x)`.
- **Hard vs soft margin?** Hard requires perfect separation (`C=∞`); soft allows `ξ>0` violations at a price `C` — the practical one.
- **How do you multiclass with SVM?** One-vs-rest (sklearn default) or one-vs-one.
- **How do SVMs output probabilities?** They don't natively; Platt scaling fits a logistic on the decision values.

### Advanced
- **State the KKT conditions — when is `αⱼ = 0`?** `α=0` → point strictly outside the margin (no support); `0<α<C` → exactly on margin (plain SV); `α=C` → inside/violated (slack).
- **Why is the RBF kernel an infinite-dimensional feature space?** `exp(−γ‖x−x'‖²)` expands (Taylor) as an infinite sum of Gaussian feature inner products.
- **Explain SMO in one breath.** Coordinate descent on the dual, updating two `α` at a time while preserving `Σαy = 0`, until all KKT conditions hold.
- **When is LinearSVC better than SVC(kernel='linear')?** For large n / sparse text: LIBLINEAR's coordinate descent is far faster and lighter than LIBSVM's SMO.
- **SVM vs Logistic Regression?** Both linear-discriminative; LR minimizes cross-entropy (probabilities), SVM minimizes hinge loss (max margin). SVM is more robust to boundary noise but gives no native probability.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
margin = 2/‖w‖
primal:    min ½‖w‖² + C·Σξ   s.t. y(w·x + b) ≥ 1 − ξ
hinge:     max(0, 1 − y(w·x + b))
dual:      max Σα − ½ ΣΣ αα·y·y·k(x,x)   s.t. Σαy = 0, 0 ≤ α ≤ C
predict:   ŷ = sign( Σ_{SV} αᵢyᵢ k(xᵢ, x) + b )
RBF:       k = exp(−γ‖x − x'‖²)
```

**Common traps:**
- Margin is `2/‖w‖`, **not** `1/‖w‖`.
- Constraints are at `±1`, **not** 0/1.
- SVM minimizes `‖w‖²` as a whole — only hinge-violating points matter, not every point.
- The kernel maps to a feature space; SVM still finds a hyperplane there — the original-space boundary is curved.

> **Representative pattern question (NOT a past GATE PYQ):** "For the points A(1,+1), B(2,+1), C(4,−1) find the SVM separating hyperplane." → `w = −2/3, b = 5/3`, margin `3` (computed exactly in Section 10).

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the Lagrangian derivation + dual + kernel theory</summary>

### From margin to a minimization

Geometric margin is `2/‖w‖`. Maximizing it ⟺ minimizing `‖w‖` ⟺ minimizing `½‖w‖²` (the ½ makes the derivative clean). So:

```text
min ½‖w‖²   s.t.  yᵢ(w·xᵢ + b) ≥ 1
```

### Lagrangian

```text
L = ½‖w‖² − Σᵢ αᵢ [ yᵢ(w·xᵢ + b) − 1 ],   αᵢ ≥ 0
```

Set gradients to zero:

```text
∂L/∂w = w − Σ αᵢ yᵢ xᵢ = 0   →  w = Σᵢ αᵢ yᵢ xᵢ
∂L/∂b = −Σ αᵢ yᵢ = 0         →  Σᵢ αᵢ yᵢ = 0
```

**Crucial insight:** `w` is a weighted sum of data points — only those with `αᵢ > 0` (support vectors) matter.

### The dual (this is where kernels enter)

Substitute `w = Σαᵢyᵢxᵢ` back into `L`. The objective now depends on the data **only through inner products `xᵢᵀxⱼ`**:

```text
max Σᵢ αᵢ − ½ ΣᵢΣⱼ αᵢαⱼyᵢyⱼ xᵢᵀxⱼ   s.t. Σαᵢyᵢ = 0, αᵢ ≥ 0
```

Replace every `xᵢᵀxⱼ` with a kernel `k(xᵢ,xⱼ)`:

```text
max Σᵢ αᵢ − ½ ΣᵢΣⱼ αᵢαⱼyᵢyⱼ k(xᵢ,xⱼ)
```

Since we never need `w` explicitly — only `k(·,·)` — we get curved boundaries in the original space for the price of inner products. Prediction:

```text
ŷ(x) = sign( Σᵢ αᵢ yᵢ k(xᵢ, x) + b )
```

### Soft margin / C

For overlap, add slack `ξᵢ ≥ 0` and bound `0 ≤ αᵢ ≤ C`:

```text
min ½‖w‖² + C Σ ξᵢ   s.t. yᵢ(w·xᵢ + b) ≥ 1 − ξᵢ
```

`C` is the "budget" of violation the model is willing to tolerate in exchange for a wide margin.

### Why max-margin generalizes

The margin is a *capacity control* (structural risk minimization): a wide margin => a smaller effective hypothesis class => better generalization guarantees. This is the formal reason "wider is safer."

### RBF kernel, concretely

```text
k(x,x') = exp(−γ‖x − x'‖²)
```

A "bump" centered at each training point; the boundary is a superposition of local Gaussian similarities → flexible curved regions.

### Complexity

```text
training (kernel):  O(n²)–O(n³), memory O(n²)
training (linear):  O(n·d), scales to millions
prediction:         O(SV · d)
space:              O(SV · d)
```

If the number of support vectors ≈ n, the model isn't really sparse — consider a linear SVM or a tree ensemble instead.

</details>

---

## 31. Teach Back

Try all four.

> **Explain in 30 seconds:** "SVM draws the widest possible empty street between two classes, decided only by the edge points that touch the street's edge — the support vectors. When the data curves, a kernel lifts it into a higher dimension where a straight street works, then maps the decision back down."

> **Explain to a 12-year-old:** "Two groups of dots on a floor. Draw the fattest line you can through the empty space between them, staying as far from both groups as possible. Only the dots that touch the line's edge actually decide where it goes."

> **Explain in an interview:** add: margin `2/‖w‖`, hinge loss, the dual and kernel trick, hard vs soft margin `C`, KKT, why scaling is required, LinearSVC for scale.

> **Explain the mathematics:** derive the Lagrangian → dual → kernel replacement, and compute margin on the small example.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define SVM.
2. Explain its intuition with the "widest street" story.
3. Write the margin, hinge, primal, and dual formulas.
4. Compute the Section 10 SVM by hand (`w, b`, margin, prediction).
5. Explain what's inside `fit()` for an SVM.
6. Why do only support vectors matter? (KKT link)
7. What does the kernel trick do, and why does it cost nothing extra?
8. Compare SVM with logistic regression and a forest.
9. Choose it for a real problem and defend the choice.
10. State one scenario where you would NOT use SVM (and what instead).

---

## 33. Cheat Sheet

```text
Algorithm : Support Vector Machine (SVC) · Supervised → Classification
Goal      : find the widest separating margin; bend it with kernels
Core      : max margin 2/‖w‖  ⟺  min ½‖w‖² + C·Σ max(0, 1 − y(w·x+b))
Dual      : max Σα − ½ ΣΣ αα·yy·k   s.t. Σαy = 0, 0 ≤ α ≤ C
Predict   : ŷ = sign( Σ_{SV} αy·k(x,x') + b )
Learn     : α multipliers, b, support-vector set (w explicit only for linear)
Tune      : C · kernel · gamma · degree · class_weight
Scaling   : REQUIRED (standardize, fit on train only)
Use when  : small/medium data, high-dim numeric, non-linear, text (linear kernel)
Avoid when: millions of rows, need calibrated probs fast, unscaled/messy data, heavy overlap
Related   : Logistic Regression · Perceptron · SVR · One-Class SVM
```

---

## 34. What Next?

You just met the non-tree branch of classification.

```text
Random Forest / Extra Trees  → (trees, votes)
SVM (max margin + kernels)   ← you are here
   └── Boosting (sequential, bias-reducing)
        ├── 08. Gradient Boosting (fit residuals of log-loss)
        └── 09. AdaBoost (reweight mistakes, exponential loss)
```

> Next recommended: **08. Gradient Boosting (Classification)** — the accuracy workhorse. Unlike SVM's single optimized boundary, it builds many *small* trees, each correcting the residuals of all the ones before, to drive down **bias**. It answers: "what if we learn from our mistakes, one small tree at a time?"
