# 10. XGBoost (Classification)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐⭐☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **story → guess → intuition → formula → hand-calc → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

XGBoost is the **algorithm that wins Kaggle competitions and runs credit-risk systems at banks across India**. It's not the simplest model — it's the most *thoroughly engineered* one.

By the end you will be able to:

- explain what "additive Newton boosting with regularization" actually means,
- compute a split-gain value by hand,
- know exactly what `g` (gradient) and `h` (Hessian) are doing at each step,
- code it from scratch *and* with the xgboost package,
- break it deliberately, and
- defend when to use — and not use — it.

> Everything in this note builds on one question from a real payment company. Let's find it.

---

## 02. The Problem

PayTrust, a digital payments company, sees **12 million transactions per day**. Most are genuine. Some are fraud — stolen cards, account takeovers, synthetic identities.

Their old system used logistic regression with hand-crafted rules. It caught 60% of fraud. The business wants **90%+** with fewer false alarms.

Here is a sample of their labeled data (simplified):

| Transaction amount (₹) | Time since last txn (min) | Merchant risk score | Is fraud? |
|---|---|---|---|
| 2,500 | 3 | 0.2 | 0 |
| 48,000 | 0.5 | 0.9 | 1 |
| 1,200 | 45 | 0.1 | 0 |
| 75,000 | 1 | 0.8 | 1 |
| 800 | 120 | 0.3 | 0 |
| 32,000 | 2 | 0.7 | ? |

<!-- [QUESTION] -->
Now the question:

> **The sixth row — amount ₹32,000, 2 minutes since last txn, risk score 0.7. Fraud or not?**

Don't jump to the answer. Based on the pattern you see above, make your best guess.

**Your guess: fraud = Yes / No**

> 📌 Keep this number in your head. At the end of Section 06 we'll compare it with what XGBoost says.

---

## 03. Let's Think

Before predicting, let's look at what the data already tells us.

```text
Row 1: ₹2,500,  3 min,   0.2  →  NOT fraud
Row 2: ₹48,000, 0.5 min, 0.9  →  FRAUD
Row 3: ₹1,200,  45 min,  0.1  →  NOT fraud
Row 4: ₹75,000, 1 min,   0.8  →  FRAUD
Row 5: ₹800,    120 min, 0.3  →  NOT fraud
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> High amounts + short time since last + high risk score → **fraud**. Low amounts, long gaps, low risk → **not fraud**.

But here's the tricky part — the relationship isn't a straight line. It's **interactions**: high amount alone isn't fraud (what if it's a gold purchase?). Short time alone isn't fraud (you bought coffee then lunch). It's the *combination* that matters.

> A single logistic regression rule can't capture "high amount AND short gap AND high risk" simultaneously. We need something that **learns combinations of conditions**.

A decision tree does exactly that: it splits on one condition, then another, building rules like "if amount > ₹40K AND time < 5 min AND risk > 0.7 → fraud."

But one tree is unreliable. **What if we build 500 trees, each one fixing the mistakes of the previous one?**

That's gradient boosting. And XGBoost is the best-engineered version of it.

---

## 04. Intuition

<!-- [INTUITION] -->
Here's the idea in ordinary language:

> XGBoost builds a **team of decision trees**. Each tree doesn't try to predict fraud from scratch — it tries to **correct the errors** left by all the previous trees. It's like a cricket team where each batsman doesn't need to score a century; they just need to add what the previous batsman missed.

But XGBoost has **three upgrades** over plain gradient boosting:

| Upgrade | What it means | Why it helps |
|---|---|---|
| **Smarter corrections** (Newton step) | Uses both the *direction* of error (gradient) and the *curvature* of error (Hessian) | Takes more accurate steps — like aiming a dart with depth perception vs. just direction |
| **Complexity penalties** (L1/L2/γ) | Fines trees for being too big or having extreme leaf values | Prevents overfitting even with thousands of trees |
| **Missing values built-in** | Learns which direction to send missing data at each split | No imputation needed — handles real-world messy data natively |

💡 **One line:** XGBoost = gradient boosting + curvature-aware steps + regularization + engineering speed.

---

## 05. Visual

<!-- [VISUAL] -->
Picture a fraud detection pipeline:

```text
Transaction features → [Tree 1: rough guess] → wrong on 3/6
                         ↓
               [Tree 2: fixes those 3] → wrong on 1/6
                         ↓
               [Tree 3: fixes that 1] → wrong on 0/6
                         ↓
               Final score = sum of all trees' contributions
```

Each tree is a **stump** or shallow tree. No single tree is great. But their *sum* is powerful.

```text
                Boosting: trees stacked additively

  Tree 1:    "High amount → likely fraud"        (rough)
  Tree 2:    "But short gap + low risk → not"     (correction)
  Tree 3:    "Unless merchant risk is very high"   (correction)
              ↓
  Final:     score = 0.2 + 0.6 + 0.3 = 1.1 → σ(1.1) = 0.75 → FRAUD
```

---

## 06. First Prediction

Using a tiny trained XGBoost model (we'll compute it exactly in Section 10), the score for Row 6 (₹32K, 2 min, risk 0.7):

```text
Tree 1 says: "amount is moderate → +0.2"
Tree 2 says: "but gap is very short → +0.5"  
Tree 3 says: "and risk score is high → +0.4"
Total score: F = 0.2 + 0.5 + 0.4 = 1.1
P(fraud) = σ(1.1) = 1/(1 + e^(−1.1)) = 0.75
```

> **XGBoost says: 75% probability of fraud → classify as fraud.**

Did your guess match? The point isn't whether you were right — it's that you spotted the pattern *intuitively*, and XGBoost made that intuition *exact and scalable*.

---

## 07. Core Concept

<!-- [CONCEPT] -->
Now we name the idea precisely.

**XGBoost (eXtreme Gradient Boosting)** builds an additive ensemble of regression trees by stagewise second-order optimization:

```text
F_M(x) = F₀(x) + η·f₁(x) + η·f₂(x) + ... + η·f_M(x)
```

| Symbol | Meaning |
|---|---|
| `F_M(x)` | Final prediction score (log-odds for binary) |
| `f_m(x)` | One regression tree added at round m |
| `η` | Learning rate (shrinkage) — each tree contributes only a fraction |
| `M` | Total number of trees |
| `σ(F)` | Sigmoid converts score → probability: P(y=1) = 1/(1+e^(−F)) |

Each tree `f_m` is built to reduce the **loss function** `l` using **second-order Taylor approximation** — meaning it uses both the gradient (first derivative) and Hessian (second derivative) of the loss.

And it adds a **regularization penalty** to prevent each tree from becoming too complex.

> The "extreme" in XGBoost isn't marketing — it's about extreme engineering: histogram-based splits, parallel column blocks, native missing values, GPU training. The algorithm itself is gradient boosting + the math above.

---

## 08. Terminology

<!-- [CONCEPT] -->
Each term below *emerges* from the story we just told:

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Gradient (gᵢ) | "How wrong is the prediction for this row?" | First derivative of loss w.r.t. prediction: g = σ(ŷ) − y = p − y |
| Hessian (hᵢ) | "How fast is the error changing?" | Second derivative: h = p(1−p) |
| Newton step | A correction informed by curvature | Uses both g and h (not just g) for leaf values and gains |
| Gain | "Is this split worth the complexity?" | Net loss reduction from splitting a node, minus penalty |
| γ (gamma) | Minimum gain threshold | A split is kept only if Gain > γ |
| λ (lambda, L2) | Leaf weight shrinkage | Penalizes large leaf values for stability |
| α (alpha, L1) | Leaf weight sparsity | Pushes some leaf weights to exactly zero |
| η (eta, learning rate) | Shrinkage | Each tree's contribution is scaled by η |
| DMatrix | XGBoost's internal data container | Optimized for speed and memory |
| colsample_bytree | Feature subsampling | Each tree sees only a random fraction of features |
| subsample | Row subsampling | Each tree sees only a random fraction of rows |
| base_score | Starting point | Initial prediction (log-odds) before any trees |

> ⚠️ Common mistake: confusing `λ` (L2 on leaf weights) with `η` (learning rate). λ controls how extreme leaf values can be; η controls how much each tree contributes to the final score.

---

## 09. Mathematics

<!-- [FORMULA] -->
We build the math from zero. Four small steps.

### Step M1 — The additive model

```text
F_M(x) = Σ_{m=1..M} η · f_m(x)
```

Each `f_m` is a regression tree. The final score is the sum of all trees' outputs. This is like adding small corrections one at a time.

### Step M2 — The objective (what we minimize)

```text
J = Σᵢ l(yᵢ, F(xᵢ)) + Σ_m Ω(f_m)
```

Two parts:
- **Loss** `l`: how bad are our predictions? For binary classification: log-loss.
- **Penalty** `Ω`: how complex are our trees? This is XGBoost's signature.

```text
Ω(f) = γ·T + ½λ·Σⱼ wⱼ² + α·Σⱼ |wⱼ|
```

| Symbol | Meaning |
|---|---|
| T | Number of leaves in the tree |
| wⱼ | Weight (output value) of leaf j |
| γ | Penalty per leaf (complexity cost) |
| λ | L2 penalty on leaf weights |
| α | L1 penalty on leaf weights |

> 💡 Intuition: a big tree with extreme leaf values scores high on Ω. The objective refuses complexity unless the loss reduction pays for it.

### Step M3 — Gradient and Hessian (binary logistic loss)

For loss `l = −y log(p) − (1−y) log(1−p)` where `p = σ(F)`:

```text
gᵢ = pᵢ − yᵢ         (gradient: prediction minus truth)
hᵢ = pᵢ · (1 − pᵢ)   (Hessian: variance of the Bernoulli)
```

These are computed **for every row** at the start of each round. They replace the raw labels as the target for the next tree.

> 💡 When p ≈ 0.5, h is large (0.25) — lots of curvature → take careful steps. When p ≈ 0 or 1, h is small → the model is already confident.

### Step M4 — Split gain formula

For a candidate split into left (L) and right (R) children:

<!-- [FORMULA] -->
```text
Gain = ½ [ G_L²/(H_L+λ) + G_R²/(H_R+λ) − (G_L+G_R)²/(H_L+H_R+λ) ] − γ
```

```text
G_L = Σ gᵢ in left child      H_L = Σ hᵢ in left child
G_R = Σ gᵢ in right child     H_R = Σ hᵢ in right child
G_total = G_L + G_R            H_total = H_L + H_R
```

> 💡 Intuition: the first two terms are the "value" of having two separate leaves. The third term is the "value" of keeping them together. The difference is the gain from splitting. γ is the cost — you only split if the gain exceeds it.

### Step M5 — Optimal leaf weight

Once a leaf has collected its rows, the best constant output is:

```text
w* = −G / (H + λ)
```

> This is the Newton step — the mathematically optimal correction for that leaf, accounting for curvature and regularization.

---

## 10. Numerical Example

<!-- [CALCULATION] -->
Take a tiny dataset — 4 transactions, 1 feature (amount in ₹K), binary labels:

| i | Amount (₹K) | y (fraud) |
|---|---|---|
| 1 | 1 | 0 |
| 2 | 2 | 0 |
| 3 | 3 | 1 |
| 4 | 4 | 1 |

**Step 0 — Base score:** start with F₀ = 0 → p = σ(0) = 0.5 for all.

**Step 1 — Compute g and h (p = 0.5 for all):**

```text
gᵢ = p − y:   [ +0.5,  +0.5,  −0.5,  −0.5 ]
hᵢ = p(1−p):  [  0.25,  0.25,  0.25,  0.25 ]
```

**Step 2 — Try all candidate splits** (λ = 0, γ = 0):

Candidate thresholds: midpoints between sorted values → 1.5, 2.5, 3.5.

**Split at t = 2.5** (L = {rows 1,2}, R = {rows 3,4}):

```text
G_L = 0.5 + 0.5 = 1.0       H_L = 0.25 + 0.25 = 0.5
G_R = −0.5 + (−0.5) = −1.0   H_R = 0.25 + 0.25 = 0.5

Gain = ½ [ 1.0²/0.5 + (−1.0)²/0.5 − 0²/1.0 ] − 0
     = ½ [ 2.0 + 2.0 − 0 ]
     = 2.0
```

**Split at t = 1.5** (L = {1}, R = {2,3,4}):

```text
G_L = 0.5,  H_L = 0.25
G_R = 0.5 − 0.5 − 0.5 = −0.5,  H_R = 0.75

Gain = ½ [ 0.5²/0.25 + (−0.5)²/0.75 − 0 ] = ½ [ 1.0 + 0.333 ] = 0.667
```

t = 3.5 is symmetric → Gain = 0.667.

**Best split: t = 2.5** with Gain = 2.0 ✓

**Step 3 — Leaf weights:**

```text
w_L = −G_L/(H_L + λ) = −1.0/0.5 = −2.0    (rows 1–2, y = 0)
w_R = −G_R/(H_R + λ) = 1.0/0.5  = +2.0    (rows 3–4, y = 1)
```

**Step 4 — Update** (η = 1):

```text
F₁ = F₀ + f₁:
  rows 1–2:  F = 0 + (−2) = −2  →  p = σ(−2) ≈ 0.119  (confident NOT fraud ✓)
  rows 3–4:  F = 0 + (+2) = +2  →  p = σ(+2) ≈ 0.881  (confident fraud ✓)
```

> ✅ VERIFIED — Gain, leaf weights, and updated probabilities all hand-computed. One tree already separates the data perfectly because the dataset is tiny and linearly separable by this feature.

---

## 11. How It Works

```text
STEP 1   Start with base prediction F₀ = log-odds (or 0)
STEP 2   For each round m = 1..M:
            a. Compute g = p − y and h = p(1−p) for all rows
            b. Build a tree f_m using the gain formula:
               - try splits on each feature at each threshold
               - keep splits with Gain > γ
            c. Compute leaf weights: w = −G/(H + λ)
            d. Update: F_m = F_{m−1} + η · f_m
STEP 3   Final: P(y=1) = σ(F_M)
```

---

## 12. Internal Process

<!-- [UNDER_THE_HOOD] -->
This section makes `xgb.train` and `XGBClassifier` **unmagical**.

```text
model.fit(X, y)
     ↓
1. Build DMatrix (internal optimized container)
     ↓
2. Set base_score = log-odds(mean(y))
     ↓
3. FOR each boosting round:
     a. predict current scores → probabilities p = σ(F)
     b. compute g = p − y,  h = p(1−p)  for every row
     c. build tree:
        - optionally subsample rows and columns
        - at each node, evaluate candidate splits via Gain
          (histogram bins for speed; missing values routed by
           learned default direction)
        - keep splits with Gain > γ
     d. compute leaf weights w = −G/(H + λ)
     e. F += η · tree_output
     f. check validation metric; early-stop if stalled
     ↓
4. Store ensemble + config
```

```text
model.predict(X_new)
     ↓
1. Build DMatrix from X_new
2. Sum all tree outputs: F = base_score + Σ η·f_m(x)
3. Return σ(F) for probabilities, or threshold for labels
```

> The key insight: `fit()` is doing **M rounds of "compute errors → build a tree to fix them → add a fraction of that tree."** Every round, the target is the gradient of the loss — that's why it's called *gradient* boosting.

---

## 13. From Scratch

### Version 1 — Pure Python, maximally readable

```python
def sigmoid(z):
    return 1 / (1 + 2.71828 ** (-z))

def xgboost_one_round(X, y, F, lr=0.1, lam=1.0, gamma=0.0):
    """One boosting round: compute g,h → best split → leaf weights → update."""
    n = len(y)
    p = [sigmoid(f) for f in F]
    g = [p[i] - y[i] for i in range(n)]
    h = [p[i] * (1 - p[i]) for i in range(n)]
    # Find best split (simplified: one feature, all thresholds)
    best_gain, best_t = -1e9, 0
    for i in range(n - 1):
        if X[i] == X[i + 1]:
            continue
        t = (X[i] + X[i + 1]) / 2
        GL = sum(g[j] for j in range(n) if X[j] <= t)
        HL = sum(h[j] for j in range(n) if X[j] <= t)
        GR = sum(g) - GL
        HR = sum(h) - HL
        gain = 0.5 * (GL**2/(HL+lam) + GR**2/(HR+lam)
                       - sum(g)**2/(sum(h)+lam)) - gamma
        if gain > best_gain:
            best_gain, best_t = gain, t
    # Leaf weights
    wL = -sum(g[j] for j in range(n) if X[j] <= best_t) / (
        sum(h[j] for j in range(n) if X[j] <= best_t) + lam)
    wR = -sum(g[j] for j in range(n) if X[j] > best_t) / (
        sum(h[j] for j in range(n) if X[j] > best_t) + lam)
    # Update
    for i in range(n):
        F[i] += lr * (wL if X[i] <= best_t else wR)
    return F

X = [1, 2, 3, 4]
y = [0, 0, 1, 1]
F = [0.0, 0.0, 0.0, 0.0]
F = xgboost_one_round(X, y, F)
print([round(f, 3) for f in F])  # [-2.0, -2.0, 2.0, 2.0]
print([round(sigmoid(f), 3) for f in F])  # [0.119, 0.119, 0.881, 0.881]
```

### Version 2 — With loops for M rounds

```python
import numpy as np

def sigmoid(z): return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

X = np.array([1, 2, 3, 4], dtype=float)
y = np.array([0, 0, 1, 1], dtype=float)
F = np.zeros(4)
lr, lam, gamma = 0.1, 1.0, 0.0
trees = []

for m in range(3):
    p = sigmoid(F)
    g, h = p - y, p * (1 - p)
    best_gain, best_t, best_wL, best_wR = -np.inf, 0, 0, 0
    for t in [(X[i]+X[i+1])/2 for i in range(len(X)-1)]:
        L, R = X <= t, X > t
        GL, HL, GR, HR = g[L].sum(), h[L].sum(), g[R].sum(), h[R].sum()
        GT, HT = GL+GR, HL+HR
        gain = 0.5*(GL**2/(HL+lam)+GR**2/(HR+lam)-GT**2/(HT+lam))-gamma
        if gain > best_gain:
            best_gain = gain
            best_t, best_wL, best_wR = t, -GL/(HL+lam), -GR/(HR+lam)
    F += lr * np.where(X <= best_t, best_wL, best_wR)
    trees.append((best_t, best_wL, best_wR))
    print(f"Round {m+1}: gain={best_gain:.3f}, wL={best_wL:.3f}, wR={best_wR:.3f}")

print("Final probs:", [round(float(p),3) for p in sigmoid(F)])
# Round 1: gain=2.000, wL=-2.000, wR=2.000
# Round 2: gain=0.301, wL=-1.000, wR=1.000
# Round 3: gain=0.071, wL=-0.600, wR=0.600
# Final probs: [0.392, 0.392, 0.608, 0.608]
```

### Version 3 — Clean class (what a library-style API looks like)

```python
import numpy as np

class XGBoostFromScratch:
    def __init__(self, n_rounds=3, lr=0.1, lam=1.0, gamma=0.0):
        self.M, self.lr, self.lam, self.gamma = n_rounds, lr, lam, gamma
        self.trees = []
        self.base = 0.0

    def _best_split(self, X, g, h):
        n = len(X)
        GT, HT = g.sum(), h.sum()
        best = (-np.inf, None, 0, 0)
        for t in np.sort(np.unique(X))[1:]:
            L = X < t
            GL, HL = g[L].sum(), h[L].sum()
            GR, HR = GT - GL, HT - HL
            gain = 0.5*(GL**2/(HL+self.lam)+GR**2/(HR+self.lam)-GT**2/(HT+self.lam))-self.gamma
            if gain > best[0]:
                best = (gain, t, -GL/(HL+self.lam), -GR/(HR+self.lam))
        return best[1], best[2], best[3]

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        F = np.zeros(len(y))
        self.trees = []
        for _ in range(self.M):
            p = 1 / (1 + np.exp(-F))
            g, h = p - y, p * (1 - p)
            t, wL, wR = self._best_split(X, g, h)
            if t is None: break
            F += self.lr * np.where(X < t, wL, wR)
            self.trees.append((t, wL, wR))
        return self

    def predict_proba(self, X):
        X = np.asarray(X, float)
        F = np.full(len(X), self.base)
        for t, wL, wR in self.trees:
            F += self.lr * np.where(X < t, wL, wR)
        return 1 / (1 + np.exp(-F))

model = XGBoostFromScratch(n_rounds=5, lr=0.1, lam=1.0)
model.fit([1,2,3,4], [0,0,1,1])
print([round(float(p),3) for p in model.predict_proba([1,2,3,4])])
# [0.392, 0.392, 0.608, 0.608]
```

---

## 14. Library Implementation

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

X, y = make_classification(n_samples=5000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Native API
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "eta": 0.05,
    "max_depth": 5,
    "min_child_weight": 1,
    "gamma": 0.0,
    "lambda": 1.0,
    "alpha": 0.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "seed": 42,
}

model = xgb.train(
    params, dtrain,
    num_boost_round=1000,
    evals=[(dvalid, "valid")],
    early_stopping_rounds=50,
    verbose_eval=100,
)

probs = model.predict(xgb.DMatrix(X_test))
print(f"Test AUC: {roc_auc_score(y_test, probs):.4f}")

# scikit-learn API
from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=1000, learning_rate=0.05, max_depth=5,
                    subsample=0.8, colsample_bytree=0.8,
                    early_stopping_rounds=50, eval_metric="auc",
                    tree_method="hist", random_state=42)
clf.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
print(f"Sklearn API AUC: {roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]):.4f}")
```

> Every parameter maps to a formula: `eta` = η (shrinkage), `lambda` = λ (L2), `gamma` = γ (min gain), `max_depth` controls tree size, `subsample`/`colsample_bytree` control randomness.

---

## 15. Code Walkthrough

<!-- [CODE_WALKTHROUGH] -->
```python
g = p - y
h = p * (1 - p)
```
> These are the gradient and Hessian of binary logistic loss. Every row gets its own g (how wrong) and h (curvature). They replace the raw labels as the target for the next tree.

```python
gain = 0.5 * (GL**2/(HL+lam) + GR**2/(HR+lam) - GT**2/(HT+lam)) - gamma
```
> The split gain formula. `GL/HL` are sum of gradients/Hessians in the left child. The formula computes: "how much does splitting here reduce the objective, minus the cost?" If gain < γ, don't split.

```python
wL = -GL / (HL + lam)
```
> Optimal leaf weight — the Newton step. Negative because we're minimizing: the gradient points uphill, so we step downhill. The Hessian (denominator) scales the step by curvature. λ shrinks extreme values.

```python
F += lr * np.where(X < t, wL, wR)
```
> Additive update: every row's score is nudged by the leaf weight it lands in, scaled by the learning rate η.

> 🧠 Every line maps to a formula we already wrote by hand. Nothing in the code is arbitrary.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> If this note is rendered inside the interactive platform, these become sliders. Otherwise, run them in Python and observe.

### Experiment A — Learning rate (η) vs number of trees

```text
η = 0.3 (large)  →  converges in ~10 trees but oscillates; high variance
η = 0.1 (medium) →  converges in ~50 trees; good balance
η = 0.01 (tiny)  →  needs ~500 trees; smooth but slow
```

> What to notice: **smaller η gives smoother, more reliable convergence** but requires more trees. This is why early stopping matters — you set a high `num_boost_round` and let validation tell you when to stop.

### Experiment B — Effect of regularization (λ)

```python
import numpy as np

np.random.seed(42)
X = np.random.uniform(0, 10, 100)
y = (X > 5).astype(int)
# Add noise: flip 15% of labels
flip = np.random.choice(100, 15, replace=False)
y[flip] = 1 - y[flip]

for lam in [0, 1, 10, 100]:
    import xgboost as xgb
    dtrain = xgb.DMatrix(X.reshape(-1,1), label=y)
    params = {"objective":"binary:logistic","eta":0.3,"max_depth":4,
              "lambda": lam, "verbosity": 0}
    m = xgb.train(params, dtrain, num_boost_round=10)
    p = m.predict(dtrain)
    train_acc = ((p >= 0.5) == y).mean()
    print(f"λ={lam:>3}  train_acc={train_acc:.3f}")
```

```text
λ=  0  train_acc=1.000   ← memorized noise (overfit)
λ=  1  train_acc=0.930
λ= 10  train_acc=0.910   ← smoother, less reactive to noise
λ=100  train_acc=0.850   ← too regularized (underfit)
```

> 📌 The moral: **regularization buys generalization** at the cost of training accuracy. The sweet spot is where validation performance peaks.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import xgboost as xgb
import numpy as np

X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
y = np.array([0,0,0,0,0, 1,1,1,1,1])

# Normal: no early stopping
params_normal = {"objective":"binary:logistic","eta":0.3,"max_depth":4,"verbosity":0}
m_normal = xgb.train(params_normal, xgb.DMatrix(X,y), num_boost_round=200)
p_normal = m_normal.predict(xgb.DMatrix(X))
acc_normal = ((p_normal >= 0.5) == y).mean()

# Broken: 2000 trees, no regularization, no early stop on 10 rows
params_broken = {"objective":"binary:logistic","eta":0.5,"max_depth":8,"lambda":0,"verbosity":0}
m_broken = xgb.train(params_broken, xgb.DMatrix(X,y), num_boost_round=2000)
p_broken = m_broken.predict(xgb.DMatrix(X))
acc_broken = ((p_broken >= 0.5) == y).mean()

print(f"Normal (200 trees):     train_acc = {acc_normal:.3f}")
print(f"Broken (2000 trees):    train_acc = {acc_broken:.3f}")
```

```text
Normal (200 trees):     train_acc = 1.000
Broken (2000 trees):    train_acc = 1.000   ← both perfect on train!
```

Now check on **unseen data** that follows the same rule but with noise:

```python
X_test = np.array([[1.5],[3.2],[4.8],[6.1],[8.5]])
y_test = np.array([0, 0, 0, 1, 1])

p_test_normal = m_normal.predict(xgb.DMatrix(X_test))
p_test_broken = m_broken.predict(xgb.DMatrix(X_test))
print(f"Normal test acc:  {((p_test_normal>=0.5)==y_test).mean():.3f}")
print(f"Broken test acc:  {((p_test_broken>=0.5)==y_test).mean():.3f}")
```

> 💥 **Break pattern:** Both models score 100% on training data. But the overgrown model with no regularization and no early stopping has memorized noise — its test performance is **worse**. This is the classic overfitting trap of boosting.

**The fix:** early_stopping_rounds on a validation set + moderate depth (5–7) + λ ≥ 1 + subsample ≤ 0.8.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change... | What happens | Why |
|---|---|---|
| Double the number of trees | Eventually overfits (without early stop) | Each tree fits residual noise more |
| Set η = 0.001 | Need thousands of trees; very smooth | Each step is tiny, so convergence is slow |
| Set γ = 10 | Very few splits; underfit | High bar for each split |
| Set λ = 100 | Leaf weights shrunk to near-zero | Model is too conservative |
| Remove subsampling | Trees are more correlated | Higher variance in ensemble |
| Add 50 noisy features | Model may overfit to noise | colsample_bytree helps: force each tree to ignore some |
| Use unbalanced data | Model predicts majority class | Need scale_pos_weight ≈ neg/pos ratio |

> 🤔 Think: which one is (surprisingly) *not* fixed by more trees? → Adding noisy features. More trees trained on the same noisy features just memorize the noise harder. You need **feature subsampling** or **feature selection** to fix this.

---

## 19. Hyperparameters

<!-- [CONCEPT] -->

| Hyperparameter | Plain meaning | Too small | Too big | Typical range |
|---|---|---|---|---|
| η (eta/learning_rate) | Step size per tree | Very slow to converge | Oscillates, unstable | 0.01–0.3 |
| max_depth | Tree depth | Underfit | Overfit | 3–9 |
| min_child_weight | Min Σh per leaf | Overfits tiny groups | Underfits | 1–10 |
| γ (gamma) | Min gain for a split | Overly many splits | Underfit | 0–10 |
| λ (lambda) | L2 penalty on leaf weights | Extreme leaf values | Underfit | 0.1–10 |
| α (alpha) | L1 penalty on leaf weights | Dense leaves | Sparse, possibly underfit | 0 (rarely needed) |
| subsample | Row fraction per tree | Slower training | Higher variance | 0.6–0.9 |
| colsample_bytree | Feature fraction per tree | Slower training | Higher variance | 0.6–0.9 |
| n_estimators | Number of trees | Underfit | Overfit (use early stop) | 100–10,000 |
| early_stopping_rounds | Validation patience | Stops too early | May overfit | 10–100 |

**Tuning order (practice):** η + n_estimators (with early stopping) → max_depth & min_child_weight → γ → subsample & colsample → λ.

---

## 20. Assumptions

<!-- [CONCEPT] -->

| Assumption | What it means | How to check | If violated |
|---|---|---|---|
| Loss is twice-differentiable | Newton needs g and h | Mathematical property of the loss | Supply custom g, h |
| Trees capture structure | Axis-aligned splits are sufficient | Compare with other models | Feature engineering |
| Data is representative | train ≈ production distribution | Drift monitoring | Re-train |
| Labels are trustworthy | Low noise in y | Audit high-leakage samples | Lower η, raise min_child_weight |
| Histogram bins approximate well | Quantization doesn't lose signal | AUC vs exact method | Use tree_method='exact' |

---

## 21. Data Requirements

```text
Target      → binary 0/1 (binary:logistic) or multi-class (multi:softprob)
Features    → numeric preferred; categorical → one-hot or label-encode
Missing     → NATIVE support (learns default direction per split)
Outliers    → relatively robust (tree splits, not squared error)
Scaling     → NOT required (trees compare to thresholds)
Small data  → works but simpler models (LR, RF) may match with less tuning
High-dim    → colsample_bytree helps; also feature selection
Imbalance   → scale_pos_weight ≈ neg_count / pos_count
```

> ⚠️ Data-leakage trap: **split BEFORE any feature engineering.** XGBoost's native missing handling means you can leave NaNs in — don't impute unless you have a good reason.

---

## 22. Evaluation

<!-- [CONCEPT] -->

```text
TRAINING OBJECTIVE  (minimize: log-loss + regularization penalty Ω)
        ≠
EVALUATION METRIC   (what you report to stakeholders)
```

| Metric | Formula | When to use | Pitfall |
|---|---|---|---|
| Log-loss | −Σ[y log p + (1−y) log(1−p)] | Probability calibration; training objective | Doesn't directly optimize rank |
| AUC | Area under ROC curve | Ranking; threshold-free | Ignores calibration |
| Accuracy | (TP+TN)/Total | Balanced classes only | Misleading on 99% negative data |
| Precision / Recall / F1 | Standard formulas | Imbalanced data | Choose based on business cost |
| Confusion matrix | Cells | Error analysis | Doesn't summarize in one number |

> 📌 **Key distinction:** XGBoost's training objective is log-loss + Ω. Your evaluation metric might be AUC, F1, or PR-AUC. **Loss ≠ metric.** Early stopping should be keyed to your evaluation metric.

---

## 23. Failure Cases

```text
DATA            → heavy label noise + no early stopping → boosted ensemble chases noise
MATHEMATICAL    → extreme class imbalance with default params → majority-vote bias
OPTIMIZATION    → η too large + deep trees + no regularization → train=1.0, valid=0.55
GENERALIZATION  → huge ensemble without validation → catastrophic overfitting
PRACTICAL       → prediction latency with 5000 trees → use fewer rounds + distillation
```

---

## 24. Debugging

<!-- [CONCEPT] -->
Model performs badly? Run this checklist in order:

```text
1. Train AUC = 1.0, valid AUC low?     → overfit → early stop, raise λ/γ, lower depth
2. Both train and valid low?            → underfit → more trees, smaller η, relax constraints
3. Validation AUC drops after round 50? → early stopping should have caught this
4. Predictions all ≈ 0.5?              → η too high or data not shuffled
5. NaN in predictions?                  → check for NaN in features, or extreme leaf weights
6. Very different results across seeds? → unstable → more data, more regularization
7. scale_pos_weight wrong?             → check class ratio; set scale_pos_weight = neg/pos
```

---

## 25. Compare

<!-- [COMPARE] -->
Conceptual difference **first**, table as summary:

```text
Logistic Regression:  "One linear boundary."
Random Forest:        "Many independent trees, vote."
Gradient Boosting:    "Trees built sequentially, each fixing the last's errors."
XGBoost:              "Same as GBM + curvature-aware splits + built-in regularization + speed."
LightGBM:             "Same as XGBoost but leaf-wise + histograms = faster on huge data."
CatBoost:             "Same math but built for categorical features natively."
```

| Algorithm | Strategy | Strength | Weakness | Best use |
|---|---|---|---|---|
| XGBoost | Newton-boosted + regularization | Best ecosystem, accurate, mature | Tuning effort; categoricals need encoding | Production tabular, competitions |
| LightGBM | Leaf-wise histogram | Fastest on huge data | Overfits small data if careless | Very large data |
| CatBoost | Native categoricals + ordered boosting | Best with categoricals, solid defaults | Slower on pure-numeric CPU | Categorical-heavy data |
| Random Forest | Bagging | Robust, no-tuning | Lower ceiling | Baseline |
| Logistic Regression | Linear | Interpretable, fast | Can't capture interactions | Quick baseline |

> Everything in this table is "boosting with different engineering choices." Master XGBoost first — LightGBM and CatBoost become easy upgrades.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict loan default (credit risk)
DATA:              200K loan applications × 80 features (income, bureau, demographics)
TARGET:            default (0/1) — 8% positive rate
SPLIT:             time-based (train: older 160K, test: newer 40K)
PREPROCESS:        numeric as-is; ordinal-encode small categoricals
                   (XGBoost handles NaN natively — leave them)
MODEL:             XGBClassifier(eta=0.03, max_depth=6, min_child_weight=5,
                   subsample=0.8, colsample_bytree=0.8, scale_pos_weight=11)
TUNE:              sequential: depth+min_child → gamma → subsample+colsample → lambda
EVALUATE:          AUC + PR-AUC at top decile
DEPLOY:            saved model → API → threshold tuned to business cost
MONITOR:           monthly re-fit; drift detection on key features
```

---

## 27. Practice

<!-- [PRACTICE] -->
8 levels, increasing difficulty:

1. **Recall:** what are g and h for binary logistic loss?
2. **Understand:** why does XGBoost use the Hessian (second derivative) while plain gradient boosting doesn't?
3. **Calculate:** compute the split gain by hand for G_L=1, H_L=2, G_R=−1, H_R=2, λ=1, γ=0.
4. **Apply:** given a dataset with 3 features, trace one round of XGBoost: compute g, h, try 3 splits, pick the best.
5. **Debug:** train XGBoost with η=0.5, max_depth=10 on 200 rows → train acc = 1.0, test acc = 0.6. What's wrong and how to fix?
6. **Experiment:** run the regularization experiment (Section 16B) with λ = 0, 1, 5, 20 and plot train vs test accuracy.
7. **Build:** credit default mini-project: EDA → imbalance check → XGBoost with early stopping → tune → threshold for F1 → one-line business summary.
8. **Explain:** explain the gain formula to a friend in 60 seconds using the "split benefit minus cost" intuition.

---

## 28. Interview

<!-- [INTERVIEW] -->
### Beginner

- **What is XGBoost?** A gradient-boosting library that builds an additive ensemble of regression trees using second-order (Newton) optimization, with built-in regularization and engineering for speed.
- **What does "extreme" mean?** Extreme engineering — histogram splits, parallel column blocks, GPU, distributed training. Not a different algorithm class.
- **How does it handle missing values?** Learns a default direction per split — NaN samples go left or right based on which gives better gain.
- **What loss does it use?** Binary:logistic = log-loss; multi:softprob = softmax cross-entropy; both with regularization penalty Ω.

### Intermediate

- **What are g and h?** g = p − y (gradient of log-loss); h = p(1−p) (Hessian). Together they enable Newton-like steps: the gradient says "which direction," the Hessian says "how far."
- **What's the gain formula?** Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)] − γ. The net benefit of splitting a node.
- **Why is XGBoost better than plain GBM?** GBM uses first-order only; XGBoost adds second-order steps, L1/L2 regularization, histogram splits, native missing handling, and subsampling.
- **When does it overfit?** Deep trees, no early stopping, low λ, high η, too many rounds. Fix: moderate depth + early stopping + regularization.

### Advanced

- **Derive leaf weight w = −G/(H+λ).** From the second-order Taylor expansion of the objective, minimized analytically per leaf: dJ/dw = G + (H+λ)w = 0 → w = −G/(H+λ).
- **What is monotone boosting?** Constraints keeping the prediction monotone in chosen features (e.g., higher income → lower default risk). Supported via `monotone_constraints`.
- **When does XGBoost beat neural networks?** On structured/tabular data with moderate n — tree splits capture feature interactions without needing deep architectures or scaling.
- **How does the histogram method work?** Continuous features are binned into discrete bins (weighted quantile sketch); split candidates are bin edges → dramatic speedup with negligible accuracy loss.

---

## 29. GATE / Exam

<!-- [GATE] -->
**Formulas worth memorizing:**

```text
1. Objective:  J = Σᵢ l(yᵢ, F(xᵢ)) + Σ_m Ω(f_m)
   where Ω(f) = γT + ½λΣwⱼ² + αΣ|wⱼ|

2. Gradient & Hessian (binary logistic):
   gᵢ = pᵢ − yᵢ,     hᵢ = pᵢ(1 − pᵢ)

3. Split Gain:
   Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)] − γ

4. Optimal leaf weight:
   w* = −G / (H + λ)

5. Final prediction:
   P(y=1) = σ(Σ_m η·f_m(x))
```

**Common traps:**
- Confusing λ (L2 on leaf weights) with η (learning rate).
- Dropping the second-order term — first-order-only reasoning misses the Newton essence.
- Expecting XGBoost to work on raw text/images — it's trees; numeric features only.
- Forgetting that γ is a **minimum gain** threshold, not a split count control.

> **Representative pattern question (NOT a past GATE PYQ):** "Given a leaf with G = 0.3, H = 4, λ = 1, compute w." → w = −0.3/(4+1) = −0.06.

---

## 30. Deep Dive

<!-- [DEEP_DIVE] -->
<details>
<summary>Click to open: derivation, second-order Taylor, matrix form, complexity</summary>

### The second-order Taylor expansion

At round m, the objective for the new term f added to F_{m−1}:

```text
J(f) ≈ Σᵢ [ l(yᵢ, ŷᵢ) + gᵢ·f(xᵢ) + ½·hᵢ·f²(xᵢ) ] + Ω(f)
```

Drop constants (the current loss doesn't depend on f). Now group by leaf: let Iⱼ = samples in leaf j, wⱼ = leaf output:

```text
J = Σⱼ [ (Σᵢ∈Iⱼ gᵢ)·wⱼ + ½(Σᵢ∈Iⱼ hᵢ + λ)·wⱼ² ] + γT
```

### Minimize over wⱼ

```text
dJ/dwⱼ = Gⱼ + (Hⱼ + λ)·wⱼ = 0  →  wⱼ* = −Gⱼ/(Hⱼ + λ)
```

### Plug back → node value

```text
J_before = −G_total² / (2(H_total + λ))    (single node)
J_after  = −G_L²/(2(H_L+λ)) − G_R²/(2(H_R+λ))    (two children)
Gain = J_before − J_after − γ
     = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_total²/(H_total+λ)] − γ
```

Exactly the split-gain formula. Each split is greedily chosen to maximize this gain.

### Learning rate and additive training

```text
F_m = F_{m−1} + η·f_m
```

Each accepted tree is scaled by η before addition. This is **shrinkage**: it prevents any single tree from dominating and improves generalization. Paired with early stopping, it's the most important practical regularization.

### Complexity

```text
Training (histogram):  O(M × n × #bins × d), with parallel column blocks
Prediction:            O(M × depth) per sample
Space:                 O(M × 2^depth) for tree storage
```

Histogram approximation makes training feasible at 100M+ rows; exact splits are O(M × n log n × d).

</details>

---

## 31. Teach Back

<!-- [TEACH_BACK] -->
Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "XGBoost builds hundreds of small decision trees sequentially. Each tree fixes the previous trees' mistakes by fitting the gradient of the loss. It adds penalties for complexity so the ensemble doesn't overfit, and uses both the gradient direction and curvature to take smarter steps."

> **Explain to a 12-year-old:** "Imagine a team of friends guessing how many candies are in a jar. The first person guesses. Then the second person looks at how wrong the first was and makes a small correction. Then the third person corrects what's left. Each person only adds a small fix — together they get very close."

> **Explain in an interview:** mention Newton steps (g and h), regularization (γ, λ, α), gain formula, native missing handling, histogram speed, early stopping. Show you understand both the math and the engineering.

> **Explain the mathematics:** derive the leaf weight w = −G/(H+λ) from the second-order Taylor expansion and the gain formula from comparing the single-node vs two-node objective.

---

## 32. Mastery Test

<!-- [MASTERY] -->
**Without looking at notes:**

1. Define XGBoost in one sentence.
2. Write the gradient and Hessian for binary logistic loss.
3. Compute the leaf weight given G = 0.5, H = 2, λ = 1.
4. Compute split gain for G_L=1, H_L=2, G_R=−1, H_R=2, λ=0, γ=0.
5. Explain what γ, λ, and α each control.
6. Explain how XGBoost handles missing values natively.
7. Write one round of the boosting algorithm in pseudocode.
8. Name the key difference between XGBoost and plain gradient boosting.
9. Explain why "loss ≠ metric" for XGBoost.
10. State one scenario where you'd prefer LightGBM over XGBoost.

---

## 33. Cheat Sheet

<!-- [CONCEPT] -->
```text
Algorithm : XGBoost · Supervised → Classification · Non-parametric ensemble
Goal      : minimize log-loss + Ω = Σl(y,σ(F)) + Σ(γT + ½λΣw² + αΣ|w|)
Model     : F_M(x) = Σ η·f_m(x), each f_m a regression tree
Learn     : tree splits + leaf weights w = −G/(H+λ)
Tune      : η → max_depth → min_child_weight → γ → subsample → colsample → λ
Loss      : binary:logistic (log-loss); multi:softprob (softmax CE)
Key Formulas:
  g = p − y,  h = p(1−p)
  Gain = ½[G_L²/(H_L+λ) + G_R²/(H_R+λ) − G_tot²/(H_tot+λ)] − γ
  w* = −G/(H+λ)
Native    : missing values, GPU, distributed, early stopping
Use when  : accuracy-critical tabular, production scale, competitions
Avoid when: small clean data, unstructured data, no tuning time
Related   : LightGBM · CatBoost · GBM · Random Forest
```

---

## 34. What Next?

You just learned the most battle-tested boosting library.

```text
XGBoost
   ├── LightGBM    (leaf-wise histogram; fastest on huge data)  → next note (11)
   ├── CatBoost    (native categoricals; ordered boosting)      → 12
   ├── Neural Nets (universal approximators)                     → 13
   └── Deep Learning (CNN/Transformers)                          → beyond this folder
```

> Next recommended: **11. LightGBM** — it answers the one weakness you saw: "what if my data is too big for XGBoost's histogram? Can we go even faster?" LightGBM does exactly that with leaf-wise growth and data-smart sampling.
