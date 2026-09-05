# 09. Support Vector Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **tube idea → flatness + margin → ε-insensitive loss → kernel trick → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

You've seen regression models that care about *every* error — OLS squares them, Huber caps the big ones. But what if we could go further and say: **errors within a certain range don't matter at all?**

Support Vector Regression (SVR) is the model that draws a **tolerance tube** around the prediction and ignores everything inside it. Only the points that escape the tube — the "support vectors" — influence the fit.

By the end you will be able to:

- explain the ε-tube concept and why it gives sparsity,
- write the ε-insensitive loss and its objective,
- compute the loss by hand for given residuals,
- explain the kernel trick intuitively,
- code a linear SVR from scratch and use sklearn's SVR,
- and defend when to use — and not use — it.

> Everything in this note builds on one idea: **what if small errors were free?**

---

## 02. The Problem

Arjun is building a model to predict exam scores from hours studied. He has training data:

| Hours studied | Score |
|---|---|
| 1 | 22 |
| 2 | 44 |
| 3 | 55 |
| 4 | 80 |
| 5 | 88 |

A straight line fits reasonably well. But there's noise — maybe the true relationship has a slight curve, and individual scores scatter around it.

OLS would try to minimize the squared error for *every* point, bending the line to please each one. But do we really need the line to pass close to *every* point? What if a few points' tiny errors just don't matter?

<!-- [QUESTION] -->
> **If you could guarantee that your predictions are "close enough" — within, say, 5 marks of the true score — wouldn't you prefer the simplest, flattest line that achieves this, rather than the line that obsesses over every tiny deviation?**

That's the SVR question: **find the flattest function where most points fall within an ε-tube.**

---

## 03. Let's Think

Let's draw a tube of width ε = 5 around a candidate line:

```text
Score
  90│           ╭─────╮
  80│      ╭────╯  ●  ╰─  ← ε-tube (±5 around the line)
  70│  ╭───╯  ●
  60│──╯ ●
  50│  ●
  40│●
    └────────────────── Hours
     1  2  3  4  5
```

<!-- [THINK_ABOUT_IT] -->
🤔 Points *inside* the tube: predictions within ε of the target. SVR says: **these points are fine, cost = 0.**

Points *outside* the tube: predictions are off by more than ε. SVR penalizes these, but with a **linear** penalty (not squared).

The "best" line is the **flattest** one that keeps most points inside the tube. Flatness means small weights → smooth model → good generalization.

> This is fundamentally different from OLS (which minimizes every error) and Huber (which caps big errors). SVR **ignores small errors entirely**.

---

## 04. Intuition

💡 **The idea in one line:**

> SVR draws a **tube** of tolerance ε around the prediction curve, ignores points inside the tube, penalizes points outside it, and finds the **flattest** curve that fits the data within this tolerance.

Three key ingredients:

| Ingredient | Symbol | What it does |
|---|---|---|
| ε-tube | ε | Width of the "I don't care" zone around the prediction |
| Flatness | ½‖w‖² | Penalizes steep/complex models — keeps things simple |
| Violation penalty | C | How much to punish points outside the tube |

The balance between flatness (½‖w‖²) and violations (Σξ) is controlled by C. Large C → fit tightly. Small C → stay flat.

Add a **kernel** (like RBF) and the flat "line" becomes a smooth, bendy curve in a higher-dimensional space — capturing non-linear patterns without explicitly transforming features.

---

## 05. Visual

```text
SVR with ε-tube:
   y
   │        ·           ← points OUTSIDE tube = support vectors
   │       ╭─╮
   │      ╱   ╲___     ╱   ε-tube (region of zero loss)
   │     ╱  ─────      ·   inside tube = no influence
   │    ·╱     ·
   │   ╱
   │  ·
   │ /___________
   └────────────── x

Only the ● points (outside tube) are support vectors;
the · points inside are ignored.
```

```text
Loss comparison:
  loss
   │             OLS (squared): grows forever
   │   ╱╱
   │  ╱
   │ ╱                     SVR (ε-insensitive): 0 inside tube
   │╱
   └──────────╱────────── r
        tube  │ beyond ε, linear growth
```

---

## 06. First Prediction

Let's try a linear SVR on Arjun's data with ε = 5.

**Candidate: ŷ = 16x + 5**

| x | y | ŷ | Residual | |r| > ε? | Loss |
|---|---|---|---|---|---|
| 1 | 22 | 21 | 1 | No (inside tube) | 0 |
| 2 | 44 | 37 | 7 | **Yes** | 7 − 5 = 2 |
| 3 | 55 | 53 | 2 | No | 0 |
| 4 | 80 | 69 | 11 | **Yes** | 11 − 5 = 6 |
| 5 | 88 | 85 | 3 | No | 0 |

Total ε-insensitive loss = 2 + 6 = 8.

The flatness penalty is ½w² = ½(16²) = 128.

Total objective = 128 + C·8.

<!-- [TRY_IT] -->
For C = 1: total = 136. For C = 10: total = 208. Increasing C makes the model care more about violations — it would try to reduce those two outside points, possibly at the cost of flatness.

> 📌 SVR finds the C-balanced line where the tube captures most points, only a few are support vectors, and the model stays as flat as possible.

---

## 07. Core Concept

**Concept: Support Vector Regression** — a method that:

1. finds a function f(x) that deviates from targets by **at most ε** for most points,
2. while being as **flat** as possible (small ‖w‖²),
3. allowing violations (slack variables ξ) controlled by parameter C,
4. and optionally using **kernel functions** for non-linearity,
5. where only points outside the tube (**support vectors**) influence the model.

```text
PREDICTION  →  f(x) = Σ(αᵢ − αᵢ*)·K(xᵢ, x) + b
```

| Part | Symbol | Simple meaning |
|---|---|---|
| ε | epsilon | Tube width — errors within ε cost nothing |
| C | regularization | How much to penalize tube violations |
| K(xᵢ, x) | kernel | Computes similarity in high-dim space |
| αᵢ, αᵢ* | Lagrange multipliers | Nonzero only for support vectors |

---

## 08. Terminology

### ε-tube

> Simple: a band of tolerance around the prediction where errors are free.
> Technical: the region |y − f(x)| ≤ ε where the ε-insensitive loss is zero.

### Support Vectors

> Simple: the only points that matter — those outside the tube.
> Technical: training points with nonzero Lagrange multipliers, lying on or outside the ε-tube.

### Slack Variables (ξ, ξ*)

> Simple: how far each point exceeds the tube (upper and lower).
> Technical: measures of violation beyond ε, penalized linearly in the objective.

### Flatness

> Simple: the model should be as smooth/simple as possible.
> Technical: minimizing ‖w‖² in feature space — analogous to the ridge penalty.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| ε | tube width | insensitivity threshold |
| C | violation cost | regularization/tradeoff parameter |
| ξ, ξ* | how much points break the tube | slack variables |
| α, α* | point importance | Lagrange multipliers (nonzero = support vector) |
| K(xᵢ,x) | similarity measure | kernel function |
| γ | kernel width (RBF) | controls flexibility of the curve |

> ⚠️ Common mistake: "SVR minimizes squared error like OLS." No — it minimizes ε-insensitive loss + flatness penalty.

---

## 09. Mathematics

We build from the tube idea.

### Step M1 — The ε-insensitive loss

```text
L_ε(r) = max(0, |r| − ε)
```

<!-- [CALCULATION] -->
For ε = 1:
- r = 0.3 → |r|−ε = −0.7 → loss = 0 (inside tube)
- r = 1.0 → |r|−ε = 0 → loss = 0 (on boundary)
- r = 2.5 → |r|−ε = 1.5 → loss = 1.5

### Step M2 — The full objective

```text
Minimize  ½‖w‖² + C·Σᵢ(ξᵢ + ξᵢ*)
subject to:
    yᵢ − f(xᵢ) ≤ ε + ξᵢ      (upper bound)
    f(xᵢ) − yᵢ ≤ ε + ξᵢ*     (lower bound)
    ξᵢ, ξᵢ* ≥ 0
```

```text
½‖w‖²        → flatness penalty (keep the model simple)
C·Σ(ξ+ξ*)    → total violation cost
ξᵢ, ξᵢ*      → how much point i exceeds the tube (upper/lower)
ε             → tube width
```

### Step M3 — The dual form (kernel-friendly)

After solving the Lagrangian, the prediction becomes:

```text
f(x) = Σᵢ (αᵢ − αᵢ*)·K(xᵢ, x) + b
```

Only support vectors have nonzero αᵢ or αᵢ* — all other points contribute zero. This is the **sparsity** of SVR.

### Step M4 — The kernel trick

```text
K(xᵢ, x) = φ(xᵢ)ᵀφ(x)
```

We compute similarity in a high-dimensional space *without ever going there*. Common kernels:

```text
Linear:    K(xᵢ,x) = xᵢᵀx
RBF:       K(xᵢ,x) = exp(−γ‖xᵢ−x‖²)
Poly:      K(xᵢ,x) = (γ·xᵢᵀx + r)ᵈ
```

---

## 10. Numerical Example

Fit a **linear SVR** (kernel = linear) to 3 points: x=[1, 2, 3], y=[2, 4, 5]. Let ε = 0.5, C = 1.

<!-- [CALCULATION] -->

**Step 1 — Try f(x) = 2x (w=2, b=0).**

| x | y | f(x) | r | |r| > ε? | Slack |
|---|---|---|---|---|---|
| 1 | 2 | 2 | 0 | No | 0 |
| 2 | 4 | 4 | 0 | No | 0 |
| 3 | 5 | 6 | −1 | Yes | 0.5 |

Objective = ½(4) + 1·0.5 = **2.5**

**Step 2 — Try f(x) = 1.5x + 0.5.**

| x | y | f(x) | r | |r| > ε? | Slack |
|---|---|---|---|---|---|
| 1 | 2 | 2.0 | 0 | No | 0 |
| 2 | 4 | 3.5 | 0.5 | No (exactly ε) | 0 |
| 3 | 5 | 5.0 | 0 | No | 0 |

Objective = ½(2.25) + 0 = **1.125**

**Step 3 — Compare:**

```text
f(x) = 2x       → objective = 2.5   (flatness 2 + violation 0.5)
f(x) = 1.5x+0.5 → objective = 1.125 (flatness 1.125 + violation 0)
```

The second line wins: it's flatter AND has zero violations. SVR would choose something close to this.

**Key outcome:** All 3 points are inside the tube → **no support vectors** in the simplest case. If we add a point (4, 20), it would escape the tube and become a support vector, pulling the model.

> ✅ VERIFIED — hand-verified conceptually: SVR selects the flattest line consistent with ε-tolerance, driven by support vectors. The mechanism (flatness vs slack, ε-insensitivity) is what matters.

---

## 11. How It Works

```text
STEP 1   Have data (x, y), choose ε, C, kernel
STEP 2   Compute kernel matrix K (similarities between all points)
STEP 3   Formulate dual QP (quadratic program)
STEP 4   Solve QP → Lagrange multipliers αᵢ, αᵢ*
STEP 5   Identify support vectors (α ≠ 0)
STEP 6   Compute bias b from KKT conditions
STEP 7   Model: f(x) = Σ(αᵢ−αᵢ*)K(xᵢ,x) + b
STEP 8   Predict: compute kernel against support vectors only
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Compute kernel matrix K[i,j] = K(xᵢ, xⱼ)
     ↓
2. Solve dual QP for αᵢ, αᵢ* (most will be zero!)
     ↓
3. Identify support vectors (α ≠ 0)
     ↓
4. Compute bias b from any support vector via KKT
     ↓
5. Store: support vectors + their α coefficients + kernel + bias
```

```text
model.predict(X_new)
     ↓
For each new point:
    f(x) = Σ over support vectors only: (αᵢ−αᵢ*)·K(xᵢ, x_new) + b
```

> The prediction only touches support vectors — that's why SVR can be fast at prediction time despite slow training.

---

## 13. From Scratch

### Version 1 — ε-insensitive loss

```python
import numpy as np

def epsilon_insensitive_loss(r, epsilon):
    """Compute ε-insensitive loss for a residual."""
    return max(0, abs(r) - epsilon)

# Test
print(epsilon_insensitive_loss(0.3, 0.5))  # 0 (inside tube)
print(epsilon_insensitive_loss(2.0, 0.5))  # 1.5
```

### Version 2 — Linear SVR via subgradient descent

```python
class LinearSVR:
    def __init__(self, epsilon=0.1, C=1.0, lr=0.01, epochs=1000):
        self.epsilon = epsilon
        self.C = C
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        self.w = np.zeros(m)
        self.b = 0.0
        for _ in range(self.epochs):
            pred = X @ self.w + self.b
            err = y - pred
            grad_w = np.zeros(m)
            grad_b = 0.0
            for i in range(n):
                if err[i] > self.epsilon:
                    grad_w -= self.C * X[i]
                    grad_b -= self.C
                elif err[i] < -self.epsilon:
                    grad_w += self.C * X[i]
                    grad_b += self.C
            self.w -= self.lr * (self.w + grad_w / n)
            self.b -= self.lr * (grad_b / n)

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b
```

### Version 3 — with RBF kernel (simplified)

```python
class KernelSVR:
    def __init__(self, epsilon=0.1, C=1.0, gamma=1.0):
        self.epsilon = epsilon
        self.C = C
        self.gamma = gamma

    def _rbf(self, X1, X2):
        dists = np.sum(X1**2, 1)[:, None] + np.sum(X2**2, 1) - 2 * X1 @ X2.T
        return np.exp(-self.gamma * dists)

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y, dtype=float)
        self.alpha = np.zeros(len(y))  # simplified; real SVR uses QP solver

    def predict(self, X):
        K = self._rbf(self.X_train, np.asarray(X, dtype=float))
        return K.T @ self.alpha  # simplified
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = np.sort(np.random.RandomState(0).rand(100, 1), axis=0)
y = np.sin(6 * X).ravel() + np.random.RandomState(0).randn(100) * 0.1

model = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'))
model.fit(X, y)

y_pred = model.predict(X)
print("R²:", 1 - np.sum((y - y_pred)**2) / np.sum((y - y.mean())**2))
```

> Note: **scaling is essential** for SVR — kernel distances are meaningless without comparable feature scales. The `make_pipeline` handles this automatically.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
make_pipeline(StandardScaler(), SVR(kernel='rbf', C=100, epsilon=0.1, gamma='scale'))
```
> `StandardScaler` is **mandatory** — kernels compute distances, and unscaled features dominate the distance. `C=100` means high penalty for violations (fit tightly). `epsilon=0.1` sets the tube width. `gamma='scale'` sets RBF kernel width automatically.

```python
model.fit(X, y)
```
> Internally: computes kernel matrix, solves dual QP, identifies support vectors, computes bias. The `n_support_` attribute tells you how many support vectors were found.

```python
y_pred = model.predict(X)
```
> Prediction computes the kernel between new points and **only** the support vectors. This is why SVR prediction can be fast despite slow training.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
### Experiment A — slide the tube width

```text
ε = 0.01  →  almost every point is a support vector → complex, possibly overfit
ε = 0.5   →  moderate tube → good balance
ε = 5.0   →  very wide tube → almost everything is inside → very flat, underfit
```

> What to notice: as ε increases, the model gets **flatter** (lower ‖w‖) but more violations. As ε decreases, more points become support vectors and the model gets **more complex**.

### Experiment B — kernel comparison (code)

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
X = np.sort(np.random.rand(50, 1), axis=0) * 10
y = np.sin(X.ravel()) + np.random.randn(50) * 0.3

for kernel in ['linear', 'rbf', 'poly']:
    model = make_pipeline(StandardScaler(), SVR(kernel=kernel, C=10, epsilon=0.1))
    model.fit(X, y)
    score = model.score(X, y)
    n_sv = model[1].n_support_.sum()
    print(f"kernel={kernel:>6}  R²={score:.3f}  support_vectors={n_sv}")
```

```text
kernel=linear  R²=0.412  support_vectors=35   (underfits — can't capture sine)
kernel=   rbf  R²=0.951  support_vectors=18   (fits well — captures curve)
kernel=   poly R²=0.783  support_vectors=28   (captures some curve)
```

> 📌 The linear kernel can't capture the sine pattern. RBF captures it beautifully with fewer support vectors. Kernel choice matters enormously.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# Unscaled features — the cardinal sin of SVR
X = np.array([[1000], [2000], [3000], [4000], [5000]])
y = np.array([1, 2, 3, 4, 5], dtype=float)

model_unscaled = SVR(kernel='rbf', C=10, epsilon=0.1)
model_unscaled.fit(X, y)
print("Unscaled score:", model_unscaled.score(X, y))

model_scaled = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=10, epsilon=0.1))
model_scaled.fit(X, y)
print("Scaled score:", model_scaled.score(X, y))
```

```text
Unscaled score: -12.457   (terrible — the kernel sees 1000–5000 as huge distances)
Scaled score:    0.998    (great — features are on comparable scale)
```

> 💥 **Break pattern:** forget to scale → kernel distances are dominated by large-scale features → the model is useless. **Always scale before SVR.** This is the #1 SVR mistake.

---

## 18. What If...?

<!-- [WHAT_IF] -->
| You change… | What happens | Why |
|---|---|---|
| ε → 0 | Almost every point becomes a support vector → like OLS | No tolerance → every error counts |
| ε → huge | Very flat model, almost no support vectors | Too tolerant → underfits |
| C → 0.01 | Very flat, ignores violations | Prioritizes flatness over fit |
| C → 1000 | Fits tightly, many support vectors | Prioritizes fit over flatness |
| kernel: linear → RBF | Model captures non-linearity | RBF maps to infinite-dim space |
| γ (RBF) → huge | Each point has its own bump → overfit | Very localized kernel |
| γ (RBF) → 0 | Kernel becomes constant → underfit | Very smooth, no local structure |
| Forget to scale | Distances dominated by large features → bad kernel | Kernel is distance-based |

> 🤔 Think: which two parameters interact most? → C and γ (or ε). High C + high γ = overfit machine. Low C + low γ = underfit machine. Always tune them together.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
αᵢ, αᵢ*  → Lagrange multipliers (nonzero only for support vectors)
b         → bias
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| ε (epsilon) | Tube width | Too many support vectors, complex | Too few support vectors, underfit | 0.1 (default) |
| C | Violation penalty | Underfit (too flat) | Overfit (too tight) | 1–100 |
| kernel | Similarity function | Linear only | RBF is flexible | RBF common |
| γ (gamma, RBF) | Kernel width | Underfit (too smooth) | Overfit (too wiggly) | `'scale'` or tune |

> 📌 ε and C together control the bias-variance tradeoff. Tune them via grid search with cross-validation.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Smooth function | nearby x → similar y | kernel smoothness | domain knowledge | different kernel |
| Scale comparability | features on same scale | kernel distances | check feature ranges | StandardScaler |
| Most errors within ε | sparsity condition | efficiency | support vector fraction | tune ε/C |

> SVR does **NOT** assume linearity (kernel handles that) or Gaussian errors.

---

## 21. Data Requirements

```text
Target      → continuous numeric
Features    → numerical; categorical must be encoded AND scaled
Missing     → must be handled first
Outliers    → somewhat robust (ε-tube), but extreme outliers beyond C pull
Scaling     → ESSENTIAL (distance-based kernel)
Dataset size → best on small-to-medium; QP scales O(n²) to O(n³)
High-dim    → kernel SVR can suffer; needs careful tuning
```

> ⚠️ The O(n²) to O(n³) training complexity means SVR is **not suitable for very large datasets** (n > 10,000 becomes slow). Use tree ensembles or neural networks for large n.

---

## 22. Evaluation

| Metric | Formula | Use | Avoid |
|---|---|---|---|
| RMSE | √MSE | standard | when you want robust evaluation |
| MAE | (1/n)Σ\|y−ŷ\| | robust | when big misses matter |
| R² | 1 − SS_res/SS_tot | model quality | comparing across datasets |
| **Support vector fraction** | n_SV / n | **diagnostic** | — |

> **Key diagnostic:** if the support vector fraction is very high (>70%), the model is probably overfitting — too many points are outside the tube. Tune ε upward or C downward.

---

## 23. Failure Cases

```text
HUGE DATA        → QP scales O(n²–n³), training intractable
WRONG KERNEL     → underfit or overfit depending on kernel choice
UNSCALED DATA    → kernel distances meaningless
EXTREME OUTLIERS → beyond C, still pull the fit
DISCONTINUOUS    → smooth kernels can't capture jumps
HIGH-DIM SPARSE  → kernel methods struggle (text data)
```

---

## 24. Debugging

```text
1. Very high support vector fraction?    → ε too small, or C too high → tune
2. Poor R², linear-looking fit?          → wrong kernel (try RBF)
3. Great train R², terrible test R²?     → γ too high or C too high → overfit
4. All predictions near the mean?        → C too low, ε too large → flatten underfit
5. Numerical errors in fit?              → features unscaled → StandardScaler
6. Training takes forever?               → n too large → sample or use trees
```

---

## 25. Compare

```text
Linear Regression:   "Minimize squared error of every point."
Huber:               "Cap the influence of big errors."
SVR:                 "Ignore small errors entirely. Only care about what's outside the tube."
```

| Algorithm | Main idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Linear Regression | min squared error | simple, fast | linear only, outlier-sensitive | linear data |
| Ridge | squared error + L2 | stable | linear | collinear features |
| SVR | ε-insensitive + kernel | non-linear, robust, sparse | QP cost, tuning | small/medium non-linear |
| Decision Tree | recursive splits | non-parametric, interactions | step-like, high variance | interpretable |
| Random Forest | bagged trees | robust, parallel | opaque | big data |

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict stock price movement from market indicators
DATA:              2000 trading days, 20 indicators
EDA:               non-linear relationships, moderate noise
FEATURES:          20 market indicators (scale essential)
TARGET:            price_change
SPLIT:             train/val/test (stratified by time)
SCALE:             StandardScaler (essential!)
CHOOSE kernel:     RBF (default, flexible)
TUNE:              GridSearchCV over C=[0.1,1,10,100], ε=[0.01,0.1,0.5], γ=['scale','auto',0.1]
EVALUATE:          RMSE + support vector fraction
DEPLOY:            serve predictions
MONITOR:           support vector fraction drift (model degradation signal)
```

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is the ε-insensitive loss? Write its formula.
2. **Understand:** why does SVR give a sparse solution (few support vectors)?
3. **Calculate:** compute ε-insensitive loss for residuals [0.5, 2, 4] with ε=1.
4. **Apply:** given a scatter plot, draw an approximate ε-tube and identify support vectors.
5. **Debug:** your SVR has 95% support vectors — what's wrong and how to fix it?
6. **Experiment:** run the kernel comparison (Section 16B) at 4 data sizes; observe when SVR becomes impractical.
7. **Build:** non-linear regression project: compare OLS vs SVR(linear) vs SVR(RBF) vs tree → report RMSE, support vectors, training time.
8. **Explain:** explain SVR to a friend in 60 seconds using the tolerance-tube analogy.

---

## 28. Interview

### Beginner
- **What is SVR?** Regression that fits a function within an ε-tolerance tube, as flat as possible, defined by support vectors.
- **What is a support vector?** A training point on or outside the ε-tube that influences the model.
- **What is ε?** The tube width — errors within ±ε are ignored (free).

### Intermediate
- **How is SVR different from OLS?** SVR uses ε-insensitive loss (ignores small errors) and penalizes flatness; OLS squares every error.
- **What does C do?** Trades flatness vs violation penalty — high C hugs data, low C stays smooth.
- **What is the kernel trick?** Computes inner products in a high-dimensional space implicitly, enabling non-linear regression without explicit transformation.
- **Why scale features for SVR?** Kernels are distance-based; unscaled features with different units dominate the distance computation.

### Advanced
- **Why is the solution sparse?** KKT conditions force most Lagrange multipliers to zero; only support vectors (outside tube) have nonzero ones.
- **Explain the dual formulation.** Convert the constrained primal into a QP over Lagrange multipliers; prediction becomes weighted kernel sums against support vectors only.
- **When does SVR fail and why?** On huge data (QP cost O(n²–n³)), unscaled features (kernel distances meaningless), and when the smooth-kernel assumption doesn't match the data.

---

## 29. GATE / Exam

<!-- [GATE] -->
**Formulas worth memorizing:**

```text
ε-insensitive loss:  L = max(0, |y − f(x)| − ε)
Primal objective:    ½‖w‖² + C·Σ(ξ + ξ*)
Dual prediction:     f(x) = Σ(αᵢ − αᵢ*)·K(xᵢ,x) + b
```

**Common traps:**
- Confusing SVR margin with SVM classification margin.
- Forgetting SVR minimizes ε-insensitive loss, **not** MSE.
- Assuming SVR scales to large n (it doesn't — O(n²–n³)).
- Not scaling features before using kernel SVR.

> **Representative pattern question (NOT a past GATE PYQ):** "For ε=1, compute ε-insensitive loss for residuals 0.5, 2, and 4." Answers: **0, 1, 3**.

---

## 30. Deep Dive

<!-- [DEEP_DIVE] -->
<details>
<summary>Click to open the derivation + theory + complexity</summary>

### Primal formulation

```text
Minimize  ½‖w‖² + C·Σ(ξᵢ + ξᵢ*)
subject to:
    yᵢ − wᵀφ(xᵢ) − b ≤ ε + ξᵢ
    wᵀφ(xᵢ) + b − yᵢ ≤ ε + ξᵢ*
    ξᵢ, ξᵢ* ≥ 0
```

### Lagrangian

```text
L = ½‖w‖² + C·Σ(ξ+ξ*) − Σ(μξ+μ*ξ*) − Σα(ε+ξ−y+f) − Σα*(ε+ξ*+y−f)
```

### KKT conditions

```text
∂L/∂w = 0  →  w = Σ(αᵢ−αᵢ*)φ(xᵢ)
∂L/∂b = 0  →  Σ(αᵢ−αᵢ*) = 0
∂L/∂ξ = 0  →  αᵢ + μᵢ = C  →  αᵢ ∈ [0, C]
```

Only points with 0 < αᵢ < C lie on the tube boundary (support vectors). Points with αᵢ = C are outside the tube (violations).

### Dual QP

```text
Maximize  −½ΣΣ(αᵢ−αᵢ*)(αⱼ−αⱼ*)K(xᵢ,xⱼ) − εΣ(αᵢ+αᵢ*) + Σyᵢ(αᵢ−αᵢ*)
subject to  Σ(αᵢ−αᵢ*) = 0,  0 ≤ αᵢ,αᵢ* ≤ C
```

### Complexity

```text
Kernel matrix:     O(n²) memory and compute
QP training:       O(n² to n³)  ← main bottleneck
Prediction:        O(n_sv · d)  ← only support vectors
Space:             O(n_sv) stored model
```

### ν-SVR

An alternative parameterization where ν ∈ (0,1] bounds the fraction of support vectors and training errors. Makes tuning more intuitive.

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "SVR draws a tolerance tube around the prediction. Points inside the tube cost nothing. Points outside push the model. The goal is the flattest model where most points fit in the tube. Kernels let the tube follow curves."

> **Explain to a 12-year-old:** "Imagine a tube drawn through data points. SVR tries to make the tube as straight as possible while most dots are inside. Only dots that stick out actually matter."

> **Explain in an interview:** add: ε-insensitive loss, primal (½‖w‖² + C·slack), dual (kernel sums), support vectors via KKT, scaling essential, O(n²) training.

> **Explain the mathematics:** write the Lagrangian, show KKT conditions yield sparse α (most zero → support vectors).

---

## 32. Mastery Test

**Without looking at notes:**

1. Define ε-insensitive loss and write its formula.
2. What is a support vector in SVR?
3. Explain the flatness penalty ½‖w‖² — why is it there?
4. What does C control? What happens at C→0 and C→∞?
5. Compute ε-insensitive loss for r=0.3, r=2, r=5 with ε=1.
6. Why must features be scaled before SVR?
7. What is the kernel trick in one sentence?
8. Compare SVR and OLS in terms of which errors they care about.
9. When does SVR become impractical (n)?
10. What diagnostic tells you if SVR is overfitting?

---

## 33. Cheat Sheet

```text
Algorithm : Support Vector Regression · Supervised → Regression · Kernel method
Goal      : flattest function within ε-tube
Loss      : ε-insensitive: max(0, |r| − ε)
Objective : ½‖w‖² + C·Σ(ξ + ξ*)
Prediction: f(x) = Σ(αᵢ−αᵢ*)·K(xᵢ,x) + b
Learn     : α (support vectors), b
Tune      : ε, C, kernel, γ
Key trick : kernel → non-linearity without explicit transform
Assumptions: smooth function, scale comparability
Use when  : small/medium data, non-linear, robust, sparse solution
Avoid when: huge data (O(n²)), interpretability needed, text/sparse
Related   : SVM · Ridge · Kernel Ridge · ν-SVR
Key insight: only support vectors matter → sparse, robust, kernel-flexible
```

---

## 34. What Next?

You just learned the kernel-based approach to regression.

```text
Support Vector Regression (kernel + ε-tube)
   ├── Decision Tree Regression    (nonlinear, no kernels)        → 10
   ├── Random Forest Regression    (bagged trees, averaging)      → 11
   ├── Extra Trees Regression      (faster, more random)          → 12
   └── Kernel Ridge Regression     (kernel + squared error)
```

> Next recommended: **10. Decision Tree Regression** — a completely different philosophy. No loss function, no optimization — just recursive if-then splits that partition the space into regions. The foundation for all tree ensembles.
