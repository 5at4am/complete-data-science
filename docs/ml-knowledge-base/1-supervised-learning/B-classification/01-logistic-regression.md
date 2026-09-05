# 01. Logistic Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐☆☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **story → guess → sigmoid curve → log-odds → loss → gradient descent → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

This is the **first classification model everyone meets** — the direct sibling of Linear Regression, but for Yes/No problems.

By the end you will be able to:

- predict a **probability** (not just a label) for any new sample,
- explain *why* a straight line fails for Yes/No targets,
- compute the sigmoid and the log-loss by hand on tiny data,
- code logistic regression from scratch and with sklearn,
- break it deliberately and fix it,
- and defend when to use — and not use — it.

> Everything builds on one small idea. Let's find it.

---

## 02. The Problem

Riya is the world's busiest sorcerer — she receives **400 SMS messages a day**. She wants an assistant that flags **spam** so she never opens a "Congratulations! You've won ₹50 lakh" again.

She shows you the last five messages she manually tagged:

| Message | # of suspicious words (WIN, FREE, PRIZE) | Spam? |
|---|---|---|
| "Meeting at 4 pm tomorrow" | 1 | No |
| "WIN free prize today" | 3 | Yes |
| "Project deadline is Friday" | 2 | No |
| "YOU ARE THE WINNER, FREE CASH" | 4 | Yes |
| "Lunch? Definitely not spam" | 1 | No |

Now a sixth message arrives:

> **"Congratulations, you have WON a FREE trip to Dubai"**  → **2 suspicious words.**

<!-- [QUESTION] -->
**Should the assistant call it Spam or Not?**

Don't scroll straight to the answer. Make your best guess first.

**Your guess: Spam ☐   Not Spam ☐**

> 📌 Keep your instinct in mind. Section 06 will see how close the model agrees with you.

---

## 03. Let's Think

Before predicting, look at the data with fresh eyes.

```text
suspicious words →  Spam?
1               →  No
3               →  Yes
2               →  No
4               →  Yes
1               →  No
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> Higher suspicious-word count → mostly **Yes**. Lower → mostly **No**. There's a threshold somewhere between 2 and 3.

Now the crucial question:

> The new message has **2** suspicious words. Rows with 1 are No, and rows with 3–4 are Yes. So what about **2**?

It's right in the **gray zone**. Neither the 5 rows nor your eyes give a clean Yes/No.

> And that's the whole point: real decisions are rarely perfectly clear. We want a **probability**, not a forced label.

---

## 04. Intuition

If we put the SMS data on a graph, something funny happens:

<!-- [VISUAL] -->
```text
Spam?
 1 ┤                                   • (4)
   │                                /
   │                             •
 0 ┤  •          •         •     ← 3 of these are "1"
   └─────────────────────────────────── suspicious words
      1       2       3       4
```

We want to draw a curve that:
- stays **between 0 and 1** (it's a probability!),
- rises from 0 upward through 1 as suspicious words increase,
- is **S-shaped** — flat at the edges, steep in the middle.

💡 **The idea in one line:**

> Logistic Regression fits an **S-shaped curve** (the sigmoid) to data, so that it can output a probability between 0 and 1 — instead of a straight line that would fly past 0 and 1.

The straight line from regression can't do this. Watch what happens if we try:

```text
A straight line would predict:  "3.0 suspicious words  →  spam score 1.4"
                                    ↑ that's not a probability!
     probabilities must live inside [0, 1]
```

So we need a **different shape**. The S-curve is that shape.

---

## 05. Visual First

Here's the S-shaped curve we'll use:

```text
 σ(z)
  1 ┤                ___________________
    │               /
    │              /
 0.5┤— — — — — — —/— — — — — — — — —  ← decision line at 0.5
    │            /
    │           /
  0 ┤__________/
    └──────|──────|──────|──────|────→  z = w·x + b
         -3      0      +3
                 ↑
          z = 0 → σ = 0.5
```

The horizontal axis here is **z = w·x + b** — exactly the linear combination from Linear Regression. The curve **squashes** whatever z we give it into a number between 0 and 1.

| z (linear score) | σ(z) (probability) |
|---|---|
| very negative | close to 0 |
| 0 | exactly 0.5 |
| very positive | close to 1 |

> 📌 The curve is symmetric, smooth, and bounded. Those three properties will do all our heavy lifting.

---

## 06. First Prediction

Which S-curve is "best"? There are many possible curves. Let's use our eyes first: a curve that reaches `Spam? = 1` around 3 suspicious words and starts near 0 at low counts looks reasonable.

At 2 suspicious words, the curve sits at roughly **0.4–0.5**:

<!-- [TRY_IT] -->
> Model's answer for 2 suspicious words: **P(spam) ≈ 0.45, so label = Not Spam.**

Does that match your earlier guess?

> 📌 If you hesitated between Spam and Not, you already understood the idea: at the boundary, the probability is close to 0.5. Logistic Regression makes that *hesitation* a precise, repeatable number.

Now the honest question that drives the rest:

> **How do we choose the "best" S-curve without drawing it by hand?**

That means two things:
1. What makes a curve good? → the **loss** function.
2. How do we find the best one? → **gradient descent**.

Let's define the curve precisely first.

---

## 07. Core Concept

Introducing the idea formally, right after we've already met it:

**Concept: Logistic Regression** — a method that:

1. computes a **linear score** `z = w·x + b` (like regression),
2. squashes it through the **sigmoid** function `σ(z) = 1/(1+e⁻ᶻ)` to get a **probability**,
3. classifies based on whether that probability crosses **0.5**,
4. finds the best weights by minimizing **cross-entropy (log-loss)**.

```text
PREDICTION  →  p = σ(w·x + b) = 1 / (1 + e^(−(w·x + b)))
```

The whole pipeline:

```text
  features x ──→  z = w·x + b ──→  p = σ(z) ──→  class = 1 if p ≥ 0.5
                 (linear score)   (probability)    else 0
```

> Everything else — loss, gradients, gradient descent — exists only to make `w` and `b` good.

---

## 08. Terminology

Each term *emerges* from our spam story:

### Sigmoid (σ)

> Simple: the S-curve that turns any number into a probability.
> Technical: `σ(z) = 1/(1+e⁻ᶻ)`, maps ℝ → (0, 1).

### Logit / Log-odds

> Simple: the "raw score" on the inside of the curve.
> Technical: `ln(p/(1−p))` = the inverse of the sigmoid.

### Odds

> Simple: how many times more likely "Yes" than "No".
> Technical: `p/(1−p)`.

### Decision boundary

> Simple: the line where the model can't decide.
> Technical: the hyperplane `w·x + b = 0`, where `p = 0.5`.

### Cross-entropy / log-loss

> Simple: how bad the probabilities are.
> Technical: `−(1/n)Σ [y·log(ŷ) + (1−y)·log(1−ŷ)]`.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| z | raw score of the message | linear combination `w·x + b` |
| σ(z) | squashed probability | sigmoid, `1/(1+e⁻ᶻ)` |
| ŷ | predicted probability | `P(y=1\|x)` |
| p/(1−p) | odds of spam | odds ratio |
| ln(p/(1−p)) | log-odds / logit | inverse sigmoid |
| w, b | how features vote | weights + bias |
| p ≥ 0.5 | "it's spam" | threshold rule |

> ⚠️ Common mistake: calling it "regression" and thinking it's for numbers. The name is historical — it uses a regression-style *score*, but predicts **categories**.

---

## 09. Mathematics (gradual)

We build the math from zero. Five small steps.

### Step M1 — The linear score

```text
z = w·x + b
```

Just like Linear Regression. A weight per feature plus a bias.

### Step M2 — The sigmoid

```text
σ(z) = 1 / (1 + e^(−z))
```

- `e` ≈ 2.718 (Euler's number). `e^(−z)` is `1/(e˻z)˼`.
- When z is huge positive: `e^(−z)` ≈ 0 → σ ≈ 1.
- When z is huge negative: `e^(−z)` is huge → σ ≈ 0.
- At z = 0: `e^0 = 1` → σ = 1/2 = 0.5.

<!-- [CALCULATION] -->
```text
z = 2   →  σ = 1/(1 + e^(−2)) = 1/(1 + 0.1353) = 1/1.1353 ≈ 0.881  → 68%? no, 88%
z = 0   →  σ = 1/(1 + 1)       = 0.5
z = −2  →  σ = 1/(1 + e²)      = 1/(1 + 7.39)  = 1/8.39  ≈ 0.119
```

### Step M3 — Why sigmoid is the natural shape

The sigmoid is not invented randomly — it *falls out* of modeling the **log-odds** linearly.

```text
ln( p / (1−p) ) = z      ← "log-odds is a straight line"
```

Solve for p and you get... the sigmoid. (See Section 30 for the full derivation.)

> 💡 Intuition: instead of predicting the probability directly (0..1), predict the **log of the odds** (any real number) — then the linear machinery from regression works as-is.

### Step M4 — The decision rule

```text
z > 0  →  σ(z) > 0.5  →  predict class 1
z < 0  →  σ(z) < 0.5  →  predict class 0
z = 0  →  the decision boundary
```

### Step M5 — The loss (why not squared error?)

Squared error (MSE) from regression seems natural, but with a sigmoid it becomes **non-convex** — it develops local minimum traps. Instead we use **cross-entropy (log-loss)**:

```text
loss(single point) =  −[ y·log(ŷ) + (1−y)·log(1−ŷ) ]
```

Why this works — think about the two cases:

- **y = 1** (actually spam): loss = `−log(ŷ)`. If ŷ is near 1 (confident & right) → loss ≈ 0. If ŷ is near 0 (confident & wrong) → loss → ∞.
- **y = 0** (actually not): loss = `−log(1−ŷ)`. If ŷ near 0 → loss ≈ 0. If ŷ near 1 → ∞.

The full objective:

```text
J(w,b) = −(1/n) Σᵢ [ yᵢ·log(ŷᵢ) + (1−yᵢ)·log(1−ŷᵢ) ]
```

> 💡 Saying "log-loss is convex with the sigmoid" is the mathematically correct reason we use it. The human reason: **a confident wrong answer hurts more than a hesitant one** — exactly what we want.

---

## 10. Numerical Example

Tiny dataset we can check **on paper**:

```text
x (suspicious words) = [1, 3, 2]
y (spam? 1/0)        = [0, 1, 0]
```

Initialize `w = 0, b = 0`, learning rate `α = 0.5`. Let's do **one iteration** of gradient descent.

<!-- [CALCULATION] -->
**Step 1 — Forward pass (z, then σ):**

```text
x=1: z = 0·1 + 0 = 0   →  ŷ = σ(0) = 0.5
x=3: z = 0             →  ŷ = 0.5
x=2: z = 0             →  ŷ = 0.5
```

**Step 2 — Gradients (prediction − truth):**

```text
∂J/∂w = (1/3)·Σ (ŷᵢ − yᵢ)·xᵢ
      = (1/3)[ (0.5−0)(1) + (0.5−1)(3) + (0.5−0)(2) ]
      = (1/3)[ 0.5  − 1.5     + 1.0 ]
      = (1/3)[ 0.0 ] = 0.0

∂J/∂b = (1/3)·Σ (ŷᵢ − yᵢ)
      = (1/3)[ (0.5−0) + (0.5−1) + (0.5−0) ]
      = (1/3)[ 0.5 − 0.5 + 0.5 ] = 0.167
```

**Step 3 — Update:**

```text
w = 0 − 0.5·0.0   = 0.0
b = 0 − 0.5·0.167 = −0.083
```

After one step: `w = 0.0, b = −0.083`. The bias pulled slightly negative (pushing all predictions down, since mostly-No data). Repeating many iterations would let `w` grow positive (suspicious words push toward spam).

> ✅ VERIFIED — gradients hand-computed with the exact formula. The sign of each update matches intuition.

> 🎯 Your turn: what would happen to `w` after enough iterations, given that higher x tends to be spam? *(Answer: w goes positive, so higher x → higher z → higher p.)*

---

## 11. How It Works

```text
STEP 1   Have labeled data (x, y ∈ {0,1})
STEP 2   Model probability:  ŷ = σ(w·x + b)
STEP 3   Define "wrong": log-loss (cross-entropy)
STEP 4   Find w, b that minimize it   ← gradient descent
STEP 5   Production: new x → p → class (p ≥ 0.5)
```

If Section 09 was clear, the only new piece is Step 4 — and even that is a small loop.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Standardize / validate data
     ↓
2. Initialize w = 0 (or small random), b = 0
     ↓
3. REPEAT until convergence:
     a. z      = w·x + b
     b. ŷ      = σ(z)                    (forward pass)
     c. loss   = cross-entropy(ŷ, y)
     d. ∂/∂w   = (1/n)·Xᵀ·(ŷ − y)        (backward pass)
     e. ∂/∂b   = (1/n)·Σ(ŷ − y)
     f. w ← w − α·∂/∂w;  b ← b − α·∂/∂b
     ↓
4. Store:  coef_ (= w), intercept_ (= b)
     ↓
model.predict_proba(X_new):
     for each new row:
         p = σ(w·x + b)
model.predict(X_new):
     (p ≥ 0.5) → 1, else 0
```

> Unlike OLS regression (a one-shot formula), logistic regression **loops**. That's the main new idea.

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
import math

def sigmoid(z):
    return 1 / (1 + math.exp(-z))

def fit_logistic(xs, ys, lr=0.5, epochs=1000):
    n = len(xs)
    w, b = 0.0, 0.0
    for _ in range(epochs):
        dw = db = 0.0
        for i in range(n):
            z = w * xs[i] + b
            y_hat = sigmoid(z)
            dw += (y_hat - ys[i]) * xs[i]
            db += (y_hat - ys[i])
        w -= lr * dw / n
        b -= lr * db / n
    return w, b

w, b = fit_logistic([1, 3, 2], [0, 1, 0])
print(round(w, 4), round(b, 4))        # w>0, b ~ small negative

def predict_proba(x, w, b):
    return sigmoid(w * x + b)

print(round(predict_proba(2, w, b), 4))   # probability at x=2
```

> This is *literally* Section 10's update, looped. Nothing hidden.

### Version 2 — numpy, vectorized, many features

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_logistic_vec(X, y, lr=0.1, epochs=1000):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    for _ in range(epochs):
        z = X @ w + b
        y_hat = sigmoid(z)
        dw = (X.T @ (y_hat - y)) / n
        db = np.sum(y_hat - y) / n
        w -= lr * dw
        b -= lr * db
    return w, b

# works for ANY number of features
```

Same math, but the whole dataset updates at once.

### Version 3 — clean class

```python
import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr, self.epochs = lr, epochs
        self.w = None
        self.b = 0.0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        n, d = X.shape
        self.w = np.zeros(d)
        for _ in range(self.epochs):
            y_hat = self._sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (y_hat - y)) / n
            self.b -= self.lr * np.sum(y_hat - y) / n
        return self

    def predict_proba(self, X):
        return self._sigmoid(np.asarray(X, float) @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[1], [3], [2]])
y = np.array([0, 1, 0])

model = LogisticRegression()        # scikit does standardization internally
model.fit(X, y)

print(model.coef_)                  # [[w]]  → the learned weight
print(model.intercept_)             # [b]    → the bias
print(model.predict_proba([[2]]))   # [[P(no), P(yes)]]  → probabilities
print(model.predict([[2]]))         # class label
```

> `model.coef_` = our `w`, `model.intercept_` = our `b`. `predict_proba` gives you both probabilities; `predict` applies the 0.5 threshold. sklearn did exactly what Section 13's class did — faster, validated, and battle-tested.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
z = X @ w + b
```
> Computes the linear score for every sample in one matrix multiply — `z = w·x + b`.

```python
y_hat = 1 / (1 + np.exp(-z))
```
> The sigmoid. Vectorized: it applies the S-curve to every z at once, giving probabilities.

```python
dw = (X.T @ (y_hat - y)) / n
```
> `(ŷ − y)` is "how wrong we are" per sample. Dotting with `X.T` accumulates the per-feature gradient. This is Section 10's `∂J/∂w` formula, vectorized.

```python
self.w -= self.lr * dw
```
> Gradient descent: step downhill. Subtract (move opposite the gradient) because we **minimize**.

> 🧠 Every line maps to a formula from Section 09–10. Nothing arbitrary.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> If rendered in the platform, these become sliders. Otherwise run them in Python and observe.

### Experiment A — slide the decision boundary

Imagine sliders for `w` and `b`, with the S-curve drawn over the SMS data:

```text
w too small (0.1) →  curve barely rises; everything predicted "No"
w too big   (4.0) →  curve jumps at 0.5 too early; x=2 → clearly Spam
b too high  (2)   →  whole curve shifts left; more "Spam"
b too low   (−2)  →  whole curve shifts right; more "Not Spam"
```

> What to notice: the **log-loss number** is minimized exactly when the curve visually separates Spam from Not. Eyes and math agree again.

### Experiment B — probabilities vs labels (code)

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.linspace(0, 5, 200).reshape(-1, 1)
rng = np.random.default_rng(7)
y = (X.ravel() + rng.normal(0, 1.2, 200) > 2.5).astype(int)

m = LogisticRegression().fit(X, y)
probs = m.predict_proba([[1.0], [2.5], [2.7], [4.0]])[:, 1]
for x, p in zip([1.0, 2.5, 2.7, 4.0], probs):
    print(f"x={x:>3}  P(spam)={p:.3f}")
```

```text
x=1.0  P(spam)=0.08x   →  strongly Not
x=2.5  P(spam)=0.4xx   →  hesitant
x=2.7  P(spam)=0.5xx   →  right at the boundary
x=4.0  P(spam)=0.9xx   →  strongly Spam
```

> 📌 The moral: logistic regression gives you *confidence*, not just a label. Two points can both be "No" yet have very different probabilities — that richness matters in decisions (credit, spam, medical screening).

---

## 17. Break the Model

<!-- [BREAK_IT] -->
**Experiment: force a "confident wrong" prediction.**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([[0.1],[0.12],[0.15],[8.9],[9.1],[9.3]])
y = np.array([1,1,1,0,0,0])     # weird: LOW x = spam, HIGH x = not

m = LogisticRegression(max_iter=5000).fit(X, y)
print("coef:", m.coef_[0][0])       # negative! (flips the curve)
print("P(spam) at 5.0:", m.predict_proba([[5.0]])[0,1])
```

```text
coef: negative            ← the learned rule is now "lower x → more likely"
P(spam) at 5.0: ≈ 0.5      ← dead center, maximally unsure
```

**What happened?** The data's pattern was the *opposite* of the natural story, so the model learned a flipped curve. The point x=5 has **no training evidence nearby** — the model is forced to guess.

Now the teaching step:

- **Add more points near x=5** → the model stops being unsure there.
- **Remove the middle** → nothing changes, the two clusters still determine the curve.
- **Lesson:** logistic regression interpolates smoothly between clusters but is **misleadingly confident far from data**. And with a **linearly separable** dataset and zero regularization, the weights can shoot to infinity (see Section 30).

> 💥 **Break pattern:** separable data + no regularization → `coef_` explodes to ±∞.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Scale features 10× | Coefficient shrinks 10× | Only the *product* `w·x` matters; meaning is unchanged |
| Make classes perfectly separable | Coefficients grow huge | To push probabilities to 0/1, sigmoid needs z → ±∞ |
| Add polynomial features (x, x²) | Curve can bend | Now the boundary can be curved/non-linear |
| Double the data | Curves stabilize | More evidence → less wobble |
| Set threshold to 0.3 | More "Spam" predictions | Higher recall, lower precision (see Section 22) |
| Target becomes a number (marks) | Wrong tool | That's regression → Linear Regression |
| Add a correlated duplicated feature | Coeffs get unstable/split unfairly | Model can't tell them apart |

> 🤔 Think: which change lets us solve the XOR problem? → Adding polynomial/interaction features — the base model is still linear, but *in the expanded feature space*.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w   → one coefficient per feature     (model.coef_)
b   → the bias                        (model.intercept_)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `C` (inverse regularization) | How much to fight big weights; `C = 1/λ` | Strong regularization → underfit | Weak → overfit, weight explosion | 1.0; tune log-scale |
| `penalty` | Type of fight: `l1`, `l2`, `none` | — | — | `l2` (default) |
| `solver` | Optimization engine | — | — | `lbfgs` (default) |
| `max_iter` | Loop cap for gradient descent | Not converged | Wasted time | 100; raise on warnings |
| `class_weight` | Fairness for imbalanced data | — | — | `'balanced'` if skewed |

> 📌 In sklearn, `C` means "larger = *less* regularization". People mix this up constantly — bigger `C` is closer to plain maximum-likelihood.

---

## 20. Assumptions

For each: what, why, how to check, what if violated.

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linearity of log-odds** | `ln(p/(1−p))` is a straight line in features | The model family is linear in `z` | residual plots, feature transforms | polynomial features, or a different family |
| **Independence** | samples don't influence each other | standard statistical assumption | domain knowledge | clustered/time-series correction |
| **No multicollinearity** | features aren't near-duplicates | weights become unstable | correlation matrix, VIF | regularize (L1/L2), drop features |
| **Enough samples per class** | MLE needs data | unreliable estimates otherwise | events-per-feature ~ ≥10 | regularization, more data |
| **Representative & separable-ish classes** | class regions roughly match | the boundary is linear | visualize; compare with non-linear model | add features or switch models |

> For pure **prediction**, linearity of log-odds matters most. The rest mostly matter for **interpretability** of the weights (inference).

---

## 21. Data Requirements

```text
Target      → categorical: binary {0,1}, or multiclass (softmax / OVR)
Features    → numerical; encode categorical (one-hot, ordinal)
Missing     → must handle first (impute or drop) — sklearn can't take NaN
Outliers    → moderate sensitivity; extreme outliers can shift the boundary
Scaling     → RECOMMENDED: faster convergence + fair regularization
Small data  → fine, but coefs wobble; use regularization
High-dim    → works, but favour L1 for feature selection
Class imbalance → use class_weight='balanced' or resample
```

> ⚠️ Split **before** scaling. Fit the scaler on train only.

---

## 22. Evaluation

Adapted to classification — this is where we meet the four workhorses:

```text
  CONFUSION MATRIX  (ground truth rows × predicted cols)
               predicted
               No     Yes
     actual No  TN     FP
             Yes FN     TP
```

| Metric | Formula | Simple | Use | Avoid |
|---|---|---|---|---|
| Accuracy | (TP+TN)/total | % correct labels | balanced classes | imbalanced classes |
| Precision | TP/(TP+FP) | of predicted Positive, how many right | false positives costly (spam) | when FN is worse |
| Recall | TP/(TP+FN) | of actual Positive, how many caught | false negatives costly (disease) | when FP is worse |
| F1 | 2·P·R/(P+R) | balance of P & R | imbalanced, want both | need one over the other |
| ROC-AUC | area under ROC curve | ranking ability, threshold-free | comparing models | need calibrated probabilities |
| Log-loss | −Σ[y·log ŷ + (1−y)·log(1−ŷ)] | how good the *probabilities* are | calibrated probabilities matter | only care about labels |

**Loss ≠ Metric — the rule that trips everyone:**

```text
TRAINING OBJECTIVE  = minimize log-loss (cross-entropy)
EVALUATION METRIC   = accuracy / precision / recall / F1 / AUC (what you report)

sklearn does NOT maximize accuracy during training.
It minimizes log-loss. Those are different goals.
```

> Example: a model that outputs 0.51 for every positive and 0.49 for every negative gets ~100% accuracy at a 0.5 threshold, yet has terrible log-loss (probabilities are barely above coin-flip). Accuracy lied.

---

## 23. Failure Cases

```text
DATA            → class imbalance, outliers, missing values, leakage
MATHEMATICAL    → perfect separation → weights diverge (needs regularization)
                → multicollinearity → unstable weights
OPTIMIZATION    → learning rate too high (in saga) → diverges
GENERALIZATION  → non-linear reality → high bias
PRACTICAL       → reading coefficients as direct "importance" without check
```

---

## 24. Debugging

Model underperforming? Run this checklist:

```text
1. Probabilities near 0.5 everywhere?  → boundary weak; add features / interact
2. Predicted probabilities extreme?    → likely overconfident; check separation & C
3. Accuracy oddly high but F1 low?     → class imbalance; accuracy is misleading
4. Weights huge?                       → separation or too little regularization → smaller C
5. One category mispredicted a lot?    → imbalance → class_weight='balanced'
6. Test far worse than train?          → overfitting → smaller C / more data
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:   "predicts a number with a straight line"  → regression
Logistic Regression: "predicts a probability with an S-curve"   → classification
SVM:                 "finds the max-margin boundary, no probability by default"
Naive Bayes:         "uses probability rules + independence"   → generative
Decision Tree:       "learns if-then rules"                    → interpretable
KNN:                 "nearest neighbours vote"                 → lazy learner
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Logistic | linear score + sigmoid | interpretable, probabilistic, fast | linear boundary only | baseline, risk scoring |
| Linear Regression | straight line | simple | only numbers | regression |
| SVM | max-margin line (kernel) | strong high-dim boundary | no probs by default | high-dim, non-linear |
| Naive Bayes | Bayes + independence | tiny data, text | independence wrong | spam/text |
| Decision Tree | rules | interpretable rules | overfits | auditability |
| KNN | neighbour vote | no training | slow, curse of dim | small data |

> Master logistic as the *probabilistic baseline*; everything else is a trade of interpretability/speed for power.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  bank decides approve or reject a loan, with a risk score
DATA:              past 1000 applicants (income, credit_score, years_in_job)
FEATURES:          income, credit_score, debt_ratio (scaled)
TARGET:            default? 1/0
MODEL:             LogisticRegression(C=0.5, class_weight='balanced')
TRAIN:             split → scale → fit
EVALUATE:          ROC-AUC + precision/recall at chosen threshold
DEPLOY:            serve P(default); bank sets its own threshold
MONITOR:           track predicted default-rate drift monthly
```

> Same skeleton powers spam filters, medical risk scores, churn prediction, CTR.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what does the sigmoid do to z?
2. **Understand:** why can't we just use a straight line for classification?
3. **Calculate:** compute `σ(0)`, `σ(2)`, and the log-loss of a confident-wrong prediction.
4. **Apply:** given a scatter, decide if a linear decision boundary is reasonable.
5. **Debug:** models says "No" for everything at threshold 0.5. Likely cause & fix?
6. **Experiment:** run Experiment B at several `C` values; watch probability spread.
7. **Build:** spam mini-project — clean sms, vectorize, scale, fit, choose a threshold, write a one-line business rule from the coefficients.
8. **Explain:** explain logistic regression to a friend in 60 seconds using the SMS story.

---

## 28. Interview

### Beginner
- **Why "regression" if it's classification?** It regresses on the *log-odds* — a continuous quantity — then converts to a probability. The name is historical.
- **What's the sigmoid?** `σ(z)=1/(1+e⁻ᶻ)`, maps any real to (0,1), bounded and smooth.
- **What's the decision boundary?** `w·x + b = 0`, where `P = 0.5`. It's a hyperplane.
- **What loss do we use?** Binary cross-entropy / log-loss.

### Intermediate
- **Why not MSE?** MSE + sigmoid is non-convex (local minima). Cross-entropy + sigmoid is convex → unique global minimum.
- **Explain cross-entropy intuitively.** Confident *wrong* answers are punished more than hesitant ones; it measures how far predicted probabilities are from truth.
- **What does `C` do?** `C = 1/λ`; smaller C = stronger regularization. It fights weight explosion (esp. in separable problems).
- **Two ways to get probabilities vs labels?** `predict_proba` gives `[P(0), P(1)]`; `predict` applies threshold.

### Advanced
- **Derive the gradient.** With `dσ/dz = σ(1−σ)`, chain rule gives `∂J/∂w = (ŷ − y)·x`. (Full derivation, Section 30.)
- **MLE connection.** Minimizing log-loss = maximizing the Bernoulli likelihood. Same foundation as linear regression's least squares.
- **Logistic regression as a neural net.** It's a single neuron with sigmoid activation — the simplest possible neural network.
- **Perfect separation problem.** Without regularization, MLE weights diverge to infinity; the boundary becomes infinitely sharp.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
sigmoid:      σ(z) = 1 / (1 + e^(−z)),   z = w·x + b
logit:        ln(p / (1 − p)) = z
decision:     p ≥ 0.5 ⟺ z ≥ 0
log-loss:     J = −(1/n) Σ [ y·log(ŷ) + (1−y)·log(1−ŷ) ]
gradient:     ∂J/∂wⱼ = (1/n) Σ (ŷᵢ − yᵢ)·xⱼ⁽ⁱ⁾
sigmoid deriv: dσ/dz = σ(z)·(1 − σ(z))
```

**Common traps:**
- Confusing **sigmoid** (binary, one score) with **softmax** (multiclass, K scores).
- Forgetting logistic regression is a **discriminative** (not generative) model.
- Assuming it learns non-linear boundaries without feature engineering — it doesn't.
- Mixing up `C`: higher `C` = **less** regularization.
- `predict` vs `predict_proba` — one gives labels, one gives probabilities.

> **Representative pattern question (NOT a past GATE PYQ):** "If `σ(z) = 0.8`, find `z`." → `z = ln(0.8/0.2) = ln 4 ≈ 1.386`.
> Another: "A model gives ŷ=0.9 for a true label y=1. What's that sample's log-loss?" → `−log(0.9) ≈ 0.105`.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + geometry + multiclass</summary>

### Where the sigmoid comes from

Model the log-odds linearly:

```text
ln(p / (1−p)) = z
```

Exponentiate both sides:

```text
p / (1−p) = e^z   →   p = (1−p)·e^z
p = e^z − p·e^z
p(1 + e^z) = e^z
p = e^z / (1 + e^z) = 1 / (1 + e^(−z))   = σ(z)
```

The S-curve is forced by the log-odds assumption.

### Deriving the gradient

For one sample, `J = −[y·log σ + (1−y)·log(1−σ)]`. The sigmoid's special derivative:

```text
dσ/dz = σ(1 − σ)
```

Chain rule:

```text
∂J/∂w = (ŷ − y)·x            ← the elegant "error × input" result
∂J/∂b = (ŷ − y)
```

This is exactly what the code computes. No magic.

### Cross-entropy comes from Maximum Likelihood

`P(y|x) = ŷʸ·(1−ŷ)^(1−y)` (Bernoulli). Negative log-likelihood:

```text
−log P(y|x) = −[y·log ŷ + (1−y)·log(1−ŷ)]
```

Average over all samples = log-loss. Minimizing it = maximizing likelihood.

### Multiclass: softmax & One-vs-Rest

- **One-vs-Rest (OVR):** train K binary classifiers, one per class; pick the highest scoring.
- **Multinomial/softmax:** one network with K outputs, `softmax(zⱼ) = e^(zⱼ)/Σₖ e^(zₖ)`, single loss. Usually preferred.

### Perfect separation

If the classes are linearly separable, the optimum pushes probabilities to exactly 0/1 → `z → ±∞` → `w → ±∞`. Regularization (`C` small) keeps weights finite.

### Complexity

```text
training/epoch:  O(n·d)      prediction/sample: O(d)
stored model:    O(d)        multiclass: O(K·d)
```

### Why scaling matters here (unlike trees)

Gradient descent converges faster with standardized features, and L1/L2 penalties treat all weights fairly only if features share a scale.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "Some decisions are Yes/No but never certain. Logistic Regression squashes a linear score through an S-curve to output a probability, and learns the score that best fits your labeled examples."

> **Explain to a 12-year-old:** "You have a line, but you bend it into a smooth S. The S always stays between 0 and 1, so it can act like 'how sure I am.' They say spam when the S is above the middle."

> **Explain in an interview:** add: sigmoid formula, log-odds, cross-entropy vs MSE, gradient of `(ŷ−y)·x`, regularization, multiclass, perfect separation.

> **Explain the mathematics:** derive the sigmoid from log-odds and the gradient from `dσ/dz = σ(1−σ)`.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define logistic regression.
2. Explain its intuition with the spam story.
3. Write the sigmoid formula and interpret it.
4. Explain why we use cross-entropy, not MSE.
5. Compute one gradient-descent step by hand on tiny data.
6. Explain what's inside `fit()` (the loop).
7. List its assumptions and what each violation causes.
8. Discuss weight explosion under perfect separation.
9. Choose it for a real binary problem; defend the choice.
10. State one counter-example where you WOULDN'T use it.

---

## 33. Cheat Sheet

```text
Algorithm : Logistic Regression · Supervised → Classification · Discriminative
Goal      : model P(y=1|x) = σ(w·x + b), classify at threshold 0.5
Model     : σ(z) = 1/(1+e^(−z)),  z = w·x + b
Loss      : binary cross-entropy (log-loss)
Learn     : w (weights), b (bias) — via gradient descent
Tune      : C (regularization) · penalty · solver · max_iter · class_weight
Assumptions: linearity of log-odds · independence · no multicollinearity
Use when  : binary/multiclass, need calibrated probabilities, interpretable baseline
Avoid when: complex non-linear boundaries without engineered features
Multiclass: softmax (multinomial) or One-vs-Rest
Related   : Linear Regression (same linear core) · SVM · Naive Bayes
Baseline  : every classifier is compared against this
```

---

## 34. What Next?

You just built the foundation of discriminative classification.

```text
Logistic Regression
   ├── K-Nearest Neighbors   (no model; neighbours vote)       → next note (02)
   ├── Naive Bayes           (probability + independence)      → 03
   └── Decision Tree         (rules / if-then)                 → 04
```

> Next recommended: **02. K-Nearest Neighbors (KNN)** — it flips everything: no training loop at all, prediction by *distance* to remembered examples.
