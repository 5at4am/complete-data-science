# 01. Linear Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐☆☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **problem → pattern → guess → line → error → formula → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Linear Regression is the **first model everyone learns** — not because it's weak, but because every other model in ML is compared against it.

By the end you will be able to:

- predict a number from a straight-line pattern,
- explain *why* the line is "best",
- compute it by hand on tiny data,
- code it both from scratch and with sklearn,
- break it deliberately and fix it,
- and defend when to use — and not use — it.

> Everything in this note builds on one small idea. Let's find it.

---

## 02. The Problem

Ankit is joining a company. HR shows you the salaries of the last five people who joined the same role:

| Years of experience | Salary (₹ lakh/yr) |
|---|---|
| 1 | 5.0 |
| 2 | 5.8 |
| 3 | 6.5 |
| 4 | 7.2 |
| 5 | 8.0 |

<!-- [QUESTION] -->
Now the question:

> **Ankit has 3.5 years of experience. What salary would you predict for him?**

Don't scroll straight to the answer. Make your best guess first. Write your number down.

**Your guess: ₹ ____ lakh/year**

> 📌 Keep this number in your head. At the end of Section 06 we'll compare it with what the model says.

---

## 03. Let's Think

Before predicting, let's actually look at the data.

```text
Experience  →  Salary
1           →  5.0
2           →  5.8     (+0.8)
3           →  6.5     (+0.7)
4           →  7.2     (+0.7)
5           →  8.0     (+0.8)
```

<!-- [THINK_ABOUT_IT] -->
🤔 What do you notice?

> Every extra year adds roughly **0.7–0.8 lakh**. There is a trend.

And a second observation — probably the most important in all of ML:

> There is **no row for 3.5 years** in the table. Yet you can still make a reasonable guess.

Why? Because you spotted a **pattern**, and you projected it forward. That is exactly what a predictive model does.

> The pattern here looks like: salary = base amount + something × experience.

That "base amount + something × experience" is a **straight line**.

---

## 04. Intuition

If we plot the five employees as dots, they almost sit on a straight line:

<!-- [VISUAL] -->
```text
Salary (₹ lakh)
  8 ┤                              • (5, 8.0)
  7 ┤                        • (4, 7.2)
  6 ┤                  • (3, 6.5)
    │            • (2, 5.8)
  5 ┤      • (1, 5.0)
    └────────────────────────────────────── Experience
           1      2      3      4      5
```

💡 **The idea in one line:**

> Linear Regression draws **the one straight line that fits the dots the best**, then uses that line to predict values we've never seen.

No magic. No memorizing five salaries. Just: find the line → read the answer off it.

---

## 05. Visual First

A line is described completely by two numbers:

```text
y = w·x + b

w  → slope:     how steep the line is          (~0.75 here: +0.75 lakh per extra year)
b  → intercept: the value when x = 0           (~4.2 here: the "base" salary)
```

```text
      y
      │                          • (5, 8.0)
  8 ──┤                          ⁄
      │                        •
  7 ──┤                      ⁄
      │                    •
  6 ──┤                  ⁄
      │                •     ← real data
  5 ──┤              ⁄
      │        •    ⁄               straight line = model
      └────────────────────────── x
               1  2  3  4  5
```

The **gap** between each dot and the line is how wrong that point is. That gap has a name — we'll meet it properly in a moment.

> 📌 A *higher* slope = steeper line = experience matters more. A *higher* intercept = the whole line shifts up.

---

## 06. First Prediction

Which line is "best"? There are thousands of possible lines through these five dots. Let's think about what "best" should even mean.

> 💡 Natural answer: the line that **misses all the points by the least amount, overall**.

Using our eyeball line (`slope ≈ 0.75`, `intercept ≈ 4.2`), the prediction for 3.5 years would be:

```text
ŷ = 0.75 × 3.5 + 4.2 = 2.625 + 4.2 = 6.825  →  ₹6.8 lakh
```

<!-- [TRY_IT] -->
Did the model's answer come close to **your** guess from Section 02?

> 📌 If you said 6.5–7.0 lakh, your intuition already agrees with Linear Regression. The math that follows only makes this intuition **exact and repeatable**.

Now the honest problem:

> **How do we decide the "best" line without eyeballing?**

That leads to error. Sit tight — next section.

---

## 07. Core Concept

Introducing the idea formally, right after we've already met it:

**Concept: Linear Regression** — a method that:

1. assumes the target `y` is a **linear combination** of a feature `x` (or many),
2. finds the slope `w` and intercept `b` that make the total **squared error** as small as possible,
3. uses that line to predict new values.

```text
PREDICTION  →  ŷ = w·x + b
```

Two parts to the model:

| Part | Symbol | Simple meaning |
|---|---|---|
| Slope | `w` | How strongly each unit of `x` changes the prediction |
| Intercept | `b` | The prediction when `x = 0` |

> Everything else (error, RSS, OLS, the formulas) is just **making these two numbers good**.

---

## 08. Terminology

Each term below *emerges* from the story we just told:

### Prediction (ŷ)

> Simple: the model's answer.
> Technical: the estimated value of the target produced by the learned model.

### Feature (x)

> Simple: the thing we know — experience, hours studied, house size.
> Technical: an independent / predictor variable.

### Target (y)

> Simple: the number we want — salary, marks, price.
> Technical: the dependent / response variable.

### Residual (error)

> Simple: how far the model missed.
> Technical: `actual − predicted = y − ŷ`.

### Fit / Training

> Simple: finding the best slope and intercept.
> Technical: estimating parameters by minimizing the chosen objective.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| ŷ | model's answer | estimated target |
| x | what we know | feature / predictor |
| y | what we want | target / response |
| w | how strong the effect is | coefficient, slope |
| b | starting point | intercept / bias |
| y − ŷ | how far we missed | residual, error |
| RSS | total squared miss | Residual Sum of Squares |
| OLS | the fitting procedure | Ordinary Least Squares |

> ⚠️ Common mistake: "ŷ is just y." No — `y` is the truth, `ŷ` is the guess.

---

## 09. Mathematics (gradual)

We build the math from zero. Four small steps.

### Step M1 — The line

```text
ŷ = w·x + b
```

Every symbol, given a human meaning *before* the formula was shown in Section 07.

### Step M2 — The error of ONE point

For a single data point:

```text
error = y − ŷ
```

> Our model predicted ₹6.8 lakh for Ankit. If his real salary turns out ₹7.0 lakh, the error is `7.0 − 6.8 = +0.2 lakh`.

### Step M3 — The problem with just adding errors

Now here's a trap.

```text
point 1 error:  +0.5
point 2 error:  −0.5
sum of errors:   0     ← looks perfect, but the line missed BOTH points!
```

Positive and negative errors **cancel each other**. A terrible line can score 0.

> So we need a smarter way to add them up.

### Step M4 — Why we square

Square each error first:

```text
+0.5 → +0.25
−0.5 → +0.25
sum  → +0.50     ← now both misses are counted
```

Squaring gives us three wins:

1. **Nothing cancels** — negatives become positive.
2. **Big misses hurt more** — a miss of 5 becomes 25, a miss of 1 becomes 1.
3. The math stays **smooth** (differentiable), so we can find the minimum with calculus.

### The objective — RSS

```text
RSS = Σᵢ (yᵢ − ŷᵢ)²
```

```text
Σ      → "sum over all data points"
yᵢ     → actual value of point i
ŷᵢ     → prediction of point i
(yᵢ − ŷᵢ)² → squared residual of point i
```

**Finding the line with the minimum RSS is called Ordinary Least Squares (OLS).**

### The answer (for one feature) — closed form

The best `w` and `b` have a direct formula:

```text
w = Σᵢ (xᵢ − x̄)(yᵢ − ȳ) / Σᵢ (xᵢ − x̄)²
b = ȳ − w·x̄
```

```text
x̄      → mean of all x
ȳ      → mean of all y
Σᵢ(xᵢ−x̄)(yᵢ−ȳ) → how x and y move together (covariance)
Σᵢ(xᵢ−x̄)²     → how spread out x is (variance)
```

> 💡 Intuition: the **numerator** is "do x and y rise together?" (→ slope sign). The **denominator** normalizes it. The intercept is forced so that the line passes through the average point `(x̄, ȳ)`.

---

## 10. Numerical Example

Take a tiny dataset we can check **on paper** (pretend numbers, just for math):

```text
x = [1, 2, 3]      (hours studied)
y = [2, 4, 5]      (test score)
```

<!-- [CALCULATION] -->

**Step 1 — Means**

```text
x̄ = (1 + 2 + 3) / 3 = 2.0
ȳ = (2 + 4 + 5) / 3 = 3.667
```

**Step 2 — Build the table**

| x | y | x − x̄ | y − ȳ | (x−x̄)(y−ȳ) | (x−x̄)² |
|---|---|---|---|---|---|
| 1 | 2 | −1 | −1.667 | +1.667 | 1 |
| 2 | 4 | 0 | +0.333 | 0 | 0 |
| 3 | 5 | +1 | +1.333 | +1.333 | 1 |

**Step 3 — Slope**

```text
w = (1.667 + 0 + 1.333) / (1 + 0 + 1) = 3.0 / 2.0 = 1.5
```

**Step 4 — Intercept**

```text
b = ȳ − w·x̄ = 3.667 − 1.5 × 2.0 = 0.667
```

**Step 5 — The fitted line**

```text
ŷ = 1.5·x + 0.667
```

**Step 6 — Check it**

```text
x=1 → 1.5+0.667 = 2.167      actual 2    miss = −0.167
x=2 → 3.0+0.667 = 3.667      actual 4    miss = +0.333
x=3 → 4.5+0.667 = 5.167      actual 5    miss = −0.167
```

```text
RSS = (−0.167)² + (0.333)² + (−0.167)² = 0.028 + 0.111 + 0.028 = 0.167
```

> ✅ VERIFIED — the formula gives the line that minimizes RSS. (Hand-computed; checks with Section 16's experiment.)

**Predict something new:**

```text
x = 2.5  →  ŷ = 1.5 × 2.5 + 0.667 = 4.417
```

<!-- [TRY_IT] -->
🎯 Your turn: predict `x = 4` with this line before reading on.

> Answer: `1.5 × 4 + 0.667 = 6.667`. Take a second — this *is* the model now. Give it any x, it multiplies by 1.5 and adds 0.667.

---

## 11. How It Works

```text
STEP 1   Have data (x, y)
STEP 2   Decide the shape: a straight line  ŷ = w·x + b
STEP 3   Define "wrong": RSS = Σ(y − ŷ)²
STEP 4   Find w, b with the smallest RSS   ← OLS
STEP 5   Production: new x → ŷ = w·x + b
```

If Chapter 09 was clear, Steps 3–4 are the only "mathematical" ones — and even they reduce to one formula.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
This is the section that makes sklearn **unmagical**.

```text
model.fit(X, y)
     ↓
1. Check shapes & data validity
     ↓
2. Build the design matrix  [1 x₁ x₂ …]   (column of 1s ➝ absorbs intercept)
     ↓
3. Solve  w = (XᵀX)⁻¹·Xᵀ·y                (the normal equation)
     ↓
4. Store result:  coef_  +  intercept_
     ↓
5. Model is now just two things:  weights + a bias
```

```text
model.predict(X_new)
     ↓
for each new row:
    ŷ = X_new · weights + bias
```

> (Note: plain OLS is a *one-shot* calculation — no training loop, no epochs. That's why it's ultrafast. Gradient descent, which loops, appears later.)

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
def fit_ols(xs, ys):
    n = len(xs)
    x_bar = sum(xs) / n
    y_bar = sum(ys) / n

    numerator = 0.0
    denominator = 0.0
    for i in range(n):
        numerator += (xs[i] - x_bar) * (ys[i] - y_bar)
        denominator += (xs[i] - x_bar) ** 2

    w = numerator / denominator
    b = y_bar - w * x_bar
    return w, b

def predict(xs, ys, x_new):
    w, b = fit_ols(xs, ys)
    return w * x_new + b

print(fit_ols([1, 2, 3], [2, 4, 5]))      # (1.5, 0.667)
print(predict([1, 2, 3], [2, 4, 5], 2.5)) # 4.417
```

> This is *literally* the formula from Section 09, line by line.

### Version 2 — numpy, vectorized, many features

```python
import numpy as np

def fit_ols_vec(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    ones = np.ones((X.shape[0], 1))
    X_design = np.hstack([ones, X])         # add the column of 1s
    theta = np.linalg.inv(X_design.T @ X_design) @ X_design.T @ y
    return theta[0], theta[1:]              # (b, w...)

def predict_vec(X, b, w):
    return X @ w + b

print(fit_ols_vec([[1], [2], [3]], [2, 4, 5]))  # (0.667, array([1.5]))
```

Same math. Vectorized version works for any number of features.

### Version 3 — clean class (what a library-style API looks like)

```python
import numpy as np

class LinearRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ones = np.ones((X.shape[0], 1))
        Xd = np.hstack([ones, X])
        theta = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y
        self.b, self.w = theta[0], theta[1:]

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.w + self.b
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1], [2], [3]])
y = np.array([2, 4, 5])

model = LinearRegression()       # creates an empty model object
model.fit(X, y)                  # solves w = (XᵀX)⁻¹Xᵀy internally

print(model.coef_)               # [1.5]     → the slope(s)
print(model.intercept_)          # 0.6667    → the intercept
print(model.predict([[2.5]]))    # [4.4167]  → new predictions
```

> `model.coef_` = our `w`. `model.intercept_` = our `b`. sklearn did **exactly** what Section 13's Version 3 did — just faster, validated, and battle-tested.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
Xd = np.hstack([ones, X])
```
> Takes `[[1],[2],[3]]` → `[[1,1],[1,2],[1,3]]`. Why? So `b` becomes just another weight in one matrix equation. Now `ŷ = Xd @ theta` handles slope **and** intercept together.

```python
theta = np.linalg.inv(Xd.T @ Xd) @ Xd.T @ y
```
> This is the **normal equation** `(XᵀX)⁻¹Xᵀy` from Section 30's derivation — the closed-form least-squares answer. `Xd.T @ Xd` builds the covariance-like matrix; `inv` inverts it; the rest projects y onto the feature space.

```python
return X @ self.w + self.b
```
> That's the model equation `ŷ = w·x + b` — a dot product (all weights × all features) plus the intercept.

> 🧠 Every line maps to a formula we already wrote by hand. Nothing in the code is arbitrary.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> If this note is rendered inside the interactive platform, these become sliders. Otherwise, run them in Python and observe.

### Experiment A — slide the slope and intercept

Imagine a slider for `w` and a slider for `b`, with the five salary dots behind:

```text
w too low (0.2)  →  line too flat, misses all high-salary points
w too high (2.0) →  line too steep, overshoots the top points
w ≈ 0.75         →  best fit
b too high (6)   →  line starts above the data, everything over-predicted
```

> What to notice: the **RSS number** is minimized exactly when the visual fit looks right. Eyes and math agree.

### Experiment B — the noise experiment (code)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

rng = np.random.default_rng(42)
X = np.linspace(0, 10, 50).reshape(-1, 1)
y = 2.0 * X.ravel() + 5 + rng.normal(0, 1, 50)   # true line 2x+5 + small noise

for noise in [0.1, 1.0, 5.0]:
    y_noisy = 2.0 * X.ravel() + 5 + rng.normal(0, noise, 50)
    m = LinearRegression().fit(X, y_noisy)
    print(f"noise={noise:>3}  →  slope≈{m.coef_[0]:.3f}  intercept≈{m.intercept_:.3f}")
```

```text
noise=0.1 → slope≈2.000  intercept≈5.000   ← recovers the truth almost exactly
noise=1.0 → slope≈1.9xx  intercept≈5.1xx
noise=5.0 → slope≈1.8xx  intercept≈5.4xx   ← wobbles more
```

> 📌 The moral: least squares **averages out noise**. More noise → less certain answer. This is the seed of the whole idea of "model uncertainty."

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2, 4, 5, 4, 6])

model = LinearRegression().fit(X, y)                  # line A
print(model.coef_, model.intercept_)

y_broken = np.array([2, 4, 5, 4, 600])                # ONE huge outlier
model_broken = LinearRegression().fit(X, y_broken)    # line B
print(model_broken.coef_, model_broken.intercept_)
```

```text
Line A (normal):    slope ≈ 0.8, intercept ≈ 2.0
Line B (outlier):   slope ≈ 120,  intercept ≈ −190   ← wild
```

**What happened?** One bad point contributed `(600 − ŷ)²`, which is so large it crushed every other point's voice. The line went out of its way to please the liar.

> 💥 **Break pattern:** normal model → add one outlier → model flips. Why? **Squared error gives huge weight to huge misses.**

Now the key teaching step — don't fix yet, understand:

- Does **deleting** the point fix it? Yes, but why was it there?
- Does **capping** it help? Yes (robust regression trick, Section 19's Huber).
- **Lesson:** outliers are not "just noise" — they hijack squared loss.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Double the data | Line stabilizes toward the truth | More evidence → noise averages out |
| Add one huge outlier | Line tilts violently | Squared error amplifies the outlier |
| Data is genuinely curved | Line underfits (`high bias`) | A straight line can't curve |
| Add a second identical feature | Coefficients explode / become unstable | Math can't separate identical effects (multicollinearity) |
| Feature in "inches" vs "metres" | Coefficient changes by 12× | Units scale the slope — the *meaning* is the same |
| Target becomes Yes/No | Wrong tool | That's classification → Logistic Regression |

> 🤔 Think: which one is (surprisingly) *not* fixed by more data? → The curved one. No amount of data helps a straight line fit a curve. Memorize that — it's the "high bias" idea.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w   → one coefficient per feature      (model.coef_)
b   → the intercept                    (model.intercept_)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `fit_intercept` | Learn the `b` term? | — | False forces line through origin | `True` |
| learning rate α *(only if using gradient descent)* | Step size of updates | Very slow | Diverges | 0.01–0.1 |
| epochs *(only GD)* | Number of update rounds | Not converged | Waste of time | stop on plateau |

> 📌 Plain OLS has almost no hyperparameters — a big reason it's a clean baseline. Real tuning appears with gradient descent and regularization later.

---

## 20. Assumptions

For each: what, why, how to check, what if violated.

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Linearity** | y ≈ linear function of features | The model is a line | scatter / residual-vs-fitted plot | polynomial features, or a different family |
| **Independence** | samples don't affect each other | statistics assume it | domain knowledge | time-series or mixed models |
| **Homoscedasticity** | error spread is constant | OLS assumes it | residual plot funnel = bad | weighted least squares / transform y |
| **Normality of errors** | residuals ~ Normal | needed for p-values/CIs | Q-Q plot | predictions still fine; robust inference |
| **No multicollinearity** | features aren't near-duplicates | makes `(XᵀX)⁻¹` unstable | correlation matrix, VIF | Ridge/Lasso or drop features |
| **Few/no outliers** | no extreme points | they hijack squared loss | boxplots, residuals | Huber, or clean data |

> For pure **prediction**, the first (linearity) matters most. The rest mostly matter for **inference** (p-values, confidence intervals). Good to know for interviews.

---

## 21. Data Requirements

```text
Target      → continuous numeric only (else → classification)
Features    → numerical; categorical must be encoded (one-hot)
Missing     → must be handled first (impute or drop) — OLS can't take NaN
Outliers    → painful (Section 17). Investigate before modeling
Scaling     → optional for closed form; RECOMMENDED for gradient descent & comparing coefficients
Small data  → fine (few params), but coefficients wobble
High-dim   → works, but favours Ridge/Lasso when features are many/correlated
```

> ⚠️ Data-leakage trap: **split BEFORE scaling.** Fit the scaler on the training set only, then transform both. Scaling on all data lets test-set statistics leak into training.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (minimize RSS / squared error)
        ≠
EVALUATION METRIC   (what you report to a manager)
```

| Metric | Formula | Simple | Use | Avoid |
|---|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | avg squared miss | standard loss | units are "squared" — odd to report |
| RMSE | √MSE | avg miss, in ₹ | most common | outliers dominate the view |
| MAE | (1/n)Σ\|y−ŷ\| | avg abs miss | robust, interpretable | when big misses must hurt → RMSE |
| R² | 1 − SS_res/SS_tot | % of variance explained | model quality, same dataset | comparing across datasets |

> Misconception to avoid: **R² can't tell you "good" by itself.** R² = 0.9 on noisy data may be useless for decisions; R² = 0.4 on a hard problem may be strong. Context decides.

---

## 23. Failure Cases

```text
DATA            → outliers, missing values, leakage
MATHEMATICAL    → XᵀX singular (perfect multicollinearity) → normal equation fails
OPTIMIZATION    → (GD mode only) learning rate too high → diverges
GENERALIZATION  → curved truth → high bias; extrapolating outside the data range
PRACTICAL       → interpreting coefficients on unscaled features; claiming causation
```

---

## 24. Debugging

Model performs badly? Run this checklist in order:

```text
1. Residual plot → curved pattern?      → nonlinear → need polynomial/other model
2. Residual plot → funnel shape?        → heteroscedasticity → transform y / WLS
3. Predictions all systematically low?  → intercept missing / scaling bug / unit error
4. One residual dominates?              → outlier → investigate & handle
5. Coefficients look insane?            → multicollinearity → VIF, Ridge
6. R² high on train, low on test?       → overfitting (rare here; check leakage!)
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:   "I'll fit one straight relationship."
Polynomial:          "I'll allow the line to bend."
Ridge:               "Same line, but keep weights small to stay stable."
Lasso:               "Same line, but force useless weights to ZERO."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Linear | straight line, min squared error | simple, interpretable, baseline | can't curve, outlier-sensitive | start here |
| Polynomial | add x², x³… | captures curvature | overfits easily | curved trends |
| Ridge | line + L2 penalty | stable, handles collinearity | keeps all features | many correlated features |
| Lasso | line + L1 penalty | auto feature selection | no closed form | many features, few matter |
| Huber | robust loss | resists outliers | extra tuning | outlier-heavy data |

> Everything in this table is "Linear Regression + one change." Master the base, and these become quick upgrades.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict hourly earnings of delivery riders by distance
DATA:              past 200 rides (distance_km, earning_₹)
FEATURES:          distance, time_of_day (encoded)
TARGET:            earning_₹
MODEL:             LinearRegression
TRAIN:             split→scale→fit
EVALUATE:          RMSE ₹/km + residual plot
DEPLOY:            serve ŷ on ride app
MONITOR:           check predictions drift as market/pricing changes
```

Same skeleton powers house price, sales forecast, fuel usage, electricity bill models.

> 🚀 ML is not `model.fit(X, y)`. It's problem → data → features → model → evaluate → deploy → monitor → repeat.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is a residual?
2. **Understand:** why do we square errors instead of summing raw residuals?
3. **Calculate:** compute `w`, `b`, RSS by hand for `(1,2),(2,4),(3,5)`.
4. **Apply:** given a scatter plot, decide if linear regression is appropriate.
5. **Debug:** test RMSE spikes only near high feature values — what's the likely cause?
6. **Experiment:** run the noise experiment (Section 16) at 5 noise levels; graph slope instability.
7. **Build:** house price mini-project: EDA → linearity check → outlier handling → fit → evaluate → one-line business summary of the main coefficient.
8. **Explain:** explain linear regression to a friend in 60 seconds using only the salary story.

---

## 28. Interview

### Beginner
- **What is linear regression?** A supervised model that predicts a continuous target as a weighted sum of features, chosen to minimize squared error.
- **What is a residual?** `y − ŷ`. How far the prediction missed.
- **Interpret slope and intercept.** Slope = change in ŷ per unit of x. Intercept = ŷ when x = 0.

### Intermediate
- **Why squared error, not absolute?** Differentiable everywhere, convex (single minimum), punishes big misses, yields a closed-form solution. Absolute error is robust but non-differentiable at 0.
- **What's the normal equation?** `w = (XᵀX)⁻¹Xᵀy`. The exact least-squares solution in one step.
- **What is multicollinearity and why does it hurt?** Near-duplicate features make `XᵀX` nearly singular → unstable, inflated coefficients.
- **Normal equation vs gradient descent?** Closed form: few features (inversion is O(m³)). Gradient descent: very many features or huge data.

### Advanced
- **What's the MLE interpretation?** If `y = w·x + b + ε`, `ε ~ N(0, σ²)`, then maximizing the likelihood of the data = minimizing RSS. This justifies p-values and confidence intervals.
- **Why is RSS convex?** It's a quadratic form in `w, b` with a positive-definite Hessian → exactly one global minimum.
- **How do outliers asymmetrically affect OLS?** Squared loss gives quadratic weight to a single residual; one outlier can dominate the fitted line, while robust losses cap its influence.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
w = Σ(x−x̄)(y−ȳ) / Σ(x−x̄)²
b = ȳ − w·x̄
R² = 1 − SS_res / SS_tot
slope = r · (SD_y / SD_x)          on standardized data → slope = r
```

**Common traps:**
- Forgetting to **center** when computing covariance by hand.
- Thinking R² measures general "goodness" — it's in-sample linear fit only.
- Confusing slope with correlation — equal only on standardized variables.
- Outlier effects: with squared loss, one far-out point can flip the line.

> **Representative pattern question (NOT a past GATE PYQ):** "Fit the least-squares line to (1,2), (2,3), (3,5)." → x̄=2, ȳ=3.333, numerator=3, denominator=2 → **w=1.5, b=0.333**.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + matrix form + theory</summary>

### The derivation

Minimize `J(w,b) = Σ(yᵢ − b − w·xᵢ)²`.

**∂J/∂b = 0:**

```text
−2 Σ(yᵢ − b − w·xᵢ) = 0  →  b = ȳ − w·x̄
```

**∂J/∂w = 0 (substitute b):**

```text
w = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)²
```

- Partial derivative = "how J changes as I nudge one variable alone."
- Setting it to zero finds the bottom of the bowl → the minimum.
- Because J is **convex**, the bottom is global — no local-minimum traps.

### Matrix form (many features)

```text
X_design = [1 x₁ x₂ … xₘ]      w = (XᵀX)⁻¹ Xᵀ y
```

Requires `XᵀX` invertible (no perfect multicollinearity). Otherwise: gradient descent or regularization.

### Bias–variance

```text
too simple  → underfit → high bias  (curved truth, straight line)
too complex → overfit  → high variance (polynomial of degree 9 on 10 points)
```

Plain linear regression: low variance, potentially high bias.

### Probabilistic view

```text
y = w·x + b + ε ,  ε ~ N(0, σ²)
L = Π P(yᵢ|xᵢ)  →  maximize L  ⇔  minimize RSS
```

This single equivalence links OLS to statistics (p-values, CIs, hypothesis tests).

### Complexity

```text
closed form training:  O(n·m² + m³)      prediction/sample: O(m)
gradient descent/epoch: O(n·m)            stored model: O(m)
```

Closed form wins on small/medium data; GD wins when `m` or `n` is huge.

### Feature scaling

For the closed form it's optional. For gradient descent it's **recommended**: unscaled features produce distorted step sizes → slow/twisted convergence. Standardize (z-score) or Min-Max.

</details>

---

## 31. Teach Back

Try all four. If any is hard, re-read the matching section.

> **Explain in 30 seconds:** "We noticed a straight-line pattern in salary data. Linear Regression finds the line that minimizes how far it misses all points, then predicts new values off that line."

> **Explain to a 12-year-old:** "You have dots on a paper. Draw the straight line that comes closest to all the dots. That line is the prediction machine."

> **Explain in an interview:** add: closed-form `w=(XᵀX)⁻¹Xᵀy`, assumptions, when to use GD, outliers, multicollinearity.

> **Explain the mathematics:** derive `b = ȳ − w·x̄` and w's formula from Section 30.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define linear regression.
2. Explain its intuition with the salary story.
3. Write and justify the core equation.
4. Compute `w`, `b` by hand for 3 points.
5. Explain what's inside `fit()`.
6. List its assumptions — and what each violation causes.
7. Explain when it fails (outlier, curve, collinearity).
8. Compare with 3 alternatives.
9. Choose it for a real problem; defend the choice.
10. State one counter-example where you WOULDN'T use it.

---

## 33. Cheat Sheet

```text
Algorithm : Linear Regression · Supervised → Regression · Parametric
Goal      : minimize RSS = Σ(y − ŷ)²
Model     : ŷ = w·x + b    (matrix: w = (XᵀX)⁻¹Xᵀy)
Learn     : w (weights), b (bias)
Tune      : fit_intercept · (α, epochs only in GD mode)
Assumptions: linearity, independence, homoscedasticity, normality (inference), no multicollinearity
Use when  : continuous target, roughly linear, need interpretability/baseline
Avoid when: curved data, heavy outliers, multicollinearity, categories (→ logistic)
Related   : Polynomial · Ridge · Lasso · Elastic Net · Bayesian · Huber · Logistic
Baseline  : every other regressor is compared against this
```

---

## 34. What Next?

You just built the foundation of all linear models.

```text
Linear Regression
   ├── Polynomial   (the line bends)      → next note (02)
   ├── Ridge        (L2 penalty)          → 03
   ├── Lasso        (L1 penalty)          → 04
   ├── Elastic Net  (both penalties)      → 05
   ├── Bayesian     (prior on weights)    → 06
   ├── Huber        (outlier-proof loss)  → 07
   └── Logistic     (the same idea for classification)  → B-classification/01
```

> Next recommended: **02. Polynomial Regression** — it answers the one weakness you saw today: "what if the line needs to bend?"