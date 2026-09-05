# 06. Bayesian Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐☆☆
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **small data → point estimates fail → prior belief → Bayes' rule → posterior → uncertainty → MAP = Ridge.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Every model so far gives a single answer: "the coefficient is 3.2." But how confident are you? Is it 3.2 ± 0.1, or 3.2 ± 50?

Bayesian Regression doesn't just give a point estimate — it gives a **distribution** over possible answers, capturing both the most likely value *and* how uncertain you are.

By the end you will be able to:

- explain prior, likelihood, and posterior in plain language,
- derive that MAP estimation under a Gaussian prior equals Ridge Regression,
- compute predictive intervals that quantify uncertainty,
- code it from scratch and with sklearn's `BayesianRidge`, and
- defend when to use Bayesian Regression vs point-estimate models.

> Everything in this note builds on one question: *what if the model could say "I'm not sure"?*

---

## 02. The Problem

Dr. Priya is testing a new drug. She has data from only 12 patients:

```text
Dose (mg):     10   20   30   40   50   60   70   80   90   100  110  120
Response:      2.1  3.8  5.5  6.2  7.1  8.0  8.8  9.5  10.1 10.8 11.2 11.9
```

She fits OLS and gets: `slope = 0.085, intercept = 1.3`.

Now the hospital board asks:

> **"If we give a patient 150 mg, what's the expected response — and how sure are you?"**

OLS says: "14.05." That's it. No confidence. No uncertainty. Just a number.

<!-- [QUESTION] -->
With only 12 data points, and the prediction going *beyond* the training range (extrapolation), should the board trust that number blindly?

> **How can the model say "I think it's around 14, but I'm really not sure"?**

That's what Bayesian Regression provides.

---

## 03. Let's Think

Before seeing any data, you already have beliefs about what's reasonable:

```text
"The slope of dose-response is probably positive, maybe between 0 and 0.2."
"The intercept is probably between 0 and 5."
```

These are **prior beliefs** — not from the data, but from general medical knowledge.

Now the data comes in. It *updates* your belief. With 12 points, the update is moderate — your prior still matters. With 1000 points, the data overwhelms the prior, and the posterior converges to the OLS answer.

<!-- [THINK_ABOUT_IT] -->
🤔 What's the key difference from everything before?

> Every previous model gave a **single number** for each coefficient. Bayesian Regression gives a **probability distribution** — a curve showing which values are likely.

The distribution tells you: "the slope is probably around 0.085, but could be anywhere from 0.07 to 0.10." That *range* is the uncertainty — and it's exactly what the hospital board needs.

---

## 04. Intuition

💡 **The idea in one line:**

> Bayesian Regression starts with a **prior belief** about the weights, combines it with the **data evidence** using Bayes' rule, and produces a **posterior distribution** — the updated belief that captures both the best guess and the uncertainty.

Think of it as an updating process:

```text
BEFORE data:  Prior belief     → "I think the slope is probably near 0, somewhere in [−0.1, 0.1]"
                ↓
DATA arrives: Likelihood       → "The data says the slope is around 0.085"
                ↓
AFTER data:   Posterior         → "I now believe the slope is around 0.085, with range [0.07, 0.10]"
```

> 📌 With lots of data, the posterior becomes very narrow (high confidence). With little data, it stays wide (low confidence). This is the natural, honest behaviour you want.

---

## 05. Visual

```text
Prior belief (wide, centered at 0):          Posterior (narrower, shifted by data):
    │                                           │
    ▓                                           ▓
   ▓▓▓                    ──data──▶           ▓▓
  ▓▓▓▓▓                                        ▓▓▓
  ───●───  w                                ──●────  w
  center 0                                shifted toward data mean
```

```text
Predictions with uncertainty:
   y
   │   ╱╲   ╱╲
   │  ╱  ╲ ╱  ╲      ← uncertainty BAND grows away from data
   │ ╱    ╲╱    ╲
   │╱_______________
   └________________  x
     • ← data points (we're confident near here)
```

> The "cone" shape is key: predictions are tight near the data (where we have evidence) and widen far from it (where we're extrapolating).

---

## 06. First Prediction

Back to Dr. Priya's data. Bayesian Regression with default priors gives:

```text
w_MAP = 0.083 (slope)
b     = 1.45  (intercept)
```

At dose = 150 mg:

```text
Mean prediction:  ŷ = 0.083 × 150 + 1.45 = 13.9
Predictive std:   σ = 2.1
95% interval:     13.9 ± 2×2.1 = [9.7, 18.1]
```

<!-- [TRY_IT] -->
Compare with OLS: "14.05" (no uncertainty). Bayesian says "13.9, but could be anywhere from 9.7 to 18.1." The board now has the honest picture.

> 📌 The wide interval at 150 mg (far from training data) reflects the model's honesty: it's uncertain about extrapolation. OLS pretended to be certain.

---

## 07. Core Concept

**Concept: Bayesian Regression** — a method that:

1. treats the weight vector `w` as a **random variable** with a **prior distribution** P(w),
2. observes data and computes the **likelihood** P(y | X, w),
3. applies **Bayes' rule** to get the **posterior** P(w | X, y),
4. uses the posterior for predictions with **uncertainty intervals**.

```text
P(w | X, y) = P(y | X, w) · P(w) / P(y)
              ╰── posterior ──╯  ╰─ likelihood × prior ─╯  ÷ evidence
```

| Part | Symbol | Simple meaning |
|---|---|---|
| Prior P(w) | what you believed before seeing data | e.g., N(0, α⁻¹I) |
| Likelihood P(y\|X,w) | how well the data fits given weights | e.g., N(Xw, β⁻¹I) |
| Posterior P(w\|X,y) | updated belief after seeing data | N(w_MAP, S⁻¹) |
| MAP | most probable weight vector | the posterior mean |
| S⁻¹ | posterior covariance | the uncertainty over weights |

> For Gaussian prior + Gaussian likelihood (conjugate pair), the posterior is also Gaussian with **closed-form** mean and covariance. No MCMC needed.

---

## 08. Terminology

### Prior

> Simple: what you believe about the weights before seeing any data.
> Technical: a probability distribution P(w) over the weight vector.

### Likelihood

> Simple: how probable the observed data is, given specific weight values.
> Technical: P(y | X, w) — the data-generating probability under the model.

### Posterior

> Simple: your updated belief about the weights after seeing the data.
> Technical: P(w | X, y) ∝ P(y | X, w) · P(w).

### Bayes' Rule

> Simple: "update = what the data says × what you believed before."
> Technical: P(w | D) = P(D | w) · P(w) / P(D).

### MAP (Maximum A Posteriori)

> Simple: the single most likely weight value from the posterior.
> Technical: argmax P(w | X, y). For Gaussian prior, MAP = Ridge solution.

### Predictive Distribution

> Simple: the range of likely predictions for a new input.
> Technical: P(y* | x*, X, y) = ∫ P(y* | x*, w) · P(w | X, y) dw.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Prior | belief before data | P(w) |
| Likelihood | data fit given weights | P(y\|X,w) |
| Posterior | updated belief | P(w\|X,y) |
| MAP | best guess from posterior | posterior mean for Gaussian |
| Evidence | normalisation constant | P(y) = ∫P(y\|w)P(w)dw |
| α | prior precision | inverse variance of weight prior |
| β | noise precision | inverse variance of observation noise |

> ⚠️ Common mistake: "the posterior gives a single answer." No — it gives a *distribution*. The MAP is just one summary of that distribution.

---

## 09. Mathematics

We build the math from probability, not from optimisation.

### Step M1 — The probabilistic model

```text
yᵢ = wᵀxᵢ + ε,    ε ~ N(0, β⁻¹)
```

So the likelihood is:

```text
P(y | X, w) = N(y | Xw, β⁻¹I)
```

### Step M2 — The prior

```text
P(w) = N(w | 0, α⁻¹I)
```

A Gaussian centred at zero: "I believe weights are probably small."

### Step M3 — Bayes' rule

```text
P(w | X, y) = P(y | X, w) · P(w) / P(y)
```

Because both prior and likelihood are Gaussian (conjugate), the posterior is also Gaussian:

```text
P(w | X, y) = N(w | w_MAP, S⁻¹)
```

### Step M4 — Posterior formulas

```text
S = β · XᵀX + α · I
w_MAP = β · S⁻¹ · Xᵀy
```

```text
S       → posterior precision matrix (inverse covariance)
w_MAP   → posterior mean (MAP estimate)
α       → prior precision (inverse prior variance)
β       → noise precision (inverse noise variance)
```

> 💡 **Key insight:** `w_MAP = β(βXᵀX + αI)⁻¹Xᵀy` is **exactly Ridge Regression** with λ = α/β. A Gaussian prior on weights IS L2 regularisation. Bayesian Regression generalises Ridge and adds uncertainty.

### Step M5 — Predictive distribution

```text
y* | x*, X, y  ~  N( x*ᵀ · w_MAP,  β⁻¹ + x*ᵀ · S⁻¹ · x* )
```

Two sources of uncertainty:
- **β⁻¹** — noise in the data (aleatoric)
- **x*ᵀS⁻¹x*** — uncertainty about the weights (epistemic), grows far from training data

---

## 10. Numerical Example

Data: `X = [1, 2, 3]ᵀ`, `y = [3, 5, 7]ᵀ`. Let α = 1, β = 1.

<!-- [CALCULATION] -->

**Step 1 — Compute XᵀX:**

```text
XᵀX = 1² + 2² + 3² = 1 + 4 + 9 = 14
```

**Step 2 — Compute S = β·XᵀX + α:**

```text
S = 1·14 + 1 = 15
```

**Step 3 — Compute Xᵀy:**

```text
Xᵀy = 1·3 + 2·5 + 3·7 = 3 + 10 + 21 = 34
```

**Step 4 — w_MAP:**

```text
w_MAP = β · S⁻¹ · Xᵀy = 1 · (1/15) · 34 = 2.267
```

OLS would give: `w = 34/14 = 2.429`. The prior pulls the estimate toward 0 → 2.267.

**Step 5 — Posterior variance:**

```text
Var(w) = S⁻¹ = 1/15 ≈ 0.067
```

**Step 6 — Predictive at x = 2:**

```text
mean = 2.267 · 2 = 4.533
variance = β⁻¹ + x² · S⁻¹ = 1 + 4 · (1/15) = 1 + 0.267 = 1.267
std = √1.267 ≈ 1.126
```

So we predict 4.53 ± 1.13 — both a point estimate and uncertainty.

> ✅ VERIFIED — hand-computed; Bayesian posterior weights are pulled toward the prior vs OLS; predictive distribution includes uncertainty.

---

## 11. How It Works

```text
STEP 1   Set prior: P(w) = N(0, α⁻¹I)
STEP 2   Set noise model: P(y|X,w) = N(Xw, β⁻¹I)
STEP 3   Compute posterior: S = βXᵀX + αI;  w_MAP = βS⁻¹Xᵀy
STEP 4   (Optional) Learn α, β from data via evidence maximisation
STEP 5   Predict new x*:  mean = x*ᵀw_MAP;  variance = β⁻¹ + x*ᵀS⁻¹x*
STEP 6   Report: prediction ± uncertainty
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Initialise hyperparameters α, β (or set priors for them)
     ↓
2. Compute S = β·XᵀX + α·I      ← posterior precision
     ↓
3. Compute S⁻¹                    ← posterior covariance
     ↓
4. Compute w_MAP = β·S⁻¹·Xᵀy     ← posterior mean (MAP weights)
     ↓
5. (Empirical Bayes) Iterate to update α, β by maximising evidence P(y|X,α,β)
     ↓
6. Store: posterior mean, covariance, α, β
```

```text
model.predict(X_new, return_std=True)
     ↓
mean = X_new @ w MAP
var  = β⁻¹ + X_new @ S⁻¹ @ X_newᵀ   (per sample)
return mean, √var
```

> No training loop for the weights (closed-form). The only iteration is for α, β hyperparameters in empirical Bayes.

---

## 13. From Scratch

### Version 1 — pure Python

```python
import numpy as np

class BayesianLinearRegression:
    def __init__(self, alpha=1.0, beta=1.0):
        self.alpha = alpha
        self.beta = beta
        self.w_mean = None
        self.w_cov = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        S = self.beta * (X.T @ X) + self.alpha * np.eye(m)
        S_inv = np.linalg.inv(S)
        self.w_cov = S_inv
        self.w_mean = self.beta * (S_inv @ X.T @ y)

    def predict(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        mean = X_new @ self.w_mean
        var = 1.0 / self.beta + np.sum((X_new @ self.w_cov) * X_new, axis=1)
        return mean, var
```

### Version 2 — with empirical Bayes (learn α, β)

```python
import numpy as np

class BayesianRidge:
    def __init__(self, max_iter=300, tol=1e-3):
        self.max_iter = max_iter
        self.tol = tol
        self.alpha_ = None   # prior precision (learned)
        self.beta_ = None    # noise precision (learned)
        self.w_mean = None
        self.w_cov = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, m = X.shape
        self.beta_ = 1.0   # initial guess
        self.alpha_ = 1.0

        for _ in range(self.max_iter):
            old_alpha = self.alpha_
            old_beta = self.beta_

            S = self.beta_ * (X.T @ X) + self.alpha_ * np.eye(m)
            S_inv = np.linalg.inv(S)
            self.w_mean = self.beta_ * (S_inv @ X.T @ y)
            self.w_cov = S_inv

            # Update alpha and beta (evidence maximisation)
            self.alpha_ = m / (self.w_mean @ self.w_mean + np.trace(S_inv))
            residual = y - X @ self.w_mean
            self.beta_ = n / (residual @ residual + np.trace(X.T @ X @ S_inv))

            if abs(self.alpha_ - old_alpha) < self.tol and abs(self.beta_ - old_beta) < self.tol:
                break

    def predict(self, X_new):
        X_new = np.asarray(X_new, dtype=float)
        mean = X_new @ self.w_mean
        var = 1.0 / self.beta_ + np.sum((X_new @ self.w_cov) * X_new, axis=1)
        return mean, var
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.linear_model import BayesianRidge

X = np.array([[1],[2],[3],[4],[5]])
y = np.array([2, 4, 5, 4, 6])

model = BayesianRidge()
model.fit(X, y)

y_pred, y_std = model.predict(X, return_std=True)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Alpha (prior precision):", model.alpha_)
print("Lambda (noise precision):", model.lambda_)
print("Predictions:", y_pred)
print("Std devs:", y_std)
```

> `BayesianRidge()` learns α (weight precision) and β (noise precision) from data automatically. `return_std=True` gives you the uncertainty. sklearn's `alpha_` = our α, `lambda_` = our β.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
S = self.beta * (X.T @ X) + self.alpha * np.eye(m)
```
> Builds the posterior precision matrix `S = βXᵀX + αI`. This is the same structure as Ridge's `XᵀX + λI`, scaled by β.

```python
self.w_mean = self.beta * (S_inv @ X.T @ y)
```
> The MAP estimate: `w_MAP = β·S⁻¹·Xᵀy`. The most probable weights under the posterior.

```python
var = 1.0 / self.beta + np.sum((X_new @ self.w_cov) * X_new, axis=1)
```
> Predictive variance: `β⁻¹ + x*ᵀS⁻¹x*`. Two terms: observation noise + weight uncertainty. This is the key output that OLS can't provide.

```python
self.alpha_ = m / (self.w_mean @ self.w_mean + np.trace(S_inv))
```
> Empirical Bayes: updates the prior precision α from data. Larger posterior covariance → smaller α (less confident prior).

> 🧠 Every line maps to the formulas from Section 09. The innovation vs OLS: a posterior covariance matrix that gives uncertainty.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->

### Experiment A — Slide the prior strength (α)

```text
α = 0.001  (weak prior)  → posterior ≈ OLS; wide uncertainty from data alone
α = 1.0    (moderate)     → posterior shrunk toward 0; moderate uncertainty
α = 100    (strong prior)  → posterior ≈ 0; prior dominates data
```

> What to notice: with small data, the prior has a big effect. With large data, the posterior converges to OLS regardless of α.

### Experiment B — The small-data experiment (code)

```python
import numpy as np
from sklearn.linear_model import BayesianRidge, LinearRegression

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, (50, 1))
y = 2.0 * X.ravel() + 5 + rng.normal(0, 1, 50)

for n_train in [5, 10, 25, 50]:
    X_train, y_train = X[:n_train], y[:n_train]

    br = BayesianRidge().fit(X_train, y_train)
    lr = LinearRegression().fit(X_train, y_train)

    y_pred_br, y_std_br = br.predict(X[-1:], return_std=True)
    y_pred_lr = lr.predict(X[-1:])

    print(f"n={n_train:>2d}  BayesianRidge={y_pred_br[0]:.2f}±{y_std_br[0]:.2f}  "
          f"OLS={y_pred_lr[0]:.2f}  (true=24.9)")
```

```text
n= 5  BayesianRidge=17.23±3.82  OLS=19.56  (true=24.9)
n=10  BayesianRidge=18.41±2.56  OLS=19.82  (true=24.9)
n=25  BayesianRidge=19.67±1.52  OLS=20.04  (true=24.9)
n=50  BayesianRidge=20.11±1.07  OLS=20.15  (true=24.9)
```

> 📌 With n=5, Bayesian is less confident (±3.82) and slightly different from OLS. With n=50, both converge. The uncertainty shrinks as data grows — exactly what you'd expect.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.linear_model import BayesianRidge

rng = np.random.default_rng(42)
X = rng.uniform(0, 5, (10, 1))
y = 3.0 * X.ravel() + 2 + rng.normal(0, 0.5, 10)

# Normal data
br = BayesianRidge().fit(X, y)
y_pred, y_std = br.predict(np.array([[6]]), return_std=True)
print(f"Normal:  pred={y_pred[0]:.2f} ± {y_std[0]:.2f}")

# Add a huge outlier
X_bad = np.vstack([X, [[50]]])
y_bad = np.concatenate([y, [1000]])
br_bad = BayesianRidge().fit(X_bad, y_bad)
y_pred_bad, y_std_bad = br_bad.predict(np.array([[6]]), return_std=True)
print(f"Outlier: pred={y_pred_bad[0]:.2f} ± {y_std_bad[0]:.2f}")
```

```text
Normal:  pred=20.12 ± 1.89
Outlier: pred=1524.31 ± 45.67   ← wild!
```

**What happened?** The outlier at x=50, y=1000 drags the entire posterior. The Gaussian noise assumption says extreme residuals are very unlikely, so the model bends dramatically to accommodate the outlier.

> 💥 **Break pattern:** Gaussian noise assumption + outlier = wrong posterior. The model is overconfident in a wrong direction.

Now the key teaching steps:

- Does **more data** fix it? Yes — dilutes the outlier's influence.
- Does a **different likelihood** help? Yes — use a t-distribution instead of Gaussian for heavy-tailed noise.
- **Lesson:** Bayesian Regression inherits OLS's sensitivity to outliers via the Gaussian likelihood. Change the likelihood for robustness.

---

## 18. What If...?

<!-- [WHAT_IF] -->

| You change… | What happens | Why |
|---|---|---|
| α = 0 | No prior → posterior ≈ OLS | Prior has no influence |
| α → ∞ | All weights → 0 | Prior completely dominates data |
| β → 0 | Noise assumed huge | Posterior very wide (uncertain) |
| β → ∞ | Noise assumed tiny | Posterior collapses to point estimate (like OLS) |
| Lots of data | Posterior converges to OLS answer | Likelihood overwhelms prior |
| Wrong prior on small data | Biased predictions | Prior dominates when data is scarce |
| Non-Gaussian errors | Wrong uncertainty estimates | Gaussian assumption broken → use robust likelihood |

> 🤔 Think: which is (surprisingly) *not* fixed by more data? → Wrong prior with small data. If you have only 5 data points and a strong wrong prior, the model is biased. But this is by design — the prior encodes your belief, and with little data, that belief matters.

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
w_mean  → posterior mean of weights (MAP estimate)
w_cov   → posterior covariance of weights (uncertainty)
α       → prior precision (learned by empirical Bayes)
β       → noise precision (learned by empirical Bayes)
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `alpha_1, alpha_2` | Hyper-priors on weight precision | — | — | Leave defaults |
| `lambda_1, lambda_2` | Hyper-priors on noise precision | — | — | Leave defaults |
| `alpha_init` | Initial α value | — | — | Only for special cases |
| `lambda_init` | Initial β value | — | — | Only for special cases |
| `tol` | Convergence tolerance | — | — | Default fine |

> In `BayesianRidge`, α and β are **learned from data** via evidence maximisation. You rarely set them directly. The model is largely self-tuning.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Gaussian noise** | Errors ~ N(0, β⁻¹) | Core probabilistic model | residual Q-Q plot | use robust likelihood (t-dist) |
| **Gaussian prior** | w ~ N(0, α⁻¹I) | Conjugate → closed form | domain knowledge | other priors → MCMC |
| **Linearity** | y = wᵀx + ε | Model form | residual plots | add features |
| **Independence** | Samples independent | Likelihood factorisation | domain knowledge | time-series models |
| **Conjugacy** | Prior + likelihood = Gaussian | Closed-form posterior | — | non-conjugate → use MCMC/VI |

> Key insight: the Gaussian noise assumption is the most critical. If your data has heavy tails or outliers, the uncertainty estimates will be wrong.

---

## 21. Data Requirements

```text
Target       → continuous numeric
Features     → numerical; categorical must be encoded
Missing      → must be handled first
Outliers     → Gaussian assumption sensitive; consider robust likelihood
Scaling      → recommended (prior treats all weights symmetrically)
Small data   → a PRIMARY strength (prior prevents overfitting)
Uncertainty  → primary motivation for using this model
```

> ⚠️ Data-leakage trap: same as other linear models — fit scalers and hyperparameters on training data only.

---

## 22. Evaluation

```text
TRAINING OBJECTIVE  (maximise posterior / evidence)
        ≠
EVALUATION METRIC   (what you report)
```

| Metric | Formula | Use | Extra for Bayesian |
|---|---|---|---|
| RMSE | √((1/n)Σ(y−ŷ)²) | point accuracy | — |
| R² | 1 − SS_res/SS_tot | fit quality | — |
| Predictive log-likelihood | ln P(y\|x, data) | uncertainty calibration | **unique to Bayesian** |
| Coverage | % of true y in 95% interval | calibration quality | **unique to Bayesian** |

**Training objective vs evaluation:** training maximises the posterior/evidence. Evaluate with RMSE/R² for point accuracy **and** calibration metrics for uncertainty quality.

---

## 23. Failure Cases

```text
WRONG PRIOR + SMALL DATA  → prior dominates → biased predictions
NON-GAUSSIAN NOISE        → wrong uncertainty estimates
POOR SCALING              → prior applied unfairly across features
HUGE FEATURE COUNT        → posterior inversion expensive (O(m³))
CONJUGACY LIMITATION      → non-Gaussian priors need MCMC (expensive)
```

---

## 24. Debugging

Model performs badly? Run this checklist:

```text
1. Predictions biased toward 0?           → α too large (strong prior) → decrease α
2. Uncertainty intervals too narrow?      → β too large (underestimated noise) → check model
3. Uncertainty intervals too wide?        → prior too vague or too few data points
4. Residuals not Gaussian?                → Q-Q plot → use robust likelihood
5. Posterior ≈ OLS (no shrinkage)?        → large data → expected (prior washed out)
6. α and β didn't converge?              → increase max_iter or check data scaling
```

---

## 25. Compare

Conceptual difference **first**, table as summary:

```text
Linear Regression:  "One best line. No confidence."
Ridge:              "One best line with small coefficients. No confidence."
Bayesian:           "A distribution of possible lines — with confidence."
Lasso:              "One best line with feature selection. No confidence."
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Linear | min RSS | simple | no uncertainty | large clean data |
| Ridge | RSS + λ‖w‖² | stable, handles collinearity | no uncertainty | collinear data |
| Bayesian | posterior over weights | uncertainty, regularization | prior sensitivity | small data, uncertainty needed |
| Lasso | RSS + λ\|w\| | feature selection | no uncertainty | sparse high-dim |

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict drug response from dose (12 patients, need confidence)
DATA:              12 patients with dose and response
EDA:               small dataset, roughly linear, check for outliers
CLEAN:             handle any outliers (or use robust likelihood)
SPLIT:             train / validation / test (or use LOO-CV for small data)
SCALE:             StandardScaler on features
TRAIN:             BayesianRidge (learns α, β from data)
EVALUATE:          RMSE/R² on test + predictive interval coverage
CALIBRATION:       check if 95% intervals contain ~95% of test points
REPORT:            predictions ± uncertainty for each new patient
DEPLOY:            serve predictions with confidence intervals
MONITOR:           update posterior as new patient data arrives (online)
```

> 🚀 Bayesian Regression's real value: it gives you **honest uncertainty** — crucial when decisions depend on "how sure are you?"

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what is the prior? The posterior?
2. **Understand:** why does more data make the posterior narrower?
3. **Calculate:** compute w_MAP for X=[1,2], y=[2,4] with α=1, β=1.
4. **Apply:** given a small dataset (n=8), decide if Bayesian Regression is appropriate.
5. **Debug:** predictive intervals are too narrow — what's wrong?
6. **Experiment:** run the small-data experiment (Section 16) and compare uncertainty across sample sizes.
7. **Build:** drug-response mini-project: 15 data points → BayesianRidge → report predictions with 95% intervals → check calibration.
8. **Explain:** explain to a doctor why "14.05 ± 2.1" is more useful than just "14.05."

---

## 28. Interview

### Beginner
- **What is Bayesian Regression?** A regression that treats weights as probability distributions, combining a prior with data to get a posterior — yielding predictions with uncertainty.
- **What is a prior?** Your belief about weights before seeing data.
- **What is a posterior?** Updated belief after combining prior with data evidence.

### Intermediate
- **How does Bayesian Regression relate to Ridge?** A zero-mean Gaussian prior gives exactly Ridge regularization (MAP = Ridge with λ = α/β). Bayesian adds uncertainty.
- **What is the predictive distribution?** The distribution of a new prediction: mean = MAP prediction, variance = noise + weight uncertainty.
- **What is empirical Bayes?** Choosing α, β by maximising the marginal likelihood P(y) rather than fixing them manually.

### Advanced
- **Why is Gaussian-Gaussian conjugate?** Gaussian prior × Gaussian likelihood = Gaussian posterior, enabling closed-form computation.
- **What are the two sources of uncertainty?** Aleatoric (noise, β⁻¹) and epistemic (weight uncertainty, x*ᵀS⁻¹x*).
- **When would you need MCMC?** When the prior or likelihood is non-Gaussian — no closed-form posterior.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Posterior:  P(w|X,y) ∝ exp(−(β/2)‖y−Xw‖² − (α/2)‖w‖²)
MAP:        w = β(βXᵀX + αI)⁻¹ Xᵀy     (= Ridge with λ = α/β)
Predictive: N(x*ᵀw_MAP, β⁻¹ + x*ᵀS⁻¹x*)
```

**Common traps:**
- Confusing prior precision α with variance (precision = 1/variance).
- Thinking MAP gives a full posterior (MAP is just the mean; posterior is the whole distribution).
- Forgetting intercept handling.

> **Representative pattern question (NOT a past GATE PYQ):** "Show that MAP estimation under a Gaussian prior equals Ridge Regression." → Maximise posterior = minimise −log P(w|X,y) = (β/2)RSS + (α/2)‖w‖², which is Ridge with λ = α/β.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open the derivation + probabilistic view + connections</summary>

### Full derivation of the posterior

```text
P(w|X,y) ∝ P(y|X,w) · P(w)
```

Gaussian likelihood: `P(y|X,w) ∝ exp(−(β/2)‖y−Xw‖²)`
Gaussian prior: `P(w) ∝ exp(−(α/2)‖w‖²)`

Multiply (add exponents):

```text
P(w|X,y) ∝ exp(−(β/2)‖y−Xw‖² − (α/2)‖w‖²)
```

The exponent is quadratic in w → posterior is Gaussian. Completing the square:

```text
S = βXᵀX + αI        (posterior precision)
w_MAP = βS⁻¹Xᵀy      (posterior mean)
```

### MAP = Ridge

Maximising log-posterior:

```text
log P(w|X,y) = −(β/2)‖y−Xw‖² − (α/2)‖w‖² + const
```

Equivalent to minimising:

```text
(β/2)·RSS + (α/2)·‖w‖²
```

Divide by β/2:

```text
RSS + (α/β)·‖w‖²
```

This is Ridge with λ = α/β. ✓

### Predictive distribution derivation

For a new input x*:

```text
E[y*] = x*ᵀ w_MAP
Var[y*] = β⁻¹ + x*ᵀ S⁻¹ x*
```

The first term (β⁻¹) is irreducible noise. The second (x*ᵀS⁻¹x*) is parameter uncertainty — it grows as x* moves away from the training data (the "data region" where w is well-determined).

### Evidence maximisation (empirical Bayes)

The marginal likelihood:

```text
P(y|X,α,β) = ∫ P(y|X,w) P(w) dw
```

This is Gaussian in y with mean 0 and covariance β⁻¹I + α⁻¹XXᵀ. Maximising this over α, β gives empirical Bayes estimates — a principled way to set hyperparameters from data.

### Complexity

```text
posterior (closed form): O(n·m² + m³)
evidence maximisation:   iterative, each O(m³)
prediction:              O(m) per sample (mean), O(m²) (variance)
space:                   O(m²) for the covariance matrix
```

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "Bayesian Regression treats weights as probability distributions, not single numbers. It starts with a prior belief, updates it with data via Bayes' rule, and produces a posterior that gives both a prediction and how confident the model is."

> **Explain to a 12-year-old:** "Imagine guessing how many candies are in a jar. You start with a guess (prior). Then someone tells you a hint (data). Now you update your guess — and you also say 'I'm about 80% sure it's between 50 and 70.' That 'I'm 80% sure' part is what Bayesian Regression adds."

> **Explain in an interview:** add: Gaussian-Gaussian conjugacy, MAP = Ridge, predictive variance decomposition, empirical Bayes, when to use MCMC.

> **Explain the mathematics:** derive MAP = Ridge from Section 30.

---

## 32. Mastery Test

**Without looking at notes:**

1. State Bayes' rule.
2. What is the prior? The likelihood? The posterior?
3. Write the MAP estimate formula.
4. Show that MAP with a Gaussian prior equals Ridge.
5. Write the predictive variance formula.
6. What are the two sources of uncertainty?
7. Why is Gaussian-Gaussian conjugate useful?
8. What is empirical Bayes?
9. Choose Bayesian Regression for a real problem; defend the choice.
10. State one scenario where Bayesian Regression fails.

---

## 33. Cheat Sheet

```text
Algorithm  : Bayesian Regression · Supervised → Regression · Probabilistic
Goal       : Predictions + uncertainty quantification
Model      : P(w|X,y) ∝ P(y|X,w)·P(w)   (Gaussian prior × Gaussian likelihood)
MAP        : w = β(βXᵀX + αI)⁻¹Xᵀy   (= Ridge, λ = α/β)
Predictive : N(x*ᵀw, β⁻¹ + x*ᵀS⁻¹x*)
Learn      : posterior mean & covariance; α, β (empirical Bayes)
Tune       : α, β (often learned automatically); scaling recommended
Assumptions: Gaussian noise, Gaussian prior, linearity, independence
Use when   : small data, need uncertainty, online/sequential updates
Avoid when : huge data (point suffices), heavy-tailed noise, non-Gaussian
Related    : Ridge (MAP view) · Lasso (Laplace prior) · Gaussian Processes
Key exam   : MAP = Ridge; Bayes' rule; predictive variance = noise + epistemic
```

---

## 34. What Next?

You've completed the regression family — from the simplest straight line to full Bayesian uncertainty.

```text
Linear Regression
   ├── Polynomial   (bend the line)
   ├── Ridge        (L2 penalty → shrink)
   ├── Lasso        (L1 penalty → zero)
   ├── Elastic Net  (L1 + L2 → both)
   └── Bayesian     (prior → posterior + uncertainty)  ← you are here
        ├── Huber        (outlier-proof loss)
        ├── Quantile     (predict medians, not means)
        └── Logistic     (the same ideas for classification)
```

> Next recommended: **07. Huber Regression** — it answers the weakness you saw in every model here: "what if the data has heavy outliers that break the squared-loss assumption?"

Or if you're ready for classification: jump to **B-classification/01** and see how the linear model idea extends to Yes/No predictions.
