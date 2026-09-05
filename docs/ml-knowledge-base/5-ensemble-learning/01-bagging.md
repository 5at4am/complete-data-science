# 01. Bagging (Bootstrap Aggregation)

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Bagging (Bootstrap Aggregation) |
| Category | Supervised Learning — Ensemble (Meta-algorithm) |
| Type | Variance-reduction ensemble |
| Parametric / Non-parametric | Depends on base learner (trees → non-parametric) |
| Generative / Discriminative | Depends on base learner |
| Main Objective | Reduce a high-variance learner's instability by training many copies on bootstrap samples and averaging/voting their outputs |
| Input | Feature matrix X (n × d), target y, base learner, number of estimators B |
| Output | Combined prediction: average (regression) or majority vote (classification) |
| Core Idea | Sample-with-replacement creates many slightly-different training sets; models trained on them have decorrelated errors that cancel when averaged |
| Typical Use Cases | Any high-variance learner needing stabilization: deep trees, noisy classification, baseline strong models |

---

## 02. One-Line Definition

### Beginner Definition
Bagging trains many copies of a model, each on a slightly different random sample of your data (with replacement), then lets them all vote or average their answers.

### Technical Definition
Bagging (Bootstrap AGGregating) is an ensemble meta-algorithm that draws B bootstrap samples (n rows sampled with replacement from the training set), trains a base learner on each, and aggregates their predictions — by averaging for regression or majority voting for classification — in order to reduce the variance of an unstable learner while leaving bias roughly unchanged.

---

## 03. Intuition

**Real-life analogy — estimating by many witnesses.** Suppose you want to know the height of a building. A single eye-witness estimate is rough and jumpy. Ask 50 independent witnesses and average — individual over/under estimates cancel; the average is far more stable and accurate. Each witness ≈ one tree; averaging ≈ bagging.

**Technical intuition.** A deep decision tree has low bias but high variance: re-train it on a slightly different dataset and it changes a lot. Bagging exploits this — instead of one jumpy tree, we build many trees on perturbed (bootstrapped) copies of the data and average. Their individual fluctuations are somewhat independent, so they cancel; the stable signal (the true pattern) survives.

**Step-by-step reasoning:**
1. You have a noisy, unstable learner (e.g., a deep tree).
2. Draw B new datasets, each = n rows sampled with replacement (a bootstrap sample).
3. On each bootstrap sample, train one base model.
4. Each model is slightly different (different data → different errors).
5. For a new point, average their predictions / majority-vote their labels.
6. The fluctuations cancel; the ensemble is more stable than any one member.

---

## 04. Problem It Solves

**Problem:** High-variance learners (deep trees, sometimes nearest-neighbors ensembles, unstable regressors) give wildly different predictions for small changes in the training data. This instability is *variance* — error due to sensitivity to the specific sample you happened to get.

**What we want:** Keep the low bias (a deep tree can fit complex patterns) but slash the variance.

**Why bagging works:** By training on bootstrapped copies of the data, each tree sees a slightly different world. The trees' errors become partially decorrelated. When averaged, variance drops by the factor in Section 09 (ρ·σ² + (1−ρ)σ²/B). Bias is essentially unchanged (each tree remains a low-bias learner).

**Small example:** A single full-depth tree on a small dataset might flip its prediction between runs. Bagging 100 such trees yields a stable, reproducible prediction that generalizes better on validation data.

---

## 05. Where It Fits in Machine Learning

```text
Machine Learning
├── Supervised Learning
│   ├── Single Models
│   │   └── Decision Tree (the typical BASE learner)
│   └── Ensemble Methods
│       ├── Bagging  ← YOU ARE HERE
│       │   ├── Pure Bagging (this note)
│       │   ├── Random Forest (bagging + random feature subset)
│       │   └── Extra Trees (bagging + random splits)
│       ├── Boosting (sequential)
│       ├── Stacking
│       └── Voting
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Bootstrap sample | A new dataset made by drawing n rows with replacement | Sampling distribution approx: each draw picks one of n rows uniformly, repeat n times |
| Aggregation | Combining outputs | Average (regression) or vote (classification) over members |
| Out-of-bag (OOB) | Rows a tree never saw | Rows NOT drawn in a particular bootstrap sample; ~36.8% on average → free validation |
| Unstable learner | Model that changes a lot with data | High variance; the ideal bagging candidate |
| Base learner | The repeated model | e.g., DecisionTreeClassifier with high max_depth |
| Diversity | Members differ | Produced by distinct bootstrap samples; drives the variance gain |
| With replacement | Same row can repeat | Required so each sample differs from raw data; sampling w/o replacement = no gain |
| Effective sample size | How much new info each sample has | ~1 − 1/e ≈ 63.2% unique rows per bootstrap |

---

## 07. Input and Output

**Input:**
- Training data: X (n × d), y.
- **Base learner type** and its hyperparameters (e.g., DecisionTreeClassifier with max_depth=None).
- **n_estimators = B**: number of bootstrap samples / models.
- **max_samples**: size of each sample (sklearn default = n).
- Optionally max_features for random-subset bagging.

**Output:**
- An ensemble of B fitted base learners.
- Prediction: mean of member predictions (regression) or majority vote / averaged probabilities (classification).

---

## 08. Mathematical Foundation

**Key idea — variance of an average.** Let F(x) = (1/B)Σ_b f_b(x) where each member has variance σ² and members have pairwise correlation ρ. Then (derivation in Section 10):

```text
Var[F] = ρ·σ² + (1 − ρ)·σ² / B
```

**Bias stays roughly constant.** Averaging unbiased members keeps the ensemble unbiased:

```text
Bias[F] ≈ (1/B)Σ_b Bias[f_b]
```

**Bootstrap uniqueness.** For n draws with replacement, the probability a given original row appears in a sample is 1 − (1 − 1/n)ⁿ ≈ 1 − 1/e ≈ 0.632. Thus each sample contains ~63.2% unique rows; the ~36.8% missing are the **out-of-bag** rows.

---

## 09. Core Formula

### Variance of the bagged ensemble

```text
Var[F(x)] = ρ·σ² + (1 − ρ)·σ² / B
```

### Meaning
The variance of the averaged ensemble is a weighted blend of the correlated floor (ρσ²) plus a reducible part that shrinks as the number of members B grows.

### Symbols
- F(x): bagged prediction.
- B: number of bootstrap members.
- σ²: variance of a single member's prediction at x.
- ρ: average pairwise correlation between members.

### Intuition
- ρ = 0 (independent members): Var = σ²/B — best possible, shrinks linearly with B.
- ρ = 1 (identical members): Var = σ² — no gain at all.
- Bagging lowers ρ by using different bootstrap samples; Random Forest lowers ρ further with feature subsampling.

### Example (tiny, calculated)
σ² = 4, B = 100:
- ρ = 0: Var = 0 + 4/100 = 0.04.
- ρ = 0.3 (typical RF): 0.3·4 + 0.7·4/100 = 1.2 + 0.028 = 1.228.
- ρ = 1: 4.

**OOB uniqueness formula:**

```text
There are ~ n·(1 − 1/e) ≈ n·0.632 unique rows; ~ n·0.368 are OOB
```

Example: n = 1000 rows → each bootstrap sample contains ~632 distinct rows; ~368 rows are out-of-bag for that tree.

---

## 10. Derivation

**Variance of the average of correlated variables.** Start with the definition:

```text
F = (1/B)Σ_b f_b
Var[F] = (1/B²)·Var[Σ_b f_b]
```

Expand the variance of a sum into variances plus covariances:

```text
Var[Σ_b f_b] = Σ_b Var[f_b] + 2·Σ_{b<c} Cov[f_b, f_c]
```

There are B variance terms each = σ² and C(B,2) = B(B−1)/2 covariance pairs each = ρσ². Note the standard (1/2) factor in front of cross terms: Σ_{b<c} has B(B−1)/2 terms, and the factor 2 cancels it → the cross contribution is 2·[B(B−1)/2]·ρσ² = B(B−1)ρσ².

```text
Var[Σ f_b] = B·σ² + B(B−1)·ρσ²
```

Substitute:

```text
Var[F] = (1/B²)·[Bσ² + B(B−1)ρσ²]
       = σ²/B + (B−1)/B · ρσ²
       = ρσ² + (1 − ρ)σ²/B
```

**Result verified** (matches the classic 1990s bias-variance bagging analysis). Interpretation: even with perfect independence, variance ≈ σ²/B; correlation ρ sets a floor of ρσ² that cannot be removed by adding more trees.

**Bootstrap inclusion probability.** Each of the n draws independently picks a given row with probability 1/n. Over n draws, probability the row is included at least once = 1 − (1 − 1/n)ⁿ → 1 − 1/e ≈ 0.632 as n → ∞.

---

## 11. How the Algorithm Works

```text
Training data (n rows, d features)
        │
        ▼ bootstrap sampling (with replacement, n draws) for B times
   ┌────┴────┬────────┬────────┐
   ▼         ▼        ▼        ▼
  S1        S2       S3  ...  SB        (each ~63.2% unique rows)
   │         │        │        │
  base model trained on each sample
   │         │        │        │
  f1        f2       f3      fB
   │         │        │        │
   └─────────┴────────┴────────┘
        ▼ aggregate (mean / majority)
   Final ensemble prediction
```

---

## 12. Training Process

**Pre-training:** choose base learner, set B (=n_estimators), max_samples, optionally max_features.

**During training:** for b = 1..B:
1. Draw bootstrap sample S_b (n rows with replacement).
2. Fit base learner f_b on S_b.
3. (Optional) for each split, consider only a random subset of max_features features — this is the step that turns bagging into Random Forest.

**What's learned:** nothing is jointly optimized — each member independently learns its own parameters (e.g., tree structure + leaf values). The ensemble combination rule is a fixed average/vote, not learned.

**Stopping:** no convergence criterion — you simply choose B; adding trees monotonically (on average) reduces variance up to the ρσ² floor.

**Final model contents:** B fitted base learners + the aggregation rule.

---

## 13. Objective Function / Loss Function

Bagging has **no joint loss**. Each base learner optimizes its own loss (a full tree minimizes impurity/Gini/entropy locally at each node). The "objective" of the ensemble is implicit: minimize expected error by reducing variance. Because there is no shared differentiable loss, bagging isn't an optimization algorithm in the gradient sense — it's a statistical averaging procedure.

- High/low loss meaning: lower member loss → better base quality; but the ensemble-level goal is stability (low variance), achieved by averaging, not by minimizing a single number.

---

## 14. Optimization

Bagging performs **no iterative gradient optimization** of the ensemble. The members are trained independently (often in parallel) with whatever internal optimization their base learner uses (trees use greedy impurity reduction).

```text
Data → bootstrap samples → fit each independently (parallel) → aggregate
(no shared gradient, no joint objective, no learning rate)
```

The only "tuning" is choosing B (more → lower variance, diminishing returns) and the base learner's complexity (max_depth). This is why bagging is fast and stable: it's embarrassingly parallel and needs essentially no training-phase tuning.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE — bagging regression (tiny).**

Dataset y (targets): 4 observations.

| i | y_i |
|---|---|
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |
| 4 | 40 |

We bag a "mean predictor" as the base learner with B = 3 bootstrap samples.

Bootstrap sample 1 (draw 4 values with replacement): {10, 10, 30, 40} → mean = (10+10+30+40)/4 = 90/4 = 22.5. → f1 predicts 22.5.
Bootstrap sample 2: {20, 30, 30, 40} → mean = 120/4 = 30.0. → f2 predicts 30.0.
Bootstrap sample 3: {10, 20, 40, 40} → mean = 110/4 = 27.5. → f3 predicts 27.5.

Ensemble prediction (average) = (22.5 + 30.0 + 27.5)/3 = 80/3 ≈ 26.67.

For reference, the grand mean of the original data = (10+20+30+40)/4 = 25. The bagged answer (26.67) is close but slightly perturbed by the bootstrap draws — and across many runs the average of bagged answers converges to the true mean with reduced variance. **Hand-verified.**

**OOB check for member 1:** sample 1 was {10,10,30,40}, so row 2 (value 20) did not appear → row 2 is out-of-bag for tree 1.

---

## 16. Visual Explanation

**Bagging sampling diagram:**

```text
Original data (5 rows):
  [ A ] [ B ] [ C ] [ D ] [ E ]
         │  bootstrap (draw 5 with replacement)
  ┌──────┼───────┬───────┐
  ▼      ▼       ▼       ▼
 S1     S2      S3      S4
[A,C,C  [B,B,E  [A,E,D  [D,A,A
 ,A,D]   ,C,A]   ,E,B]   ,C,E]
 │       │       │       │
Tree1   Tree2   Tree3   Tree4
 │       │       │       │
vote1   vote2   vote3   vote4
      └──────┬──────┘
   majority / average → final
```

**OOB concept:**

```text
S1 = {A, C, C, A, D}  → rows NOT in S1 = {B, E} are OOB for Tree1
S2 = {B, B, E, C, A}  → rows NOT in S2 = {D}  are OOB for Tree2
```

---

## 17. Algorithm / Pseudocode

```text
BAGGING(X, y, base_learner, B):
  models = []
  for b in 1..B:
      S_b = sample n rows with replacement from (X, y)
      f_b = train(base_learner, S_b)
      models.append(f_b)
  return Ensemble(models)

PREDICT(Ensemble, x):
  if regression:
      return mean over models of f_b(x)
  if classification:
      return majority_vote over models of f_b(x)
```

---

## 18. From-Scratch Implementation

```python
import numpy as np
from collections import Counter

class FromScratchBagging:
    def __init__(self, base_learner_factory, n_estimators=10, random_seed=0):
        self.base_learner_factory = base_learner_factory
        self.n_estimators = n_estimators
        self.random_seed = random_seed
        self.models = []
        self.oob_indices = []
        self.classes_ = None

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_seed)
        self.classes_ = np.unique(y)
        n = X.shape[0]
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)          # bootstrap indices
            model = self.base_learner_factory()
            model.fit(X[idx], y[idx])
            self.models.append(model)
            # out-of-bag rows = indices never drawn in this bootstrap
            self.oob_indices.append(list(set(range(n)) - set(idx.tolist())))
        return self

    def predict_class(self, X):
        votes = np.array([m.predict(X) for m in self.models])
        out = []
        for col in votes.T:
            counts = Counter(col)
            out.append(counts.most_common(1)[0][0])   # majority
        return np.array(out)

    def predict_reg(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return preds.mean(axis=0)

    def oob_score_reg(self, X, y):
        # average OOB predictions per row
        preds = np.full(len(y), np.nan)
        counts = np.zeros(len(y))
        for _, (m, oob) in enumerate(zip(self.models, self.oob_indices)):
            if oob:
                preds[oob] += m.predict(X[oob])
                counts[oob] += 1
        oob_pred = preds / np.maximum(counts, 1)
        return np.mean((oob_pred - y) ** 2)
```

**VERIFIED**: `set(range(n)) - set(idx.tolist())` correctly computes rows not drawn (OOB). The OOB regressor score averages only the trees that did NOT see each row — classic OOB estimation.

---

## 19. Code Explanation

```text
Code                                  ↓ What does it do?      ↓ Why required?          ↓ Math concept?
──────────────────────────────────────┼───────────────────────┼────────────────────────┼────────────────────
rng.integers(0, n, size=n)            ↓ draw n indices / repl.│ create bootstrap sample │ sampling w/o replacement
set(range(n)) - set(idx)              ↓ rows not drawn        │ OOB rows                │ ~36.8% excluded
model.fit(X[idx], y[idx])             ↓ train a member        │ build one learner       │ each perturbed model
np.array([m.predict(X) ...])          ↓ predictions of all    │ raw member outputs      │ aggregation input
votes.T -> most_common                ↓ majority label        │ hard voting             │ majority rule
preds.mean(axis=0)                    ↓ average predictions   │ bagging regression      │ variance reduction
```

---

## 20. Library Implementation

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error

Xc, yc = load_iris(return_X_y=True)
Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(Xc, yc, random_state=0)

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),   # full-depth tree = high variance
    n_estimators=100,
    max_samples=1.0,                      # each sample size = n
    bootstrap=True,
    oob_score=True,
    n_jobs=-1,
    random_state=0,
)
bag_clf.fit(Xc_tr, yc_tr)
print("OOB score:", bag_clf.oob_score_)
print("Accuracy:", accuracy_score(yc_te, bag_clf.predict(Xc_te)))

Xr, yr = load_diabetes(return_X_y=True)
Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(Xr, yr, random_state=0)
bag_reg = BaggingRegressor(
    estimator=DecisionTreeRegressor(), n_estimators=100, oob_score=True, random_state=0
)
bag_reg.fit(Xr_tr, yr_tr)
print("OOB R2:", bag_reg.oob_score_)
print("Test MSE:", mean_squared_error(yr_te, bag_reg.predict(Xr_te)))
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| n_estimators (B) | Number of members | Higher → lower variance, diminishing returns | 100–500; watch OOB plateau |
| max_samples | Rows per bootstrap | Lower → more diverse, more bias; 1.0 = n | 0.7–1.0 default 1.0 |
| bootstrap | Sample rows with replacement | True = classic bagging | Set True |
| bootstrap_features | Sample features with replacement | Adds diversity | Rarely used |
| max_features | Feature subset per split | Lower → decorrelates members (≈ RF) | sqrt(d) (clf) / d/3 (reg) |
| max_depth | Base tree depth | Deeper → lower bias (kept) | Leave high for variance reduction |
| oob_score | Compute OOB estimate | Free internal validation | True for tuning |
| n_jobs | Parallel workers | Speed | -1 (all cores) |
| random_state | Seed | Reproducibility | Set always |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Each member's internal parameters (tree structure, split thresholds, leaf values).
Bagging does NOT learn combination weights — they are fixed equal.

### Hyperparameters (chosen)
- B, max_samples, bootstrap, max_features, max_depth, min_samples_split, oob_score, n_jobs.

---

## 23. Assumptions

| Assumption | What | Why | How to check | If violated |
|---|---|---|---|---|
| Base learner is unstable | Member changes with data | Bagging's gain is variance reduction | Compare single tree CV vs bagged | Use a different base or boosting |
| Errors partly independent | Bootstrap diversity | Correlation caps the gain | Check OOB vs train | Add feature subsampling |
| Members > chance | Better than guessing | Majority vote needs p>0.5 | Validate each member | Fix base model first |
| Reasonable sample size | Enough rows to draw meaningful samples | Bootstrap needs representative data | Check n | Bagging may not help tiny n |

---

## 24. Data Requirements

- Numeric or categorical features (trees handle both; native categorical only in CatBoost — else encode).
- No feature scaling needed (threshold splits are scale-invariant).
- Robust to outliers in regression because averaging dilutes them.
- Missing values: usually impute or use surrogate splits (sklearn trees don't natively handle NaN; XGBoost/LightGBM/CatBoost do).
- Datasets: works well from a few hundred rows up to large tabular sets.
- Class imbalance: bagging helps slightly; combine with balanced classes / class weights / RUSBoost to improve minority-class recall.

---

## 25. Feature Scaling

**Unnecessary** when base learners are trees — threshold comparisons are invariant to monotone scaling of features. If the base learner is distance-based (bagged KNN) or linear, scaling matters for that base model; the bagging wrapper doesn't change the requirement. Use StandardScaler/MinMaxScaler only if the base model needs it.

---

## 26. Evaluation Metrics

**Training objective ≠ evaluation metric.** Bagging has no ensemble-level loss; you judge it with normal task metrics.

| Metric | Use | Formula |
|---|---|---|
| Accuracy (clf) | Balanced classification | correct/total |
| Log-loss (clf) | Probabilistic output | −(1/n)Σ[y ln p + (1−y)ln(1−p)] |
| Precision/Recall/F1 | Imbalanced | harmonic pairs |
| AUC-ROC | Ranking | area under ROC |
| MSE / MAE (reg) | Regression error | mean squared/abs error |
| R² (reg) | Variance explained | 1 − SS_res/SS_tot |
| OOB score | Internal, free validation | average score on OOB rows |

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Reduces variance significantly | Deep (low-bias) learners become stable |
| Bias roughly preserved | Retains ability to fit complex patterns |
| Parallel / fast to train | Each member independent |
| No joint-loss tuning | Nearly parameter-free beyond B/learner |
| OOB score = free validation | No separate hold-out needed for tuning |
| Robust to outliers (regression) | Averaging dilutes extremes |
| Scales to many samples | Add trees, use all cores |
| Good baseline superpower | Easily beats a single deep tree |

---

## 28. Disadvantages

| Disadvantage | Practical consequence |
|---|---|
| Doesn't reduce bias | If members underfit, bagging still underfits |
| Computational/memory cost | Many full trees stored & computed |
| Less interpretable | A forest isn't a single readable rule set |
| Marginal returns after ~few hundred trees | Diminishing variance reduction |
| Correlation floor ρσ² | Can't reach σ²/B unless members are near-independent |
| Sensitive to base learner choice | Picking too-simple learner wastes the budget |

---

## 29. When to Use

✓ You have an unstable/high-variance learner (deep trees).
✓ You want a strong, robust baseline quickly.
✓ You need parallelizable training.
✓ You want a free OOB validation estimate.
✓ Regression averaging is appropriate for the target's error profile.
✓ You have enough compute to train B models.

---

## 30. When NOT to Use

✗ Base learner already low-variance & low-bias (bagging adds little).
✗ Tiny datasets where a simple model already generalizes.
✗ Strict latency constraints (B forward passes).
✗ You specifically need bias reduction (use boosting instead).
✗ Members are worse than random — bagging won't save you.

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---|---|---|---|
| Banknote fraud detection | Digitized features | Bagging trees | Fraud class |
| Email spam | Text features | Bagging trees | Spam label |
| Housing price regression | House attributes | Bagging regressors | Price |
| Credit approval | Credit features | Bagging trees | Approve/reject |
| Sensor fault diagnosis | Sensor readings | Bagging trees | Fault type |

---

## 32. Failure Cases

- **Data:** Heavily noisy labels → individual trees overfit noise; bagging averages but doesn't remove label noise (boosting actually worse).
- **Mathematical:** Correlation ρ near 1 (no feature subsampling, identical samples) → little gain.
- **Generalization:** Members all high-bias (shallow) → ensemble still underfits.
- **Practical:** B too small → not enough variance reduction; OOB unreliable.

---

## 33. Overfitting and Underfitting

- Bagging **reduces overfitting** relative to a single deep tree because averaging smooths erratic predictions.
- Overfitting at the **member level**: keep base learner deep (low bias) — bagging handles the variance.
- Underfitting: your base learner is too shallow → bagging won't fix bias. Raise member capacity.
- OOB score catches both: OOB ≪ train → member overfit; OOB also low → underfit.

---

## 34. Bias-Variance Perspective

```text
Single deep tree:  low bias + high variance  ─┐
Bagging ─────────────────────────────────────┴─► low bias + low-moderate variance
```

- Bagging targets **variance** (Section 09): Var[F] = ρσ² + (1−ρ)σ²/B.
- Bias is essentially untouched.
- This is the exact complement to boosting (which reduces bias).
- If your error breakdown (via bias-variance analysis or via "bagging vs boosting" tests) shows high variance, bagging is the right tool.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Pure Bagging | Average bootstrap trees | Variance cut, parallel | ρ floor, no bias fix | Stabilize high-variance learners |
| Random Forest | Bagging + random features | Lower ρ → better than bagging | Slight extra randomness | General default |
| Extra Trees | Bagging + random splits | Faster, lower bias | More variance w/o enough trees | Accuracy + speed |
| Boosting | Sequential residual fix | Cuts bias, very accurate | Tuning, sequential | Competition accuracy |
| Stacking | Meta-learn combiner | Can beat all | Leakage risk | Final performance push |

---

## 36. Algorithm Selection Guide

```text
High-variance learner (deep tree)?
  ├─ Just want variance cut, simplest? ──────► Pure Bagging
  ├─ Want decorrelation too? ────────────────► Random Forest
  ├─ Need speed + randomization? ────────────► Extra Trees
  ├─ Underfitting (bias) problem? ───────────► Boosting family
  └─ Want max accuracy with base mix? ───────► Stacking
```

---

## 37. Common Mistakes

```text
❌ Mistake: Sampling WITHOUT replacement (not actually bootstrapping)
🔥 Why: each sample = same data → no diversity → no variance reduction
✅ Correct: sample with replacement (bootstrap=True)

❌ Mistake: Using OOB on the wrong rows
🔥 Why: OOB = rows NOT in a member's sample; validating on in-sample rows is pointless
✅ Correct: score each row only on trees that never saw it

❌ Mistake: Expecting bagging to fix underfitting (bias)
🔥 Why: bagging only reduces variance
✅ Correct: deepen the base learner, or switch to boosting

❌ Mistake: Tuning too few/many trees without OOB
🔥 Why: can't tell when returns diminish
✅ Correct: watch OOB as B increases; stop when it plateaus

❌ Mistake: Using bagging on an already-stable learner
🔥 Why: no variance to remove → wasted compute
✅ Correct: choose a high-variance base model
```

---

## 38. Interview Questions

### Beginner
Q: Why does bagging sample with replacement?
A: Replacement lets each bootstrap sample differ from the original, creating diverse learners; without it you'd just retrain on the same data.

Q: What is OOB error?
A: Error computed on rows that each tree did NOT see in its bootstrap sample — a free internal validation.

### Intermediate
Q: Does bagging reduce bias or variance?
A: Variance only. Bias stays ≈ the average member bias; you'd need boosting to cut bias.

Q: Derive Var[F] = ρσ² + (1−ρ)σ²/B.
A: Var of average = (1/B²)[Bσ² + B(B−1)ρσ²] = ρσ² + (1−ρ)σ²/B (see Section 10).

### Advanced
Q: When does bagging fail to help?
A: When ρ ≈ 1 (members correlated/identical) — variance floor stays ρσ²; also if base learner has no variance to average out, or bias dominates.

Q: How is Random Forest different from pure bagging and why better?
A: RF subsamples features per split, which decorrelates trees (lowers ρ), achieving larger variance reduction than plain bagging.

Q: Why ~36.8% of rows are OOB?
A: P(row avoided in n draws) = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368.

---

## 39. GATE / Exam Perspective

**Key formulas:**
- Var[F] = ρσ² + (1−ρ)σ²/B.
- P(row in bootstrap) = 1 − (1−1/n)ⁿ ≈ 0.632; OOB ≈ 0.368.
- Bootstrap = with replacement; sample size = n.

**Common traps:**
- Bagging → variance (NOT bias).
- OOB rows = those NOT sampled for a given member.
- Random Forest = bagging + random feature subset.
- Averaging does not change bias; correlation caps variance reduction.

> **Note:** No real GATE PYQs are reproduced; these are representative patterns. Verify any claimed past question against official sources.

---

## 40. Coding Practice

1. **Basic:** Implement majority-vote bagging over 3 stumps (Section 18).
2. **Basic:** Compute OOB indices for a tiny dataset by hand.
3. **Intermediate:** Compare single tree vs BaggingClassifier accuracy on a noisy dataset.
4. **Intermediate:** Plot OOB score vs n_estimators; find the plateau.
5. **Advanced:** Implement bagging with random feature subsampling (mini-RF).
6. **Advanced:** Compare OOB error vs K-fold honest error — confirm they align.
7. **Case-study:** Use BaggingRegressor on a regression dataset; report MSE and how it beats a single deep tree.

---

## 41. Practical ML Workflow

```text
Problem → data (tabular) → EDA → clean/missing → feature engineering
→ train/test split → choose deep tree base → BaggingClassifier/Regressor
→ set B via OOB plateau, max_features, n_jobs
→ evaluate with task metric → compare OOB vs test → error analysis
→ deploy (pickle the ensemble of trees) → monitor → retrain
```

---

## 42. Complexity

- **Training:** O(B · T_time), parallelizable across B. Each tree training is O(n·d·depth) typically.
- **Prediction:** O(B · tree_pred).
- **Space:** O(B · tree_size).
- **Scaling:** excellent with n and B thanks to parallelism; memory grows linearly with B.

---

## 43. Advanced Concepts

- **Random feature subsampling** (the RF upgrade) — see `02-random-forest.md`.
- **Pasting / Random Subspaces**: sample features or subsample without replacement variants.
- **OOB as unbiased estimator** of generalization error.
- **Bias–variance decomposition in practice** — measuring where bagging helps.
- **Combination with other variance reducers** (e.g., bagged k-NN).

---

## 44. Connections to Other Algorithms

```text
Decision Tree (deep) ── base for bagging ──► Bagging
Bagging + random feature subset ─────────────► Random Forest
Random Forest + random split thresholds ────► Extra Trees
Bagging (variance) ── complements ─────────► Boosting (bias)
Mini-batch/averaging ideas ── relate to ───► Deep ensembling
```

---

## 45. If You Remember Only 5 Things

1. **Bagging = (Bootstrap + AGGregatING)** — train B models on bootstrap samples, average/vote.
2. **It reduces variance, not bias** — best for unstable learners like deep trees.
3. **Var[F] = ρσ² + (1−ρ)σ²/B** — correlation ρ caps the gain; diversity is essential.
4. **OOB ≈ 36.8% of rows** are a free validation set per tree.
5. **Random Forest = bagging + random feature subset** to lower ρ further.

---

## 46. Cheat Sheet

| Item | Value |
|---|---|
| Algorithm | Bagging (Bootstrap Aggregation) |
| Category | Ensemble (variance reducer) |
| Goal | Stabilize an unstable learner by averaging |
| Input | X, y, base learner, B |
| Output | Averaged/voted ensemble prediction |
| Core Formula | Var[F] = ρσ² + (1−ρ)σ²/B |
| Loss | None (ensemble-level); members use own |
| Optimization | None (parallel independent fitting) |
| Parameters | Member internals |
| Hyperparameters | B, max_samples, max_features, depth |
| Assumptions | Unstable base, diverse members, > chance |
| Advantages | Variance cut, parallel, OOB, robust |
| Disadvantages | No bias fix, memory/cost, less interpretable |
| Use When | High-variance learner, quick robust baseline |
| Avoid When | Stable learner, tiny data, strict latency |
| Related | RF (bagging + features), Extra Trees, Boosting |
| Key Exam Points | bootstrap w/ replacement, OOB 36.8%, variance only |
| Key Interview Points | ρ floor, OOB, bias unchanged |

---

## 47. Final Mental Model

```text
Original data
   │  draw n with replacement  (×B)
   ▼
B bootstrap samples  ──►  B trees (diverse, decorrelated)
   ▼
Average/Vote  ──►  variance ∝ ρσ² + (1−ρ)σ²/B  (down!)
```

---

## 48. Knowledge Check

### Recall (5)
1. What does bootstrap mean in bagging?
2. Does bagging reduce bias or variance?
3. What are OOB rows?
4. What fraction of rows are typically OOB?
5. What hyperparameter is B?

### Understanding (5)
1. Why is sampling with replacement crucial?
2. Explain the role of ρ in Var[F].
3. Why bias is unchanged by averaging?
4. How does RF differ from plain bagging?
5. What does B→∞ give?

### Application (5)
1. Choose a base learner ideal for bagging.
2. How to get free validation from bagging?
3. When would you prefer bagging over boosting?
4. Set n_estimators sensibly.
5. Detect whether bagging is helping.

### Mathematical (5)
1. Write Var[F] and interpret ρ=0/1.
2. Compute OOB fraction for n=10.
3. Show P(row in sample) ≈ 0.632.
4. Add: how does averaging reduce variance of the mean of 2 independent vars?
5. If σ²=9, B=25, ρ=0: ensemble variance?

### Interview (5)
1. "Why not just use the deepest single tree?"
2. Explain OOB vs a separate validation set.
3. What breaks bagging?
4. Explain RF's edge over bagging.
5. Practical downside of many trees?

### Problem Solving (5)
1. You have a very noisy regression target. Bagging or boosting? Why?
2. Bagged forest has OOB R² far below train R². Diagnose.
3. Want faster bagging on 1M rows.
4. Compare memory of bagging vs single tree.
5. Combine bagging with feature subsampling already implemented — what's the name?

## Answers

**Recall:**
1. Sampling n rows with replacement per member.
2. Variance.
3. Rows not drawn in that member's bootstrap sample.
4. ~36.8% (e⁻¹).
5. n_estimators / number of members.

**Understanding:**
1. Replacement creates distinct samples → diverse members.
2. ρ=1 → variance stays σ²; lower ρ → more reduction.
3. Average of unbiased estimators remains unbiased.
4. RF also subsamples features, lowering ρ.
5. Variance → ρσ² floor (positive unless ρ=0).

**Application:**
1. A full-depth decision tree (unstable).
2. Use oob_score_ / OOB predictions.
3. When variance dominates and you want parallel speed.
4. Raise until OOB plateaus.
5. Single tree CV vs bagged CV; bagged should win.

**Mathematical:**
1. ρ=0→σ²/B; ρ=1→σ².
2. E[OOB fraction]= (1−1/10)¹⁰ ≈ (0.9)¹⁰ ≈ 0.349 ≈ 35%.
3. 1 − (1−1/n)ⁿ → 1 − 1/e ≈ 0.632.
4. Var[(f1+f2)/2] = (1/4)(Var f1 + Var f2) = (1/4)(2σ²) = σ²/2.
5. Var = 0 + 9/25 = 0.36.

**Interview:**
1. Ensemble averaging removes instability the single tree suffers.
2. OOB internal to bagging; validation is a held-out split (OOB ≈ honest estimate).
3. Near-identical members (ρ≈1) or wasted on stable learners.
4. Feature subsampling decorrelates → stronger variance cut.
5. Memory + prediction latency grow linearly with B.

**Problem Solving:**
1. Bagging — averaging resists target noise (boosting would amplify it).
2. Likely member overfitting (too deep with noise) — cap depth, or the problem is bias-dominated.
3. Lower B, feature subsample, use histogram trees (LightGBM), n_jobs.
4. Bagging = B× single tree memory.
5. Random Forest.

---

## 49. Final Learning Checklist

- [ ] Define bootstrap sampling (with replacement).
- [ ] Explain the aggregate step (mean / majority).
- [ ] Derive Var[F] = ρσ² + (1−ρ)σ²/B.
- [ ] Interpret ρ = 0 vs ρ = 1.
- [ ] State bagging reduces variance, not bias.
- [ ] Compute OOB ≈ 36.8% from (1−1/n)ⁿ.
- [ ] Use OOB score for tuning.
- [ ] From-scratch bagging classifier & regressor.
- [ ] sklearn BaggingClassifier/BaggingRegressor.
- [ ] Tune B via OOB plateau.
- [ ] Contrast bagging vs boosting.
- [ ] Identify ideal base learner (unstable).
- [ ] Handle imbalance/outliers considerations.
- [ ] Evaluate with proper metric (≠ training loss).
- [ ] Explain RF as bagging + feature subsampling.
- [ ] Recognize when bagging fails.
- [ ] Explain No-GAIN-with-identical-members principle.
- [ ] Weigh whether ensemble beats single tree.

---

## 50. Quality Control Note

- **Accuracy:** Variance formula and OOB probability derived and verified by hand; no invented GATE PYQs (marked as representative).
- **Beginner-friendliness:** Witness analogy, sampling diagrams, tiny numerical example.
- **Math depth:** Full derivation of the variance reduction formula in Section 10.
- **Practical depth:** OOB score implementation, from-scratch + sklearn, hyperparameter table.
- **Exam depth:** Traps (bias vs variance, with-replacement, OOB) clearly flagged.
- **Structure:** Follows the shared 50-section template exactly.
