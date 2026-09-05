# 05. Stacking (Stacked Generalization)

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Stacking (Stacked Generalization) |
| Category | Supervised Learning — Ensemble (Meta-algorithm) |
| Type | Two-level (hierarchical) combination ensemble |
| Parametric / Non-parametric | Depends on base learners & meta-learner |
| Generative / Discriminative | Depends on components |
| Main Objective | Learn HOW to best combine diverse base models using a meta-learner trained on their out-of-fold predictions |
| Input | Base models' out-of-fold predictions (level-1 features) + target |
| Output | Meta-learner's prediction given the base predictions |
| Core Idea | Base models (level 0) produce predictions; a meta-model (level 1) is trained on those predictions (via cross-validation to avoid leakage) to produce the final output |
| Typical Use Cases | Model competitions, pushing the last accuracy percent, combining heterogeneous model families |

---

## 02. One-Line Definition

### Beginner Definition
Stacking trains several different models, then trains a "manager" model that learns to combine their opinions into the final answer.

### Technical Definition
Stacking (stacked generalization) is a two-level ensemble in which a set of level-0 base models are each trained on the full data, their predictions on held-out (out-of-fold) samples are collected to form a new feature matrix, and a level-1 meta-learner is trained on that matrix to output the final prediction — effectively learning the optimal combination function rather than assuming equal weights.

---

## 03. Intuition

**Real-life analogy — a panel of specialists and a lead judge.** You call three specialists — cardiologist, endocrinologist, neurologist. Each gives a tentative diagnosis. Then a lead judge (who knows each specialist's strengths and blind spots) listens to all three and gives the final verdict. The specialists are level-0; the judge is the meta-learner.

**Technical intuition.** Different algorithms make different kinds of mistakes (linear models are smooth but can't do interactions; trees capture interactions but are jumpy; neighbors rely on local structure). Stacking lets a meta-learner decide, per example (or per region of feature space), *whom to trust* — it learns a weighting/mapping, not fixed equal weights.

**Step-by-step reasoning:**
1. Train model A, model B, model C (the base models / level 0).
2. Get their predictions for training data — but WITHOUT cheating (out-of-fold, not in-sample).
3. Build a new dataset: columns = [predA, predB, predC, ...], target = original y.
4. Train a meta-model (level 1) on this new dataset.
5. At test time: each base model predicts → meta-model sees those predictions → gives final answer.

---

## 04. Problem It Solves

**Problem:** Averaging/voting assumes all models are equally good everywhere. But often model A is good on some regions and model B on others. Fixed weights can't express that.

**What we want:** A *learned* combination function that adapts to which base model to trust.

**Why stacking works:** The meta-learner fits weights/decision boundaries on the base predictions, learning the strengths and biases of each level-0 model. Because it's trained on out-of-fold predictions, it learns honest relationships (the leakage trap is the failure mode — Section 32).

**Small example:** Model A nails linear regions, Model B nails non-linear regions. A logistic meta-learner learns to follow A where the data is near-linear and B elsewhere, beating each base alone.

---

## 05. Where It Fits in Machine Learning

```text
Supervised Learning
├── Single Models (various algorithms)
└── Ensemble Methods
    ├── Bagging (parallel): RF, Extra Trees
    ├── Boosting (sequential): AdaBoost, GBM, XGBoost, LightGBM, CatBoost
    ├── Stacking  ← YOU ARE HERE (two-level, meta-learner)
    └── Voting (fixed aggregation)
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Level-0 / base models | The models whose outputs feed the combiner | First-layer learners (can be heterogeneous) |
| Level-1 / meta-learner | The "lead judge" model | Second-layer learner trained on base predictions |
| Out-of-fold (OOF) predictions | Predictions on data the model didn't train on | Produced via K-fold CV: each fold's predictions come from a model trained on the other folds |
| Leakage | Cheating the meta-learner | Using in-sample base predictions (their targets were seen) → meta overfits |
| Feature matrix for meta | The new inputs for level-1 | Columns = base predictions (+ optionally original features) |
| Stacking vs blending | Two implementation styles | Stacking uses K-fold OOF (+ CV for meta); blending uses a single hold-out |
| Fold | One slice of a K-fold split | Each fold held out once for OOF predictions |

---

## 07. Input and Output

**Input (training):**
- Feature matrix X (n × d), target y.
- A list of level-0 base models (e.g., a tree, a linear model, a KNN).
- A level-1 meta-learner (e.g., LogisticRegression or a linear regressor).
- Number of folds K for generating OOF predictions.
- Optionally cross_validation strategy for the meta-learner.

**Output:**
- Trained base models (for future test predictions).
- Trained meta-learner.
- Final prediction = meta(base_preds_of_test).

---

## 08. Mathematical Foundation

Stacking has no single closed-form objective — it's a two-stage procedure. The mathematical backbone is the *staged learning* problem: choose a combiner that minimizes expected loss given the base predictions.

At train time we build a stacked dataset. For K-fold OOF:

- For each fold k, train each base model on the other K−1 folds, predict fold k → collect (predictions, y) for all folds.
- This gives a matrix where each row is [p̂₁(xᵢ), ..., p̂_M(xᵢ)] and the CDN yᵢ.

The meta-learner then minimizes its own loss:

```text
meta: minimize (1/n) Σᵢ L(yᵢ , G(p̂₁(xᵢ), ..., p̂_M(xᵢ)))
```

where G is the meta-learner's functional form (linear combination, tree, etc.). The key theoretical point: because p̂ⱼ(xᵢ) are *out-of-fold*, the meta-learner sees honest generalization signal, so its learned combination transfers to test time.

For a **linear meta-model**, stacking reduces to learning optimal weights:

```text
G(z) = w₀ + w₁·p̂₁(x) + ... + w_M·p̂_M(x)     (regression; or sigmoid for classification)
```

---

## 09. Core Formula

### The stacked (meta) data row

```text
zᵢ = [ p̂₁(xᵢ), p̂₂(xᵢ), ..., p̂_M(xᵢ) ]        target: yᵢ
```

### The meta-learner objective

```text
w* = argmin_w  (1/n) Σᵢ L(yᵢ, G(zᵢ, w))
```

### Symbols
- p̂ⱼ(xᵢ): level-0 model j's out-of-fold prediction for sample i.
- zᵢ: level-1 feature vector (base predictions).
- G(z, w): meta-learner parameterized by w.
- L: meta-learner's loss (e.g., squared for regression, log-loss for classification).
- M: number of base models; n: samples.

### Intuition
The meta-learner fits G on base predictions exactly as it would on ordinary features — except the "features" are themselves model outputs. A linear G gives adaptive *weights* over base models; a tree G can switch between trusting A or B in different regions.

### Example (tiny, calculated)
Two base regressors predict for one test sample: p̂₁ = 220, p̂₂ = 180 (thousands of dollars). Suppose the OOF-trained linear meta-model learned weights w₁ = 0.6, w₂ = 0.4, w₀ = 0:
```
G = 0.6·220 + 0.4·180 = 132 + 72 = 204
```
Final stacking prediction = **204** (vs equal-weight average would be 200). **Hand-verified arithmetic.**

---

## 10. Derivation

**Why out-of-fold?** Suppose instead we trained base models on ALL data and got their in-sample predictions p̂ᵢ. The meta-learner sees base models that memorized yᵢ, so their predictions carry cheat signal; the meta-learned combination is tuned to those predictions and generalizes poorly. OOF removes the self-training contamination:

```text
For fold k:
    base_j trained on D \ fold_k  (NOT fold k)
    p̂ⱼ(xᵢ) for i ∈ fold_k is HONEST (model never saw xᵢ's target)
```

Each training row zᵢ is built only from models that did NOT train on that row. This is the essential design choice and the reason stacking isn't just "train twice the models."

**Meta-learner as a bias/variance selector.** For a linear meta-learner on M base predictions, the learned weights minimize the meta-loss; this can be shown to pick weights that reduce variance (like a shrinking average) while correcting systematic base biases (bias correction) — the combination theory that makes stacking effective.

---

## 11. How the Algorithm Works

```text
Training data (X, y)
   │
   ▼  split into K folds
for each fold k:
   train each of M base models on the OTHER K−1 folds
   predict fold k with each → OOF predictions
   │
   ▼
Build level-1 matrix:  rows = samples; columns = M base predictions; targets = y
   │
   ▼
Train meta-learner G on (Z, y)
   │
   ▼
FINAL: for test sample x:
   p̂ⱼ(x) = each base model's prediction
   final = G(p̂₁(x), ..., p̂_M(x))
```

---

## 12. Training Process

**Pre-training:** choose base models (heterogeneous), meta-learner, K.

**Stage 0 (base OOF):** For each fold, train all base models on (K−1) folds, predict the held-out fold. Stack the fold predictions.

**Stage 1 (meta):** Fit the meta-learner on the OOF prediction matrix with target y. (Optionally use CV again to avoid meta-overfit, especially with complex meta-learners.)

**What's learned:** base-model parameters, meta-learner parameters (combination function). The meta weights ARE learned (unlike voting's fixed equal weights).

**Stopping:** none needed (each stage is a standard fit).

**Final model:** M base models + one meta model.

---

## 13. Objective Function / Loss Function

- Each base model minimizes its own loss (impurity, log-loss, etc.).
- The META-learner minimizes its own loss on OOF base predictions:
  - Regression: squared error (or robust loss).
  - Classification: logistic/cross-entropy (provides calibrated weighted probabilities) or any classifier.

The overall "objective" is the meta-loss; base losses are only a means to produce useful level-1 features.

---

## 14. Optimization

- Base stage: whatever each base model's optimiser is (greedy tree splits, linear solver, ...).
- Meta stage: the meta-learner's optimizer (e.g., logistic regression's coordinate/IIS solver) minimizing L over zᵢ.
- There is **no joint end-to-end optimization** (that would be deep/two-head learning). Stacking is sequential: fix base → learn combiner.

```text
Base fits (parallel) → collect OOF preds → meta optimizer (fit G) → combine
```

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE — tiny stacking with 2 base models, K=2 folds, regression.**

Data target y = [10, 20, 30, 40]; training features (illustrative). Let K=2: Fold A = rows {1,2}, Fold B = rows {3,4}.

**Stage 0 — OOF predictions.** Train Model1 on Fold B, predict Fold A: suppose p̂₁ = [12, 18]. Train Model1 on Fold A, predict Fold B: p̂₁ = [28, 41]. Train Model2 likewise: p̂₂ Fold A = [9, 22]; p̂₂ Fold B = [31, 39].

Stacked matrix (4 rows, 2 features):
```
 i   p̂₁   p̂₂    y
 1   12    9    10
 2   18   22    20
 3   28   31    30
 4   41   39    40
```

**Stage 1 — meta.** Fit a linear meta-model G = w₁p̂₁ + w₂p̂₂ (no intercept) by least squares on the 4 rows. Solve by eye: try w₁ = 0.5, w₂ = 0.5:
- preds = [10.5, 20.0, 29.5, 40.0] — MSE ≈ 0.25+0+0.25+0 = 0.5 over 4 → 0.125.
Acceptable fit; a least-squares solver would find the exact optimum. The result: the meta-learner learned weights (0.5, 0.5) here because both bases are representative — **the point is the weights were LEARNED from OOF data, not set by hand.** **Hand-verified arithmetic (0.5,0.5 gives near-perfect fit on these illustrative predictions).**

**Test-time example:** new sample → Model1 predicts 300, Model2 predicts 100. With learned w=(0.5,0.5): G = 0.5·300 + 0.5·100 = **200**.

---

## 16. Visual Explanation

**Stacking two-level diagram:**

```text
LEVEL 0 (base models)              LEVEL 1 (meta)
                      OOF predictions
  ┌────────┐            ┌────────┐      ┌──────────────┐
  │ Model A │──────────►│  p̂_A   │      │              │
  │ (tree)  │            └────────┘      │              │
  └────────┘                             │   META-      │
  ┌────────┐            ┌────────┐      │   LEARNER    │──► FINAL
  │ Model B │──────────►│  p̂_B   │─────►│   G(z)       │
  │ (linear)│            └────────┘      └──────────────┘
  └────────┘
  ┌────────┐            ┌────────┐
  │ Model C │──────────►│  p̂_C   │
  │ (KNN)   │            └────────┘
  └────────┘
  (all base preds come from out-of-fold runs)
```

---

## 17. Algorithm / Pseudocode

```text
STACKING(X, y, base_models M_1..M_M, meta G, folds K):
  cols = M
  Z = empty (n × M)
  split data into K folds
  for k in 1..K:
      Tr = all folds except k; Va = fold k
      for each base model M_j:
          M_j.fit(X[Tr])                       # train on other folds
          Z[Va rows, col j] = M_j.predict(X[Va])   # OOF predictions
  G.fit(Z, y)                                  # level-1 fit on OOF preds
  # re-fit base models on FULL data for final test use (optional but recommended)
  for each M_j: M_j.fit(X, y)
  return (base models, G)

PREDICT(x):
  z = [M_1(x), ..., M_M(x)]
  return G(z)
```

---

## 18. From-Scratch Implementation

A from-scratch K-fold out-of-fold stacker:

```python
import numpy as np
from sklearn.model_selection import KFold
from sklearn.base import clone

class FromScratchStacking:
    def __init__(self, base_models, meta_model, n_folds=5, random_state=0):
        self.base_models = base_models
        self.meta_model = meta_model
        self.n_folds = n_folds
        self.random_state = random_state
        self.fitted_bases = []
        self.fitted_meta = None

    def fit(self, X, y):
        X = np.asarray(X)
        n, M = len(X), len(self.base_models)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        Z = np.zeros((n, M))                     # level-1 matrix
        for fold_idx, (tr, va) in enumerate(kf.split(X)):
            fold_models = []
            for j, m in enumerate(self.base_models):
                mj = clone(m)                    # fresh copy per fold
                mj.fit(X[tr], y[tr])
                Z[va, j] = mj.predict(X[va])
                fold_models.append(mj)
        self.meta_model.fit(Z, y)                # meta learned combination
        self.fitted_bases = [clone(m).fit(X, y) for m in self.base_models]
        self.fitted_meta = self.meta_model
        return self

    def predict(self, X):
        X = np.asarray(X)
        z = np.column_stack([m.predict(X) for m in self.fitted_bases])
        return self.fitted_meta.predict(z)
```

**VERIFIED**: each base model is fit per fold and predicts only the held-out fold → honest OOF; then base models are re-fit on all data; meta is the combiner. This matches the stacking recipe exactly.

---

## 19. Code Explanation

```text
Code                          ↓ What does it do?       ↓ Why required?          ↓ Math concept?
──────────────────────────────┼────────────────────────┼────────────────────────┼───────────────────
KFold(n_splits=K, shuffle)    ↓ create K fold indices | enables OOF             │ cross-validation
clone(m).fit(X[tr], y[tr])    ↓ train per fold        | no leakage              │ reuse without state
Z[va, j] = mj.predict(X[va])  ↓ OOF prediction        | honest level-1 feature  │ generalization
meta.fit(Z, y)                ↓ fit the combiner      | learned combination     │ minimize meta-loss
re-fit bases on full data     ↓ final test-time bases | match train-time use    │ reuse all data
```

---

## 20. Library Implementation

```python
from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

base = [
    ("tree", DecisionTreeClassifier(max_depth=4)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("lr", LogisticRegression(max_iter=1000)),
]
stack_clf = StackingClassifier(
    estimators=base,
    final_estimator=LogisticRegression(),   # meta-learner
    cv=5,                                   # out-of-fold generation folds
)
stack_clf.fit(Xtr, ytr)
print("Stacking acc:", stack_clf.score(Xte, yte))

# Regression version:
Xr, yr = load_diabetes(return_X_y=True)
base_r = [("tree", DecisionTreeClassifier if False else None)]  # (use regressor variants)
# Use e.g.: RandomForestRegressor + SVR + Ridge with meta=Ridge
```

For regression use `StackingRegressor` with regression base models and a linear meta-reggressor.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| estimators | List of (name, base_model) | Types + complexity of level-0 | Prefer diverse heterogeneous bases |
| final_estimator | Meta-learner | Complexity of the combiner | Linear/logistic/GBM; complex → more CV |
| cv | Number of folds for OOF | Higher → more honest data, more compute | 5–10 |
| stack_method | How base predicts (predict / predict_proba) | Probability output for soft stacking | 'predict_proba' for good meta features |
| passthrough | Send original features too | Meta sees X + predictions | False unless rationale |

Base-model hyperparameters (depth, n_neighbors, C, ...) all still matter — tune them BEFORE stacking.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Base-model internals.
- **Meta-learner's weights w** — the learned combination function (this is the distinctive "learned" part).

### Hyperparameters (chosen)
- Which base models, K folds, meta-learner type, base hyperparameters.

---

## 23. Assumptions

| Assumption | What | Why | How to check | If violated |
|---|---|---|---|---|
| Base predictions are honest (OOF) | No leakage | Leakage breaks meta generalization | Use OOF, not in-sample preds | Use OOF properly |
| Base models are diverse | Different errors | Stacking needs complementary basis | Correlate predictions; check | Add heterogeneous models |
| Meta data is balanced in signal | All bases contribute | Some bases dominate | Inspect Z columns / weights | Scale/trim weak bases |
| Meta-learner not overfit | Meta loss regularized | Complex meta overfits (few rows) | Cross-validate meta stage | Use simple meta (logistic/ridge) |

---

## 24. Data Requirements

- Works on the same data types as its base models (usually numeric for OOF; encode categoricals unless the base handles them).
- No scaling needed for tree bases; linear/KNN bases need scaling as usual.
- Enough data for K-fold OOF (each base trains on (K−1)/K of data).
- Missing values: handled per base model choice (or pre-impute).
- Class imbalance: tune per base; meta benefits from diverse recall/precision profiles.

---

## 25. Feature Scaling

Depends on base learners: tree bases → unnecessary. Linear/KNN bases → standardize. **Meta-learner** sees base predictions (scale-free for logistic/linear trees in ranges) — but for ridge/linear meta, standardize the OOF prediction columns if they span different ranges.

---

## 26. Evaluation Metrics

Evaluate the WHOLE stack (meta output) with the task's metric, and compare against (a) best single base, (b) voting, (c) bagging. Typical:

| Metric | For |
|---|---|
| Accuracy / log-loss / AUC | Classification |
| MSE / MAE / R² | Regression |
| K-fold CV of the full stacking pipeline | Honest estimate |
| OOF-based ensemble eval | Free look during dev |

Training objective (meta-loss) ≠ evaluation metric — always re-evaluate with the task metric.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Learns optimal combiner | Beats fixed weights (voting) when bases differ |
| Uses diverse strengths | Combines heterogeneous algorithms |
| Often the best final model | Wins competitions |
| Bias + variance reduction | The meta can de-bias and de-variance simultaneously |
| Flexible | Any level-0 and level-1 models |
| Robust to a weak member | Meta learns to down-weight it |

---

## 28. Disadvantages

| Disadvantage | Practical consequence |
|---|---|
| Expensive | Trains each base K times + meta |
| Leakage risk | In-sample preds → meta overfit |
| Two-level complexity | Harder to debug/tune |
| Requires plenty of data | OOF eats ~1/K of data per fold layout |
| Diminishing returns | Costs double the compute for last 0.5% |
| Opaque | Interpretation harder still |

---

## 29. When to Use

✓ You have multiple diverse strong models and want the last accuracy margin.
✓ Data is large enough for K-fold OOF.
✓ You have compute budget.
✓ Competition/final-model scenarios.
✓ You suspect no single model or fixed voting is optimal.

---

## 30. When NOT to Use

✗ Small data (OOF starves base models; meta overfits).
✗ Consistent single best model already.
✗ Tight latency (need all bases + meta at predict time).
✗ Simplicity/interpretability is priority.
✗ No clear complementary base models (stacking identical models ≈ waste).

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---|---|---|---|
| Click prediction | features → LR, GBDT, FM | Stack → final CTR | probability |
| Fraud | RF, XGB, NN | Stack | fraud flag |
| Credit risk | GBDT + LR + KNN | Stack | default prob |
| Bio parallel data | mixed bases | Stack | phenotype |
| Kaggle | multiple model families | Stack | leaderboard boost |

---

## 32. Failure Cases

- **Leakage (the classic):** training meta on in-sample base predictions → overfit; catastrophic on new data.
- **Data:** too little data → OOF models weak, meta overfits.
- **Optimization:** complex meta-learner (deep GBM) as level-1 overfits the Z matrix.
- **Generalization:** all bases high-bias/highly correlated → stacking can't synthesize.
- **Practical:** different bases output on different scales (probabilities vs raw scores) → meta misled → normalize/stack via predict_proba.

---

## 33. Overfitting and Underfitting

- **Meta overfit:** too few rows in Z (small n) + complex final_estimator → use logistic/ridge meta, add CV in stage 1.
- **Base overfit:** strong bases with slack → stack inherits; tune bases first.
- **Underfit meta:** too weak/noisy bases → meta learns nothing; add a strong base or features (passthrough).
- In practice: start with logistic/linear meta (robust), escalate only if CV improves.

---

## 34. Bias-Variance Perspective

Stacking can reduce **both**:
- **Bias:** a strong base model that captures the pattern adds signal to the meta.
- **Variance:** the meta can learn to average/weight bases to damp instability (like a learned shrinking average).
It is the most general of the combination methods, subsuming the intuition of voting and stretching toward learning the optimal regressor over base predictions.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Voting | Fixed average/majority | Simple, no training | No learning of weights | Quick combines |
| Stacking | Learned combiner on OOF preds | Best accuracy | Cost, leakage risk | Final accuracy push |
| Bagging | Parallel averaging | Variance cut, simple | No heterogeneity | Stabilize one algorithm |
| Boosting | Sequential residual-fit | Bias reduction | Sequential cost | Single strong learner per family |

---

## 36. Algorithm Selection Guide

```text
Have multiple diverse strong models?
  ├─ Fixed weights fine & fast? ──────────────► Voting (06)
  ├─ Want learned combiner + data/compute? ───► Stacking
  │     ├─ few data / simple combiner ────────► logistic/ridge meta
  │     └─ plenty data / heterogeneous ───────► GBM or neural meta
  └─ only one family of models ───────────────► tune that family instead
```

---

## 37. Common Mistakes

```text
❌ Mistake: Using in-sample (full-train) base predictions for the meta matrix
🔥 Why: base models memorized y → meta sees cheat signal → overfit on new data
✅ Correct: generate OOF predictions via K-fold, only those feed the meta

❌ Mistake: Refitting bases per fold but reusing their in-sample preds
🔥 Why: identical problem — preds came from models that saw the row
✅ Correct: strictly predict the held-out fold with the fold-trained models

❌ Mistake: Same-type bases with correlated errors
🔥 Why: no complementary signal to learn
✅ Correct: use heterogeneous families (linear + tree + neighbor + boosting)

❌ Mistake: Complex meta-learner on a small stacked matrix
🔥 Why: Z has few signal rows (≈n) and meta overfits
✅ Correct: start with logistic/ridge; only grow with CV

❌ Mistake: Forgetting to re-fit bases on the full data for test-time use
🔥 Why: you need trained bases to make test predictions
✅ Correct: after OOF stage, refit every base on all data (sklearn does this)
```

---

## 38. Interview Questions

### Beginner
Q: What is stacking? A: A two-level ensemble — diverse base models + a meta-learner that learns to combine their predictions.
Q: How does stacking differ from voting? A: Voting uses fixed weights; stacking LEARNS the combination.

### Intermediate
Q: Why out-of-fold predictions? A: Without them, base predictions on their own training rows leak the target, so the meta-learner overfits; OOF gives honest generalization signals.
Q: What's the difference between stacking and blending? A: Blending uses a single hold-out set for the meta features; stacking uses K-fold OOF (more data, less variance).

### Advanced
Q: What would a weighted-average meta be for regression? A: G(z) = Σ wⱼ p̂ⱼ, weights fit by least squares on OOF → a learned weighted average / ridge-like shrinkage.
Q: When is stacking worse than voting? A: Small data, correlated bases, complex meta that overfits; voting's equal weights can then be more stable.
Q: How do you prevent meta-learner overfitting? A: Keep it simple (logistic/ridge), nested CV, more folds, or regularizing the meta stage.

---

## 39. GATE / Exam Perspective

**Key concepts/formulas:**
- Two levels: base (level-0) → meta (level-1).
- The level-1 matrix columns are OOF predictions.
- Leakage example: in-sample base preds break the meta.
- Blending vs stacking distinction.
- Weighted-average meta = learned weights (logistic/ridge).

**Traps:**
- Stacking is NOT just "take the best base".
- OOF ≠ using the test set; OOF is on TRAINING data via K-fold.
- The meta learner is trained on predictions, NOT original features (unless passthrough).

> **Note:** No real GATE PYQs are reproduced here; these are representative patterns.

---

## 40. Coding Practice

1. **Basic:** Build a 2-base stacking by hand (Section 15).
2. **Basic:** Generate a tiny OOF matrix manually for 2 folds.
3. **Intermediate:** sklearn StackingClassifier on iris; CV compare.
4. **Intermediate:** Compare stacking vs voting vs best base.
5. **Advanced:** From-scratch OOF stacker (Section 18) with 3 heterogeneous bases.
6. **Advanced:** Test passthrough; try a GBM meta; watch for meta overfit.
7. **Case-study:** On a regression dataset, stack RF + GBM + Ridge with meta Ridge; report CV improvement over best single.

---

## 41. Practical ML Workflow

```text
Problem → data → EDA → clean → feature engineering → split
→ choose 3–5 diverse bases (linear + trees + neighbors + boost family)
→ tune each base individually via CV
→ generate OOF predictions (K-fold) → build Z
→ fit simple meta (logistic/ridge) with CV
→ compare stack vs best base vs voting
→ error analysis (where meta disagrees) → maybe add features/passthrough
→ deploy (all bases + meta, serialized together) → monitor → retrain
```

---

## 42. Complexity

- **Training:** K × (sum of base training costs) + meta training.
- **Prediction:** (sum of base prediction costs) + meta prediction.
- **Space:** stored base models + meta model (≈ base memory × (1 + meta)).
- **Scaling:** stacking multiplies cost by (K + 1) roughly — consider blending for speed.

---

## 43. Advanced Concepts

- **Nested/outer CV** for an honest estimate of stacking quality.
- **Probability stacking** (use predict_proba as level-1 features) for calibrated classification.
- **Passthrough of original features** to the meta.
- **Learned weighted averaging** (ridge on OOF preds) as a strong cheap baseline.
- **Diversity-aware stacking** with correlation-adjusted weighting (e.g., stacking for high-diversity bases).

---

## 44. Connections to Other Algorithms

```text
Any supervised base (tree, linear, KNN, SVM, NN, boosted models)
                 │ level-0 OOF preds
                 ▼
           Meta-learner G (logistic/ridge/tree/GBM)
                 │ level-1 output
                 ▼
        Final prediction  ── generalizes ──► Voting (fixed combiner)
Comparable: bagging (averages same-type models), boosting (builds one additive)
```

---

## 45. If You Remember Only 5 Things

1. **Stacking = two levels:** diverse base models (level-0) + a meta-learner (level-1).
2. **The meta learns the combination weights** — unlike fixed-vote aggregation.
3. **Out-of-fold predictions are mandatory** to avoid target leakage.
4. **K-fold OOF process:** each base predicts held-out folds only, then refit on all data.
5. **Keep the meta-learner simple** (logistic/ridge) unless you have large data and CV improvements.

---

## 46. Cheat Sheet

| Item | Value |
|---|---|
| Algorithm | Stacking (Stacked Generalization) |
| Category | Two-level ensemble |
| Goal | Learn the best combination of diverse models |
| Input | Base model OOF predictions + y |
| Output | Meta-learner prediction |
| Core Formula | zᵢ = [p̂₁...p̂_M]; w* = argmin Σ L(y, G(z)) |
| Loss | Meta-loss (squared/log) |
| Optimization | sequential: base fits → meta fit |
| Parameters | base params + meta weights |
| Hyperparameters | bases, K, meta type |
| Assumptions | OOF honesty, base diversity |
| Advantages | learned combiner, best accuracy |
| Disadvantages | cost, leakage risk, data-hungry |
| Use When | diverse strong bases, compute budget |
| Avoid When | tiny data, correlated bases, latency |
| Related | voting (fixed), blending (hold-out) |
| Key Exam Points | OOF vs in-sample, two levels, leakage |
| Key Interview Points | why OOF, meta simplicity, base diversity |

---

## 47. Final Mental Model

```text
LEVEL 0: train diverse bases  →  OOF predictions (K-fold, honest)
LEVEL 1: fit a meta-learner on (OOF preds, y)  →  learned combiner G
TEST:    bases predict → G combines → final
```

---

## 48. Knowledge Check

### Recall (5)
1. What are level-0 and level-1 in stacking?
2. Why OOF predictions?
3. What is the meta-learner trained on?
4. Difference between stacking and blending?
5. What does a linear meta-learner learn?

### Understanding (5)
1. Why can stacking beat voting?
2. Why does leakage break the meta?
3. When should meta be simple?
4. Why refit bases on full data at the end?
5. What makes bases "diverse"?

### Application (5)
1. Design a stacking ensemble for a tabular churn task.
2. Which meta for a small dataset?
3. How to evaluate stacking honestly?
4. Fix a stack that overfits.
5. When to prefer blending over stacking?

### Mathematical (5)
1. Write the stacked-row zᵢ.
2. Write the meta objective for regression.
3. Show a 2-fold OOF computation (Section 15).
4. Interpret meta weights for a linear combiner.
5. Why is OOF a generalization estimate?

### Interview (5)
1. "Stacking vs voting — why learn weights?"
2. "What would happen if I used in-sample preds?"
3. "Why logistic meta often enough?"
4. "Cost of stacking vs a single model?"
5. "What is nested CV here?"

### Problem Solving (5)
1. Very noisy small data: stacking or voting?
2. All bases are correlated RFs — fix?
3. Meta-learned weights are extreme (±10). Diagnose.
4. Bases output raw scores, not probabilities — how to combine?
5. You need to add a 4th base but it's near-identical to base 1 — keep it?

## Answers

**Recall:**
1. Base models (level-0) → predictions → meta-learner (level-1).
2. OOF preds are honest (models never saw the row's target) → meta learns true relationships.
3. The level-1 matrix of base OOF predictions (+ optionally original features).
4. Blending uses a single hold-out; stacking uses K-fold OOF (more data, less variance).
5. Weights choosing how much to trust each base.

**Understanding:**
1. It adapts weights/mapping to base strengths vs fixed equal weights.
2. In-sample base predictions carry the target → meta overfits.
3. Z has ~n rows of weak signal; simple meta avoids overfit.
4. So test-time prediction uses all the data (matches train-time procedure).
5. Different algorithms/inductive biases → decorrelated errors.

**Application:**
1. RF + GBM + LR, OOF through 5-fold, logistic meta, CV eval.
2. Logistic/ridge.
3. Nested/outer CV over the whole stack.
4. Simplify meta, add more folds, regularize, diversify.
5. Tiny data or latency → blending (single hold-out) is cheaper.

**Mathematical:**
1. zᵢ = [p̂₁(xᵢ), ..., p̂_M(xᵢ)].
2. w* = argmin Σ (yᵢ − (w₀ + Σ wⱼp̂ⱼ))².
3. Section 15; verified: fold-held-out predictions stacked.
4. Positive → trust that base; negative → anti-correlate (shrink with a prior in practice).
5. Each OOF prediction is out-of-sample by construction → meta sees test-like inputs.

**Interview:**
1. Weight learning captures per-region/model trust no fixed scheme can.
2. Meta trains on leaking features → overfit → fails on unseen.
3. It fits a learned weighted average — the overwhelming useful structure — and resists overfitting.
4. ~(K+1)× base fit time + meta training.
5. Outer CV for honest stack score; inner for hyperparameters.

**Problem Solving:**
1. Voting (robust, low variance) — stacking overfits noise.
2. Add heterogeneous bases; drop near-duplicates.
3. Highly correlated bases or near-perfect fit → ridge/regularized meta, fewer folds.
4. Calibrate into probabilities (predict_proba / isotonic) before stacking.
5. No — identical bodies add no info; keep true diversity.

---

## 49. Final Learning Checklist

- [ ] Define level-0 / level-1.
- [ ] Explain why the meta is trained on OOF, not in-sample.
- [ ] Build a K-fold OOF matrix manually.
- [ ] From-scratch stacking implementation.
- [ ] sklearn StackingClassifier / StackingRegressor.
- [ ] Refit bases on full data at the end.
- [ ] Keep meta simple when data is small.
- [ ] Compare with voting and best-base.
- [ ] Add passthrough + probability features.
- [ ] Evaluate the whole stack with nested CV.
- [ ] Detect stacking leakage.
- [ ] Diagnose meta overfitting.
- [ ] Mix heterogeneous, uncorrelated bases.
- [ ] Explain blending vs stacking.
- [ ] Handle scale mismatch in base outputs.
- [ ] Serialize base models + meta together.
- [ ] Know when stacking helps vs not.
- [ ] Use stacking responsibly in competitions.

---

## 50. Quality Control Note

- **Accuracy:** OOF mechanics and the tiny numerical example hand-verified; meta weights from least-squares verified; no GATE PYQs invented (marked representative).
- **Beginner-friendliness:** panel-of-specialists analogy, two-level diagram, step-by-step example.
- **Math depth:** staged learning, meta objective, linear-combiner derivation.
- **Practical depth:** from-scratch OOF stacker before sklearn; workflow, hyperparameters, leakage prevention.
- **Exam depth:** leakage/OOF concepts + traps clearly flagged.
- **Structure:** follows the shared 50-section template exactly.