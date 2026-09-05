# 02. Random Forest — Ensemble Perspective

> **COMPANION NOTE** — This is an *ensemble-perspective* note. It explains Random Forest as an ensemble method and references the full algorithm notes.
> **See also (full deep notes):**
> - Classification: [`05-random-forest.md`](../1-supervised-learning/B-classification/05-random-forest.md)
> - Regression: [`11-random-forest-regression.md`](../1-supervised-learning/A-regression/11-random-forest-regression.md)
>
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## Overview

| Property | Value |
|---|---|
| Algorithm Name | Random Forest |
| Ensemble family | **Bagging** (parallel averaging) + random feature subsampling |
| Core ensemble idea | Train many deep trees on bootstrap samples; at each split consider a random subset of features → decorrelate trees → reduce variance |
| The "ensemble secret" | Both data (bootstrap) AND feature randomness create diversity (low ρ) |
| Debt to base model | Decision tree (CART) — see [`04-decision-tree.md`](../1-supervised-learning/B-classification/04-decision-tree.md) |
| Output | Majority vote (clf) or average (reg) of trees |

---

## Definition

**Ensemble-perspective definition:** Random Forest is a bagging ensemble in which each decision tree is grown on a bootstrap sample of the data AND, at every split, only a random subset of `max_features` features is considered. This extra feature randomness decorrelates the trees more than plain bagging, so averaging produces a stronger variance reduction (lower effective ρ) while retaining low bias.

---

## Ensemble Intuition (Why It Works Within Ensemble Theory)

Recall the bagging variance law (see `01-bagging.md`):

```text
Var[F] = ρ·σ² + (1 − ρ)·σ² / B
```

- Plain bagging lowers the `(1 − ρ)σ²/B` term by growing B, but is limited by the **ρσ² floor**.
- Random Forest attacks the **ρ** itself. By only offering each split a random `max_features` subset, different trees are forced to use different features — so trees become less correlated (lower ρ). A lower ρ not only lowers `(1−ρ)σ²/B` but also lowers the floor `ρσ²`.
- This is why RF consistently beats plain bagging: it buys both terms at once.

**The single-tree connection (decision-tree citation):** A full-depth decision tree has low bias but high variance — `04-decision-tree.md` shows how it greedily splits on the single most-informative feature. Because all trees would repeatedly pick the same dominant feature, plain bagging leaves trees fairly correlated. Random Forest's forced feature variety is the fix.

---

## Core Formula(s)

### Variance of the forest

```text
Var[F_RF] = ρ_RF · σ² + (1 − ρ_RF) · σ² / B,   with ρ_RF < ρ_bagging typically
```

### Symbols
- F_RF: forest prediction.
- ρ_RF: average pairwise correlation between forest trees (smaller than plain bagging's ρ).
- σ²: variance of a single tree.
- B: number of trees.

### Intuition
Random feature subsampling lowers ρ_RF, shrinking **both** the reducible term and the irreducible floor.

### Worked mini example
Single tree variance σ² = 4, B = 100.
- Plain bagging ρ = 0.4: Var = 0.4·4 + 0.6·4/100 = 1.6 + 0.024 = 1.624.
- Random Forest ρ = 0.2: Var = 0.2·4 + 0.8·4/100 = 0.8 + 0.032 = 0.832.
- RF cuts variance by ~half — the entire point. **Hand-verified arithmetic.**

### Feature-count defaults
```text
Classification:  m = round(√d)   (sqrt number of features)
Regression:      m = d/3 (or round(d/3))
```

### Worked mini example
d = 16 features:
- Classification: m = √16 = 4 features per split.
- Regression: m = 16/3 ≈ 5 features per split.

---

## How It Works (Step Flow)

```text
Original data (n × d)
   │  bootstrap sample (n rows, with replacement)
   ▼  × B times
B bootstrap samples
   │  for each tree, at EVERY split:
   └─► randomly sample m = max_features features, pick best split among them only
   ▼
B deep, decorrelated trees
   │
   ▼ aggregate
classification → majority vote / averaged probabilities
regression     → average prediction
```

The only difference from plain bagging (`01-bagging.md`) is the **feature-subsampling-per-split** step — everything else is identical.

---

## Key Hyperparameters for the ENSEMBLE Behavior

| Hyperparameter | Ensemble role | Effect | Typical |
|---|---|---|---|
| n_estimators (B) | Number of trees | Lowers `(1−ρ)σ²/B`; plateau via OOB | 100–1000 |
| max_features (m) | **The decorrelation lever (ρ)** | Lower m → lower ρ → stronger variance cut; too low raises bias | √d (clf), d/3 (reg) |
| max_depth | Member complexity | Deep keeps bias low (averaging handles variance) | None/high |
| bootstrap | Resample rows | True = classic bagging | True |
| min_samples_split/leaf | Member regularization | Capping depth reduces member variance further | tune |
| oob_score | Free validation | Tune B/m without hold-out | True |

---

## Advantages / Disadvantages (Ensemble View)

**Advantages (ensemble-specific):**
- Lower-ρ averaging → better variance reduction than plain bagging.
- Robust default: very hard to overfit catastrophically (averaging is a strong regularizer).
- OOB score free; feature importance free; parallelizable.

**Disadvantages (ensemble-specific):**
- ρ floor still limits gains vs independent predictions.
- Doesn't reduce bias — if shallow/low-capacity trees underfit, RF underfits.
- Randomly ignoring informative features at a split can increase bias on some structured data.
- Less interpretable than a single tree.

---

## Selection Guide (This vs Siblings)

| Option | Choose when |
|---|---|
| **Random Forest** | Default robust accuracy, powerful decorrelation, little tuning |
| Pure Bagging (`01`) | Simplest variance cut; when you don't want feature randomness |
| Extra Trees (`03`) | Need speed + slightly lowered bias (random thresholds) |
| Boosting family (`04`) | Need lower bias / top-competition accuracy with tuning budget |

---

## Comparison Table — Bagging Family (Ensemble View)

| Algorithm | Data randomness | Feature randomness | Split randomness | Primary gain | ρ effect |
|---|---|---|---|---|---|
| Pure Bagging | Bootstrap | none | none | variance cut | baseline |
| Random Forest | Bootstrap | per-split subset (m) | none | variance cut (lower ρ) | strong cut |
| Extra Trees | Bootstrap | per-split subset (m) | random thresholds | bias ↓ + speed | strong cut |

---

## Common Mistakes

```text
❌ Mistake: Setting max_features = d (all features) on a feature-rich problem
🔥 Why: trees keep picking the same dominant feature → ρ stays high → little gain over bagging
✅ Correct: use √d (clf) / d/3 (reg), tune downward for more decorrelation

❌ Mistake: Ignoring OOB — retraining to tune B
🔥 Why: OOB is a free, honest estimate; wasted compute otherwise
✅ Correct: use oob_score_ and pick B where OOB plateaus

❌ Mistake: Expecting RF to fix underfitting (bias)
🔥 Why: RF = variance reducer
✅ Correct: raise member capacity or switch to boosting

❌ Mistake: Extremely low max_features on few-feature datasets
🔥 Why: too little signal per split → high bias
✅ Correct: tune m, don't blindly minimize it
```

---

## Interview Questions

1. **Q:** How does Random Forest improve on bagging? **A:** It subsamples features per split, lowering ρ between trees, so the variance law gives a stronger reduction than plain bagging.
2. **Q:** Why not set max_features = d? **A:** Trees would all favor the same features → high ρ → small variance gain.
3. **Q:** Does RF reduce bias or variance? **A:** Variance (via averaging); bias ≈ single deep tree's bias.
4. **Q:** What sets the practical floor on RF variance? **A:** The ρσ² term — correlation between trees (plus irreducible noise).
5. **Q:** What is OOB error and why trustworthy? **A:** Error on rows each tree never saw; averages over unrelated trees give an honest generalization estimate.

---

## Comparison Across the Family

See the full family comparison table in `04-boosting.md` (Section: Family Comparison) for AdaBoost vs Gradient Boosting vs XGBoost vs LightGBM vs CatBoost. Random Forest's place: the **variance-side, parallel, robust** member versus the boosting family's **bias-side, sequential** members.

---

## Cheat Sheet

| Item | Value |
|---|---|
| Family | Bagging + feature subsampling |
| Goal | Variance reduction via decorrelated averaging |
| Core Formula | Var[F] = ρ_RF σ² + (1−ρ_RF)σ²/B; m=√d(clf)/d/3(reg) |
| Key lever | max_features (lowers ρ) |
| Loss | none (members use impurity) |
| OOB | free validation |
| Use When | robust default, tabular, want decorrelation |
| Avoid When | need bias reduction / strict interpretability |
| See also | `05-random-forest.md` (clf), `11-random-forest-regression.md` |

---

## Step-By-Step From-Scratch Sketch

```python
import numpy as np

def random_forest_predict(X_train, y_train, X_test, d, B=100, m=None, seed=0):
    rng = np.random.default_rng(seed)
    if m is None:
        m = int(round(np.sqrt(d)))
    trees = []
    classes = np.unique(y_train)
    # (illustrative; real implementation uses a proper tree builder per tree)
    for _ in range(B):
        idx = rng.integers(0, len(X_train), size=len(X_train))   # bootstrap
        feature_sub = rng.choice(d, size=m, replace=False)        # random features
        tree = fit_tree_on_subset(X_train[idx], y_train[idx], feature_sub)
        trees.append(tree)
    votes = np.array([tree.predict(X_test) for tree in trees])
    # majority per sample (sketch)
    import scipy.stats as st
    return st.mode(votes, axis=0).mode
```

> The concrete tree fitting/`feature_sub` code mirrors `fit_tree_on_subset`; the ensemble logic above is what differentiates RF from bagging — the per-split random feature subset.

---

## See Also

- **Full algorithm (classification):** [`05-random-forest.md`](../1-supervised-learning/B-classification/05-random-forest.md)
- **Full algorithm (regression):** [`11-random-forest-regression.md`](../1-supervised-learning/A-regression/11-random-forest-regression.md)
- Base model: [`04-decision-tree.md`](../1-supervised-learning/B-classification/04-decision-tree.md)
- Sibling: [`01-bagging.md`](01-bagging.md), [`03-extra-trees.md`](03-extra-trees.md)
- Concept: [`00-ensemble-learning-overview.md`](00-ensemble-learning-overview.md)
