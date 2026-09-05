# 04e. CatBoost — Ensemble Perspective

> **COMPANION BRIDGE NOTE** — Explains CatBoost's *ensemble* theory and bridges to the full algorithm notes.
> **See also (full deep notes):**
> - Classification: [`12-catboost.md`](../1-supervised-learning/B-classification/12-catboost.md)
> - Regression: [`17-catboost.md`](../1-supervised-learning/A-regression/17-catboost.md)
> - Family concept: [`04-boosting.md`](04-boosting.md)
>
> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐⭐⭐

---

## Overview

| Property | Value |
|---|---|
| Algorithm Name | CatBoost (Categorical Boosting) |
| Ensemble family | Boosting (sequential), asymmetric/oblivious trees |
| Loss | Any differentiable loss + regularization |
| Base learner | **Oblivious (symmetric) decision trees** |
| Core ensemble idea | Solves the **prediction shift** problem with **ordered boosting** and handles categorical features with **ordered target statistics** — target leakage-free encodings that use only past observations |
| Output | F(x) = Σ_t η·f_t(x) |

---

## Definition

**Ensemble-perspective definition:** CatBoost is a gradient-boosting framework specialized for categorical features. It converts categorical variables with *ordered target statistics* (each sample's category encoding uses only the target values of previously-seen samples — a bagging-like, leak-free scheme) and trains trees with *ordered boosting* (each tree is trained on a random permutation's prefix, and only earlier-model predictions are used for gradients), eliminating the *prediction shift* that normally biases gradient boosting. Its trees are **oblivious/symmetric** — the same split across all leaves — which are fast, regularized, and less prone to overfitting.

---

## Ensemble Intuition (Why It Works Within Ensemble Theory)

### The prediction-shift problem (why CatBoost exists)
Standard gradient boosting uses the SAME data to compute gradients AND to fit the tree that should predict those gradients. That reuse creates *prediction shift* — a systematic bias (test-time predictions get distorted). CatBoost's **ordered boosting** breaks this dependency: for each permutation, gradients for sample i use only a model trained on the *previous* samples in the permutation.

### Categorical-feature leakage
Naive target encoding (replace category with mean target) leaks the sample's own label into its feature → overfitting. CatBoost uses **ordered target statistics** — encode category k for sample i using targets of earlier samples only (with the sample's own target EXCLUDED where needed) — so no self-leak.

### Oblivious trees
Every split at a given depth uses the SAME feature+threshold for all leaves (global split per level). This symmetric structure:
- reduces overfitting (fewer parameters),
- makes trees shallow & regularized,
- gives GPU-friendly, fast training,
- and yields a natural "feature interaction" interpretation.

---

## Core Formula(s)

### Ordered Target Statistics (OTS) for a categorical feature, sample i

```text
OTS_i = (Σ_{j<i in permutation} 1[cat_j = cat_i]·y_j + prior·a) / (count_before + a)
```

### Meaning
The category's encoded value for sample i is the mean target of *earlier-permutation* samples sharing that category, smoothed by a prior.

### Symbols
- j < i: samples appearing earlier in the random permutation.
- 1[cat_j = cat_i]: indicator of same category.
- prior: global mean target (smoothing).
- a: prior weight / smoothing strength.

### Intuition
Each sample's encoding never uses its own target (leak-free) and uses only observed-history → reduces overfitting vs standard target encoding.

### Worked mini example
Permutation [A?] of 3 samples with category `color` and binary targets:
```
index 1: color=red, y=1   → no earlier red → OTS = (prior·a)/(0+a) = prior
index 2: color=red, y=0   → earlier red count=1 (y=1) → OTS = (1 + prior·a)/(1+a)
index 3: color=blue, y=1  → earlier blue=0 → OTS = prior
```
Self-targets excluded per sample; history matters only. **Hand-verified structure.**

### Ordered boosting gradient rule

```text
For permutation σ, sample i's gradient at round t uses F_t built on σ[1..i−1] ONLY.
```

---

## How It Works (Step Flow)

```text
For each categorical feature, build ordered target statistics (multiple permutations);
encode features
for t = 1..T:
   for each permutation σ:
      fit an oblivious tree whose splits only use observations σ[1..i−1]'s model (ordered)
      (symmetric tree → same split per level; learn split via gradient gain)
   average the trees across permutations; F += η·(tree)
final additive model F
```

---

## Key Hyperparameters for the ENSEMBLE Behavior

| Hyperparameter | Ensemble role | Effect | Typical |
|---|---|---|---|
| iterations | rounds | capacity | 100–3000 |
| learning_rate | shrinkage | robustness | 0.01–0.3 |
| depth | oblivious tree depth | deeper → capacity but fewer params than level-wise | 4–10 |
| l2_leaf_reg | L2 on leaves | regularization | 1–10 |
| cat_features | which columns are categorical | enables OTS handling | pass explicitly |
| one_hot_max_size | auto-one-hot small cardinality | sharp shortcut for low-card cats | e.g., 2–10 |
| random_strength | random noise | augments diversity | tune small |
| bagging_temperature | permutation/bagging entropy | diversity | 0–1 |

---

## Advantages / Disadvantages

**Advantages:** state-of-the-art categorical handling (ordered statistics, leakage-free); reduces prediction shift via ordered boosting; symmetric trees → strong built-in regularization + low overfit; often needs little tuning; great GPU; good on small datasets.
**Disadvantages:** slower than LightGBM on big numeric-only data (sorted-based splits + permutations); ordered statistics cost extra memory; the pitfalls appear when you forget `cat_features` (then it treats them as numeric or silently encodes).

---

## Selection Guide (This vs Siblings)

| Option | Choose when |
|---|---|
| **CatBoost** | Categorical-heavy data, want leakage-free categorical encoding, robustness to shift |
| XGBoost (`04c`) | Numeric features, exact-split, mature ecosystem |
| LightGBM (`04d`) | Very large numeric/dense data, speed |
| Gradient Boosting (`04b`) | sklearn-only, simplest |

---

## Comparison Table — Boosting Family

| Factor | AdaBoost | Gradient Boost | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|---|
| Core idea | re-weight mistakes | fit negative gradient | Newton + reg | histogram + GOSS/EFB | ordered stats + obliv trees |
| Loss | exponential | arbitrary | any + reg | any + reg | any + reg |
| Missing values | no | no | yes | yes | yes |
| Categorical support | no | no | some | yes (native-ish) | **best (ordered TS)** |
| Speed | fast | medium | fast | **fastest** | medium |
| Fine-tuning sensitivity | high | medium | medium | high | **low-medium** |
| Best-use | simple 2-class | regression | competition | big data | categorical-heavy |
| Full deep note | `09-adaboost.md` | `08-gradient-boosting.md` | `10-xgboost.md` | `11-lightgbm.md` | `12-catboost.md` |

---

## Common Mistakes

```text
❌ Mistake: Forgetting to pass cat_features (or using object dtype naively)
🔥 Why: categoricals wrongly treated as numbers → distorted OTS never used
✅ Correct: pass cat_features explicitly (and keep dtype category/str)

❌ Mistake: Expecting CatBoost speed on numeric-only massive data
🔥 Why: sorted-based split logic + permutation overhead costs more than histograms
✅ Correct: for pure-numeric speed use LightGBM

❌ Mistake: Ignoring prediction-shift intuition in interviews/test questions
🔥 Why: very often tested: "why does CatBoost outperform on categoricals?"
✅ Correct: mention ordered target statistics (no self-leak) + ordered boosting (no shift)

❌ Mistake: Tuning like LightGBM/XGBoost (num_leaves, etc.)
🔥 Why: CatBoost exposes depth/l2_leaf_reg, not equivalent leaf-wise knobs
✅ Correct: use its native params (iterations, depth, l2_leaf_reg, cat_features)
```

---

## Interview Questions

1. **Q:** What is the prediction shift problem in GBDT? **A:** Reusing the same data to compute gradients and build the tree biases the model; test predictions become distorted. CatBoost's ordered boosting avoids it.
2. **Q:** What are ordered target statistics? **A:** Category encodings using only earlier-permutation targets (with own target excluded), smoothing with a prior — leak-free vs naive target encoding.
3. **Q:** Why oblivious/symmetric trees? **A:** Same split per level → fewer parameters → strong regularization, speed, low overfit, interpretable "global" splits.
4. **Q:** When is CatBoost NOT the best pick? **A:** Massive numeric-only data where LightGBM's histograms are much faster; also if custom target encoding is already done and you only have numerics.

---

## Cheat Sheet

| Item | Value |
|---|---|
| Family | Boosting (ordered, categorical-aware) |
| Goal | Bias reduction without categorical leakage/shift |
| Core Formula | OTS_i = (Σ_{j<i}1[y_j]+prior·a)/(count+a); ordered boosting gradient |
| Loss | any + regularization |
| Optimization | ordered gradient, symmetric trees |
| Use When | categorical-heavy, leakage-averse, robust defaults |
| Avoid When | pure-numeric speed-critical, massive dense data |
| See also (full) | `12-catboost.md` (clf), `17-catboost.md` (reg) |
| Concept | `04-boosting.md` |

---

## See Also

- **Full algorithm (classification):** [`12-catboost.md`](../1-supervised-learning/B-classification/12-catboost.md)
- **Full algorithm (regression):** [`17-catboost.md`](../1-supervised-learning/A-regression/17-catboost.md)
- Family concept: [`04-boosting.md`](04-boosting.md)
- Siblings: [`04a-adaboost.md`](04a-adaboost.md), [`04b-gradient-boosting.md`](04b-gradient-boosting.md), [`04c-xgboost.md`](04c-xgboost.md), [`04d-lightgbm.md`](04d-lightgbm.md)