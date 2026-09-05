# 10. Decision Tree Regression

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Decision Tree Regression |
| Category | Supervised Learning |
| Type | Regression |
| Parametric / Non-parametric | Non-parametric |
| Generative / Discriminative | Discriminative |
| Main Objective | Recursively partition the feature space into regions and predict the mean target value within each region |
| Input | Feature matrix X (n×m), target y (continuous) |
| Output | Continuous prediction ŷ (region mean) |
| Core Idea | Learn a series of if-then rules on features that split data to minimize variance of targets within leaves |
| Typical Use Cases | Interpretable regression, non-linear and interaction-rich data, baseline for ensembles, feature importance |

---

## 02. One-Line Definition

### Beginner Definition
A decision tree predicts by asking a sequence of "yes/no" questions about your data (like "is size > 50?") and giving the average value of all similar past examples that end up in the same final box.

### Technical Definition
Decision Tree Regression builds a binary tree by repeatedly splitting features at thresholds to minimize the variance (or MSE) of the target within each child node, assigning each leaf the mean target of its training samples.

---

## 03. Intuition

Imagine sorting people into price brackets for used phones. You ask: "Is it an iPhone?" → if yes, "Is it newer than 2020?" → each yes/no path leads to a final group, and within each group you quote the average resale price.

That's a decision tree: it asks feature questions, splitting data into ever-smaller, more homogeneous groups. Each split tries to make the target values *inside* each branch as similar as possible (low variance). The final answer for a new point is the average of the training prices that followed the same path.

No math assumptions, no need to draw a line — just nested rules.

---

## 04. Problem It Solves

**Problem:** Linear models miss non-linear patterns and interactions between features, and aren't interpretable in complex forms.

**Example:** House prices where the effect of bedrooms depends on location (an interaction) and the relationship is non-linear. A tree naturally captures interactions and non-linearity purely from data splits.

Why useful: interpretable (visualize as rules), handles non-linearity & interactions automatically, needs no scaling, works on mixed feature types.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning
│   └── Regression
│       ├── Linear Models
│       ├── SVR
│       ├── Tree-based
│       │   ├── Decision Tree Regression   ← YOU ARE HERE
│       │   ├── Random Forest
│       │   ├── Extra Trees
│       │   └── Gradient Boosting (trees)
│       └── Neural Networks
└── (Trees = foundation of ensembles)
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Node | A decision point | Where data is split by a feature-threshold |
| Root | The first decision | Top node; whole dataset |
| Leaf | A final box (no more splits) | Terminal node; holds prediction (mean) |
| Branch | A path taken | A feature-condition test |
| Split | A question that separates data | (feature, threshold) dividing samples |
| Impurity / variance | How mixed a node's targets are | Variance of y within node; lowered by split |
| Depth | Tree length | Number of nested splits |
| Pruning | Trimming the tree | Removing branches to prevent overfitting |
| Greedy splitting | One best split at a time | Local optimization, not global |

---

## 07. Input and Output

**Input:** X (n×m) arbitrary numeric/categorical, y continuous.
**Output:** prediction ŷ (leaf mean).

**Parameters learned:** tree structure (splits: feature index + threshold at each node), leaf means.

**Hyperparameters:** max_depth, min_samples_split, min_samples_leaf, max_features, criterion (e.g., squared_error / variance), ccp_alpha (pruning).

---

## 08. Mathematical Foundation

At a node with a set S of samples, we want a split (feature j, threshold t) that partitions S into S_left and S_right, minimizing the weighted target variance of the children.

**Variance of a node (targets y):**
```text
Var(S) = (1/|S|)·Σᵢ∈S (yᵢ − ȳ_S)²
```
where ȳ_S is the mean of y in S.

**Split quality (regression):**
```text
Gain = Var(S) − ( |S_left|/|S|·Var(S_left) + |S_right|/|S|·Var(S_right) )
```

We choose the (feature, threshold) maximizing Gain.

**Notation:**
- `S` = sample set at a node
- `|S|` = count of samples
- `ȳ_S` = mean target in S
- `Var` = target variance (used as impurity for regression)
- `S_left`, `S_right` = child groups after split

**Required math:** variance, means, basic combinatorics.

---

## 09. Core Formula

### Node Variance (impurity)

```text
Var(S) = (1/|S|)·Σᵢ∈S (yᵢ − ȳ_S)²
```

#### Meaning
How spread out the target values are within a node. Low variance = homogeneous = good.

#### Symbols
- `S` = set of samples in node
- `|S|` = number of samples
- `yᵢ` = target of sample i
- `ȳ_S` = mean of targets in node
- `Var(S)` = node impurity

#### Intuition
Splits aim to reduce this variance — grouping similar targets together.

#### Example
Node targets: [2, 4, 6]. Mean = 4. Variance = ((2−4)² + (4−4)² + (6−4)²)/3 = (4+0+4)/3 = 2.67.

---

### Split Gain (squared error reduction)

```text
Gain = Var(S) − [ (|S_L|/|S|)·Var(S_L) + (|S_R|/|S|)·Var(S_R) ]
```

#### Meaning
How much a candidate split reduces impurity, weighted by child sizes.

#### Symbols
- `Var(S)` = parent variance
- `|S_L|, |S_R|` = sizes of left/right children
- `Var(S_L), Var(S_R)` = child variances

#### Intuition
We pick the split with the largest gain — the one making both children as pure as possible.

#### Example
Parent [2,4,6,8], Var = (9+1+1+9)/4 = 5. Split into {2,4} (Var=1) and {6,8} (Var=1), sizes 2 & 2:
```text
Gain = 5 − [ (2/4)·1 + (2/4)·1 ] = 5 − (0.5 + 0.5) = 4.0
```
Big gain → great split.

---

## 10. Derivation

**Step 1 — Start at root with all data. Compute node variance.**

**Step 2 — Enumerate candidate splits.** For each feature j, sort samples by feature value; try all midpoints between consecutive distinct values as thresholds t.

**Step 3 — For each candidate split, compute the gain** (variance reduction as in §9).

**Step 4 — Choose the split with maximum gain** — this is the *greedy* step: best local split at the root.

**Step 5 — Recurse.** Apply the same procedure to each child node until a stopping criterion (depth limit, min leaf size, no gain).

**Step 6 — Leaf prediction** = mean of targets in that leaf.

**Why greedy?** Finding the globally optimal tree is NP-hard; greedy one-split-at-a-time gives a practical local optimum.

---

## 11. How the Algorithm Works

```text
Input (X, y)
    ↓
Start at root with all samples
    ↓
Compute node variance
    ↓
For each feature × threshold:
    compute split gain
    ↓
Pick the best (feature, threshold)
    ↓
Split into left/right children
    ↓
Recurse into each child
    ↓
Repeat until stopping criterion (depth / min leaf / no gain)
    ↓
Leaves store mean target
    ↓
Predict: route new x down tree → leaf mean
```

---

## 12. Training Process

**Pre-training:** choose hyperparameters (max_depth, min_samples_split, etc.), criterion.

**During training:** recursively find best splits greedily, building the tree structure.

**What is learned:** the set of (feature, threshold) splits and each leaf's mean.

**Stopping criteria:** max_depth reached, min_samples_split, min_samples_leaf, no gain from further splits.

**Final model:** a tree of nodes + leaves with means.

---

## 13. Objective Function / Loss Function

The objective at each split is to **minimize the total child variance (sum of squared errors)**:

```text
Minimize  (|S_L|/|S|)·Var(S_L) + (|S_R|/|S|)·Var(S_R)
```

Equivalently, minimize the sum of squared errors of predictions (leaf means) within each region.

Why variance/MSE? It leads to splitting regions that are homogeneous in target — intuitively "similar values grouped."

High variance in a node = poor homogeneity; low = good.

---

## 14. Optimization

**Definition:** choose the sequence of splits minimizing total leaf variance.

**Method:** greedy recursive binary splitting — at each node, evaluate all feature-threshold candidates and pick the best reduction. No gradient/learning rate; the discrete search is the "optimization."

**Complexity** of evaluating splits: for each feature, sort values O(n log n), scan thresholds O(n). Cheap per node.

**No smooth objective:** this is combinatorial optimization solved greedily; trees don't use gradients.

---

## 15. Complete Numerical Example

Data: x = [1, 2, 3, 4], y = [2, 4, 3, 8].

**Step 1 — Root variance:**
```text
mean = (2+4+3+8)/4 = 4.25
Var = ((2−4.25)²+(4−4.25)²+(3−4.25)²+(8−4.25)²)/4
    = (5.0625+0.0625+1.5625+14.0625)/4 = 20.75/4 = 5.1875
```

**Step 2 — Candidate split x≤2.5** (midpoint between 2 and 3):
```text
Left:  x=[1,2], y=[2,4],  mean=3,  Var=(1+1)/2=1, size 2
Right: x=[3,4], y=[3,8],  mean=5.5, Var=(6.25+6.25)/2=6.25, size 2
Gain = 5.1875 − [ (2/4)·1 + (2/4)·6.25 ] = 5.1875 − (0.5+3.125) = 1.5625
```

**Step 3 — Candidate split x≤1.5:**
```text
Left:  [2] Var=0 size1
Right: [4,3,8] mean=5, Var=(1+4+9)/3=4.667 size3
Gain = 5.1875 − [ (1/4)·0 + (3/4)·4.667 ] = 5.1875 − 3.5 = 1.6875
```

**Step 4 — Candidate split x≤3.5:**
```text
Left:  [2,4,3] mean=3 Var=(1+1+0)/3=0.667 size3
Right: [8] Var=0 size1
Gain = 5.1875 − [ (3/4)·0.667 + (1/4)·0 ] = 5.1875 − 0.5 = 4.6875
```

**Step 5 — Best is x≤3.5 (gain 4.6875).** Split:
```text
Left:  x=[1,2,3], y=[2,4,3], mean=3 → leaf prediction 3
Right: x=[4], y=[8], mean=8 → leaf 8
```

**(If we stop here) Predictions:** x≤3.5 → 3; x>3.5 → 8.

**VERIFIED EXAMPLE** — hand-verified. Best first split is x≤3.5 with gain 4.6875; resulting leaves predict 3 and 8.

---

## 16. Visual Explanation

```text
Tree structure:
        [x ≤ 3.5]
        /        \
   y=[2,4,3]    [8]
    mean=3      mean=8
      LEAF        LEAF

Partition of x-axis:
   y
   8|                 •(4,8)  → predict 8
   4|     •(2,4)
   3|  •(1,2) •(3,3)        → predict 3
   2|
    └───|───────|──────  x
        1   2   3   4
        └── x≤3.5 ──┘   └ x>3.5 ┘
```

---

## 17. Algorithm / Pseudocode

```text
1. Input: X, y; stopping criteria
2. Define function build_node(S):
     if stopping_criterion(S) or no valid split:
         create leaf with mean of y in S
         return
     best = None
     for each feature j:
         sort S by feature j
         for each threshold t between distinct values:
             split S into S_L, S_R
             gain = Var(S) − weighted child variances
             if gain > best.gain: best = (j, t, S_L, S_R)
     create internal node with best split
     build_node(S_L); build_node(S_R)
3. Return the tree
4. Predict: route x through splits to a leaf, return leaf mean
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _variance(self, y):
        if len(y) == 0:
            return 0.0
        return np.var(y)

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.tree = self._build(X, y, depth=0)

    def _best_split(self, X, y):
        n, m = X.shape
        parent_var = self._variance(y)
        best_gain, best_j, best_t = -1, None, None
        for j in range(m):
            order = np.argsort(X[:, j])
            xs = X[order, j]
            ys = y[order]
            for i in range(1, n):
                if xs[i] == xs[i-1]:
                    continue
                t = (xs[i] + xs[i-1]) / 2.0
                y_left = ys[:i]
                y_right = ys[i:]
                w = i / n
                gain = parent_var - (w * self._variance(y_left)
                                     + (1 - w) * self._variance(y_right))
                if gain > best_gain:
                    best_gain, best_j, best_t = gain, j, t
        return best_j, best_t

    def _build(self, X, y, depth):
        node = {'value': np.mean(y)}
        if (self.max_depth is not None and depth >= self.max_depth) \
           or len(y) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            node['leaf'] = True
            return node
        j, t = self._best_split(X, y)
        if j is None:
            node['leaf'] = True
            return node
        left = X[:, j] <= t
        node['leaf'] = False
        node['feature'] = j
        node['threshold'] = t
        node['left'] = self._build(X[left], y[left], depth + 1)
        node['right'] = self._build(X[~left], y[~left], depth + 1)
        return node

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.array([self._predict_one(x, self.tree) for x in X])
```

---

## 19. Code Explanation

```text
Line:  def _variance(self, y): return np.var(y)
   What: node impurity measure
   Why: regression splits minimize target variance
   Math: Var = (1/n)Σ(y−ȳ)²

Line:  def _best_split(self, X, y):
   What: searches all features × thresholds
   Why: greedy best local split
   Math: maximize gain = Var(S) − weighted children

Line:  gain = parent_var - (w*varL + (1-w)*varR)
   What: variance reduction of the split
   Why: choose split with largest reduction
   Math: weighted child variance subtraction

Line:  node['value'] = np.mean(y)
   What: leaf stores mean target
   Why: prediction is region mean
   Math: ŷ = ȳ_S
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error

X = np.sort(np.random.RandomState(0).rand(100, 1), axis=0)
y = np.sin(6*X).ravel() + np.random.RandomState(0).randn(100)*0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = DecisionTreeRegressor(max_depth=4, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("R²:", r2_score(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("Feature importances:", model.feature_importances_)

params = {'max_depth': [2, 4, 6, 8, None], 'min_samples_leaf': [1, 5, 10]}
grid = GridSearchCV(DecisionTreeRegressor(random_state=0), params, cv=5)
grid.fit(X_train, y_train)
print("Best:", grid.best_params_)
```

Also visualize: `from sklearn.tree import plot_tree; plot_tree(model)`.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| max_depth | Max tree depth | Deeper → more complex/overfit | 3–10, or None + prune |
| min_samples_split | Min samples to split a node | Higher → simpler | ~2–20 |
| min_samples_leaf | Min samples a leaf must have | Higher → smoother | ~1–10 |
| max_features | Features considered per split | Reduces variance | sqrt for some; all default |
| criterion | Split measure | squared_error (variance) | Keep default |
| ccp_alpha | Cost-complexity pruning | Higher → smaller tree | Tune when overfitting |

**Too deep:** overfits noise. **Too shallow:** underfits. **Tune:** grid search / cross-validation.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Tree structure (splits at each node)
- Leaf mean values

### Hyperparameters (chosen)
- max_depth, min_samples_split, min_samples_leaf, max_features, criterion, pruning alpha

---

## 23. Assumptions

| Assumption | What | Why | Check | If violated |
|---|---|---|---|---|
| Piecewise-constant structure | Target roughly constant within regions | Tree predicts leaf means | Residual vs feature | Tree still works, just step-like |
| Enough data per region | Reliable leaf means | Statistics | Leaf sample counts | Limit depth |
| Representative features | Splitting features available | Finding good splits | EDA | Feature engineering |
| No strong linear extrapolation needs | Tree can't extrapolate | Leaf means | — | Use linear model |

Note: trees do NOT assume linearity, normality, homoscedasticity, or feature scaling — big advantages.

---

## 24. Data Requirements

- **Type:** numeric or categorical (sklearn needs numeric; shared trees handle categorical).
- **Missing:** sklearn tree needs filled values; some libs handle NaN.
- **Outliers:** fairly robust (split-based, not mean-sensitive globally).
- **Scaling:** unnecessary (threshold-based splits unaffected by monotone transforms).
- **Dataset size:** moderate; deep trees overfit small data.
- **Non-linearity/interactions:** naturally handled.

---

## 25. Feature Scaling

**Unnecessary:** Trees split on raw feature thresholds; monotone scaling (min-max, z-score) doesn't change the split structure. No scaling needed.

---

## 26. Evaluation Metrics

(Same regression family: MSE, RMSE, MAE, R².)

**Additional tree-specific diagnostics:**
- **Feature importance:** fraction of variance reduction by each feature across splits.
- **Leaf size distribution:** very tiny leaves ⇒ overfitting.

**Training objective vs evaluation:** training minimizes leaf variance (in-sample); evaluate on held-out data with your metric (RMSE/R²). A deep tree gets 0 training error (each point its own leaf) but poor test R² — the classic overfit signal.

---

## 27. Advantages

| Advantage | Why matters |
|---|---|
| Highly interpretable | Visual rules, easy to explain |
| Non-parametric | No distribution assumptions |
| Handles non-linearity & interactions | Splits capture them automatically |
| No scaling needed | Threshold-based |
| Mixed feature types | Numeric + categorical |
| Robust to outliers (somewhat) | Split-based |
| Foundation for ensembles | Cores of RF, boosting |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| High variance | Small data changes → very different trees |
| Overfits easily | Deep trees memorize noise |
| Step-like predictions | Not smooth |
| Poor extrapolation | Can't predict beyond training range |
| Greedy, locally optimal | May miss globally optimal tree |
| Unstable structure | Feature importance varies |

---

## 29. When to Use

✓ Need interpretable rules.
✓ Non-linear / interaction-rich relationships.
✓ Mixed feature types.
✓ No scaling desired.
✓ Baseline before ensembles / to visualize splits.
✓ Anomaly/outlier robustness matters.

---

## 30. When NOT to Use

✗ Very large data (forests/boosting better).
✗ Need smooth continuous predictions.
✗ Need extrapolation beyond training range.
✗ High-dimensional data with weak signal (single tree unstable).
✗ Performance-critical (use ensemble for accuracy).

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Loan/credit decisioning (explainable) | borrower features | Decision Tree | Risk score |
| Medical diagnosis support | symptoms/tests | Decision Tree | Risk level |
| Churn prediction (with regression target) | usage features | Decision Tree | Churn prob/avg |
| Cost estimation | project attributes | Decision Tree | Estimated cost |
| Feature importance screening | many features | Decision Tree | Key drivers |

---

## 32. Failure Cases

- **Overfitting:** unbounded deep tree memorizes each training point → awful test R².
- **Unstable trees:** two correlated splits; tiny data change flips structure.
- **Extrapolation failure:** predicting outside training range impossible.
- **Imbalanced/step function:** inherently discrete predictions.
- **High-dim:** many irrelevant features → splits may pick noise.

---

## 33. Overfitting and Underfitting

- **Overfitting:** too deep, min_samples too small → each leaf tiny, training perfect, test poor.
- **Underfitting:** too shallow, min_samples large → coarse regions miss pattern.
- **Control:** limit depth, set min_samples_leaf, prune (ccp_alpha), or bag/boost.

---

## 34. Bias-Variance Perspective

- **Bias:** a single tree has low bias (can fit anything given depth) but **high variance** — the classic single-tree problem.
- **Variance reduction:** bagging (Random Forest) and pruning reduce variance.
- **Tradeoff:** deep tree = low bias high variance; shallow = higher bias lower variance. This variance is why single trees are often replaced by ensembles.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Decision Tree | Recursive splits | Interpretable | High variance, step-like | Explainable rules |
| Random Forest | Bagged trees | Reduced variance | Less interpretable | Accuracy + robustness |
| Extra Trees | Random splits | Faster, lower variance | Randomness | Speed |
| Gradient Boosting | Sequential trees | High accuracy | Sequential, tuning | Top performance |

---

## 36. Algorithm Selection Guide

```text
Need interpretable rules?
├── YES → single DECISION TREE
├── Accuracy, stable → RANDOM FOREST
├── Best accuracy → GRADIENT BOOSTING / XGB
└── Smooth non-linear, small data
        → SVR / Kernel methods
```

---

## 37. Common Mistakes

```text
❌ Letting the tree grow to depth and overfit
Why wrong: memorizes noise.
Correct: limit depth / min_samples_leaf / prune.

❌ Expecting extrapolation from a tree
Why wrong: leaves predict means; cannot extend beyond.
Correct: linear models for extrapolation.

❌ Interpreting a single tree's importances as stable truth
Why wrong: high variance → unstable.
Correct: use many trees (RF) for stable importance.

❌ Forgetting trees don't need scaling BUT look for interactions
Why wrong: missing the point that splits capture interactions.
Correct: add relevant features, set depth.

❌ Judging tree quality on training error
Why wrong: overfits.
Correct: evaluate on validation/test.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is a decision tree?**
A: A model that splits data recursively on feature thresholds, predicting the mean target in each final region.

**Q2. What's a leaf?**
A: A terminal node holding the prediction (mean of its training targets).

**Q3. How does it decide splits?**
A: Pick (feature, threshold) that minimizes child variance / maximizes gain.

### Intermediate
**Q4. Why do trees overfit?**
A: Depth unlimited → each point its own leaf, memorizing noise.

**Q5. How do you prevent overfitting?**
A: Limit max_depth, set min_samples_leaf, prune.

**Q6. Why no feature scaling needed?**
A: Splits are threshold-based; monotone scaling doesn't change them.

### Advanced
**Q7. What is greedy splitting and why?**
A: Choose best local split at each node (global optimum is NP-hard).

**Q8. Why is a single tree high-variance?**
A: Small data changes reorder the best split → very different trees.

**Q9. How is feature importance computed?**
A: Sum of variance reductions (weighted by samples) over splits using each feature.

---

## 39. GATE / Exam Perspective

**Key formulas:**
```text
Node impurity (regression): Var = (1/n)Σ(y−ȳ)²
Split gain: Var(S) − [ (|SL|/|S|)Var(SL) + (|SR|/|S|)Var(SR) ]
```

**Concepts:**
- Greedy splitting; stopping criteria.
- Overfitting & pruning.
- Non-parametric nature.
- No scaling needed.

> **Representative pattern question (NOT a past GATE PYQ):** "Given node {2,4,6,8}, find the split gain for {2,4}|{6,8}." Answer: 4.0 as in §9.

**Traps:**
- Confusing regression impurity (variance) with classification impurity (Gini/entropy).
- Forgetting trees can't extrapolate.
- Assuming tree prediction is smooth (it's piecewise constant).

---

## 40. Coding Practice

**Level 1:** Implement variance impurity.
**Level 2:** Implement best-split search on one feature.
**Level 3:** Implement full recursive tree (as §18).
**Level 4:** Debug overfitting: vary max_depth, plot train/test R².
**Level 5:** Compare with sklearn DecisionTreeRegressor.
**Level 6:** Prune with ccp_alpha; visualize tree.
**Level 7:** Case study — regression on continuous data (e.g., mpg/auto), build tree, tune, interpret, report feature importance & test R².

---

## 41. Practical ML Workflow

```text
Problem → interpretable non-linear regression
   ↓
EDA → density, relationships, interactions
   ↓
Clean → impute (trees tolerate some), outliers
   ↓
Encode categorical (one-hot for sklearn)
   ↓
Split → train/val/test
   ↓
No scaling needed
   ↓
Train → DecisionTreeRegressor
   ↓
Tune → depth, min_samples_leaf via CV
   ↓
Prune → ccp_alpha if overfitting
   ↓
Evaluate → RMSE/R² on test, importance
   ↓
Error analysis → mispredicted regions
   ↓
Deploy → export tree (rules) for interpretation
   ↓
Monitor → drift, retrain
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Building tree | O(m·n·log n) per depth level | Sorting by feature |
| Prediction | O(depth) | Route down tree |
| Space | O(nodes) | Number of nodes |
| Scaling with n | O(n log n) build | |
| Worst case | O(n·m) deep | |

---

## 43. Advanced Concepts

- **Pruning (cost-complexity):** ccp_alpha trades tree size vs fit.
- **Gini vs variance:** classification uses Gini/entropy; regression uses variance/MSE.
- **Feature importance:** weighted variance reduction.
- **C4.5/CART differences:** CART does binary splits; C4.5 handles categorical & missing.
- **Oblique trees:** splits on linear combinations of features (more expressive).
- **Trees as base learners** for Random Forest & boosting.

---

## 44. Connections to Other Algorithms

```text
Decision Tree Regression
   ├── foundation → Random Forest (bagging)
   ├── foundation → Extra Trees (random splits)
   ├── foundation → Gradient Boosting / XGB / LightGBM / CatBoost
   └── relation → Regression to variance impurity
```

---

## 45. If You Remember Only 5 Things

1. Trees split data on feature thresholds to minimize target variance (impurity).
2. They're non-parametric, need no scaling, capture non-linearity & interactions.
3. Prediction = leaf mean → piecewise-constant, no extrapolation.
4. Single trees overfit easily and are high-variance & unstable.
5. They're the building block for Random Forest and boosting ensembles.

---

## 46. Cheat Sheet

```text
Algorithm   : Decision Tree Regression
Category    : Supervised, Regression, non-parametric
Goal        : Recursive region partition
Input       : X (n×m), y
Output      : ŷ (leaf mean)
Core Formula: split minimize weighted child variance
Loss        : sum of squared errors (variance)
Optimization: greedy recursive splitting
Parameters  : tree splits + leaf means
Hyperparams : max_depth, min_samples_leaf/split, max_features, ccp_alpha
Assumptions : piecewise-constant structure, enough data/leaf
Advantages  : interpretable, non-parametric, no scaling, interactions
Disadvantages: high variance, overfit, step-like, no extrapolation
Use When    : interpretability, small/med data, mixed types
Avoid When  : huge data (ensembles), smooth/extrapolation needs
Related     : RF, Extra Trees, Boosting, CART/C4.5
Key Exam    : variance impurity; gain; greedy; overfit
Key Interv  : why high variance, prune, importance, scaling-unneeded
```

---

## 47. Final Mental Model

```text
Data (X, y)
   ↓
Greedy splits: choose (feature, threshold) minimizing child variance
   ↓
Recurse until stop (depth / min leaf / no gain)
   ↓
Leaves store mean target
   ↓
New x routed down splits → leaf mean = prediction
```

---

## 48. Knowledge Check

### Recall (5)
1. How does a tree pick a split?
2. What is variance impurity?
3. What does a leaf store?
4. Name 3 stopping criteria.
5. Why no scaling?

### Understanding (5)
6. Why do trees overfit?
7. Why are they high-variance?
8. Why can't they extrapolate?
9. What is greedy splitting?
10. How do they capture interactions?

### Application (5)
11. Compute variance & split gain for given data.
12. Choose hyperparams to avoid overfitting.
13. Interpret a small tree as rules.
14. Decide tree vs linear for a problem.
15. Read feature importance.

### Mathematical (5)
16. Write the node variance formula.
17. Write the split gain formula.
18. Explain weighted child variance.
19. Why is global optimal NP-hard?
20. How does depth relate to bias/variance?

### Interview (5)
21. "Why is a tree high-variance?"
22. "How do you prune?"
23. "Tree vs linear for extrapolation?"
24. "Why no scaling?"
25. "How compute feature importance?"

### Problem Solving (5)
26. Training R²=1, test poor — diagnosis?
27. Need explainable model to business — what?
28. Step-like predictions unwanted — alternative?
29. Leaf too small/noisy — hyperparam fix?
30. Correlated features unstable splits — improvement?

## Answers (explained)
1. Choose feature-threshold minimizing weighted child variance.
2. Var(S) = (1/n)Σ(y−ȳ)² — target spread in node.
3. The mean target of its training samples.
4. max_depth, min_samples_leaf, min_gain/no further reduction.
5. Threshold splits are invariant to monotone scaling.
6. Unlimited depth → memorizes noise in small leaves.
7. Small data changes flip the best split → unstable structure.
8. Predictions are leaf means inside the training envelope; outside unknown.
9. Choose best local split at each node (not global optimum).
10. Splits on different features in sequence create nested conditions (interactions).
11–30: apply formulas. For (28): use smoother models (linear/forest averaging/SVR). For (30): bagging (RF) averages over bootstrap instability.

---

## 49. Final Learning Checklist

- [ ] I can define decision tree regression
- [ ] I understand variance impurity
- [ ] I can compute split gain
- [ ] I understand greedy splitting
- [ ] I know leaf mean prediction
- [ ] I understand overfitting & pruning
- [ ] I know stopping criteria
- [ ] I understand the high-variance problem
- [ ] I know why no scaling needed
- [ ] I can interpret rules from a tree
- [ ] I can implement from scratch
- [ ] I can use sklearn DecisionTreeRegressor
- [ ] I can tune hyperparameters via CV
- [ ] I understand feature importance
- [ ] I know tree can't extrapolate
- [ ] I can compare with linear/forest/boosting
- [ ] I understand CART/C4.5 basics
- [ ] I know cost-complexity pruning
- [ ] I can apply in a workflow
- [ ] I know when NOT to use it

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** Variance/gain formulas and worked example verified (split x≤3.5, gain 4.6875, leaves 3 & 8).
- **Beginner-friendliness:** Phone-pricing analogy, tree ASCII, short paragraphs, tables.
- **Math depth:** Variance impurity, gain, derivation of greedy splitting.
- **Practical depth:** From-scratch tree, sklearn, pruning, feature importance, workflow.
- **Exam depth:** Impurity, gain, overfitting, non-PYQ representative questions.
- **Structure:** All 50 sections in order.

**Verified:** Section 15 worked example hand-verified.
