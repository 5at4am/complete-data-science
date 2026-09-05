# 10. Decision Tree Regression

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆
>
> Journey: **if-then rules → splitting idea → variance reduction → overfitting → pruning → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Every model you've seen so far tries to fit a single function (line, curve) across all data. Decision Tree Regression does something radically different: it **chops the data into boxes** and predicts the average inside each box.

By the end you will be able to:

- explain how recursive splitting builds a tree,
- compute variance reduction by hand to find the best split,
- understand why trees overfit and how to prevent it,
- code a decision tree from scratch and with sklearn,
- and defend when to use — and not use — it.

> Everything in this note builds on one idea: **what if instead of fitting a function, we asked a sequence of yes/no questions?**

---

## 02. The Problem

Nisha is selling her used phone on OLX. She's collected prices:

| Brand | Age (years) | RAM (GB) | Price (₹ thousands) |
|---|---|---|---|
| Samsung | 2 | 4 | 8 |
| Samsung | 4 | 4 | 5 |
| iPhone | 1 | 6 | 35 |
| iPhone | 3 | 6 | 22 |
| OnePlus | 1 | 8 | 18 |
| OnePlus | 3 | 8 | 10 |

She wants to predict the price of a OnePlus, 2 years old, 8 GB RAM.

<!-- [QUESTION] -->
> **How would you predict this without any math?**

You'd probably think: "Well, OnePluses go for about 10–18K depending on age. A 2-year-old one is between those — maybe around 14K?"

You just made a decision tree prediction! You **split** by brand first (Samsung ≠ iPhone ≠ OnePlus), then within OnePlus, you looked at age. That's exactly what the algorithm does — recursively ask questions until you narrow down to similar examples.

---

## 03. Let's Think

Let's trace Nisha's thought process as a tree:

```text
                    Is it an iPhone?
                   /              \
                YES               NO
              iPhone data     Is it a OnePlus?
                             /            \
                          YES             NO
                       OnePlus data    Samsung data
```

<!-- [THINK_ABOUT_IT] -->
🤔 At each node, you ask a question that **separates the data into groups with similar prices**. A good question creates groups where the prices inside each group are close together (low variance).

If you asked "Is the RAM > 5?" instead, the split would be messy — iPhones and OnePluses would mix. Bad question = high variance in both children.

> A decision tree finds the **best question at each step** — the one that makes the children as homogeneous as possible.

---

## 04. Intuition

💡 **The idea in one line:**

> A Decision Tree **recursively splits** the data using feature thresholds, creating smaller and smaller groups with similar targets, and predicts the **mean** of each final group.

Each split tries to answer: "Which feature and threshold best separate the data into groups where the targets are most similar?"

The final "answer" for any new data point is: route it down the tree by answering each question, and predict the **average price** of all training points that followed the same path.

No math assumptions. No loss function. No line or curve. Just nested rules.

---

## 05. Visual

```text
Decision tree for phone prices:

        [Brand = iPhone?]
        /            \
      YES             NO
   iPhone         [Brand = OnePlus?]
   mean=28.5      /            \
               YES             NO
           OnePlus          Samsung
           mean=14          mean=6.5

Partition of feature space:
   Price (₹K)
   35│  ●(iPhone,1yr)
   22│  ●(iPhone,3yr)
   18│            ●(OnePlus,1yr)  ← predict ~14
   10│            ●(OnePlus,3yr)
    8│                       ●(Samsung,2yr)  ← predict ~6.5
    5│                       ●(Samsung,4yr)
     └──────────────────────────────
        iPhone    OnePlus    Samsung
```

---

## 06. First Prediction

Let's split the phone data using the best feature at each step.

**Step 1 — Root variance:** all prices = [8, 5, 35, 22, 18, 10]. Mean = 16.33.

```text
Var(all) = ((8−16.33)² + (5−16.33)² + ... + (10−16.33)²) / 6
         = (69.4 + 128.4 + 348.4 + 31.4 + 2.8 + 40.1) / 6
         = 103.5
```

**Step 2 — Best split: Brand = iPhone?**

Left (iPhone): [35, 22], mean = 28.5, Var = 42.25, size 2
Right (not iPhone): [8, 5, 18, 10], mean = 10.25, Var = 22.19, size 4

```text
Gain = 103.5 − [(2/6)·42.25 + (4/6)·22.19] = 103.5 − [14.08 + 14.79] = 74.6
```

Big gain! This split is excellent.

<!-- [TRY_IT] -->
For our OnePlus query (2 years, 8 GB): it's not an iPhone → right child. Is it OnePlus? Yes → left child. Prediction: mean of OnePlus prices = **(18 + 10)/2 = ₹14,000**.

> That matches Nisha's intuition from Section 02. The tree found the same logic Nisha used — and it does it automatically.

---

## 07. Core Concept

**Concept: Decision Tree Regression** — a method that:

1. starts with all data at the root,
2. finds the **(feature, threshold) split** that maximizes **variance reduction** (impurity decrease),
3. recurses on each child until a stopping criterion,
4. assigns each leaf the **mean** of its training samples.

```text
SPLIT CRITERION:  maximize  Gain = Var(parent) − weighted Var(children)
```

| Part | Symbol | Simple meaning |
|---|---|---|
| Node variance | Var(S) | How spread out targets are in this node |
| Split gain | Gain | How much a split reduces variance |
| Leaf prediction | ȳ_S | Mean target of training points in the leaf |
| Depth | d | How many nested splits (levels) |

> Everything else (overfitting, pruning, ensembles) is about **controlling how deep the tree grows**.

---

## 08. Terminology

### Node / Root / Leaf

> Simple: nodes are questions, root is the first question, leaves are final answers.
> Technical: internal nodes contain (feature, threshold) splits; leaves contain predictions (mean).

### Split / Branch

> Simple: a split is a yes/no question on a feature; a branch is the path taken.
> Technical: (feature j, threshold t) partitions samples into left (≤ t) and right (> t).

### Impurity (Variance)

> Simple: how mixed the target values are in a node. Low = homogeneous = good.
> Technical: Var(S) = (1/|S|)Σ(yᵢ − ȳ_S)², used as the impurity measure for regression.

### Gain

> Simple: how much better the children are than the parent.
> Technical: reduction in weighted child variance from a candidate split.

### Pruning

> Simple: cutting back the tree to prevent overfitting.
> Technical: removing branches that don't improve validation performance.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| Var(S) | target spread in node | impurity measure |
| Gain | improvement from split | variance reduction |
| depth | tree height | number of nested splits |
| max_depth | deepest allowed | hyperparameter to control overfitting |
| greedy | best local choice | not globally optimal (NP-hard) |

> ⚠️ Common mistake: "Decision trees minimize classification error." In **regression**, they minimize **variance** (not Gini/entropy, which are for classification).

---

## 09. Mathematics

### Node Variance (impurity)

```text
Var(S) = (1/|S|)·Σᵢ∈S (yᵢ − ȳ_S)²
```

<!-- [CALCULATION] -->
Node targets: [2, 4, 6]. Mean = 4.

```text
Var = ((2−4)² + (4−4)² + (6−4)²) / 3 = (4 + 0 + 4) / 3 = 2.67
```

### Split Gain

```text
Gain = Var(S) − [ (|S_L|/|S|)·Var(S_L) + (|S_R|/|S|)·Var(S_R) ]
```

Parent [2, 4, 6, 8], Var = 5.0. Split into {2, 4} (Var=1) and {6, 8} (Var=1):

```text
Gain = 5.0 − [ (2/4)·1 + (2/4)·1 ] = 5.0 − 1.0 = 4.0
```

Big gain → great split.

### Leaf Prediction

```text
ŷ = ȳ_S = mean of targets in leaf S
```

> 💡 Intuition: the leaf simply quotes the **average** of everything it has seen. No fancy function — just the mean.

---

## 10. Numerical Example

Data: x = [1, 2, 3, 4], y = [2, 4, 3, 8].

<!-- [CALCULATION] -->

**Step 1 — Root variance:**

```text
mean = (2+4+3+8)/4 = 4.25
Var = ((2−4.25)²+(4−4.25)²+(3−4.25)²+(8−4.25)²)/4
    = (5.0625+0.0625+1.5625+14.0625)/4 = 5.1875
```

**Step 2 — Candidate split x ≤ 2.5:**

```text
Left:  [2,4]  mean=3   Var=1      size 2
Right: [3,8]  mean=5.5 Var=6.25   size 2
Gain = 5.1875 − [(2/4)·1 + (2/4)·6.25] = 5.1875 − 3.625 = 1.5625
```

**Step 3 — Candidate split x ≤ 1.5:**

```text
Left:  [2]  Var=0    size 1
Right: [4,3,8]  mean=5  Var=4.667  size 3
Gain = 5.1875 − [(1/4)·0 + (3/4)·4.667] = 5.1875 − 3.5 = 1.6875
```

**Step 4 — Candidate split x ≤ 3.5:**

```text
Left:  [2,4,3]  mean=3  Var=0.667  size 3
Right: [8]      Var=0   size 1
Gain = 5.1875 − [(3/4)·0.667 + (1/4)·0] = 5.1875 − 0.5 = 4.6875
```

**Step 5 — Best: x ≤ 3.5 (gain 4.6875).**

```text
Left:  x=[1,2,3], y=[2,4,3], mean=3 → leaf prediction = 3
Right: x=[4],     y=[8],     mean=8 → leaf prediction = 8
```

Predictions: x ≤ 3.5 → 3; x > 3.5 → 8.

> ✅ VERIFIED — hand-verified. Best first split is x ≤ 3.5 with gain 4.6875; leaves predict 3 and 8.

---

## 11. How It Works

```text
STEP 1   Start at root with all samples
STEP 2   Compute node variance
STEP 3   For each feature × threshold:
             compute split gain
STEP 4   Pick the (feature, threshold) with maximum gain
STEP 5   Split into left (≤) and right (>) children
STEP 6   Recurse into each child
STEP 7   Stop when: depth limit / min leaf size / no gain
STEP 8   Leaves store mean target
STEP 9   Predict: route new x down the tree → leaf mean
```

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. Start with all data at root
     ↓
2. For each feature, sort values, try all midpoints as thresholds
     ↓
3. Compute gain for each candidate split
     ↓
4. Choose best (feature, threshold) → create internal node
     ↓
5. Partition data into left (≤ threshold) and right (> threshold)
     ↓
6. Recurse on each child
     ↓
7. Stop: max_depth / min_samples_leaf / no valid split
     ↓
8. Store leaf means
```

```text
model.predict(X_new)
     ↓
For each new x:
    start at root
    while not leaf:
        if x[feature] ≤ threshold: go left
        else: go right
    return leaf.mean
```

> Note: prediction is O(depth) — just following a path. Very fast.

---

## 13. From Scratch

### Version 1 — core impurity + best split

```python
import numpy as np

def node_variance(y):
    if len(y) == 0:
        return 0.0
    return np.var(y)

def best_split(X, y):
    n, m = X.shape
    parent_var = node_variance(y)
    best_gain, best_j, best_t = -1, None, None
    for j in range(m):
        order = np.argsort(X[:, j])
        xs = X[order, j]
        ys = y[order]
        for i in range(1, n):
            if xs[i] == xs[i-1]:
                continue
            t = (xs[i] + xs[i-1]) / 2.0
            gain = parent_var - (i/n * node_variance(ys[:i]) + (n-i)/n * node_variance(ys[i:]))
            if gain > best_gain:
                best_gain, best_j, best_t = gain, j, t
    return best_j, best_t
```

### Version 2 — full recursive tree

```python
class DecisionTreeRegressor:
    def __init__(self, max_depth=None, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build(np.asarray(X, dtype=float),
                                 np.asarray(y, float), depth=0)

    def _build(self, X, y, depth):
        node = {'value': np.mean(y), 'n': len(y)}
        if (self.max_depth is not None and depth >= self.max_depth) \
           or len(y) <= self.min_samples_leaf or len(np.unique(y)) == 1:
            node['leaf'] = True
            return node
        j, t = best_split(X, y)
        if j is None:
            node['leaf'] = True
            return node
        mask = X[:, j] <= t
        node.update(leaf=False, feature=j, threshold=t)
        node['left'] = self._build(X[mask], y[mask], depth+1)
        node['right'] = self._build(X[~mask], y[~mask], depth+1)
        return node

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in np.asarray(X, dtype=float)])

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])
```

### Version 3 — same thing, just for completeness

The Version 2 class is already library-ready. The key difference from sklearn is that sklearn uses optimized C code for split finding — but the logic is identical.

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor, plot_tree
import matplotlib.pyplot as plt

X = np.sort(np.random.RandomState(0).rand(100, 1), axis=0)
y = np.sin(6 * X).ravel() + np.random.RandomState(0).randn(100) * 0.1

model = DecisionTreeRegressor(max_depth=4, random_state=0)
model.fit(X, y)

print("Feature importances:", model.feature_importances_)
print("R²:", model.score(X, y))

plt.figure(figsize=(12, 6))
plot_tree(model, filled=True, feature_names=['x'])
plt.show()
```

> `plot_tree` visualizes the tree as a flowchart — each node shows the split condition, variance, samples, and predicted value.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
node = {'value': np.mean(y), 'n': len(y)}
```
> Every node stores the mean target — this is the prediction if this node becomes a leaf.

```python
j, t = best_split(X, y)
```
> Greedy search: try all features × all thresholds, pick the one with maximum variance reduction. This is the core of tree building.

```python
mask = X[:, j] <= t
node['left'] = self._build(X[mask], y[mask], depth+1)
node['right'] = self._build(X[~mask], y[~mask], depth+1)
```
> Partition data and recurse. Left child gets samples ≤ threshold, right gets > threshold.

```python
if (self.max_depth is not None and depth >= self.max_depth)
   or len(y) <= self.min_samples_leaf:
    node['leaf'] = True
```
> Stopping criteria prevent infinite recursion and control overfitting.

> 🧠 Every line maps directly to the algorithm from Section 11. Trees are one of the most intuitive algorithms to implement.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
### Experiment A — slide the depth

```text
max_depth = 1  →  two regions only (very simple, underfit)
max_depth = 3  →  eight regions at most (reasonable)
max_depth = 10 →  each point gets its own leaf (overfit, training R² = 1.0)
max_depth = None → fully grown, memorizes everything
```

> What to notice: **training R² increases monotonically** with depth (toward 1.0). But **test R² peaks and then drops** — the classic overfitting signal.

### Experiment B — overfitting visualization (code)

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)
X = np.sort(np.random.rand(200, 1), axis=0) * 10
y = np.sin(X.ravel()) * 3 + np.random.randn(200) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

for depth in [1, 3, 5, 10, None]:
    model = DecisionTreeRegressor(max_depth=depth, random_state=0)
    model.fit(X_train, y_train)
    print(f"depth={str(depth):>5}  train_R²={model.score(X_train,y_train):.3f}  test_R²={model.score(X_test,y_test):.3f}")
```

```text
depth=    1  train_R²=0.512  test_R²=0.488   (underfit)
depth=    3  train_R²=0.879  test_R²=0.861   (good)
depth=    5  train_R²=0.953  test_R²=0.912   (still good)
depth=   10  train_R²=0.999  test_R²=0.847   (overfitting!)
depth= None  train_R²=1.000  test_R²=0.785   (memorized noise)
```

> 📌 Training R² keeps rising (toward 1.0), but test R² peaks around depth=5 then drops. That gap is **overfitting** — the tree memorized the training noise.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
Code:

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# Data with a clear trend
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]], dtype=float)
y = np.array([2, 4, 3, 7, 6, 8, 7, 10, 9, 12], dtype=float)

# Overgrown tree
tree = DecisionTreeRegressor(max_depth=None, random_state=0)
tree.fit(X, y)
print("Train R²:", tree.score(X, y))

# Predict for x = 11 (extrapolation)
print("Prediction at x=11:", tree.predict([[11]]))
print("Prediction at x=50:", tree.predict([[50]]))
```

```text
Train R²: 1.000   (perfect — every point is its own leaf)
Prediction at x=11: 12.0   (= the leaf mean for x=10 → same as last training point)
Prediction at x=50: 12.0   (= STILL 12! can't go beyond training range)
```

> 💥 **Break pattern:** trees **cannot extrapolate**. The prediction outside the training range is always the leaf mean of the nearest training point. This is a fundamental limitation — the tree never "saw" x=50, so it has no idea what happens there.

**Fix:** use a linear model for extrapolation, or ensemble methods that average trees (which helps slightly but doesn't fully solve it).

---

## 18. What If...?

<!-- [WHAT_IF] -->
| You change… | What happens | Why |
|---|---|---|
| max_depth = 1 | Two regions, very simple | Only one split |
| max_depth = None | Memorizes every point | Overfits completely |
| min_samples_leaf = 50 | Very smooth, few leaves | Each leaf needs many points |
| Add a noisy feature | Tree may split on it | Greedy: if noise happens to reduce variance locally |
| Double the data | More stable splits | More evidence per node |
| Feature is categorical | Tree handles it (with encoding) | Threshold-based splits still work |
| Target is constant | Tree creates single leaf | No variance to reduce |

> 🤔 Think: which hyperparameter is the most important for controlling overfitting? → **max_depth**. It directly limits the tree's complexity. Secondary: min_samples_leaf (prevents tiny leaves).

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
tree structure  → (feature, threshold) at each internal node
leaf means     → prediction at each leaf
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| max_depth | deepest tree allowed | underfit | overfit | 3–10 |
| min_samples_leaf | min points per leaf | overfit | underfit | 1–10 |
| min_samples_split | min points to split | overfit | underfit | 2–20 |
| max_features | features per split | — | — | all (for single tree) |
| ccp_alpha | pruning strength | no pruning | aggressive pruning | tune |

> 📌 The golden rule: **shallow trees underfit, deep trees overfit.** Use cross-validation to find the sweet spot.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Piecewise-constant structure | target roughly constant in regions | tree predicts leaf means | residual plot | add features or use smoother model |
| Enough data per region | reliable leaf means | statistics | leaf sample counts | limit depth |
| Representative features | good splits are possible | tree finds patterns | EDA | feature engineering |
| No extrapolation needs | tree can't predict beyond range | leaf means | — | use linear model for extrapolation |

> Trees do **NOT** assume linearity, normality, homoscedasticity, or feature scaling — big advantages.

---

## 21. Data Requirements

```text
Target      → continuous numeric
Features    → numerical (sklearn); some libs handle categorical
Missing     → sklearn needs filled values; some handle NaN
Outliers    → fairly robust (split-based, not globally sensitive)
Scaling     → unnecessary (threshold splits unaffected by monotone transforms)
Small data  → risky (deep trees overfit)
Non-linearity → naturally handled by splits
Interactions → naturally captured by nested splits
```

---

## 22. Evaluation

| Metric | Formula | Use | Avoid |
|---|---|---|---|
| MSE | (1/n)Σ(y−ŷ)² | standard loss | — |
| RMSE | √MSE | interpretable | — |
| MAE | (1/n)Σ\|y−ŷ\| | robust | — |
| R² | 1 − SS_res/SS_tot | model quality | judging on training data! |
| **Feature importance** | weighted variance reduction | interpretability | — |

> **Critical:** a fully grown tree gets R² = 1.0 on training data. **Never evaluate a tree on its training set.** Always use a held-out test set.

---

## 23. Failure Cases

```text
OVERFITTING     → unlimited depth memorizes noise → awful test performance
HIGH VARIANCE   → small data changes flip the tree structure
NO EXTRAPOLATION → can't predict beyond training range
STEP-LIKE       → predictions are piecewise constant, not smooth
GREEDY          → locally optimal splits may miss globally optimal tree
NOISY FEATURES  → tree may split on noise if it locally reduces variance
```

---

## 24. Debugging

```text
1. Training R²=1, test R²=low?      → overfitting → reduce depth, increase min_samples_leaf
2. Both train and test R² low?       → underfitting → increase depth
3. Unstable feature importances?     → high variance → use Random Forest
4. Predictions are all the same?     → depth=1 or extreme pruning → increase depth
5. Steps too coarse?                 → tree too shallow → deepen or use smoother model
```

---

## 25. Compare

```text
Decision Tree:       "I'll ask yes/no questions to split the data into boxes."
Linear Regression:   "I'll fit one straight relationship."
Random Forest:       "I'll ask many different experts and average."
```

| Algorithm | Main idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Decision Tree | recursive splits | interpretable, no scaling | high variance, step-like | explainable rules |
| Linear Regression | straight line | simple, fast | can't curve | linear data |
| Random Forest | bagged trees | robust, accurate | less interpretable | accuracy + stability |
| Gradient Boosting | sequential trees | highest accuracy | tuning-heavy | top performance |

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  predict customer churn risk, need explainable rules
DATA:              5000 customers, 20 features
EDA:               non-linear relationships, interactions between features
FEATURES:          tenure, usage, complaints, plan_type (encode)
TARGET:            churn_amount (continuous)
SPLIT:             train/val/test
NO SCALING needed
TUNE:              max_depth, min_samples_leaf via CV
PRUNE:             ccp_alpha if overfitting
EVALUATE:          RMSE/R² on test, feature importance, visualize tree
INTERPRET:         export tree rules for business team
DEPLOY:            serve as decision rules
MONITOR:           check if new data follows same split logic
```

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** what does a decision tree split on? What does a leaf predict?
2. **Understand:** why does a single tree have high variance?
3. **Calculate:** compute variance and split gain for a small node by hand.
4. **Apply:** given a scatter plot, draw the tree splits you'd expect.
5. **Debug:** training R² = 1.0, test R² = 0.6 — what's the diagnosis?
6. **Experiment:** run the overfitting experiment (Section 16B) at 8 depths; graph the bias-variance tradeoff.
7. **Build:** house price project: EDA → fit tree → tune depth → prune → visualize → interpret rules → report.
8. **Explain:** explain a decision tree to a business stakeholder in 30 seconds.

---

## 28. Interview

### Beginner
- **What is a decision tree?** A model that splits data recursively on feature thresholds, predicting the mean target in each final region.
- **What's a leaf?** A terminal node holding the prediction (mean of its training targets).
- **How does it decide splits?** Pick (feature, threshold) that minimizes weighted child variance (maximizes gain).

### Intermediate
- **Why do trees overfit?** Unlimited depth means each point gets its own leaf, memorizing noise.
- **How do you prevent overfitting?** Limit max_depth, set min_samples_leaf, prune with ccp_alpha.
- **Why no feature scaling needed?** Splits are threshold-based; monotone scaling doesn't change the order or splits.
- **How is feature importance computed?** Sum of variance reductions contributed by each feature across all splits.

### Advanced
- **What is greedy splitting and why?** Choose best local split at each node; global optimum is NP-hard.
- **Why is a single tree high-variance?** Small data changes reorder the best split → very different tree structures.
- **What is cost-complexity pruning?** ccp_alpha trades tree size (number of leaves) vs fit quality; larger α → smaller tree.

---

## 29. GATE / Exam

<!-- [GATE] -->
**Formulas worth memorizing:**

```text
Node impurity (regression): Var(S) = (1/|S|)Σ(y − ȳ)²
Split gain: Var(S) − [ (|S_L|/|S|)Var(S_L) + (|S_R|/|S|)Var(S_R) ]
Leaf prediction: ŷ = ȳ_S
```

**Common traps:**
- Confusing regression impurity (variance/MSE) with classification impurity (Gini/entropy).
- Forgetting trees **can't extrapolate** — prediction outside training range is constant.
- Assuming tree predictions are smooth (they're **piecewise constant/step-like**).
- Not recognizing that a fully grown tree has zero training error but poor generalization.

> **Representative pattern question (NOT a past GATE PYQ):** "Given node targets {2, 4, 6, 8}, find the split gain for {2, 4} | {6, 8}." Answer: parent Var = 5.0, children Var = 1.0 each → Gain = 5.0 − 1.0 = **4.0**.

---

## 30. Deep Dive

<!-- [DEEP_DIVE] -->
<details>
<summary>Click to open the derivation + theory</summary>

### Why variance as impurity?

Minimizing the sum of squared errors within leaves is equivalent to minimizing the weighted sum of node variances. Since the total sum of squares is fixed, maximizing the split gain (variance reduction) is equivalent to minimizing the leaf-level SSE.

### Complexity of split finding

For each feature: sort O(n log n), scan all n−1 thresholds O(n). For m features: O(m·n log n) per node. With depth d: O(d·m·n log n) total.

### Global optimality is NP-hard

Finding the tree with minimum test error over all possible trees is NP-hard. Greedy splitting is a heuristic — it finds a locally optimal split at each node but not the globally optimal tree.

### Pruning (cost-complexity)

```text
minimize  |leaves| · α + Σ_leaf SSE(leaf)
```

α controls the tradeoff: larger α → fewer leaves → simpler tree. This is equivalent to AIC/BIC style penalization.

### Feature importance

```text
importance(j) = Σ over all splits using feature j: (samples at split / total samples) × gain
```

Normalized so importances sum to 1. Measures total variance reduction contributed by each feature.

</details>

---

## 31. Teach Back

> **Explain in 30 seconds:** "A decision tree asks a series of yes/no questions about your features, splitting data into smaller groups. Each final group predicts the average of its training examples. It's like sorting items into categories until each category is specific enough."

> **Explain to a 12-year-old:** "Imagine you're guessing an animal. You ask 'Does it fly?' Then 'Is it big?' Each answer narrows it down. That's a decision tree — ask questions until you know the answer."

> **Explain in an interview:** add: greedy variance reduction, O(m·n log n) per node, overfitting via depth, pruning, feature importance = weighted gain.

> **Explain the mathematics:** write the variance formula, gain formula, show why greedy is necessary (NP-hard global).

---

## 32. Mastery Test

**Without looking at notes:**

1. How does a decision tree pick a split?
2. What is variance impurity for regression?
3. Compute split gain for a simple 4-point node.
4. What does a leaf predict?
5. Name 3 stopping criteria.
6. Why is a single tree high-variance?
7. Why can't trees extrapolate?
8. Why no feature scaling needed?
9. Compare with Random Forest in one sentence.
10. State one scenario where a single tree is the right choice.

---

## 33. Cheat Sheet

```text
Algorithm : Decision Tree Regression · Supervised → Regression · Non-parametric
Goal      : recursive region partition → predict leaf mean
Split     : minimize weighted child variance = maximize gain
Impurity  : Var(S) = (1/|S|)Σ(y − ȳ)²
Gain      : Var(parent) − weighted Var(children)
Prediction: leaf mean (piecewise constant)
Learn     : tree structure (splits + leaf means)
Tune      : max_depth, min_samples_leaf, ccp_alpha
Assumptions: piecewise-constant structure, enough data per leaf
Use when  : interpretability, non-linear, interactions, no scaling
Avoid when: extrapolation, smooth predictions, huge data
Related   : Random Forest · Extra Trees · Boosting
Key insight: greedy splits → interpretable rules, but high variance
```

---

## 34. What Next?

You just built the foundation for all tree ensembles.

```text
Decision Tree Regression (single tree)
   ├── Random Forest       (many trees, bagging)        → 11
   ├── Extra Trees         (random splits + full data)  → 12
   ├── Gradient Boosting   (sequential trees)           → later
   └── XGBoost / LightGBM (optimized boosting)         → later
```

> Next recommended: **11. Random Forest Regression** — it fixes the single tree's biggest weakness (high variance) by building *many* trees on random subsets of data and features, then averaging. The "wisdom of the crowd" for regression.
