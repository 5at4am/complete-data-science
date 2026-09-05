# 04. Decision Tree (Classification)

<!-- [STORY] -->
> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐
>
> Journey: **story → guess → impurity → Gini/entropy → information gain → greedy split → prune → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

The Decision Tree is the **"twenty questions" classifier** — it learns human-readable if-then rules. It's also the building block of random forests and gradient boosting, so mastering it unlocks most of modern ML.

By the end you will be able to:

- predict a class by following if-then questions,
- compute entropy, Gini, and information gain by hand,
- explain *greedy* recursive splitting,
- control overfitting with depth and pruning,
- code a tree from scratch and with sklearn,
- break it deliberately and fix it,
- and defend when to use — and not use — it.

> One number tells the tree how good a question is. Let's find it.

---

## 02. The Problem

Priya works at a small bank. Her manager wants a **loan-approval rule** that a regulator, an auditor, and a panchayat member can all read and check — not a black box.

She shows you past decisions for 5 applicants:

| Income (₹k/mo) | Has job? | Loan approved? |
|---|---|---|
| 30 | No | No |
| 40 | Yes | Yes |
| 50 | No | No |
| 60 | Yes | Yes |
| 70 | Yes | Yes |

A new applicant comes in:

> **Income = ₹55k/month, Has job = Yes. Approve or reject?**

<!-- [QUESTION] -->
Before any math: glance at the pattern and guess.

**Your guess: Approve ☐   Reject ☐**

> 📌 The bank's real ask isn't just an answer — it's a **rule you can explain**. That's what makes trees special.

---

## 03. Let's Think

Look at the data and spot the pattern:

```text
Income     HasJob     Approved?
30,000     No         No
40,000     Yes        Yes
50,000     No         No
60,000     Yes        Yes
70,000     Yes        Yes
```

<!-- [THINK_ABOUT_IT] -->
🤔 What separates the "No" from the "Yes"?

> Everyone with **No job** was rejected. Everyone with a **job** was approved. The single most useful question is "Does this person have a job?"

That's the natural rule:

```text
IF has job == Yes  THEN approve
IF has job == No   THEN reject
```

> A human finds this in seconds. The decision tree, too — but it needs a **score** to decide *which* question (job? income? what threshold?) is the best one to ask first. That score is coming.

---

## 04. Intuition

A Decision Tree is the game **"20 Questions"** bottled as code.

You start with a pile of people. You pick the question that best **splits the pile into purer groups** — preferably every group having mostly one answer. Then you repeat on each group until no useful split remains.

<!-- [VISUAL] -->
```text
                    [ all 5 applicants ]
                    ( 3 Yes, 2 No )          ← mixed, impure
                           │
              "Has a job?" (best first question)
                    ┌──────┴──────┐
                    │             │
              No job (2)      job (3)
             (0 Yes, 2 No)  (3 Yes, 0 No)
             → reject       → approve
              PURE           PURE
```

💡 **The idea in one line:**

> A Decision Tree keeps asking "which question makes the groups most pure?" until every group is a single answer, then turns each group's path into a readable if-then rule.

The word that matters everywhere: **pure** — a group where everyone (or almost everyone) shares one class.

---

## 05. Visual First

Here's what "pure" vs "mixed" looks like, and the two rulers we use to measure it:

```text
Node with 8 people: 6 approve, 2 reject

        "how mixed is this group?"
   Gini = 1 − Σ p²            Entropy = −Σ p·log₂(p)
   = 1 − (p_A² + p_R²)        = −(p_A·log₂p_A + p_R·log₂p_R)
   = 1 − (0.5625 + 0.0625)    = −(0.75(−0.415) + 0.25(−2))
   = 0.375                    ≈ 0.811 bits
```

| Group | Gini | Entropy | Meaning |
|---|---|---|---|
| all one class (pure) | 0 | 0 | no uncertainty |
| 50/50 (max mix) | 0.5 | 1 | maximum uncertainty |
| 75/25 (moderately mixed) | 0.375 | 0.811 | in between |

> 📌 **Bigger = more mixed = worse.** The tree wants to drive every node's Gini/entropy toward 0.

Both rulers measure "how mixed" — they usually pick the same splits, so don't stress the choice.

---

## 06. First Prediction

Let's use our eyeball rule from Section 03 to predict the new applicant:

```text
Rule:  has job?  Yes → approve, No → reject
New applicant: income 55k, Has job = Yes  →  APPROVE
```

<!-- [TRY_IT] -->
> Model's first answer: **Approve.**

Bet it matches your Section 02 guess. A human would say this is obvious — but the tree doesn't "see" the pattern as we do. It has to *prove* which question is best using numbers.

So the real question, which drives the whole algorithm:

> **Among all possible questions ("has job?", "income ≤ 40?", "income ≤ 45?", ...), which one should the tree ask FIRST?**

That's where **information gain** comes in — next section.

---

## 07. Core Concept

Introducing the idea formally:

**Concept: Decision Tree (CART)** — a method that:

1. measures how **impure** a node is (Gini or entropy),
2. tries every feature × threshold split,
3. picks the split with the largest **information gain** (biggest impurity drop),
4. recursively repeats on the children,
5. stops when a node is pure, too small, or too deep — then labels each leaf with its majority class.

```text
Information Gain = impurity(parent) − weighted-avg impurity(children)
```

```text
Gini  = 1 − Σⱼ pⱼ²
Entropy = −Σⱼ pⱼ·log₂(pⱼ)      pⱼ = fraction of class j in the node
```

> The tree IS the set of rules. That readability is its superpower — and why banks and hospitals love it.

---

## 08. Terminology

### Node

> Simple: a question in the tree.
> Technical: a point where data is split by one feature test.

### Root node

> Simple: the first question.
> Technical: the top of the tree where splitting starts.

### Leaf node

> Simple: a final answer.
> Technical: a terminal node holding a class label (majority of its samples).

### Impurity

> Simple: how mixed the classes are.
> Technical: Gini `1−Σp²` or entropy `−Σp·log₂p` of a node's labels.

### Information gain (IG)

> Simple: how much cleaner a split makes things.
> Technical: `IG = I(parent) − Σ(n_k/n)·I(child_k)`.

### Greedy

> Simple: always take the locally-best split now.
> Technical: choose the best split at each node, without looking ahead.

### Pruning

> Simple: trimming overgrown branches.
> Technical: removing subtrees (pre- or post-) to beat overfitting.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| node | a question | split point |
| leaf | an answer | terminal class label |
| impurity | how mixed | Gini / entropy |
| IG | how much cleaner | parent − weighted children |
| greedy | best split now | no global lookahead |
| depth | how many questions deep | splits root→leaf |
| CART | the algorithm family | Classification And Regression Trees |

> ⚠️ Common mistake: confusing **Gini impurity** (a classification-split score) with the **Gini coefficient** (an economic inequality measure). Different things, same name.

---

## 09. Mathematics (gradual)

### Step M1 — Gini impurity

```text
G = 1 − Σⱼ pⱼ²
```

- `pⱼ` = fraction of class j in the node.
- Pure node (one class) → `1 − 1 = 0`.
- Two balanced classes → `1 − (0.5² + 0.5²) = 0.5`.

<!-- [CALCULATION] -->
```text
Node: 4 of class A, 2 of class B
p_A = 4/6 = 0.667, p_B = 2/6 = 0.333
G = 1 − (0.667² + 0.333²) = 1 − (0.444 + 0.111) = 0.444
```

### Step M2 — Entropy

```text
H = −Σⱼ pⱼ·log₂(pⱼ)
```

- Same node: `H = −(0.667·log₂0.667 + 0.333·log₂0.333) = −(−0.390 − 0.528) = 0.918` bits.

### Step M3 — Weighted impurity of children

A split sends samples left and right. Their average impurity, weighted by group size:

```text
I_split = (n_left / n_node)·I(left) + (n_right / n_node)·I(right)
```

### Step M4 — Information gain

```text
IG = I(parent) − I_split
```

- **Higher IG = better split.** The greedy algorithm picks the split with max IG.

---

## 10. Numerical Example

Tiny dataset — the exact 5 loan applicants:

| Applicant | Income | HasJob | Approved? |
|---|---|---|---|
| A | 30 | No | No |
| B | 40 | Yes | Yes |
| C | 50 | No | No |
| D | 60 | Yes | Yes |
| E | 70 | Yes | Yes |

We'll use Gini. First, the root's impurity:

```text
p_Yes = 3/5 = 0.6, p_No = 2/5 = 0.4
Gini(root) = 1 − (0.6² + 0.4²) = 1 − (0.36 + 0.16) = 0.48
```

<!-- [CALCULATION] -->
**Candidate split 1: income ≤ 45** → left {A(30,No), B(40,Yes)}, right {C,D,E}

```text
Gini(left)  = 1 − (0.5² + 0.5²) = 0.5
Gini(right) = 1 − ((2/3)² + (1/3)²) = 1 − (0.444 + 0.111) = 0.444
I_split = (2/5)(0.5) + (3/5)(0.444) = 0.2 + 0.267 = 0.467
IG = 0.48 − 0.467 = 0.013        ← tiny gain
```

**Candidate split 2: income ≤ 55** → left {A,B,C}, right {D,E}

```text
Gini(left)  = 1 − ((1/3)² + (2/3)²) = 0.444
Gini(right) = 1 − (0 + 1) = 0                     (pure!)
I_split = (3/5)(0.444) + (2/5)(0) = 0.267
IG = 0.48 − 0.267 = 0.213        ← better
```

**Candidate split 3: HasJob (≤0.5)** → left {A, C}, right {B, D, E}

```text
Gini(left)  = 1 − (0 + 1) = 0     (all No)
Gini(right) = 1 − (1 + 0) = 0     (all Yes)
I_split = 0
IG = 0.48 − 0 = 0.48              ← perfect split, best of all
```

**Winner: "Has job?" (IG = 0.48).**

```text
                  [root]  IG=0.48
             HasJob ≤ 0.5?
              ┌──────┴──────┐
        Job=No (2)      Job=Yes (3)
       0 Yes, 2 No      3 Yes, 0 No
       → No             → Yes
        (pure)           (pure)
```

Both children are pure, so we stop. Prediction for (55k, HasJob=Yes) → right branch → **Approve**.

> ✅ VERIFIED — every Gini value and IG hand-computed. This is exactly what Section 17's code will do.

> 🎯 Your turn: why did "income ≤ 45" score so poorly? *(Because it split the mostly-Yes group in half too — left was 50/50.)*

---

## 11. How It Works

```text
STEP 1   Have labeled data (x, y)
STEP 2   At the root, measure impurity
STEP 3   For every feature × candidate threshold:
             compute the weighted child impurity and IG
STEP 4   Pick the feature/threshold with max IG, split
STEP 5   Recurse into each child
STEP 6   Stop when pure / too small / too deep → make leaves
```

> That's the entire loop. "Recurse" just means "do Steps 2–5 again on each child."

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
build(root, depth=0):
    if stopping rule hit (pure / depth / min_samples): return leaf
    for each feature f:
        for each threshold t (midpoint between adjacent sorted values):
            split data on (f ≤ t)
            compute IG
    pick (f*, t*) with max IG
    node.test = (f*, t*)
    node.left  = build(X[f*≤t*], depth+1)
    node.right = build(X[f*>t*], depth+1)
     ↓
model.predict(X_new):
    for each row:
        walk root→leaf following the tests
        return leaf majority class (+ class probabilities from leaf counts)
```

> `fit()` is the recursion; `predict()` is just walking the tree.

---

## 13. From Scratch

### Version 1 — pure Python, readable

```python
from collections import Counter

def gini(y):
    if len(y) == 0:
        return 0.0
    p = list(Counter(y).values())
    n = len(y)
    return 1 - sum((c / n) ** 2 for c in p)

def information_gain(y, y_left, y_right):
    n = len(y)
    child = (len(y_left) / n) * gini(y_left)
    child += (len(y_right) / n) * gini(y_right)
    return gini(y) - child

def best_split(X, y):
    best_ig, best = -1, None
    for f in range(len(X[0])):
        vals = sorted(set(row[f] for row in X))
        for i in range(len(vals) - 1):
            t = (vals[i] + vals[i + 1]) / 2
            left = [j for j, row in enumerate(X) if row[f] <= t]
            right = [j for j in range(len(X)) if j not in left]
            yl = [y[j] for j in left]; yr = [y[j] for j in right]
            ig = information_gain(y, yl, yr)
            if ig > best_ig:
                best_ig, best = ig, (f, t)
    return best

X = [[30,0],[40,1],[50,0],[60,1],[70,1]]
y = [0,1,0,1,1]
print(best_split(X, y))     # (1, 0.5)  → feature 1 (HasJob) at 0.5 → IG max
```

> This is Section 10 in code: scan features × thresholds, score each by IG, return the best.

### Version 2 — recursive build with a node class

```python
import numpy as np
from collections import Counter

class Node:
    def __init__(self):
        self.feature = self.threshold = None
        self.left = self.right = None
        self.label = None
        self.is_leaf = False

class DecisionTree:
    def __init__(self, max_depth=5, min_samples=2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def _gini(self, y):
        if len(y) == 0:
            return 0.0
        p = Counter(y)
        n = len(y)
        return 1 - sum((v / n) ** 2 for v in p.values())

    def _best(self, X, y):
        best_ig, best = -1, None
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            for i in range(len(vals) - 1):
                t = (vals[i] + vals[i + 1]) / 2
                left = y[X[:, f] <= t]; right = y[X[:, f] > t]
                ig = self._gini(y) - (len(left)/len(y))*self._gini(left) \
                                   - (len(right)/len(y))*self._gini(right)
                if ig > best_ig:
                    best_ig, best = ig, (f, t)
        return best

    def _build(self, X, y, depth):
        node = Node()
        counts = Counter(y)
        node.label = counts.most_common(1)[0][0]
        if (depth >= self.max_depth or len(y) < self.min_samples
                or len(counts) == 1):
            node.is_leaf = True
            return node
        split = self._best(X, y)
        if split is None:
            node.is_leaf = True
            return node
        f, t = split
        node.feature, node.threshold = f, t
        left = X[:, f] <= t
        node.left = self._build(X[left], y[left], depth + 1)
        node.right = self._build(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return [self._predict_one(r, self.root) for r in np.asarray(X, float)]
```

### Version 3 — clean class (with score)

```python
import numpy as np
from collections import Counter

class DecisionTreeClassifier:
    def __init__(self, max_depth=5, min_samples_split=2, criterion="gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None

    def _impurity(self, y):
        if len(y) == 0:
            return 0.0
        p = np.array(sorted(Counter(y).values())) / len(y)
        if self.criterion == "gini":
            return 1 - (p ** 2).sum()
        return -np.sum(p * np.log2(p + 1e-12))

    def _best_split(self, X, y):
        best_ig, best = -1, None
        for f in range(X.shape[1]):
            vals = np.unique(X[:, f])
            for i in range(len(vals) - 1):
                t = (vals[i] + vals[i + 1]) / 2
                left = y[X[:, f] <= t]; right = y[X[:, f] > t]
                child = (len(left)/len(y))*self._impurity(left) \
                      + (len(right)/len(y))*self._impurity(right)
                ig = self._impurity(y) - child
                if ig > best_ig:
                    best_ig, best = ig, (f, t)
        return best

    def _build(self, X, y, depth):
        node = Node()
        node.label = Counter(y).most_common(1)[0][0]
        if (depth >= self.max_depth or len(y) < self.min_samples_split
                or len(set(y)) == 1):
            node.is_leaf = True
            return node
        split = self._best_split(X, y)
        if split is None:
            node.is_leaf = True
            return node
        f, t = split
        node.feature, node.threshold = f, t
        left = X[:, f] <= t
        node.left = self._build(X[left], y[left], depth + 1)
        node.right = self._build(X[~left], y[~left], depth + 1)
        return node

    def fit(self, X, y):
        self.root = self._build(np.asarray(X, float), np.asarray(y), 0)
        return self

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.label
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def predict(self, X):
        return np.array([self._predict_one(r, self.root)
                         for r in np.asarray(X, float)])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))
```

---

## 14. Library Implementation

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
model.fit(X_train, y_train)

print(f"Test accuracy: {model.score(X_test, y_test):.4f}")
print(export_text(model, feature_names=["sepal_len","sepal_wid","petal_len","petal_wid"]))
print(model.feature_importances_)     # impurity-based feature ranking

# Tune the key hyperparameters
param_grid = {"max_depth": [2,3,5,8,None],
              "min_samples_leaf": [1,3,5],
              "criterion": ["gini","entropy"]}
grid = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
```

> `export_text()` turns the tree into readable rules; `plot_tree()` draws it — both are golden for the "explain to a stakeholder" story. No feature scaling needed.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
t = (vals[i] + vals[i + 1]) / 2
```
> Candidate thresholds are **midpoints between adjacent distinct feature values**. Splitting exactly at a data value would be ambiguous, so we split between values.

```python
child = (len(left)/len(y))*self._impurity(left) + (len(right)/len(y))*self._impurity(right)
```
> The **weighted** child impurity from Section 09 — weighted by group size so big groups count more.

```python
ig = self._impurity(y) - child
```
> Information gain = parent impurity − weighted child impurity. The bigger, the better.

```python
if (depth >= self.max_depth or len(y) < self.min_samples_split or len(set(y)) == 1):
    node.is_leaf = True
```
> The stopping criteria: too deep, too few samples, or already pure → stop and make a leaf.

> 🧠 Every line maps to a formula from Sections 09–10. Nothing arbitrary.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> Sliders in the platform; otherwise run the code.

### Experiment A — slide the depth

A slider for `max_depth` on the loan data:

```text
depth = 1   →  just "HasJob?" — the stump. Simple, readable, but no nuance
depth = 3   →  "HasJob? then income ≤ 55? ..." — more rules
depth = None →  grows until every leaf is pure → 100% train, shaky test
```

> What to notice: deeper trees fit training better but generalize worse. Watch the **train vs test gap** grow with depth — that gap is overfitting.

### Experiment B — depth vs overfitting (code)

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(5)
X = rng.normal(0, 1, (300, 2))
y = (X[:, 0] * X[:, 1] > 0.3).astype(int)     # mildly noisy boundary

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
for depth in [1, 3, 5, None]:
    m = DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
    print(f"depth={str(depth):>4}  train={m.score(X_train,y_train):.3f}  test={m.score(X_test,y_test):.3f}")
```

```text
depth=1    train=0.66x  test=0.6xx   ← underfits
depth=3    train=0.7xx  test=0.7xx   ← good balance
depth=5    train=0.8xx  test=0.6xx   ← overfitting starts
depth=None train=1.000  test=0.5xx   ← memorized noise
```

> 📌 The moral: **the depth slider IS the bias-variance trade-off.** There's a sweet spot — find it with cross-validation, not eyeballing.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
**Experiment 1 — a unique ID column.**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier

X = np.arange(50).reshape(-1, 1)      # a near-unique "ID" column
y = (np.arange(50) % 2).astype(int)   # alternating labels

m = DecisionTreeClassifier().fit(X, y)
print("train:", m.score(X, y))
print("test :", np.mean(m.predict(X + 0.5) == y))   # new values between IDs
```

```text
train: 1.000     ← memorized every single ID perfectly
test : 0.00x     ← total garbage on new IDs
```

**What happened?** A unique-valued column lets the tree split off *single samples* — each its own pure leaf. Perfect memorization, zero generalization. Classic data-leak-style failure.

**Experiment 2 — max_depth=None on noisy small data.**

```python
import numpy as np
from sklearn.tree import DecisionTreeClassifier
rng = np.random.default_rng(2)
X = rng.normal(0, 1, (30, 2))
y = (X[:,0] + X[:,1] + rng.normal(0, 2, 30) > 0).astype(int)

m = DecisionTreeClassifier().fit(X, y)     # no limits
print("train:", m.score(X, y), " test:", np.mean(1 - m.predict(X) + 0))  # no held-out
# held-out would show the real story
```

```text
train: 1.000    ← memorizes the 30 noisy points
```
> 💥 **Break pattern:** greedy + unlimited depth + few samples = the tree carves leaves around every noise point. Fix with `max_depth`/`min_samples_leaf` (pre-prune) or cost-complexity `ccp_alpha` (post-prune, Section 33).

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Set max_depth=None on small data | 100% train, poor test | memorizes noise |
| Raise min_samples_leaf | Smoother leaves, less overfit | forces bigger, purer-ish groups |
| Switch Gini↔Entropy | Nearly identical trees | both measure the same mixing |
| Add a unique ID column | Tree memorizes IDs | each ID becomes a pure leaf |
| Add many irrelevant features | Some noise splits | greedy finds spurious patterns |
| Require approximating a diagonal boundary | Many splits needed | trees are axis-aligned only |
| Use a stump (depth=1) | High bias, stable | one question too coarse |
| Use many trees (forest) | Variance drops | bagging averages the trees |

> 🤔 Think: why does a tree need **no** feature scaling, unlike KNN? → It only compares a feature to a threshold in its own units; rescaling just rescales the threshold. No distances, no gradient.

---

## 19. Hyperparameters

**Learned by the model:**
- The tree structure: which feature test at each node + threshold.
- Leaf labels and class proportions.

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `criterion` | the impurity ruler | — | — | `'gini'` (default) |
| `max_depth` | how many levels | underfit (stump) | overfit | 3–10; tune |
| `min_samples_split` | min samples to split further | overfit | underfit | 2; raise to regularize |
| `min_samples_leaf` | min samples in a leaf | overfit | underfit | 1–5; raise for noise |
| `max_features` | features considered per split | more random | fast, less power | `'sqrt'` for forests |
| `max_leaf_nodes` | cap on leaves | underfit | overfit | prefer for size control |
| `class_weight` | fairness for imbalance | — | — | `'balanced'` |
| `ccp_alpha` | cost-complexity pruning penalty | less pruning | heavy pruning | tune via CV |

> 📌 **The two you'll tune first on noisy data: `max_depth` and `min_samples_leaf`.** They're the cheapest way to stop overfitting.

---

## 20. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Features are informative** | splitting reduces impurity meaningfully | greedy picks feature splits | compare vs random features | feature selection/engineering |
| **Separable by thresholds** | class regions cut by feature values | axis-aligned splits only | visualize; try oblique/kernel model | need more splits, or other model |
| **Representative data** | train ≈ test distribution | any model assumes it | drift checks | retrain on representative data |
| **Reasonable class balance** | impurity handles skew poorly | majority dominates leaves | class counts | `class_weight` / resample |

> Trees make **few functional assumptions** (flexible), but the one real structural limit is **axis-aligned splits** — they partition along feature axes, which is awkward for rotated/diagonal class regions.

---

## 21. Data Requirements

```text
Target      → categorical (classification); trees also do regression
Features    → numeric simplest (sklearn); encode categorical; CatBoost handles natively
Missing     → sklearn DT: no native handling → impute; CatBoost oh
Outliers    → moderate robustness (extremes end up isolated in leaves)
Scaling     → NOT required (thresholds are in the feature's units)
Feature engineering → minimal; trees auto-select features at splits
Size        → needs enough data per leaf; flexible to size
High-dim    → wasteful on very sparse text; linear/SVM better there
Class imbalance → can bias to majority; use class_weight or resample
```

---

## 22. Evaluation

Classification metrics again — plus one tree-specific one:

| Metric | Formula / Notes | Simple | Use | Avoid |
|---|---|---|---|---|
| Accuracy | (TP+TN)/total | % correct | balanced | imbalanced |
| Precision | TP/(TP+FP) | of predicted Yes how many right | FP costly | when FN worse |
| Recall | TP/(TP+FN) | of actual Yes how many caught | FN costly | when FP worse |
| F1 | 2·P·R/(P+R) | balance | imbalanced | need one alone |
| ROC-AUC | area under ROC | ranking | comparing | need calibrated probs |
| Confusion matrix | TP/TN/FP/FN | error structure | diagnosis | single num needed |
| **Feature importance** | impurity decrease × samples | which features matter | insight, selection | as causal proof |

**Loss ≠ Metric:**

```text
THE TREE IS TRAINED BY MAXIMIZING INFORMATION GAIN (minimizing impurity),
NOT by maximizing accuracy/F1/AUC.
Greedy impurity reduction ≠ maximizing precision, recall, or AUC.
Report the classification metrics separately — they're evaluation, not the
training objective.
```

---

## 23. Failure Cases

```text
DATA            → many features, few samples → spurious splits (overfit)
MATHEMATICAL    → XOR / diagonal classes → needs many jagged axis-aligned splits
OPTIMIZATION    → greedy local splits → globally sub-optimal tree (NP-hard in general)
GENERALIZATION  → unpruned → high variance, train≫test
PRACTICAL       → splits on an ID column → memorization, useless model
```

---

## 24. Debugging

Model underperforming? Checklist:

```text
1. Train≈100%, test poor?           → overfit → max_depth / min_samples_leaf / ccp_alpha
2. Problem is diagonal/XOR-shaped?  → few splits can't do it → use an oblique/kernel/forest
3. One feature is a unique ID?      → tree memorizes it → drop the column
4. Accuracy high but minority class missed? → imbalance → class_weight='balanced'
5. Test ~ train but both low?       → underfit → increase depth / features
6. Feature importance looks odd?    → correlated features split; use permutation importance
```

---

## 25. Compare

Conceptual difference **first**:

```text
Decision Tree:   "learn if-then questions, one at a time"
Logistic Reg.:   "learn one weighted boundary + probability"
Naive Bayes:     "combine independent probability votes"
KNN:             "ask the nearest neighbours"
Random Forest:   "many trees average away each other's mistakes"
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Decision Tree | greedy impurity splits | readable rules, no scaling | overfits, high variance | interpretability, rules |
| Logistic Regression | linear + sigmoid | calibrated probs, baseline | linear only | risk scoring |
| Naive Bayes | Bayes + independence | tiny data, text | naive assumption | spam/text |
| KNN | neighbour vote | no training | slow, curse of dim | small data |
| Random Forest | many deep trees | low variance, robust | less interpretable | default strong classifier |

> The tree's superpower is **human-readable rules**; its weakness is **variance**. Forests and boosting exist precisely to fix that weakness.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  auto-router flags support tickets as "urgent" or "normal",
                   where the decision must be explainable to the team
DATA:              20K tickets + labels
FEATURES:          response days, customer tier, complaint keywords (encoded)
TARGET:            urgent? 1/0
MODEL:             DecisionTreeClassifier(criterion='gini', max_depth=5)
TRAIN:             split → (no scaling!) → fit
EVALUATE:          F1 + confusion matrix + print the rules for review
DEPLOY:            export rules (JSON/export_text) for an auditable, fast service
MONITOR:           re-examine rules as data drifts; retune depth
```

> Same skeleton powers medical triage, fraud rule-cards, and regulatory credit decisions.

---

## 27. Practice

8 levels:

1. **Recall:** what is information gain?
2. **Understand:** why are decision boundaries axis-aligned?
3. **Calculate:** compute Gini and entropy for the node [4 A, 2 B]; then IG for a pure split.
4. **Apply:** given a scatter, sketch a reasonable decision tree boundary.
5. **Debug:** 100% train, 62% test — diagnose & give two fixes.
6. **Experiment:** run Experiment B at depth 1, 3, 5, None; graph the train/test gap.
7. **Build:** loan mini-project — fit depth-tuned tree, `export_text` the rules, present them as a business document.
8. **Explain:** explain a decision tree to a friend in 60 seconds using the loan story.

---

## 28. Interview

### Beginner
- **How does a tree choose a split?** For each feature, try thresholds (midpoints between values), compute information gain (impurity reduction), pick the max, split, recurse.
- **What is Gini impurity?** `1 − Σp²`, the chance of misclassifying a random sample in the node given its distribution. 0 = pure.
- **What is information gain?** `IG = I(parent) − Σ(n_k/n)I(child_k)`. Higher = better split.
- **When does it stop growing?** Pure node, depth/min-sample limits, or no positive-IG split.

### Intermediate
- **Why does an unpruned tree overfit?** Greedy recursive growth fits training noise; variance is high because small data changes rebuild the whole structure.
- **How to control overfitting?** Pre-pruning (`max_depth`, `min_samples_leaf`, `max_features`) and post-pruning (`ccp_alpha`), selected by cross-validation.
- **Gini vs entropy?** Nearly identical trees; Gini faster, entropy has a nice information-theoretic meaning. sklearn defaults to gini.
- **Why no feature scaling?** Splits compare one feature to a threshold; rescaling just rescales thresholds. No distance/gradient.
- **What's a stump?** A depth-1 tree. High bias; used as the weak learner in AdaBoost.

### Advanced
- **Why feature subsampling (`max_features`)?** Decorrelates trees in a forest, slashing ensemble variance — the key to Random Forest.
- **What's cost-complexity pruning?** Minimize `R(T) + α|T|`; α penalizes leaf count; cross-validate α's pruning path.
- **Why are single trees high-variance?** A small perturbation near the root can redirect entire subtrees — amplified along the tree.
- **Why are trees easy to overfit and forests not?** Averaging many decorrelated trees cancels variance; a single tree has nowhere to hide.
- **Axis-aligned vs oblique?** Trees split one feature against a threshold (axis-parallel). Oblique trees use linear combinations — fewer nodes for rotated boundaries, less interpretable.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Entropy:   H = −Σⱼ pⱼ·log₂(pⱼ)
Gini:      G = 1 − Σⱼ pⱼ²
IG:        IG = I(parent) − Σ_k (n_k/n)·I(child_k)
Gain ratio (C4.5): GR = IG / intrinsic_information   (fixes multi-valued-feature bias)
Cost-prune: Cost = R(T) + α·|T|
```

**Common traps:**
- Computing IG but forgetting the **child-weighting** `n_k/n`.
- Assuming trees produce linear boundaries (they're piecewise-constant).
- Believing "more depth is always better" (overfitting).
- Confusing **Gini impurity** with the **Gini coefficient** (inequality).
- Forgetting ID3 uses entropy/multi-way, C4.5 uses gain ratio, CART uses Gini/binary.

> **Representative pattern question (NOT a past GATE PYQ):** "A node has 8 samples, 5 class-A and 3 class-B. A split sends them to pure children (4,0) and (1,3). Compute the entropy-based information gain." → H(parent) = −(5/8·log₂5/8 + 3/8·log₂3/8) ≈ 0.954; children: H(4,0)=0, H(1,3)=−(1/4·log₂1/4 + 3/4·log₂3/4) ≈ 0.811; I_split = (4/8)(0)+(4/8)(0.811)=0.406; **IG ≈ 0.954 − 0.406 = 0.549**.

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open impurity theory, greedy search & pruning</summary>

### Why Gini and entropy, formally

We need a node impurity measure I(p) with:
- I(0) = I(1) = 0 (pure → no uncertainty),
- maximized at equal class proportions (max uncertainty).

Both `1 − Σp²` and `−Σp·log₂p` satisfy this. Entropy is literally the expected number of bits to encode the class of a random draw — so information gain in bits is `H(parent) − H(children)`.

### Greedy search and its cost

At each node, the tree:
1. sorts each feature's values,
2. generates thresholds = midpoints between adjacent distinct values,
3. scores each split, keeps the max-IG one, recurses.

Finding the **globally optimal** tree is NP-hard, so greedy is a practical approximation — and why it overfits and is unstable. Sorting dominates training cost: **O(n·log n · d)** typical per node.

### Why trees are axis-aligned

Each split tests a single feature against a threshold → the boundary at every step is perpendicular to one feature axis. Diagonal class regions need many small axis-aligned steps (jagged), or an oblique method.

### Pre-pruning vs post-pruning

- **Pre-pruning:** stop growing early (`max_depth`, `min_samples_leaf`, `max_features`).
- **Post-pruning (cost-complexity, CART):** grow fully, then prune subtrees where removing them reduces `R(T) + α·|T|`. sklearn exposes this via `ccp_alpha`; pick α by cross-validation along the pruning path.

### Bias–variance for trees

```text
deep trees  →  low bias, high variance
shallow     →  high bias, low variance
single tree →  sits at the high-variance end
→ this is WHY bagging (Random Forest) and boosting were invented:
  they average / combine many trees to slash variance (and often bias).
```

</details>

---

## 31. Teach Back

Try all four.

> **Explain in 30 seconds:** "A decision tree asks a sequence of if-then questions. At each step it picks the question that separates the classes best — the biggest drop in impurity — and stops when every branch is a single answer."

> **Explain to a 12-year-old:** "It's twenty questions. You keep asking the question that splits people into the most clear-cut groups, until every person left belongs to one group — then you've got a rule."

> **Explain in an interview:** add: Gini/entropy, information gain, greedy recursion, axis-aligned limits, overfitting, pre/post-pruning, cost-complexity, why forests fix variance.

> **Explain the mathematics:** derive Gini/entropy, weighted child impurity, and information gain from a worked example.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define a decision tree.
2. Explain the intuition with the loan story.
3. Write and interpret the Gini, entropy, and IG formulas.
4. Compute IG by hand for a candidate split.
5. Explain greedy recursive splitting and why it overfits.
6. Explain pre-pruning vs post-pruning.
7. Explain why trees need no feature scaling.
8. Discuss axis-aligned limits and high variance.
9. Choose it for a real problem; defend — or choose a forest.
10. State one counter-example where you WOULDN'T use a single tree.

---

## 33. Cheat Sheet

```text
Algorithm : Decision Tree (CART) · Supervised → Classification · Non-parametric
Goal      : partition feature space into pure regions with if-then rules
Core      : IG = I(parent) − Σ(n_k/n)·I(child_k),  pick max at each node
Impurity  : Gini 1−Σp²  ·  Entropy −Σp·log₂p  (0 = pure)
Learn     : tree structure: features, thresholds, leaf labels
Tune      : max_depth · min_samples_split/leaf · max_features · criterion · ccp_alpha
Scaling   : NOT required
Fails     : unpruned overfits · axis-aligned only · high variance
Use when  : interpretability/rules, non-linear, no scaling, ensemble building block
Avoid when: max accuracy alone, diagonal boundaries, extreme sparse high-dim
Prune     : pre- (depth/leaf) or post- (ccp_alpha)
Related   : Random Forest · Extra Trees · Gradient Boosting · XGBoost · CatBoost
```

---

## 34. What Next?

You've now met all four classification families — a learned boundary, a lazy voter, a probability combiner, and a rule learner.

```text
Decision Tree (Classification)
   ├── Random Forest   (bag many trees → lower variance)
   ├── Gradient Boosting / XGBoost / LightGBM / CatBoost  (combine weak trees)
   └── (trees also do regression → Decision Tree Regressor)
```

> Next recommended: after this unit, jumping to **Random Forest and the ensemble models** is the natural step — they exist specifically to fix the single tree's overfitting and variance.
