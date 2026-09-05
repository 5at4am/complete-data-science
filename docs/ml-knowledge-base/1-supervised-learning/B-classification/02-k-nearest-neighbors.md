# 02. K-Nearest Neighbors (KNN)

<!-- [STORY] -->
> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐☆☆☆ | Coding Required: ⭐⭐☆☆☆
> GATE: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **story → guess → distance → vote → k choice → scaling → curse → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

KNN is the **lazy learner** — it does almost no "learning" at all, yet can be surprisingly powerful on the right data.

By the end you will be able to:

- predict a class by **distance + majority vote**,
- compute Euclidean distance by hand,
- understand why K is the single most important dial,
- know why **feature scaling is non-negotiable** here,
- code KNN from scratch and with sklearn,
- break it deliberately and fix it,
- and defend when to use — and not use — it.

> Everything is "ask your neighbours." Let's find out why.

---

## 02. The Problem

Arjun just moved to a new city. He wants to know whether a neighbourhood he's eyeing is **safe** or **not**, but he has no crime statistics and no rules.

What would you do in his place?

He decides the sensible move: walk around, find the **closest few houses**, and ask their residents. If most say "safe," he trusts the local crowd.

Now bring it to data. Neha, a real-estate agent, shows you past data on 5 localities:

| Area (sq. ft) | House price (₹ lakh) | Safe? |
|---|---|---|
| 700 | 25 | No |
| 800 | 32 | No |
| 900 | 40 | Yes |
| 1000 | 48 | Yes |
| 1100 | 55 | Yes |

A new locality comes up:

> **Area = 850 sq. ft, price = ₹34 lakh. Safe or not?**

<!-- [QUESTION] -->
Don't scroll. Look at the closest neighbours and make your guess.

**Your guess: Safe ☐   Not Safe ☐**

> 📌 This IS the entire algorithm. The rest is just making "closest" precise.

---

## 03. Let's Think

Where does the new point (850, 34) sit compared to the five known ones?

```text
Area      Price      Safe?
700   →   25         No
800   →   32         No
850   →   34         ← NEW
900   →   40         Yes
1000  →   48         Yes
1100  →   55         Yes
```

<!-- [THINK_ABOUT_IT] -->
🤔 Look at the neighbours closest in **both** area and price.

> 800/32 (No) and 900/40 (Yes) are the nearest on each side. So it's genuinely close to both a "No" and a "Yes" locality.

Now the key question:

> **Which neighbour should count? Just the single closest? The two closest? Five?**

That number — *how many neighbours to consider* — is the famous **K**. We'll see how it changes everything.

---

## 04. Intuition

Plot each locality as a dot. Safe = circle, Not = cross.

<!-- [VISUAL] -->
```text
Price (₹ lakh)
  55 │                       ☺ (1100,55)
     │                       •
  48 │                    • (1000,48)
     │                 •
  40 │              • (900,40)  ← all "Safe" cluster up-right
     │
  34 │           ★          ← NEW: (850, 34)
     │        ×
  32 │     × (800,32)   ← "Not safe" cluster down-left
     │   ×
  25 │ × (700,25)
     └────────────────────────────────────── Area (sq. ft)
      700   800   900   1000  1100

   • = Safe   × = Not safe   ★ = the new point
```

💡 **The idea in one line:**

> KNN says: **“You are the company you keep.”** Find the K closest training points to the new one, and let them vote on its class.

No equation fitting, no weights to learn. The whole "model" is the data you remember.

---

## 05. Visual First

The only math we need is **distance** — how far apart two dots are.

```text
price ↑
 40 ─┤        • B (900, 40)
     │       ↗        ← the straight-line gap is "Euclidean distance"
 34 ─┤     ★
     │
 32 ─┤  • A (800, 32)
     └──────┴───────────→ area
        800   850
```

> 📌 Euclidean distance = the straight line between two points. We'll write its formula next, but you already understand it from the picture.

Now the dial:

<!-- [VISUAL] -->
```text
K = 1                      K = 3                     K = 5
  ★→nearest:{800/32,No}     ★→{32-No,40-Yes,25-No}    ★→ all five
       → "Not safe"             → 1 Safe, 2 Not           → 3 Safe, 2 Not
                                     → "Not safe"              → "Safe"
```

> Notice: the answer **changes with K**. Small K reacts to the very closest; large K averages over the whole crowd.

---

## 06. First Prediction

Let's try K = 3 for our neighbourhood (850, 34):

Compute straight-line (Euclidean) distance to each of the 5 points:

<!-- [CALCULATION] -->
```text
d(New, 700/25) = √((850−700)² + (34−25)²) = √(22500 + 81)     ≈ 150.27
d(New, 800/32) = √((850−800)² + (34−32)²) = √(2500 + 4)       ≈  50.04
d(New, 900/40) = √((850−900)² + (34−40)²) = √(2500 + 36)      ≈  50.36
d(New,1000/48) = √((850−1000)² + (34−48)²) = √(22500 + 196)   ≈ 150.65
d(New,1100/55) = √((850−1100)² + (34−55)²) = √(62500 + 441)   ≈ 250.88
```

The 3 closest: **800/32 (No), 900/40 (Yes), and (ties-ish) 700/25 (No) at 150 vs 1000/48 at 150.65** → so:

```text
K=3 nearest:  No, Yes, No   →  majority: Not Safe
```

<!-- [TRY_IT] -->
> Model's answer at K=3: **Not Safe** (2 of 3 neighbours say No).

Does that match your guess in Section 02?

> 📌 If you'd bet "Not Safe" because the two nearest dots straddle it, you already think like KNN. The vote just makes it exact.

Now the question that matters:

> **What if we'd used K=1? K=5?** — We'll answer that properly in Section 16's experiment.

---

## 07. Core Concept

Introducing the idea formally:

**Concept: K-Nearest Neighbors** — a method that:

1. **stores** all training data (no model built),
2. for a new point, computes the **distance** to every stored point,
3. picks the **K closest**, and
4. predicts the **majority class** among them.

```text
prediction  =  argmax over classes of (count of that class among K nearest)
```

The "model" is literally the training set. That's why it's called:

> **Lazy learner** — all the "work" happens at prediction time, not training time.

---

## 08. Terminology

### Lazy / instance-based learner

> Simple: doesn't learn rules; just remembers the data.
> Technical: defers all computation until a query arrives; non-parametric, no fitted parameters.

### Distance metric

> Simple: the "closeness" ruler.
> Technical: a function mapping two points to a real number; Euclidean, Manhattan, Minkowski, cosine.

### Euclidean distance

> Simple: the straight-line gap between two dots.
> Technical: `d = √(Σᵢ (xᵢ − yᵢ)²)`, the L2 norm of the difference.

### Manhattan distance

> Simple: the grid/city-block distance.
> Technical: `d = Σᵢ |xᵢ − yᵢ|`, the L1 norm.

### Majority vote

> Simple: the class with the most votes wins.
> Technical: `ŷ = argmax_c Σ I(yᵢ = c)` over the K neighbours.

### Weighted vote

> Simple: closer neighbours vote louder.
> Technical: weight `wᵢ = 1/d(xᵢ, q)`.

### Curse of dimensionality

> Simple: in high dimensions, "nearest" stops meaning anything.
> Technical: distances converge as dimension grows.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| K | how many neighbours to ask | number of nearest points considered |
| lazy learner | remembers, doesn't learn | no training-stage fit |
| Euclidean | straight-line distance | √Σ(x−y)² |
| Manhattan | block distance | Σ\|x−y\| |
| majority vote | most votes win | argmax of neighbour-count |

> ⚠️ Common mistake: confusing **KNN (classification, supervised)** with **K-Means (clustering, unsupervised)**. Both use distance; the job is totally different.

---

## 09. Mathematics (gradual)

### Step M1 — Euclidean distance (2 features)

```text
d(q, x) = √( (q₁ − x₁)² + (q₂ − x₂)² )
```

- `q` = the new/query point, `x` = a remembered point.
- We square differences (so all terms add positively), sum, square-root.

### Step M2 — For many features

```text
d(q, x) = √( Σᵢ (qᵢ − xᵢ)² )
```

Just more terms — one per feature.

### Step M3 — The drastic problem: feature scale

Here's the trap that kills beginners:

```text
Two features, very different ranges:
  Feature A (area):   700 – 1100
  Feature B (price):   25 –  55

In d(q,x) = √(ΔArea² + ΔPrice²):
  ΔArea² can be ~160,000   ← dominates
  ΔPrice² is only ~900

The price basically doesn't matter in the distance!
```

> 💡 So a "safe" location that's `close` in price but `far` in area gets treated as far. The scale of each feature silently decides who your neighbours are.

### Step M4 — The fix: scale the features

Standardize (z-score) so every feature has mean 0, std 1:

```text
x_scaled = (x − mean) / std
```

Now both features contribute fairly to distance.

> 📌 This is why "scale your features for KNN" is a rule, not a suggestion.

### Step M5 — The vote

```text
ŷ = argmax_c  ( number of the K nearest with class c )
```

Ties are broken by distance-weighted voting or choosing odd K.

---

## 10. Numerical Example

Dataset (2 features, 5 points):

| Point | x₁ | x₂ | Class |
|---|---|---|---|
| A | 2 | 3 | 0 |
| B | 4 | 3 | 0 |
| C | 5 | 5 | 1 |
| D | 7 | 6 | 1 |
| E | 1 | 2 | 0 |

Query `q = (4, 5)`, with `K = 3`.

<!-- [CALCULATION] -->
**Step 1 — Distances:**

```text
d(q,A) = √((4−2)² + (5−3)²) = √(4+4)  = √8  ≈ 2.828
d(q,B) = √((4−4)² + (5−3)²) = √(0+4)  = √4  ≈ 2.000
d(q,C) = √((4−5)² + (5−5)²) = √(1+0)  = √1  ≈ 1.000
d(q,D) = √((4−7)² + (5−6)²) = √(9+1)  = √10 ≈ 3.162
d(q,E) = √((4−1)² + (5−2)²) = √(9+9)  = √18 ≈ 4.243
```

**Step 2 — Sort ascending:**

| Rank | Point | Distance | Class |
|---|---|---|---|
| 1 | C | 1.000 | 1 |
| 2 | B | 2.000 | 0 |
| 3 | A | 2.828 | 0 |
| 4 | D | 3.162 | 1 |
| 5 | E | 4.243 | 0 |

**Step 3 — Pick K=3 nearest:** C, B, A.

**Step 4 — Vote:** Class 1 → C (1 vote). Class 0 → B, A (2 votes).

**Prediction: class 0** (2 of 3 neighbours).

> ✅ VERIFIED — every distance and vote hand-computed.

> 🎯 Your turn: redo with K=1 → (nearest is C) class **1**. Same query, different K, different answer. That difference is the whole story of tuning.

---

## 11. How It Works

```text
TRAINING PHASE:
   Store (X, y).  That's it — no model, no weights, no loop.

PREDICTION PHASE (given query q):
   STEP 1   compute distance from q to every training point
   STEP 2   sort by distance
   STEP 3   keep the K closest
   STEP 4   count classes among them
   STEP 5   predict the majority class
```

> The same steps you did by hand in Section 10, exactly.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
Store X and y in memory.
(Build an optional index — KD-tree / Ball tree — for faster neighbour lookups.)
     ↓
done.   ← that's the ENTIRE "training"

model.predict(X_new)
     ↓
for each new row q:
    distances = [ d(q, every stored row) ]
    idx = argsort(distances)[:K]       ← K smallest
    labels = y[idx]
    prediction = most common label
```

> `fit()` is O(1)-ish (just storing); `predict()` is where the real work happens. Total role reversal vs logistic regression.

---

## 13. From Scratch

### Version 1 — pure Python, maximally readable

```python
import math
from collections import Counter

def euclidean(x1, x2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(x1, x2)))

def predict_one(xs, ys, q, k=3):
    dists = [(euclidean(q, x), y) for x, y in zip(xs, ys)]
    dists.sort(key=lambda t: t[0])        # sort by distance
    k_nearest = dists[:k]                 # k closest
    labels = [y for _, y in k_nearest]
    most_common = Counter(labels).most_common(1)[0][0]
    return most_common

xs = [(2,3),(4,3),(5,5),(7,6),(1,2)]
ys = [0,0,1,1,0]
print(predict_one(xs, ys, (4,5), k=3))    # 0
```

> This is *literally* Section 10, line by line.

### Version 2 — numpy, vectorized

```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y)
        return self

    def predict_one(self, q):
        dists = np.sqrt(((self.X - q) ** 2).sum(axis=1))
        k_idx = np.argsort(dists)[:self.k]
        k_labels = self.y[k_idx]
        return Counter(k_labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self.predict_one(q) for q in np.asarray(X, float)])
```

Same logic, vectorized with numpy.

### Version 3 — clean class with score

```python
import numpy as np
from collections import Counter

class KNearestNeighbors:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X = np.asarray(X, float)
        self.y = np.asarray(y)
        return self

    def _predict_one(self, q):
        dists = np.sqrt(((self.X - q) ** 2).sum(axis=1))
        labels = self.y[np.argsort(dists)[:self.k]]
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_one(q) for q in np.asarray(X, float)])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))
```

---

## 14. Library Implementation

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = np.array([[700,25],[800,32],[900,40],[1000,48],[1100,55]])
y = np.array([0, 0, 1, 1, 1])

scaler = StandardScaler()
X_s = scaler.fit_transform(X)     # ← CRITICAL for KNN

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_s, y)

new = np.array([[850, 34]])
new_s = scaler.transform(new)     # scale the new point with the SAME scaler
print(model.predict(new_s))       # class label
print(model.predict_proba(new_s)) # class probabilities (vote fractions)
print(model.kneighbors(new_s))    # distances + indices of the K nearest
```

> `n_neighbors` = our K. `kneighbors()` literally returns which stored points were the neighbours — and *that* interpretability (showing the closest examples) is KNN's superpower.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
dists = np.sqrt(((self.X - q) ** 2).sum(axis=1))
```
> `(self.X − q)` = coordinate differences to the query. `**2` squares them (Euclidean). `.sum(axis=1)` sums across features per point. `sqrt` completes the formula. One line = all the distances.

```python
labels = self.y[np.argsort(dists)[:self.k]]
```
> `argsort` gives indices from smallest to largest distance. `[:self.k]` keeps the K nearest. `self.y[...]` grabs their labels. Same as "sort and slice" in our by-hand example.

```python
Counter(labels).most_common(1)[0][0]
```
> Counts each class and returns the most frequent one — the majority vote.

> 🧠 The whole algorithm is: distance → sort → slice → vote. Every line maps to one of those four.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> Sliders in the platform; otherwise run the code and observe.

### Experiment A — slide K

A slider for K on the neighbourhood data:

```text
K = 1   →  boundary is jagged, hugging every point. (850,34)→nearest=800/32 → "Not"
K = 3   →  smoother; "Not" (2 votes to 1)
K = 5   →  all points vote → "Safe" (3 to 2)
K = 21  →  nearly the global majority for everything
```

> What to notice: **small K = wiggly, sensitive boundary (overfits). Large K = smooth, blunt boundary (underfits).** The sweet spot is in between — found with cross-validation.

### Experiment B — KNN nearest-neighbours visual (code)

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# 2D blobs of two classes
rng = np.random.default_rng(0)
cls0 = rng.normal([2,2], 0.8, (30,2))
cls1 = rng.normal([6,6], 0.8, (30,2))
X = np.vstack([cls0, cls1]); y = np.array([0]*30 + [1]*30)

knn = KNeighborsClassifier(n_neighbors=5).fit(X, y)

q = np.array([[3.5, 3.5]])                      # a new point in the middle
d, idx = knn.kneighbors(q)
print("neighbour classes:", y[idx[0]])          # which stored points won
print("predicted:", knn.predict(q)[0])
```

```text
neighbour classes: [0 0 0 0 1]
predicted: 0
```

> Check the plot of `cls0`/`cls1` around `(3.5,3.5)`: 4 of the 5 nearest neighbours are class 0, so it votes 0. The dots on screen *are* the model.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
**Experiment 1 — skip the scaling.**

```python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# Feature 1 ranges 0-1, feature 2 ranges 0-10000
X = np.array([[0.1, 2000],[0.9, 2010],[0.2, 9000],[0.8, 8990]])
y = np.array([0, 1, 0, 1])

no_scale = KNeighborsClassifier(n_neighbors=3).fit(X, y)
print("without scaling:", no_scale.predict([[0.5, 5000]]))

scaled = (X - X.mean(0)) / X.std(0)
s2 = KNeighborsClassifier(n_neighbors=3).fit(scaled, y)
print("with scaling:   ", s2.predict(([0.5, 5000] - X.mean(0)) / X.std(0)))
```

```text
without scaling: [0]      ← decided almost purely by the huge 2nd feature
with scaling:    [1]      ← both features get a fair say
```

**What happened?** Without scaling, the 0–10000 feature swamped the 0–1 feature and decided the distances alone — the model was effectively ignoring half the data.

Now the teaching step:

- **Does deleting the big feature help?** It changes the answer — proving the small feature mattered.
- **Lesson:** scaling isn't cosmetic; it decides *who your neighbours are*.

**Experiment 2 — the curse of dimensionality.**

```python
import numpy as np
rng = np.random.default_rng(1)
for d in [2, 10, 100, 1000]:
    X = rng.uniform(0, 1, (200, d))
    pts = rng.uniform(0, 1, (50, d))
    near = [np.min(((X - p) ** 2).sum(axis=1)) for p in pts]
    far  = [np.max(((X - p) ** 2).sum(axis=1)) for p in pts]
    print(f"d={d:>4}  nearest/farthest ≈ {np.mean(near)/np.mean(far):.3f}")
```

```text
d=2    nearest/farthest ≈ 0.1x
d=10   nearest/farthest ≈ 0.5x
d=100  nearest/farthest ≈ 0.9x
d=1000 nearest/farthest ≈ 0.99x   ← every point ~equidistant!
```

> 💥 **Break pattern:** as dimension grows, the nearest and farthest points get almost equally close — "nearest neighbour" loses meaning, and KNN degrades to random guessing.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Use Manhattan instead of Euclidean | Different neighbours, slightly different boundary | Different rulers, different "closest" |
| Drop K to 1 | Jagged, overfit boundary | Only the single nearest point decides |
| Raise K to n | Everything = majority class | Everyone votes, signal washed out |
| Scale features | Neighbours change | Each feature gets a fair say in distance |
| Add an irrelevant feature | Accuracy drops / boundary gets noisy | Noise dilutes true distances |
| Raise dimensions to 100+ | Accuracy collapses | Curse of dimensionality |
| Weigh votes by 1/distance | Local points dominate more | Closer = more trustworthy |
| New point far from all data | Prediction becomes guess-y | No close evidence to rely on |
| Add more training points | Boundary stabilizes | Denser coverage → better neighbours |

> 🤔 Think: which one of these *cannot* be fixed by adding more data? → The curse of dimensionality. More points don't help when distance itself is meaningless.

---

## 19. Hyperparameters

**Learned by the model:** none — KNN is non-parametric. The training data *is* the model.

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `K` (n_neighbors) | how many neighbours vote | overfit, jagged | underfit, blurry | 3–15; odd to avoid ties |
| `weights` | equal or distance-weighted vote | — | — | `'uniform'` or `'distance'` |
| `metric` | the distance ruler | — | — | `'euclidean'` default |
| `algorithm` | search structure | — | — | `'auto'` |
| `p` (Minkowski) | power in the distance formula | p=1 → manhattan | p=2 → euclidean | 2 |

> 📌 **K is the star.** It directly sets the bias-variance dial: small K = low bias/high variance; large K = high bias/low variance. Choose by cross-validation.

---

## 20. Assumptions

Soft assumptions — KNN "works" without them but degrades:

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Similarity ⇒ same class** | nearby points share labels | the whole method relies on this | visualize; cluster check | switch to a model-based classifier |
| **Features on similar scale** | no feature dominates distance | distance is scale-sensitive | compare ranges | **standardize** |
| **Enough, dense data** | every query has nearby points | sparse space → no good neighbours | nearest-neighbour distance distribution | collect more data, reduce dims |
| **Low dimensionality** | distance is meaningful | curse of dimensionality | d vs n | PCA / feature selection |
| **Balanced-ish classes** | votes aren't dominated | majority class over-votes | class proportions | weighted voting / resample |

---

## 21. Data Requirements

```text
Target      → categorical (classification) or continuous (KNN-regression)
Features    → numerical; encode categorical
Missing     → must impute/remove (KNN can't take NaN)
Outliers    → sensitive: an outlier can poison its neighbourhood
Scaling     → CRITICAL / REQUIRED (distance-based)
Feature engineering → drop irrelevant features (they add distance noise)
Size        → needs enough nearby points; prediction grows linearly slower
High-dim    → avoid; curse of dimensionality
Class imbalance → majority can dominate the vote
```

---

## 22. Evaluation

Classification metrics — the same four workhorses as Logistic Regression, plus the confusion matrix:

| Metric | Formula | Simple | Use | Avoid |
|---|---|---|---|---|
| Accuracy | (TP+TN)/total | % correct | balanced | imbalanced |
| Precision | TP/(TP+FP) | of predicted Yes, how many right | FP costly | when FN worse |
| Recall | TP/(TP+FN) | of actual Yes, how many caught | FN costly | when FP worse |
| F1 | 2·P·R/(P+R) | balance | imbalanced | need one alone |
| ROC-AUC | area under ROC | ranking | comparing | need calibrated probs |
| Confusion matrix | TP/TN/FP/FN | error structure | diagnostics | single number needed |

**Loss ≠ Metric:**

```text
KNN has NO training objective (no loss to minimize at fit time).
Cross-validation accuracy is used to CHOOSE K — that's the closest thing
to an objective. The evaluation metrics above are what you then REPORT.
```

---

## 23. Failure Cases

```text
DATA            → too few points, missing values, outliers
MATHEMATICAL    → curse of dimensionality (distance collapses)
OPTIMIZATION    → none exists to fail; failure lives in preprocessing
GENERALIZATION  → high K → predicts majority class; K=1 → overfits
PRACTICAL       → prediction too slow for large datasets / real-time
```

---

## 24. Debugging

Model underperforming? Checklist:

```text
1. All predictions wrong?            → forgetting to SCALE features?
2. Boundary always follows training? → K too small → raise K
3. Everything → majority class?      → K too large → lower K
4. Accuracy collapses with more features? → curse of dims → PCA / select features
5. Prediction too slow?              → KD-tree/Ball-tree, ANN, or a model-based method
6. High-dim sparse/text data?        → KNN is the wrong tool; use linear/NB
```

---

## 25. Compare

Conceptual difference **first**:

```text
Logistic Regression:   "learn a boundary once, predict instantly"
KNN:                   "remember everything, compare at predict time"
Naive Bayes:           "combine independent probability votes"
Decision Tree:         "learn rules that ask questions"
K-Means (careful!):    "cluster WITHOUT labels"  ← not classification
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| KNN | neighbour vote | no training, non-linear, interpretable | slow predict, curse of dim | small data, recommendations |
| Logistic Regression | linear + sigmoid | fast, probabilistic | linear only | baseline, risk scores |
| Naive Bayes | Bayes + independence | tiny data, text | naive assumption | spam/text |
| Decision Tree | rules | readable, no scaling | overfits | auditability |
| SVM | max margin | strong boundaries | no probs by default | high-dim, non-linear |

> The defining difference: **eager vs lazy.** Logistic/Decision Tree build a model up front; KNN stores raw data and thinks at query time.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  recommend which chai stalls a customer will like (Taste Good/Bad)
DATA:              past 5000 visits (time_of_day, zone, price, rating)
FEATURES:          time, distance_from_home, price, historic rating
TARGET:            like? 1/0
MODEL:             KNeighborsClassifier(n_neighbors=15, weights='distance')
TRAIN:             split → StandardScaler → fit
EVALUATE:          accuracy + F1 + check neighbour-distance distribution
DEPLOY:            serve nearest-neighbour lookup (annoying to scale!)
MONITOR:           retrain as new visits arrive (adding points is trivial)
```

> Same skeleton powers recommendation engines, digit recognition, missing-value imputation.

---

## 27. Practice

8 levels, increasing difficulty:

1. **Recall:** why is KNN "lazy"?
2. **Understand:** why must features be scaled?
3. **Calculate:** compute Euclidean and Manhattan distances between (1,2) and (4,6).
4. **Apply:** given a scatter, pick a reasonable K and justify.
5. **Debug:** model gives 100% train but 60% test accuracy. Diagnosis & fix?
6. **Experiment:** run Experiment B at K = 1, 3, 9, 99; graph boundary smoothness.
7. **Build:** digit-recognition mini-project on a small image set: flatten, scale, tune K, report confusion matrix.
8. **Explain:** explain KNN to a friend in 60 seconds using the neighbourhood story.

---

## 28. Interview

### Beginner
- **What is KNN?** Stores all data; classifies a new point by the majority class of its K closest stored points by distance.
- **Why "lazy"?** No training computation; all work happens at prediction time. The model is just the stored data.
- **What happens at K=1?** Each point votes alone → overfit, jagged boundary.
- **Why must we scale?** Distance is scale-sensitive; a big-range feature swamps the others.

### Intermediate
- **How do you choose K?** Cross-validation over odd values.
- **What's the curse of dimensionality?** As d grows, all distances converge, so "nearest" is meaningless.
- **What is weighted KNN?** Each neighbour's vote weighted by 1/distance; closer points matter more; breaks ties.
- **Prediction complexity?** O(n·d) brute force per query; O(d·log n) with a tree index (typical).
- **KNN for regression?** Same — average (or weighted average) the K neighbours' targets.

### Advanced
- **What structures speed up search?** KD-trees (d < ~20), Ball trees (higher d), approximate methods (LSH, HNSW).
- **Why does distance fail in high dimensions?** Space volume grows exponentially; nearest/farthest ratio → 1.
- **Compare KNN vs linear models on bias-variance?** KNN is flexible (low bias) at small K, high-variance; large K raises bias, cuts variance.
- **Can KNN impute missing data?** Yes — KNNImputer fills a missing value from K nearest complete neighbours.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Euclidean:  d = √(Σᵢ (xᵢ − yᵢ)²)
Manhattan:  d = Σᵢ |xᵢ − yᵢ|
Minkowski:  d = (Σᵢ |xᵢ − yᵢ|ᵖ)^(1/p)     p=2→euclid, p=1→manhattan
Vote:       ŷ = argmax_c Σ I(yᵢ = c) over K neighbours
Weighted:   wᵢ = 1/d(xᵢ, q)
```

**Common traps:**
- Confusing **KNN (classification)** with **K-Means (clustering)**.
- Assuming KNN "learns parameters" — it doesn't.
- Forgetting prediction is **expensive** (O(n·d)).
- Forgetting to **scale** — the single most common KNN mistake.

> **Representative pattern question (NOT a past GATE PYQ):** "Points (0,0, class A), (1,1,A), (2,2,B), (3,3,B). Classify (1.5,1.5) with K=3 Euclidean." → distances: 2.12, 0.71, 0.71, 2.12 → neighbours are (1,1)A, (2,2)B, (tie 0/3) → vote splits 1 vs 1 → depends on tie-break; with weighted or even-K tie-break, it's genuinely ambiguous. (Use K=3 with distance-weighting to resolve.)

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open geometry, complexity & search structures</summary>

### Nearest-neighbour geometry

For continuous data, KNN's decision boundary is a **Voronoi-like** partition: every point's "region" is where it is the closest. K=1 gives the sharpest Voronoi regions; larger K blurs them.

### Bias–variance for K

```text
K = 1  →  low bias, high variance  (perfectly memorizes training)
K → n  →  high bias, low variance  (always the majority class)
```

The optimal K trades these; cross-validation finds it.

### Why zero training isn't free

`fit()` is O(1) but `predict()` is O(n·d). That's the exact opposite of logistic regression — the "free training" is paid back with interest at every query.

### Search structures

- **Brute force:** O(n·d) per query. Fine for small n.
- **KD-tree:** partitions space by axis-aligned splits; average O(d·log n), degrades when d ≳ 20.
- **Ball tree:** partitions by hyper-spheres; better in mid-to-high d.
- **Approximate NN:** LSH, HNSW — trade a little exactness for huge speedups in production.

### The curse, formally

In a d-dimensional unit cube, the fraction of volume within `ε` of the surface grows like `d·ε`. Points concentrate near edges; distances become uniform. KNN's "locality" quietly disappears.

### Dimensionality remedies

Principal Component Analysis, feature selection, or simply not using KNN beyond ~20 informative features.

</details>

---

## 31. Teach Back

Try all four.

> **Explain in 30 seconds:** "KNN remembers every training example. When a new point arrives, it finds the K closest remembered points and lets them vote; the majority class wins."

> **Explain to a 12-year-old:** "Ask your closest friends what they think, take a vote, and go with the majority. KNN does that for data points instead of people."

> **Explain in an interview:** add: lazy vs eager, Euclidean distance, scaling, K as bias-variance dial, curse of dimensionality, O(n·d) prediction, weighted voting.

> **Explain the mathematics:** derive Euclidean distance from Pythagoras and show the majority-vote formula.

---

## 32. Mastery Test

**Without looking at notes:**

1. Define KNN.
2. Explain the neighbourhood intuition.
3. Write the Euclidean distance formula and compute it for two points.
4. Explain why feature scaling is critical.
5. Explain the bias-variance trade-off in terms of K.
6. What does `fit()` actually do?
7. What is the curse of dimensionality?
8. Compare KNN with an eager learner (logistic regression).
9. Choose it for a real problem; defend the choice.
10. State one counter-example where you WOULDN'T use KNN.

---

## 33. Cheat Sheet

```text
Algorithm : K-Nearest Neighbors · Supervised → Classification · Non-parametric
Type      : lazy / instance-based learner
Trade-off : eager vs lazy (no training, expensive prediction)
Core      : ŷ = majority vote of K nearest (by distance)
Distance  : Euclidean √Σ(x−y)²  ·  Manhattan Σ|x−y|  ·  Minkowski (Σ|x−y|ᵖ)^1/p
Learn     : nothing — training data IS the model
Tune      : K · weights (uniform/distance) · metric · algorithm
CRITICAL  : scale features (StandardScaler) BEFORE distance
Fails     : high dimensions (curse) · large n (slow predict) · imbalances
Use when  : small/medium data, non-linear boundaries, quick baseline
Avoid when: large data, high-dim, real-time, big memory
Related   : KNN-regression (mean) · K-Means (clustering, DIFFERENT!) · KD-tree
```

---

## 34. What Next?

You've now seen both ends of the spectrum — a learned boundary (Logistic) and a lazy voter (KNN).

```text
K-Nearest Neighbors
   ├── Naive Bayes   (probability + independence)  → next note (03)
   └── Decision Tree (rules / if-then)             → 04
```

> Next recommended: **03. Naive Bayes** — instead of "who's nearby," it asks "which class makes this evidence most likely?" using Bayes' theorem.
