# 17. CatBoost (Categorical Boosting)

> Difficulty: ⭐⭐⭐⭐⭐ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐☆☆☆ | Interview: ⭐⭐⭐☆☆ | Industry: ⭐⭐⭐⭐☆

---

<!-- [STORY] -->
## 01. Start Here

Flipkart lists 50 million products. Every product has categories — brand, colour, size, material, seller_type, city. A price-prediction model must read these columns as-is. Naive one-hot encoding of `brand` (4,000 values) creates a 4,000-column sparse matrix; naïve target encoding leaks the answer back into the same row. CatBoost solves both problems at once: it encodes every category **leakage-free** using *ordered target statistics* and reduces **prediction shift** via *ordered boosting* — all while growing simple, stable symmetric trees.

We will build this story by hand, step by step.

---

<!-- [QUESTION] -->
## 02. Core Question

**If categories leak their own answer into their encoding, and gradient boosting accumulates biased leaf estimates round after round, how do you build a model that is both category-safe and shift-free?**

---

<!-- [HINT] -->
## 03. Intuition Before Math

You have a column `city` with values Mumbai, Delhi, Chennai, Kolkata.

**Naïve approach:** compute average price for each city and plug it in as a number.
But wait — if Mumbai has only one product priced at ₹999, that ₹999 *is* the average. The model sees the answer for that exact product. That is **leakage**.

**CatBoost fix:** when encoding Mumbai for a particular product, look only at *other* products from Mumbai that appeared *earlier* in a random permutation. The current product's own price is never used. Add a global average (prior) to smooth cases where few earlier samples exist.

**Prediction shift fix:** standard GBM computes residuals using a model already trained on those same rows. Each round's leaf estimates carry a small bias. Over hundreds of rounds, bias accumulates. CatBoost trains each tree using *only earlier samples in a permutation*, making each leaf estimate essentially out-of-fold.

Two simple ideas — zero leakage, zero shift. Everything else (symmetric trees, priors, smoothing) flows from these.

---

<!-- [MAP] -->
## 04. Where It Fits

```text
SUPERVISED LEARNING
└── Regression
    └── Ensembles
        └── Boosting
            ├── Gradient Boosting
            ├── AdaBoost
            ├── XGBoost
            ├── LightGBM
            └── CATBOOST        ← YOU ARE HERE

CatBoost's classification counterpart → 17-catboost.md (B-classification folder)
```

---

<!-- [TABLE] -->
## 05. Key Vocabulary

| Term | Plain English | Formal |
|---|---|---|
| Categorical feature | Column with non-numeric labels | Discrete nominal variable |
| Ordered target statistic (TS) | Encode a category using only *earlier* samples' targets | Permutation-based leakage-free encoding |
| Prediction shift | Model's leaf estimates drift from the true gradient distribution | Accumulated bias in residual estimation |
| Ordered boosting | Train each tree's leaves using only earlier samples in a permutation | Out-of-fold leaf estimation |
| Symmetric (oblivious) tree | Every node at the same depth uses the *same* split | Depth-wise shared-split decision tree |
| Prior (smoothing) | Global average blended into TS to avoid extreme values for rare categories | Regularisation constant in TS formula |
| Permutation | A random reordering of the dataset | Random bijection σ : {1..n} → {1..n} |

---

<!-- [INPUTS] -->
## 06. Input → Process → Output

```text
INPUT                          PROCESS                           OUTPUT
─────────────────────────────────────────────────────────────────────────
X: n × m matrix               1. Encode categoricals via         ŷ = Σ η·fₜ(x)
   (numeric + categorical)        ordered TS (permutation-based)
y: continuous target           2. Compute gradient of loss
                               3. Grow symmetric tree
                                  (same split at every depth)
                               4. Leaf values from earlier-sample gradients
                               5. ŷ̂ += η · tree(x)
                               6. Repeat steps 2-5
```

**Parameters learned:** tree structures, leaf weights, per-category ordered-TS encodings.

**Key hyperparameters:** `iterations`, `learning_rate`, `depth`, `l2_leaf_reg`, `cat_features`, `border_count`, `bagging_temperature`, `random_strength`.

---

<!-- [DEFINITION] -->
## 07. One-Line Definition

**Beginner:** CatBoost is a boosting library that handles categorical columns directly — safely encoding them without leaking the answer — and uses symmetric trees to stay stable and fast.

**Technical:** CatBoost is a gradient-boosting framework that replaces categorical features with ordered target statistics (leakage-free), reduces prediction shift via ordered boosting, and grows symmetric (oblivious) trees for stable, fast inference.

---

<!-- [SCENARIO] -->
## 08. The Problem CatBoost Solves

**The leakage trap:** you have a dataset with 20 categorical columns, some with thousands of unique values. You try target encoding — replace each category with the mean of its targets. But each product's own price feeds into its own encoding. The model memorises. Train accuracy is 99%. Test accuracy is 60%. You have no idea why.

**The prediction shift trap:** you switch to XGBoost with one-hot encoding. Leakage is gone, but the model trains residuals computed from a model already fitted to those same rows. Each round's gradient estimate is slightly biased. After 500 rounds, bias accumulates, and generalisation plateaus early.

**CatBoost attacks both traps simultaneously:** ordered TS eliminates leakage; ordered boosting eliminates shift. You can pass raw categorical columns without encoding, and the model just works.

---

<!-- [TREES] -->
## 09. Taxonomy

```text
Ensemble Methods
├── Bagging
│   └── Random Forest
├── Boosting                          ← CatBoost lives here
│   ├── AdaBoost
│   ├── Gradient Boosting (GBM)
│   ├── XGBoost
│   ├── LightGBM
│   └── CatBoost
└── Stacking
```

---

<!-- [VOCAB] -->
## 10. Terminology Deep Dive

**Ordered Target Statistics (ordered TS):**
For sample *i* in permutation σ, the encoding of its categorical value *c* is:

```
enc(i) = (a · prior + Σ_{j < i in σ} [cⱼ = cᵢ] · yⱼ)
          ─────────────────────────────────────────────────
              a + |{j < i in σ : cⱼ = cᵢ}|
```

- `prior` = global mean of y (default)
- `a` = smoothing strength (default = 1)
- Sample *i*'s own y is **never** used → no leakage

**Prediction shift:**
In standard GBM, leaf value for row *i* is computed from the gradient at *i*, which depends on the current model's prediction at *i*. But the current model was trained using leaf values that also depended on *i*'s gradient. This circular dependency creates a systematic bias called **prediction shift**.

**Ordered boosting:** maintain separate model copies per permutation. For sample *i*, only gradients from samples *j < i* in that permutation are used to compute *i*'s leaf value. The circular dependency is broken.

**Symmetric (oblivious) trees:** at each depth level, every node uses the *same* split condition. A depth-3 tree has exactly 2³ = 8 leaves, and the path to each leaf is a sequence of the same three decisions applied in the same order. Simple, regularised, and very fast at inference.

---

<!-- [RECIPE] -->
## 11. Step-by-Step Algorithm

```text
INPUT: X (with categorical columns), y, iterations T, depth d, learning rate η

STEP 1 — Encode categoricals:
    For each categorical column, compute ordered TS using a random permutation σ
    (run P independent permutations, average the encodings)

STEP 2 — Initialise:
    ŷ = prior  (global mean of y)

STEP 3 — For each boosting round t = 1..T:
    a) Compute gradients gᵢ = ∂L(yᵢ, ŷᵢ) / ∂ŷᵢ  using ordered estimates
    b) Grow a symmetric tree of depth d:
       - At each level, try all (feature, threshold) pairs
       - Pick the ONE split that maximises total gain across ALL current leaves
       - Both child nodes of every existing leaf use the same split
    c) For each leaf, compute increment using only earlier-sample gradients (ordered)
    d) ŷ̂ += η · tree(x)

STEP 4 — Return ŷ̂ = Σ η · fₜ(x)
```

---

<!-- [DEMO] -->
## 12. Toy Example — Ordered TS Encoding

**Data:** Categorical column `colour` and target `price`.

| Row | colour | price |
|-----|--------|-------|
| 1 | Red | ₹500 |
| 2 | Blue | ₹800 |
| 3 | Red | ₹600 |
| 4 | Blue | ₹900 |
| 5 | Red | ₹550 |

Permutation order = row order. `prior` = mean(price) = ₹670. Smoothing `a` = 1.

**Row 1 (Red, ₹500):** No earlier Red rows. enc = (1 × 670 + 0) / (1 + 0) = **₹670**

**Row 2 (Blue, ₹800):** No earlier Blue rows. enc = (1 × 670 + 0) / (1 + 0) = **₹670**

**Row 3 (Red, ₹600):** Earlier Red = Row 1 (₹500). enc = (1 × 670 + 500) / (1 + 1) = 1170 / 2 = **₹585**

**Row 4 (Blue, ₹900):** Earlier Blue = Row 2 (₹800). enc = (1 × 670 + 800) / (1 + 1) = 1470 / 2 = **₹735**

**Row 5 (Red, ₹550):** Earlier Red = Row 1 (₹500), Row 3 (₹600). enc = (1 × 670 + 500 + 600) / (1 + 2) = 1770 / 3 = **₹590**

Notice: Row 3's own price (₹600) never appears in Row 3's encoding. Row 5's encoding uses only Rows 1 and 3. **Zero leakage.**

---

<!-- [NUMERICAL] -->
## 13. Complete Worked Example — One Boosting Round

Continuing from §12, now we add a numeric feature `weight_kg`:

| Row | weight_kg | colour (enc) | price (y) |
|-----|-----------|--------------|-----------|
| 1 | 1 | 670 | 500 |
| 2 | 2 | 670 | 800 |
| 3 | 1.5 | 585 | 600 |
| 4 | 2.5 | 735 | 900 |
| 5 | 1.2 | 590 | 550 |

**Round 1:** Base prediction = prior = ₹670 for all rows.

Residuals (gradient for L2 loss = y − ŷ):
```
r = [500-670, 800-670, 600-670, 900-670, 550-670]
  = [-170, 130, -70, 230, -120]
```

Try split: `weight_kg ≤ 1.5` → Left = {Row 1, 3, 5}, Right = {Row 2, 4}.

Left increment = mean([-170, -70, -120]) = -360/3 = **-120**
Right increment = mean([130, 230]) = 360/2 = **+180**

With learning rate η = 0.3:
- Left rows: 670 + 0.3 × (-120) = 670 − 36 = **634**
- Right rows: 670 + 0.3 × (180) = 670 + 54 = **724**

**After Round 1:** predictions = [634, 724, 634, 724, 634].

Errors: |500−634| + |800−724| + |600−634| + |900−724| + |550−634| = 134 + 76 + 34 + 176 + 84 = **504**

**With ordered boosting** (the key CatBoost twist): when computing the increment for Row 3, we would use only gradients from Rows 1 and 2 (earlier in the permutation), not Row 3's own gradient. This eliminates the circular dependency that causes prediction shift.

Round 2 would then use the new residuals and repeat. Each round chips away at the error.

**VERIFIED** — ordered TS from §12 hand-checked; leaf increments arithmetic confirmed.

---

<!-- [CODE] -->
## 14. From-Scratch Implementation

### Version 1 — Ordered TS + Symmetric Stump

```python
import numpy as np

class CatBoostFromScratch:
    """Minimal CatBoost: ordered TS encoding + symmetric stump boosting."""

    def __init__(self, iterations=20, lr=0.3, depth=1, prior_smoothing=1.0):
        self.iterations = iterations
        self.lr = lr
        self.depth = depth
        self.a = prior_smoothing
        self.trees = []
        self.prior = None

    def _ordered_ts(self, cat_col, y):
        """Encode each category using only earlier samples (leakage-free)."""
        n = len(y)
        enc = np.zeros(n)
        sums, counts = {}, {}
        for i in range(n):
            c = cat_col[i]
            s = sums.get(c, 0.0)
            cnt = counts.get(c, 0)
            enc[i] = (self.prior * self.a + s) / (self.a + cnt)
            sums[c] = s + y[i]
            counts[c] = cnt + 1
        return enc

    def fit(self, X, y, cat_indices=None):
        """
        X: ndarray (n, m) — numeric features
        y: ndarray (n,)   — target
        cat_indices: list of column indices to treat as categorical
        """
        X = X.astype(float).copy()
        y = y.astype(float)
        self.prior = y.mean()

        # Replace categorical columns with ordered TS
        if cat_indices:
            for ci in cat_indices:
                X[:, ci] = self._ordered_ts(X[:, ci].astype(str), y)

        pred = np.full(len(y), self.prior)

        for _ in range(self.iterations):
            residuals = y - pred
            best_gain, best_col, best_thr = -np.inf, None, None

            for col in range(X.shape[1]):
                for thr in np.unique(X[:, col])[1:]:
                    left = X[:, col] <= thr
                    if left.sum() == 0 or (~left).sum() == 0:
                        continue
                    inc_l = residuals[left].mean()
                    inc_r = residuals[~left].mean()
                    gain = -(
                        np.mean((residuals[left] - inc_l) ** 2)
                        + np.mean((residuals[~left] - inc_r) ** 2)
                    )
                    if gain > best_gain:
                        best_gain, best_col, best_thr = gain, col, thr

            if best_col is None:
                inc = residuals.mean()
                tree = {"col": None, "thr": None, "left": inc, "right": inc}
            else:
                left = X[:, best_col] <= best_thr
                tree = {
                    "col": best_col,
                    "thr": best_thr,
                    "left": residuals[left].mean(),
                    "right": residuals[~left].mean(),
                }

            self.trees.append(tree)
            pred += self.lr * self._apply_tree(X, tree)

        return self

    def _apply_tree(self, X, tree):
        if tree["col"] is None:
            return np.full(len(X), tree["left"])
        return np.where(X[:, tree["col"]] <= tree["thr"], tree["left"], tree["right"])

    def predict(self, X, cat_indices=None):
        X = X.astype(float).copy()
        if cat_indices:
            for ci in cat_indices:
                X[:, ci] = self._ordered_ts(X[:, ci].astype(str), np.zeros(len(X)))
        pred = np.full(len(X), self.prior)
        for tree in self.trees:
            pred += self.lr * self._apply_tree(X, tree)
        return pred


# --- Demo ---
rng = np.random.RandomState(42)
n = 200
colour = rng.choice(["Red", "Blue", "Green"], n)
weight = rng.rand(n) * 3 + 0.5
y = np.where(colour == "Red", 500, np.where(colour == "Blue", 800, 650)) + weight * 100 + rng.randn(n) * 50

X = np.column_stack([weight, np.array(colour, dtype=float)])
model = CatBoostFromScratch(iterations=30, lr=0.3)
model.fit(X, y, cat_indices=[1])
preds = model.predict(X, cat_indices=[1])
rmse = np.sqrt(np.mean((y - preds) ** 2))
print(f"RMSE (from scratch): {rmse:.1f}")
```

### Version 2 — Multi-Permutation Ordered Boosting (Conceptual)

```python
import numpy as np

class OrderedBoostingCatBoost:
    """
    Demonstrates the ordered boosting concept:
    For each permutation, maintain a separate model that only uses
    earlier samples for leaf estimation.
    """

    def __init__(self, n_permutations=3, iterations=10, lr=0.3):
        self.P = n_permutations
        self.T = iterations
        self.lr = lr
        self.permutations = []
        self.models = []  # one model per permutation

    def fit(self, X, y):
        n = len(y)
        self.prior = y.mean()

        # Generate permutations
        for p in range(self.P):
            perm = np.random.permutation(n)
            self.permutations.append(perm)

            # For this permutation, maintain prediction array
            # ordered_est[p, i] uses only gradients from perm[:position_of_i]
            model_preds = np.full(n, self.prior)

            for t in range(self.T):
                # Build gradient order: process samples in permutation order
                # When building the tree, leaf value for sample i uses only
                # gradients from samples earlier in this permutation
                grad = -(y - model_preds)  # L2 gradient

                # Simplified: fit a stump using only the gradients of the
                # samples in the first half of the permutation (concept demo)
                half = n // 2
                train_idx = perm[:half]

                # Find best split using only training subset gradients
                best_gain, best_col, best_thr = -np.inf, None, None
                for col in range(X.shape[1]):
                    vals = np.unique(X[train_idx, col])
                    for thr in vals[1:]:
                        mask = X[:, col] <= thr
                        left_t = mask[train_idx]
                        right_t = ~left_t
                        if left_t.sum() == 0 or right_t.sum() == 0:
                            continue
                        inc_l = grad[train_idx][left_t].mean()
                        inc_r = grad[train_idx][right_t].mean()
                        gain = -(
                            np.mean((grad[train_idx][left_t] - inc_l) ** 2)
                            + np.mean((grad[train_idx][right_t] - inc_r) ** 2)
                        )
                        if gain > best_gain:
                            best_gain, best_col, best_thr = gain, col, thr

                if best_col is not None:
                    left_mask = X[:, best_col] <= best_thr
                    inc_l = grad[train_idx][left_mask[train_idx]].mean()
                    inc_r = grad[train_idx][~left_mask[train_idx]].mean()
                    update = np.where(left_mask, inc_l, inc_r)
                    model_preds += self.lr * update

            self.models.append(model_preds)

    def predict(self, X):
        # Average predictions across all permutation-models
        avg = np.mean(self.models, axis=0)
        return avg
```

### Version 3 — Library CatBoost (Production)

```python
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# --- Synthetic e-commerce product data ---
rng = np.random.RandomState(0)
n = 2000
df = pd.DataFrame({
    "weight_kg": rng.rand(n) * 5 + 0.1,
    "brand": rng.choice(["Samsung", "Apple", "OnePlus", "Xiaomi", "Realme"], n),
    "colour": rng.choice(["Black", "White", "Blue", "Red", "Gold"], n),
    "seller_type": rng.choice(["Official", "Reseller", "Refurbished"], n),
    "city": rng.choice(["Mumbai", "Delhi", "Chennai", "Kolkata", "Bangalore"], n),
})

# True relationship (hidden)
brand_premium = {"Apple": 30000, "Samsung": 15000, "OnePlus": 12000, "Xiaomi": 8000, "Realme": 6000}
seller_disc = {"Official": 0, "Reseller": -500, "Refurbished": -2000}
y = (
    df["weight_kg"].values * 2000
    + df["brand"].map(brand_premium).values
    + df["seller_type"].map(seller_disc).values
    + rng.randn(n) * 1000
)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42)

cat_cols = ["brand", "colour", "seller_type", "city"]

model = CatBoostRegressor(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3.0,
    cat_features=cat_cols,
    loss_function="RMSE",
    verbose=100,
    random_seed=42,
)
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

y_pred = model.predict(X_test)
print(f"\nR²:    {r2_score(y_test, y_pred):.4f}")
print(f"RMSE:  {np.sqrt(mean_squared_error(y_test, y_pred)):.1f}")
print(f"Best iteration: {model.get_best_iteration()}")
print(f"Feature importances: {dict(zip(df.columns, model.feature_importances_))}")
```

---

<!-- [EXPLAINER] -->
## 15. Code Walkthrough

### Version 1 — `_ordered_ts`

```
Line: _ordered_ts(self, cat_col, y)
  What:  Walk through rows in order; for each row, compute the TS using
         only earlier rows sharing the same category.
  Why:   Leakage-free — the current row's own target never enters its encoding.
  Math:  enc(i) = (a · prior + Σ_{j<i} yⱼ) / (a + count_{j<i})

Line: self.prior = y.mean()
  What:  Global mean used as the prior.
  Why:   When no earlier rows exist for a category, the prior fills in.
  Math:  prior = (1/n) Σ yᵢ

Line: best_gain, best_col, best_thr  (split search loop)
  What:  For each feature and threshold, compute the gain from splitting.
  Why:   Symmetric tree requires ONE best split at each depth level.
  Math:  gain = -[variance(left) + variance(right)]
```

### Version 3 — Library usage

```
Line: cat_features=cat_cols
  What:  Tell CatBoost which columns are categorical.
  Why:   Without this, CatBoost treats them as numeric (wrong).

Line: eval_set=(X_test, y_test), early_stopping_rounds=50
  What:  Monitor test loss; stop if no improvement for 50 rounds.
  Why:   Prevents overfitting; paired with ordered boosting for double safety.

Line: model.feature_importances_
  What:  How much each feature contributed to splits.
  Why:   Interpretability; verify that brand/seller carry weight.
```

---

<!-- [HYPER] -->
## 16. Hyperparameters That Matter

| Hyperparameter | What it does | Increase → | Decrease → | Typical |
|---|---|---|---|---|
| `iterations` | Number of boosting rounds | Lower bias, higher overfit risk | Higher bias, less overfit | 500–2000 |
| `learning_rate` | Step size per tree | Faster learning, less stable | Slower, more robust | 0.01–0.1 |
| `depth` | Tree depth (symmetric) | More complex interactions | Simpler, more regularised | 4–8 |
| `l2_leaf_reg` | L2 penalty on leaf weights | More shrinkage, less overfit | Less regularisation | 1–10 |
| `border_count` | Number of split thresholds tried | Finer splits, more compute | Coarser, faster | 32–254 |
| `bagging_temperature` | Randomness in row sampling | More diversity, less overfit | Deterministic sampling | 0–1 |
| `random_strength` | Noise added to split gains | More exploration, less overfit | Greedy splitting | 0–10 |
| `cat_features` | Columns to encode via ordered TS | — | — | list of indices |

**Good news:** CatBoost's Bayesian-initialized defaults often work well out of the box — less tuning than XGBoost or LightGBM.

---

<!-- [COMPARE] -->
## 17. How It Compares to Siblings

```text
                    Leakage      Prediction     Tree         Speed on       Default
                    control      shift control  structure    big numeric    quality
─────────────────────────────────────────────────────────────────────────────────────
Gradient Boosting   None         None           Unbalanced   Medium         Low
AdaBoost            Reweighting  None           Unbalanced   Medium         Medium
XGBoost             Regularise   None           Level-wise   Medium         High
LightGBM            Regularise   None           Leaf-wise    Very fast      High
CATBOOST            Ordered TS   Ordered boost  Symmetric    Medium         Very high
```

**The CatBoost edge:** when your data is full of categorical columns (product categories, user segments, city codes), CatBoost's ordered TS avoids leakage that other methods are vulnerable to (unless you carefully implement your own leakage-free target encoding). Its symmetric trees also make inference very fast — a depth-6 oblivious tree has exactly 64 leaves and a fixed 6-comparison path.

---

<!-- [BREAK] -->
## 18. Break the Model — Experiment

**Experiment 1: Demonstrate leakage**

Take the data from §12. Replace ordered TS with naïve target encoding (use the row's own mean):

```
Naïve enc for Row 3 (Red, ₹600):
  mean of ALL Red rows = (500 + 600 + 550) / 3 = ₹550
  → Row 3's own price (₹600) IS included in its encoding
```

Now remove Row 3 from the dataset and re-encode: mean of remaining Red = (500 + 550) / 2 = ₹525. The encoding shifted by ₹25. That shift IS the leakage signal. In ordered TS, removing Row 3 doesn't change its encoding at all (because Row 3's own target was never used).

**Experiment 2: Prediction shift amplification**

Train two identical GBMs for 500 rounds on the same data:
- Model A: standard residual boosting (all samples used for each leaf)
- Model B: ordered boosting (leaf for sample *i* uses only samples before *i*)

Plot train vs test RMSE after each round. Model A's train-test gap widens noticeably after round 100. Model B's gap stays narrow. That gap IS prediction shift.

**Try it yourself:** increase `iterations` to 2000 on both. Model A may even start getting worse on test while improving on train (overfitting amplified by shift). Model B degrades more gracefully.

---

<!-- [WORKED] -->
## 19. Full Numerical Walkthrough — Ordered TS by Hand

**Dataset:** 6 products with categorical `brand` and numeric `price`.

| Row | brand | price |
|-----|-------|-------|
| 1 | A | ₹100 |
| 2 | B | ₹200 |
| 3 | A | ₹150 |
| 4 | C | ₹300 |
| 5 | B | ₹250 |
| 6 | A | ₹180 |

Permutation order = row order. Prior = mean = (100+200+150+300+250+180)/6 = 1180/6 ≈ ₹196.67. Smoothing `a` = 1.

| Row | brand | Earlier same-brand rows | enc = (a·prior + Σy_earlier) / (a + count) |
|-----|-------|------------------------|---------------------------------------------|
| 1 | A | none | (196.67 + 0) / 1 = **₹196.67** |
| 2 | B | none | **₹196.67** |
| 3 | A | Row 1 (₹100) | (196.67 + 100) / 2 = **₹148.33** |
| 4 | C | none | **₹196.67** |
| 5 | B | Row 2 (₹200) | (196.67 + 200) / 2 = **₹198.33** |
| 6 | A | Row 1 (₹100), Row 3 (₹150) | (196.67 + 250) / 3 = **₹148.89** |

**Verification:** Row 6's encoding uses ₹100 and ₹150 (Rows 1 and 3) but NOT its own ₹180. ✓ No leakage.

**Effect of smoothing:** if brand C had appeared only in Row 4, without smoothing (a=0), the formula breaks (0/0). The prior fills this gap, giving ₹196.67 — the global average. As more C rows appear, the prior influence fades and the category's true signal emerges.

---

<!-- [SCENARIOS] -->
## 20. When to Use / When NOT to Use

### ✓ Use CatBoost when:
- Your data has **many categorical features** (especially high-cardinality)
- You want **robust defaults** without extensive tuning
- **Leakage prevention** is critical (finance, healthcare)
- You want **fast inference** (symmetric trees are hardware-friendly)
- You're building **competition ensembles** (diversity with XGBoost/LightGBM)

### ✗ Avoid CatBoost when:
- Data is **purely numeric and very large** (LightGBM is faster)
- You need **images / text / audio** (use deep learning)
- **Extreme memory constraints** (ordered TS permutations cost memory)
- You want **maximum interpretability** (use a single tree or linear model)
- **Tiny datasets** (<100 rows) — ordered TS has too few "earlier" samples

---

<!-- [GATE] -->
## 21. GATE / Exam Perspective

**Key formulas to remember:**

```text
Ordered TS:    enc(i) = (a·prior + Σ_{j<i in σ} [cⱼ=cᵢ]·yⱼ) / (a + |{j<i : cⱼ=cᵢ}|)

Additive model: ŷ̂ = f₀ + Σ η·fₜ(x)

Prediction shift: leaf gradient at i depends on model trained using i → bias accumulates

Symmetric tree: depth d → exactly 2^d leaves, each defined by d same-split decisions
```

**Core concepts:** ordered target statistics, prediction shift, ordered boosting, symmetric/oblivious trees, prior smoothing.

> **Representative pattern question (NOT a past GATE PYQ):**
> "A dataset has a categorical column `city` with 500 unique values. You want to use gradient boosting. Explain why naïve target encoding is dangerous, and describe two mechanisms CatBoost uses to mitigate it."
>
> **Answer sketch:** (1) Naïve target encoding includes the sample's own target in its category's mean → target leakage → overfitting. (2) CatBoost uses **ordered TS** — encode each sample using only earlier samples in a permutation (no self-information). (3) CatBoost uses **ordered boosting** — leaf estimates use only earlier-sample gradients, eliminating prediction shift (the circular dependency between current model and gradient estimation).

**Common traps:**
- Assuming all boosters handle categoricals the same way (they don't)
- Thinking target encoding is safe by default (it leaks unless done carefully)
- Forgetting prediction shift exists (it's subtle but real)
- Assuming CatBoost is always the fastest (LightGBM wins on large numeric data)

---

<!-- [DECISIONS] -->
## 22. Practical Decision Framework

```text
Your data has categorical columns?
├── YES, many/high-cardinality
│   ├── Need fast training → CatBoost (good defaults, minimal tuning)
│   ├── Need fast inference → CatBoost (symmetric trees)
│   └── Competition ensemble → CatBoost + XGBoost + LightGBM (diversity)
├── YES, few/low-cardinality
│   ├── Data is small → XGBoost with one-hot
│   └── Data is large → LightGBM with label encoding
└── NO, all numeric
    ├── Large dataset → LightGBM (fastest)
    ├── Medium dataset → XGBoost (robust)
    └── Small dataset → Any (or simple model)
```

---

<!-- [CHECK] -->
## 23. Sanity Checks

- [ ] Pass `cat_features` parameter — otherwise CatBoost treats categories as numeric (silently wrong)
- [ ] Verify train-test RMSE gap — ordered boosting should make it smaller than XGBoost/LightGBM on the same data
- [ ] Check `model.get_best_iteration()` — if it hits `iterations`, increase the limit or reduce `learning_rate`
- [ ] Inspect `feature_importances_` — categorical columns should appear if they carry signal
- [ ] Compare with XGBoost/LightGBM on the same data — CatBoost should win on categorical-heavy, lose on pure-numeric large data

---

<!-- [CHEAT] -->
## 24. Cheat Sheet

```text
CatBoost
──────────────────────────────────────────────────
WHAT:     Gradient boosting with leakage-free categorical handling
WHEN:     Categorical-heavy tabular data, robust defaults needed
CORE IDEA: Ordered TS (no leakage) + Ordered boosting (no shift) + Symmetric trees

FORMULA:  enc(i) = (a·prior + Σ_{j<i} yⱼ) / (a + count_{j<i})
          ŷ̂ = f₀ + Σ η·fₜ(x)

PYTHON:   from catboost import CatBoostRegressor
          model = CatBoostRegressor(cat_features=[...], iterations=500)
          model.fit(X_train, y_train, eval_set=(X_test, y_test),
                    early_stopping_rounds=50)

PROS:     Handles categoricals natively, fast inference, good defaults
CONS:     Slower training than LightGBM on pure-numeric, more memory
AVOID:    Pure numeric huge data → LightGBM; images/text → deep learning

VARIANT:  Classification → CatBoostClassifier (B-classification folder)
```

---

<!-- [MISTAKES] -->
## 25. Common Mistakes

```text
❌ Forgetting to pass cat_features
   → CatBoost silently treats categories as numeric. Results look okay but are wrong.
   FIX: Always pass cat_features=[list of column names or indices].

❌ One-hot encoding before feeding to CatBoost
   → Defeats the purpose; you lose ordered TS, increase dimensionality.
   FIX: Pass raw categorical columns; let CatBoost handle them.

❌ Not using eval_set / early_stopping
   → Risk of overfitting despite ordered boosting.
   FIX: Always hold out a validation set and use early_stopping_rounds.

❌ Using CatBoost on a 10-million-row purely-numeric dataset
   → LightGBM would be 3-5x faster with similar accuracy.
   FIX: Benchmark both; prefer LightGBM for speed on numeric-only data.

❌ Expecting CatBoost to be interpretable like a single tree
   → It's an ensemble of hundreds of trees.
   FIX: Use feature_importances_ and SHAP for interpretation.
```

---

<!-- [DEEPDIVE] -->
## 26. Deep Dive — The Math of Prediction Shift

In standard gradient boosting, at round *t*, the model prediction is ŷₜ(xᵢ). The gradient (residual) is:

```
gₜ(xᵢ) = -∂L(yᵢ, ŷₜ(xᵢ)) / ∂ŷₜ(xᵢ)
```

The tree fₜ is fit to these gradients. But ŷₜ(xᵢ) was computed using earlier trees that were *also* fit to gradients from the same rows. This creates a circular dependency:

```
f₁ depends on g₁ (which uses ŷ₀ = constant — OK, no shift yet)
f₂ depends on g₂ (which uses ŷ₁ = η·f₁ — f₁ was fit to ALL rows' g₁)
f₃ depends on g₃ (which uses ŷ₂ — all earlier trees trained on all rows)
...
```

Each fₜ's gradient gₜ is biased because the model used to compute gₜ was trained using the same rows' gradients. This bias is **prediction shift** — the gradient distribution at training time differs from what you'd see at test time.

**CatBoost's fix:** For permutation σ, maintain model copies. When computing the leaf value for sample *i* in permutation σ, use only gradients from samples *j* that appear before *i* in σ. Since those earlier samples were trained using even earlier samples, the circular dependency is broken — each leaf estimate is essentially "out-of-fold."

The final prediction averages over P permutations, smoothing out any residual bias.

**Intuition:** Imagine a teacher grading homework. If the teacher uses a student's own homework to decide how to grade that same student, bias creeps in. CatBoost's ordered boosting is like the teacher grading each student using only the answer keys from students who submitted *earlier*.

---

<!-- [CHEAT] -->
## 27. Mathematical Formulation

### Objective

```
min Σ L(yᵢ, ŷᵢ) + Σ Ω(fₜ)
```

where Ω(fₜ) = l2_leaf_reg · Σ (leaf_weight)² (L2 regularisation on leaf values).

### Ordered Target Statistic

```
enc(xᵢ, σ) = (a · μ + Σ_{k=1}^{i-1} [x_σ(k) = x_σ(i)] · y_σ(k))
              ────────────────────────────────────────────────────────
                   a + |{k < i : x_σ(k) = x_σ(i)}|
```

Averaged over P permutations for stability.

### Symmetric Tree Split

At each depth level, find ONE (feature, threshold) that maximises:

```
gain = Σ_leaves [ (N_L · μ_L² + N_R · μ_R²) ]
```

where N_L, N_R are left/right counts and μ_L, μ_R are mean gradients in each child. The same split is applied to ALL leaves at that depth.

### Prediction

```
ŷ = μ + Σ_{t=1}^{T} η · fₜ(x)
```

where fₜ is a symmetric tree and μ = prior = mean(y).

---

<!-- [GATE] -->
## 28. GATE Quick-Revision Card

```text
CatBoost = Ordered TS + Ordered Boosting + Symmetric Trees

Ordered TS:  enc(i) = (a·prior + Σ_{j<i} yⱼ) / (a + count)
             → No leakage (own target excluded)

Ordered Boosting: leaf(i) uses only gradients from j < i in permutation
                  → No prediction shift

Symmetric Tree: same split at every depth level
                → 2^d leaves, d comparisons, fast inference

Prior smoothing: prevents extreme values for rare categories
                  → stabilises TS when count is low

Prediction Shift: circular dependency between model and gradients
                  → CatBoost breaks it with ordered estimation

Compare:
  XGBoost:   level-wise, regularise, no ordered anything
  LightGBM:  leaf-wise, histogram, fast on numeric
  CatBoost:  symmetric, ordered, best on categoricals
```

---

<!-- [CASES] -->
## 29. Case Studies

### Case 1 — Flipkart Product Pricing (Regression)

**Problem:** Predict listed price for 50M products. Features: brand (4000 values), category (500), colour (50), seller_type (3), city (800), weight, rating.

**Why CatBoost:** 5 categorical columns with high cardinality. One-hot → 4500+ columns, sparse, slow. Label encoding → ordinal assumption wrong. Target encoding → leakage. CatBoost's ordered TS handles all of this natively.

**Result:** CatBoost typically outperforms XGBoost (with manual target encoding) by 3-8% RMSE on such data, with less preprocessing.

### Case 2 — Credit Risk Scoring (Classification counterpart)

**Problem:** Predict loan default probability. Features: income, employment_type (5 values), city_tier (4), education (4), loan_type (10).

**Why CatBoost:** Categorical-heavy, leakage prevention critical (financial data). Ordered boosting adds a second safety layer against the gradual overfitting that plagues standard GBM on financial time-series.

### Case 3 — When NOT to use CatBoost

**Problem:** Predict house prices from 200 numeric features (square footage, rooms, age, GPS coordinates, etc.) on 2M rows.

**Why LightGBM instead:** Purely numeric, large dataset. LightGBM's histogram binning and leaf-wise growth are 3-5x faster. CatBoost's ordered TS overhead adds no value here (no categoricals to encode).

---

<!-- [VERIFY] -->
## 30. Verification Checklist

Before deploying a CatBoost model, verify:

- [ ] **No leakage:** if you replaced ordered TS with naïve target encoding, did test RMSE get worse? (It should — confirming ordered TS helps.)
- [ ] **No shift:** plot train vs test loss per iteration. The gap should be smaller than equivalent XGBoost/LightGBM on the same data.
- [ ] **cat_features passed:** run `model.get_feature_importance()` — if categorical columns appear, they were treated correctly.
- [ ] **Early stopping worked:** `model.get_best_iteration()` < `iterations`. If not, increase iterations or decrease learning_rate.
- [ ] **Symmetric tree depth:** depth 6 gives 64 leaves — check that this is sufficient for your data's complexity.

---

<!-- [PATTERN] -->
## 31. Representative Pattern Question

> **Question:** You have a dataset with 30 features, 15 of which are categorical (some with 1000+ unique values). Your colleague one-hot encodes all categoricals and trains XGBoost. The model overfits badly (train R²=0.99, test R²=0.65). What are the three likely causes, and how does CatBoost address each?

**Answer sketch:**

1. **Target leakage from naïve target encoding** (if they used it) — CatBoost uses ordered TS with permutation-based exclusion, so no sample sees its own target.
2. **Dimensionality explosion from one-hot** (5000+ sparse columns) — CatBoost handles categories natively, no one-hot needed.
3. **Prediction shift** (standard GBM trains on biased gradients) — CatBoost's ordered boosting eliminates this.

The test-train gap of 0.34 would likely shrink to 0.05–0.10 with CatBoost on this data.

---

<!-- [COMPARISON] -->
## 32. Boosting Family Comparison

| Property | GBM | AdaBoost | XGBoost | LightGBM | CatBoost |
|---|---|---|---|---|---|
| Core idea | Residual fitting | Reweight misclassified | 2nd-order optimisation | Leaf-wise histogram | Ordered TS + ordered boosting |
| Categorical handling | Manual | Manual | Manual/sorted | Sorted by target | **Native ordered TS** |
| Prediction shift | Yes | Yes | Yes | Yes | **No** (ordered boosting) |
| Tree structure | Unbalanced | Unbalanced | Level-wise | Leaf-wise | **Symmetric (oblivious)** |
| Speed (numeric, large) | Slow | Slow | Medium | **Very fast** | Medium |
| Speed (categorical) | Slow | Slow | Medium | Medium | **Fast** (native) |
| Default quality | Low | Medium | High | High | **Very high** |
| Overfitting control | Weak | Medium | Strong (L1/L2) | Medium | **Strong** (ordered + L2) |
| Inference speed | Slow | Slow | Medium | Medium | **Very fast** (symmetric) |

---

<!-- [PRACTICE] -->
## 33. Mastery Test

**Conceptual (8 questions):**

1. What is ordered target statistics, and how does it prevent leakage?
2. What is prediction shift, and why does it occur in standard GBM?
3. How does ordered boosting break the circular dependency causing shift?
4. What is a symmetric (oblivious) tree? How many leaves does a depth-5 symmetric tree have?
5. What role does the prior play in ordered TS? When is it most important?
6. Why might CatBoost have a smaller train-test gap than XGBoost on categorical data?
7. Explain why one-hot encoding high-cardinality categoricals is problematic for GBM.
8. When would you choose LightGBM over CatBoost?

**Numerical (2 questions):**

9. Given: `prior=100, a=1, category "Gold" has earlier rows with targets [80, 120, 90]`. Compute the ordered TS for a new "Gold" row.

10. A symmetric depth-3 tree is trained. At depth 1, the best split is `x₃ ≤ 5`. At depth 2, the best split (applied to BOTH children) is `x₁ ≤ 2`. At depth 3, the best split is `x₅ ≤ 7`. How many leaves does this tree have, and what is the prediction path for the sample x = [3, 4, 6, 1, 8]?

**Answers:**
- Q9: enc = (1×100 + 80+120+90) / (1+3) = 390/4 = **97.5**
- Q10: 2³ = **8 leaves**. Path: x₃=6>5 → right; x₁=3>2 → right; x₅=8>7 → right → right-right-right leaf.

---

<!-- [NEXT] -->
## 34. What Next?

You have now completed the boosting family:
- 13. Gradient Boosting → the foundation
- 14. AdaBoost → reweighting intuition
- 15. XGBoost → regularised, 2nd-order, production-grade
- 16. LightGBM → histogram speed, leaf-wise growth
- 17. CatBoost → leakage-free categoricals, ordered boosting

**Natural next steps:**
- **Stacking** — combine XGBoost + LightGBM + CatBoost into a meta-learner (often the winning move in competitions)
- **Random Forest** (B-classification folder) — bagging alternative, no boosting shift, simpler
- **Model selection & ensembling** — when to use which, how to blend

CatBoost's classification counterpart is in the **B-classification** folder (17-catboost.md).
