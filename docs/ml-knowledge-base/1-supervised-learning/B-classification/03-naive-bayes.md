# 03. Naive Bayes

<!-- [STORY] -->
> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐☆☆☆
> GATE: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆
>
> Journey: **story → guess → Bayes rule → priors/likelihoods → the naive twist → smoothing → variants → code → break → when to use → deep dive.**
> Level 1 = sections 01–18. Level 2 = 19–26. Level 3 = 27–34.

---

## 01. Start Here

Naive Bayes is the **probability-first classifier** — it turns classification into "which class makes this evidence most likely?" It's the engine behind the spam filters on your phone.

By the end you will be able to:

- apply Bayes' theorem to a classification problem,
- compute priors and likelihoods by hand,
- explain the "naive" assumption and why it *works anyway*,
- use log-probabilities and Laplace smoothing,
- pick the right variant (Gaussian / Multinomial / Bernoulli),
- code it from scratch and with sklearn,
- and break it on purpose.

> One tiny math rule powers it all. Let's find it.

---

## 02. The Problem

Aman is the host of a radio quiz show where listeners send in SMS answers. Annoyingly, some messages are **junk/spam** ("FREE credit, claim now") and some are **real entries** ("Kohli's century, answer is 200").

He tags 5 messages and gives you their features:

| Message | Contains "FREE"/"WIN" | Contains "answer" | Real entry / Spam |
|---|---|---|---|
| "FREE CASH for you" | Yes | No | Spam |
| "WIN a new phone" | Yes | No | Spam |
| "the answer is 200" | No | Yes | Real |
| "FREE WIN winner" | Yes | No | Spam |
| "answer please reply" | No | Yes | Real |

A new message arrives:

> **"WIN the FREE answer guide"**  → Contains "FREE/WIN": **Yes** · Contains "answer": **Yes**

<!-- [QUESTION] -->
**Is it Real or Spam?**

Think: it has a spammy word ("WIN/FREE") *and* a real word ("answer"). What do you predict?

**Your guess: Spam ☐   Real ☐**

> 📌 Keep your reasoning visible — you probably weighed *both* clues. Naive Bayes does exactly that, mathematically.

---

## 03. Let's Think

Look at the data as **clues** instead of rows:

```text
"FREE/WIN" present  →  4 messages:  3 Spam, 1 Real
"answer" present    →  3 messages:  2 Real, 1 Spam
```

<!-- [THINK_ABOUT_IT] -->
🤔 What does each clue suggest by itself?

> "FREE/WIN" leans **Spam** (3 of 4). "answer" leans **Real** (2 of 3).

Now the new message has *both* clues. It's pulling in two directions.

> The twist: naive Bayes **multiplies** the evidence from both clues together, as if each clue votes independently, then picks whichever class gets the bigger combined score.

---

## 04. Intuition

Imagine a doctor with a hunch and a test result. She starts from a **prior belief** (how common the disease is), then **updates** with the test's likelihood. Bayes' rule is the formal way to do that update — it flips "the probability of this test given the disease" into "the probability of the disease given this test."

Naive Bayes uses the same idea for classification:

1. Start with the **prior**: how common is each class?
2. For each feature, look at how likely this value is **given each class**.
3. **Multiply** all the evidence together per class.
4. Pick the class with the biggest product.

💡 **The idea in one line:**

> Naive Bayes predicts the class with the highest **prior × (product of feature likelihoods)** — every feature gets a vote, and the votes multiply.

The "naive" part: it pretends **all clues are independent**. "FREE" and "answer" are treated as if they never co-occur meaningfully — obviously false in real language, yet the method still works. We'll see why.

---

## 05. Visual First

Here's the whole flow as a picture:

<!-- [VISUAL] -->
```text
                    NEW MESSAGE
                         │
      ┌──────────────────┼──────────────────┐
      │ (given class)    │                   │
      ▼                  ▼                   ▼
   "FREE/WIN"?        "answer"?         prior P(class)
   P(Yes|Spam)      P(Yes|Spam)        (how common)
   vs P(Yes|Real)   vs P(Yes|Real)
      │                  │                   │
      └──────── declanetra ────────────┘
       multiply: P(Spam|msg) ∝ P(Spam)·P(free|Spam)·P(ans|Spam)
                  P(Real|msg) ∝ P(Real)·P(free|Real)·P(ans|Real)
                         │
                         ▼
                 bigger score wins
```

> Two parallel "score ladders," one per class. Each feature climbs its own ladder by multiplying its likelihood. Reach the top and compare totals.

---

## 06. First Prediction

Let's do a quick count-based estimate. Priors from the data:

```text
P(Spam) = 3/5 = 0.6      P(Real) = 2/5 = 0.4
```

Clue likelihoods (we'll make them exact in Section 10):

```text
P("FREE/WIN"=Yes | Spam) = 3/3 = 1.0      P("FREE/WIN"=Yes | Real) = 0/2 = 0
P("answer"=Yes   | Spam) = 0/3 = 0        P("answer"=Yes   | Real) = 2/2 = 1.0
```

For the new message (both Yes):

```text
Score_Spam = 0.6 × 1.0 × 0    = 0     ← zero! one clue "kills" it
Score_Real = 0.4 × 0   × 1.0  = 0     ← also zero!
```

<!-- [TRY_IT] -->
> Oops. The raw counts give **zero for both** — because no Spam message in training had "answer," and no Real message had "FREE/WIN."

> 📌 This is the **zero-frequency problem**, and it's exactly why a plain count fails. Before we can answer the question, we need **smoothing** — Section 10 shows the fix (and the correct answer).

For now, hold the question. The important lesson:

> Even a tiny dataset can produce a zero that wipes out all other evidence. Real Naive Bayes never lets that happen — it adds a small constant so no probability is ever 0.

---

## 07. Core Concept

Introducing the idea formally:

**Concept: Naive Bayes** — a classifier based on **Bayes' theorem**:

```text
P(class | features)  ∝  P(class) · P(features | class)
```

with the **conditional-independence ("naive")** assumption:

```text
P(features | class)  =  P(feature₁|class) · P(feature₂|class) · … · P(feature_d|class)
```

The prediction is the class maximizing:

```text
ŷ = argmax_c  P(c) · Πᵢ P(xᵢ | c)
```

> It's **generative**: it models how each class *produces* its features (P(x|y)), then flips it with Bayes to get the class given features (P(y|x)).

---

## 08. Terminology

### Bayes' Theorem

> Simple: a rule to flip "probability of A given B" into "probability of B given A."
> Technical: `P(A|B) = P(B|A)·P(A)/P(B)`.

### Prior

> Simple: how common the class is before looking at features.
> Technical: `P(c)` — the class frequency in training data.

### Likelihood

> Simple: how well this feature value fits a class.
> Technical: `P(xᵢ | c)` — feature probability given class.

### Posterior

> Simple: what we believe after seeing the features.
> Technical: `P(c | x)` — class probability given features.

### Evidence (P(x))

> Simple: how common the overall feature combo is.
> Technical: the normalizing constant — the same for all classes, so we drop it.

### Conditional independence / "naive"

> Simple: each clue votes alone, ignoring the others.
> Technical: `P(x₁,x₂|c) = P(x₁|c)·P(x₂|c)`.

### Laplace smoothing

> Simple: add a tiny number so no probability is zero.
> Technical: `(count + α)/(total + α·V)`.

| Term | Simple meaning | Technical meaning |
|---|---|---|
| P(c) | how common the class is | prior |
| P(xᵢ\|c) | how well this clue fits the class | likelihood |
| P(c\|x) | belief after seeing clues | posterior |
| Πᵢ P(xᵢ\|c) | multiply all the clues | the "naive" product |
| P(x) | same for all classes | evidence (dropped) |
| α | the smoothing constant | Laplace parameter |

> ⚠️ Common mistake: thinking the "naive" part means Naive Bayes is bad. It means it *assumes independence* — a deliberate simplification that buys speed and small-data power.

---

## 09. Mathematics (gradual)

### Step M1 — Bayes' theorem

```text
P(class | features) = P(features | class) · P(class) / P(features)
```

- Left side: what we want (class given the features) — the **posterior**.
- `P(features|class)`: the **likelihood** — how the class produces these features.
- `P(class)`: the **prior** — class frequency.
- `P(features)`: the **evidence** — constant across classes for a fixed input.

> 💡 The denominator doesn't change the *winner*, only the scale. For picking the best class, we can ignore it:

```text
P(c | x)  ∝  P(c) · P(x | c)      (∝ means "proportional to")
```

### Step M2 — The problem: P(features | class) is hard

If features interact, computing `P(x₁, x₂, …, x_d | c)` requires modeling all their joint behaviour — impossible with limited data.

### Step M3 — The naive assumption

Assume conditional independence:

```text
P(x₁,…,x_d | c) = P(x₁|c) · P(x₂|c) · … · P(x_d|c) = Πᵢ P(xᵢ|c)
```

Now we only need simple per-feature probabilities, which are easy to estimate. This is the entire "naive" simplification.

### Step M4 — The classification rule

```text
ŷ = argmax_c  P(c) · Πᵢ P(xᵢ | c)
```

### Step M5 — The log trick

Multiplying many small probabilities underflows to 0.0 in floating point (e.g. `0.9⁵⁰⁰ ≈ 0`). Since log is monotonic, the argmax is unchanged:

```text
ŷ = argmax_c  [ log P(c) + Σᵢ log P(xᵢ | c) ]
```

> 💡 We never actually multiply; we **add log-probabilities**. Stable and preserves the answer.

### Step M6 — Smoothing (the zero fix)

If a feature value was never seen in a class, `P = 0` kills the whole product. Add α:

```text
P(xᵢ | c) = (count(xᵢ, c) + α) / (total_c + α·V)
```

where `V` = number of possible values. α = 1 gives **Laplace smoothing** — no probability is ever exactly 0.

---

## 10. Numerical Example

Back to Aman's SMS, now with smoothing so we can actually answer the question. Ignore α=1 for a moment and let's compute the **exact likelihoods with Laplace smoothing** using a tiny trick: add 1 to every count (α = 1), and use vocabulary V = 2 features.

Dataset:

| Msg | FREE/WIN | answer | Class |
|---|---|---|---|
| 1 | Yes | No | Spam |
| 2 | Yes | No | Spam |
| 3 | Yes | No | Spam |
| 4 | No | Yes | Real |
| 5 | No | Yes | Real |

**Priors:**

```text
P(Spam) = 3/5 = 0.6      P(Real) = 2/5 = 0.4
```

**Smoothed likelihoods (α=1, V=2):**

For Spam (3 messages):

```text
P(FREE=Yes|S) = (count_Yes + 1)/(total + α·V) = (3 + 1)/(3 + 2) = 4/5 = 0.8
P(FREE=No |S) = (0 + 1)/(3 + 2)               = 1/5 = 0.2
P(ans=Yes |S) = (0 + 1)/(3 + 2)               = 1/5 = 0.2
P(ans=No  |S) = (3 + 1)/(3 + 2)               = 4/5 = 0.8
```

For Real (2 messages):

```text
P(FREE=Yes|R) = (0 + 1)/(2 + 2)   = 1/4 = 0.25
P(FREE=No |R) = (2 + 1)/(2 + 2)   = 3/4 = 0.75
P(ans=Yes |R) = (2 + 1)/(2 + 2)   = 3/4 = 0.75
P(ans=No  |R) = (0 + 1)/(2 + 2)   = 1/4 = 0.25
```

**Posterior scores** for the new message (FREE=Yes, answer=Yes):

```text
Score(S) = log 0.6  + log 0.8 + log 0.2
         = −0.511   − 0.223   − 1.609        = −2.343

Score(R) = log 0.4  + log 0.25 + log 0.75
         = −0.916   − 1.386   − 0.288        = −2.590
```

<!-- [CALCULATION] -->
> Score(S) = −2.343 > Score(R) = −2.590, so the model predicts **Spam.**

> ✅ VERIFIED — every smoothed probability and log-score hand-computed.

**Why Spam?** The strong spam clue ("FREE/WIN") outweighs the real clue ("answer"). No zero killed anything this time — that's smoothing doing its job.

> 🎯 Your turn: without smoothing, we saw both scores hit exactly 0 (Section 06). With α=1, we get a real, sensible answer. That single difference is why smoothing exists.

---

## 11. How It Works

```text
TRAINING PHASE:
   For each class c:
      prior[c] = count(c) / n
      For each feature i:
         estimate P(xᵢ|c) from training data
         (Gaussian: mean+var per class · Multinomial: counts
          · Bernoulli: probabilities)

PREDICTION PHASE (query x):
   For each class c:
      score[c] = log prior[c] + Σᵢ log P(xᵢ|c)
   ŷ = the class with the largest score[c]
```

> One pass to build the "score table." Then any new message is a few lookups + adds.

---

## 12. Internal Process (what fit() really does)

<!-- [UNDER_THE_HOOD] -->
```text
model.fit(X, y)
     ↓
1. For each class, count samples → priors P(c)
2. For each class × feature, estimate likelihood params:
     - Gaussian:     mean μ_c and variance σ_c² per feature per class
     - Multinomial:  word counts per class
     - Bernoulli:    P(present|c), P(absent|c)
     ↓
3. Store: priors + a compact likelihood table
     ↓
done.  (No iterations, no gradient descent — closed-form statistics)

model.predict_proba(X_new)
     ↓
for each class c in log-space:
    score = log P(c) + Σ feature log P(xᵢ|c)
softmax the scores into probabilities, argmax → label
```

> The "model" is a small table of probabilities — not the raw data (unlike KNN), and no weight vector (unlike logistic regression).

---

## 13. From Scratch

### Version 1 — Gaussian Naive Bayes, readable

```python
import math

class GaussianNB:
    def __init__(self):
        self.classes = None
        self.priors = {}
        self.means = {}
        self.vars = {}

    def fit(self, X, y):
        self.classes = sorted(set(y))
        n = len(y)
        for c in self.classes:
            rows = [X[i] for i in range(n) if y[i] == c]
            self.priors[c] = len(rows) / n
            d = len(rows[0])
            self.means[c] = [sum(r[j] for r in rows) / len(rows) for j in range(d)]
            self.vars[c] = [
                sum((r[j] - self.means[c][j]) ** 2 for r in rows) / len(rows)
                for j in range(d)
            ]

    def _log_pdf(self, x, c):
        s = 0.0
        for j, xj in enumerate(x):
            mu, v = self.means[c][j], self.vars[c][j] + 1e-9
            s += -0.5 * math.log(2 * math.pi * v) - (xj - mu) ** 2 / (2 * v)
        return s

    def predict(self, X):
        out = []
        for x in X:
            scores = {c: math.log(self.priors[c]) + self._log_pdf(x, c)
                      for c in self.classes}
            out.append(max(scores, key=scores.get))
        return out

X = [[2, 3], [3, 4], [5, 1], [6, 2]]
y = [0, 0, 1, 1]
print(GaussianNB().fit(X, y).predict([[3, 3]]))
```

> `_log_pdf` is the Gaussian density `1/√(2πσ²)·exp(−(x−μ)²/2σ²)`, in log space.

### Version 2 — numpy, vectorized

```python
import numpy as np

class GaussianNBVec:
    def fit(self, X, y):
        X = np.asarray(X, float)
        self.classes = np.unique(y)
        self.means = {}
        self.vars = {}
        self.priors = {}
        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / len(y)
            self.means[c] = Xc.mean(axis=0)
            self.vars[c] = Xc.var(axis=0) + 1e-9
        return self

    def _log_likelihood(self, X, c):
        var = self.vars[c]
        return (-0.5 * np.log(2 * np.pi * var)
                - (X - self.means[c]) ** 2 / (2 * var)).sum(axis=1)

    def predict(self, X):
        X = np.asarray(X, float)
        scores = np.array([
            np.log(self.priors[c]) + self._log_likelihood(X, c)
            for c in self.classes
        ])
        return np.array([self.classes[i] for i in scores.argmax(axis=0)])
```

### Version 3 — clean class (with score)

```python
import numpy as np

class NaiveBayesGaussian:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y)
        self.classes = np.unique(y)
        self.means, self.vars, self.priors = {}, {}, {}
        for c in self.classes:
            Xc = X[y == c]
            self.priors[c] = len(Xc) / len(y)
            self.means[c] = Xc.mean(axis=0)
            self.vars[c] = Xc.var(axis=0) + self.var_smoothing
        return self

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

    def _scores(self, X):
        return np.array([
            np.log(self.priors[c])
            + (-0.5 * np.log(2 * np.pi * self.vars[c])
               - (X - self.means[c]) ** 2 / (2 * self.vars[c])).sum(axis=1)
            for c in self.classes
        ])

    def predict(self, X):
        X = np.asarray(X, float)
        return np.array([self.classes[i] for i in self._scores(X).argmax(axis=0)])
```

---

## 14. Library Implementation

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Continuous features → Gaussian
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.4f}")
print(gnb.class_prior_)          # learned priors P(c)
print(gnb.theta_)                # per-class means (μ_c)
print(gnb.var_)                  # per-class variances (σ_c²)
print(gnb.predict_proba(X_test[:3]))   # posterior probabilities
```

```python
# Text / word counts → Multinomial
from sklearn.feature_extraction.text import CountVectorizer
texts = ["free cash now", "win a prize", "meeting tomorrow", "answer is 200"]
labels = ["spam", "spam", "real", "real"]

vec = CountVectorizer()
X = vec.fit_transform(texts)

mnb = MultinomialNB(alpha=1.0)      # α = Laplace smoothing
mnb.fit(X, labels)
print(mnb.predict(vec.transform(["free prize win"])))
```

> `class_prior_`, `theta_`, `var_` let you see exactly the numbers we computed by hand — nothing hidden.

---

## 15. Code Walkthrough — why each line exists

<!-- [CODE_WALKTHROUGH] -->
```python
self.vars[c] = Xc.var(axis=0) + 1e-9
```
> Variance per feature per class; `+1e-9` prevents division by zero when a feature is constant in a class. That tiny epsilon is sklearn's `var_smoothing`.

```python
sigma2 = self.vars[c]
(n ll) = -0.5*np.log(2*np.pi*sigma2) - (X - self.means[c])**2/(2*sigma2)
```
> The Gaussian log-density, summed over features. Log form = numerically stable.

```python
scores = np.log(self.priors[c]) + self._log_likelihood(X, c)
```
> `log prior + Σ log likelihood` per class — the full classification score from Section 09.

```python
self._scores(X).argmax(axis=0)
```
> The argmax over classes — "pick the class with the biggest score."

> 🧠 Every line is Section 09's formula in code. Nothing arbitrary.

---

## 16. Interactive Experiment

<!-- [EXPERIMENT] -->
> Sliders in the platform; otherwise run the code.

### Experiment A — slide the smoothing α

A slider for α on the SMS data:

```text
α = 0     →  one unseen word kills everything (zeros everywhere)
α = 0.1   →  tiny smoothing; probabilities still extreme
α = 1.0   →  Laplace; no zeros, sensible answers (our Section 10)
α = 5.0   →  heavy smoothing; all likelihoods pulled toward 1/V, less opinionated
```

> What to notice: as α grows, the model becomes **less confident** (likelihoods flatten toward uniform). α = 0 is dangerous; α too large loses signal.

### Experiment B — irrelevant-feature robustness (code)

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

rng = np.random.default_rng(3)
X = np.hstack([
    rng.normal([0,0],[1,1],(200,2)),   # two informative features
    rng.normal(0, 100, (200, 50)),     # 50 pure-noise features
])
y = np.array([0]*100 + [1]*100)
noise = np.random.default_rng(0).normal(0, 100, (200, 50))

nb = GaussianNB().fit(X, y)
print("with 50 noise features:", nb.score(X, y))

X_clean = X[:, :2]
nb2 = GaussianNB().fit(X_clean, y)
print("clean (2 features):    ", nb2.score(X_clean, y))
```

```text
with 50 noise features: ≈ same / slightly worse
clean (2 features):     ≈ same / slightly better
```

> 📌 The moral: **irrelevant features hurt Naive Bayes far less than KNN.** KNN's distances drown in noise; NB's per-feature likelihoods are individually estimated and mostly cancel out. That's a big reason NB shines on text with thousands of noisy words.

---

## 17. Break the Model

<!-- [BREAK_IT] -->
**Experiment 1 — a word that never appears in one class (zero-frequency).**

```python
from sklearn.naive_bayes import MultinomialNB

X = [[1,0],[1,0],[0,1],[0,1]]          # 2 features: [free, answer]
y = ["spam","spam","real","real"]

nb = MultinomialNB(alpha=0.0)          # NO smoothing!
nb.fit(X, y)
print(nb.predict([[1, 1]]))            # both words present
```

```text
predict([[1,1]]) →  error / nonsense (log(0) = -inf → 0 chance for both)
```

With α = 0, a single zero likelihood turns `log(0) = −∞` and nukes the class. **This is the break that smoothing fixes.**

**Experiment 2 — correlated features (double-counting).**

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB

base = np.random.default_rng(1).normal([0,0],[1,1],(200,2))
y = np.array([0]*100 + [1]*100)

# duplicate the same underlying signal 20 times
X_dup = np.hstack([base for _ in range(20)])
nb = GaussianNB().fit(X_dup, y)
probs = nb.predict_proba(X_dup[:1])[0]
print("probabilities (correlated dups):", probs.round(3))
```

```text
probabilities: [0.001, 0.999]    ← wildly overconfident
```

**What happened?** The model counted the *same* evidence 20 times (it thinks all 20 duplicated features are independent signals). Real probability might be ~80/20, but naive Bayes says 0.1%/99.9%. **Overconfidence is the classic naive-Bayes failure.**

> 💥 **Break pattern:** naive = treats correlated clues as independent → overcounts them → overconfident probabilities.

---

## 18. What If...?

| You change… | What happens | Why |
|---|---|---|
| Add Laplace α=1 | No more zeros | every count gets a small tail |
| Remove smoothing (α=0) | A single unseen word → −∞ | zero likelihood kills the product |
| Correlate features (duplicate signal) | Overconfident probabilities | double-counts evidence |
| Add many irrelevant features | Barely hurts | per-feature likelihoods cancel out |
| Use Gaussian on word counts | Poor performance | wrong distribution for the data |
| Use Multinomial on continuous values | Poor performance | wrong variant |
| Imbalanced classes | Priors dominate | P(c) reflects the bigger class |
| Very small dataset | Still trains fine | just counts/means — few parameters |
| Text with rare words | Without smoothing it breaks | zero-frequency on unseen words |

> 🤔 Think: which change is *not* fixed by more data? → The overconfidence from correlated features. More data doesn't remove the double-counting — that's built into the naive assumption. (Calibration or a better model handles it.)

---

## 19. Hyperparameters

**Learned by the model (parameters):**

```text
Priors P(c)                         →  class_prior_
Per-class likelihood parameters     →  theta_ (μc), var_ (σc²)  [Gaussian]
                                      word-count table          [Multinomial]
```

**Chosen by you (hyperparameters):**

| Hyperparameter | Simple meaning | Too small | Too big | Typical |
|---|---|---|---|---|
| `alpha` (smoothing) | how much to pad counts | 0 → zeros break it | too flat, no signal | 1.0 (Laplace) |
| `var_smoothing` | eps for stability (Gaussian) | division-by-zero risk | too flat | 1e-9 |
| `class_prior` | override the priors | — | — | leave default |
| `binarize` (Bernoulli) | threshold to make features binary | — | — | 0.0 |
| series variant | Gaussian / Multinomial / Bernoulli | — | — | match the data |

> 📌 Naive Bayes has **very few** hyperparameters — that's a strength. You mostly tune α and pick the right variant.

---

## 20. Assumptions

For each: what, why, how to check, if violated.

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| **Conditional independence** | features independent given class | allows the simple product | correlation per class | often still works (see why, Sec 16/25) |
| **Right distribution** | features match Gaussian/Multinomial/Bernoulli | likelihoods must be sensible | histograms per class | use the matching variant |
| **Zero-free (smoothing)** | no feature is impossible in a class | avoids −∞ | — | always smooth |
| **Representative priors** | class frequency ≈ data frequency | prior drives decisions | class counts | set class_prior |

> The independence assumption is **violated in almost every real dataset** — and Naive Bayes still often works. Why? For classification we only need the *relative* ordering of scores; correlations that affect all classes similarly cancel out in the argmax.

---

## 21. Data Requirements

```text
Target      → categorical
Features    → depends on variant:
               Gaussian (continuous) · Multinomial (counts) · Bernoulli (binary)
Missing     → not handled natively; impute (0-counts work naturally for text)
Outliers    → Gaussian NB sensitive (they distort μ, σ²); others fine
Scaling     → NOT needed (each feature has its own distribution)
Feature engineering → text: bag-of-words / TF-IDF, remove stopwords, n-grams
Size        → works with very little data (major strength)
High-dim    → excellent (text-scale features) — unlike KNN
Class imbalance → severe imbalance → correct/override priors
```

---

## 22. Evaluation

Same classification workhorses as the previous two notes:

| Metric | Formula | Simple | Use | Avoid |
|---|---|---|---|---|
| Accuracy | (TP+TN)/total | % correct | balanced | imbalanced |
| Precision | TP/(TP+FP) | of predicted Yes how many right | FP costly | when FN worse |
| Recall | TP/(TP+FN) | of actual Yes how many caught | FN costly | when FP worse |
| F1 | 2·P·R/(P+R) | balance | imbalanced | need one alone |
| Log-loss | −Σ[y log ŷ + (1−y)log(1−ŷ)] | how good probabilities are | probs matter | only labels matter |
| ROC-AUC | area under ROC | ranking | comparing | calibrated probs needed |

**Loss ≠ Metric:**

```text
NAIVE BAYES IS TRAINED BY MAXIMIZING LIKELIHOOD (MLE), not by minimizing
accuracy/F1/AUC. Those metrics are computed AFTER training.
NOTE: naive-Bayes probabilities are often overconfident (Sec 17) → if you
report log-loss, expect it to look worse than a well-calibrated model's.
```

---

## 23. Failure Cases

```text
DATA            → a feature value unseen in a class (zero) — fixed by smoothing
MATHEMATICAL    → strongly correlated features → overconfident posteriors
OPTIMIZATION    → none (closed-form stats can't "fail to converge")
GENERALIZATION  → wrong distribution assumed (e.g. Gaussian on counts)
PRACTICAL       → treating overconfident probabilities as calibrated truth
```

---

## 24. Debugging

Model underperforming? Checklist:

```text
1. Predictions always one class?      → priors skewed → check class balance
2. Probabilities ~0 or ~1 always?     → correlated features double-count → calibrate
3. Any log(0) / -inf errors?          → you disabled smoothing → turn α back on
4. Continuous data far from Gaussian? → use a different likelihood or transform
5. Count data with GaussianNB?        → wrong variant → use MultinomialNB
6. New vocabulary words break it?      → unseen-word zeros → add smoothing
```

---

## 25. Compare

Conceptual difference **first**:

```text
Logistic Regression:   "learn a boundary, output calibrated probability"
Naive Bayes:           "combine independent probability votes (generative)"
KNN:                   "nearest neighbours vote by distance"
Decision Tree:         "learn rules by impurity reduction"
```

| Algorithm | Idea | Strength | Weakness | Best use |
|---|---|---|---|---|
| Naive Bayes | Bayes + independence | tiny data, text-scale, fast | naive assumption, overconfident | spam/text, small data |
| Logistic Regression | linear + sigmoid | calibrated probs, tractable | needs more data to be stable | baseline, risk scores |
| KNN | neighbour vote | no training, non-linear | slow, curse of dim | small data |
| Decision Tree | rules | readable | overfits | auditability |

> **Special note:** on very sparse, high-dimensional text with skewed classes, Multinomial NB often *beats* logistic regression — its per-class word statistics stay robust where LR's global optimization suffers from sparse gradients. That's the one place naive Bayes genuinely wins.

---

## 26. Real-World Workflow

```text
BUSINESS PROBLEM:  auto-tag incoming support emails as "complaint" or "inquiry"
DATA:              5000 historical emails + labels
FEATURES:          bag-of-words counts (or TF-IDF)
TARGET:            complaint? 1/0
MODEL:             MultinomialNB(alpha=1.0)
TRAIN:             CountVectorizer → split → fit
EVALUATE:          F1 + confusion matrix; sanity-check priors
DEPLOY:            serialize with joblib, serve via API
MONITOR:           as vocabulary drifts, retrain; streaming updates are cheap
```

> Same skeleton powers spam filters, sentiment, topic labeling, and insurance auto-routing.

---

## 27. Practice

8 levels:

1. **Recall:** state Bayes' theorem.
2. **Understand:** what does "naive" mean, and why doesn't it always hurt?
3. **Calculate:** compute the posterior for Aman's message by hand with α=1.
4. **Apply:** which variant for word counts? for heart-rate? for symptom presence?
5. **Debug:** probabilities are ~0/1 always. Likely cause and fix?
6. **Experiment:** vary α across [0, 0.1, 1, 5]; observe when zeros/overconfidence appear.
7. **Build:** spam filter mini-project — clean, vectorize, MultinomialNB, report precision/recall, choose a threshold.
8. **Explain:** explain Naive Bayes to a friend in 60 seconds using the quiz-show story.

---

## 28. Interview

### Beginner
- **State Bayes' theorem.** `P(A|B) = P(B|A)·P(A)/P(B)`.
- **Why "naive"?** Assumes features are conditionally independent given the class.
- **What is Laplace smoothing and why?** Adds α to counts so no probability is 0 (zero-frequency).
- **Three variants?** Gaussian (continuous), Multinomial (counts), Bernoulli (binary).

### Intermediate
- **Why does it work despite violating independence?** Only the relative ordering of class scores matters; correlations that affect all classes similarly cancel out.
- **Generative vs discriminative?** NB models P(x|c)·P(c) then flips via Bayes; logistic models P(c|x) directly.
- **Why log instead of product?** Underflow; summing logs is stable and preserves the argmax.
- **What's the zero-frequency problem?** An unseen value → 0 probability → whole product 0. Smoothing fixes it.

### Advanced
- **Why are NB probabilities overconfident?** Correlated features are counted as independent, so evidence is double-counted; calibrate with Platt/isotonic.
- **When does NB beat logistic regression?** Very few samples, skewed priors, or many irrelevant/sparse features — LR's global MLE wobbles where NB's per-class statistics stay stable.
- **Relation to Bayes-optimal classifier?** NB approximates the Bayes-optimal rule (argmax P(c|x)); it's optimal only if independence truly holds.
- **Incremental learning?** Estimation uses cumulative counts/means — update them as data arrives, no retrain.

---

## 29. GATE / Exam

**Formulas worth memorizing:**

```text
Bayes:         P(A|B) = P(B|A)·P(A)/P(B)
Naive rule:    P(c|x) ∝ P(c)·Πᵢ P(xᵢ|c)
Log form:      ŷ = argmax_c [ log P(c) + Σᵢ log P(xᵢ|c) ]
Smoothing:     P(xᵢ|c) = (count + α)/(total + α·V)
Gaussian PDF:  P(x) = (1/√(2πσ²))·exp(−(x−μ)²/(2σ²))
```

**Common traps:**
- Forgetting the **prior** P(c) term.
- Confusing Gaussian-NB (a model) with "assume the data is Normal."
- Dropping P(x) without knowing why — it's constant across classes.
- Treating the zero-frequency problem as harmless.
- Thinking "naive" means "wrong" — it's a deliberate assumption.

> **Representative pattern question (NOT a past GATE PYQ):** "P(disease)=0.02, P(positive|disease)=0.99, P(positive|healthy)=0.05. Find P(disease|positive)." → numerator = 0.99·0.02 = 0.0198; denominator = 0.0198 + 0.05·0.98 = 0.0688 → **≈ 0.288**. The classic "false-positive paradox."

---

## 30. Deep Dive (gated — optional)

<details>
<summary>Click to open derivation, log trick & variants</summary>

### From Bayes to naive

Start with the exact chain rule:

```text
P(x₁,…,x_d|c) = P(x₁|c)·P(x₂|x₁,c)·P(x₃|x₁,x₂,c)·…
```

Exact, but requires modeling all dependencies. The naive assumption replaces each conditional with the marginal:

```text
P(x₁,…,x_d|c) ≈ Πᵢ P(xᵢ|c)
```

Huge simplification — from exponential data needs down to per-feature estimates.

### Why logs, formally

A product of d probabilities each ~0.1 is ~10⁻ᵈ. For large d, this underflows to 0.0. Since `log` is strictly increasing, `argmax_c` is unchanged if we maximize sums of logs. Safe and exact-in-ordering.

### Variants in one table

| Variant | Data | Likelihood P(xᵢ\|c) |
|---|---|---|
| Gaussian | continuous | 1/√(2πσc²)·exp(−(x−μc)²/2σc²) |
| Multinomial | word counts | (count(xᵢ,c)+α)/(total_c+α·V) |
| Bernoulli | binary | P(present\|c), P(absent\|c) |

### MLE estimation

The parameters are just the empirical frequencies (Multinomial) or sample mean/variance (Gaussian). These are the maximum-likelihood estimates — Th no iteration needed, which is why training is instant.

### Calibration

Because of double-counting, posteriors are overconfident. Platt scaling (fit a logistic on the scores) or isotonic regression correct this for reliable probability estimates.

### Complexity

```text
training:  O(n·d)     prediction/sample: O(d·K)      stored: O(K·d) or O(K·V)
```

Linear in everything — one of the fastest classifiers that exists.

</details>

---

## 31. Teach Back

Try all four.

> **Explain in 30 seconds:** "Naive Bayes predicts a class by multiplying how common the class is with how well each clue fits it, then picks the class with the biggest score. It assumes clues are independent."

> **Explain to a 12-year-old:** "Ask every clue for its opinion — 'this smells like spam' or 'this smells like a real note' — then combine all the opinions and go with whichever side has more total belief."

> **Explain in an interview:** add: Bayes' theorem, conditional-independence assumption, log trick, smoothing, generative vs discriminative, calibration.

> **Explain the mathematics:** derive the product rule from Bayes and show why P(x) is dropped; show the log form preserves argmax.

---

## 32. Mastery Test

**Without looking at notes:**

1. State Bayes' theorem and define prior/likelihood/posterior.
2. Explain the naive assumption with the quiz-show story.
3. Compute a posterior by hand on tiny data (with smoothing).
4. Explain the zero-frequency problem and the smoothing fix.
5. Explain the log trick and why it's needed.
6. Give the three variants and their data types.
7. Explain why NB often works despite violating independence.
8. Discuss overconfidence and how to fix it.
9. Choose it for a real problem; defend the choice.
10. State one counter-example where you WOULDN'T use it.

---

## 33. Cheat Sheet

```text
Algorithm : Naive Bayes · Supervised → Classification · Generative
Core      : ŷ = argmax_c P(c)·Πᵢ P(xᵢ|c)   (use log-sum form)
Bayes     : P(c|x) ∝ P(c)·P(x|c)            (evidence P(x) dropped)
Assumption: conditional independence (the "naive" part)
Trick     : sum logs instead of multiplying  ·  add α (Laplace) to avoid 0
Variants  : Gaussian (continuous) · Multinomial (counts) · Bernoulli (binary)
Learn     : priors + per-class likelihood params (closed-form, no iteration)
Tune      : α (smoothing) · variant · class_prior · var_smoothing
Scaling   : NOT needed
Fails     : not zero-smooth → -inf; correlated dup features → overconfident
Use when  : text, small data, high-dim sparse, speed
Avoid when: strongly correlated features + need calibrated probabilities
Related   : Logistic Regression (discriminative twin) · Bayes decision theory
```

---

## 34. What Next?

You've met the probability-first classifier — the third distinct style of thinking.

```text
Naive Bayes
   ├── Decision Tree   (rules / if-then)   → next note (04)
   └── (after trees)   Random Forest, Boosted trees
```

> Next recommended: **04. Decision Tree (Classification)** — it replaces "combine probability votes" with "ask a sequence of if-then questions."
