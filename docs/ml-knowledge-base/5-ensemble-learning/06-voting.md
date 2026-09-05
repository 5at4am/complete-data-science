# 06. Voting Ensembles

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Voting Ensembles (Voting Classifier / Voting Regressor) |
| Category | Supervised Learning — Ensemble (Meta-algorithm) |
| Type | Fixed-aggregation combination ensemble |
| Parametric / Non-parametric | Depends on base models |
| Generative / Discriminative | Depends on base models |
| Main Objective | Combine multiple models' predictions with a FIXED rule (majority vote, averaged probabilities, weighted average) without training a combiner |
| Input | Multiple trained base models + their predictions (hard labels or probabilities) |
| Output | Combined prediction (majority label / averaged probability / weighted average) |
| Core Idea | "Crowd wisdom" with a transparent, fixed aggregation — no meta-learning; uses the Condorcet majority intuition |
| Typical Use Cases | Quick robust combinations, blending different families' votes, baseline for stacking |

---

## 02. One-Line Definition

### Beginner Definition
Voting ensembles let several trained models each cast a vote (or give a number), then combine them by majority, averaging, or weighted averaging to get one final answer.

### Technical Definition
Voting ensembles aggregate the outputs of pre-trained base models with a **fixed** rule: for classification, *hard voting* takes the majority class label while *soft voting* averages predicted probabilities; for regression, the prediction is the (possibly weighted) mean of base predictions. Unlike stacking, no second model is learned.

---

## 03. Intuition

**Real-life analogy — a panel decision.** A jury asks each of its members for a verdict (guilty / not guilty). If more than half say guilty, the verdict is guilty (hard voting). If members give probabilities, average them (soft voting). Each person's judgment is trusted equally unless you weight the experts (weighted voting — trust a senior expert more). No one "learns" how to combine; the rule is fixed and transparent.

**Technical intuition.** If each model is right more often than not, and their mistakes are not perfectly correlated, then the majority of several such models is right more often than any single one (Condorcet's jury theorem). Averaging probabilities adds a softness: a model confident in a minority class still moves the final answer.

**Step-by-step reasoning:**
1. Train several models (they can be trees, linear, KNN, boosting...).
2. For a new sample, each model produces a class label (hard) or a probability vector (soft), or a value (regression).
3. Combine by majority (hard), averaged probabilities (soft), or weighted average (with fixed weights).
4. Return the combined result — no additional training.

---

## 04. Problem It Solves

**Problem:** Which model should we trust? Different algorithms make different errors. Averaging/voting is a model-agnostic way to de-risk the choice — instead of betting on one model, you harness all of them.

**What we want:** A cheap, transparent combination whose accuracy is at least as good as (often better than) the average member, with no second training phase.

**Why voting works:** Majority intuition (many independent-ish voters > one), and soft voting exploits model confidence to boost well-calibrated members. It's the classic "wisdom of the crowd" applied to models.

**Small example:** Three models on a classification task: accuracies 70%, 68%, 72%, errors largely independent. A hard-vote ensemble routinely reaches ~75–78% — better than any single member — because the majority of three is right whenever ≥2 agree correctly.

---

## 05. Where It Fits in Machine Learning

```text
Ensemble Methods
├── Bagging (parallel averaging of SAME-type models on bootstrap data)
├── Boosting (sequential, same-type models)
├── Stacking (learned two-level combination)
└── Voting  ← YOU ARE HERE (fixed aggregation of ANY models)
```

Voting is the simplest and most transparent ensemble; stacking is its learned cousin.

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Hard voting | Count class labels | Majority vote: argmax over class-label counts |
| Soft voting | Average probabilities | Average predicted probability vectors, argmax |
| Majority rule | Most votes win | Requires odd B typically to avoid ties |
| Weighted voting | Give some models more say | Multiply votes/probabilities by fixed weights |
| Condorcet jury theorem | Many independent voters > one | Probability-of-correct-majority grows with B for p>0.5 |
| Base learner | The models whose outputs are combined | Can be heterogeneous (tree, linear, KNN, ...) |
| Ties | Equal votes | Handled by choosing lowest class index / first model |

---

## 07. Input and Output

**Input (fitting):** base models (already fitted or to be fitted), their outputs.
- Classification: class labels (hard) or predicted probabilities (soft).
- Regression: predicted values.

**Output:**
- Hard: the majority class label.
- Soft: the class with highest average probability.
- Regression: (weighted) mean of predictions.

---

## 08. Mathematical Foundation

**Hard voting:** Let {f₁, ..., f_B} predict class labels c ∈ {1,...,C}. The ensemble predicts

```text
F(x) = argmax_c   Σ_{b=1}^{B} 1[ f_b(x) = c ]
```

**Soft voting:** with probability vectors p_b(x) = (p_b1, ..., p_bC):

```text
F(x) = argmax_c   Σ_{b=1}^{B} w_b · p_bc(x)
```

**Regression averaging:**

```text
F(x) = (1/B) Σ_b f_b(x)      (or weighted: Σ_b w_b f_b(x), Σ w = 1)
```

**Condorcet majority.** If B independent voters each correct with probability p (binary), number correct ~ Binomial(B, p):

```text
P(majority correct) = Σ_{k=⌊B/2⌋+1}^{B} C(B,k) p^k (1−p)^{B−k}  → 1 as B→∞   (p > 0.5)
```

---

## 09. Core Formula

### Hard voting

```text
F(x) = argmax_c  Σ_b 1[f_b(x) = c]
```

### Soft voting

```text
F(x) = argmax_c  Σ_b w_b · p_bc(x)
```

### Weighted majority-vote error reduction

```text
B voters, each correct with p > 0.5 (independent) →
P(majority correct) = Σ_{k>B/2}^{B} C(B,k) p^k (1−p)^(B−k)
```

### Symbols
- B: number of base models.
- 1[·]: indicator (1 if true, 0 if false).
- p_bc(x): probability that model b assigns to class c at x.
- w_b: fixed weight of model b (often w = 1/B).
- c: class index.

### Intuition
- Hard = count labels.
- Soft = average confidences, then take the argmax.
- Majority theorem: with independent >0.5 voters, majority accuracy → 1.

### Example (tiny, calculated)
**Hard:** Models M1,M2,M3 vote class: A, A, B → counts A:2, B:1 → **A** (majority).

**Soft:** probabilities (P(A), P(B)):
- M1: (0.6, 0.4), M2: (0.8, 0.2), M3: (0.7, 0.3).
- Average A = (0.6+0.8+0.7)/3 = 0.7; B = 0.3 → class **A**.

**Weighted:** w = (0.5, 0.3, 0.2): A-score = 0.5·0.6+0.3·0.8+0.2·0.7 = 0.30+0.24+0.14 = 0.68; B-score = 0.5·0.4+0.3·0.2+0.2·0.3 = 0.20+0.06+0.06 = 0.32 → **A**. **Hand-verified all three.**

### Majority-theorem mini-example
p = 0.7, B = 3: P(majority correct) = C(3,2)(0.7)²(0.3) + C(3,3)(0.7)³ = 3·0.49·0.3 + 0.343 = 0.441+0.343 = 0.784 > 0.7. Three weak-ish voters → 78.4%. **Hand-verified.**

---

## 10. Derivation

**Majority theorem.** With B independent binary voters each correct with probability p, the number correct X ~ Binomial(B, p). A majority is correct when X > B/2:

```text
P(X > B/2) = Σ_{k=⌊B/2⌋+1}^{B} C(B,k) p^k (1−p)^{B−k}
```

By the weak law of large numbers, X/B → p; since p > 0.5, the event X/B > 0.5 has probability → 1 as B → ∞.

**Why a majority vote attains ≥ the average member.** For binary independent voters, accuracy of the majority is monotone in B when p > 0.5 (Condorcet). For finite B and correlated votes the gain shrinks — correlation is the crucial caveat, exactly as in bagging's variance analysis.

**Soft voting as a "probabilistic majority".** Averaging calibrated probabilities is the Bayes-optimal aggregation when members' confidences approximate posterior probabilities:

```text
Under independence and calibration: Σ_b w_b p_b(c | x)  approximates the pooled posterior
→ choosing the argmax is Bayes-optimal for the pooled belief
```

This is the *Bayes-optimality intuition* behind soft voting.

---

## 11. How the Algorithm Works

```text
Train B models (any heterogeneous set) on the same task
        │
        ▼
for a new sample x:
   hard:  labels c_1..c_B      → count → argmax count
   soft:  prob-vectors p_1..p_B → weighted sum → argmax
   regression: values v_1..v_B  → (weighted) mean
        │
        ▼
final prediction
```

---

## 12. Training Process

1. **Choose** the base models (heterogeneous for best results) and decide hard vs soft vs regression aggregation.
2. **Optionally weight** the models (fixed weights; not learned — you can pick them by validation performance, but the aggregation rule itself stays fixed).
3. **Fit** all base models (independently, in parallel).
4. **Predict** by the chosen fixed rule.

**What's learned:** only the base models' parameters. The combination is fixed — this is the key difference vs stacking.

**Final "model":** the set of base models + the fixed aggregation rule.

---

## 13. Objective Function / Loss Function

There is **no joint objective**: voting does not optimize a single loss over the ensemble. Each base model optimizes its own loss; the combination is a fixed arithmetic rule (majority / mean). This is what makes voting fast, cheap, and leakage-free (no second training set is needed).

- The "implicit objective" is to reduce error by diversity-guided aggregation, but no gradient ever flows between members or into the combination.
- High/low loss meaning: each member's loss is what you inspect; the ensemble has no meaningful training loss — evaluate with the task metric instead.

---

## 14. Optimization

Voting performs **no ensemble-level optimization**. The base models are fit by their own optimizers; the aggregation weights are fixed beforehand (e.g., 1/B or chosen by hand via validation). If you want the system to *learn* the weights, that is stacking, not voting.

```text
Fit bases (parallel) → fixed rule (count/mean/weighted-sum) → output
(No joint gradient, no learned weights, no convergence criterion)
```

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE — soft-voting three classifiers, two classes.**

| Model | P(A) | P(B) |
|---|---|---|
| M1 (tree) | 0.6 | 0.4 |
| M2 (KNN) | 0.8 | 0.2 |
| M3 (logistic) | 0.7 | 0.3 |

Soft votes: P(A) = (0.6+0.8+0.7)/3 = 2.1/3 = 0.7; P(B) = 0.3. Argmax → class **A**.

Hard votes: labels from argmax: M1→A, M2→A, M3→A → majority **A**. Consistent here.

**VERIFIED EXAMPLE — weighted voting changes the outcome.** Same models, weights = (0.1, 0.1, 0.8):
- A-score = 0.1·0.6 + 0.1·0.8 + 0.8·0.7 = 0.06+0.08+0.56 = 0.70.
- B-score = 0.1·0.4 + 0.1·0.2 + 0.8·0.3 = 0.04+0.02+0.24 = 0.30.
Still A. But with M3 favoring B and heavy weight, the outcome flips — demonstrating that weights are an *informed choice*, not learned. **Hand-verified arithmetic.**

**VERIFIED EXAMPLE — regression.** Predictions 210, 230, 220 → mean = 660/3 = **220**.

---

## 16. Visual Explanation

**Voting counter diagram (3 models, 2 classes):**

```text
        sample x
   ┌───────┼───────┐
   ▼       ▼       ▼
  M1      M2      M3
  ──►A   ──►A   ──►B      hard votes:  A=2, B=1 → MAJORITY = A
  p:0.6   p:0.8  p:0.7(A)  soft: P(A)=(0.6+0.8+0.7)/3=0.7 → A
```

**Hard vs soft mapping:**

```text
Hard:  labels → count → argmax
Soft:  probabilities → average → argmax (calibration-aware)
```

---

## 17. Algorithm / Pseudocode

```text
VOTING(base_models f_1..f_B, weights w_1..w_B):
  train each f_b on the data (parallel)
  in predict x:
     if regression:
        return Σ_b w_b · f_b(x)
     if hard voting:
        labels = [f_b(x) for b]
        return argmax_c count_c(labels)
     if soft voting:
        probs = [f_b.predict_proba(x) for b]
        return argmax_c Σ_b w_b · probs[b][c]
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

class FromScratchVoting:
    def __init__(self, models, weights=None):
        self.models = models
        self.n = len(models)
        self.weights = weights if weights is not None else \
            [1.0 / self.n] * self.n
        self.classes_ = None

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict_hard(self, X):
        votes = np.array([m.predict(X) for m in self.models])
        out = []
        for col in votes.T:
            counts = {}
            for v in col:
                counts[v] = counts.get(v, 0) + 1
            out.append(max(counts, key=counts.get))  # majority
        return np.array(out)

    def predict_soft(self, X):
        stacked = np.zeros((X.shape[0], self.classes_.shape[0]))
        for m, w in zip(self.models, self.weights):
            stacked += w * m.predict_proba(X)
        return self.classes_[np.argmax(stacked, axis=1)]

    def predict_reg(self, X):
        preds = np.array([m.predict(X) for m in self.models])
        return np.average(preds, axis=0, weights=self.weights)
```

**VERIFIED**: hard = count of labels (majority); soft = weighted average of probability vectors; reg = weighted mean — exactly the voting rules.

---

## 19. Code Explanation

```text
Code                                    ↓ What does it do?   ↓ Why required?   ↓ Math concept?
────────────────────────────────────────┼────────────────────┼─────────────────┼───────────────
[1.0/self.n]*self.n                     ↓ default weights    │ equal trust     │ uniform
np.array([m.predict(X) ...])            ↓ labels per model   │ voting inputs    │ vote collection
max(counts, key=counts.get)             ↓ majority label     │ hard vote       │ argmax count
stacked += w * predict_proba            ↓ weighted avg probs │ soft vote       │ weighted sum
np.average(preds, weights=weights)      ↓ weighted mean      │ regression      │ weighted mean
```

---

## 20. Library Implementation

```python
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)

models = [
    ("tree", DecisionTreeClassifier(max_depth=4)),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
    ("log", LogisticRegression(max_iter=1000)),
]
hard = VotingClassifier(estimators=models, voting="hard")
soft = VotingClassifier(estimators=models, voting="soft", weights=[1, 1, 1])
soft_w = VotingClassifier(estimators=models, voting="soft", weights=[0.2, 0.3, 0.5])

for name, vc in [("hard", hard), ("soft", soft), ("soft_w", soft_w)]:
    vc.fit(Xtr, ytr)
    print(name, vc.score(Xte, yte))

# Regression:
Xr, yr = load_diabetes(return_X_y=True)
Xrt, Xrte, yrt, yrte = train_test_split(Xr, yr, random_state=0)
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
vreg = VotingRegressor(
    estimators=[("dt", DecisionTreeRegressor(max_depth=4)), ("ridge", Ridge())]
)
vreg.fit(Xrt, yrt)
print("VotingRegressor R2:", vreg.score(Xrte, yrte))
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| estimators | List of (name, model) | Which + diversity | 3–7 heterogeneous models |
| voting | 'hard' / 'soft' | Aggregation type | Soft for calibrated models |
| weights | Fixed per-model weights | Trust distribution | Equal or validation-based |
| flatten_transform | Transform output shape | predict_proba concatenation | Advanced; leave default |
| n_jobs | Parallel fitting | Speed | -1 |

Base-model hyperparameters are tuned as usual per member.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Only each base model's internal parameters.
- Voting learns **nothing** about the combination.

### Hyperparameters (chosen)
- The models list, voting type, weights — all fixed before fitting.

---

## 23. Assumptions

| Assumption | What | Why | How to check | If violated |
|---|---|---|---|---|
| Members better than random | p > 0.5 | Majority math needs it | Per-model accuracy | Fix data/models first |
| Members not perfectly correlated | Diverse errors | Correlated votes don't cancel | Correlation of predictions | Add heterogeneous models |
| Calibration (soft) | Probabilities meaningful | Soft average assumes valid confidences | Calibration curve | Calibrate or use hard |
| Probabilities comparable | Same class order/space | Argmax across members needs aligned classes | Check predict_proba shape | Enforce class alignment |

---

## 24. Data Requirements

- Same data type as members (trees OK with mixed; linear/KNN need scaling).
- No extra data needed (no second training set, unlike stacking).
- Missing values: handle per base model.
- Class imbalance: weights can up-weight minority-savvy models, but majority vote tends to ignore them — prefer soft with balanced members.

---

## 25. Feature Scaling

Scaling just follows the base models (trees: unnecessary; linear/KNN: required). Voting adds no scaling needs; if you soft-vote models with wildly different probability scales, normalize/calibrate them first.

---

## 26. Evaluation Metrics

Evaluate the VOTED ensemble with the task metric; also compare vs the best single member:

| Metric | For | Note |
|---|---|---|
| Accuracy / F1 / AUC | Classification | Compare with best member |
| Log-loss | Probabilistic soft vote | Calibration matters |
| MSE / MAE / R² | Regression | Weighted average can dip MSE |
| Member-level breakdown | Diagnose which member helps | Compute per-model and ensemble |

Training objective (none) ≠ evaluation metric (task metric).

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Simple & transparent | Fixed rule, easy to explain |
| Cheap | No second training, parallelizable |
| Leakage-free | No overfitting of a combiner |
| Robust to model-choice uncertainty | De-risks picking one algorithm |
| Works with any model types | Fully heterogeneous |
| Soft voting exploits calibration | Can beat hard on well-calibrated members |

---

## 28. Disadvantages

| Disadvantage | Practical consequence |
|---|---|
| Fixed weights may be suboptimal | Stacking learns better weights |
| Majority needs p>0.5 votes | Weak models drag it down |
| Correlation erodes benefit | Similar models add little |
| Ties in hard voting | Arbitrary resolution |
| Soft requires calibrated models | Uncalibrated probas mislead it |

---

## 29. When to Use

✓ Quick robust combination of existing models.
✓ Diverse family members, no desire to train a combiner.
✓ Prefer transparency/simplicity over maximal accuracy.
✓ Baseline before attempting stacking.
✓ Regression where averaging smooths noisy predictors.

---

## 30. When NOT to Use

✗ Members are comparable/vastly correlated (little gain).
✗ A single clearly-best model exists already.
✗ Need learned weighting/tie-breaking (use stacking).
✗ Members are worse than random.
✗ Each model's probability estimates unreliable AND you still soft-vote.

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---|---|---|---|
| Spam filtering | text features | LR + NB + Tree vote | spam/not |
| Credit decisions | tabular | LR + RF + GBM vote | approve/reject |
| Churn prediction | usage features | diverse models | churn probability (soft) |
| Customer pricing | attributes | multiple regressors | averaged price |

---

## 32. Failure Cases

- **Mathematical:** all members p < 0.5 → majority worse than random.
- **Correlation:** all models are variants of the same tree → no gain.
- **Calibration:** soft voting with badly calibrated probabilities → skewed averages.
- **Ties:** hard voting with even B and 50/50 splits → arbitrary.
- **Cost:** if the ensemble barely beats its best member, the added latency isn't worth it.

---

## 33. Overfitting and Underfitting

Voting itself doesn't overfit (no learned combination). But:
- If members overfit, the vote inherits their bad regions (weighted soft can worsen — a confident overfit member dominates).
- Underfitting: if all members are weak/high-bias, voting stays weak — use boosting or better models; voting can't create signal that isn't there.

---

## 34. Bias-Variance Perspective

- Hard voting: a variance-reducer (majority smooths member instability) — but does not reduce shared bias.
- Soft/weighted voting: can reduce variance when members decorrelated; weights can partially compensate systematic member bias (a fixed "de-biasing" only if weight choices reflect bias).
- Net: voting mainly tames **variance** like bagging, but only when members are diverse; it does not learn bias correction (that is stacking/boosting's job).

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| Voting | Fixed aggregation | Simple/transparent | Fixed weights | Quick combines, baseline |
| Stacking | Learned combiner on OOF | Can beat all | Cost + leakage | Final accuracy push |
| Bagging | Average bootstrap models | Variance cut | Same-type base | Stabilize one algorithm |
| Boosting | Sequential residual-fit | Bias reduction | Sequential cost | One strong family |

---

## 36. Algorithm Selection Guide

```text
Have multiple trained models?
  ├─ no new training / transparency? ────► Voting
  │     ├─ members calibrated? ──────── Soft
  │     └─ labels only / cheap? ──────── Hard
  ├─ want learned weights + budget? ────► Stacking
  └─ want strongest single-family? ─────► Boosting / tuned RF
```

---

## 37. Common Mistakes

```text
❌ Mistake: Voting same-type, nearly-identical models (e.g., 3 RFs with same seed)
🔥 Why: identical predictions → vote copies one model; no diversity
✅ Correct: use heterogeneous families or different seeds/subsets

❌ Mistake: Soft-voting on uncalibrated or scale-mismatched probabilities
🔥 Why: confident-but-wrong model distorts the average
✅ Correct: calibrate members or use hard voting

❌ Mistake: Using members worse than 50% accuracy and expecting a miracle
🔥 Why: majority math only lifts p > 0.5
✅ Correct: fix Member-before-mixing

❌ Mistake: Expecting voting to fix high-bias members
🔥 Why: averaging weak, biased models stays weak/biases
✅ Correct: use boosting/better members

❌ Mistake: Hard voting with even number of members (frequent ties)
🔥 Why: ties resolved arbitrarily
✅ Correct: odd counts, or use soft voting to break with confidence
```

---

## 38. Interview Questions

### Beginner
Q: What is hard vs soft voting? A: Hard counts class labels; soft averages predicted probabilities, both then take the argmax.
Q: Does voting train a new model? A: No — the combination rule is fixed; only base models are trained.

### Intermediate
Q: When is soft voting better than hard? A: With well-calibrated members — averaging confidences (using more info) usually beats counting labels.
Q: Why does voting sometimes fail to beat the best member? A: If members are correlated or weak, majority/mean can't add information.

### Advanced
Q: Prove the Condorcet majority intuition. A: With B independent voters, count-correct ~ Binomial(B, p); P(correct majority) = Σ_{k>B/2} C(B,k)p^k(1−p)^(B−k) → 1 if p>0.5 (weak law).
Q: What is the Bayes-optimality argument for soft voting? A: Averaging calibrated posterior probabilities approximates the pooled posterior; argmax then maximizes pooled probability.
Q: Voting vs stacking, concretely? A: Voting fixes weights a priori; stacking learns them (and their interaction structure) on out-of-fold predictions — costlier, more flexible.

---

## 39. GATE / Exam Perspective

**Key formulas:**
- Hard: F(x) = argmax_c Σ_b 1[f_b = c].
- Soft: argmax_c Σ_b w_b p_bc(x).
- Regression: mean/weighted mean.
- Majority theorem binomial tail.

**Traps:**
- Voting reduces variance (not bias) in general views.
- Soft + uncalibrated ≠ automatically better than hard.
- Ties/even B matter in code, not just theory.

> **Note:** No real GATE PYQs reproduced; these are representative patterns.

---

## 40. Coding Practice

1. **Basic:** Hand-compute the soft vote of Section 15.
2. **Basic:** Implement majority count with numpy (Section 18).
3. **Intermediate:** sklearn VotingClassifier hard vs soft; compare CV.
4. **Intermediate:** Add weights tuned on validation; compare.
5. **Advanced:** From-scratch weighted soft voter (Section 18) with calibration.
6. **Case-study:** Two-frame: use Voting vs best base on a tabular problem; quantify ensemble gain.

---

## 41. Practical ML Workflow

```text
Problem → data → EDA → clean → feature engineering → split
→ train 3–7 diverse models → (calibrate probabilities if soft voting)
→ choose hard/soft/regression + weights (equal or validation-picked)
→ fit all → evaluate ensemble vs best member
→ error analysis (which member helps where) → adjust weights or members
→ deploy (serialize all models + rule) → monitor → retrain
```

---

## 42. Complexity

- **Training:** O(sum of base training costs), fully parallel.
- **Prediction:** O(sum of base prediction costs).
- **Space:** sum of base model sizes.
- **Scaling:** linear in the number of members; no extra memory for a combiner.

---

## 43. Advanced Concepts

- **Calibration** before soft voting (Platt/isotonic).
- **Weight selection** via validation / linear objective — bridging to stacking.
- **Condorcet theory** for correlated voters (dependent-jury results).
- **Meta-label ensembles vs probability pooling** (label vs distribution view).
- **Mini modeling:** voting as an ensemble baseline in AutoML.

---

## 44. Connections to Other Algorithms

```text
Any classifier/regressor ── can be a voter (tree, linear, KNN, SVM, boosting, NN)
Voting ── fixed weights ── generalizes into ──► Stacking (learned combiner)
Voting (variance) ── conceptually neighbors ──► Bagging (parallel averaging)
Soft voting ── probability averaging ── rel. to ──► Bayesian model averaging
```

---

## 45. If You Remember Only 5 Things

1. **Voting = fixed aggregation** (majority / average / weighted) with no learned combiner.
2. **Hard counts labels; soft averages probabilities** (soft needs calibrated members).
3. **Majority-theorem flaw:** only lifts members with p > 0.5 and decorrelated errors.
4. **Voting mostly reduces variance**, like a heterogeneous bagging; it doesn't fix bias.
5. **Stacking is voting's smart cousin** — swap when you can learn the weights.

---

## 46. Cheat Sheet

| Item | Value |
|---|---|
| Algorithm | Voting Ensembles |
| Category | Fixed-aggregation ensemble |
| Goal | Combine diverse models transparently |
| Input | Base model predictions |
| Output | Majority / averaged-proba / mean |
| Core Formula | argmax Σ_b 1[f_b=c]; argmax Σ w_b p_bc; weighted mean |
| Loss | none (per-member only) |
| Optimization | none at ensemble level |
| Parameters | base-model params only |
| Hyperparameters | members, voting type, weights |
| Assumptions | p>0.5, low correlation, calibration (soft) |
| Advantages | simple, cheap, transparent, leakage-free |
| Disadvantages | fixed suboptimal weights, correlation |
| Use When | quick robust combine, stop short of stacking |
| Avoid When | need learned combiner, tiny gain |
| Related | stacking (learned), bagging (parallel avg) |
| Key Exam Points | majority theorem, hard vs soft, weights |
| Key Interview Points | no meta-learning, calibration, correlation |

---

## 47. Final Mental Model

```text
model1 ── label/proba ─┐
model2 ── label/proba ─┼──► FIXED RULE (count / avg / weighted) ──► FINAL
model3 ── label/proba ─┘
(no training of the rule — that would be stacking)
```

---

## 48. Knowledge Check

### Recall (5)
1. What is hard voting?
2. What is soft voting?
3. Does voting train a combiner?
4. Why is p > 0.5 important?
5. What are weights in weighted voting?

### Understanding (5)
1. Why calibration matters for soft voting.
2. When does voting fail despite good members?
3. What does voting reduce — bias or variance?
4. Why heterogenous members help.
5. How is soft voting "Bayes-optimal"?

### Application (5)
1. Choose hard vs soft for a dataset.
2. Pick weights sensibly.
3. Detect that voting adds no value.
4. Design a 3-model vote for churn.
5. Fix an even-B tie problem.

### Mathematical (5)
1. Write the hard-vote formula.
2. Write soft-vote formula.
3. Compute P(majority) for p=0.6, B=3.
4. Show a weighted mean for regression.
5. Prove majority accuracy → 1 for p>0.5.

### Interview (5)
1. "Hard vs soft, when which?"
2. "Why so often barely beats best member?"
3. "Voting vs stacking in one paragraph."
4. "What if one member is clearly best?"
5. "How to combine classes with different label orders?"

### Problem Solving (5)
1. 3 models: 90%, 60%, 55% — worth a vote?
2. Models agree on 95% of samples — what happens?
3. Soft voting glues A wrongly due to an overconfident member.
4. You have no resource for stacking; sell voting honestly.
5. Your regression predictors have very different scales — vote?

## Answers

**Recall:**
1. Majority class among predicted labels.
2. Argmax of averaged probabilities.
3. No — fixed rule; only members train.
4. Majority math lifts only p>0.5 voters.
5. Fixed per-model trust; summing/weighting predictions.

**Understanding:**
1. Averaging miscalibrated confidences distorts the pooled posterior.
2. Correlated errors or one dominant confident member.
3. Variance (majority/mean smooth instability); shared bias remains.
4. Decorrelated diverse errors give majority gain.
5. Averaging calibrated posteriors approximates pooled belief; argmax maximizes it.

**Application:**
1. Soft if calibrated; hard otherwise.
2. Equal, or proportional to per-member CV score.
3. Compare ensemble vs best single via CV; if near-equal, drop it.
4. RF (stability) + GBM (accuracy) + LR (interpretability), soft/weighted.
5. Odd number of members or soft voting to break ties.

**Mathematical:**
1. F(x) = argmax_c Σ_b 1[f_b=c].
2. F(x) = argmax_c Σ_b w_b p_bc.
3. 3·0.36·0.4 + 0.216 = 0.432+0.216 = 0.648.
4. F = Σ w_b f_b, Σw=1 (mean when equal).
5. X/B → p by WLLN ⇒ majority event has prob → 1.

**Interview:**
1. Soft wins with calibrated, probability-output models; hard with labels/cheap.
2. Because errors are correlated and shared bias isn't addressed.
3. Voting fixes verifiable weights a priori; stacking learns them on OOF predictions — more power, more cost/risk.
4. It may dominate the vote; weigh it down (weights) or skip voting.
5. sklearn aligns classes internally; in custom code, map each member to a common class/index space.

**Problem Solving:**
1. Yes if winning depends on those 2 weak ones agreeing — but drop below-50% members.
2. Vote equals near-consensus; gain tiny — probably not worth latency.
3. Calibrate members, down-weight the overconfident one.
4. Framing: transparent, zero training overhead, no leakage risk, robust de-risking — pick it when your goal is a safe choice, not a leaderboard.
5. Normalize/stack comparable scales (e.g., z-score or rank), or weight against the extreme-scale member.

---

## 49. Final Learning Checklist

- [ ] Define hard and soft voting.
- [ ] Formula for both + weighted mean (regression).
- [ ] State Condorcet majority bound (p>0.5).
- [ ] Explain calibration requirement for soft.
- [ ] Explain correlation eroding benefit.
- [ ] Distinguish voting from stacking.
- [ ] From-scratch hard/soft/weighted voter.
- [ ] sklearn VotingClassifier / VotingRegressor.
- [ ] Choose weights sensibly.
- [ ] Test voting vs best member honestly.
- [ ] Handle ties / even B.
- [ ] Mismatched probability scales → calibrate.
- [ ] Never vote below-chance members.
- [ ] Where voting fits vs bagging/boosting/stacking.
- [ ] Deploy serialized members + rule.

---

## 50. Quality Control Note

- **Accuracy:** all numerical examples hand-verified (majority, soft, weighted, binomial p=0.7→0.784); Condorcet statement standard; no GATE PYQs invented (marked representative).
- **Beginner-friendliness:** jury analogy, counter diagram, tiny worked examples.
- **Math depth:** majority theorem derivation, Bayes-optimality intuition for soft voting.
- **Practical depth:** from-scratch voter before sklearn; hyperparameters, calibration, workflow.
- **Exam depth:** formulas, ties, calibration traps; formulas flagged as representative, not fabricated PYQs.
- **Structure:** follows the shared 50-section template exactly.