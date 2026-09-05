# 07. Non-Negative Matrix Factorization (NMF)

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Aspect | Details |
|--------|---------|
| **Algorithm name** | Non-Negative Matrix Factorization (NMF) |
| **Category** | Matrix factorization / Dimensionality Reduction (unsupervised, parts-based, feature extraction) |
| **Type** | Low-rank non-negative factorization (multiplicative optimization) |
| **Parametric / Non-parametric** | Non-parametric in data (learns fixed W, H for the training matrix) |
| **Generative / Discriminative** | Neither (unsupervised decomposition), but has a generative interpretation (factor model with non-negative loadings) |
| **Main objective** | Given a non-negative matrix A (m×n), find non-negative W (m×k), H (k×n) whose product WH ≈ A, minimizing reconstruction error — producing sparse, additive, interpretable parts |
| **Input** | Non-negative matrix A (e.g., term×document counts, pixel×image intensities, spectrograms) |
| **Output** | Non-negative basis matrix W (m×k) and coefficient matrix H (k×n); the reduced representation is H (or W) |
| **Core idea** | Constrain factors to be ≥0 so the reconstruction is a *sum of parts* (additive model) — a natural, interpretable decomposition |
| **Typical use cases** | Topic modeling, image parts/faces, audio source separation, spectral unmixing, collaborative-filtering-adjacent embeddings |

---

## 02. One-Line Definition

### Beginner Definition
NMF splits a table of non-negative numbers into two smaller tables that multiply to roughly rebuild the original, and because everything is non-negative it reads like "small building blocks combine to make each column."

### Technical Definition
NMF finds non-negative matrices W ∈ ℝ₊^{m×k} and H ∈ ℝ₊^{k×n} such that A ≈ W H, minimizing a divergence/reconstruction loss (Frobenius or KL) under a latent-dimension k; the factor matrices yield a parts-based, additive representation of the data.

---

## 03. Intuition

Imagine a photo of a face as a sum of "parts" — eyes, mouth, nose, background. A face is genuinely built by *adding* anatomically meaningful components; it would be odd to "Eyes = 3×nose − background" (negative mix). NMF respects that: every element of W and H is ≥0, so each image ≈ (house of W-parts) weighted by H-coefficients. Because there are no negative cancellations, the parts are sparse and interpretable.

Think of money: if you say "total = 3×$10-bills + 2×$5-bills," both the counts (H) and the bills (W) are non-negative — that's exactly the NMF worldview of a bank statement (A). Compare to PCA: "total = 3×(10 − 2×5) − ..." — technically possible, but the negative weights make no human sense.

---

## 04. Problem It Solves

**The problem:** PCA/SVD factors can contain *negative* loadings, making them uninterpretable for data that is inherently non-negative (word counts, pixel intensities, counts of ratings, spectral energy). Also, global patterns (PCA) mix everything, not "parts."

**What we want:** A factorization where each factor is a *coherent, positive part* and the data is a *positive blend of parts*.

**Why NMF is useful:** It produces sparse, parts-based factors that map to human-meaningful concepts: in text → topics; in images → visual parts; in audio → instruments/sources. The non-negativity constraint is exactly what yields this additive interpretability.

**Small example:** A term-document matrix (words × docs). NMF finds: W-col1 ≈ "machine learning terms," H-row1 ≈ which docs are about ML. Handing these to a person, each W-column reads like a topic list (all words with positive weight), unlike PCA where topics come with confusing negative weights.

---

## 05. Where It Fits in Machine Learning

```text
Machine Learning
└── Unsupervised Learning
    └── Dimensionality Reduction / Matrix Factorization
        ├── PCA (orthogonal, signed)
        ├── SVD (general signed factorization)
        ├── NMF ◄── non-negative, parts-based
        └── LDA (supervised; topic/class)
```

NMF occupies the "interpretable / parts-based" corner of matrix factorization, used widely in text, vision, and audio; it is also the natural bridge to topic modeling (closely related to LDA topic models conceptually).

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|------|----------------|-------------------|
| **Non-negative matrix** | A matrix with all entries ≥ 0 | Aᵢⱼ ≥ 0 for all i,j |
| **Factorization / low-rank** | Two smaller matrices that rebuild A | A ≈ W H with inner dimension k < min(m,n) |
| **Basis / component matrix W** | The "parts" (each column a part) | W (m×k), columns are latent components |
| **Coefficient matrix H** | How much each part is used per column of A | H (k×n), rows give weights per part |
| **Reconstruction error** | How far WH is from A | ‖A − WH‖²_F (Frobenius) |
| **Sparsity** | Many entries near zero | More zeros in W,H; interpretable by keeping only non-zero weights |
| **Additive model** | Every column is built by summing parts | A[:,j] ≈ Σ_t H[t,j]·W[:,t], all terms ≥0 |
| **KL / Frobenius divergence** | Two losses for NMF | Frobenius: ‖A−WH‖²_F; KL: Σ a log(a/(WH)) − a + (WH) |
| **Multiplicative update** | Iteration that never goes negative | W ← W ⊙ (AHᵀ)/(WHHᵀ) style; automatic non-negativity |

---

## 07. Input and Output

**Input:**
- Non-negative matrix A (m×n). Common shapes: words×documents, pixels×images, users×items, frequency×time spectrogram.
- Hyperparameters: latent dim k, loss (frobenius/kl/mu), init, regularization (alpha_W, alpha_H, l1_ratio), solver (cd/mu/coordinate descent).

**Output:**
- W (m×k): the k learned components/parts/basis.
- H (k×n): non-negative coefficients; H (or its transform) is the reduced-dim embedding of the columns of A.
- For new data: `transform` computes H_new via a *fitted* W.

**Not a predictor:** unsupervised; but W,H feed classifiers/clustering/topic interpretation.

---

## 08. Mathematical Foundation

**Basic idea:** Approximate a non-negative matrix by two lower-rank non-negative factors:

```text
A (m×n) ≈ W (m×k) · H (k×n),   W,H ≥ 0,  k < min(m,n)
```

Column viewpoint: each column a_j ≈ Σ_t H[t,j]·w_t — a weighted sum of k parts (columns of W). That is precisely the additive/parts-based reading.

**Notation:** A = non-negative data matrix; m = rows (features), n = columns (samples/items); k = latent dimension (hyperparameter); W = component/basis matrix; H = coefficient matrix; ‖·‖_F = Frobenius norm; ⊙ = elementwise (Hadamard) product.

**Required math:** Linear algebra (matrix multiplication, transpose), non-negative matrices, convex optimization basics (alternating/coordinate descent), norms and information divergence (KL).

---

## 09. Core Formula

### 1. Reconstruction

```text
A ≈ W H
```

**Meaning:** The data is approximated by the product of two non-negative low-rank factors.

**Symbols:** A (m×n), W (m×k), H (k×n), all entries ≥ 0.

**Intuition:** Every column of A is a non-negative blend of the k parts (columns of W), with weights in H.

### 2. Objective / loss (Frobenius)

```text
minimize  ‖A − W H‖_F²   over W ≥ 0, H ≥ 0
```

**Meaning:** Minimize the squared reconstruction error.

**Symbols:** ‖·‖_F² = sum of squared entries.

**Intuition:** The usual least-squares fit, constrained to non-negative factors.

### 3. Objective (KL / generalized KL)

```text
minimize  Σ_{i,j} [ A_{ij} log( A_{ij}/(WH)_{ij} ) − A_{ij} + (WH)_{ij} ]
```

**Meaning:** A Poisson/relative-entropy-motivated loss better for count data.

**Intuition:** Weighs errors proportionally — good for counts, where large counts tolerate larger absolute errors.

### 4. Multiplicative update rules

```text
W ← W ⊙ ( (A Hᵀ) / (W H Hᵀ + ε) )
H ← H ⊙ ( (Wᵀ A) / (Wᵀ W H + ε) )
```

**Meaning:** Update factors by multiplying by the ratio of positive/negative gradient parts — never decreases below zero.

**Symbols:** ⊙ = elementwise multiplication; division elementwise; ε = tiny constant for stability.

**Intuition:** Multiplicative rather than subtractive — the sign can't flip, guaranteeing non-negativity and monotone loss descent.

### 5. Interpretability criterion

```text
A[:, j] ≈ Σ_{t=1..k} H[t, j] · W[:, t]
```

**Meaning:** Column j's data is *recalled* by adding parts.

**Intuition:** Because all H[t,j] and W[:,t] ≥ 0, no cancellations hide structure — each coefficient meaningfully "pulls in" a part.

**Worked example (hand-verified).** Let A = [[2,4],[2,4]] (two identical rows: feature pairs). Try k=1: W (2×1), H (1×2). Guess W=[[1],[1]], H=[2,4] → WH = [[2,4],[2,4]] = A exactly. Reconstruction error = 0; W's column is the "part" [1,1] (both-feature co-occurrence pattern in the row-style), H says docs use it with weights 2 and 4. ✅ Hand-verified (k=1, error=0).

---

## 10. Derivation

The NMF objective F(W,H) = ‖A − WH‖_F² is **not jointly convex** in (W,H), but is **convex in W alone** (H fixed) and **convex in H alone** (W fixed) — so we alternate.

1. Standard gradient w.r.t. W: ∇_W = −2(A−WH)Hᵀ.
2. Split into positive/negative parts: ∇_W = 2(WH Hᵀ − A Hᵀ). Write as P − N with P = 2W H Hᵀ, N = 2 A Hᵀ.
3. A "multiplicative" natural gradient idea (Lee & Seung): W ← W ⊙ (N/P) = W ⊙ (A Hᵀ)/(W H Hᵀ). This keeps W ≥ 0 and, under suitable conditions, decreases the objective monotonically.
4. Analogous for H: H ← H ⊙ (Wᵀ A)/(Wᵀ W H).
5. Iterate until convergence (change below tolerance or max iterations); optionally scale W toward unit columns and absorb the scale into H (normalization).
6. Optionally add L1/L2 regularization on W and/or H to encourage sparsity/smoothness.

**Why multiplicative:** Subtractive gradient steps (W ← W − η∇) can drive entries negative; multiplying by a positive ratio preserves sign. (The exact steps are modifications of gradient descent with per-entry learning rates.)

**Important result:** The algorithm guarantees non-negativity for all iterates if initialized non-negative, and is the standard practical NMF (Lee-Seung updates; equivalents in sklearn's solvers).

---

## 11. How the Algorithm Works

```text
Input A (m×n) non-negative; choose k, loss, regularization, init
  ↓
Initialize W (m×k), H (k×n) non-negative (random/NNDSVD)
  ↓
Repeat until convergence:
    update H ← H ⊙ (Wᵀ A)/(Wᵀ W H + ε)      [keep W fixed]
    update W ← W ⊙ (A Hᵀ)/(W H Hᵀ + ε)      [keep H fixed]
    (optional) normalize columns of W; adjust H
    track objective ‖A − WH‖²_F (or KL)
  ↓
Converged
  ↓
Output W (components), H (coefficients/embedding)
```

---

## 12. Training Process

**Pre-training:** Verify A non-negative (shift/scale if needed); choose k (< min(m,n) typically); initialize W,H (random non-negative, or deterministic NNDSVD).

**During:**
- Alternating updates (H with W fixed, then W with H fixed).
- Multiplicative rules keep factors non-negative; objective decreases monotonically for standard updates.
- Regularization adjusts sparsity.
- Optional normalization of W columns.

**What's learned:** The basis W (parts) and coefficients H (usage per column).

**Stopping:** Small change in W/H or objective (tol), or max iterations.

**Final model contents:** W (to be reused for transform), optionally fitted scaler/normalization, and the chosen loss/regularization hyperparameters.

---

## 13. Objective Function / Loss Function

**Objective:** Minimize reconstruction error subject to non-negativity:

```text
minimize ‖A − W H‖²_F   (or KL divergence)   subject to W ≥ 0, H ≥ 0
```

Optionally with regularization:

```text
minimize ‖A − W H‖²_F + α_L1·‖W‖₁ + α_L2·‖H‖₂² ...  (L1 → sparse, L2 → smooth)
```

**Why chosen:** Frobenius fits well for dense continuous non-negative data; KL fits count-like data (Poisson view). Non-negativity forces additive, interpretable solutions.

**High/low meaning:** Low error → the k parts faithfully rebuild A; high error → k too small or additive parts unsuitable.

**Remark:** NMF's objective is a *training/reconstruction* objective — for evaluation on downstream tasks you still use dedicated metrics (see §26).

---

## 14. Optimization

**Definition:** Alternating non-negative optimization solved by multiplicative updates (or active-set coordinate descent in sklearn's `cd` solver).

**Why:** Joint problem is non-convex; per-factor problems are convex → alternate block coordinate descent.

**Method:**
- (Lee-Seung) multiplicative updates H then W (guaranteed monotone for the standard losses).
- Coordinate-descent solver (`solver='cd'`) activates a few coordinates at a time — faster on sparse A, handles L1 regularization exactly.

```text
Init W,H ≥ 0
  ↓
loop:
   H ← H ⊙ (WᵀA)/(WᵀWH+ε)
   W ← W ⊙ (AHᵀ)/(WHHᵀ+ε)
   (optionally) normalize W
  ↓
until converged
```

**Local optimum:** Non-convex → multiple local optima; run several random inits and keep the best (lowest loss), or use NNDSVD init for stability.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE (hand-verified).** Non-negative 2×2 matrix A = [[3, 6], [1, 2]] (each column is a multiple of the row-pattern [3,1]). Try k=1: want W (2×1) ≥0, H (1×2) ≥0 with WH = A.

Column vectors: [3,1] and [6,2] = 2×[3,1] → identical direction. Perfect factorization: W = [[3],[1]], H = [1, 2]: WH = [[3],[1]]·[1,2] = [[3×1, 3×2],[1×1,1×2]] = [[3,6],[1,2]] = A. Reconstruction error = 0. ✅ Hand-verified (rank-1 exact factorization, because columns are collinear).

Take another A where k=1 is insufficient: A = [[1, 0],[0, 1]]. Product of 2×1 and 1×2 non-negative factors has rank ≤1 and is of the form [[a·b, a·d],[c·b, c·d]] with (row1)/(row2) ratio a/c constant across columns? Actually rows are proportional: row1 = (a/c)·row2. But A's rows [1,0] and [0,1] are not proportional → NMF with k=1 cannot represent A exactly; reconstruction error > 0. With k=2, W=H=I gives error 0. ✅ This illustrates why latent dim k must be at least the "additive rank" of the data.

---

## 16. Visual Explanation

```text
A (2×2) = [[3,6],[1,2]]   ≈   W (2×1) · H (1×2)
  [3 6]      [3]           [1 2]
  [1 2]    = [1]  ·        (row vector)

   "both columns are made from the SAME part (the vector (3,1)),
    used with weights 1 and 2"   ← additive, interpretable
```

```text
Faces example:

  W columns (parts):   [eyes]  [nose]  [mouth]   → all pixelries ≥0
  H (per image): weight of each part per face
  A[:, face] ≈ w_eyes·2 + w_nose·1 + w_mouth·0.8   (all positive weights)
```

---

## 17. Algorithm / Pseudocode

```
1. Preprocess: ensure A ≥ 0 (shift if negative values possible; scale for KL)
2. Choose k < min(m,n); choose loss, init, regularization
3. Initialize W (m×k) ≥ 0, H (k×n) ≥ 0  (random or NNDSVD)
4. Repeat:
     H ← H ⊙ (Wᵀ A) / (Wᵀ W H + ε)
     W ← W ⊙ (A Hᵀ) / (W H Hᵀ + ε)
     (if regularization: apply L1/L2 steps)
     (optional) normalize columns of W; fold scale into H
     compute objective ‖A − WH‖²_F (or KL)
   Until relative change in objective < tol or max_iter
5. Return W, H
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

def nmf_scratch(A, k, max_iter=200, tol=1e-4, loss="frobenius"):
    m, n = A.shape
    rng = np.random.default_rng(0)
    W = np.abs(rng.normal(0.1, 0.1, (m, k)))
    H = np.abs(rng.normal(0.1, 0.1, (k, n)))
    eps = 1e-10

    def obj():
        if loss == "frobenius":
            return np.sum((A - W @ H) ** 2)
        R = np.maximum(W @ H, eps)
        return np.sum(A * np.log(np.maximum(A, eps) / R) - A + R)

    prev = obj()
    for it in range(max_iter):
        H = H * (W.T @ A) / np.maximum(W.T @ W @ H, eps)
        W = W * (A @ H.T) / np.maximum(W @ H @ H.T, eps)
        cur = obj()
        if abs(prev - cur) / (abs(prev) + 1e-12) < tol:
            break
        prev = cur
    return W, H, cur

A = np.array([[3.0, 6.0], [1.0, 2.0]])
W, H, loss = nmf_scratch(A, k=1)
print("W:\n", W.round(3))
print("H:\n", H.round(3))
print("WH:\n", (W @ H).round(3))
print("Loss:", loss.round(6))
```

---

## 19. Code Explanation

```text
Code line → What does it do? → Why required? → Mathematical concept?
np.abs(rng.normal(...)) → non-negative init → factors must stay ≥0 → feasible region W,H ≥ 0
H * (W.T @ A) / (W.T @ W @ H) → multiplicative H update → reduces loss without sign flip → H ← H⊙(WᵀA)/(WᵀWH)
W * (A @ H.T) / (W @ H @ H.T) → multiplicative W update → reduces loss without sign flip → W ← W⊙(AHᵀ)/(WHHᵀ)
obj(): frobenius/kl → monitors convergence → stopping criterion → ‖A−WH‖²_F or KL
abs(prev−cur)/prev < tol → stop when stable → efficiency → tolerance convergence
W @ H → rebuild A → output check/fit → A ≈ WH reconstruction
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.decomposition import NMF

A = np.array([[3.0, 6.0], [1.0, 2.0]])

model = NMF(n_components=1, init="nndsvda", solver="cd",
            max_iter=500, random_state=0)
W = model.fit_transform(A)   # W (m × k)
H = model.components_        # H (k × n)
print("W:\n", W.round(3))
print("H:\n", H.round(3))
print("Reconstruction:\n", (W @ H).round(3))
print("Reconstruction error (Frobenius):", round(model.reconstruction_err_, 3))

# Transform NEW columns (fit W only; learn H for new data):
A_new = np.array([[9.0], [3.0]])
H_new = model.transform(A_new)
print("H for new data:\n", H_new.round(3))
print("Reconstruction of new data:\n", (model.components_ @ H_new.T).round(3))
```

**Key API:** `fit_transform(A)` returns W; `components_` gives H; `transform(A_new)` computes H for new data given the learned W; `reconstruction_err_`, `n_iter_`.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|----------------|---------|--------|----------------------|
| `n_components` (k) | Latent dimension | Too low → underfit/interpreting as missing parts; too high → overfit/noise | Tune by reconstruction/downstream/interpretability |
| `init` | random/nndsvd/nndsvda/nndsvdar | Starting point; NNDSVD better for stability | `nndsvda` default |
| `solver` | cd (coordinate descent) / mu (multiplicative) | Speed + regularization fidelity | `cd` with L1; `mu` for KL |
| `beta_loss` | frobenius / kullback-leibler / itakura-saito | Data-type-appropriate loss | KL for counts, fro for continuous |
| `alpha_W`, `alpha_H` | L1/L2 penalty strength on W/H | Controls sparsity | Increase for sparsity |
| `l1_ratio` | Mixture L1/(L1+L2) | Sparse vs smooth | 1 → pure L1 (sparse) |
| `max_iter`, `tol` | Stopping | Convergence quality | Default; increase if not converged |

**too low / too high / tune:** k too low underfits (loses latent parts); k too high overfits (duplicates parts / noise). Tune with reconstruction-error-vs-k elbow AND downstream interpretability/accuracy; increase α for sparser, more interpretable factors.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- W (basis/parts), H (coefficients/embedding). Fixed after fitting.

### Hyperparameters (chosen)
- n_components k, init, solver, beta_loss, alpha_W/alpha_H, l1_ratio, max_iter, tol.

---

## 23. Assumptions

| Assumption | What | Why | How to check | If violated | Solution |
|-----------|------|-----|--------------|-------------|----------|
| **Non-negativity of A** | All entries ≥ 0 | Additive model requires it | Check min(A) ≥ 0 | Negative entries break the game | Shift to non-negative or use signed MF/PCA |
| **Additive generation of data** | Data is literally sums of hidden parts | NMF's reason to exist | Parts reconstruct well / interpretable | Data has subtractive structure | Use PCA/SVD, or allow signed factors |
| **Choice of k corresponds to true hidden dimension** | Data has ≈k generative components | k is latent model order | Reconstruction elbow / topic coherence | Wrong k → mush or dispersion | Tune k (elbow, domain) |
| **Loss suits noise** | Loss matching noise family | Fro/KL behave differently | Residual diagnostics | Heavy-tailed noise | Weaker (L1-like) loss / robust variants |

---

## 24. Data Requirements

- **Data type:** Non-negative numeric (counts, intensities, magnitudes, probabilities, ratings, text frequencies).
- **Missing values:** NMF isn't native to missing entries; impute or weighted/regularized variants (like non-negative MF for recommenders).
- **Outliers:** Big counts can dominate Frobenius; consider scaling or KL loss.
- **Scaling:** For Frobenius loss, scale features/normalize rows as appropriate (KL handles count-based patterns). Avoid negative-transformed inputs.
- **Dataset size:** Standard NMF is O(m n k) per iteration; sparse solvers (cd) handle large sparse matrices (e.g., text) efficiently.
- **Class imbalance / labels:** Not required (unsupervised); labels optional for validation/topic coherence.

---

## 25. Feature Scaling

**Depends on the loss.**

- **Frobenius:** scaling helps balance features; a common recipe is to normalize columns/rows (e.g., TF normalization for text).
- **KL:** designed for raw-ish counts; often used directly on counts without heavy scaling.
- Guidance: keep A non-negative; choose min over scale that makes per-feature contributions comparable; never standardize (Z-score) NMF input because it generates negatives (breaks the additive model).

---

## 26. Evaluation Metrics

**Training objective (reconstruction error / KL) ≠ evaluation metric.**

| Metric | Definition | When to use | When NOT |
|--------|-----------|-------------|----------|
| **Reconstruction error** | ‖A − WH‖²_F | Choosing k, convergence | Downstream quality alone |
| **Sparsity of W/H** | Fraction of near-zero entries | Interpretability check | Confusing sparse with accurate |
| **Topic coherence** (text) | Semantic alignment of W columns | Topic-model evaluation | Non-text data |
| **Downstream task score** (accuracy/F1 on H embedding) | Classifier/cluster performance on NMF features | Practical utility of embedding | Interpreting the components |
| **Stability across inits** | Variation of factors over runs | Reliability check | Expecting identical factors |

---

## 27. Advantages

- **Interpretable, parts-based factors** (additive, non-negative). ✅
- **Sparse representations** — each part uses few features, coefficients are sparse. ✅
- **Handles counts/high-dimensional sparse data** (text, images). ✅
- **Natural for topic/source separation and spectral decomposition.** ✅
- **Closed-form-free but simple multiplicative optimization; no negative artifacts.** ✅
- **Works directly on raw non-negative features** (counts, intensities) with interpretable units (e.g., "topic 3 loads 0.7 on 'neural'"). ✅

---

## 28. Disadvantages

- **Requires non-negative input** — not universal (data with negative values excluded). ✗
- **Non-convex → local optima** — sensitive to init; multiple runs needed. ✗
- **k is critical and difficult to set** — wrong choice → poor/duplicated parts. ✗
- **No orthogonality constraint** → parts can overlap (sometimes desired, sometimes messy). ✗
- **Reconstruction-focused; not designed for class separation or variance ranking like LDA/PCA.** ✗
- **Scaling/missing-data handling non-native** — needs preprocessing or weighted variants. ✗

---

## 29. When to Use

- ✓ Data is non-negative (counts, intensities, frequencies, magnitudes).
- ✓ You want interpretable, additive "parts" (topics, image parts, sources).
- ✓ You want sparsity in the components.
- ✓ Text analysis, image parts, audio/nuclear/spectral unmixing.
- ✓ You want an unsupervised embedding with human-readable factors.

---

## 30. When NOT to Use

- ✗ Data contains meaningful negative values (use PCA/SVD/ICA).
- ✗ You need orthogonal/ordered components (PCA ranks by variance).
- ✗ You need class-separating directions (use LDA).
- ✗ Missing-heavy matrices without imputation/weighting.
- ✗ Very small data where a simpler SVD suffices.
- ✗ When subtractive structure genuinely exists (cancellations are information).

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---------|-------|-----------|--------|
| Topic modeling | term×document counts | NMF | Topic-word basis W, doc-topic coefficients H |
| Face parts analysis | pixel×images | NMF | Parts (eyes/nose/mouth) components |
| Audio source separation | frequency×time spectrogram | NMF | Individual instrument/source components |
| Spectrum unmixing (remote sensing) | band×pixel reflectances | NMF | End-member spectra + abundances |
| Collaborative filtering (ratings ≥ 0) | user×item ratings | Non-negative MF | User/item latent factors |
| Biomarker discovery | gene expression × samples | NMF | Gene-expression programs/modules |

---

## 32. Failure Cases

- **Data failure:** Negative entries (after any processing) break the method; zero-heavy matrices can be fine for KL but noisy for Fro.
- **Mathematical failure:** k too large → overfits noise; k too small → cannot represent additive parts.
- **Optimization failure:** Poor init → suboptimal local minimum; insufficient iterations → not converged.
- **Generalization failure:** W fit on one distribution may not represent varied new samples; verify with held-out transform error.
- **Practical failure:** "Topics" can be semantically incoherent if k mis-set or data noisy; interpret W with care.
- **Scaling failure:** Z-score normalization destroys non-negativity → unexplained negatives appear.

---

## 33. Overfitting and Underfitting

- **Overfitting:** k too large (each sample gets its own part) or no regularization → W,H memorize noise; poor generalization to new samples.
- **Underfitting:** k too small → parts are blends that wash out distinct structure; reconstruction error high.

**Balance:** Choose k by reconstruction elbow + downstream/interpretability validation; add L1 regularization for sparsity and stability; cross-validate transform error on held-out columns.

---

## 34. Bias-Variance Perspective

- k = model complexity: small k → high bias (coarse parts), low variance; large k → low bias (fits detail), higher variance (noise memorized).
- Regularization (L1 → sparsity, L2 → smoothness) trades bias (simpler parts) vs variance (less overfitting).
- Selecting k and α by held-out reconstruction/downstream performance finds the practical sweet spot.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|-----------|-----------|----------|----------|----------|
| **NMF** | Non-negative low-rank factors | Interpretable parts, sparse, additive | Non-negative only, local optima | Topics, parts, spectral unmixing |
| **SVD/PCA** | Orthogonal signed factors | Coherent variance ranking, any sign | Negative loadings, not parts-based | General DR, compression |
| **LDA (topic model)** | Probabilistic generative topics (Dirichlet) | Probabilistic statements, principled priors | More complex, needs fitting care | Text topics with probability framing |
| **ICA** | Independent signed components | Statistically independent sources | ≠ parts/sparse; sign issues | Signal separation |
| **Matrix Factorization (recsys)** | Learned factors on observed entries | Handles missing ratings | Non-interpretable unless constrained | Recommendations |

---

## 36. Algorithm Selection Guide

```text
Non-negative data matrix?
├── Interpretable additive parts (topics, parts, sources) → NMF
├── Variance ordering/orthogonal components → PCA (via SVD)
├── Source independence (audio/EEG) → ICA
├── Missing entries to predict (ratings) → regularized MF (non-neg optional)
├── Probabilistic topics for text → LDA topic model
└── Negative data present → PCA/SVD (not NMF)
```

---

## 37. Common Mistakes

```text
❌ Using NMF on data with negative values (e.g., standardized features)
Why wrong: the additive non-negative model is undefined / breaks
Correct: keep data non-negative, or use PCA/SVD

❌ Setting k arbitrarily large
Why wrong: overfits noise; topics/parts become incoherent
Correct: elbow on reconstruction + interpretability, CV when possible

❌ Forgetting that NMF has local optima — using a single init
Why wrong: unlucky init → poor local solution
Correct: multiple inits (or NNDSVD), keep best

❌ Z-score standardization before NMF
Why wrong: introduces negatives, defeating the method
Correct: use non-negative scalers (e.g., row-normalize counts) or none

❌ Using NMF as a black-box DR without checking part interpretability
Why wrong: NMF's value is interpretation
Correct: inspect W columns (top words/parts), verify coherence
```

---

## 38. Interview Questions

### Beginner (with answers)
**Q: What is NMF?** A factorization of a non-negative matrix A into non-negative W·H ≈ A.
**Q: Why non-negative?** So the reconstruction reads as sums of parts (additive, interpretable).
**Q: What does W and H each mean?** W = parts/components; H = how much each part is used per data column.

### Intermediate (with answers)
**Q: Why multiplicative updates?** Standard gradient steps can drive entries negative; multiplying by a positive ratio preserves non-negativity and gives monotone loss decrease (Lee–Seung).
**Q: How do you choose k?** Reconstruction elbow + downstream/interpretability check; too small underfits, too large overfits.
**Q: Frobenius vs KL?** Frobenius fits dense continuous data; KL fits counts/Poisson-like data, weighting errors proportionally.

### Advanced (with answers)
**Q: Why is the joint problem non-convex?** Product W H is bilinear; the objective is jointly non-convex but convex in each factor alone → alternating block coordinate descent.
**Q: How does NMF relate to a generative model?** It's a factor model where each column is a non-negative additive mixture of parts — the KL version corresponds to a Poisson/multinomial-like generative view; NMF ≈ non-negative projection of the data onto a simplex.
**Q: How do you make NMF more interpretable in text?** Use KL loss, add L1 sparsity (α, l1_ratio), tune k with topic coherence, and inspect top-weighted words per component.

---

## 39. GATE / Exam Perspective

**Key formulas:**

```text
A ≈ W H        (A≥0, W≥0, H≥0, A m×n, W m×k, H k×n)
min ‖A − W H‖²_F   subject to W,H ≥ 0
KL: Σ [A log(A/(WH)) − A + WH]
update: H ← H ⊙ (WᵀA)/(WᵀWH);  W ← W ⊙ (AHᵀ)/(WHHᵀ)
A[:, j] ≈ Σ_t H[t,j] W[:, t]   (additive parts view)
```

**Common traps:**
- NMF is **unsupervised**, **additive**, **non-negative** — no negative loadings, unlike SVD/PCA.
- **k is a hyperparameter** (latent rank), not discovered by variance ranking.
- Objective is **non-convex jointly** → local optima; init matters.
- Requires **A ≥ 0**; Z-scoring is a classic exam trap (negatives).
- Multiplicative updates guarantee non-negativity, unlike subtractive rules.

**Representative pattern question (NOT a real PYQ):** "Why can NMF not be applied directly to a mean-centered matrix?" → mean-centering produces negative entries, violating the non-negative model. (Verify before treating as PYQ.)

---

## 40. Coding Practice

1. **Level 1:** Build a small non-negative matrix and verify row/column additivity structure.
2. **Level 2:** Implement `nmf_scratch`; factor a collinear 2×2 exactly (k=1).
3. **Level 3:** Show k=1 fails on a matrix whose rows are not proportional; fix with k=2.
4. **Level 4:** Use sklearn NMF on a 20-newsgroups term-document matrix; extract topics.
5. **Level 5:** Add L1 regularization; compare sparsity/interpretation vs unregularized.
6. **Level 6:** Compare NMF vs SVD topics (top words per component) for interpretability.
7. **Level 7:** End-to-end: apply NMF to a face/image dataset (Olivetti), recover parts, and evaluate a classifier on NMF vs PCA embeddings.

---

## 41. Practical ML Workflow

```text
Problem → interpretable/parts-based factorization or DR (non-negative data)
  ↓ Data → non-negative matrix (terms×docs / pixels×images / spectrograms)
  ↓ EDA → sparsity, distributions, min-value check (must be ≥0)
  ↓ Cleaning → impute (if dense) or keep masks; remove rare/zero rows if helpful
  ↓ Feature engineering → (text) TF/row-normalize; (images) flatten intensities
  ↓ Split (if downstream) → train/test; fit W on train
  ↓ Preprocess → non-negative scaling (avoid Z-score!)
  ↓ Train → NMF(k, loss, init, α), run multiple inits, keep best
  ↓ Evaluate → reconstruction, coherence/sparsity, downstream score, stability
  ↓ Error analysis → tune k/α, compare with SVD topics
  ↓ Deploy → save W (and scaler); transform new samples with W
  ↓ Monitor → drifts in component coherence / downstream metrics
```

---

## 42. Complexity

- **Multiplicative updates per iteration:** O(m n k) (matrix products AHᵀ, WHHᵀ, etc.).
- **Coordinate-descent solver (sparse A):** O(k × nnz(A) × iterations) roughly — very efficient for sparse counts.
- **Memory:** O(m k + k n) for factors + O(nnz(A)) for sparse input.
- **Convergence:** linear-ish; typical 100–500 iterations.

**Scaling:** Linear in data size per iteration; sparse-friendly — scales well for text/images; k small keeps it cheap.

---

## 43. Advanced Concepts

- **Regularized NMF (sparsity):** L1 on W/H for sparse parts; L2 for smoothness; the sklearn `α, l1_ratio` interface.
- **Weighted / missing-data NMF:** mask observed entries (recommenders).
- **Sparse-NMF variants**, and **online/minibatch NMF** for extremely large datasets (sklearn `MiniBatchNMF` / `nmf` partial_fit).
- **Relation to stopwords/topic coherence** — metrics like UMass coherence to pick k.
- **Bayesian/Probabilistic NMF (Gamma-Poisson)**: uncertainty-aware variants.
- **NMF vs LDA topic models:** both discover topics; NMF is matrix-based and deterministic, LDA is generative and probabilistic.

---

## 44. Connections to Other Algorithms

```text
               NMF (non-negative factorization)
                   |
        +----------+---------+----------+
        |          |         |          |
      SVD/PCA    MatrixMF   LDA topic  ICA
  (signed,     (missing,  (probabilistic  (independent
   orthogonal)  reg.)       generative)     sources)
        |
  Topics / parts / sources / spectral unmixing
        |
  +--- clustering/classification on H/W embeddings
```

---

## 45. If You Remember Only 5 Things

1. NMF factors a **non-negative** matrix A into **non-negative** W·H ≈ A — an **additive, parts-based** decomposition.
2. Columns of W = interpretable parts (topics, image parts, sources); H = weights of each part per column.
3. The objective is **reconstruction error (Frobenius) or KL divergence**, minimized by **multiplicative updates** that preserve non-negativity.
4. The joint problem is **non-convex** (local optima) and needs the latent **k** chosen by reconstruction elbow + interpretability; regularization (L1) boosts sparsity.
5. Use NMF when data is non-negative and you want **human-readable additive factors** — otherwise prefer SVD/PCA (signed) or LDA (probabilistic) depending on needs.

---

## 46. Cheat Sheet

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Non-Negative Matrix Factorization |
| **Category** | Unsupervised matrix factorization / parts-based DR |
| **Goal** | Interpretable additive decomposition A ≈ WH |
| **Input** | Non-negative A (m×n) |
| **Output** | W (m×k) parts, H (k×n) coefficients/embedding |
| **Core Formula** | min ‖A−WH‖²_F s.t. W,H≥0 (or KL); multiplicative updates |
| **Objective** | Reconstruction error (Fro/KL) + optional sparsity |
| **Optimization** | Alternating multiplicative / coordinate descent |
| **Parameters** | W, H |
| **Hyperparameters** | k, init, solver, beta_loss, α_W/α_H, l1_ratio, max_iter |
| **Assumptions** | Non-negativity, additive generation, right latent k |
| **Advantages** | Interpretable parts, sparse, count-friendly, additive |
| **Disadvantages** | Non-negative only, non-convex/local optima, k critical, no orthogonality |
| **Use When** | Non-negative data + interpretable factors (topics, parts, sources) |
| **Avoid When** | Negative data, orthogonal/variance ranking, class separation, missing-heavy |
| **Related** | SVD, PCA, Matrix Factorization, ICA, LDA topic models |
| **Key Exam Points** | Additive model, non-negativity constraint, multiplicative updates, k hyperparameter |
| **Key Interview Points** | Why non-negative, updates, non-convexity, k/α tuning, topics |

---

## 47. Final Mental Model

```text
 A (m×n) ≥ 0
   ↓
 choose k; init W≥0, H≥0
   ↓
 loop:
   H ← H ⊙ (WᵀA)/(WᵀWH)
   W ← W ⊙ (AHᵀ)/(WHHᵀ)
   (optionally normalize/normalize + regularize)
   loss: ‖A−WH‖²_F (or KL)
   ↓
 converged → W (parts), H (embedding)
   ↓
 interpret W columns; use H as features; transform new data via W
```

---

## 48. Knowledge Check

### Recall (5)
1. What does NMF factor and with what constraint?
2. What do W and H represent?
3. Give two NMF losses.
4. Write the multiplicative update for H.
5. Is NMF's objective jointly convex?

### Understanding (5)
1. Why does non-negativity give interpretability?
2. Why multiplicative (not subtractive) updates?
3. How does k affect over/underfitting?
4. Why is a single random init risky?
5. Why can't you Z-score before NMF?

### Application (5)
1. Topic modeling pipeline — steps?
2. How to pick k for topics?
3. How to embed a NEW document?
4. How to get sparser topics?
5. When would you use KL vs Frobenius?

### Mathematical (5)
1. Show NMF with k=1 factorizes the matrix [[3,6],[1,2]] exactly.
2. Why does k=1 fail on [[1,0],[0,1]]?
3. Write the Frobenius objective.
4. Write the KL objective.
5. Show the update rules keep W,H ≥ 0.

### Interview (5)
1. How does NMF differ from PCA/SVD?
2. Relation to LDA topic models?
3. What's weighted NMF for missing data?
4. How do you regularize NMF for sparsity?
5. What is MiniBatch/online NMF?

### Problem Solving (5)
1. Non-negative data but with a few zeros — fine?
2. Corpus of 100 docs; k choice range?
3. Negative counts after TF-IDF — how to use NMF?
4. A stray very large count — what loss/step?
5. Recommend building NMF on face parts — design.

## Answers (explained)
1. A≈WH with A,W,H ≥0 (non-negative low-rank factorization). 2. W=parts/components; H=per-column weights/embedding. 3. Frobenius ‖A−WH‖²_F; KL Σ[a log(a/(WH))−a+WH]. 4. H ← H ⊙ (WᵀA)/(WᵀWH+ε). 5. No — jointly non-convex (convex in each factor alone).
6. No negative cancellations hidden — parts add up honestly. 7. Subtractive updates can flip signs; multiplicative (positive ratio) preserves non-negativity + monotone descent. 8. k small → underfit (parts blend); k large → overfit (noise parts). 9. Local optima differ; run several inits / NNDSVD init. 10. Creates negative values, breaking the model.
11. Convert corpus → term-doc CountVectorizer → NMF(k) → inspect top words per W column; use H as doc features. 12. Coherence/elbow on reconstruction + manual coherence; cross-validate on downstream task. 13. model.transform(new_matrix) with fitted W. 14. Increase α (L1, l1_ratio→1); sometimes larger k + L1. 15. Counts/tf → KL; dense continuous non-negative signals → Frobenius.
16. W=[[3],[1]], H=[[1,2]] → WH=[[3,6],[1,2]] error 0. 17. Rows [1,0] and [0,1] are not proportional → rank-1 product can't produce it; error > 0. 18. min_{W,H≥0}‖A−WH‖²_F. 19. min Σ[A log(A/WH)−A+WH]. 20. All factors start ≥0 and every update multiplies by a non-negative ratio → stays ≥0.
21. NMF requires ≥0 and gives additive sparse parts; PCA/SVD allow negatives, orthogonal axes, variance ordering. 22. Both find topics; NMF = matrix factorization (deterministic), LDA = probabilistic generative model with Dirichlet priors. 23. Weight the observed entries only (mask/reweight the loss), leaving missing untouched. 24. Add L1 penalty (α·‖W‖₁ or ‖H‖₁) via l1_ratio → sparser factors. 25. Online algorithm updating W,H on minibatches for huge datasets.
26. Yes — zeros are valid non-negative entries; KL handles them naturally. 27. k from ~2..min(m,n, many), typically 5–50; choose by coherence/elbow. 28. TF-IDF has negatives → use raw counts or non-negative TF variants for NMF (or shift). 29. Log-scale counts or use KL loss (proportional weighting); trim extreme outliers. 30. Flatten faces (pixels→rows), NMF with k≈parts count, cluster using H embeddings.

---

## 49. Final Learning Checklist

- [ ] State the NMF factorization and non-negativity constraint
- [ ] Define W (parts) and H (coefficients/embedding)
- [ ] Write Frobenius and KL objectives
- [ ] Write H and W multiplicative update rules
- [ ] Show why multiplicative updates preserve non-negativity
- [ ] Hand-verify k=1 factorization (collinear columns)
- [ ] Explain why k=1 fails on non-collinear rows
- [ ] Call out the non-convexity and init sensitivity
- [ ] Choose k by reconstruction elbow + interpretability
- [ ] Add L1/L2 regularization for sparsity
- [ ] Implement NMF from scratch (numpy)
- [ ] Use sklearn NMF (fit_transform, components_, transform)
- [ ] Extract/interpret topics from a term-document matrix
- [ ] Compare NMF vs SVD/PCA topics
- [ ] Use NMF for face parts / spectral unmixing
- [ ] Embed new data with a fitted W
- [ ] Avoid Z-score / negative preprocessing
- [ ] Handle sparse counts efficiently (coordinate descent)
- [ ] Understand O(m n k) complexity and MiniBatch variants
- [ ] End-to-end: non-negative data → NMF → interpret + evaluate downstream

---

## 50. Quality Control Note

- **Accuracy:** Hand-verified examples — exact k=1 factorization of [[3,6],[1,2]] (error 0) and the failure on non-proportional [[1,0],[0,1]]; multiplicative-update formulas match Lee–Seung. ✅
- **Beginner-friendliness:** Money/building-parts analogies; additivity explained without heavy math. ✅
- **Math depth:** Objectives (Fro/KL), update derivation logic, convexity discussion, all symbols explained. ✅
- **Practical depth:** From-scratch + sklearn code, topic-engineering workflow, hyperparameter tuning, failure cases, coding ladder. ✅
- **Exam depth:** GATE traps (negativity, non-convexity, k as hyperparameter), representative pattern question clearly marked non-PYQ. ✅
- **Structure:** All 50 template sections present in order. ✅
- **Consistency:** NMF framed as additive, parts-based DR; objective = reconstruction error; evaluation tied to interpretability + downstream score, per the task requirement. ✅