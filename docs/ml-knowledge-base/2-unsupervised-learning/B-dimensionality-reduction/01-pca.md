# 01. Principal Component Analysis

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Aspect | Details |
|--------|---------|
| **Algorithm name** | Principal Component Analysis (PCA) |
| **Category** | Dimensionality Reduction (Unsupervised feature extraction) |
| **Type** | Linear transformation / linear feature extraction |
| **Parametric / Non-parametric** | Non-parametric (no learned weights tied to labels; but does compute a fixed transformation matrix fitted from data) |
| **Generative / Discriminative** | Neither (it is an unsupervised projection; not a classifier) |
| **Main objective** | Project data onto a lower-dimensional subspace that preserves maximum variance of the original data |
| **Input** | Feature matrix **X** of shape (n samples × p features) |
| **Output** | Embedded/scores matrix of shape (n × k), columns = k principal components |
| **Core idea** | Find orthogonal directions (eigenvectors of the covariance matrix) along which data variance is maximized; keep the top k |
| **Typical use cases** | Data visualization (to 2D/3D), noise reduction, feature compression, preprocessing before clustering/classification |

---

## 02. One-Line Definition

### Beginner Definition
PCA finds the directions in which your data spreads out the most, then re-draws your data on only those few directions, throwing away the least important ones.

### Technical Definition
PCA is a linear orthogonal transformation that maps an n×p data matrix to an n×k matrix (k < p) whose columns are the projections onto the eigenvectors of the covariance matrix ranked by their corresponding eigenvalues (explained variance), so that the retained dimensions capture maximum variance with minimum reconstruction error.

---

## 03. Intuition

Imagine you photograph a long, thin cloud of points (a "cigar") in 3D. Instead of storing all three coordinates, you could describe every point by saying how far along the cigar's long axis it lies, plus a small offset for its thickness. If the cloud is almost perfectly thin, the thickness offset is nearly the same for all points, so you can drop that coordinate and lose almost nothing. PCA does exactly this: it re-orients your axes so the first axis points along the direction of greatest spread, the second along the next greatest spread perpendicular to the first, and so on.

Step-by-step reasoning:

1. Center the data by subtracting the mean of each feature.
2. Measure how features vary with each other (the covariance matrix).
3. Find the "preferred" directions of spread — the eigenvectors.
4. Rank directions by how much variance they capture — the eigenvalues.
5. Keep only the top k directions and project the data onto them.

The key insight: real data is often redundant. Many features move together (e.g., a person's height and shoe size). PCA collapses these redundant axes into a small number of meaningful ones, saving space and discarding noise.

---

## 04. Problem It Solves

**The problem:** Real datasets often have hundreds or thousands of features (e.g., 60,000-pixel images, 10,000-gene microarrays). High-dimensional data is hard to visualize, expensive to store, slow to train on, and prone to the "curse of dimensionality" — distances between points become uninformative and models overfit.

**What we want:** A smaller set of features that still captures the essential structure/variance of the data.

**Why PCA is useful:** It provably finds the linear projection that preserves the most variance (and, equivalently, minimizes mean squared reconstruction error) for a given output dimension k.

**Small example:** Suppose you record two variables — height (cm) and weight (kg) — for 100 people. These are strongly correlated. PCA reveals one dominant principal component (roughly a weighted combination of height and weight capturing "overall body size") and a tiny second component (describing shape, e.g., tall-and-thin vs short-and-stout). You can reduce 2 features to 1 while keeping ~95% of the information.

---

## 05. Where It Fits in Machine Learning

```text
Machine Learning
├── Supervised Learning          (labeled data)
├── Unsupervised Learning        (unlabeled data)
│   ├── Clustering               (group similar points: K-Means, DBSCAN)
│   ├── Dimensionality Reduction ← PCA lives here
│   │   ├── Linear: PCA, SVD, LDA (supervised), NMF
│   │   └── Nonlinear: Kernel PCA, t-SNE, UMAP
│   └── Association Rule Learning (market baskets)
├── Semi-supervised Learning
├── Reinforcement Learning
└── Ensemble Learning
```

PCA is the foundational *linear* unsupervised dimensionality-reduction method; almost every other approach (Kernel PCA, SVD, and connections to LDA) is built on or compared against it.

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|------|----------------|-------------------|
| **Variance** | How much a feature spreads out | Expected squared deviation from the mean; Var(X) = (1/n)Σ(xᵢ − x̄)² |
| **Covariance matrix** | Table of how every pair of features varies together | p×p symmetric matrix Σ with Σᵢⱼ = Cov(fᵢ, fⱼ) |
| **Eigenvector** | A direction that a matrix only stretches, not tilts | Non-zero vector v such that Σv = λv |
| **Eigenvalue** | How much the matrix stretches along that eigenvector | Scalar λ in Σv = λv; equals variance captured by PC |
| **Principal component (PC)** | A new "axis" = eigenvector of covariance | The eigenvector v; its score = X·v |
| **Explained variance** | Share of total spread captured by a component | λᵢ / Σλⱼ |
| **Score / loading** | Position of a point along a PC / weight of an original feature in a PC | Score = Xv; loading = entry of v |
| **Projection** | Dropping a perpendicular onto an axis | x·v (dot product of point with unit eigenvector) |
| **Reconstruction** | Rebuilding the original data from few components | X̂ = scores·loadingsᵀ (+ mean) |

---

## 07. Input and Output

**Input:**
- Data matrix **X** (n × p): n samples, p features (numeric, continuous preferred).
- No target/label needed (unsupervised).

**Preprocessing:** Features are usually centered and often standardized (see §25).

**Hyperparameters:**
- `n_components` (k): number of components to keep.
- Optional: whiten, solver, tol, iterated_power.

**Output:**
- Transform/embedding matrix (n × k) — the projected data (scores).
- The components matrix (k × p) — the loadings (eigenvectors).
- Explained variance ratio per component.
- The learned mean vector and the transformation matrix **W** (p × k).

**Not a classifier:** PCA does not "predict" a label; its output is a lower-dimensional representation fed into downstream models.

---

## 08. Mathematical Foundation

**Basic idea:** Center the data, compute the covariance, and find its eigenvectors. The eigenvectors form a new basis where variance is axis-aligned and ranked.

**Notation:**
- **X** = n×p data matrix, columns mean-centered: X_c.
- **Σ** (Sigma) = p×p covariance matrix = (1/(n−1)) X_cᵀ X_c (sample covariance) or (1/n) X_cᵀ X_c (population).
- **vᵢ** = i-th eigenvector (column vector, length p).
- **λᵢ** = i-th eigenvalue (scalar).

**Core equation:**

```text
Σ v = λ v
```

**Required math concepts:** Linear algebra (matrix multiplication, dot product, matrix transpose), eigendecomposition, basic statistics (mean, variance, covariance), orthogonality.

**Interpretation:** The eigenvectors of the covariance matrix define the directions of maximum variance. The eigenvalue λ equals the variance of the data when projected onto its corresponding eigenvector. Because Σ is real and symmetric, its eigenvectors are orthonormal (mutually perpendicular, unit length), which makes PCA an orthogonal rotation — no information is distorted, only re-oriented.

---

## 09. Core Formula

### 1. Covariance matrix

```text
Σ = (1/(n-1)) (X_c)ᵀ (X_c)
```

**Meaning:** Measures pairwise linear co-variation of features.

**Symbols:** X_c = mean-centered data (each column has its mean subtracted); n = number of samples; superscript ᵀ = transpose.

**Intuition:** Each entry Σᵢⱼ says whether features i and j increase together (positive), decrease together (negative), or are unrelated (≈0). The diagonal Σᵢᵢ is the variance of feature i.

### 2. Eigendecomposition

```text
Σ V = V Λ
```

**Meaning:** Packs all eigenvector/eigenvalue facts into one equation.

**Symbols:** V = p×p matrix whose columns are eigenvectors vᵢ; Λ = diagonal matrix with eigenvalues λᵢ on the diagonal.

**Intuition:** Multiplying Σ by V stretches each eigenvector by its eigenvalue without changing its direction.

### 3. Projection / Scores

```text
T = X_c W      (equivalently T = X_c V[:, :k])
```

**Meaning:** Projects centered data onto the subspace spanned by the top k eigenvectors.

**Symbols:** T = n×k scores matrix; W = p×k matrix of the top k eigenvectors.

**Intuition:** Each score tᵢⱼ = dot product of point i with eigenvector j — "how much of direction j is in point i."

### 4. Explained variance ratio

```text
Explained_variance_ratio_i = λᵢ / (Σⱼ λⱼ)
```

**Meaning:** Fraction of total variance captured by component i.

**Intuition:** Sums to 1 across all p components; the cumulative sum up to k tells you how much you kept.

### 5. Reconstruction

```text
X̂ = T Wᵀ + mean_X
```

**Meaning:** Rebuild original data (approximately) from the reduced scores.

**Intuition:** If you kept all p components, X̂ = X exactly. Keeping k < p loses the variance along the dropped eigenvectors.

**Worked example (hand-verified):** Take centered data with 2 samples and 2 features:

```text
X_c = [[1, 1],
       [-1, -1]]
```

Covariance (n=2, use n): Σ = (1/2) X_cᵀ X_c.

X_cᵀ X_c = [[1,-1],[1,-1]] · [[1,1],[-1,-1]] = [[1+1, 1+1],[1+1, 1+1]] = [[2,2],[2,2]].

Σ = (1/2)[[2,2],[2,2]] = [[1,1],[1,1]].

Eigenvalues: solve det(Σ − λI) = 0 → (1−λ)² − 1 = 0 → λ² − 2λ = 0 → λ = 0 or λ = 2.

Eigenvector for λ=2: (Σ − 2I)v = 0 → [[−1,1],[1,−1]]v = 0 → −a + b = 0 → v = [1,1]/√2 (normalized).

Eigenvector for λ=0: [1,−1]/√2.

Explained variance: component 1 keeps 2/(2+0) = 100% of variance. Projecting X_c onto v₁ = [1,1]/√2 gives scores T = X_c v₁ = [√2, −√2]. Thanks to symmetry this reduces 2 features → 1 with zero information loss. ✅ Hand-verified.

---

## 10. Derivation

Goal: find unit vector v maximizing the variance of projections X_c v.

1. Projected variance: Var(X_c v) = vᵀ Σ v (because variance of a linear combination of centered data is vᵀ Covariance v).
2. Constraint: vᵀ v = 1 (unit length, otherwise scaling makes variance unbounded).
3. Optimize with Lagrange multiplier λ: maximize vᵀ Σ v − λ(vᵀ v − 1).
4. Take derivative w.r.t. v: 2Σv − 2λv = 0 → Σv = λv.
5. This is exactly the eigen-equation. The Lagrange multiplier λ is the eigenvalue and equals the projected variance.
6. To get the next component, restrict to the subspace orthogonal to previously found v's; the same argument yields the next eigenvector. Because Σ is symmetric positive semi-definite, its eigenvectors are orthonormal and span ℝᵖ, so this gives all p components in decreasing order of eigenvalue.

**Important result:** The k eigenvectors with the largest eigenvalues are the optimal k-dimensional linear projection in two equivalent senses: (a) max variance retained, (b) min squared reconstruction error (this is the Eckart–Young theorem, also connecting to SVD).

---

## 11. How the Algorithm Works

```text
Input (n×p) 
  ↓ 
Center columns (subtract mean) [optional: standardize] 
  ↓ 
Compute covariance matrix Σ (p×p) 
  ↓ 
Eigendecompose Σ → eigenvectors V, eigenvalues Λ 
  ↓ 
Sort by descending eigenvalue 
  ↓ 
Select top k eigenvectors → W (p×k) 
  ↓ 
Project: T = X_c W (n×k) 
  ↓ 
Output scores T + components W + explained variance
```

---

## 12. Training Process

**Pre-training:** Center the data (and optionally scale).

**During:** There is no iterative loss-minimization loop. PCA is a closed-form (direct) solution:
1. Compute the covariance matrix.
2. Compute its eigendecomposition.
3. Sort and select components.

Nothing is "learned" by gradient descent; the transformation is fully determined by the data's second-order statistics.

**Stopping:** No convergence loop — you pick k up front.

**Final model contents:** The mean vector, the p×k component matrix **W**, and optionally the explained variance ratios. These are saved and reused to transform future (unseen) data.

---

## 13. Objective Function / Loss Function

**Objective:** Maximize the variance of the projected data subject to orthogonality/unit-length constraints.

```text
maximize  vᵀ Σ v   subject to  vᵀ v = 1
```

**Equivalent loss (reconstruction) view:** Minimize the mean squared reconstruction error:

```text
minimize  (1/n) ‖ X_c − T Wᵀ ‖²_F     (Frobenius norm)
```

**Why chosen:** Variance is a natural proxy for "information" in continuous data; reconstruction error directly measures how much is lost.

**High/low meaning:** Higher retained variance = less information discarded; lower reconstruction error = projection is more faithful.

**Note:** PCA's training objective (explained variance) differs from the downstream evaluation metric (e.g., classifier accuracy on the projection). See §26.

---

## 14. Optimization

**Definition:** Because PCA's optimum is obtained analytically (closed-form eigendecomposition), no numeric gradient-based optimization is required.

**Why:** The objective vᵀΣv with vᵀv=1 is a Rayleigh quotient whose extrema are exactly the eigenvectors — a solved linear-algebra problem.

**Method:** Power iteration or an off-the-shelf eigensolver (LAPACK) computes the top eigenvectors quickly.

**Gradient/learning rate:** None needed (no gradient descent step). If you choose iterative eigensolvers like power iteration, the "iterations" just converge to the dominant eigenvector.

```text
Start vector u0
  ↓ 
u_{t+1} = Σ u_t / ‖Σ u_t‖   (power iteration)
  ↓ 
Converges to largest-eigenvalue eigenvector
  ↓ 
Deflate (subtract component), repeat for next PC
```

**Global optimum:** The top-k eigenvectors give the global (not local) optimum because it's an exactly solvable eigen-problem.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE (hand-verified).** Consider 4 points in 2D:

```text
x1 = (1, 2)
x2 = (2, 2)
x3 = (3, 4)
x4 = (4, 4)
```

**Step 1 — Center:** Mean of feature1 = (1+2+3+4)/4 = 2.5; mean of feature2 = (2+2+4+4)/4 = 3. Centered:

```text
C = [[-1.5, -1],
     [-0.5, -1],
     [ 0.5,  1],
     [ 1.5,  1]]
```

**Step 2 — Covariance (population, ÷n):**
Cᵀ = [[-1.5,-0.5,0.5,1.5],[-1,-1,1,1]].

CᵀC (2×2): 
- entry(1,1) = 2.25+0.25+0.25+2.25 = 5
- entry(1,2) = (-1.5)(−1)+(−0.5)(−1)+(0.5)(1)+(1.5)(1) = 1.5+0.5+0.5+1.5 = 4
- entry(2,2) = 1+1+1+1 = 4

CᵀC = [[5,4],[4,4]]. Σ = (1/4)CᵀC = [[1.25,1],[1,1]].

**Step 3 — Eigenvalues:** det([[1.25−λ,1],[1,1−λ]]) = 0 → (1.25−λ)(1−λ) − 1 = 0 → 1.25 −1.25λ −λ +λ² −1 = 0 → λ² −2.25λ +0.25 = 0. Roots: λ = [2.25 ± √(5.0625−1)]/2 = [2.25 ± √4.0625]/2 = [2.25 ± 2.0156]/2. λ₁ = 2.1328, λ₂ = 0.1172. (Note the tiny second eigenvalue — data is nearly one-dimensional.)

**Step 4 — Total variance:** 2.1328 + 0.1172 = 2.25. Ratio for PC1 = 2.1328/2.25 ≈ 0.9478 → PC1 explains ~94.8%.

**Step 5 — Eigenvector for λ₁:** ([[1.25−2.1328,1],[1,1−2.1328]])v=0 → [[−0.8828,1],[1,−1.1328]]v=0 → v = [1, 0.8828] normalized → v₁ ≈ [0.75, 0.66].

**Step 6 — Project to k=1:** scores T = C v₁ ≈ [-1.5*0.75 + (-1)*0.66, -0.5*0.75+(-1)*0.66, 0.5*0.75+1*0.66, 1.5*0.75+1*0.66] = [-1.785, -1.035, 1.035, 1.785].

The 4 points collapse to 1D positions preserving ~95% of variance. ✅ Hand-verified.

---

## 16. Visual Explanation

```text
Original 2D scatter (centered):

           f2
           4 |          x4
           3 |      x3
           2 |  x1      x2
           1 |
           --+------------------
              f1

Cigar-shaped cloud → one dominant axis of spread.

After PCA:

           f2
            |      ^ PC2 (small eigenvalue)
            |       |  
            |       +-------> PC1 (large eigenvalue: direction of max variance)
```

```text
Projection onto PC1 (drop PC2):

   <-1.8   <-1.0    1.0    1.8     (score axis)
    x1      x2      x3     x4
```

The data becomes a ranking of points along the single most-informative axis.

---

## 17. Algorithm / Pseudocode

```
1. Center the data: X_c ← X − mean(X, axis=0)
2. (Optional) Standardize: divide each column by its std
3. Compute covariance matrix: Σ ← (1/(n-1)) X_cᵀ X_c   [or using SVD]
4. Eigen-decompose:  [V, Λ] ← eig(Σ)
5. Sort eigenvectors by descending eigenvalue
6. Select top k eigenvectors → W (p×k)
7. Project:  T ← X_c · W
   return T (scores), W (loadings), explained_variance_ratios
```

**Numerically robust alternative:** perform SVD of X_c directly (X_c = U D Vᵀ); the right-singular vectors V equal the eigenvectors of Σ, and the singular values squared ∝ eigenvalues. See note 06-svd.

---

## 18. From-Scratch Implementation

```python
import numpy as np

def center(X):
    return X - np.mean(X, axis=0)

def pca_scratch(X, n_components):
    Xc = center(X)
    cov = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    W = eigenvectors[:, :n_components]
    scores = Xc @ W
    total = eigenvalues.sum()
    explained = eigenvalues / total
    return scores, W, explained

X = np.array([[1, 2], [2, 2], [3, 4], [4, 4]], dtype=float)
scores, W, explained = pca_scratch(X, n_components=1)
print("Scores:", scores.ravel().round(3))
print("Loadings:", W.ravel().round(3))
print("Explained ratio:", explained[:1].round(3))
```

**Numerically stable SVD variant:**

```python
def pca_svd(X, n_components):
    Xc = center(X)
    U, D, Vt = np.linalg.svd(Xc, full_matrices=False)
    scores = Xc @ Vt.T[:, :n_components]
    var = (D ** 2) / (X.shape[0] - 1)
    explained = var / var.sum()
    return scores, Vt[:n_components], explained
```

---

## 19. Code Explanation

```text
Code line → What does it do? → Why required? → Mathematical concept?
X - mean → mean-centers each column → makes covariance zero-mean → centers data so projection is about variance, not offset
Xc.T @ Xc / (n-1) → computes covariance → needed to find axes of variance → symmetric p×p matrix Σ
np.linalg.eigh → eigendecomposition → obtains directions + variance → solves Σ v = λ v
argsort[::-1] → rank by eigenvalue desc → PC1 is highest-variance → sorted eigenvalues
eigenvectors[:, :k] → pick top k → keeps only signal axes → dimensionality reduction
Xc @ W → dot product of data with loadings → produces lower-dim coordinates → projection score t = x·v
eigenvalues / total → normalize → share of variance each PC holds → explained variance ratio
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[1, 2], [2, 2], [3, 4], [4, 4]], dtype=float)

model = PCA(n_components=1)
scores = model.fit_transform(X)
print("Scores:", scores.ravel().round(3))
print("Components:", model.components_.ravel().round(3))
print("Explained variance ratio:", model.explained_variance_ratio_.round(3))
print("Mean:", model.mean_)

X_reconstructed = model.inverse_transform(scores)
print("Reconstruction:", X_reconstructed.round(2))
```

Key API: `fit` (learn mean + components), `fit_transform` (fit then project), `inverse_transform` (reconstruct), `explained_variance_ratio_`, `components_`.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|----------------|---------|--------|----------------------|
| `n_components` (k) | Number of axes to keep | Too low → lose signal; too high → keep noise | Choose via explained-variance elbow, or k where cumsum ≥ 0.9–0.95 |
| `whiten` | Scales components to unit variance | Removes correlation/scaling artifacts; changes downstream interpretation | True if features have very different scales/units |
| `svd_solver` | Algorithm for decomposition (auto/full/arpack/randomized) | Speed vs accuracy on large data | `randomized` for huge matrices |
| `tol` | Tolerance for arpack | Convergence of iterative solver | Default usually fine |
| `iterated_power` | Power iterations for randomized SVD | Quality of randomized SVD | Increase for more accuracy |

**too low / too high / how to tune:** k too low underfits (drops structure); k too high overfits (keeps noise) and defeats the purpose. Tune by plotting cumulative explained variance and choosing the elbow.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Mean vector of each feature.
- Component matrix **W** (k eigenvectors).
- Explained variance values (λᵢ).
These are stored in the fitted model (`components_`, `mean_`, `explained_variance_`).

### Hyperparameters (chosen)
- `n_components` (k).
- `whiten` (True/False).
- `svd_solver`, `tol`, `iterated_power`.

---

## 23. Assumptions

| Assumption | What | Why | How to check | What if violated | Solution |
|-----------|------|-----|--------------|------------------|----------|
| **Linearity** | Data lies near a linear subspace | PCA finds linear axes | Scatter plots / residual variance after projection | Non-linear structure misrepresented | Kernel PCA, t-SNE, UMAP |
| **Variance = information** | High-variance directions matter most | Objective maximizes variance | Compare to domain knowledge | Low-variance directions may be critical | Consider domain-informed preprocessing |
| **Orthogonality** | New axes are perpendicular | Eigenvectors of symmetric Σ | Mathematically guaranteed | Sometimes non-orthogonal axes fit better | ICA (independent components) |
| **Continuous, roughly Gaussian-ish features** | Feasible covariance interpretation | Covariance assumed adequate | Histograms, Q-Q plots | Heavy skew / outliers skew covariance | Standardize, transform, robust PCA |
| **Mean/covariance characterize the data** | Uses only 2nd-order statistics | PCA uses Σ only | Not always obvious | Higher-order structure ignored | Kernel PCA |

---

## 24. Data Requirements

- **Data type:** Numeric/continuous features. Categorical must be encoded or excluded.
- **Missing values:** Not handled natively — impute first or the covariance breaks.
- **Outliers:** Sensitive; a few extreme points can rotate principal components. Robust PCA or removal helps.
- **Scaling:** Very important when features have different units (see §25).
- **Feature engineering:** Not part of PCA; PCA itself performs feature extraction/compression.
- **Dataset size:** Works for small and large n; the p×p covariance is cheap when p is small; for huge p (e.g., 10⁵ genes) use SVD/randomized methods or the "kernel trick" version.
- **Class imbalance:** Irrelevant (unsupervised; no labels involved).

---

## 25. Feature Scaling

**Recommended (often required) when feature scales differ.**

- If features have different units (cm vs kg vs dollars), the high-variance feature dominates the covariance and biases PCA. Standardize each feature to zero mean and unit variance (Z-score) so every feature contributes fairly.
- If features are already comparable units, centering alone (mean-only) suffices, and PCA then preserves relative variance meaningfully (this is common in some signal/spectral settings).
- **Methods:** StandardScaler (z = (x − μ)/σ), sometimes MinMax scaling.

**Rule of thumb:** If you cannot decide, standardize — it is the safest default for PCA.

---

## 26. Evaluation Metrics

**Training objective ≠ evaluation metric.** PCA optimizes explained variance; downstream tasks evaluate real usefulness.

| Metric | Definition | Formula | Interpretation | When to use | When NOT to use |
|--------|-----------|---------|----------------|-------------|-----------------|
| **Explained variance ratio** | Share of variance per component | λᵢ/Σλⱼ | How much signal each axis holds | Choosing k, checking sufficiency | Not a measure of downstream utility |
| **Cumulative explained variance** | Sum of top-k ratios | Σᵢ≤ₖ λᵢ/Σλ | Info retained by k dims | Selecting k (elbow) | Ignoring noise-vs-signal subtleties |
| **Reconstruction error (MSE)** | How faithfully projection reproduces data | (1/n)‖X−T Wᵀ‖² | Fidelity of compression | Testing lossless-ness | Downstream accuracy |
| **Downstream task score** | Accuracy/F1 of classifier on projection | e.g., Accuracy | Practical benefit of reduction | Validating DR helps | Interpreting alone |
| **Separability visualization** | Do classes separate in 2D projection | Scatter plot | Sanity check, cluster discovery | Exploratory visualization | Over-trusting visual separation |

---

## 27. Advantages

- **Reduces dimensionality** (k < p): speeds up training, storage, and inference. ✅
- **Removes correlation / redundancy** among features (orthonormal axes). ✅
- **De-noises data** by dropping low-variance (often noise) directions. ✅
- **Hard to overfit** (fewer parameters than the full feature space) when done correctly with proper k. ✅
- **Tractable & fast** — closed-form solution, no iterative tuning. ✅
- **Interpretable loadings** — weights show which original features drive each component. ✅
- **Enables 2D/3D visualization** of high-dimensional data. ✅

---

## 28. Disadvantages

- **Linear only** — cannot capture non-linear manifolds (curves, spirals). ✗
- **Sensitive to outliers** — extreme points can dominate variance and rotate axes. ✗
- **Sensitive to scaling** — misleading results if features have different units. ✗
- **Assumes variance is informative** — sometimes low-variance directions carry the actual signal. ✗
- **Global, not local** — a single global subspace may miss local structure. ✗
- **Hard to interpret** PCs when many original features mix into one component. ✗
- **Ignores labels** — even when class labels exist (LDA uses them better). ✗

---

## 29. When to Use

- ✓ Data is high-dimensional and you want a lower-dim summary.
- ✓ You need 2D/3D visualization of samples.
- ✓ Features are redundant/correlated and you want decorrelation.
- ✓ You want noise reduction as preprocessing.
- ✓ You want a fast, interpretable, linear projection.
- ✓ Downstream model is sensitive to many features (curse of dimensionality).

---

## 30. When NOT to Use

- ✗ Data has strong non-linear structure (use Kernel PCA, t-SNE, UMAP).
- ✗ You have labels and care about class separation (use LDA).
- ✗ Outliers are abundant (consider Robust PCA).
- ✗ You must interpret features individually (NMF may suit sparse/binary data).
- ✗ Feature scales are incomparable and you can't standardize meaningfully.
- ✗ You need exact preservation of distances in high-d (PCA is a global linear map; t-SNE/UMAP are local).

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---------|-------|-----------|--------|
| Face recognition (Eigenfaces) | 10k-pixel grayscale face images | PCA (via SVD) | A few dozen eigenfaces compressing each face |
| Gene expression analysis | Microarray samples × 10k genes | PCA | Low-dim sample embedding showing subgroups |
| Stock/portfolio analysis | Daily returns of many assets | PCA | Principal factors (market, sector) |
| Image compression | Image pixel matrix | PCA on patches | Compressed representation, low reconstruction loss |
| Anomaly detection | High-dim sensor logs | PCA | Reconstruction error flags unusual points |
| Preprocessing for clustering | High-dim features | PCA → K-Means | Clean clusters in reduced space |

---

## 32. Failure Cases

- **Data failure:** Highly skewed distributions and gross outliers rotate principal components toward outliers, hiding true structure.
- **Mathematical failure:** Under the linearity assumption, curved manifolds (e.g., a Swiss roll) get crushed; PCA sees a plane cutting through the roll.
- **Scaling failure:** Mixing cm and kg without standardization makes the larger-scale feature dominate.
- **Choosing k without evidence:** Picking too few components discards signal; too many keeps noise and defeats the purpose.
- **Generalization failure:** Fitting PCA on all data (including test set) leaks information — fit only on training data, then transform.
- **Interpretation failure:** Believing PC1 alone is meaningful when loadings spread across many features.

---

## 33. Overfitting and Underfitting

- **Overfitting:** Keeping too many components (k near p) so that noise is retained, or fitting PCA on the whole dataset (data leakage) repeat the training distribution too literally. Also, choosing k by maximizing validation performance on a tiny set can chase noise.
- **Underfitting:** Keeping too few components so real structure (classes, clusters) is erased.

**Balance:** Select k with a variance elbow / cross-validation on the downstream task, fit PCA only on training data, and validate on held-out data.

---

## 34. Bias-Variance Perspective

- PCA imposes a strong constraint (few axes) → high bias if the true structure needs more dimensions; low variance because few parameters.
- Many components → low bias (can represent more) but higher variance (noise gets fit).
- There is a bias-variance trade-off in choosing k: too small k underfits structure (bias), too large k overfits noise (variance). The optimal k balances reconstruction fidelity against downstream generalization.
- As a regularizer: PCA-denoising (project → reconstruct) can reduce variance of downstream models, at the cost of some bias if signal is dropped.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|-----------|-----------|----------|----------|----------|
| **PCA** | Linear max-variance projection | Fast, interpretable, closed-form | Linear only, sensitive to scaling/outliers | General linear reduction, visualization |
| **Kernel PCA** | PCA in a high-dim feature space via kernel | Captures non-linear structure | Kernel choice, less interpretable, slower | Non-linear manifolds, curved blobs |
| **LDA** | Linear projection maximizing class separation | Uses labels → better for classification | Needs labels, ≤ C−1 dims, Gaussian assumptions | Supervised reduction for classification |
| **SVD** | Factor matrix into U·D·Vᵀ | Works on any rectangular matrix, basis of PCA | Not a reduction objective by itself | Low-rank approximation, PCA via SVD, recommendation |
| **NMF** | Factor into non-negative matrices | Interpretable parts-based decomposition | Requires non-negative data | Topic modeling, image parts |
| **t-SNE** | Preserves local pairwise similarities | Great 2D/3D visualization of clusters | No preserving big distances, expensive, non-unique | Visualization only |
| **UMAP** | Preserves local + some global structure via manifold | Fast, better global structure, scalable | Less standardized than t-SNE, hyperparams | Visualization + embedding for downstream |

---

## 36. Algorithm Selection Guide

```text
Need to reduce dimensions?
├── Data has labels & goal is class separation → LDA
├── Need linear, interpretable, fast → PCA (or SVD for exact/robust)
├── Data is non-linear (curved manifold) →
│   ├── Need visualization in 2D/3D → t-SNE / UMAP
│   └── Need a reproducible embedding → Kernel PCA / UMAP
├── Data is non-negative (counts, images) → NMF
└── Need a low-rank approximation of a big matrix → SVD
```

Default first try: **PCA** — fast, robust baseline; escalate only if structure is clearly non-linear or labels demand supervised reduction.

---

## 37. Common Mistakes

```text
❌ Not standardizing before PCA when units differ
Why wrong: high-magnitude features dominate covariance, biasing axes
Correct: StandardScaler on training data before fitting

❌ Fitting PCA on the FULL dataset including test set
Why wrong: data leakage — test info leaks into the transform
Correct: fit on train only; transform both train and test with the same model

❌ Interpreting PC1 loadings literally with many mixed features
Why wrong: loadings can diffuse across features; not necessarily a clean meaning
Correct: inspect loadings, scree/elbow plots, and domain context

❌ Choosing k purely by "95% variance" with no downstream check
Why wrong: variance ≠ task usefulness
Correct: validate on downstream metric too

❌ Applying PCA to handle multicollinearity then expecting to recover original features
Why wrong: PCA replaces features with new axes; original names are lost
Correct: accept transformed space or choose a different method if interpretability matters
```

---

## 38. Interview Questions

### Beginner (with answers)
**Q: What does PCA do in one sentence?** Finds the directions of maximum variance and projects data onto the top few.
**Q: Why center the data?** So variance is measured about the mean (covariance is zero-mean by construction).
**Q: What are eigenvalues and eigenvectors in PCA?** Eigenvectors are directions of max variance; eigenvalues are amounts of variance along them.

### Intermediate (with answers)
**Q: Why must you scale features before PCA?** Because covariance weights by magnitude; differing units make larger-scaled features dominate.
**Q: What does "explained variance ratio" mean?** The fraction of total variance captured by a component, λᵢ/Σλⱼ.
**Q: Is PCA the same as eigendecomposition of the covariance matrix?** Yes, when done that way; equivalently SVD of the centered matrix.

### Advanced (with answers)
**Q: How is PCA related to SVD?** Σ ∝ XᵀX; the right singular vectors of X equal eigenvectors of XᵀX, and singular values squared are proportional to eigenvalues. SVD is numerically more stable.
**Q: Why are PCA directions orthogonal?** The covariance matrix is symmetric, and symmetric matrices have orthonormal eigenvectors.
**Q: What's the best linear projection in reconstruction-error terms?** By the Eckart–Young theorem, the top-k PCA subspace minimizes ‖X − Xₖ‖ for all rank-k projections.

---

## 39. GATE / Exam Perspective

**Key formulas to remember:**

```text
Σ = (1/n) XᵀX (centered data)   →   Covariance
Σ v = λ v                        →   Eigen equation
Score_i = X_i · v_1              →   Projection on PC1
Explained variance = λ1 / Σλ     →   Fraction kept
PCA ≡ SVD of centered X          →   Numerical connection
```

**Common traps:**
- PCA is **unsupervised** (no labels); LDA is supervised — don't confuse.
- Eigenvectors must be **unit length**; eigenvalues rank the components.
- PCA maximizes **variance**, not class separability.
- The number of possible components is min(n−1, p) distinct eigenvectors.
- Standardization matters when feature units differ.

**Representative pattern question (NOT a real PYQ):** "If the eigenvalues of a data covariance matrix are [4, 2, 1, 1], what fraction of total variance is captured by keeping the top 2 components?" → Total = 8; kept = 6; fraction = 6/8 = 0.75. (Never claim this is a past GATE question — verify first.)

---

## 40. Coding Practice

1. **Level 1 — Basics:** Compute the mean, covariance, eigenvalues/eigenvectors of a small matrix with numpy; verify Σv = λv.
2. **Level 2 — From scratch:** Implement `pca_scratch(X, k)` exactly as in §18 and compare scores to sklearn.
3. **Level 3 — Scaling:** Fit PCA with and without StandardScaler on mixed-unit data; compare explained variance.
4. **Level 4 — Choosing k:** Plot cumulative explained variance; pick k at the elbow.
5. **Level 5 — Reconstruction:** Compute reconstruction error at several k; plot error vs k.
6. **Level 6 — Data leakage:** Fit on train, transform test; show the wrong way leaks info.
7. **Level 7 — Real-world case:** Load a real dataset (e.g., faces or digits from sklearn), reduce to 2D, visualize classes, then train a classifier on k-PCA features and report accuracy vs k.

---

## 41. Practical ML Workflow

```text
Problem → Define DR goal (visualize / compress / denoise / preprocess)
  ↓ Data → collect n×p matrix
  ↓ EDA → correlations, distributions, outliers (heatmap of cov/corr)
  ↓ Cleaning → impute missing, remove/contain outliers
  ↓ Feature engineering → encode categoricals if needed
  ↓ Split → train/test (fit PCA only on train!)
  ↓ Preprocess → standardize on train, reuse on test
  ↓ Train → fit PCA, choose k (elbow + downstream CV)
  ↓ Evaluate → explained variance, reconstruction error, downstream score, separability plot
  ↓ Error analysis → inspect loadings; check if non-linearity warrants Kernel PCA
  ↓ Deploy → store scaler + PCA model; transform new data
  ↓ Monitor → drift in explained variance / downstream performance over time
```

---

## 42. Complexity

- **Computing covariance:** O(n p²).
- **Eigendecomposition:** O(p³) for a p×p dense matrix (full solver).
- **Randomized/partial SVD:** O(n p k) roughly, for top-k — much cheaper when k ≪ p.
- **Transforming new data:** O(n k p).
- **Space:** Covariance is O(p²); components O(k p).

**Scaling notes:** PCA scales well with samples (n) but the O(p³) eigen-cost can hurt when p is large — use SVD / randomized solvers. Samples much smaller than features (n < p) yields at most n−1 non-trivial components.

---

## 43. Advanced Concepts

- **SVD connection:** PCA computed via SVD is numerically stabler than forming the covariance explicitly (avoids squaring condition number).
- **Probabilistic PCA (PPCA):** PCA as maximum-likelihood in a linear-Gaussian latent-variable model — handles missing data elegantly.
- **Sparse PCA:** Adds sparsity on loadings for interpretability.
- **Robust PCA:** Decomposes into low-rank + sparse outliers (used in anomaly detection).
- **Kernel PCA:** Non-linear extension via kernel trick (see note 02).
- **Total variance theorem:** trace(Σ) = Σλᵢ = total variance; PCA reallocates it among orthonormal axes.
- **Randomized SVD:** For huge matrices, approximate top singular vectors cheaply.

---

## 44. Connections to Other Algorithms

```text
            PCA
          /  |  \
         /   |   \
    SVD      Kernel PCA    PPCA
(cov via  (non-linear)   (probabilistic
 SVD)        |              view)
             +--- t-SNE / UMAP (visualization alternatives)
             +--- LDA (supervised linear DR)
             +--- NMF (non-negative linear DR)
             +--- K-Means (often applied on PCA embedding)
```

---

## 45. If You Remember Only 5 Things

1. PCA re-orients data onto orthogonal directions of maximum **variance**, ranked by **eigenvalues**.
2. It is **unsupervised and linear**; it uses eigendecomposition of the covariance matrix (or SVD of centered data).
3. The k eigenvectors with the largest eigenvalues are the optimal linear projection for both max-variance and min-reconstruction-error.
4. **Standardize** features with different units, and **fit on training data only** to avoid leakage.
5. Use it to visualize, compress, denoise, and preprocess — but switch to Kernel PCA / t-SNE / UMAP for non-linear structure, and LDA when labels and class separation matter.

---

## 46. Cheat Sheet

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Principal Component Analysis |
| **Category** | Unsupervised linear dimensionality reduction / feature extraction |
| **Goal** | Project data to lower dims preserving maximum variance |
| **Input** | Feature matrix X (n×p) |
| **Output** | Scored embedding (n×k), loadings (k×p), explained variance |
| **Core Formula** | Σv = λv; T = X_c W; explained = λᵢ/Σλⱼ |
| **Loss / Objective** | max vᵀΣv s.t. ‖v‖=1 (equivalently min reconstruction error) |
| **Optimization** | Closed-form eigendecomposition / SVD |
| **Parameters** | Mean vector, component matrix W, eigenvalues |
| **Hyperparameters** | n_components, whiten, svd_solver |
| **Assumptions** | Linearity, variance=info, orthogonality, roughly Gaussian |
| **Advantages** | Fast, interpretable, denoises, decorrelates, no overfitting when k small |
| **Disadvantages** | Linear only, sensitive to scaling/outliers, ignores labels |
| **Use When** | Linear reduction, visualization, denoising, redundancy removal |
| **Avoid When** | Non-linear structure, label-driven separation, need per-feature interpretability |
| **Related** | SVD, Kernel PCA, LDA, NMF, t-SNE, UMAP, PPCA |
| **Key Exam Points** | Eigen equation, explained variance sum, unsupervised, standardize, SVD link |
| **Key Interview Points** | Eigen vs singular vectors, scaling, data leakage, choice of k |

---

## 47. Final Mental Model

```text
  X (n×p) ──centering──▶ X_c ──▶ Σ = XᵀX/n ──▶ eig(Σ) → (λ, v)
                                        │
                                        ▼
              Sort λ desc, take top k, W = [v1..vk]
                                        │
                                        ▼
                 T = X_c W  (n×k)  =  the compressed embedding
                                        │
                                        ▼
        visualize / cluster / classify / reconstruct (X̂ = T Wᵀ + mean)
```

---

## 48. Knowledge Check

### Recall (5)
1. What matrix does PCA decompose?
2. What does an eigenvalue represent?
3. What is the projection formula?
4. Define explained variance ratio.
5. Is PCA supervised or unsupervised?

### Understanding (5)
1. Why standardize before PCA? 
2. Why are components orthogonal?
3. What is the difference between scores and loadings?
4. Why does PCA reduce noise?
5. When does keeping many components cause overfitting?

### Application (5)
1. How would you reduce 500 features to 2 for plotting?
2. How do you choose k?
3. How do you apply a trained PCA to new data?
4. What if data has gross outliers?
5. When would reconstruction error be high despite few components?

### Mathematical (5)
1. Write the covariance matrix formula.
2. Write the eigen-equation and explain λ.
3. Show explained-variance for eigenvalues [5,3,2].
4. Relate PCA to SVD.
5. What does trace(Σ) equal?

### Interview (5)
1. What is the Eckart–Young theorem's role here?
2. How is PPCA different?
3. What is the "curse of dimensionality" and how does PCA help?
4. Why is fitting PCA on the test set wrong?
5. When would you pick Kernel PCA over PCA?

### Problem Solving (5)
1. Eigenvalues [4,2,1,1]: fraction kept by top 2?
2. Two fully-correlated features: how many effective components?
3. 100 samples, 1000 features: how many non-trivial PC's?
4. Which method preserves non-linear manifold?
5. Design a pipeline to visualize labeled high-dim data with PCA + color.

## Answers (explained)
1. The covariance matrix (or SVD of centered data). 2. The variance of data projected onto that eigenvector — the "importance." 3. T = X_c W. 4. λᵢ/Σλⱼ. 5. Unsupervised.
6. Differing units let large-scale features dominate covariance. 7. Symmetric matrices have orthonormal eigenvectors. 8. Scores = projections (positions); loadings = eigenvectors (weights on original features). 9. Low-variance directions are often noise; dropping them smooths data. 10. Keeping near-p components fits noise → variance.
11. PCA(n_components=2) then plot scores. 12. Variance elbow + downstream CV. 13. new_Xc = new_X − stored_mean; then new_Xc @ W. 14. Use Robust PCA / remove outliers / trimmed covariance. 15. When true structure is recessed along low-variance curved directions.
16. Σ=(1/n) XᵀX (centered). 17. Σv=λv; λ = variance. 18. total=10; top two=8 → 0.8. 19. Right singular vectors of X = eigenvectors of XᵀX; singular values² ∝ eigenvalues. 20. trace = Σλᵢ = total variance.
21. It guarantees top-k subspace gives min reconstruction error over rank-k projections. 22. PCA = MLE of a linear-Gaussian latent model; handles missing data. 23. Distances become uninformative in high-d; PCA concentrates signal in few dims. 24. Data leakage inflates apparent quality. 25. When data is non-linear (curved manifold).
26. Fraction = 6/8 = 0.75. 27. One (any correlation=±1 → rank 1 covariance → one eigenvector has all variance). 28. At most min(n−1, p) = 99. 29. t-SNE / UMAP / Kernel PCA. 30. PCA → fit on train → transform → color by label in 2D scatter.

---

## 49. Final Learning Checklist

- [ ] Explain PCA in one sentence
- [ ] Define eigenvector, eigenvalue, covariance, explained variance
- [ ] Write the covariance matrix formula
- [ ] Write and interpret the eigen-equation Σv = λv
- [ ] Derive why eigenvectors maximize projected variance
- [ ] Recompute a 2×2 covariance eigendecomposition by hand
- [ ] Compute explained-variance ratio from eigenvalues
- [ ] State the projection formula T = X_c W
- [ ] State the reconstruction formula X̂ = T Wᵀ + mean
- [ ] Implement PCA from scratch with numpy
- [ ] Implement PCA via SVD
- [ ] Use sklearn PCA (fit, transform, inverse_transform)
- [ ] Choose k via elbow and downstream validation
- [ ] Explain why standardization matters (and when to center only)
- [ ] Explain data leakage when fitting PCA on all data
- [ ] List 3 advantages and 3 disadvantages
- [ ] Contrast PCA vs LDA vs Kernel PCA vs t-SNE vs NMF vs SVD
- [ ] Analyze complexity (n p², p³)
- [ ] Explain the SVD–PCA relationship
- [ ] Apply PCA in a full workflow end-to-end
- [ ] Build a 2D visualization of a real dataset colored by label

---

## 50. Quality Control Note

- **Accuracy:** Formulas and hand-verified examples (2×2 and 4×2 datasets) are correct: covariance, eigenvectors, eigenvalues, explained-variance ratio all recomputed by hand. ✅
- **Beginner-friendliness:** Intuition (§03), beginner definition (§02), and everyday analogies included before math. ✅
- **Math depth:** Full derivation, symbol-by-symbol formula explanation, tiny worked numerical example. ✅
- **Practical depth:** From-scratch code, sklearn code, hyperparameters, workflow, coding practice ladder, failure cases. ✅
- **Exam depth:** GATE-style traps, representative pattern questions (clearly marked not real PYQs), interview at 3 levels. ✅
- **Structure:** All 50 template sections present in order. ✅
- **Consistency:** Terminology defined before use; dimensional-reduction framing (objective/input/output/eval) applied throughout. ✅
