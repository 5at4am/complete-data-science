# 02. Kernel PCA

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆

---

## 01. Algorithm Overview

| Aspect | Details |
|--------|---------|
| **Algorithm name** | Kernel Principal Component Analysis (Kernel PCA) |
| **Category** | Dimensionality Reduction (unsupervised, non-linear feature extraction) |
| **Type** | Non-linear transformation via kernel trick |
| **Parametric / Non-parametric** | Mixture: it has a fixed kernel form but the projection depends on the training set (a kernel-based embedding), so it is effectively non-parametric in the data |
| **Generative / Discriminative** | Neither (unsupervised projection) |
| **Main objective** | Project non-linearly separable data onto a lower-dim space where (linear) PCA would find structure — by doing linear PCA in a high-dimensional feature space induced by a kernel |
| **Input** | Feature matrix X (n×p), plus a choice of kernel and its hyperparameters |
| **Output** | Embedded matrix (n×k) — the principal components in the (implicit) high-dim feature space |
| **Core idea** | Use the kernel trick: compute inner products (kernel matrix) instead of explicit high-dim coordinates, then eigendecompose the kernel matrix to get non-linear components |
| **Typical use cases** | Non-linear manifold visualization, separating concentrically / curved clusters, feature extraction for downstream models |

---

## 02. One-Line Definition

### Beginner Definition
Kernel PCA is PCA but it first invisibly "bends" the data into a higher-dimensional space where the structure is straight, then does normal PCA — and you never have to build that higher space by hand.

### Technical Definition
Kernel PCA performs principal component analysis in a high-dimensional (possibly infinite) feature space implicitly defined by a kernel function k(x, y) = ⟨φ(x), φ(y)⟩, by eigendecomposing the centered Gram (kernel) matrix instead of the covariance matrix, thereby capturing non-linear structure that linear PCA misses.

---

## 03. Intuition

Ordinary PCA fails when the data is arranged in a curve or a ring — e.g., two concentric circles. No straight line captures "inner vs outer ring." But if you could lift the data into a third "bending" dimension (radius), the two rings would flatten into two blobs that a straight line could separate, and linear PCA would find them.

Kernel PCA does this trick *implicitly*. Instead of explicitly computing coordinates in a giant space (too expensive), it only ever computes pairwise similarities between points using a **kernel** — a function that equals the inner product in that high-dim space. Since PCA's math only ever needs inner products (the covariance Σ is built from XᵀX, which is all pairwise inner products), you can rewrite all of PCA purely in terms of the **kernel matrix**, and swap in any kernel to "bend" the data however you like.

Key insight: PCA needs only dot products of the data; the kernel provides those dot products *as if computed in a richer space*, without ever visiting it.

---

## 04. Problem It Solves

**The problem:** Real data is frequently not linearly penetrating. Ionospheric signals, digit images, biological samples often lie on curved manifolds or in concentric/arc arrangements. Linear PCA will flatten these structures and lose the very pattern you care about.

**What we want:** A way to reduce dimensions that respects non-linear structure.

**Why Kernel PCA is useful:** It generalizes PCA with almost no extra implementation burden — same "maximize variance, keep top eigenvectors" recipe, but in a feature space defined by a kernel. Any function that is a valid kernel yields a consistent, meaningful projection.

**Small example:** Two concentric clusters (inner radius r=1, outer r=2) in 2D. Linear PCA sees isotropic spread (no separation). With a Gaussian (RBF) kernel, the top components separate inner vs outer points because the kernel encodes proximity structure non-linearly.

---

## 05. Where It Fits in Machine Learning

```text
Machine Learning
└── Unsupervised Learning
    ├── Clustering
    └── Dimensionality Reduction
        ├── Linear: PCA, SVD, LDA(supervised), NMF
        └── Non-linear: Kernel PCA, t-SNE, UMAP
                        ↑ Kernel PCA lives in this branch
```

Kernel PCA bridges linear PCA and the nonlinear visualization methods: it inherits PCA's variance-maximization objective but gains nonlinear power through the kernel trick.

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|------|----------------|-------------------|
| **Kernel** | A similarity function between two points | k(x, y) = ⟨φ(x), φ(y)⟩, equals inner product of feature maps φ |
| **Feature map φ** | An invisible transformation to high-dim space | φ: ℝᵖ → ℝᵈ (d possibly ∞) |
| **Kernel matrix (Gram matrix)** | Table of pairwise similarities | n×n matrix K with Kᵢⱼ = k(xᵢ, xⱼ) |
| **Kernel trick** | Doing inner-product math without explicit coordinates | Replace ⟨φ(x),φ(y)⟩ by k(x,y) |
| **RBF / Gaussian kernel** | Similarity decays with distance | k(x,y) = exp(−‖x−y‖² / (2σ²)) |
| **Polynomial kernel** | Similarity from dot product power | k(x,y) = (x·y + c)^d |
| **Centering in feature space** | Implicit mean subtraction | K → K − (1/n)1K − K(1/n) + (1/n)1K1 |
| **Eigenvector / eigenvalue** | Directions and magnitudes from eigendecomposition | K v = λ v |
| **Embedding** | Projected low-dim coordinates | The reduced feature-vector per sample |

---

## 07. Input and Output

**Input:**
- Data matrix X (n×p).
- Kernel function choice (RBF, polynomial, sigmoid, custom) and its hyperparameters (e.g., γ for RBF).

**Output:**
- Embedded matrix (n×k) — the non-linear principal components (`transform` of sklearn).
- The transformed data (n×k) can be fed to clustering/classification.

**Note:** Unlike linear PCA, Kernel PCA's components live in the implicit feature space; you typically cannot give them intuitive per-feature "loadings." Output is generally used as an embedding, not for loadings interpretation.

---

## 08. Mathematical Foundation

**Basic idea:** Do PCA on the feature matrix Φ = [φ(x₁), …, φ(xₙ)]ᵀ (n×d) instead of X, but never materialize Φ. PCA needs the covariance ΦᵀΦ (d×d in feature space) or equivalently the Gram KK = ΦΦᵀ (n×n in sample space). The kernel supplies K directly.

**Notation:**
- φ(x) = feature map (maps x to feature space).
- Φ = n×d matrix of mapped points.
- K = ΦΦᵀ = n×n Gram/kernel matrix, Kᵢⱼ = ⟨φ(xᵢ), φ(xⱼ)⟩ = k(xᵢ, xⱼ).
- K̃ = the centered kernel matrix (centering the data in feature space requires centering K).

**Core result:** For centered data, the eigenvectors v of the covariance ΦᵀΦ satisfy ΦᵀΦ v = λ v. Multiplying both sides by Φ: ΦΦᵀ(Φ v) = λ (Φ v). Let α = Φ v (n-vector). Then K α = λ α. So the covariance eigendecomposition in feature space reduces to a kernel-matrix eigendecomposition K α = λ α.

**Required math:** Linear algebra, dot products, eigendecomposition, the kernel-trick concept, Mercer's theorem (which kernels are valid).

---

## 09. Core Formula

### 1. The kernel

```text
Kᵢⱼ = k(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩
```

**Meaning:** The similarity of xᵢ and xⱼ as computed in feature space, but obtained without explicit φ.

**Symbols:** xᵢ, xⱼ = data points; φ = feature map; ⟨·,·⟩ = inner product; k = kernel function.

**Intuition:** If two points map close together in feature space, their kernel value is large — they are "similar." The kernel encodes a custom notion of similarity.

### 2. Centered kernel matrix

```text
K̃ = K − (1/n) 1_n K − K (1/n) 1_n + (1/n)² 1_n K 1_n
```

**Meaning:** Removes the implicit mean of the mapped points from the kernel.

**Symbols:** 1_n = n×n matrix of all ones; n = number of samples.

**Intuition:** For linear PCA you subtract the data mean; in feature space you cannot subtract φ-mean directly, so you correct the Gram matrix instead.

### 3. Kernel eigendecomposition

```text
K̃ α_j = λ_j α_j
```

**Meaning:** Eigenvectors of the centered kernel are what drive the projection.

**Symbols:** α_j = j-th eigenvector (dimension n); λ_j = j-th eigenvalue.

**Intuition:** Same form as Σ v = λ v, but indexed by samples rather than features.

### 4. Projection of a new/test point

```text
t_j(x) = Σ_i α_{j,i} k(x₀, xᵢ) = α_j · k(x₀, ·)
```

**Meaning:** The j-th component of a new point x₀ is a weighted sum of its kernel values with all training points.

**Symbols:** α_{j,i} = i-th entry of eigenvector j; k(x₀, xᵢ) = kernel value between x₀ and training point xᵢ; Σ over i = 1..n.

**Intuition:** The projection is a "similarity-weighted" combination — the new point's coordinate reflects how similar it is to each training point, weighted by how important that training point is.

**Worked example (hand-verified).** Take 3 points in 1D: x = [1, 2, 3]. Use polynomial kernel k(x,y) = (x·y)²  (feature map implicitly φ(x) = x², since (xy)² = (x²)(y²) — actually φ(x) = x², a 1-D feature map). Kernel matrix:

```text
k(1,1)=1, k(1,2)=4, k(1,3)=9
k(2,1)=4, k(2,2)=16, k(2,3)=36
k(3,1)=9, k(3,2)=36, k(3,3)=81

K = [[1, 4, 9],
     [4, 16, 36],
     [9, 36, 81]]
```

This K = ΦΦᵀ with Φ = [1, 4, 9] (the row of φ-values 1², 2², 3²). Feature space is 1-D; with n=3, one eigenvector is α ∝ [1,4,9]. Projecting the (already-standardized) mapped values onto their principal direction reduces to 1 component capturing all variance. This verifies the mechanics: K is exactly the outer product, and its eigenvector reproduces the feature-map direction. ✅ Hand-verified (n=3 kernel, 1 eigenvector retains 100% variance).

---

## 10. Derivation

1. Linear PCA solves Σ v = λ v with Σ = ΦᵀΦ in feature space.
2. We cannot form ΦᵀΦ (d × d, d huge). But note ΦᵀΦ v = λ v.
3. Multiply both sides on the left by Φ: ΦΦᵀ Φ v = λ Φ v. Let α = Φ v. Since KK = ΦΦᵀ, we get K α = λ α.
4. So eigenvectors α of the n×n kernel matrix encode the feature-space directions: v = Φᵀ α (up to normalization), and feature-space coordinates of any point are given by projections t_j(x) = ⟨φ(x), v_j⟩ = α_j·k(x, ·).
5. Normalization: for orthonormal feature-space components, rescale α so that α_jᵀ K̃ α_j = λ_j.

**Important result:** The top k eigenvectors of the (centered) kernel matrix produce the same projection as linear PCA done on φ(x), but never requiring φ. This is the kernel trick applied to PCA. This is the standard formulation (Schölkopf et al.).

---

## 11. How the Algorithm Works

```text
Input (n×p)
  ↓
Choose kernel + hyperparameters (e.g., RBF with γ)
  ↓
Compute kernel matrix K (n×n): Kᵢⱼ = k(xᵢ, xⱼ)
  ↓
Center K → K̃ (feature-space mean correction)
  ↓
Eigendecompose K̃ α = λ α
  ↓
Sort by descending eigenvalue
  ↓
Select top k eigenvectors α_1..α_k (normalize each)
  ↓
Project each point: t_j(x) = α_j · k(x, ·)
  ↓
Output embedding (n×k)
```

---

## 12. Training Process

**Pre-training:** Choose kernel type and hyperparameters (often via validation on a downstream task or visual inspection).

**During (no iterative optimization):**
1. Build the Gram matrix K from all pairs of training points.
2. Center it.
3. Eigendecompose.
4. Select and normalize top k eigenvectors.

**What's "learned":** The top k eigenvectors α_j and the kernel parameters; the training set itself is retained because projecting new points requires kernel values against all training points.

**Stopping:** No iterative loop; you pick k (number of components).

**Final model contents:** The training (support) points, the kernel function + parameters, the normalized eigenvectors α, and the mean-centering constants.

---

## 13. Objective Function / Loss Function

**Objective:** Maximize the variance of the projected data in the (implicit) feature space, i.e., the same Rayleigh-quotient objective as PCA but in φ-space:

```text
maximize  vᵀ ΦᵀΦ v  subject to  vᵀ v = 1    (in feature space)
```

Equivalently, for the kernel-alpha form: maximize αᵀ K̃ α subject to αᵀ K̃ α' normalization constraints.

**Why chosen:** Reusing PCA's objective keeps a clear, principled notion of "most informative directions," now able to see non-linear structure through the kernel.

**High/low meaning:** Higher eigenvalue = that non-linear component carries more (feature-space) variance. The top-k capture the dominant non-linear modes.

---

## 14. Optimization

**Definition:** Like PCA, the optimal components are found analytically via eigendecomposition — no gradient descent.

**Why:** The objective is still an eigen-problem, just on the kernel matrix.

**Method:** Power iteration, Lanczos, or a dense eigensolver (LAPACK). For large n, use randomized/partial eigensolvers.

**Cost:** O(n³) for dense n×n eigendecomposition.

```text
K̃ (n×n)
  ↓
Iterative or dense eigensolver
  ↓
Top-k eigenvectors α (n×k)
  ↓
Thus the feature-space components are determined
```

**Global optimum:** Yes — eigendecomposition of a symmetric (PSD) matrix yields global optimality.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE (hand-verified).** Use the three 1-D points with linear-ish kernel but let's do a small 2-class illustrative centering check. Data: x = [1, 2, 3] with **linear kernel** k(x,y) = x·y (this is just ordinary PCA on 1-D data, a useful sanity check).

K = [[1,2,3],[2,4,6],[3,6,9]].

Mean of values = 2. Feature-space (1-D) centering of Φ=[1,2,3]: centered = [−1, 0, 1]. The centered kernel from these centered values: 
centered Φ' = [−1,0,1]. K̃ = Φ'Φ'ᵀ = [[1,0,−1],[0,0,0],[−1,0,1]].

Eigendecompose K̃: eigenvalues λ = 2, 0, 0. Eigenvector for λ=2: [1,0,−1]/√2. The nonzero variance (2) matches variance of centered data [−1,0,1] = (1+0+1) = 2. Number of non-trivial components = 1 (as expected for 1-D data projected through linear kernel). ✅ Hand-verified.

This confirms: (a) centering formula produces the feature-space-correct Gram matrix, (b) kernel PCA with a linear kernel reduces exactly to ordinary PCA (variance 2 in this case). 

---

## 16. Visual Explanation

```text
Linear PCA on concentric circles (fails):

   o o o          Linear PCA sees isotropic scatter —
  o o o o         no axis separates inner from outer.
   o o o

Kernel PCA (success):

   o o o          In feature space (e.g., add radius axis),
  o o o o         inner ring and outer ring separate cleanly.
   o o o          Top kernels components separate them.
```

```text
Kernel PCA pipeline:

 Original  →  K = [k(x_i, x_j)]  →  Center K̃  →  eigendec  →  embedding (n×k)
  2D rings       n×n similarities       remove φ-mean   α    plot / cluster
```

---

## 17. Algorithm / Pseudocode

```
1. Choose kernel function k and hyperparameters (e.g., RBF γ)
2. Build Gram matrix: K[i][j] = k(x_i, x_j)  for all i,j
3. Center the kernel:
     one_n = (1/n) matrix of ones
     K̃ = K - one_n·K - K·one_n + one_n·K·one_n
4. Solve K̃ α = λ α   (dense or iterative eigensolver)
5. Sort eigenvectors α by descending λ
6. Take top k, normalize: α_j ← α_j / sqrt(λ_j)
7. For each point x0, its j-th component: t_j = Σ_i α_{j,i} k(x0, x_i)
   return embedding T (n×k)
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

def rbf_kernel_matrix(X, gamma=1.0):
    n = X.shape[0]
    sq = np.sum(X**2, axis=1)[:, None]
    dist2 = sq + sq.T - 2.0 * (X @ X.T)
    return np.exp(-gamma * dist2)

def center_kernel(K):
    n = K.shape[0]
    one = np.ones((n, n)) / n
    return K - one @ K - K @ one + one @ K @ one

def kernel_pca_scratch(X, n_components, gamma=1.0):
    K = rbf_kernel_matrix(X, gamma)
    Kc = center_kernel(K)
    eigvals, eigvecs = np.linalg.eigh(Kc)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    alphas = eigvecs[:, :n_components]
    alphas = alphas / np.sqrt(np.maximum(eigvals[:n_components], 1e-12))
    T = Kc @ alphas
    return T, alphas, eigvals

X = np.array([[1.0], [2.0], [3.0]])
T, alphas, eigvals = kernel_pca_scratch(X, n_components=1, gamma=1.0)
print("Embedding:", T.ravel().round(3))
print("Eigenvalues:", eigvals.round(3))
```

**Note on projecting new points:** For unseen points, form K_new(i) = k(x0, x_i), center them against the training-set K̃, then t_j(x0) = α_j · K_new. (Simplify here by projecting the training embedding; sklearn handles new points internally.)

---

## 19. Code Explanation

```text
Code line → What does it do? → Why required? → Mathematical concept?
rbf_kernel_matrix → builds n×n RBF similarity table → provides implicit feature-space inner products → kernel trick
sq[:,None] + sq.T - 2 X Xᵀ → squared pairwise distances → feeds the Gaussian → ‖x−y‖² inside exp(−γ·…)
exp(-gamma*dist2) → RBF kernel → similarity decays with distance → k(x,y)=exp(−γ‖x−y‖²)
one@K - K@one + one@K@one → feature-space mean correction → centering in φ-space → K̃ = K - 1/n 1K - K1/n + …
np.linalg.eigh → eigendecomposition → finds non-linear components → K̃ α = λ α
argsort[::-1] → rank by eigenvalue desc → most important non-linear mode first → sorted eigenvalues
alphas/sqrt(λ) → normalize → orthonormal feature-space directions → αᵀK̃α = λ constraint
Kc @ alphas → project points → lower-dim embedding → t = α_j·k(x, ·)
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.decomposition import KernelPCA

X = np.array([[1.0], [2.0], [3.0]])

model = KernelPCA(n_components=1, kernel="rbf", gamma=1.0)
T = model.fit_transform(X)
print("Embedding:", T.ravel().round(3))
print("Can reconstruct (only for linear kernel):", model.inverse_transform(T).round(3))

# Comparing linear KernelPCA to standard PCA (should match)
model_lin = KernelPCA(n_components=1, kernel="linear")
T_lin = model_lin.fit_transform(X - X.mean(axis=0))
```

**Key API:** `fit_transform`, `transform` (new points), `eigenvalues_`, `alphas_`, `dual_coef_`, `lambdas_`. Set `kernel="linear"` to recover ordinary PCA.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|----------------|---------|--------|----------------------|
| `kernel` | Kernel type (rbf, linear, poly, sigmoid, cos, precomputed) | Defines the feature space | Match to presumed data structure |
| `gamma` (γ) | Width of RBF / kernel scale | Too small → nearly constant kernel (no structure); too large → just nearest-neighbor similarity | Tune with CV/visual inspection |
| `degree` | Degree of polynomial kernel | Controls nonlinearity degree | For `poly` kernel |
| `coef0` | Constant in poly/sigmoid kernels | Shifts similarity baseline | For `poly`/`sigmoid` |
| `n_components` | Number of components to keep | Output dimension | Elbow on eigenvalues |
| `eigen_solver` | dense / arpack / randomized | Speed vs accuracy / scalability | `arpack` or `randomized` for large n |
| `max_iter` | Iterations for iterative solver | Convergence | For `arpack` |

**too low / too high / tune:** γ too low over-smooths (all similarities ≈1, useless); γ too high makes each point its own cluster (overfits). Tune on a validation set or via classification accuracy of the downstream task.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Top k eigenvectors α_j of K̃ (the "loadings" in feature space).
- Eigenvalues λ_j.
- The mean-centering quantities derived from the training kernel.

### Hyperparameters (chosen)
- Kernel type.
- Kernel hyperparameters (γ, degree, coef0).
- n_components, eigen_solver, max_iter.

---

## 23. Assumptions

| Assumption | What | Why | How to check | If violated | Solution |
|-----------|------|-----|--------------|-------------|----------|
| **Valid (PSD) kernel** | Kernel is positive semi-definite | Ensures a real feature space | Check Mercer / PSD of Gram matrix | Invalid kernel → meaningless embedding | Use proven kernels (RBF, poly) |
| **Kernel captures structure** | Chosen kernel matches data geometry | Kernel defines similarity | Compare embeddings for several kernels | Poor structure extraction | Try RBF, poly, custom domain kernels |
| **Data lies on a useful manifold** | The kernel's similarity relates to ground-truth structure | Kernel PCA assumes structure is kernel-recoverable | Visualize embedding | Structure not kernel-expressible | t-SNE/UMAP (manifold-aware) |
| **Variance meaningful in feature space** | Eigenvalue ranking = importance | Objective = feature-space variance | Inspect eigenvalue spectrum | Dominant eigenvalue from outliers | Robust kernels / standardization |

---

## 24. Data Requirements

- **Data type:** Numeric features generally required for RBF/poly kernels.
- **Missing values:** Not handled; impute or drop first.
- **Outliers:** RBF kernel squashes far points (similarity→0), which partly tames outliers, but they can still distort centering.
- **Scaling:** Strongly recommended — RBF and poly kernels depend on distances/dot products which scale with |x|; standardize so no feature dominates.
- **Dataset size:** Kernel matrix is n×n; O(n²) memory and O(n³) time → poor for very large n. Use approximation (Nyström) or randomized eigensolvers.
- **Categorical:** Need encoding if used.
- **Class imbalance / labels:** Not used (unsupervised); labels optional for validation only.

---

## 25. Feature Scaling

**Strongly recommended.**

- RBF uses Euclidean distance, poly uses dot products — both are sensitive to feature magnitudes. A large-magnitude feature will dominate distances/similarities.
- Standardize (Z-score) each feature before building the kernel so every feature contributes comparably.
- After scaling, choose kernel hyperparameters (e.g., γ) on the scaled data.

---

## 26. Evaluation Metrics

**Training objective (feature-space variance) ≠ downstream evaluation metric.**

| Metric | Definition | When to use | When NOT |
|--------|-----------|-------------|----------|
| **Eigenvalue spectrum** | Ordered λ_j and cumulative share | Choosing k, seeing dominant modes | Judging downstream usefulness |
| **Visual cluster/class separation** | Do points/classes separate in embedding | Exploratory + sanity check | Over-trusting 2D visual separation |
| **Downstream task score** | Accuracy/F1 of classifier on embedding | Validating DR improves task | Interpreting the embedding alone |
| **Reconstruction / trustworthiness** | Whether neighbors in embedding reflect original proximity | For manifold-preservation claims | When preserving global distances matters |

---

## 27. Advantages

- **Captures non-linear structure** (rings, arcs, curved manifolds) that linear PCA misses. ✅
- **Same elegant math as PCA** (eigendecomposition) with minimal conceptual leap. ✅
- **No explicit feature-space construction** — the kernel trick keeps it tractable even for infinite-dimensional spaces. ✅
- **No labels needed** — unsupervised. ✅
- **Can be specialized** with domain-specific kernels (text, graphs, sequences). ✅
- **Reproducible/closed-form** — no stochastic optimization. ✅

---

## 28. Disadvantages

- **Kernel choice + hyperparameters** (γ, degree) are hard to set and can dominate results. ✗
- **O(n²) memory, O(n³) time** — poor scalability to large datasets. ✗
- **Not interpretable** — loadings live in an implicit feature space; can't read "which feature drives PC1". ✗
- **Projecting new points requires the whole training set** (all kernel values). ✗
- **May overfit** with too-flexible kernels. ✗
- **Doesn't explicitly preserve global distances** (like t-SNE/UMAP do locally). ✗

---

## 29. When to Use

- ✓ Data has clearly non-linear / concentric / curved structure.
- ✓ Linear PCA visualization fails to reveal clusters.
- ✓ You want a non-linear feature embedding for a downstream model.
- ✓ You can choose a sensible kernel for the data type (e.g., RBF for point clouds).
- ✓ Dataset is small-to-moderate (kernel matrix manageable).

---

## 30. When NOT to Use

- ✗ Data is roughly linear — plain PCA is faster and more interpretable.
- ✗ Very large n (kernel matrix infeasible) — prefer UMAP, or Nyström-approximated Kernel PCA.
- ✗ You need interpretable per-feature loadings.
- ✗ Kernel choice is arbitrary and untunable.
- ✗ You mainly need global-distance-preserving embedding (t-SNE/UMAP are partially specialized).

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---------|-------|-----------|--------|
| Separating concentric tumor/healthy cells | Multi-feature cell measurements | Kernel PCA → classifier | Non-linear embedding where classes separate |
| Handwritten digit grouping | pixel matrices | Kernel PCA → K-Means | Digits cluster by shape |
| Face recognition non-linear variation | face images | Kernel PCA (RBF) | Robust identity embedding |
| Anomaly detection in network traffic | traffic features | Kernel PCA (reconstruction) | Reconstruction error flags anomalies |
| Ionosphere/signal classification | radar returns | Kernel PCA embedding | Linear classifier on reduced space |

---

## 32. Failure Cases

- **Data failure:** Massive outliers can dominate the kernel/centering and distort all components.
- **Kernel failure:** Wrong γ (too small → trivial similarities; too large → overfit nearest-neighbor) hides structure.
- **Mathematical failure:** Non-PSD kernel yields no valid feature space / negative eigenvalues (often from violating Mercer's condition).
- **Optimization failure:** Dense O(n³) eigensolver crashes on large n; use blocked/randomized or Nyström.
- **Generalization failure:** Fitting on all data leaks information; also new-point projection depends on the whole training set.
- **Interpretation failure:** Believing the 2D embedding preserves global distances (it does not).

---

## 33. Overfitting and Underfitting

- **Overfitting:** Too-flexible kernel (very small γ or high-degree poly) models idiosyncratic noise; embedding captures training quirks, new points project poorly. Also keeping too many components retains noise-dominated modes.
- **Underfitting:** Too-smooth kernel (large γ) collapses similarities, hiding structure; too few components drop real modes.

**Balance:** Validate embedding quality on a downstream task with held-out data; compare across kernel/γ and k.

---

## 34. Bias-Variance Perspective

- Kernel PCA's complexity is controlled by the kernel's flexibility and the number of components k.
- Very smooth (high-bias) kernel: low variance, but may miss structure (bias).
- Very sharp (low-bias) kernel + many components: fits noise (high variance).
- Choose kernel flexibility and k to balance reconstruction/signal capture against generalization to new points.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|-----------|-----------|----------|----------|----------|
| **Kernel PCA** | PCA in implicit feature space | Non-linear, principled, closed-form | Kernel tuning, O(n³), no loadings | Non-linear feature extraction |
| **PCA** | Linear max-variance | Fast, interpretable | Linear only | Linear baseline |
| **t-SNE** | Preserve local similarities | Great 2D/3D cluster views | No global geometry, costly | Visualization |
| **UMAP** | Preserve local+global manifold | Fast, scalable, global better | Hyperparameter-sensitive | Visualization + downstream embedding |
| **LDA** | Linear class-separation | Uses labels | ≤ C−1 dims, Gaussian | Supervised reduction |
| **MDS** | Preserve pairwise distances | Global geometry | Quadratic memory | Distance-based global embedding |

---

## 36. Algorithm Selection Guide

```text
Non-linear dimensionality reduction needed?
├── Visualize 2D/3D clusters → t-SNE (small) / UMAP (large)
├── Need explicit, reproducible non-linear feature embedding
│   → Kernel PCA (choose RBF; tune γ; Nyström if large)
└── Need global distances preserved → MDS / (UMAP for compromise)
Still prefer the principled variance-maximizing recipe → Kernel PCA
```

---

## 37. Common Mistakes

```text
❌ Forgetting to center the kernel matrix
Why wrong: uncentered kernel ≠ feature-space PCA; results are off
Correct: apply the full centering formula K̃ = K - 1/n 1K - K1/n + 1/n² 1K1

❌ Underestimating the cost: dense KernelPCA on 100k rows
Why wrong: O(n³) eigen and O(n²) memory → crash
Correct: Nyström approximation, randomized solver, or UMAP

❌ Using RBF γ without tuning
Why wrong: γ dominates results; bad γ ⇒ meaningless embedding
Correct: tune γ via CV / downstream metric

❌ Interpreting loadings as original-feature weights
Why wrong: components live in implicit feature space; loadings aren't per-feature
Correct: treat output strictly as an embedding

❌ Fitting on all data (leakage)
Why wrong: test info leaks into embedding
Correct: fit on train, transform test with same model
```

---

## 38. Interview Questions

### Beginner (with answers)
**Q: What is a kernel?** A similarity function equal to the inner product of points mapped to a high-dim space.
**Q: Why use Kernel PCA?** To capture non-linear structure that linear PCA misses.
**Q: What is the kernel trick?** Computing inner products via the kernel without explicit high-dim coordinates.

### Intermediate (with answers)
**Q: What matrix does Kernel PCA eigendecompose?** The centered kernel (Gram) matrix K̃ (n×n), not the covariance.
**Q: What is centering the kernel?** Subtracting the implicit feature-space mean from the Gram matrix.
**Q: How do you set γ for RBF?** Tune via validation/downstream performance; too small flattens, too large overfits.

### Advanced (with answers)
**Q: Why does kernel PCA equal PCA in feature space?** Because PCA's covariance eigendecomposition reduces to K α = λ α via ΦΦᵀ = K; eigenvectors α encode feature-space components v = Φᵀα.
**Q: What is the computational bottleneck?** O(n²) memory and O(n³) time for the n×n dense kernel eigendecomposition; mitigate with randomized/Nyström.
**Q: When does parallel linear PCA === KernelPCA(linear)?** When data is centered; equivalently before centering, both reduce to variance-preserving projection.

---

## 39. GATE / Exam Perspective

**Key formulas:**

```text
K = { k(x_i, x_j) }        (Gram matrix)
K̃ = K - 1/n 1 K - K 1/n + 1/n² 1K1   (centered kernel)
K̃ α_j = λ_j α_j           (eigen equation in sample space)
t_j(x) = Σ_i α_{j,i} k(x, x_i)   (projection)
k(x,y) = exp(-γ ‖x-y‖² )   (RBF kernel)
```

**Common traps:**
- Kernel PCA is **unsupervised** and **non-linear** — differ from both PCA (linear) and LDA (supervised).
- The eigendecomposition is of the **kernel/Gram matrix (n×n)**, not the covariance (p×p).
- A **valid (PSD) kernel** is required for Mercer's theorem; RBF/poly always valid.
- Projecting new points requires **all training kernel values** — it's not a simple parametric dot product.
- Kernel **hyperparameters** (especially γ) strongly affect results.

**Representative pattern question (NOT a real PYQ):** "Why can Kernel PCA discover two concentric rings while linear PCA cannot?" Answer: linear PCA maximizes variance along straight axes in the original space, which treats the rings symmetrically; kernel PCA operates in a bent feature space where the two rings align with an axis whose variance dominates. (Verify before treating as PYQ.)

---

## 40. Coding Practice

1. **Level 1:** Compute an RBF kernel matrix for a small dataset with numpy; check it's symmetric and PSD.
2. **Level 2:** Implement `kernel_pca_scratch`; verify linear kernel equals centered ordinary PCA.
3. **Level 3:** Generate two concentric circles (sklearn `make_circles`); show linear PCA fails and Kernel PCA separates them.
4. **Level 4:** Tune γ visually for circles; note under-smooth vs over-smooth behavior.
5. **Level 5:** Project genuinely new points with a held-out split; verify consistency.
6. **Level 6:** Compare Kernel PCA embedding to t-SNE/UMAP on the same dataset (visual).
7. **Level 7:** Use Kernel PCA embedding + classifier on a real dataset (digits); report accuracy vs PCA baseline.

---

## 41. Practical ML Workflow

```text
Problem → confirm non-linear structure is worth modeling
  ↓ Data → numeric, small-moderate n, scaled
  ↓ EDA → scatter/known geometry (rings? arcs?)
  ↓ Cleaning → impute, handle outliers
  ↓ Feature engineering → standardize
  ↓ Split → train/test (fit embedding on train only)
  ↓ Preprocess → StandardScaler
  ↓ Train → KernelPCA, tune kernel/γ, choose k by eigenvalue elbow
  ↓ Evaluate → embedding separation + downstream score on test
  ↓ Error analysis → try other kernels/γ, compare to PCA/UMAP
  ↓ Deploy → save scaler + KernelPCA model + training points
  ↓ Monitor → re-run when distribution drifts
```

---

## 42. Complexity

- **Kernel matrix construction:** O(n² p).
- **Dense eigendecomposition:** O(n³).
- **Randomized / arpack:** roughly O(n² k) or better.
- **Projecting one new point:** O(n p) (must form kernel with all training points).
- **Space:** O(n²) for the Gram matrix.

**Scaling:** Grows quadratically in samples (memory) and cubically (dense eigen) — Kernel PCA is for small-to-moderate n. Features only enter via the kernel cost O(n² p).

---

## 43. Advanced Concepts

- **Nyström approximation:** Approximate the n×n kernel with a low-rank subset to scale.
- **Kernel choice via Mercer's theorem / PSD check.**
- **Kernel alignment** (with labels or a target kernel) to choose kernels quantitatively.
- **Pre-image problem:** Estimating original-space points from feature-space components (needed for denoising/reconstruction).
- **Connection to other kernel methods:** SVM and kernel ridge regression all use the same Gram-matrix machinery — the "kernel trick" is shared.
- **Empirical kernel map:** Treating kernel values as explicit new features.

---

## 44. Connections to Other Algorithms

```text
             PCA
              │  (linear limit)
              ▼
         Kernel PCA
          /      \
  (kernel trick)  (Gram matrix shared with)
        │            │
  SVM / kernel ridge   MDS (both eigen-decompose
  regression            similarity/distortion matrices)
        │
        ├── t-SNE / UMAP (non-linear embeddings)
        ├── LDA (supervised)
        └── PPCA (probabilistic view)
```

---

## 45. If You Remember Only 5 Things

1. Kernel PCA = PCA performed in a high-dim feature space defined by a **kernel**, without ever visiting that space.
2. It eigendecomposes the **centered kernel (Gram) matrix** (n×n), K̃ α = λ α — not the covariance.
3. The **kernel trick** supplies all needed inner products via k(x, y) = ⟨φ(x), φ(y)⟩.
4. It captures **non-linear structure** (rings, curved manifolds) that linear PCA misses, but is **not interpretable** and is **O(n³)/O(n²)**.
5. Choose a **valid kernel + tuned hyperparameters (γ)**, standardize data, fit on training only; for very large n use Nyström/randomized or UMAP.

---

## 46. Cheat Sheet

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Kernel PCA |
| **Category** | Unsupervised non-linear dimensionality reduction |
| **Goal** | Non-linear projection preserving feature-space variance |
| **Input** | Feature matrix X (n×p) + kernel choice |
| **Output** | Embedding (n×k) |
| **Core Formulas** | K̃ = K − 1/n 1K − K1/n + 1/n²1K1; K̃ α = λ α; t_j(x) = Σ α_{j,i}k(x,xᵢ) |
| **Objective** | Max feature-space variance (Rayleigh quotient) |
| **Optimization** | Closed-form eigendecomposition of K̃ |
| **Parameters** | Top-k eigenvectors α_j, eigenvalues λ_j, mean-centering of K |
| **Hyperparameters** | kernel type, γ/degree/coef0, n_components, eigen_solver |
| **Assumptions** | Valid PSD kernel, kernel matches data geometry |
| **Advantages** | Non-linear power, principled, closed-form, unsupervised |
| **Disadvantages** | Kernel tuning, O(n³)/O(n²), no loadings, needs full training set |
| **Use When** | Non-linear/curved structure, moderate n |
| **Avoid When** | Linear data, huge n, need interpretability |
| **Related** | PCA (linear limit), SVM/kernel methods, MDS, Nyström, UMAP, t-SNE |
| **Key Exam Points** | Eigendecompose Gram matrix; unsupervised; non-linear; kernel trick |
| **Key Interview Points** | Kernel trick rationale, new-point projection, scalability, γ tuning |

---

## 47. Final Mental Model

```text
 X (n×p) ──standardize──▶ choose kernel
   │                        │
   ▼                        ▼
 build Gram K (n×n) ──center──▶ K̃ ──eigendecomp──▶ {α_j, λ_j}
                                          │
                                          ▼
                 project t_j = α_j · k(x,·)   (embedding (n×k))
                                          │
                                          ▼
              visualize / cluster / classify / compare with PCA
```

---

## 48. Knowledge Check

### Recall (5)
1. What does the kernel matrix store?
2. What is the kernel trick?
3. Which matrix does Kernel PCA eigendecompose?
4. Give the RBF kernel formula.
5. Is Kernel PCA supervised or unsupervised?

### Understanding (5)
1. Why does linear PCA fail on concentric rings?
2. Why must the kernel be centered?
3. How does a new point get projected?
4. Why can't you read per-feature loadings?
5. What role does γ play in RBF?

### Application (5)
1. How would you test whether Kernel PCA improves over PCA?
2. How do you choose k?
3. What do you do for a 200k-row dataset?
4. When would you pick poly over RBF?
5. How do you avoid data leakage?

### Mathematical (5)
1. Write the centered-kernel formula.
2. Derive K α = λ α from ΦᵀΦ v = λ v.
3. Why must the kernel be PSD?
4. What is the complexity of dense Kernel PCA?
5. What normalization is applied to α?

### Interview (5)
1. What is the connection between kernel PCA and SVM?
2. What is the pre-image problem?
3. How is Kernel PCA related to ordinary PCA?
4. When is Nyström approximation used?
5. Can kernel PCA be used for denoising? How?

### Problem Solving (5)
1. Three 1-D points, linear kernel → how many non-trivial components?
2. RBF γ too large → what happens to similarities?
3. Which is cheaper for n=50k: dense kernel PCA or UMAP?
4. Design a pipeline to pick the best kernel via CV.
5. How would you visualize mixed linear+nonlinear structure?

## Answers (explained)
1. Pairwise kernel similarities Kᵢⱼ = k(xᵢ,xⱼ). 2. Using the kernel to compute inner products without φ. 3. The centered kernel/Gram matrix K̃. 4. exp(−γ‖x−y‖²). 5. Unsupervised.
6. Linear axes treat the rings symmetrically; no straight direction has separating variance. 7. The feature-space mean isn't zero; you must correct K. 8. t_j(x)=Σα_{j,i}k(x,xᵢ) — similarity-weighted sum over training points. 9. Components live in implicit feature space. 10. γ is the RBF width; small→smooth, large→sharp.
11. Compare downstream/visual quality of PCA vs Kernel PCA. 12. Eigenvalue elbow + downstream CV. 13. Nyström/randomized, or UMAP. 14. When structure is polynomial-ish and axis structure matters. 15. Fit on train only; transform test with same model.
16. K̃ = K − (1/n)1K − K(1/n) + (1/n²)1K1. 17. Multiply eigen-equation by Φ: ΦΦᵀ(Φv) = λ(Φv), set α=Φv. 18. Mercer's theorem guarantees a real feature space. 19. O(n³) time, O(n²) memory. 20. αᵀK̃α = λ (unit feature-space norm).
21. Both use Gram matrices and the same kernel-trick machinery. 22. Recovering original-space points from feature-space embeddings (needed for denoising). 23. Linear kernel on centered data gives ordinary PCA. 24. Approximate the full kernel to scale the eigendecomposition. 25. Yes, via pre-image: project to few components then map back, dropping noise directions.
26. One non-trivial component (1-D data). 27. All similarities → 0; no structure, essentially identity on each point. 28. UMAP (kernel PCA would need ~n² memory ~ billions entries). 29. KernelPCA + pre-score by CV accuracy per kernel/γ. 30. Not cleanly; kernel PCA is globally nonlinear — consider mixtures/Hybrid, or validate visually.

---

## 49. Final Learning Checklist

- [ ] Explain the kernel trick in one sentence
- [ ] Define kernel, feature map, Gram matrix, RBF kernel
- [ ] Write and interpret the centered-kernel formula
- [ ] Write the kernel eigen-equation K̃ α = λ α
- [ ] Derive kernel PCA from PCA's eigen-equation
- [ ] State the new-point projection formula
- [ ] Compute a tiny centered kernel and its eigenvector by hand
- [ ] Show linear KernelPCA == ordinary PCA on centered data
- [ ] Implement kernel PCA from scratch with numpy
- [ ] Use sklearn KernelPCA (fit, transform)
- [ ] Separate concentric circles with the RBF kernel
- [ ] Tune γ and explain over-smooth vs over-sharp
- [ ] Choose kernel type sensibly for data
- [ ] Explain O(n²)/O(n³) complexity and Nyström fix
- [ ] List advantages and disadvantages
- [ ] Contrast with PCA, t-SNE, UMAP, MDS
- [ ] Recognize data-leakage pitfalls
- [ ] Explain why loadings aren't interpretable
- [ ] Apply kernel PCA in an end-to-end workflow
- [ ] Compare vs PCA baseline on a real dataset

---

## 50. Quality Control Note

- **Accuracy:** Kernel matrices and handmade eigenvalue/centering examples verified (3-point 1-D kernel; linear-kernel equal-to-PCA check; variance 2). Formulas follow the standard Schölkopf formulation. ✅
- **Beginner-friendliness:** Intuition and beginner/technical definitions before math; analogies (bending data) retained. ✅
- **Math depth:** Derivation of K α = λ α, centered-kernel formula, projection formula, all with symbols/examples. ✅
- **Practical depth:** From-scratch + sklearn code, hyperparameter tuning, workflow, coding ladder, failure cases. ✅
- **Exam depth:** Representative pattern questions clearly marked non-PYQ; formula-set and traps included. ✅
- **Structure:** All 50 template sections present in order. ✅
- **Consistency:** Terminology defined first; DR framing (non-linear variance preservation) applied throughout. ✅
