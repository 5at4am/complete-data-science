# 03. Linear Discriminant Analysis (LDA)

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐☆☆
> GATE Relevance: ⭐⭐⭐⭐⭐ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐☆☆

---

## 01. Algorithm Overview

| Aspect | Details |
|--------|---------|
| **Algorithm name** | Linear Discriminant Analysis (LDA) |
| **Category** | Supervised Linear Dimensionality Reduction (also used as a classifier). Note: LDA is **SUPERVISED**, unlike the other techniques in this folder |
| **Type** | Linear projection / linear classifier (Gaussian generative model) |
| **Parametric / Non-parametric** | Parametric (assumes Gaussian class-conditional distributions) |
| **Generative / Discriminative** | Generative (models P(x|class) and uses Bayes) |
| **Main objective** | Find a projection that maximizes the separation between class means while minimizing within-class scatter — i.e., maximize the ratio of between-class to within-class variance |
| **Input** | Feature matrix X (n×p) **and class labels y** (C classes) |
| **Output** | A discriminative low-dim embedding (n×k, k ≤ C−1), plus a linear classifier |
| **Core idea** | Project onto directions that separate classes best: maximize (between-class scatter)/(within-class scatter) via generalized eigendecomposition of scatter matrices |
| **Typical use cases** | Face recognition, marketing analytics, dimensional reduction for classification, Gaussian classifier |

---

## 02. One-Line Definition

### Beginner Definition
LDA finds the few best directions to look at your data so that different groups (classes) are as far apart as possible and each group is as tight as possible — then it can also classify new points by which group they land nearest.

### Technical Definition
LDA is a supervised technique that finds a projection matrix W maximizing the generalized Rayleigh quotient J(W) = |Wᵀ S_B W| / |Wᵀ S_W W|, where S_B is the between-class scatter matrix and S_W the within-class scatter matrix, thereby producing a low-dimensional representation that maximally separates C classes; the same Gaussian generative model yields class posterior probabilities for classification.

---

## 03. Intuition

Imagine you have three groups of points: tall people, short people, and mid-height people, measured on height and weight. You want one number per person that best tells which group they belong to. A good direction is one where, projecting everyone onto it, the three groups occupy separate, tight clusters — pulled apart (large between-group gaps) and compact within each group (small spread).

LDA balances two opposing desires:
1. **Between-class scatter** — get the class means far apart.
2. **Within-class scatter** — keep each class tightly packed.

It finds the direction that maximizes the ratio of these two. That single direction is the best 1-D "view" for telling classes apart. This is fundamentally different from PCA, which maximizes total variance and ignores class labels entirely.

---

## 04. Problem It Solves

**The problem:** When you have labeled data and want to reduce dimensions, PCA may pick directions that are good for spread but useless for distinguishing classes (e.g., a direction of huge variance that both classes share). You want a reduction that improves **class separability**.

**What we want:** Directions that separate the known classes.

**Why LDA is useful:** It uses the labels directly, so the reduced space is tailored for discrimination. It also doubles as a generative classifier with closed-form (non-iterative) solution.

**Small example:** Two overlapping classes in 2D scattered along a shared axis (large total variance there) but separated along an orthogonal thin axis (small total variance). PCA keeps the high-variance axis (bad for separation); LDA picks the thin separating axis (great for separation).

---

## 05. Where It Fits in Machine Learning

```text
Machine Learning
├── Supervised Learning
│   └── Classification
│       └── Linear Discriminant Analysis (also dimensional reduction)
└── Unsupervised Learning
    └── Dimensionality Reduction
        ├── PCA, SVD, NMF (unsupervised, linear)
        ├── Kernel PCA, t-SNE, UMAP (non-linear)
        └── LDA ← placed here in this folder BUT it is SUPERVISED;
                   it uses class labels. Don't confuse with unsupervised PCA.
```

**Important distinction:** LDA is listed under "dimensionality reduction" in this folder for convenience, but it is a **supervised** method — it requires labeled training data. It is the supervised counterpart to PCA for the specific goal of class separation.

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|------|----------------|-------------------|
| **Class** | A labeled group (category) | One of C known output categories |
| **Between-class scatter S_B** | How far class means are from the overall mean | S_B = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ |
| **Within-class scatter S_W** | How spread out points are within each class | S_W = Σ_c Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)ᵀ |
| **Total scatter S_T** | Overall spread = S_W + S_B | S_T = Σ_i (x_i − μ)(x_i − μ)ᵀ |
| **Class mean μ_c** | Mean of class c | Average of feature vectors in class c |
| **Overall mean μ** | Mean of all data | Average of all feature vectors |
| **Generalized eigenproblem** | Solves S_B v = λ S_W v | Not the ordinary eigenproblem because S_W appears on both sides |
| **Discriminant direction** | The best axis to separate classes | Eigenvector of S_W⁻¹ S_B |
| **Posterior probability** | Probability a point belongs to class c | P(C=c | x) from Bayes |

---

## 07. Input and Output

**Input:**
- Feature matrix X (n×p).
- **Labels y** (n,) with C distinct classes. **LDA REQUIRES these (supervised).**

**Parameters/hyperparameters:**
- `n_components` (k ≤ C−1): number of discriminants to keep.
- `solver` (svd/lsqr/eigen), `shrinkage`, priors.

**Output:**
- Reduced space X_lda (n×k).
- The discriminant vectors (n×p) = eigenvectors of S_W⁻¹S_B.
- Class means, priors, covariance (shared or per-class) — used for classification.
- Probabilities P(y=c|x) via the generative model.

---

## 08. Mathematical Foundation

**Basic idea:** Define scatter matrices that measure class separation (S_B) and class tightness (S_W), then maximize the separation-per-spread ratio.

**Notation:**
- n = total samples; C = number of classes; n_c = samples in class c.
- μ_c = mean of class c; μ = global mean; x_i = feature vector.
- S_W, S_B = within/between-class scatter (p×p symmetric matrices).
- W = projection matrix (p×k); v = a single discriminant direction (eigenvector).

**Core equations:**

```text
S_W = Σ_{c=1..C} Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)ᵀ
S_B = Σ_{c=1..C} n_c (μ_c − μ)(μ_c − μ)ᵀ
S_T = S_W + S_B
```

**Rayleigh quotient to maximize:**

```text
J(W) = |Wᵀ S_B W| / |Wᵀ S_W W|
```

**Required math:** Linear algebra, determinant, generalized eigenvalue problems, Gaussian pdfs, Bayes theorem.

**Interpretation:** Maximizing J separates class means (numerator) while keeping each class compact (denominator). For a single direction v this becomes the scalar Rayleigh quotient J(v) = (vᵀS_B v)/(vᵀS_W v).

---

## 09. Core Formula

### 1. Rayleigh quotient (single direction)

```text
J(v) = (vᵀ S_B v) / (vᵀ S_W v)
```

**Meaning:** Ratio of between-class to within-class variance achievable by projecting on v.

**Symbols:** v = unit direction vector; S_B, S_W = scatter matrices; vᵀ … v = quadratic form.

**Intuition:** Big numerator (means far apart) and small denominator (compact classes) → big J → good separation.

### 2. Generalized eigen-equation

```text
S_B v = λ S_W v        ⟹        (S_W⁻¹ S_B) v = λ v
```

**Meaning:** The optimal directions v are eigenvectors of S_W⁻¹ S_B; λ = J(v) (the "separation gain").

**Symbols:** S_W⁻¹ = inverse of within-class scatter; λ = generalized eigenvalue.

**Intuition:** The first discriminant is the eigenvector with the largest λ — the direction with the best separation-per-spread.

### 3. Number of discriminants

```text
k ≤ C − 1   (rank of S_B is at most C−1)
```

**Meaning:** With C classes you can have at most C−1 meaningful discriminant dimensions.

**Symbols:** C = number of classes; k = number of components.

**Intuition:** S_B is a sum of C rank-1 outer products minus one (global mean) → rank ≤ C−1, so no more than C−1 non-zero eigenvalues.

### 4. Classification (Gaussian generative)

```text
P(c|x) ∝ P(c) · N(x; μ_c, Σ)
```

**Meaning:** Posterior probability via Bayes: prior P(c) times Gaussian likelihood.

**Symbols:** P(c) = class prior; N = multivariate Gaussian density with mean μ_c and shared covariance Σ.

**Intuition:** Predict the class whose Gaussian is most likely at x; the decision boundary between equal-covariance Gaussians is a hyperplane (hence "linear" DA).

**Worked example (hand-verified).** Two classes, one feature (p=1):
- Class A: values {2, 4}, n_A=2, μ_A=3.
- Class B: values {6, 8}, n_B=2, μ_B=7.
- Overall mean μ=(3+7)/2=5 (equal sizes).

Within-class scatter S_W = (2−3)²+(4−3)² + (6−7)²+(8−7)² = (1+1)+(1+1)=4.
Between-class scatter S_B = n_A(μ_A−μ)² + n_B(μ_B−μ)² = 2(3−5)² + 2(7−5)² = 2·4+2·4 = 16.

Rayleigh quotient (scalar, p=1) J = S_B/S_W = 16/4 = 4. The single "discriminant direction" is along the feature axis v=1. Projection onto this axis keeps all samples; A={2,4}, B={6,8} are clearly separated. ✅ Hand-verified.

---

## 10. Derivation

1. Choose a direction v; project all points → values z = vᵀx.
2. The projected class means are vᵀμ_c; between-class variance (scalar) = vᵀ S_B v.
3. The projected within-class variance = vᵀ S_W v.
4. Want to maximize the ratio; the Fisher criterion: maximize J(v) = (vᵀS_B v)/(vᵀS_W v).
5. Set derivative to zero (calculate ∂J/∂v of the determinant/quadratic form). Standard result: at optimum, (vᵀS_B v) S_W v = (vᵀS_W v) S_B v → substituting J: S_W⁻¹ S_B v = J v. Thus v is a generalized eigenvector and J is its eigenvalue.
6. This holds for every discriminant; the eigenvectors of S_W⁻¹S_B give the k optimum directions (orthogonal in a weighted sense). 

**Important result:** (Optional derivation detail — one can also show the equivalence to solving the ordinary eigenproblem of S_W⁻¹ S_B or, when S_W is singular, to a simultaneous diagonalization / SVD-based approach used in the sklearn `svd` solver.) For classification, with shared covariance Σ = S_W/n, the Bayes decision boundary between two classes remains a hyperplane — hence the name *Linear* DA.

---

## 11. How the Algorithm Works

```text
Input X (n×p) + labels y (n,), C classes
  ↓
Compute class means μ_c, global mean μ
  ↓
Compute within-class S_W and between-class S_B scatter
  ↓
Solve generalized eigenproblem: S_W⁻¹ S_B v = λ v
  ↓
Sort eigenvectors by descending λ
  ↓
Take top k (k ≤ C−1) → W (p×k)
  ↓
Project: X_lda = (X − μ) W  →  embedding (n×k)
  ↓
Classification: assign x to class maximizing P(c)·N(x;μ_c,Σ)
```

---

## 12. Training Process

**Pre-training:** No iterativative learning; LDA has a closed-form solution using class statistics.

**During:**
1. Estimate μ_c, μ from labels.
2. Build S_W, S_B.
3. Solve S_W⁻¹S_B eigenproblem (or SVD route when S_W singular).
4. Select top k discriminants.

**What's learned:** Discriminant vectors W, class means μ_c, class priors, covariance Σ.

**Stopping:** None — k chosen up front, capped at C−1.

**Final model contents:** Projection matrix W, class means, covariance, priors.

---

## 13. Objective Function / Loss Function

**Objective (Fisher's criterion):** Maximize between-class relative to within-class separation:

```text
maximize  J(W) = |Wᵀ S_B W| / |Wᵀ S_W W|
```

**For classification view:** Maximize the likelihood Σ log P(c) + log N(x;μ_c,Σ) under the Gaussian model, equivalently minimize misclassification probability via Bayes.

**Why chosen:** Separation of means vs tight classes is exactly what makes classes linearly separable.

**High/low meaning:** Higher J → better class separation in the projection; lower → classes overlap heavily.

---

## 14. Optimization

**Definition:** Closed-form generalized eigendecomposition — no iterative gradient descent.

**Why:** The Fisher criterion's optimum is another eigen-problem (Rayleigh quotient), which is exactly solvable.

**Method:** Compute S_W⁻¹ S_B and eigendecompose; or, for numerical stability when S_W is singular (n < p, the small-sample-size problem), use simultaneous diagonalization / SVD.

```text
(n, C) labels → scatter matrices S_W, S_B → solve S_W⁻¹ S_B v = λ v
  → sort λ → top k eigenvectors → W
```

**Global optimum:** Yes, generalized eigen-solution is the global optimum for the Fisher criterion.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE (hand-verified): 4 samples, 2 classes, 2 features.**

```text
Class A: (1,1), (2,1)   → μ_A = (1.5, 1), n_A=2
Class B: (3,4), (4,4)   → μ_B = (3.5, 4), n_B=2
Global mean μ = ((1+2+3+4)/4, (1+1+4+4)/4) = (2.5, 2.5)
```

**Within-class scatter S_W:**
- (1,1)−(1.5,1) = (−0.5,0): outer product [[0.25,0],[0,0]]
- (2,1)−(1.5,1) = (0.5,0): [[0.25,0],[0,0]]
- (3,4)−(3.5,4) = (−0.5,0): [[0.25,0],[0,0]]
- (4,4)−(3.5,4) = (0.5,0): [[0.25,0],[0,0]]
S_W = [[1,0],[0,0]].

**Between-class scatter S_B:**
- (μ_A − μ) = (−1, −1.5)
- (μ_B − μ) = (1, 1.5)
n_A·(μ_A−μ)(μ_A−μ)ᵀ = 2·[[1,1.5],[1.5,2.25]] = [[2,3],[3,4.5]]
n_B·(μ_B−μ)(μ_B−μ)ᵀ = 2·[[1,1.5],[1.5,2.25]] = [[2,3],[3,4.5]]
S_B = [[4,6],[6,9]].

**Generalized eigenproblem S_W⁻¹ S_B:** S_W = [[1,0],[0,0]] is singular → can't invert directly. This is the small-sample-size problem. With k limited: S_B has rank 1 (only one separating direction between two classes), so k ≤ C−1 = 1. The single discriminant direction is that of S_B's nonzero eigenvector: S_B v ∝ v with eigenvector (1, 1.5)/norm. Projecting: scores = X·(1,1.5):
- A: 1+1.5=2.5, 2+1.5=3.5
- B: 3+6=9, 4+6=10
Class A ≈ {2.5, 3.5}, class B ≈ {9, 10} — perfectly separated on one score. ✅ Hand-verified (shows rank limitation and class separation).

---

## 16. Visual Explanation

```text
Original 2D:

        C
       B
      C        (data)
   A      B
   A    C

PCA might pick the high-variance diagonal (spreads classes together),
LDA picks the direction separating A from B,C.

After LDA projection to 1D:

   A  A      B B
   |---gap---|
   B ... C    ← actually want classes compact & separated
```

```text
Classification boundary (shared covariance → linear):

   A A | B B
   A A | B B     ← hyperplane (line) boundary
```

---

## 17. Algorithm / Pseudocode

```
1. For each class c: compute mean μ_c, count n_c
2. Compute global mean μ
3. Compute S_W = Σ_c Σ_{i∈c} (x_i−μ_c)(x_i−μ_c)ᵀ
4. Compute S_B = Σ_c n_c (μ_c−μ)(μ_c−μ)ᵀ
5. Solve generalized eigenproblem: S_W^{-1} S_B v = λ v
   (or SVD route if S_W is singular)
6. Sort eigenvectors v by descending λ
7. Choose top k (≤ C−1) → W
8. Project: X_lda = X · W   (with mean shift)
9. For classification: predict argmax_c P(c)·N(x; μ_c, Σ)
```

---

## 18. From-Scratch Implementation

```python
import numpy as np

def lda_scratch(X, y, n_components=None):
    classes = np.unique(y)
    C = len(classes)
    if n_components is None:
        n_components = C - 1
    n, p = X.shape
    mu = X.mean(axis=0)

    Sw = np.zeros((p, p))
    Sb = np.zeros((p, p))
    for c in classes:
        Xc = X[y == c]
        nc = len(Xc)
        muc = Xc.mean(axis=0)
        Xcc = Xc - muc
        Sw += Xcc.T @ Xcc
        diff = (muc - mu)[:, None]
        Sb += nc * (diff @ diff.T)

    eigvals, eigvecs = np.linalg.eigh(np.linalg.pinv(Sw) @ Sb)
    order = np.argsort(eigvals)[::-1]
    W = eigvecs[:, order[:n_components]]
    X_lda = (X - mu) @ W
    return X_lda, W, eigvals[order]

X = np.array([[1.,1],[2,1],[3,4],[4,4]])
y = np.array([0, 0, 1, 1])
X_lda, W, evals = lda_scratch(X, y)
print("Projection scores:\n", X_lda.round(3))
print("Discriminant direction:", W.ravel().round(3))
```

**Note on pinv:** When S_W is singular (small-sample problem), use `np.linalg.pinv` (pseudo-inverse) — this matches the SVD-based robust approach.

---

## 19. Code Explanation

```text
Code line → What does it do? → Why required? → Mathematical concept?
np.unique(y) → list class ids → needed to loop per class → C classes
X[y==c] → select class c samples → used to compute μ_c, S_W → class-conditional statistics
Xc - muc → center within class → builds within-class deviations → (x−μ_c) terms
Xcc.T @ Xcc → sum outer products within class → within-class scatter per class → S_W
(muc - mu)[:,None] → class-mean deviation column → between-class term → (μ_c − μ)
nc * diff@diff.T → weighted between-class scatter → adds n_c weight → S_B term
np.linalg.pinv(Sw) @ Sb → generalized eigenproblem → left-multiply to invert S_W → S_W⁻¹S_B v=λv
argsort[::-1] → rank by eigenvalue desc → best separation first → λ ordering
X - mu → center data → project about global mean → mean-centering for projection
(X - mu) @ W → project → discriminative embedding → scores = (x−μ)·v
```

---

## 20. Library Implementation

```python
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X = np.array([[1.,1],[2,1],[3,4],[4,4]])
y = np.array([0, 0, 1, 1])

model = LinearDiscriminantAnalysis(n_components=1, solver="svd")
X_lda = model.fit_transform(X, y)
print("Embedding:", X_lda.ravel().round(3))
print("Class means:", model.means_)

# As a classifier on new point:
X_new = np.array([[1.5, 1.2], [3.5, 4.5]])
pred = model.predict(X_new)
proba = model.predict_proba(X_new)
print("Predictions:", pred)
print("Probabilities:", proba.round(3))
```

**Key API:** `fit(X, y)` (labels required!), `transform`, `fit_transform`, `predict`, `predict_proba`, `means_`, `coef_`, `intercept_`, `explained_variance_ratio_`.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|----------------|---------|--------|----------------------|
| `n_components` | Number of discriminants (≤ C−1) | Output dimension | Usually C−1 or a small value for visualization |
| `solver` | svd / lsqr / eigen | How to solve the eigenproblem | `svd` robust + fast (recommended) |
| `shrinkage` | Regularize S_W (None/auto/0..1) | Fixes singular S_W (n<p), reduces overfitting | `auto` for small samples |
| `priors` | Class prior probabilities | Adjusts Bayes rule | Default → from data |
| `tol` | Tolerance for solvers | Numerical precision | Default fine |

**too low / too high / tune:** n_components can't exceed C−1. shrinkage too high over-regularizes (squashes to identity); too low leaves singular S_W. Tune shrinkage or use Ledoit–Wolf `auto`.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Discriminant vectors W (eigenvectors of S_W⁻¹S_B).
- Class means μ_c, priors, shared covariance Σ.
- Eigenvalues (separation scores).

### Hyperparameters (chosen)
- n_components, solver, shrinkage, priors, tol.

---

## 23. Assumptions

| Assumption | What | Why | How to check | If violated | Solution |
|-----------|------|-----|--------------|-------------|----------|
| **Gaussian class-conditional** | Each class ~ Normal | Bayes classifier uses Gaussians | Histograms, Q-Q plots | Heavy skew/outliers | Robust covariance, or QDA for unequal Σ |
| **Shared covariance** | Classes have (nearly) equal Σ | Enables linear boundary; maximizes J via common S_W | Box's M test, compare covariances | Very different covariance → QDA (quadratic) |
| **Class means differ from global** | S_B informative | Separation needs distinct means | Compare means | Classes share same mean → LDA fails | Use higher-order features |
| **Linearity/separable projection** | A linear map suffices | LDA is a linear projection | Visualize/check train accuracy | Non-linear separation → Kernel LDA/QDA/neural |
| **Full-rank S_W** (for inverse) | S_W invertible | Need S_W⁻¹ | Check rank | n < p or collinear features → singular | shrinkage, SVD solver, regularize |

---

## 24. Data Requirements

- **Labels:** Required (supervised!). C ≥ 2 classes.
- **Data type:** Numeric continuous features best.
- **Missing values:** Impute first.
- **Outliers:** Can inflate scatter matrices and skew means — remove/contain.
- **Scaling:** Helps when features differ in units (though scatter matrices are unit-consistent; still recommended for balance).
- **Dataset size:** Needs enough samples per class to estimate μ_c and Σ reliably (n > p ideally, or use shrinkage).
- **Class imbalance:** Priors can correct it; LDA can be biased toward majority class without priors.
- **n < p (small-sample) problem:** S_W singular — use shrinkage/solver="svd".

---

## 25. Feature Scaling

**Recommended.** Although scatter matrices are invariant to affine transforms in a mathematical sense, when features have wildly different units the shared-covariance Gaussian assumption is less reasonable and conditioning worsens. Standardizing (Z-score) each feature makes scatter computations numerically stable and comparable. For classification with shared diagonal-ish covariance, scaling also affects regularization behavior.

---

## 26. Evaluation Metrics

**Training objective (Fisher ratio / likelihood) ≠ downstream evaluation metric.**

| Metric | Definition | When to use | When NOT |
|--------|-----------|-------------|----------|
| **Between/within ratio J** | Separation achieved per discriminant | Ranking discriminants, choosing k | Judging classification performance |
| **Classification accuracy** | Fraction correct predictions | Evaluating the LDA classifier | Ignoring class imbalance |
| **Precision/Recall/F1** | Per-class correctness | Imbalanced classes | Balanced problems only |
| **Reduced-space class separation** (visual/scatter) | Do classes separate after projection | Validating DR goal | Over-interpreting |
| **Log-likelihood / BIC** | Fit of Gaussian model | Model selection (LDA vs QDA) | Large n dominated by fit |

---

## 27. Advantages

- **Uses labels** — reduces to a space tailored for class separation (unlike PCA which ignores labels). ✅
- **Simple & fast** — closed-form, non-iterative, no gradient descent. ✅
- **Doubles as a strong baseline classifier** with well-calibrated probabilities. ✅
- **Interpretable directions** — discriminants tell you which features separate classes. ✅
- **Low variance** — fewer parameters than complex models; hard to overfit when assumptions hold. ✅
- **Naturally handles multi-class** (C−1 discriminants). ✅

---

## 28. Disadvantages

- **Supervised** — requires labels; useless for fully-unlabeled exploration. ✗
- **Assumes Gaussian classes + shared covariance** — strongly violated → poor boundaries (use QDA). ✗
- **Limited to C−1 discriminants** — can't reduce below C−1 dims. ✗
- **Sensitive to small sample size** (n < p → singular S_W). ✗
- **Outlier-sensitive** — means and scatter skewed by extremes. ✗
- **Linear boundary** — can't separate non-linearly separable classes. ✗

---

## 29. When to Use

- ✓ You have labeled data and want class-separating dimensionality reduction.
- ✓ You want a fast, interpretable, well-calibrated generative classifier.
- ✓ Classes are roughly Gaussian with similar covariance.
- ✓ You need probabilities, not just labels.
- ✓ Data size n > p (or use shrinkage).

---

## 30. When NOT to Use

- ✗ You have no labels (use PCA).
- ✗ Classes clearly non-Gaussian or with very different covariance (use QDA or non-parametric).
- ✗ Non-linear class boundaries (use Kernel LDA / neural nets).
- ✗ n << p with no shrinkage/regularization.
- ✗ Heavy outliers dominate statistics.
- ✗ You only care about variance preservation, not class separation (use PCA).

---

## 31. Real-World Applications

| Problem | Input | Algorithm | Output |
|---------|-------|-----------|--------|
| Face recognition | pixel intensity vectors + person labels | LDA (Fisherfaces) | Discriminant face space |
| Credit risk | borrower attributes + default label | LDA classifier | Default probability per applicant |
| Marketing segmentation | customer features + segments | LDA projection | Reduced space separating segments |
| Genome-wide association | SNP features + case/control label | LDA / penalized LDA | Discriminant genes |
| Speech/emotion recognition | spectral features + emotion label | LDA | Separated acoustic classes |

---

## 32. Failure Cases

- **Data failure:** Class outliers drag means and inflate scatter → bad discriminants.
- **Mathematical failure:** Singular S_W (n<p) collapses → use shrinkage/SVD.
- **Assumption failure:** Unequal covariance makes linear boundary wrong → QDA.
- **Optimization failure:** None (closed-form), but numerical instability from near-singular matrices.
- **Generalization failure:** Overfitting when estimating C covariances/means from too little data.
- **Practical failure:** Non-linear boundaries misclassified heavily.

---

## 33. Overfitting and Underfitting

- **Overfitting:** Using too many features (p large) without regularization → estimates noisy, misclassifies test set; or fitting per-class covariances on sparse data.
- **Underfitting:** Assumption of shared/equal covariance too simple when classes differ → boundary misses clusters that QDA would catch.

**Balance:** Choose k ≤ C−1 sensibly; use shrinkage when n small; consider QDA if unequal covariance dominates.

---

## 34. Bias-Variance Perspective

- LDA with shared covariance is a low-variance, somewhat high-bias model (strong Gaussian/equal-covariance assumptions) → good for small data, often a strong baseline.
- Relaxing to per-class covariance (QDA) reduces bias but increases variance (more params) → needs more data.
- Feature/regularization choice (shrinkage) trades bias vs variance; binning or adding features can help non-linearity at variance cost.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|-----------|-----------|----------|----------|----------|
| **LDA** | Max between/within class separation (supervised) | Uses labels, well-calibrated classifier | Gaussian/equal-cov assumptions, C−1 cap | Labeled separable-class problems |
| **PCA** | Max total variance (unsupervised) | Ignores labels → any data | Ignores class separation | Unlabeled linear reduction |
| **QDA** | Same idea but per-class covariance | Handles unequal covariance | More params, more data needed | Unequal class covariance |
| **Logistic regression** | Direct P(class|x) linear model (discriminative) | No Gaussian assumption, robust | Not a projection/DR method | Classification w/o DR |
| **Kernel LDA / LDA in feature space** | Non-linear discriminant | Handles non-linear classes | More complex, tuning | Non-linearly separable data |

---

## 36. Algorithm Selection Guide

```text
Dimensionality reduction with LABELS?
├── Goal = class separation → LDA (linear) / Kernel LDA (non-linear)
├── Goal = capture overall structure (ignore labels) → PCA
└── Equality of covariance unclear → compare LDA vs QDA
Unlabeled data → PCA / Kernel PCA / t-SNE / UMAP
```

---

## 37. Common Mistakes

```text
❌ Calling LDA unsupervised / using it without labels
Why wrong: LDA REQUIRES labels (supervised)
Correct: only use when class labels exist

❌ Asking for more than C−1 components
Why wrong: rank of S_B ≤ C−1 → no more discriminants
Correct: cap k at C−1

❌ Inverting singular S_W directly
Why wrong: n < p or collinear features → singular, NumPy error
Correct: use pinv, shrinkage, or solver="svd"

❌ Assuming equal covariance when it's wildly unequal
Why wrong: linear boundary is wrong
Correct: consider QDA

❌ Confusing LDA (supervised DR) with NMF/PCA (unsupervised)
Why wrong: different objectives and label requirements
Correct: match method to label availability and objective
```

---

## 38. Interview Questions

### Beginner (with answers)
**Q: What's the main difference between LDA and PCA?** LDA is supervised (uses labels, maximizes class separation); PCA is unsupervised (maximizes total variance).
**Q: What does LDA maximize?** The ratio of between-class to within-class scatter.
**Q: How many components can LDA produce?** At most C−1, where C is the number of classes.

### Intermediate (with answers)
**Q: Why is the boundary linear?** With shared covariance Σ, Bayes between two Gaussians simplifies to a linear (hyperplane) boundary.
**Q: What is the small-sample-size problem?** When n < p, S_W is singular and S_W⁻¹ doesn't exist → use shrinkage/SVD/pseudo-inverse.
**Q: Can LDA be used for regression?** No — it's for classification of discrete classes; there's LDA for multi-class but not regression targets.

### Advanced (with answers)
**Q: Derive why eigenvectors of S_W⁻¹S_B maximize the Fisher criterion.** Take the derivative of the Rayleigh quotient; at the optimum J·S_W v = S_B v → S_W⁻¹S_B v = J v, eigen-relation.
**Q: When is LDA equivalent to least-squares?** In the binary (C=2) case, the LDA direction is closely related to Fisher's linear discriminant which is proportional to S_W⁻¹(μ₁−μ₂).
**Q: What happens with unequal class covariance?** The Bayes boundary becomes quadratic (QDA); LDA (shared covariance) then suboptimal.

---

## 39. GATE / Exam Perspective

**Key formulas:**

```text
S_W = Σ_c Σ_{i∈c} (x_i − μ_c)(x_i − μ_c)ᵀ
S_B = Σ_c n_c (μ_c − μ)(μ_c − μ)ᵀ
S_T = S_W + S_B
J(v) = (vᵀ S_B v) / (vᵀ S_W v)   [Fisher criterion]
S_W⁻¹ S_B v = λ v                 [generalized eigenproblem]
k ≤ C − 1                         [max number of discriminants]
```

**Common traps:**
- LDA is **supervised**; PCA is **unsupervised** — the single most-tested distinction.
- Fisher criterion maximizes **between/within**, NOT total variance.
- Rank of S_B ≤ **C−1** bounds the number of discriminants.
- Assumes **Gaussian, equal covariance** → otherwise QDA.
- Compute S_W and S_B correctly (weight by n_c in S_B).

**Representative pattern question (NOT a real PYQ):** "In the two-class case, on which quantity does Fisher's linear discriminant depend?" → S_W⁻¹(μ₁−μ₂) after scaling. (Verify before treating as PYQ.)

---

## 40. Coding Practice

1. **Level 1:** Compute class means and global mean of a small dataset by hand/numpy.
2. **Level 2:** Implement S_W and S_B from scratch; verify S_T = S_W + S_B.
3. **Level 3:** Implement `lda_scratch`; project the 4-sample example and verify separation.
4. **Level 4:** Use sklearn LDA to transform and classify the Iris dataset; compare to PCA+classifier.
5. **Level 5:** Demonstrate the singular-S_W problem on n<p data; fix with shrinkage.
6. **Level 6:** Compare LDA vs QDA on a dataset with unequal class covariance.
7. **Level 7:** Real-world: reduce a face/scale dataset to C−1 dims with LDA, train classifier, report accuracy vs PCA baseline.

---

## 41. Practical ML Workflow

```text
Problem → supervised classification/reduction on labeled data
  ↓ Data → features + labels
  ↓ EDA → check class distributions, means, covariance similarity
  ↓ Cleaning → impute, handle outliers, check balance
  ↓ Feature engineering → standardize, encode if needed
  ↓ Split → train/test (stratify by class)
  ↓ Preprocess → StandardScaler
  ↓ Train → LDA (choose n_components ≤ C−1, solver, shrinkage if n small)
  ↓ Evaluate → accuracy/F1 on test; visualize separated classes
  ↓ Error analysis → check assumption (Gaussian/equal cov); compare QDA
  ↓ Deploy → save scaler+LDA; predict_proba for new points
  ↓ Monitor → drift in class distributions/probabilities
```

---

## 42. Complexity

- **Computing scatter matrices:** O(n p²) (summing outer products).
- **Generalized eigenproblem:** O(p³) (or O(p²·k) with partial solvers).
- **SVD solver:** O(n p²) roughly.
- **Transforming new points:** O(n·k·p).
- **Space:** O(p²) for scatter/covariance.

**Scaling:** Good for moderate p. Small n (n < p) needs shrinkage/SVD. Multi-class caps components at C−1, keeping the projection tiny.

---

## 43. Advanced Concepts

- **Regularized LDA (RLDA):** shrink S_W toward identity: S_W' = (1−λ)S_W + λ·diag → fixes singular.
- **Kernel LDA:** LDA in a kernel feature space for non-linear separation.
- **QDA:** per-class covariance instead of shared.
- **Fisher's Linear Discriminant:** the classic C=2 version = direction S_W⁻¹(μ₁−μ₂).
- **Probabilistic view:** LDA = maximum a posteriori from Gaussian generative model; decision rule is Bayes.
- **Relation to least squares / logistic regression** for the binary case (equivalence up to scaling under full-rank conditions).

---

## 44. Connections to Other Algorithms

```text
                 LDA (supervised DR + classifier)
                   |
        +----------+-----------+
        |          |           |
       PCA    QDA/Kernel LDA   Logistic reg
 (unsupervised (relax/changed  (discriminative
  counterpart)  assumptions)    alternative)
        |
        +--- t-SNE / UMAP (visualization for labeled data)
        +--- NMF (unsupervised parts, non-negative)
```

---

## 45. If You Remember Only 5 Things

1. LDA is **supervised** — it uses class labels to find directions that maximize **between-class / within-class** separation.
2. It solves the **generalized eigenproblem** S_W⁻¹ S_B v = λ v; directions are eigenvectors, eigenvalues = separation gain.
3. The number of discriminants is at most **C−1**.
4. It doubles as a **generative Gaussian classifier** (with shared covariance → linear boundary); relax shared covariance → QDA.
5. Use it instead of PCA when you have labels and want **class separation**; watch out for singular S_W (n<p) via shrinkage/SVD.

---

## 46. Cheat Sheet

| Aspect | Detail |
|--------|--------|
| **Algorithm** | Linear Discriminant Analysis |
| **Category** | SUPERVISED linear dimensional reduction + classifier |
| **Goal** | Separate classes in reduced space; classify via Bayes |
| **Input** | X (n×p) + labels y (C classes) |
| **Output** | Embedding (n×k, k≤C−1) + class probabilities |
| **Core Formula** | J(v)=(vᵀS_B v)/(vᵀS_W v); S_W⁻¹S_B v=λv |
| **Loss / Objective** | Max Fisher ratio (between/within); Bayes likelihood |
| **Optimization** | Closed-form generalized eigendecomposition |
| **Parameters** | Discriminant vectors W, class means μ_c, priors, covariance Σ |
| **Hyperparameters** | n_components, solver, shrinkage, priors |
| **Assumptions** | Gaussian classes, equal covariance, distinct means |
| **Advantages** | Uses labels, fast, well-calibrated, interpretable |
| **Disadvantages** | Supervised, Gaussian/equal-cov assumptions, C−1 cap |
| **Use When** | Labeled, roughly Gaussian/separable classes, n>p |
| **Avoid When** | No labels, non-linear boundaries, unequal covariance, n<<p |
| **Related** | PCA, QDA, Kernel LDA, Logistic regression, Bayes classifier |
| **Key Exam Points** | Supervised vs PCA; Fisher criterion; C−1 bound; S_W/S_B |
| **Key Interview Points** | Label requirement, generalized eigenproblem, linear boundary, QDA |

---

## 47. Final Mental Model

```text
 X (n×p) + y (C classes)
   ↓
 μ_c, μ → S_W, S_B
   ↓
 S_W⁻¹S_B v = λ v  → top k eigenvectors W (k ≤ C−1)
   ↓
 Project X_lda = (X − μ)·W
   ↓
 Classify: argmax_c P(c)·N(x; μ_c, Σ)  → linear boundary
```

---

## 48. Knowledge Check

### Recall (5)
1. Is LDA supervised or unsupervised?
2. What is S_W, S_B?
3. What is the Fisher criterion?
4. Max number of discriminants?
5. What covariance assumption does LDA (not QDA) make?

### Understanding (5)
1. Why does LDA separate classes better than PCA with labels?
2. Why is the boundary linear?
3. What is the small-sample-size problem?
4. Why weight S_B by n_c?
5. When would QDA outperform LDA?

### Application (5)
1. How would you reduce a labeled 50-feature problem for classification with LDA?
2. How to fix singular S_W?
3. How to handle imbalanced classes?
4. When to standardize before LDA?
5. How to pick n_components?

### Mathematical (5)
1. Write S_W and S_B formulas.
2. Write the generalized eigen-equation.
3. Show S_T = S_W + S_B.
4. Compute J for two 1-D classes {2,4} vs {6,8}.
5. Why is rank(S_B) ≤ C−1?

### Interview (5)
1. Derive the eigen-condition from the Fisher criterion.
2. How is binary LDA related to least squares?
3. What is QDA and when used?
4. What is regularized LDA?
5. Why might LDA fail with n < p?

### Problem Solving (5)
1. Two classes with equal means — can LDA separate? 
2. Convert LDA to handle 3 classes.
3. Which is better when class covariance differs: LDA/QDA?
4. How to use LDA for outlier detection?
5. Design pipeline: reduce + classify + calibrate.

## Answers (explained)
1. Supervised (requires labels). 2. Within-class scatter (tightness) and between-class scatter (mean separation). 3. J = (vᵀS_B v)/(vᵀS_W v). 4. C−1. 5. Shared/equal covariance Σ.
6. LDA uses labels directly toward class separation; PCA ignores labels. 7. Shared covariance makes Bayes boundary a hyperplane. 8. n < p → S_W singular → can't invert → shrinkage/SVD. 9. Larger classes contribute more to between-scatter (importance weighting). 10. When classes have very different covariance/spread.
11. LDA with n_components ≤ C−1. 12. Shrinkage/Ledoit-Wolf or solver="svd"/pseudo-inverse. 13. Set priors or resample. 14. When features in different units. 15. Elbow on eigenvalue/separation + downstream CV (≤ C−1).
16. S_W = ΣΣ(x−μ_c)(x−μ_c)ᵀ; S_B=Σn_c(μ_c−μ)(μ_c−μ)ᵀ. 17. S_W⁻¹S_B v = λ v. 18. Holds by construction (adding scatter decompositions). 19. S_B/S_W = 16/4 = 4. 20. S_B is sum of C outer products minus one (global mean) → rank ≤ C−1.
21. Differentiate J, optimum satisfies S_W⁻¹S_B v = Jv. 22. Binary Fisher discriminant ∝ S_W⁻¹(μ₁−μ₂), related to OLS coefficient direction. 23. QDA uses per-class covariance → quadratic boundaries. 24. Shrink S_W toward diagonal to invert. 25. S_W singular, unstable estimates.
26. No — no between-class separation (means equal) → LDA yields no discriminative direction. 27. Generalize S_B/S_W to all C classes; k≤C−1. 28. QDA (unequal covariance). 29. Model P(x|c); low-likelihood points = outliers. 30. Standardize → stratified split → LDA reduce → calibrate probabilities (Platt/isotonic).

---

## 49. Final Learning Checklist

- [ ] State that LDA is supervised
- [ ] Define within/between/class scatter
- [ ] Write S_W and S_B formulas
- [ ] Write and interpret the Fisher criterion
- [ ] Write the generalized eigen-equation S_W⁻¹S_B v=λv
- [ ] Cap k at C−1 and explain why
- [ ] Derive the eigen-condition manually
- [ ] Compute S_W, S_B for a tiny 2-class dataset
- [ ] Show the linear boundary from shared covariance
- [ ] Implement LDA from scratch with numpy
- [ ] Use sklearn LDA for transform + predict_proba
- [ ] Contrast LDA vs PCA (supervised vs not)
- [ ] Contrast LDA vs QDA (covariance assumption)
- [ ] Explain small-sample-size problem + shrinkage
- [ ] Choose n_components ≤ C−1
- [ ] List advantages and disadvantages
- [ ] Recognize outlier/imbalance sensitivities
- [ ] Use LDA in an end-to-end classification workflow
- [ ] Explain probability calibration
- [ ] Compare LDA baseline vs PCA+classifier on a real dataset

---

## 50. Quality Control Note

- **Accuracy:** Hand-verified examples: two 1-D classes (J=S_B/S_W=4) and a 4-sample 2-class 2-feature LDA (showing rank-1 S_B and perfect 1-D separation). Scatter-matrix algebra rechecked row by row. ✅
- **Beginner-friendliness:** Distinction from PCA emphasized in Overview, One-Line, Intuition, and Where-It-Fits before math. ✅
- **Math depth:** Fisher criterion, generalized eigenproblem, derivation, scatter matrices, Bayes view all explained with symbols and examples. ✅
- **Practical depth:** From-scratch + sklearn code, hyperparameters, workflow, coding ladder, failure cases, small-sample handling. ✅
- **Exam depth:** Supervised-vs-unsupervised trap, Fisher criterion, C−1 bound; representative pattern questions clearly marked non-PYQ. ✅
- **Structure:** All 50 template sections present in order. ✅
- **Consistency:** The supervised nature and the DR (class-separation) framing are consistent throughout. ✅
