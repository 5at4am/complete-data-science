# Phase 02 — Mathematics for ML

> **Goal:** Learn the mathematics that makes ML implementation work — taught when it becomes useful.

**Difficulty:** 🟢 Beginner → 🟡 Intermediate  
**Priority:** Essential  
**Prerequisites:** Phase 01 (NumPy)  
**Mastery target:** Level 4 — debugging and decision making for core math concepts

---

## Why This Phase Exists

Machine learning is built on linear algebra, calculus, optimization, and probability. But learning all math upfront is overwhelming and disconnected from practice. This phase teaches each concept when it is needed by the next ML step.

### Phase Mental Model

Math is the language of ML models:

```text
Vectors/matrices → data representation
    ↓
Matrix operations → linear transformations
    ↓
Derivatives/gradients → how to improve parameters
    ↓
Gradient descent → optimization loop
    ↓
Probability → uncertainty and classification
    ↓
Entropy/cross-entropy → classification loss
    ↓
Eigenvectors → dimensionality reduction
```

### What This Phase Prepares For

- linear/logistic regression in Phase 05
- neural network backpropagation in Phase 06
- embeddings and vector search in Phases 07–11
- PCA and feature engineering in Phase 05
- probabilistic models throughout the roadmap

---

## Units

### Unit 02.1 — Linear Algebra: Vectors & Matrices

**What is it?**  
Vectors and matrices are the fundamental data structures of ML. A vector is a point or direction in space. A matrix is a collection of vectors or a linear transformation.

**Why does it matter?**  
Every ML input, weight, embedding, and gradient is a vector or matrix.

**Why learn it here?**  
Phase 01 gave NumPy. Now we use it to make math concrete instead of abstract.

**Prerequisites:** NumPy arrays, shapes, broadcasting.

**Mental Model:**  
A vector is an arrow in space. A matrix is a machine that transforms vectors.

**Core Concepts:**

- scalars, vectors, matrices, tensors
- vector addition, scalar multiplication
- dot product (inner product)
- vector norms (L1, L2)
- orthogonality
- matrix as a collection of vectors

**Implementation:** NumPy vector/matrix operations.

**Simple Example:**

```python
import numpy as np

x = np.array([1, 2, 3])
w = np.array([0.1, 0.2, 0.3])

# Dot product = weighted sum
prediction = np.dot(x, w)
```

**Common Mistakes:**

- confusing row vs column vectors
- shape mismatch in dot product
- treating 1D arrays as row or column without reshaping
- forgetting that `np.dot` on 1D arrays is inner product, not matrix multiplication

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| `ValueError: shapes not aligned` | Dimension mismatch | Print `.shape` of both arrays | Reshape with `.reshape(-1, 1)` or `.T` |
| Result is scalar but expected vector | Used 1D arrays | Check `ndim` | Use 2D arrays for matrix multiplication |

**Hands-On Practice:**

1. Basic: create vectors and compute dot products.
2. Guided: implement cosine similarity.
3. Independent: compute weighted sums for a small dataset.
4. Challenge: explain why dot product measures similarity.

**Exit Criteria:**

- You can create and manipulate vectors/matrices in NumPy.
- You can explain what a dot product represents geometrically.

**Next Step:** Matrix operations and multiplication.

---

### Unit 02.2 — Matrix Operations & Multiplication

**What is it?**  
Matrix multiplication composes linear transformations. It is the core operation in linear layers, attention, and many ML algorithms.

**Why does it matter?**  
Neural network layers, linear regression, and PCA all use matrix multiplication.

**Mental Model:**  
Matrix multiplication applies one transformation after another.

**Core Concepts:**

- matrix multiplication rules
- transpose
- identity matrix
- inverse (when it exists)
- matrix-vector multiplication
- broadcasting rules

**Implementation:** Matrix operations in NumPy.

**Simple Example:**

```python
X = np.array([[1, 2], [3, 4], [5, 6]])  # 3 samples, 2 features
W = np.array([[0.1, 0.2], [0.3, 0.4]])  # 2 features, 2 outputs

Y = X @ W  # 3 samples, 2 outputs
```

**Decision Guidance: `@` vs `np.dot` vs `np.matmul`**

| Use | When |
|---|---|
| `@` | Modern, readable, preferred for matrix multiplication |
| `np.matmul` | Explicit function form, handles broadcasting |
| `np.dot` | Legacy, also does inner product on 1D arrays |

**Common Mistakes:**

- element-wise multiplication (`*`) instead of matrix multiplication (`@`)
- forgetting that `(AB)ᵀ = BᵀAᵀ`
- assuming all matrices are invertible

**Hands-On Practice:**

1. Basic: multiply matrices and verify shapes.
2. Guided: implement a linear layer forward pass.
3. Independent: chain two linear transformations.
4. Realistic: debug a shape mismatch in a multi-layer computation.

**Exit Criteria:**

- You can multiply matrices correctly and explain the shape rules.
- You can implement a linear transformation from scratch.

**Next Step:** Derivatives and gradients for optimization.

---

### Unit 02.3 — Calculus: Derivatives & Gradients

**What is it?**  
A derivative measures how a function changes. A gradient is the vector of partial derivatives — the direction of steepest increase.

**Why does it matter?**  
Gradient descent uses gradients to find parameters that minimize loss.

**Mental Model:**  
The gradient points uphill. Negative gradient points downhill toward lower loss.

**Core Concepts:**

- derivative as rate of change
- partial derivatives
- chain rule
- gradient vector
- Jacobian (for vector-valued functions)
- numerical vs analytical gradients

**Implementation:** Numerical vs analytical derivatives.

**Simple Example:**

```python
def f(x):
    return x**2

def grad_f(x):
    return 2 * x

# Numerical check
eps = 1e-5
x = 3.0
numerical = (f(x + eps) - f(x - eps)) / (2 * eps)
analytical = grad_f(x)
print(numerical, analytical)  # Should be close
```

**Common Mistakes:**

- confusing derivative with gradient
- forgetting the chain rule in multi-layer functions
- numerical gradient errors from large epsilon
- gradient shape mismatches

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Gradient check fails | Wrong analytical gradient | Compare numerical vs analytical | Derive carefully or use autograd |
| Gradient is huge/small | Scaling issue | Check input/output scales | Normalize inputs, check learning rate |

**Hands-On Practice:**

1. Basic: compute derivatives of simple functions.
2. Guided: implement gradient of MSE loss.
3. Independent: derive gradient of a two-layer function using chain rule.
4. Challenge: implement gradient checking for a small neural network.

**Exit Criteria:**

- You can compute gradients analytically and verify numerically.
- You can explain the chain rule in your own words.

**Next Step:** Use gradients for optimization.

---

### Unit 02.4 — Optimization: Gradient Descent

**What is it?**  
Gradient descent iteratively updates parameters in the direction opposite to the gradient to minimize a loss function.

**Why does it matter?**  
It is the optimization engine behind almost all ML training.

**Mental Model:**  
Repeatedly adjusting parameters to reduce error — like walking downhill in fog.

**Core Concepts:**

- gradient descent update rule
- learning rate
- convergence
- local vs global minima
- stochastic/mini-batch gradient descent
- momentum

**Implementation:** Gradient descent from scratch.

**Simple Example:**

```python
w = 0.0
lr = 0.1

for step in range(100):
    grad = 2 * w  # derivative of w^2
    w = w - lr * grad
    print(w)
```

**Common Mistakes:**

- learning rate too large (diverges) or too small (slow)
- not checking convergence
- using full-batch when mini-batch is needed
- forgetting to zero gradients in frameworks

**Decision Guidance: Learning Rate**

| Situation | Start With |
|---|---|
| Simple convex problem | 0.1 or 0.01 |
| Neural network | 1e-3 to 1e-4 (Adam) |
| Fine-tuning | 1e-5 to 1e-4 |

**Hands-On Practice:**

1. Basic: minimize a quadratic function.
2. Guided: implement gradient descent for linear regression.
3. Independent: add momentum and compare convergence.
4. Realistic: debug a diverging optimization.

**Exit Criteria:**

- You can implement gradient descent from scratch.
- You can explain how learning rate affects convergence.

**Next Step:** Probability for classification and uncertainty.

---

### Unit 02.5 — Probability Fundamentals

**What is it?**  
Probability quantifies uncertainty. ML models often output probabilities or make decisions under uncertainty.

**Why does it matter?**  
Classification, Naive Bayes, language models, and evaluation all use probability.

**Mental Model:**  
Probability is a measure of belief or frequency, constrained to [0, 1].

**Core Concepts:**

- sample space, events
- conditional probability
- Bayes' theorem
- independence
- random variables
- expectation and variance

**Implementation:** Probability in Python.

**Simple Example:**

```python
# P(A|B) = P(B|A) * P(A) / P(B)
p_spam = 0.1
p_word_given_spam = 0.8
p_word = 0.2

p_spam_given_word = p_word_given_spam * p_spam / p_word
```

**Common Mistakes:**

- confusing P(A|B) with P(B|A)
- assuming independence when it does not hold
- ignoring the prior
- probability values outside [0, 1]

**Hands-On Practice:**

1. Basic: compute conditional probabilities from a contingency table.
2. Guided: implement Naive Bayes from scratch.
3. Independent: apply Bayes' theorem to a medical test scenario.
4. Challenge: explain why a rare disease with a 99% accurate test can still give mostly false positives.

**Exit Criteria:**

- You can apply Bayes' theorem.
- You can explain conditional probability and independence.

**Next Step:** Information theory for classification loss.

---

### Unit 02.6 — Information Theory

**What is it?**  
Information theory quantifies uncertainty and information. Entropy measures average uncertainty. Cross-entropy measures the cost of using one distribution to represent another.

**Why does it matter?**  
Cross-entropy is the standard loss for classification. Entropy guides decision tree splits.

**Mental Model:**  
Entropy = average surprise. Cross-entropy = average surprise when using the wrong distribution.

**Core Concepts:**

- entropy
- cross-entropy
- KL divergence
- relationship to log-likelihood
- bits vs nats

**Implementation:** Compute entropy in Python.

**Simple Example:**

```python
import numpy as np

def entropy(p):
    return -np.sum(p * np.log2(p))

def cross_entropy(p, q):
    return -np.sum(p * np.log2(q))

# True distribution: [1, 0] (certain)
# Predicted: [0.9, 0.1]
print(cross_entropy([1, 0], [0.9, 0.1]))  # Low
print(cross_entropy([1, 0], [0.5, 0.5]))  # Higher
```

**Common Mistakes:**

- log(0) errors (add epsilon)
- confusing entropy with cross-entropy
- using natural log vs base-2 inconsistently

**Hands-On Practice:**

1. Basic: compute entropy of uniform vs skewed distributions.
2. Guided: implement cross-entropy loss for logistic regression.
3. Independent: show that minimizing cross-entropy maximizes likelihood.
4. Challenge: explain why cross-entropy is better than MSE for classification.

**Exit Criteria:**

- You can compute entropy and cross-entropy.
- You can explain why cross-entropy is the right loss for classification.

**Next Step:** Eigenvectors for dimensionality reduction.

---

### Unit 02.7 — Linear Algebra: Eigenvalues & Eigenvectors

**What is it?**  
Eigenvectors are directions that a matrix stretches without rotating. Eigenvalues are the stretch factors.

**Why does it matter?**  
PCA finds directions of maximum variance using eigenvectors of the covariance matrix.

**Mental Model:**  
An eigenvector is a "natural axis" of a transformation. The eigenvalue tells you how much it stretches.

**Core Concepts:**

- eigenvalue equation: Av = λv
- eigendecomposition
- covariance matrix
- principal components
- variance explained

**Implementation:** Eigen decomposition in NumPy.

**Simple Example:**

```python
cov = np.cov(X.T)  # X is centered data
eigenvals, eigenvecs = np.linalg.eigh(cov)
# eigenvecs[:, -1] is the first principal component
```

**Common Mistakes:**

- not centering data before PCA
- using `eig` instead of `eigh` for symmetric matrices
- assuming eigenvalues are sorted

**Hands-On Practice:**

1. Basic: compute eigenvalues of a 2x2 matrix.
2. Guided: implement PCA from scratch using eigendecomposition.
3. Independent: compare variance explained by different components.
4. Realistic: debug a PCA that gives negative variance.

**Exit Criteria:**

- You can explain what eigenvectors represent in PCA.
- You can implement PCA from scratch.

**Next Step:** Connect all math concepts to ML applications.

---

### Unit 02.8 — Math Synthesis & Review

**What is it?**  
A cumulative integration unit connecting all math concepts to their ML applications.

**Why does it matter?**  
Math should not feel like isolated topics. Each concept solves a specific ML problem.

**Mini Project:** Math-to-ML Connection Notebook

**Objective:** Create a notebook that traces each math concept to its ML use case.

**Requirements:**

- vector/matrix → linear regression
- matrix multiplication → neural network layer
- gradient → gradient descent
- probability → logistic regression/Naive Bayes
- cross-entropy → classification loss
- eigenvectors → PCA

**Expected Output:** Annotated notebook with code and explanations.

**Evaluation Criteria:**

- each math concept linked to ML
- code runs correctly
- explanations are clear
- limitations noted

**Knowledge Check:**

- Why does linear regression use matrix multiplication?
- How does the chain rule enable backpropagation?
- Why is cross-entropy the right loss for classification?
- What does an eigenvector of the covariance matrix represent?

**Exit Criteria:**

- You can trace any ML algorithm to its mathematical foundations.
- You can implement the core math operations from scratch.

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Analytical vs numerical gradient | You need exact gradients for simple functions | Function is complex or from a library | Precision vs convenience |
| Full-batch vs mini-batch GD | Dataset is small | Dataset is large or streaming | Stability vs speed/memory |
| L1 vs L2 regularization | You want sparse solutions | You want small dense weights | Sparsity vs smoothness |
| PCA vs feature selection | You can accept transformed features | You need original feature meaning | Compression vs interpretability |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Gradient descent diverges | Learning rate too large | Plot loss curve | Reduce learning rate |
| Gradient descent stalls | Learning rate too small or flat region | Check gradient magnitude | Increase LR or add momentum |
| PCA gives negative variance | Numerical precision | Check eigenvalues | Use `eigh`, clip small negatives |
| Naive Bayes gives zero probability | Unseen feature value | Check feature counts | Add Laplace smoothing |
| Matrix multiplication shape error | Wrong transpose or order | Print shapes at each step | Draw shape diagram |

---

## Phase Review Checklist

- [ ] All units complete.
- [ ] Vectors, matrices, and dot products understood.
- [ ] Matrix multiplication and transpose practiced.
- [ ] Derivatives, gradients, and chain rule implemented.
- [ ] Gradient descent from scratch working.
- [ ] Probability and Bayes' theorem applied.
- [ ] Entropy and cross-entropy computed.
- [ ] Eigenvectors and PCA implemented.
- [ ] Math-to-ML connection notebook completed.
- [ ] Cumulative review passed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Explain what a gradient is and why it matters.
2. Implement gradient descent from scratch.
3. Compute dot products and matrix products correctly.
4. Explain entropy and cross-entropy.
4. Understand eigenvectors' role in PCA.
5. Trace each math concept to its ML application.

## Interview / Explain-Back Questions

- What is the geometric meaning of a dot product?
- Why does gradient descent use the negative gradient?
- How does the chain rule work in a two-layer network?
- Why is cross-entropy better than MSE for classification?
- What does an eigenvector of the covariance matrix represent?
- When would you use numerical gradients instead of analytical?

## Exit Criteria

Move to Phase 03 only when you can implement the core math operations from scratch and explain how each connects to ML algorithms.