# Mathematics Reference

> **A quick-reference guide to the mathematics used throughout this curriculum.**
> Each topic links to the concept it supports. Full derivations live in the
> relevant notebooks.

---

## Linear Algebra

### Scalars, Vectors, Matrices, Tensors
- **Scalar**: a single number
- **Vector**: an ordered list of numbers (1D array)
- **Matrix**: a 2D grid of numbers
- **Tensor**: an N-dimensional array (generalization)

**Used by:** everything — data representation, neural networks, embeddings.

### Vector Operations
- Addition, subtraction (element-wise)
- Dot product: `a · b = Σ aᵢbᵢ`
- Magnitude (norm): `||v|| = √(Σ vᵢ²)`
- Cosine similarity: `cos(θ) = (a·b) / (||a|| ||b||)`

**Used by:** linear regression, similarity search, embeddings.

### Matrix Multiplication
- `(A·B)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ`
- Requires inner dimensions to match
- The workhorse of neural networks

**Used by:** neural networks, linear transformations.

### Matrix Transpose
- Flip rows and columns: `(Aᵀ)ᵢⱼ = Aⱼᵢ`

### Identity Matrix
- Diagonal of 1s, zeros elsewhere
- `A·I = A`

### Inverse Matrix
- `A·A⁻¹ = I`
- Only square, non-singular matrices have inverses

**Used by:** closed-form linear regression, solving linear systems.

### Eigenvalues and Eigenvectors
- `Av = λv` — the vector `v` only scales (by λ) under transformation `A`
- Principal components are eigenvectors of the covariance matrix

**Used by:** PCA, dimensionality reduction.

---

## Calculus

### Derivatives
- Rate of change of a function
- `f'(x) = lim(h→0) [f(x+h) - f(x)] / h`
- Power rule: `d/dx xⁿ = nxⁿ⁻¹`
- Chain rule: `d/dx f(g(x)) = f'(g(x))·g'(x)`

**Used by:** gradient descent, backpropagation.

### Partial Derivatives
- Derivative with respect to one variable, holding others constant
- The gradient is a vector of all partial derivatives

**Used by:** multi-variable optimization.

### Gradient
- `∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]`
- Points in the direction of steepest ascent
- Gradient descent moves opposite the gradient

**Used by:** all optimization in ML.

### Integrals (conceptually)
- Area under a curve
- Used in probability (area under PDF = 1)

---

## Probability

### Basic Concepts
- **Sample space**: all possible outcomes
- **Event**: a subset of outcomes
- **Probability**: `P(A) = favorable / total`

### Conditional Probability
- `P(A|B) = P(A∩B) / P(B)`

### Bayes' Theorem
- `P(A|B) = P(B|A)·P(A) / P(B)`
- Updates beliefs given evidence

**Used by:** Naive Bayes, Bayesian thinking.

### Random Variables
- A variable whose value is a random outcome
- Discrete vs continuous

### Probability Distributions
- **Bernoulli**: single binary outcome
- **Binomial**: number of successes in n trials
- **Normal (Gaussian)**: bell curve, `N(μ, σ²)`
- **Uniform**: all values equally likely
- **Poisson**: count of events in a fixed interval

### Expectation and Variance
- `E[X]` = mean (average value)
- `Var(X) = E[(X - E[X])²]` = spread

---

## Statistics

### Descriptive Statistics
- Mean, median, mode
- Variance, standard deviation
- Quartiles, percentiles, IQR

### Inferential Statistics
- Sampling, confidence intervals
- Hypothesis testing
- p-values

### Correlation
- Pearson correlation: linear relationship between two variables
- Range [-1, 1]

---

## Optimization

### Gradient Descent
- `θ ← θ - η·∇J(θ)`
- η = learning rate
- Iteratively reduces the loss function

### Convexity
- A convex function has one global minimum
- Gradient descent on convex functions converges to the global minimum

---

## Information Theory

### Entropy
- `H = -Σ pᵢ log(pᵢ)`
- Measure of uncertainty/impurity

### Cross-Entropy
- `H(p,q) = -Σ pᵢ log(qᵢ)`
- Used as a loss function for classification

### KL Divergence
- `D_KL(p||q) = Σ pᵢ log(pᵢ/qᵢ)`
- Measures how one distribution differs from another

**Used by:** decision trees (entropy), classification loss, LLM training.

---

## When Each Topic Is Used

| Math Topic | Used In |
|------------|---------|
| Vectors, dot product | Linear regression, embeddings, similarity |
| Matrix multiplication | Neural networks |
| Derivatives, gradients | Gradient descent, backpropagation |
| Chain rule | Backpropagation |
| Probability | Classification, Naive Bayes, generative models |
| Bayes' theorem | Naive Bayes, Bayesian methods |
| Eigenvectors | PCA |
| Entropy, cross-entropy | Decision trees, classification loss |
| Expectation, variance | Statistics, regularization |
