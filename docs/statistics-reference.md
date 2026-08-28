# Statistics Reference

> **A quick-reference guide to statistics used throughout the curriculum.**

---

## Descriptive Statistics

### Measures of Central Tendency
- **Mean**: `μ = (1/n) Σ xᵢ` — arithmetic average
- **Median**: middle value when sorted
- **Mode**: most frequent value

### Measures of Spread
- **Variance**: `σ² = (1/n) Σ (xᵢ - μ)²` — average squared deviation from mean
- **Standard deviation**: `σ = √σ²`
- **Range**: max - min
- **IQR**: Q3 - Q1 (robust to outliers)

### Quantiles
- Quartiles divide data into 4 parts
- Percentiles divide into 100 parts

---

## Probability Distributions

### Discrete
- **Bernoulli(p)**: single trial, success/failure
- **Binomial(n, p)**: n independent Bernoulli trials
- **Poisson(λ)**: count of events in fixed interval

### Continuous
- **Uniform(a, b)**: all values equally likely
- **Normal(μ, σ²)**: the bell curve
- **Exponential(λ)**: time between events

---

## Inferential Statistics

### Sampling
- A sample is a subset of the population
- Random sampling avoids bias

### Central Limit Theorem
- The distribution of sample means approaches normal as sample size grows
- Foundation for many statistical tests

### Confidence Intervals
- A range that likely contains the true parameter
- `CI = estimate ± margin of error`

### Hypothesis Testing
1. State null hypothesis `H₀` and alternative `H₁`
2. Choose significance level α (usually 0.05)
3. Compute test statistic
4. Compute p-value
5. Reject `H₀` if p < α

### p-value
- Probability of observing the data (or more extreme) if `H₀` is true
- Small p-value → evidence against `H₀`

---

## Correlation

### Pearson Correlation
- `r = cov(X,Y) / (σₓ σᵧ)`
- Measures **linear** relationship
- Range [-1, 1]

### Spearman Correlation
- Rank-based, measures monotonic relationship
- Robust to outliers

---

## Key Concepts for ML

### Bias-Variance Tradeoff
- **Bias**: error from overly simple assumptions
- **Variance**: error from sensitivity to training data
- Tradeoff: reducing one often increases the other

### Overfitting vs Underfitting
- **Overfitting**: model memorizes training data, fails on new data
- **Underfitting**: model too simple, fails on both

### Cross-Validation
- Split data into k folds, train on k-1, validate on 1
- Repeat k times, average results
- More reliable than a single train/test split

---

## Common Statistical Tests

| Test | Purpose | When to Use |
|------|---------|-------------|
| t-test | Compare means of 2 groups | Small samples, unknown variance |
| ANOVA | Compare means of 3+ groups | Multiple groups |
| Chi-square | Test independence of categorical variables | Contingency tables |
| Shapiro-Wilk | Test normality | Check if data is normal |

---

## Statistics in the ML Pipeline

| Statistical Concept | ML Application |
|---------------------|----------------|
| Mean, variance | Feature scaling, normalization |
| Correlation | Feature selection, multicollinearity |
| Distributions | Data understanding, generative models |
| Hypothesis testing | A/B testing, model comparison |
| Confidence intervals | Model uncertainty |
| Bias-variance | Model selection, regularization |
