# Phase 03 — Statistics & Probability

> **Goal:** Build statistical intuition and tools for understanding data, evaluating models, and reasoning about uncertainty.

**Difficulty:** 🟡 Intermediate  
**Priority:** Essential  
**Prerequisites:** Phase 02  
**Mastery target:** Level 5 — decision making for statistical concepts

---

## Why This Phase Exists

Machine learning is applied statistics. Every model evaluation, hyperparameter choice, and error analysis depends on statistical reasoning. Without it, you cannot distinguish signal from noise, choose the right metric, or trust your results.

### Phase Mental Model

Statistics is the toolkit for reasoning about data you cannot fully observe:

```text
Descriptive stats → summarize what you see
    ↓
Probability distributions → model the data-generating process
    ↓
Sampling & inference → generalize from samples to populations
    ↓
Hypothesis testing → make decisions with evidence
    ↓
Correlation & regression → understand relationships
    ↓
Bayesian thinking → update beliefs with data
    ↓
Bias-variance & CV → connect to ML generalization
```

### What This Phase Prepares For

- data exploration and cleaning in Phase 04
- model evaluation and metric selection in Phase 05
- understanding overfitting/underfitting in Phase 05
- probabilistic models throughout the roadmap
- A/B testing and experiment design in Phase 15

---

## Units

### Unit 03.1 — Descriptive Statistics

**What is it?**  
Descriptive statistics summarize the main features of a dataset.

**Why does it matter?**  
Before modeling, you must understand what your data looks like.

**Prerequisites:** NumPy, Pandas, basic Python.

**Mental Model:**  
Descriptive stats are the "vital signs" of your data.

**Core Concepts:**

- measures of center: mean, median, mode
- measures of spread: variance, standard deviation, IQR, range
- measures of shape: skewness, kurtosis
- percentiles and quartiles
- correlation vs causation

**Implementation:** Compute in NumPy/Pandas.

**Simple Example:**

```python
import pandas as pd
import numpy as np

df = pd.read_csv("data.csv")
print(df.describe())
print("Skewness:", df["target"].skew())
print("Correlation matrix:\n", df.corr())
```

**Common Mistakes:**

- using mean for skewed data (median is better)
- ignoring outliers when computing statistics
- treating correlation as causation
- reporting only mean without spread

**Decision Guidance: Mean vs Median**

| Use Mean When | Use Median When |
|---|---|
| Distribution is symmetric | Distribution is skewed |
| No extreme outliers | Outliers are present |
| You need mathematical properties | You need robustness |

**Hands-On Practice:**

1. Basic: compute mean, median, std for numeric columns.
2. Guided: compare mean vs median on skewed data.
3. Independent: write a function that auto-selects appropriate summary stats.
4. Realistic: find a dataset where mean is misleading and explain why.

**Exit Criteria:**

- You can choose and compute appropriate descriptive statistics.
- You can explain when mean vs median is appropriate.

**Next Step:** Probability distributions as models of data.

---

### Unit 03.2 — Probability Distributions

**What is it?**  
A probability distribution describes how likely different outcomes are. It is a model of the data-generating process.

**Why does it matter?**  
Choosing the right distribution lets you simulate, test, and model data correctly.

**Mental Model:**  
A distribution is a recipe for generating data. The parameters control the recipe.

**Core Concepts:**

- discrete: Bernoulli, Binomial, Poisson, Categorical
- continuous: Normal, Uniform, Exponential, Beta, Gamma
- parameters: location, scale, shape
- sampling from distributions
- probability density/mass functions
- cumulative distribution functions

**Implementation:** Sample and visualize distributions.

**Simple Example:**

```python
import numpy as np
import matplotlib.pyplot as plt

# Normal distribution
samples = np.random.normal(loc=0, scale=1, size=10000)
plt.hist(samples, bins=50, density=True, alpha=0.5)

# Overlay theoretical PDF
x = np.linspace(-4, 4, 100)
pdf = 1/np.sqrt(2*np.pi) * np.exp(-x**2/2)
plt.plot(x, pdf, 'r-')
plt.show()
```

**Decision Guidance: Choosing a Distribution**

| Data Type | Candidate Distributions |
|---|---|
| Binary outcome | Bernoulli |
| Count of successes in n trials | Binomial |
| Count of events in interval | Poisson |
| Continuous, symmetric | Normal |
| Continuous, positive, skewed | Log-normal, Gamma, Exponential |
| Proportions/percentages | Beta |

**Common Mistakes:**

- using Normal for bounded or skewed data
- ignoring parameter constraints (e.g., variance > 0)
- confusing probability mass with density
- assuming independence when data is correlated

**Hands-On Practice:**

1. Basic: sample from and plot 5 common distributions.
2. Guided: fit a distribution to real data using MLE.
3. Independent: simulate a process (e.g., customer arrivals) with appropriate distribution.
4. Challenge: explain why heights are approximately Normal but incomes are not.

**Exit Criteria:**

- You can identify the right distribution for common data types.
- You can sample, visualize, and fit basic distributions.

**Next Step:** Inferential statistics — generalizing from samples.

---

### Unit 03.3 — Inferential Statistics & Sampling

**What is it?**  
Inferential statistics uses sample data to make claims about a population. Sampling theory explains how reliable those claims are.

**Why does it matter?**  
You almost never have all data. You must generalize from samples.

**Mental Model:**  
A sample is a noisy window into the population. The Central Limit Theorem tells you how noisy.

**Core Concepts:**

- population vs sample
- sampling distributions
- Central Limit Theorem
- standard error
- confidence intervals
- margin of error
- sample size determination

**Implementation:** Simulate sampling.

**Simple Example:**

```python
import numpy as np

population = np.random.exponential(scale=2, size=100000)
sample_means = []

for _ in range(1000):
    sample = np.random.choice(population, size=30, replace=False)
    sample_means.append(sample.mean())

print("Population mean:", population.mean())
print("Sampling distribution mean:", np.mean(sample_means))
print("Standard error:", np.std(sample_means))
```

**Common Mistakes:**

- confusing standard deviation with standard error
- assuming CLT applies for tiny samples
- ignoring sampling bias
- treating a confidence interval as a probability statement about the parameter

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| CI too wide | Small sample or high variance | Check sample size and std | Increase sample size or accept uncertainty |
| CI doesn't contain true value | Bad luck (5% for 95% CI) | Repeat simulation | Accept that CIs have coverage probability |

**Hands-On Practice:**

1. Basic: simulate sampling distribution of the mean.
2. Guided: compute confidence intervals for a proportion.
3. Independent: determine sample size needed for a desired margin of error.
4. Realistic: identify sampling bias in a real-world scenario.

**Exit Criteria:**

- You can explain the Central Limit Theorem in your own words.
- You can compute and interpret confidence intervals.

**Next Step:** Hypothesis testing for decision making.

---

### Unit 03.4 — Hypothesis Testing

**What is it?**  
Hypothesis testing is a framework for making decisions using data. It quantifies how surprising your data would be if a null hypothesis were true.

**Why does it matter?**  
ML decisions (is model A better than B? is this feature useful?) are hypothesis tests.

**Mental Model:**  
Assume nothing happened (null). If data is very surprising, reject the null.

**Core Concepts:**

- null and alternative hypotheses
- test statistic
- p-value
- significance level (α)
- Type I and Type II errors
- power
- t-tests, chi-squared, ANOVA
- multiple testing correction

**Implementation:** Run statistical tests.

**Simple Example:**

```python
from scipy import stats

# Two-sample t-test
group_a = np.random.normal(0, 1, 50)
group_b = np.random.normal(0.5, 1, 50)

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"t={t_stat:.3f}, p={p_value:.4f}")

if p_value < 0.05:
    print("Reject null: groups differ")
else:
    print("Fail to reject null")
```

**Decision Guidance: Choosing a Test**

| Question | Test |
|---|---|
| Compare two means (independent) | t-test |
| Compare two means (paired) | paired t-test |
| Compare multiple means | ANOVA |
| Compare proportions | chi-squared or z-test |
| Compare distributions | Kolmogorov-Smirnov |
| Non-parametric alternative | Mann-Whitney U |

**Common Mistakes:**

- p-hacking (trying many tests until one is significant)
- interpreting p-value as P(null is true)
- ignoring effect size
- not correcting for multiple comparisons
- testing on the same data used for exploration

**Hands-On Practice:**

1. Basic: run t-tests and interpret p-values.
2. Guided: simulate Type I and Type II errors.
3. Independent: design an A/B test with power analysis.
4. Challenge: explain why "p < 0.05" does not mean "95% chance the effect is real."

**Exit Criteria:**

- You can choose and run appropriate hypothesis tests.
- You can explain p-values, Type I/II errors, and power.

**Next Step:** Correlation and regression for relationships.

---

### Unit 03.5 — Correlation & Regression Basics

**What is it?**  
Correlation measures linear association. Regression models the relationship between variables.

**Why does it matter?**  
Understanding relationships guides feature engineering and model selection.

**Mental Model:**  
Correlation = how much two variables move together. Regression = predicting one from the other.

**Core Concepts:**

- Pearson correlation (linear)
- Spearman correlation (monotonic)
- simple linear regression
- coefficient interpretation
- R²
- assumptions: linearity, independence, homoscedasticity, normality

**Implementation:** Compute correlations.

**Simple Example:**

```python
import pandas as pd
import seaborn as sns

df = pd.read_csv("data.csv")
print(df.corr(method="pearson"))
print(df.corr(method="spearman"))

# Visualize
sns.pairplot(df)
```

**Common Mistakes:**

- correlation ≠ causation
- Pearson only captures linear relationships
- ignoring outliers that drive correlation
- extrapolating regression beyond data range
- assuming correlation implies predictive power

**Hands-On Practice:**

1. Basic: compute and visualize correlations.
2. Guided: fit simple linear regression and interpret coefficients.
3. Independent: find a spurious correlation and explain why it's misleading.
4. Realistic: check regression assumptions on real data.

**Exit Criteria:**

- You can compute and interpret correlation coefficients.
- You can fit and interpret simple linear regression.

**Next Step:** Bayesian thinking for probabilistic reasoning.

---

### Unit 03.6 — Bayesian Thinking

**What is it?**  
Bayesian thinking updates beliefs with evidence using Bayes' theorem. It treats parameters as random variables with distributions.

**Why does it matter?**  
It provides a coherent framework for uncertainty, regularization, and decision making.

**Mental Model:**  
Prior belief + data → posterior belief. The posterior becomes the new prior.

**Core Concepts:**

- prior, likelihood, posterior
- conjugate priors
- Bayesian vs frequentist interpretation
- credible intervals vs confidence intervals
- Bayesian model comparison
- regularization as prior

**Implementation:** Update beliefs with data.

**Simple Example:**

```python
# Beta-Binomial conjugate prior
# Prior: Beta(alpha=2, beta=2) - weakly favors 0.5
# Data: 7 heads, 3 tails
# Posterior: Beta(2+7, 2+3) = Beta(9, 5)

import numpy as np
from scipy import stats

prior = stats.beta(2, 2)
posterior = stats.beta(9, 5)

print("Prior mean:", prior.mean())
print("Posterior mean:", posterior.mean())
print("95% credible interval:", posterior.interval(0.95))
```

**Common Mistakes:**

- choosing an informative prior without justification
- confusing credible intervals with confidence intervals
- ignoring the prior's influence on small data
- computational complexity for complex models

**Hands-On Practice:**

1. Basic: update a Beta prior with Binomial data.
2. Guided: compare Bayesian and frequentist intervals.
3. Independent: implement Bayesian linear regression with simple priors.
4. Challenge: explain how L2 regularization corresponds to a Gaussian prior.

**Exit Criteria:**

- You can apply Bayes' theorem to update beliefs.
- You can explain the difference between Bayesian and frequentist intervals.

**Next Step:** Connect statistics to ML generalization.

---

### Unit 03.7 — Statistics for ML

**What is it?**  
This unit connects statistical concepts directly to ML: bias-variance tradeoff, overfitting/underfitting, and cross-validation.

**Why does it matter?**  
These are the statistical foundations of model generalization.

**Mental Model:**  
Bias = error from wrong assumptions. Variance = error from sensitivity to training data. Total error = bias² + variance + irreducible noise.

**Core Concepts:**

- bias-variance decomposition
- overfitting (high variance)
- underfitting (high bias)
- cross-validation as estimation of generalization error
- k-fold CV
- stratified CV
- nested CV for hyperparameter tuning

**Implementation:** Demonstrate bias-variance.

**Simple Example:**

```python
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Bias-variance tradeoff with polynomial degree
for degree in [1, 2, 5, 10, 20]:
    model = make_pipeline(PolynomialFeatures(degree), Ridge(alpha=1.0))
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"Degree {degree}: CV MSE = {-scores.mean():.4f} (+/- {scores.std():.4f})")
```

**Common Mistakes:**

- using test set for model selection (data leakage)
- not using stratified CV for imbalanced data
- reporting only mean CV score without variance
- assuming CV score equals test performance

**Decision Guidance: CV Strategy**

| Situation | CV Strategy |
|---|---|
| Standard classification/regression | 5-fold or 10-fold |
| Imbalanced classes | Stratified k-fold |
| Time series | TimeSeriesSplit |
| Hyperparameter tuning | Nested CV |
| Small dataset | Leave-one-out |

**Hands-On Practice:**

1. Basic: run k-fold CV and interpret results.
2. Guided: plot bias-variance tradeoff with model complexity.
3. Independent: implement nested CV for hyperparameter tuning.
4. Challenge: explain why CV can be optimistic for small datasets.

**Exit Criteria:**

- You can explain bias-variance tradeoff.
- You can choose and run appropriate cross-validation.

**Next Step:** Synthesis and statistical analysis project.

---

### Unit 03.8 — Statistics Synthesis & Review

**What is it?**  
A cumulative integration unit applying statistical thinking to a real dataset.

**Mini Project:** Statistical Analysis Report

**Objective:** Perform a complete statistical analysis on a real dataset.

**Requirements:**

- descriptive statistics and visualizations
- distribution fitting for key variables
- confidence intervals for key parameters
- hypothesis tests for meaningful questions
- correlation/regression analysis
- Bayesian analysis for at least one question
- bias-variance discussion for a simple model
- clear report with limitations

**Expected Output:** Jupyter notebook or PDF report.

**Evaluation Criteria:**

- appropriate methods chosen
- assumptions checked
- results interpreted correctly
- limitations acknowledged
- reproducible code

**Knowledge Check:**

- When is a t-test inappropriate?
- What does a 95% confidence interval actually mean?
- How does sample size affect power?
- Why does regularization reduce variance?
- What is the difference between a credible interval and a confidence interval?

**Exit Criteria:**

- You can design and execute a statistical analysis independently.
- You can explain the statistical reasoning behind each choice.

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Mean vs median | Symmetric, no outliers | Skewed or outliers | Efficiency vs robustness |
| Pearson vs Spearman | Linear relationship | Monotonic but non-linear | Specificity vs generality |
| t-test vs Mann-Whitney | Normal data, equal variance | Non-normal or ordinal | Power vs assumptions |
| Frequentist vs Bayesian | Large data, standard questions | Small data, prior knowledge, decisions | Objectivity vs coherence |
| Simple CV vs nested CV | Model fixed, estimating performance | Tuning hyperparameters | Simplicity vs unbiased estimate |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Significant p-value but tiny effect | Large sample size | Check effect size | Report effect size and CI |
| CI very wide | Small sample or high variance | Check n and std | Accept uncertainty or collect more data |
| Model overfits despite CV | Data leakage in CV | Check preprocessing inside CV | Put all preprocessing in pipeline |
| Correlation disappears with more data | Spurious correlation | Check with larger sample | Don't trust small-sample correlations |
| Bayesian posterior dominated by prior | Small data, strong prior | Compare prior vs posterior | Use weaker prior or collect more data |

---

## Phase Review Checklist

- [ ] All units complete.
- [ ] Descriptive statistics practiced with real data.
- [ ] Probability distributions identified, sampled, and fitted.
- [ ] Sampling distributions and CLT simulated.
- [ ] Confidence intervals computed and interpreted.
- [ ] Hypothesis tests run with correct interpretation.
- [ ] Correlation and regression analyzed.
- [ ] Bayesian updating practiced.
- [ ] Bias-variance and CV connected to ML.
- [ ] Statistical analysis mini project completed.
- [ ] Cumulative review passed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Summarize a dataset with appropriate statistics.
2. Choose the right distribution for data.
3. Run and interpret a hypothesis test.
4. Explain bias-variance tradeoff.
5. Understand cross-validation.
6. Apply Bayesian reasoning to a simple problem.

## Interview / Explain-Back Questions

- What is the difference between standard deviation and standard error?
- Why does the Central Limit Theorem matter for ML?
- When is a p-value misleading?
- What does "95% confidence" actually mean?
- How does regularization relate to Bayesian priors?
- Why is nested CV needed for hyperparameter tuning?

## Exit Criteria

Move to Phase 04 only when you can independently design a statistical analysis, choose appropriate methods, check assumptions, and interpret results for real data.