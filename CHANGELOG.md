# CHANGELOG

All notable changes to this learning system are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial project scaffolding and recovery infrastructure.
- Project architecture (directories for docs, roadmap, notebooks, projects, datasets, src, tests, templates, tracking).
- Learning system specification and dependency-aware roadmap.
- Notebook templates (concept, implementation, experiment, review, project).
- Tracking systems (progress, mistakes, concepts, experiments, review log, project log).
- Reference documentation (learning philosophy, prerequisites, environment setup, troubleshooting, math, stats, ML, DL, NLP, GenAI, engineering).
- Phase 00 (Environment Setup) notebooks: environment verification, Jupyter workflow, Git & version control, Python environments.
- Environment verified: all core packages installed and working.
- `requirements.txt`, `.env.example`, `datasets/README.md`, `src/README.md`, `tests/README.md`.
- Phase 01 (Python Foundations) Unit 01.1: Python basics notebook (variables, types, operators, control flow).
- Phase 01 Unit 01.2: Data structures notebook (lists, tuples, sets, dicts, comprehensions).
- Phase 01 Unit 01.3: Functions & scope notebook (functions, args/kwargs, scope, lambdas, closures).
- Phase 01 Unit 01.4: OOP notebook (classes, inheritance, dunder methods, encapsulation).
- Phase 01 Unit 01.5: NumPy notebook (arrays, vectorization, broadcasting, matrix ops, normal equation).
- Phase 01 Unit 01.6: Pandas notebook (DataFrames, Series, filtering, groupby, merging, missing values).
- Phase 01 Unit 01.7: Matplotlib notebook (line, scatter, bar, histogram, box, subplots, saving).
- Phase 01 Unit 01.8: File I/O & errors notebook (file handling, with statement, try/except, custom exceptions).
- Phase 01 Unit 01.9: Synthesis notebook (mini data analysis project tying together all Phase 01 skills).
- **Phase 01 (Python Foundations) COMPLETE** — all 9 units verified.
- Phase 02 (Mathematics for ML) Unit 02.1: Vectors & matrices notebook (vectors, matrices, norms, dot product, cosine similarity).
- Phase 02 Unit 02.2: Matrix operations notebook (matrix multiplication, inverse, linear systems, normal equation).
- Phase 02 Unit 02.3: Derivatives & gradients notebook (derivatives, partial derivatives, gradients, chain rule, autograd).
- Phase 02 Unit 02.4: Gradient descent notebook (update rule, learning rate, batch/SGD/mini-batch, divergence).
- Phase 02 Unit 02.5: Probability notebook (conditional probability, Bayes' theorem, independence, distributions, CLT).
- Phase 02 Unit 02.6: Information theory notebook (entropy, cross-entropy, KL divergence).
- Phase 02 Unit 02.7: Eigenvalues notebook (eigenvalues, eigenvectors, eigendecomposition, PCA).
- Phase 02 Unit 02.8: Synthesis notebook (full pipeline: data, model, train, evaluate, PCA).
- **Phase 02 (Mathematics for ML) COMPLETE** — all 8 units verified.
- Phase 03 (Statistics & Probability) Unit 03.1: Descriptive statistics notebook (mean, median, mode, variance, std, quartiles, skew, outliers).
- Phase 03 Unit 03.2: Probability distributions notebook (Bernoulli, Binomial, Normal, Uniform, Poisson).
- Phase 03 Unit 03.3: Inferential statistics notebook (sampling, CLT, standard error, confidence intervals).
- Phase 03 Unit 03.4: Hypothesis testing notebook (null/alt hypotheses, p-values, t-tests, Type I/II errors).
- Phase 03 Unit 03.5: Correlation & regression basics notebook (Pearson correlation, covariance, simple linear regression, R-squared).
- Phase 03 Unit 03.6: Bayesian thinking notebook (Bayes' theorem, priors, likelihoods, posteriors, sequential updating).
- Phase 03 Unit 03.7: Statistics for ML notebook (train/test split, cross-validation, bias-variance, metrics).
- Phase 03 Unit 03.8: Synthesis notebook (full workflow: explore, infer, model, evaluate).
- **Phase 03 (Statistics & Probability) COMPLETE** - all 8 units verified and committed.
- Phase 04 Unit 04.1: Data exploration & EDA notebook (summary stats, distributions, correlations, visualizations).
- Phase 04 Unit 04.2: Data cleaning notebook (duplicates, formats, typos, types, consistency).
- Phase 04 Unit 04.3: Missing values notebook (MCAR/MAR/MNAR, imputation strategies).
- Phase 04 Unit 04.4: Outliers notebook (IQR, z-score, visualization, keep/cap/remove).

### Fixed
- `opendatasets` blocked on Python 3.13 (removed `cgi` module). Use `kagglehub` instead.

---

## Versioning Convention

- **0.x.y** — Pre-release / in-development builds.
- Each completed learning unit increments the patch version.
- Each completed phase increments the minor version.
- A full curriculum release increments the major version.

## Commit Convention

Every completed learning unit is committed with a meaningful message:

```
phase-00-environment-unit-01-complete
phase-01-foundations-unit-03-complete
phase-02-statistics-unit-05-complete
phase-03-ml-unit-02-complete
```

See `BUILD_STATUS.md` for the current build state and `PROJECT_STATE.md` for the
full project state snapshot.
