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
