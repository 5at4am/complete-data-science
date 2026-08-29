# ROADMAP

> **A dependency-aware ML/AI training system.**
> This roadmap is not optimized for the number of topics covered. It is optimized for what the learner can understand, build, debug, compare, explain, improve, and solve independently.

---

## What Changed in This Restructure

The original roadmap already had a strong 18-phase structure. This revision preserves that intent and improves it by adding:

- clearer phase dependencies and transitions
- difficulty and priority labels
- practical exercise progression
- project milestones before the final capstone
- debugging/failure-case expectations
- decision-making comparisons
- interview preparation checkpoints
- revision and mastery gates
- industry-readiness criteria
- a final audit table for future improvements

The detailed learning standard is now defined in [`roadmap/optimized-learning-system.md`](roadmap/optimized-learning-system.md). The dependency map is in [`roadmap/dependency-graph.md`](roadmap/dependency-graph.md).

---

## How to Use This Roadmap

1. Start at Phase 00 and move forward in order unless a phase is explicitly marked optional.
2. For every unit, learn in this order:

   ```text
   Understand → Apply → Practice → Build → Debug → Compare → Optimize → Explain → Solve independently
   ```

3. Do not skip prerequisites. If a later topic feels confusing, return to the dependency graph.
4. Complete the hands-on work, not only the reading.
5. Track progress in `tracking/progress.md`.
6. Record mistakes in `tracking/mistakes.md`.
7. Record experiments in `tracking/experiments.md`.
8. Record project outcomes in `tracking/project-log.md`.
9. Use phase reviews as gates. Do not move forward just because all topics were read.

---

## Difficulty & Priority Labels

| Label | Meaning |
|---|---|
| 🟢 Beginner | New learner-friendly foundation |
| 🟡 Intermediate | Requires earlier foundations and independent coding |
| 🔴 Advanced | Requires comfort with theory, debugging, and trade-offs |
| ⚫ Expert / Optional | Useful for specialization, not required before industry entry |

| Priority | Meaning |
|---|---|
| Essential | Must know before moving forward |
| Important | Strongly recommended for real projects |
| Recommended | Valuable, but can be learned after essentials |
| Optional | Useful only for some paths |
| Specialization | Deep path for a specific role or domain |

---

## The Progressive Learning Path

```text
Phase 00 — Environment Setup
        ↓
Phase 01 — Python Foundations
        ↓
Phase 02 — Mathematics for ML
        ↓
Phase 03 — Statistics & Probability
        ↓
Phase 04 — Data Analysis & Preparation
        ↓
Phase 05 — Machine Learning
        ↓
Phase 06 — Deep Learning
        ↓
Phase 07 — NLP
        ↓
Phase 08 — Transformers
        ↓
Phase 09 — Generative AI Foundations
        ↓
Phase 10 — Applied LLM Engineering
        ↓
Phase 11 — RAG Systems
        ↓
Phase 12 — LangChain / Framework Abstractions
        ↓
Phase 13 — LangGraph / Stateful Workflows
        ↓
Phase 14 — AI Agents
        ↓
Phase 15 — Evaluation & Experimentation
        ↓
Phase 16 — Deployment & MLOps
        ↓
Phase 17 — Capstone Engineering
```

Framework phases are intentionally placed after manual concepts. Learn the problem first, then the abstraction.

---

## Phase Details

| Phase | Difficulty | Priority | Goal | Units | Mastery | Details |
|---|---:|---|---|---:|---:|---|
| 00 — Environment Setup | 🟢 | Essential | Get a verified, reproducible workspace. | 4 | 3 | [`phase-00.md`](roadmap/phase-00.md) |
| 01 — Python Foundations | 🟢 | Essential | Write Python confidently for data and ML work. | 9 | 5 | [`phase-01.md`](roadmap/phase-01.md) |
| 02 — Mathematics for ML | 🟢/🟡 | Essential | Understand the math used by ML algorithms. | 8 | 4 | [`phase-02.md`](roadmap/phase-02.md) |
| 03 — Statistics & Probability | 🟡 | Essential | Reason about uncertainty, sampling, metrics, and evidence. | 8 | 5 | [`phase-03.md`](roadmap/phase-03.md) |
| 04 — Data Analysis & Preparation | 🟡 | Essential | Explore, clean, split, and prepare data without leakage. | 8 | 5 | [`phase-04.md`](roadmap/phase-04.md) |
| 05 — Machine Learning | 🟡 | Essential | Build, evaluate, debug, and compare classical ML models. | 18 | 5 | [`phase-05.md`](roadmap/phase-05.md) |
| 06 — Deep Learning | 🟡/🔴 | Important | Train neural networks, debug training, and use PyTorch. | 13 | 5 | [`phase-06.md`](roadmap/phase-06.md) |
| 07 — NLP | 🟡 | Important | Process text and build NLP systems before transformers. | 9 | 5 | [`phase-07.md`](roadmap/phase-07.md) |
| 08 — Transformers | 🔴 | Important | Understand attention and transformer architectures. | 10 | 5 | [`phase-08.md`](roadmap/phase-08.md) |
| 09 — Generative AI Foundations | 🔴 | Important | Understand generation, inference, prompting, APIs, and multimodal basics. | 15 | 5 | [`phase-09.md`](roadmap/phase-09.md) |
| 10 — Applied LLM Engineering | 🔴 | Important | Choose, secure, tune, and operate LLM-based systems. | 8 | 5 | [`phase-10.md`](roadmap/phase-10.md) |
| 11 — RAG Systems | 🔴 | Important | Build retrieval-grounded systems from search to evaluation. | 17 | 5 | [`phase-11.md`](roadmap/phase-11.md) |
| 12 — LangChain / Framework Abstractions | 🟡/🔴 | Recommended | Learn framework abstractions after manual pipelines. | 9 | 4 | [`phase-12.md`](roadmap/phase-12.md) |
| 13 — LangGraph / Stateful Workflows | 🔴 | Recommended | Model LLM apps as stateful graphs and controlled loops. | 8 | 4 | [`phase-13.md`](roadmap/phase-13.md) |
| 14 — AI Agents | 🔴 | Important | Build tool-using, stateful, secure agent systems. | 12 | 5 | [`phase-14.md`](roadmap/phase-14.md) |
| 15 — Evaluation & Experimentation | 🔴 | Essential | Evaluate ML, LLM, RAG, and agent systems rigorously. | 7 | 5 | [`phase-15.md`](roadmap/phase-15.md) |
| 16 — Deployment & MLOps | 🔴 | Important | Serve, monitor, secure, and maintain ML/AI systems. | 8 | 4 | [`phase-16.md`](roadmap/phase-16.md) |
| 17 — Capstone Engineering | 🔴/⚫ | Essential | Prove independent engineering ability through portfolio systems. | 6 | 6 | [`phase-17.md`](roadmap/phase-17.md) |

**Total scope:** 18 phases, 177 listed units, multiple mini/intermediate/advanced projects, and 6 capstones.

---

## Project Ladder

Projects are introduced throughout the roadmap to prevent tutorial hell.

| Milestone | Project Type | Expected Independence |
|---|---|---|
| Phase 01 | Python data utility | Follow instructions |
| Phase 04 | EDA + data-cleaning report | Partial guide |
| Phase 05 | End-to-end classical ML project | Solve with hints |
| Phase 06 | Neural-network training project | Debug with guidance |
| Phase 07–08 | NLP/transformer project | Make modeling decisions |
| Phase 10–11 | LLM/RAG application | Design retrieval/evaluation choices |
| Phase 14 | Tool-using agent | Handle failures and security constraints |
| Phase 16 | Deployed ML/AI service | Operate and monitor a system |
| Phase 17 | Portfolio capstones | Design independently |

---

## Core Decision Comparisons

The learner should not only know tools, but know when to choose them.

| Comparison | Learned Around | Decision Skill |
|---|---|---|
| Python list vs NumPy array | Phase 01 | General programming vs numerical computation |
| Pandas vs Polars | Phase 04 | Familiar ecosystem vs performance/lazy execution |
| SQL vs NoSQL | Phase 04/16 | Structured querying vs flexible document/key-value storage |
| Train/validation/test vs cross-validation | Phase 04/05 | Reliable estimate vs limited-data robustness |
| Linear model vs tree model | Phase 05 | Interpretability/simplicity vs nonlinear interactions |
| Random Forest vs XGBoost/LightGBM | Phase 05 | Stable default vs high-performance tuned model |
| Accuracy vs F1/ROC-AUC/PR-AUC | Phase 05 | Balanced data vs imbalanced/high-cost errors |
| CNN vs Transformer | Phase 06/08 | Local visual structure vs sequence/global attention |
| Fine-tuning vs RAG | Phase 10/11 | Change model behavior vs inject external knowledge |
| RAG vs long-context prompting | Phase 10/11 | Scalable retrieval vs simple context packing |
| LangChain vs direct API implementation | Phase 12 | Faster composition vs explicit control/debuggability |
| LangGraph vs simple chains | Phase 13 | Stateful cycles/control flow vs linear workflows |
| Single agent vs workflow | Phase 14 | Flexible autonomy vs predictable execution |

---

## Improvement Methodology

When a solution performs poorly, follow this loop before changing tools randomly:

```text
Define the failure clearly
        ↓
Check data quality and leakage
        ↓
Check preprocessing and splits
        ↓
Build/compare against a simple baseline
        ↓
Choose the correct metric
        ↓
Analyze errors by segment/example
        ↓
Tune the current approach
        ↓
Try a justified alternative
        ↓
Validate again on held-out data
        ↓
Document the decision and trade-offs
```

This loop applies to ML models, deep learning, prompts, RAG systems, and agents.

---

## Final Roadmap Validation

| Area | Status | Required Improvement |
|---|---:|---|
| Fundamentals | ✅ | Keep Python/math/statistics before ML; add SQL/database practice as an extension in Phase 04/16. |
| Prerequisites | ✅ | Dependency graph now clarifies phase, spiral, and unit-track dependencies. |
| Topic ordering | ✅ | Manual concepts precede frameworks; framework phases are recommended rather than treated as fundamentals. |
| Practical exercises | ✅ | Progressive exercise ladder added; phase files should keep expanding concrete notebooks over time. |
| Projects | ✅ | Project ladder and capstone rubric added. |
| Industry skills | ✅ | Deployment, monitoring, security, evaluation, documentation, and trade-offs included. |
| Interview preparation | ✅ | Interview prompts added to the learning-system standard. |
| Revision | ✅ | Revision checklists and mastery gates added. |
| Advanced topics | ✅ | Advanced and optional material is labeled separately. |
| Career readiness | ✅ | Final capstone requires independent design, implementation, evaluation, deployment, and explanation. |

### Highest-Priority Remaining Improvements

1. Add exact notebook paths for every unit as notebooks are created.
2. Add dataset recommendations for each project level.
3. Add automated checks for selected exercises.
4. Expand SQL/database coverage in a future Phase 04 or Phase 16 supplement.
5. Add model cards, data cards, and evaluation reports to every major project.

---

## Definition of Complete

The roadmap is complete when:

- [ ] Dependencies are mapped.
- [ ] Each unit has a purpose, prerequisites, practice, and exit criteria.
- [ ] Notebooks are implemented and executable from a clean environment.
- [ ] Concepts are connected instead of isolated.
- [ ] Mistakes and failure cases are documented.
- [ ] Projects exist at increasing independence levels.
- [ ] Evaluation is included from early ML through advanced AI systems.
- [ ] Engineering, security, deployment, and monitoring are included.
- [ ] Capstones demonstrate independent problem solving.
- [ ] Progress, experiments, mistakes, reviews, and project outcomes are tracked.

**Most importantly:** the learner can leave the final capstone and independently design, build, evaluate, debug, secure, deploy, and explain a new AI/ML system never explicitly taught in the curriculum.
