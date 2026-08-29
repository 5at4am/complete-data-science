# Dependency Graph

> **The complete dependency-aware learning progression.**
> A topic appears when it solves a problem created by earlier topics and prepares the learner for later systems.

---

## High-Level Dependency Chain

```text
00 Environment Setup
    ↓ gives reproducible tools
01 Python Foundations
    ↓ gives programming and data manipulation
02 Mathematics for ML
    ↓ explains vectors, gradients, optimization, probability
03 Statistics & Probability
    ↓ explains uncertainty, sampling, metrics, evidence
04 Data Analysis & Preparation
    ↓ produces clean, trustworthy training data
05 Machine Learning
    ↓ introduces modeling, evaluation, error analysis
06 Deep Learning
    ↓ scales learning to representation learning
07 NLP
    ↓ applies modeling to language before modern transformers
08 Transformers
    ↓ explains the architecture behind modern LLMs
09 Generative AI Foundations
    ↓ introduces inference, prompting, APIs, structured generation
10 Applied LLM Engineering
    ↓ teaches selection, cost, security, fine-tuning decisions
11 RAG Systems
    ↓ grounds LLMs in external knowledge
12 LangChain / Framework Abstractions
    ↓ wraps manual pipelines after the learner understands them
13 LangGraph / Stateful Workflows
    ↓ controls loops, state, and conditional execution
14 AI Agents
    ↓ combines tools, memory, planning, evaluation, safety
15 Evaluation & Experimentation
    ↓ formalizes system-level measurement and improvement
16 Deployment & MLOps
    ↓ makes systems usable, monitored, secure, maintainable
17 Capstone Engineering
    ↓ proves independent portfolio-level capability
```

---

## Why Each Phase Comes Next

| From | To | Why this transition exists |
|---|---|---|
| 00 | 01 | Once the environment works, the learner needs Python to express ideas. |
| 01 | 02 | NumPy/Python make math concrete instead of abstract. |
| 02 | 03 | ML needs both optimization math and statistical reasoning. |
| 03 | 04 | Data preparation decisions require statistical intuition. |
| 04 | 05 | Models are only meaningful after data can be cleaned and split safely. |
| 05 | 06 | Classical ML teaches pipelines and metrics before neural-network complexity. |
| 06 | 07 | NLP uses DL ideas but adds language-specific preprocessing and evaluation. |
| 07 | 08 | Transformers solve sequence-model limits learned in NLP. |
| 08 | 09 | Generative AI is easier after attention, tokenization, and transformer mechanics. |
| 09 | 10 | LLM apps require engineering decisions beyond prompting. |
| 10 | 11 | RAG solves knowledge freshness, grounding, and source-traceability limits. |
| 11 | 12 | Frameworks become useful after manual RAG/LLM pipelines are understood. |
| 12 | 13 | Stateful graphs solve limitations of linear chains. |
| 13 | 14 | Agents combine graph control, tools, memory, and safety. |
| 14 | 15 | Complex systems need formal evaluation beyond demo success. |
| 15 | 16 | Measured systems can be deployed and monitored responsibly. |
| 16 | 17 | Capstones require the whole lifecycle, not isolated notebooks. |

---

## Detailed Dependency Map

### Phase 00 — Environment Setup
**Prerequisites:** None
**Units:** Environment verification → Jupyter workflow → Git/version control → Python environments/packages
**Exit dependency:** The learner can create, run, save, version, and reproduce work.

### Phase 01 — Python Foundations
**Prerequisites:** Phase 00
**Units:** Python basics → data structures → functions/scope → OOP → NumPy → Pandas → Matplotlib → file I/O/errors → synthesis
**Exit dependency:** The learner can load, transform, visualize, and organize data with code.

### Phase 02 — Mathematics for ML
**Prerequisites:** Phase 01, especially NumPy
**Units:** Vectors/matrices → matrix operations → derivatives/gradients → gradient descent → probability fundamentals → entropy/cross-entropy → eigenvectors/eigenvalues → synthesis
**Exit dependency:** The learner can reason about model inputs, parameters, optimization, and loss.

### Phase 03 — Statistics & Probability
**Prerequisites:** Phase 02
**Units:** Descriptive stats → distributions → sampling/inference → hypothesis testing → correlation/regression basics → Bayesian thinking → stats for ML → synthesis
**Exit dependency:** The learner can reason about uncertainty, evidence, and metric reliability.

### Phase 04 — Data Analysis & Preparation
**Prerequisites:** Phase 01, Phase 03
**Units:** EDA → cleaning → missing values → outliers → scaling/encoding → splits → leakage → synthesis
**Add-on dependency:** SQL/database basics should be introduced here or in Phase 16 for real-world data access.
**Exit dependency:** The learner can create a trustworthy modeling dataset.

### Phase 05 — Machine Learning
**Prerequisites:** Phase 02, Phase 03, Phase 04
**Units:** Problem formulation → linear regression → logistic regression → evaluation → trees → forests → boosting → KNN → Naive Bayes → SVM → clustering → PCA → feature engineering → imbalanced learning → cross-validation/tuning → interpretation → ensembling → synthesis
**Spiral dependency:** Evaluation starts here and continues through Phases 06, 07, 10, 11, 14, and 15.
**Exit dependency:** The learner can build, improve, and justify classical ML pipelines.

### Phase 06 — Deep Learning
**Prerequisites:** Phase 02, Phase 05
**Units:** Perceptron/activations → losses → backprop → PyTorch MLP → regularization → optimizers/schedules → datasets/dataloaders → training loops → checkpointing/transfer learning → CNNs → RNNs/LSTMs/GRUs → attention → synthesis
**Exit dependency:** The learner can train neural networks and debug training behavior.

### Phase 07 — NLP
**Prerequisites:** Phase 05, Phase 06
**Units:** Text preprocessing/tokenization → BoW/TF-IDF → n-grams → text classification → embeddings → sequence models → attention for NLP → NLP evaluation → synthesis
**Exit dependency:** The learner understands why traditional NLP methods lead toward transformers.

### Phase 08 — Transformers
**Prerequisites:** Phase 06, Phase 07
**Units:** Transformer from scratch → self-attention → positional encoding → encoder-decoder → BERT-style models → GPT-style models → tokenizers → Hugging Face → fine-tuning → synthesis
**Exit dependency:** The learner can explain and use transformer models without treating them as magic APIs.

### Phase 09 — Generative AI Foundations
**Prerequisites:** Phase 08
**Units:** Language models → tokens/tokenization → embeddings → attention recap → pretraining → inference/decoding → temperature/sampling → instruction following/alignment → LLM APIs → prompt engineering → structured output → tool calling → streaming/caching/retries → multimodal overview → synthesis
**Clarification:** Repeated topics are applied here, not reintroduced from scratch.
**Exit dependency:** The learner can call generative models safely and reason about outputs.

### Phase 10 — Applied LLM Engineering
**Prerequisites:** Phase 09
**Units:** Model landscape/selection → context windows → fine-tuning concepts → RAG vs fine-tuning vs long context → cost/latency → security → LLM evaluation → synthesis
**Exit dependency:** The learner can choose an LLM strategy based on requirements, not hype.

### Phase 11 — RAG Systems
**Prerequisites:** Phase 09, Phase 10
**Units:** Why RAG → keyword search → embeddings → vector similarity → vector databases → ingestion → chunking → embedding generation → retrieval → reranking → context construction → generation/grounding → RAG evaluation → naive RAG → advanced RAG → agentic RAG → synthesis
**Exit dependency:** The learner can build grounded systems and diagnose retrieval failures.

### Phase 12 — LangChain / Framework Abstractions
**Prerequisites:** Phase 09, Phase 11
**Units:** Manual LLM pipeline → models/prompts → output parsers → chains → memory → tools → RAG with framework → framework vs manual → synthesis
**Clarification:** This phase teaches abstraction trade-offs, not dependency on one tool.
**Exit dependency:** The learner can decide when framework speed is worth reduced transparency.

### Phase 13 — LangGraph / Stateful Workflows
**Prerequisites:** Phase 12 and basic agent-loop intuition from Phase 14.1/14.2 if needed
**Units:** Manual state-machine agent → state/nodes → edges/routing → loops/cycles → memory → human approval → failure handling → synthesis
**Clarification:** LangGraph is a framework for controlled stateful workflows. It should be learned after simple manual loops are understood.
**Exit dependency:** The learner can model controlled, inspectable, resumable LLM workflows.

### Phase 14 — AI Agents
**Prerequisites:** Phase 12, Phase 13 or equivalent manual state-machine understanding
**Units:** Tool calling → single-tool agent → planning → execution → reflection → memory → multi-tool agent → stateful agent → multi-agent systems → failure modes → security → synthesis
**Exit dependency:** The learner can build agents with boundaries, evaluation, and safety controls.

### Phase 15 — Evaluation & Experimentation
**Prerequisites:** Phase 05, Phase 11, Phase 14
**Units:** ML evaluation deep dive → LLM evaluation → RAG evaluation → agent evaluation → evaluation datasets → experiment tracking → synthesis
**Clarification:** This is the formal evaluation phase; basic evaluation begins earlier.
**Exit dependency:** The learner can measure quality instead of trusting demos.

### Phase 16 — Deployment & MLOps
**Prerequisites:** Phase 15
**Units:** FastAPI → model serving → Docker → CI/CD → monitoring/observability → LLMOps → production security → synthesis
**Add-on dependency:** Include data/model versioning, model registry, drift detection, retraining triggers, secrets management, and cost monitoring.
**Exit dependency:** The learner can operate systems beyond notebooks.

### Phase 17 — Capstone Engineering
**Prerequisites:** All phases
**Units:** ML capstone → DL capstone → NLP capstone → RAG capstone → agent capstone → final AI engineering capstone
**Exit dependency:** The learner can independently design and build a novel ML/AI system.

---

## Spiral Learning Tracks

Repeated topics are intentional only when the role changes.

### Vectors → Embeddings → Vector Search → RAG
```text
Phase 02: vectors as math objects
    ↓
Phase 07: embeddings as language representations
    ↓
Phase 09: embeddings as model inputs and semantic representations
    ↓
Phase 11: embeddings as retrieval infrastructure
```

### Tokenization Track
```text
Phase 07: simple text tokenization/preprocessing
    ↓
Phase 08: BPE/WordPiece and transformer tokenizers
    ↓
Phase 09: tokens as context, cost, latency, and generation units
```

### Attention Track
```text
Phase 06: attention as a neural-network mechanism
    ↓
Phase 07: attention for language sequences
    ↓
Phase 08: self-attention as transformer foundation
    ↓
Phase 09: attention effects on generation and context behavior
```

### Evaluation Track
```text
Phase 03: statistical reliability
    ↓
Phase 05: ML metrics and baselines
    ↓
Phase 06: validation curves and training diagnostics
    ↓
Phase 07: NLP task metrics
    ↓
Phase 10: LLM task evaluation
    ↓
Phase 11: retrieval and answer-grounding evaluation
    ↓
Phase 14: agent task success and safety evaluation
    ↓
Phase 15: formal evaluation datasets and experiment tracking
```

### Production Track
```text
Phase 00: reproducible environment
    ↓
Phase 01: organized code and file I/O
    ↓
Phase 04: reproducible preprocessing
    ↓
Phase 05: pipelines, baselines, model artifacts
    ↓
Phase 10: cost, latency, API reliability
    ↓
Phase 11: indexes, ingestion, grounding
    ↓
Phase 16: serving, CI/CD, monitoring, security
```

### Security / Responsible AI Track
```text
Phase 04: privacy, leakage, sensitive data handling
    ↓
Phase 05: bias, fairness, inappropriate metrics
    ↓
Phase 09: prompt injection basics and unsafe outputs
    ↓
Phase 10: LLM security, privacy, policy constraints
    ↓
Phase 11: poisoned documents and retrieval attacks
    ↓
Phase 14: tool permissions, sandboxing, untrusted instructions
    ↓
Phase 16: secrets, access control, monitoring, governance
```

---

## Unit-Level Standard

Every important unit should specify:

- What it is
- Why it matters
- Why it appears here
- Prerequisites
- Mental model
- Core concepts
- Implementation/lab
- Simple example
- Real-world example
- Common mistakes
- Debugging/failure cases
- Alternatives and trade-offs
- Best practices
- Hands-on practice
- Mini project when useful
- Knowledge check
- Exit criteria
- Next step

Small units may use a compressed version, but they still need purpose, practice, and exit criteria.
