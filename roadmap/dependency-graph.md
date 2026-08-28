# Dependency Graph

> **The complete dependency-aware learning progression.**
> Each phase depends on the phases above it. Each unit depends on the units
> listed as prerequisites.

---

## High-Level Dependency Chain

```
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
Phase 09 — Generative AI
        ↓
Phase 10 — LLMs
        ↓
Phase 11 — RAG
        ↓
Phase 12 — LangChain
        ↓
Phase 13 — LangGraph
        ↓
Phase 14 — AI Agents
        ↓
Phase 15 — Evaluation
        ↓
Phase 16 — Deployment & MLOps
        ↓
Phase 17 — Capstone Engineering
```

---

## Detailed Dependency Map

### Phase 00 — Environment Setup
**Prerequisites:** None
**Units:**
- 00.1 Environment verification
- 00.2 Jupyter workflow
- 00.3 Git & version control
- 00.4 Python environments & packages

### Phase 01 — Python Foundations
**Prerequisites:** Phase 00
**Units:**
- 01.1 Python basics (variables, types, control flow)
- 01.2 Data structures (lists, dicts, sets, tuples)
- 01.3 Functions & scope
- 01.4 Object-oriented programming
- 01.5 NumPy fundamentals
- 01.6 Pandas fundamentals
- 01.7 Matplotlib & visualization
- 01.8 File I/O & error handling
- 01.9 Python for data work (review + synthesis)

### Phase 02 — Mathematics for ML
**Prerequisites:** Phase 01 (NumPy)
**Units:**
- 02.1 Linear algebra: vectors & matrices
- 02.2 Matrix operations & multiplication
- 02.3 Calculus: derivatives & gradients
- 02.4 Optimization: gradient descent
- 02.5 Probability fundamentals
- 02.6 Information theory: entropy & cross-entropy
- 02.7 Linear algebra: eigenvalues & eigenvectors
- 02.8 Math synthesis & review

### Phase 03 — Statistics & Probability
**Prerequisites:** Phase 02
**Units:**
- 03.1 Descriptive statistics
- 03.2 Probability distributions
- 03.3 Inferential statistics & sampling
- 03.4 Hypothesis testing
- 03.5 Correlation & regression basics
- 03.6 Bayesian thinking
- 03.7 Statistics for ML (bias-variance, evaluation)
- 03.8 Statistics synthesis & review

### Phase 04 — Data Analysis & Preparation
**Prerequisites:** Phase 01, Phase 03
**Units:**
- 04.1 Data exploration & EDA
- 04.2 Data cleaning
- 04.3 Missing values
- 04.4 Outliers
- 04.5 Feature scaling & encoding
- 04.6 Train/validation/test splits
- 04.7 Data leakage (critical)
- 04.8 Data preparation synthesis & review

### Phase 05 — Machine Learning
**Prerequisites:** Phase 02, Phase 03, Phase 04
**Units:**
- 05.1 ML fundamentals & problem formulation
- 05.2 Linear regression (from scratch)
- 05.3 Logistic regression (from scratch)
- 05.4 Model evaluation (regression & classification)
- 05.5 Decision trees
- 05.6 Random forests
- 05.7 Gradient boosting (XGBoost, LightGBM)
- 05.8 k-Nearest Neighbors
- 05.9 Naive Bayes
- 05.10 Support Vector Machines
- 05.11 Clustering (K-Means, Hierarchical, DBSCAN)
- 05.12 Dimensionality reduction (PCA)
- 05.13 Feature engineering
- 05.14 Imbalanced learning
- 05.15 Cross-validation & hyperparameter tuning
- 05.16 Model interpretation
- 05.17 Ensembling
- 05.18 ML synthesis & review

### Phase 06 — Deep Learning
**Prerequisites:** Phase 02, Phase 05
**Units:**
- 06.1 Perceptron & activation functions
- 06.2 Loss functions
- 06.3 Gradient descent & backpropagation (from scratch)
- 06.4 MLP with PyTorch
- 06.5 Regularization (dropout, batch norm)
- 06.6 Optimizers & learning rate schedules
- 06.7 PyTorch datasets & dataloaders
- 06.8 Training loops & validation
- 06.9 Checkpointing & transfer learning
- 06.10 CNNs for computer vision
- 06.11 RNNs, LSTMs, GRUs
- 06.12 Attention mechanism
- 06.13 Deep learning synthesis & review

### Phase 07 — NLP
**Prerequisites:** Phase 05, Phase 06
**Units:**
- 07.1 Text preprocessing & tokenization
- 07.2 Bag of Words & TF-IDF
- 07.3 N-grams
- 07.4 Text classification
- 07.5 Word embeddings (Word2Vec concepts)
- 07.6 Sequence models for NLP
- 07.7 Attention for NLP
- 07.8 NLP evaluation
- 07.9 NLP synthesis & review

### Phase 08 — Transformers
**Prerequisites:** Phase 06, Phase 07
**Units:**
- 08.1 Transformer architecture (from scratch)
- 08.2 Self-attention deep dive
- 08.3 Positional encoding
- 08.4 Encoder-decoder architecture
- 08.5 BERT-style models
- 08.6 Causal language models (GPT-style)
- 08.7 Tokenizers (BPE, WordPiece)
- 08.8 Hugging Face ecosystem
- 08.9 Fine-tuning transformers
- 08.10 Transformers synthesis & review

### Phase 09 — Generative AI
**Prerequisites:** Phase 08
**Units:**
- 09.1 What is a language model?
- 09.2 Tokens & tokenization
- 09.3 Embeddings
- 09.4 Attention & transformer recap
- 09.5 Pretraining
- 09.6 Inference & decoding
- 09.7 Temperature & sampling
- 09.8 Instruction following & alignment
- 09.9 LLM APIs
- 09.10 Prompt engineering
- 09.11 Structured output
- 09.12 Function/tool calling
- 09.13 Streaming, caching, retries
- 09.14 Generative AI synthesis & review

### Phase 10 — LLMs
**Prerequisites:** Phase 09
**Units:**
- 10.1 LLM landscape & model selection
- 10.2 Context windows
- 10.3 Fine-tuning concepts
- 10.4 RAG vs fine-tuning vs long-context
- 10.5 LLM cost & latency
- 10.6 LLM security
- 10.7 LLM evaluation
- 10.8 LLM synthesis & review

### Phase 11 — RAG
**Prerequisites:** Phase 09, Phase 10
**Units:**
- 11.1 Why RAG exists
- 11.2 Keyword search
- 11.3 Embeddings
- 11.4 Vector similarity
- 11.5 Vector databases
- 11.6 Document ingestion
- 11.7 Chunking
- 11.8 Embedding generation
- 11.9 Retrieval
- 11.10 Reranking
- 11.11 Context construction
- 11.12 Generation & grounding
- 11.13 RAG evaluation
- 11.14 Naive RAG (build)
- 11.15 Advanced RAG (hybrid, metadata, multi-query)
- 11.16 Agentic RAG
- 11.17 RAG synthesis & review

### Phase 12 — LangChain
**Prerequisites:** Phase 09, Phase 11
**Units:**
- 12.1 Manual LLM pipeline (no framework)
- 12.2 LangChain models & prompts
- 12.3 Output parsers
- 12.4 Chains
- 12.5 Memory
- 12.6 Tools
- 12.7 RAG with LangChain
- 12.8 LangChain vs manual implementation
- 12.9 LangChain synthesis & review

### Phase 13 — LangGraph
**Prerequisites:** Phase 12
**Units:**
- 13.1 Manual state-machine agent
- 13.2 LangGraph state & nodes
- 13.3 Edges & conditional routing
- 13.4 Loops & cycles
- 13.5 Memory
- 13.6 Human approval
- 13.7 Failure handling
- 13.8 LangGraph synthesis & review

### Phase 14 — AI Agents
**Prerequisites:** Phase 12, Phase 13
**Units:**
- 14.1 LLM call → tool calling
- 14.2 Single-tool agent
- 14.3 Planning
- 14.4 Execution
- 14.5 Reflection
- 14.6 Memory
- 14.7 Multi-tool agent
- 14.8 Stateful agent
- 14.9 Multi-agent systems
- 14.10 Agent failure modes
- 14.11 Agent security
- 14.12 Agent synthesis & review

### Phase 15 — Evaluation
**Prerequisites:** Phase 05, Phase 11, Phase 14
**Units:**
- 15.1 ML evaluation deep dive
- 15.2 LLM evaluation
- 15.3 RAG evaluation
- 15.4 Agent evaluation
- 15.5 Evaluation datasets
- 15.6 Experiment tracking
- 15.7 Evaluation synthesis & review

### Phase 16 — Deployment & MLOps
**Prerequisites:** Phase 15
**Units:**
- 16.1 API design (FastAPI)
- 16.2 Model serving
- 16.3 Docker
- 16.4 CI/CD
- 16.5 Monitoring & observability
- 16.6 LLMOps
- 16.7 Security in production
- 16.8 Deployment synthesis & review

### Phase 17 — Capstone Engineering
**Prerequisites:** All phases
**Units:**
- 17.1 ML Capstone
- 17.2 Deep Learning Capstone
- 17.3 NLP Capstone
- 17.4 RAG Capstone
- 17.5 Agent Capstone
- 17.6 Final AI Engineering Capstone

---

## Cross-Phase Dependencies (Spiral Learning)

### Vectors → Embeddings → Vector Search → RAG
```
Phase 02 (vectors, linear algebra)
    ↓
Phase 07 (word embeddings)
    ↓
Phase 09 (embeddings)
    ↓
Phase 11 (vector search, RAG)
```

### Functions → Python → APIs → Tools → Agents
```
Phase 01 (Python functions)
    ↓
Phase 09 (LLM APIs)
    ↓
Phase 12 (tools)
    ↓
Phase 14 (agents)
```

### Statistics → ML → Deep Learning → Evaluation
```
Phase 03 (statistics)
    ↓
Phase 05 (ML)
    ↓
Phase 06 (deep learning)
    ↓
Phase 15 (evaluation)
```

### Probability → Classification → Generative Models
```
Phase 02 (probability)
    ↓
Phase 05 (classification)
    ↓
Phase 09 (generative AI)
```
