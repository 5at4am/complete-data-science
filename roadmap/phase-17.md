# Phase 17 — Capstone Engineering

> **Goal:** Complete capstones that require architectural decisions and independent engineering. Each capstone is a complete system — not a bigger notebook — with a problem, constraints, decisions, implementation, evaluation, documentation, and clear trade-off reasoning.

**Difficulty:** 🔴 Advanced  
**Priority:** Essential  
**Prerequisites:** Phases 00–16  
**Mastery target:** Level 6 — teaching and independent system design

---

## Why This Phase Exists

Every prior phase teaches isolated skills: data cleaning, modeling, deployment, evaluation. Capstone engineering forces the learner to combine those skills into working systems that solve real problems. This is the phase where the learner proves they can work without a tutorial — making choices, defending them, debugging failures, and delivering results. It is also where weak foundations surface, because a capstone exposes gaps that small exercises can hide.

### Phase Mental Model

A capstone is a self-contained engineering project. The learner acts as the architect: defining the problem, choosing the tools, designing the system, implementing it, evaluating it honestly, and communicating the results. The key shift is from "follow instructions" to "make decisions and own them."

```text
Problem definition → constraints → data → baseline → model selection
       ↓
Architecture design → implementation → evaluation → error analysis
       ↓
Documentation → presentation → monitoring plan → security review
```

### What This Phase Prepares For

- independent project ownership in professional ML/AI roles
- interview problem-solving and system design conversations
- portfolio demonstration for job applications
- the ability to learn new domains and build systems without guidance
- mentoring and teaching others

---

## Units

---

### Unit 17.1 — ML Capstone

**What is it?**  
An end-to-end classical machine learning system that takes raw data, engineers features, trains a model, evaluates it rigorously, and serves predictions through an API.

**Why does it matter?**  
Most production ML systems in industry are still classical ML — gradient boosting, logistic regression, random forests — not deep learning. The ability to build a complete classical ML pipeline from scratch is the baseline expectation for any ML engineer.

**Why learn it here?**  
After completing Phases 01–16, the learner has all the pieces: data manipulation, feature engineering, model training, evaluation, and API development. This capstone forces them to assemble those pieces into a coherent system with real constraints.

**Prerequisites:** Phases 01–10 (Python, data prep, classical ML, evaluation, deployment basics).

**Mental Model:**  
Think of this as building a predictive service. The model is one component — the data pipeline, feature engineering, evaluation, and serving infrastructure matter just as much.

**Problem Statement:**  
Build a complete ML system that predicts a meaningful target from tabular or structured data. The system must include a data pipeline, feature engineering, model training with baseline comparison, evaluation with multiple metrics, error analysis, a simple API, and documentation.

**Requirements:**

- real or realistic dataset with enough complexity to require cleaning
- data validation and preprocessing pipeline
- at least 3 feature engineering decisions with justification
- baseline model (e.g., majority class, mean prediction, simple logistic regression)
- at least 2 competing models with comparison
- cross-validation or proper train/validation/test split
- evaluation with task-appropriate metrics
- error analysis on misclassified or high-error examples
- model card documenting data, performance, and limitations
- simple API or script for inference
- reproducible setup (requirements file, data script, or README)

**Concepts Used:**

- data cleaning and missing value handling
- feature engineering and selection
- train/validation/test splitting
- cross-validation
- hyperparameter tuning
- model comparison (accuracy, precision, recall, F1, AUC-ROC as appropriate)
- error analysis
- API development (FastAPI or Flask)

**Suggested Architecture:**

```text
Raw data → validation → cleaning → feature engineering → split
                                                          ↓
                                    baseline model ← training pipeline → candidate models
                                                          ↓
                                                    evaluation → error analysis
                                                          ↓
                                                    model card + API + docs
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| Data understanding | Day 1–2 | EDA notebook, data quality report |
| Baseline | Day 3 | Simple model with metrics |
| Feature engineering | Day 4–6 | Pipeline with documented decisions |
| Model comparison | Day 7–9 | Results table with metric justification |
| Error analysis | Day 10 | Failure case examples and patterns |
| API + docs | Day 11–14 | Working inference endpoint, README, model card |

**Expected Output:**

- reproducible Python project (not just a notebook)
- evaluation report with metric tables
- error analysis notebook
- model card
- README with setup instructions and decisions

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| Problem framing | Clear user/business problem with measurable criteria | Vague or no measurable goal |
| Data handling | Source documented, quality inspected, leakage audited | Data used blindly |
| Baseline | Simple baseline implemented and compared | Only complex model shown |
| Feature engineering | At least 3 decisions with justification | Features added without reasoning |
| Evaluation | Task-appropriate metrics, cross-validation or proper split | Only accuracy on one split |
| Error analysis | Weak segments and failure examples inspected | No analysis of failures |
| Engineering | Reproducible setup, tests, artifacts | Notebook-only, hard to rerun |
| Documentation | README, model card, limitations documented | No explanation of decisions |

**Common Mistakes:**

- using test data for feature selection or hyperparameter tuning (data leakage)
- reporting accuracy on imbalanced data without precision/recall
- skipping baseline comparison
- no error analysis
- notebook-only delivery with no reproducible setup

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| High validation accuracy but poor real performance | Data leakage or bad split | Inspect features and split logic | Re-split before preprocessing |
| Model performs worse than baseline | Bad features or preprocessing | Check feature distributions and target encoding | Fix preprocessing or try simpler features |
| API returns errors on new data | Inconsistent input format | Test with edge cases | Add input validation |

**Possible Improvements / Advanced Extensions:**

- experiment tracking with MLflow or Weights & Biases
- automated feature selection
- SHAP or LIME explanations
- CI/CD pipeline for retraining
- Docker containerization
- A/B testing framework

**Hands-On Practice (before the capstone):**

1. Complete a Kaggle tabular competition end-to-end.
2. Build a baseline model for a new dataset in under 30 minutes.
3. Implement cross-validation from scratch with sklearn.
4. Write a model card for an existing model.

**Knowledge Check:**

- Why is a baseline important before trying complex models?
- How do you detect data leakage in a feature engineering pipeline?
- When should you use precision vs recall vs F1?
- What is the difference between model validation and model evaluation?
- How would you explain your model choice to a non-technical stakeholder?

**Exit Criteria:**

- You can build a complete ML system from raw data to serving without a tutorial.
- You can justify every architectural decision.
- You can identify and explain failure cases.

---

### Unit 17.2 — Deep Learning Capstone

**What is it?**  
A complete deep learning system that preprocesses data, designs an architecture, trains with proper optimization, evaluates rigorously, and serves predictions — with attention to GPU memory, reproducibility, and training stability.

**Why does it matter?**  
Deep learning dominates vision, audio, and increasingly tabular and sequential tasks. Building a DL system end-to-end teaches the unique challenges: architecture selection, training dynamics, hyperparameter sensitivity, compute management, and debugging strategies that classical ML does not require.

**Why learn it here?**  
After completing Phases 01–12, the learner understands neural network fundamentals, PyTorch or TensorFlow, training loops, and regularization. This capstone forces them to apply that knowledge to a real problem with real constraints.

**Prerequisites:** Phases 01–12 (Python, data prep, DL fundamentals, training loops, regularization, evaluation).

**Mental Model:**  
A deep learning system has three core concerns: data (preprocessing and augmentation), model (architecture and initialization), and training (optimizer, learning rate, regularization). Problems in any area manifest as training instability or poor generalization.

**Problem Statement:**  
Build a complete deep learning system for a non-trivial task (image classification, time series forecasting, sequence modeling, or similar). The system must include data preprocessing, a justified architecture choice, training with monitoring, evaluation with multiple metrics, error analysis, and serving.

**Requirements:**

- dataset with enough complexity to require thoughtful preprocessing
- data augmentation or preprocessing pipeline
- architecture choice with justification (why this architecture for this problem?)
- training loop with loss curves monitoring
- learning rate scheduling or adaptive optimizer
- regularization (dropout, weight decay, early stopping, or augmentation)
- evaluation on held-out test set with multiple metrics
- error analysis on failure cases
- GPU/memory-aware training (if applicable)
- reproducible setup (seed setting, requirements, README)

**Concepts Used:**

- data preprocessing and normalization
- data augmentation
- architecture design (CNN, RNN, Transformer, or hybrid)
- loss function selection
- optimizer selection (SGD, Adam, AdamW)
- learning rate scheduling
- batch size effects
- regularization techniques
- gradient monitoring
- checkpointing

**Suggested Architecture:**

```text
Raw data → preprocessing → augmentation → DataLoader
                                              ↓
                              Architecture → loss → optimizer
                                              ↓
                                   Training loop (with monitoring)
                                              ↓
                              Checkpoint → evaluation → error analysis
                                              ↓
                                    Model card + serving + docs
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| Data pipeline | Day 1–2 | Preprocessed data with augmentation |
| Architecture | Day 3 | Justified architecture selection |
| Training | Day 4–7 | Loss curves, converged model |
| Evaluation | Day 8–9 | Test metrics, confusion matrix or equivalent |
| Error analysis | Day 10 | Failure case inspection |
| Documentation | Day 11–14 | README, model card, training logs |

**Expected Output:**

- reproducible training script
- training loss and metric curves
- evaluation report with multiple metrics
- error analysis examples
- model card
- README with architecture rationale

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| Data pipeline | Preprocessing and augmentation justified | Default transforms without reasoning |
| Architecture | Choice motivated by problem and constraints | Architecture chosen because "it's popular" |
| Training | Loss curves monitored, stable convergence | No monitoring, unstable training ignored |
| Regularization | Appropriate techniques applied and measured | No regularization or untested |
| Evaluation | Multiple metrics, test set used correctly | Only training accuracy reported |
| Error analysis | Failure cases inspected and explained | No analysis |
| Engineering | Reproducible, seeded, documented | Non-reproducible, no seed |

**Common Mistakes:**

- training on the test set (including test data in preprocessing statistics)
- not monitoring loss curves (missing overfitting or underfitting)
- using a learning rate that is too high or too low
- ignoring GPU memory limits
- not setting random seeds (non-reproducible results)
- data leakage through augmentation on test data

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Loss does not decrease | Learning rate too high/low, bad data, wrong loss | Plot loss curve, check gradients | Adjust LR, check data pipeline |
| Training accuracy high, validation low | Overfitting | Compare train/val curves | Add regularization, more data, simpler model |
| Loss goes to NaN | Exploding gradients or bad data | Check gradient norms | Gradient clipping, lower LR, check data |
| Out of memory | Batch too large or model too big | Monitor GPU memory | Reduce batch size, use mixed precision |

**Possible Improvements / Advanced Extensions:**

- mixed-precision training (AMP)
- distributed training
- architecture search or ablation study
- interpretability (Grad-CAM, attention visualization)
- model compression (pruning, quantization)
- deployment to edge or mobile

**Hands-On Practice (before the capstone):**

1. Train a CNN on CIFAR-10 and beat a simple baseline.
2. Implement a training loop from scratch in PyTorch.
3. Diagnose overfitting by comparing train/val curves.
4. Experiment with learning rate scheduling and compare results.

**Knowledge Check:**

- Why does architecture choice depend on the data modality?
- How do you detect overfitting before it wastes compute?
- What is the effect of batch size on training dynamics?
- Why is data leakage through preprocessing statistics dangerous?
- How do you make a DL experiment reproducible?

**Exit Criteria:**

- You can design, train, and evaluate a DL system for a given problem.
- You can diagnose and fix training failures.
- You can justify architecture and hyperparameter choices.

---

### Unit 17.3 — NLP Capstone

**What is it?**  
A complete NLP system that processes text data, applies tokenization and embedding strategies, trains or fine-tunes a model, evaluates with language-appropriate metrics, and serves predictions — with attention to preprocessing, vocabulary, and evaluation beyond accuracy.

**Why does it matter?**  
NLP problems require domain-specific preprocessing (tokenization, stopword handling, subword splitting), evaluation (BLEU, ROUGE, F1 per class, hallucination rate), and architecture choices (classical ML on TF-IDF vs. transformer fine-tuning). This capstone teaches those unique considerations.

**Why learn it here?**  
After completing Phases 01–13, the learner understands text preprocessing, embeddings, transformers, and evaluation. This capstone combines those skills into a working NLP application.

**Prerequisites:** Phases 01–13 (Python, data prep, NLP fundamentals, transformers, evaluation).

**Mental Model:**  
An NLP system transforms raw text into structured representations, applies a model, and produces text-level or token-level outputs. The key challenge is that text is messy, high-dimensional, and ambiguous — preprocessing and evaluation must account for this.

**Problem Statement:**  
Build a complete NLP system for a non-trivial text task (sentiment analysis, named entity recognition, text classification, summarization, or similar). The system must include text preprocessing, tokenization, model training or fine-tuning, evaluation with language-appropriate metrics, error analysis, and serving.

**Requirements:**

- text dataset with preprocessing needs (noise, inconsistent formatting, class imbalance)
- tokenization strategy with justification (word-level, subword, sentencepiece)
- baseline model (e.g., TF-IDF + logistic regression or simple rules)
- main model (e.g., transformer fine-tuning or custom architecture)
- evaluation with task-appropriate metrics (F1, BLEU, ROUGE, accuracy as relevant)
- error analysis on misclassified or low-confidence examples
- inference script or API
- reproducible setup

**Concepts Used:**

- text preprocessing (lowercasing, lemmatization, stopword removal)
- tokenization (BPE, WordPiece, SentencePiece)
- embeddings (Word2Vec, GloVe, contextual embeddings)
- sequence modeling (RNN, LSTM, or transformer)
- fine-tuning pre-trained models
- evaluation metrics for text tasks
- class imbalance handling in text

**Suggested Architecture:**

```text
Raw text → cleaning → tokenization → vocabulary → embedding/encoding
                                                          ↓
                                        baseline (TF-IDF + classifier)
                                                          ↓
                                        main model (transformer fine-tune)
                                                          ↓
                                        evaluation → error analysis
                                                          ↓
                                        model card + serving + docs
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| Data understanding | Day 1 | Text EDA, vocabulary analysis |
| Preprocessing pipeline | Day 2–3 | Tokenization and cleaning pipeline |
| Baseline | Day 4 | Simple model with metrics |
| Main model | Day 5–8 | Fine-tuned or trained model |
| Evaluation | Day 9–10 | Metrics table, error analysis |
| Documentation | Day 11–14 | README, model card, serving script |

**Expected Output:**

- reproducible NLP pipeline
- baseline and main model comparison
- evaluation report with language-appropriate metrics
- error analysis notebook
- model card
- README

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| Text preprocessing | Justified tokenization and cleaning choices | Default preprocessing without reasoning |
| Baseline | Simple baseline compared against | No baseline |
| Model choice | Architecture motivated by task and data size | Model chosen without justification |
| Evaluation | Task-appropriate metrics, not just accuracy | Only accuracy on balanced test set |
| Error analysis | Failure cases inspected, patterns identified | No error analysis |
| Reproducibility | Seeded, documented, runnable | Non-reproducible |

**Common Mistakes:**

- using accuracy on imbalanced text classification
- preprocessing that destroys signal (aggressive stopword removal)
- fine-tuning a large model on a tiny dataset without regularization
- not handling out-of-vocabulary tokens
- evaluating text generation with only perplexity

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model predicts one class for all inputs | Class imbalance or broken labels | Check label distribution | Resample, use class weights |
| BLEU/ROUGE scores are low | Generation quality or reference mismatch | Inspect outputs vs references | Improve decoding strategy |
| Out of memory during fine-tuning | Sequence length too long or batch too large | Check sequence lengths | Truncate, reduce batch, use gradient accumulation |

**Possible Improvements / Advanced Extensions:**

- active learning for labeling
- few-shot or zero-shot classification
- multilingual support
- text generation with evaluation (human eval, GPT-4 as judge)
- deployment with streaming output
- multilabel classification

**Hands-On Practice (before the capstone):**

1. Fine-tune BERT on a text classification dataset.
2. Implement a TF-IDF + logistic regression baseline.
3. Compare tokenization strategies on the same dataset.
4. Analyze misclassified examples and identify patterns.

**Knowledge Check:**

- Why is accuracy often a poor metric for text classification?
- How does tokenization choice affect model performance?
- When should you use a pre-trained model vs. train from scratch?
- What is the difference between BLEU and ROUGE?
- How do you handle class imbalance in text tasks?

**Exit Criteria:**

- You can build an NLP system from raw text to serving.
- You can choose tokenization and model strategies with justification.
- You can evaluate with language-appropriate metrics and analyze failures.

---

### Unit 17.4 — RAG Capstone

**What is it?**  
A complete Retrieval-Augmented Generation system that ingests documents, chunks and embeds them, retrieves relevant context, generates grounded answers, and evaluates both retrieval quality and answer faithfulness.

**Why does it matter?**  
RAG is the dominant pattern for grounding LLMs in private or changing knowledge. Production RAG systems require careful attention to chunking, embedding, retrieval, reranking, generation, and evaluation — each with unique failure modes. Building one end-to-end teaches these challenges.

**Why learn it here?**  
After completing Phases 11–14, the learner understands embeddings, vector databases, retrieval, reranking, and LLM generation. This capstone combines those into a production-style system.

**Prerequisites:** Phases 01–14 (Python, data prep, embeddings, vector search, LLMs, evaluation).

**Mental Model:**  
RAG is a pipeline: ingest → chunk → embed → store → retrieve → rerank → generate → evaluate. Each step has quality levers. Retrieval failure cascades into generation failure — you cannot generate a good answer from irrelevant context.

**Problem Statement:**  
Build a complete RAG system that ingests a document collection, retrieves relevant chunks, generates grounded answers, and evaluates retrieval quality, faithfulness, and citation correctness. The system must handle unanswerable questions gracefully.

**Requirements:**

- document collection (at least 100 documents or pages)
- document ingestion pipeline (parsing, cleaning)
- chunking strategy with justification (fixed-size, semantic, recursive)
- embedding model selection with justification
- vector store setup
- retrieval pipeline (basic + hybrid or reranking)
- generation with context injection
- evaluation: retrieval metrics (recall@k, MRR), answer quality (faithfulness, correctness)
- unanswerable question handling
- citation or source attribution
- latency and cost tracking
- reproducible setup

**Concepts Used:**

- document parsing and cleaning
- chunking strategies
- embedding models and similarity search
- vector databases (FAISS, Chroma, Pinecone, Weaviate, Qdrant)
- hybrid retrieval (keyword + semantic)
- reranking (cross-encoder)
- LLM generation with context
- faithfulness evaluation
- hallucination detection

**Suggested Architecture:**

```text
Documents → parsing → cleaning → chunking → embedding → vector store
                                                                ↓
User query → query embedding → retrieval → reranking → context selection
                                                                ↓
                                              LLM generation → answer + citations
                                                                ↓
                                              evaluation (retrieval + faithfulness)
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| Ingestion pipeline | Day 1–3 | Chunked, embedded documents in vector store |
| Retrieval | Day 4–5 | Retrieval pipeline with basic metrics |
| Generation | Day 6–7 | LLM generating grounded answers |
| Reranking | Day 8 | Reranker improving retrieval quality |
| Evaluation | Day 9–11 | Retrieval + faithfulness + citation metrics |
| Documentation | Day 12–14 | README, eval report, cost analysis |

**Expected Output:**

- working RAG pipeline (ingestion to generation)
- evaluation report with retrieval and faithfulness metrics
- chunking strategy comparison
- error analysis on retrieval failures and hallucinations
- cost and latency report
- README with architecture decisions

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| Chunking | Strategy justified, compared with alternatives | Default chunking without reasoning |
| Retrieval | Recall@k and MRR reported, hybrid or reranked | Only "it seems to work" |
| Generation | Faithfulness measured, citations provided | No faithfulness check |
| Unanswerable questions | System says "I don't know" when appropriate | Hallucinates on unknowns |
| Evaluation | Retrieval + answer quality metrics, error analysis | Only demo examples |
| Cost/latency | Tracked and reported | Ignored |
| Engineering | Reproducible, documented, runnable | Notebook-only |

**Common Mistakes:**

- chunking too large (retrieval includes irrelevant context) or too small (loss of context)
- using cosine similarity without normalizing embeddings
- not evaluating retrieval separately from generation
- generating answers without faithfulness checking
- ignoring latency and cost
- not handling unanswerable questions

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Retrieved chunks are irrelevant | Poor chunking or embedding | Inspect top-k results | Adjust chunk size, try different embedding model |
| Answer is fluent but wrong | Hallucination or poor retrieval | Check retrieved context | Improve retrieval, add faithfulness constraints |
| System is slow | Embedding or retrieval latency | Profile each step | Cache embeddings, use faster retrieval |
| Unanswerable question gets an answer | No confidence threshold | Test with out-of-scope questions | Add "I don't know" logic |

**Possible Improvements / Advanced Extensions:**

- multi-modal RAG (images, tables)
- graph-based retrieval (knowledge graphs)
- self-querying or query decomposition
- adaptive chunking based on query
- evaluation with human annotations
- streaming answers with source highlighting

**Hands-On Practice (before the capstone):**

1. Build a basic RAG pipeline with FAISS and an LLM API.
2. Compare fixed-size vs. semantic chunking on the same documents.
3. Implement recall@k evaluation for a retrieval pipeline.
4. Test a RAG system with unanswerable questions.

**Knowledge Check:**

- Why does retrieval quality dominate generation quality?
- How does chunk size affect retrieval precision and recall?
- What is the difference between a cross-encoder reranker and a bi-encoder?
- How do you measure faithfulness in generated answers?
- When should you use RAG vs. fine-tuning?

**Exit Criteria:**

- You can build a complete RAG system from ingestion to evaluation.
- You can measure and improve retrieval quality independently.
- You can detect and handle hallucinations.

---

### Unit 17.5 — Agent Capstone

**What is it?**  
A complete stateful, tool-using agent system that plans, executes multi-step tasks, manages memory, uses external tools safely, and evaluates task success with trace analysis.

**Why does it matter?**  
Agents are the frontier of LLM applications. They require understanding of tool use, planning, state management, memory, safety, and evaluation — all of which are harder than single-turn generation. This capstone teaches these challenges in a controlled setting.

**Why learn it here?**  
After completing Phases 14–16, the learner understands LLM tool use, planning patterns, memory systems, and evaluation. This capstone combines those into a working agent with safety constraints.

**Prerequisites:** Phases 01–16 (Python, LLMs, tool use, planning, memory, evaluation, deployment).

**Mental Model:**  
An agent is an LLM in a loop: observe → think → act → observe → think → act → ... until done. The key challenges are: (1) choosing the right tool at each step, (2) maintaining coherent state across steps, (3) knowing when to stop, and (4) operating safely within constraints.

**Problem Statement:**  
Build a stateful agent that can accomplish multi-step tasks using at least 3 external tools, maintain conversation memory across turns, and execute safely within defined boundaries. Evaluate task completion, trace quality, and failure handling.

**Requirements:**

- at least 3 tool integrations (API, file system, database, web search, code execution, etc.)
- tool schema definitions with clear descriptions
- planning or reasoning step before each tool call
- conversation memory across turns
- step limit or budget constraint
- permission or approval gate for sensitive operations
- error handling for tool failures
- task completion detection
- trace logging for evaluation
- evaluation on a benchmark of tasks (at least 10)
- reproducible setup

**Concepts Used:**

- LLM tool use (function calling)
- planning and reasoning (ReAct, Chain-of-Thought)
- state management across steps
- conversation memory (short-term, long-term)
- error handling and retries
- safety constraints and approval gates
- trace logging and analysis

**Suggested Architecture:**

```text
User request → LLM (reasoning) → tool selection → tool execution
                                                       ↓
                                                 result → LLM (reasoning)
                                                       ↓
                                                 ... repeat until done
                                                       ↓
                                                 final answer + trace log
                                                       ↓
                                                 evaluation (task success, traces)
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| Tool definitions | Day 1–2 | At least 3 tools with schemas |
| Basic agent loop | Day 3–4 | ReAct or similar loop working |
| Memory | Day 5–6 | Multi-turn conversation support |
| Safety | Day 7–8 | Step limits, approval gates |
| Evaluation | Day 9–11 | Task benchmark, trace analysis |
| Documentation | Day 12–14 | README, eval report, safety docs |

**Expected Output:**

- working agent with tool use
- task benchmark results (success rate, step count)
- trace analysis (tool call patterns, failure modes)
- safety documentation
- README with architecture decisions

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| Tool design | Clear schemas, good descriptions, error handling | Vague tool definitions |
| Planning | Reasoning step before each action | Blind tool calls |
| Memory | Coherent multi-turn state management | State lost between turns |
| Safety | Step limits, approval gates, budget tracking | No safety constraints |
| Evaluation | Benchmark tasks, trace analysis, failure cases | Only "it worked on my example" |
| Error handling | Graceful recovery from tool failures | Crashes on tool errors |

**Common Mistakes:**

- no step limit (agent loops forever)
- no approval gate for destructive actions
- tool schemas too vague (LLM picks wrong tool)
- no error handling for API failures
- conversation state lost between turns
- no evaluation beyond a single example

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Agent loops without stopping | No step limit or completion detection | Trace logs | Add step limit and stop condition |
| Agent picks wrong tool | Vague tool descriptions | Review schemas | Improve tool descriptions |
| Agent crashes on tool error | No error handling | Test with failing tools | Add try/catch and retry logic |
| Multi-turn memory is broken | State not persisted | Test across turns | Implement proper state management |

**Possible Improvements / Advanced Extensions:**

- multi-agent collaboration
- human-in-the-loop approval workflow
- tool use with streaming results
- self-reflection and plan revision
- cost-aware planning
- evaluation with human preference ranking

**Hands-On Practice (before the capstone):**

1. Build a simple ReAct agent with 2 tools.
2. Implement conversation memory for a chatbot.
3. Add a step limit and test with a task that requires 5+ steps.
4. Trace and log all tool calls in a session.

**Knowledge Check:**

- Why is a step limit important for agent safety?
- What is the difference between ReAct and Chain-of-Thought?
- How do you evaluate whether an agent completed a task successfully?
- What is the role of tool schema quality in agent performance?
- When should you use an agent vs. a fixed workflow?

**Exit Criteria:**

- You can build a stateful, tool-using agent with safety constraints.
- You can evaluate agent performance with traces and benchmarks.
- You can identify and fix common agent failure modes.

---

### Unit 17.6 — Final AI Engineering Capstone

**What is it?**  
The capstone of capstones: a full AI/ML system that combines data processing, classical ML or deep learning, LLM integration, RAG, tool-using agents, evaluation, API serving, frontend/backend, deployment, monitoring, and security — all in one coherent project.

**Why does it matter?**  
This is the final proof that the learner can independently design and build a complex AI system. It is not about using every tool — it is about making the right choices for a real problem and delivering a complete, production-quality system.

**Why learn it here?**  
After completing all 16 prior phases, the learner has all the building blocks. This capstone forces them to integrate everything into a system that could be deployed in the real world.

**Prerequisites:** All phases (00–16).

**Mental Model:**  
This is a systems engineering project, not a model training exercise. The architecture must balance performance, cost, latency, safety, and maintainability. Every component must justify its inclusion.

**Problem Statement:**  
Design and build a complete AI/ML system that solves a real problem. The system must include at least 3 of the following: data pipeline, ML model, LLM integration, RAG, agent, tool use, evaluation, API, frontend, deployment, monitoring, security. The system must be documented, reproducible, and demonstrable.

**Requirements:**

- real problem definition with measurable success criteria
- data pipeline (collection, cleaning, validation)
- model or LLM component with architecture justification
- at least one of: RAG, agent, or tool use
- evaluation with task-appropriate metrics
- error analysis
- API or serving layer
- frontend or CLI interface
- deployment plan (Docker, cloud, or local serving)
- monitoring plan (quality, cost, latency)
- security review (secrets, permissions, input validation)
- documentation (README, architecture diagram, setup instructions)
- 5-minute teach-back presentation

**Concepts Used:**

- everything from Phases 01–16
- system design and architecture
- integration of multiple components
- production engineering
- monitoring and observability
- security and privacy
- technical communication

**Suggested Architecture:**

```text
Problem definition → constraints → component selection → architecture design
       ↓
Data pipeline → model/LLM → RAG/agent → evaluation → serving
       ↓
Frontend/CLI → deployment → monitoring → security review
       ↓
Documentation → teach-back → portfolio
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| Problem definition | Day 1 | Problem statement, success criteria |
| Architecture | Day 2–3 | Architecture diagram, component selection |
| Data + model | Day 4–7 | Working data pipeline and core model |
| Integration | Day 8–10 | RAG/agent/LLM integration |
| Evaluation | Day 11–13 | Metrics, error analysis |
| Deployment | Day 14–17 | Serving, frontend, deployment |
| Documentation | Day 18–21 | README, architecture docs, security review |
| Teach-back | Day 22 | 5-minute presentation |

**Expected Output:**

- complete, runnable system
- architecture diagram
- evaluation report with metrics
- error analysis
- deployment configuration
- monitoring plan
- security review
- README with setup instructions
- 5-minute teach-back script

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| Problem definition | Clear problem, measurable criteria, real context | Vague or no measurable goal |
| Architecture | Components justified, trade-offs documented | Components chosen randomly |
| Data pipeline | Robust, validated, reproducible | Ad-hoc data handling |
| Model/LLM | Appropriate choice with justification | Default model without reasoning |
| Integration | RAG/agent/tools work together coherently | Components loosely connected |
| Evaluation | Comprehensive metrics, error analysis, failure cases | Only demo examples |
| Deployment | Working deployment with health checks | Notebook-only |
| Monitoring | Quality, cost, latency tracking | No monitoring plan |
| Security | Secrets managed, permissions documented, input validated | Security ignored |
| Documentation | Complete, clear, reproducible | Missing or unclear |
| Communication | Teach-back explains decisions clearly | Cannot explain choices |

**Common Mistakes:**

- trying to use every tool instead of choosing the right ones
- skipping evaluation until the end
- no error analysis
- ignoring security and secrets management
- non-reproducible setup
- no monitoring plan
- cannot explain architectural decisions

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| System works locally but not in deployment | Environment differences | Check dependencies and configs | Use Docker, pin versions |
| Latency is too high | Unoptimized component | Profile each step | Cache, async, reduce model size |
| Costs are unexpected | Untracked LLM calls | Log all API calls | Add cost tracking and limits |
| Security scan fails | Secrets in code or open permissions | Review code and configs | Use env vars, restrict access |

**Possible Improvements / Advanced Extensions:**

- A/B testing framework
- automated retraining pipeline
- multi-region deployment
- cost optimization (model distillation, caching)
- user feedback loop
- compliance documentation

**Hands-On Practice (before the capstone):**

1. Build a complete project from a previous capstone and add deployment.
2. Design an architecture diagram for a system you use daily.
3. Write a security review for an existing project.
4. Practice a 5-minute technical presentation.

**Knowledge Check:**

- How do you decide which components a system needs?
- What is the trade-off between RAG and fine-tuning?
- How do you monitor a deployed AI system?
- What security risks are unique to LLM-based systems?
- How would you explain your architecture to a non-technical stakeholder?

**Exit Criteria:**

- You can independently design and build a complex AI system.
- You can justify every architectural decision.
- You can deploy, monitor, and secure a production system.
- You can communicate technical decisions clearly.

---

### Unit 17.7 — Company-Scoped RAG Support Chatbot (TaxKraft)

**What is it?**  
A production-grade, company-scoped Retrieval-Augmented Generation support chatbot for TaxKraft (a real Indian CA/tax/GST services firm). The system ingests TaxKraft's public website information, answers user questions strictly from that knowledge base, and enforces comprehensive safety guardrails — all while operating fully offline with extractive generation or optionally with an LLM backend.

**Why does it matter?**  
Real-world RAG deployments need more than just retrieval + generation. They require domain scoping (answer only from company info), PII protection, injection defense, hallucination detection, and honest evaluation — all measurable and auditable. This capstone delivers that full stack.

**Why learn it here?**  
After completing Phases 11–16, the learner understands embeddings, vector databases, retrieval, reranking, LLM generation, guardrails, evaluation, API design, and deployment. This capstone combines all of them into a single, verifiable system with a real company's data.

**Prerequisites:** Phases 01–16 (Python, data prep, embeddings, vector search, LLMs, evaluation, deployment, security).

**Mental Model:**  
A company-scoped RAG is a pipeline: ingest (with provenance) → chunk → embed → store → retrieve (hybrid) → generate (grounded) → guard (5 layers) → evaluate (retrieval + guardrails + answers). Every step must be measured: retrieval recall, guardrail confusion matrices, answer faithfulness, attack success rate.

**Problem Statement:**  
Build a complete RAG chatbot for TaxKraft that:
- Ingests only TaxKraft's verified public information (website sitemap, known facts)
- Answers questions strictly from that corpus — no general knowledge, no competitor comparisons
- Deflects off-topic, PII-containing, and adversarial queries with clear messages
- Evaluates itself rigorously (retrieval metrics, guardrail metrics, answer quality metrics)
- Runs fully offline (extractive) or with optional LLM — no mandatory API costs
- Exposes a FastAPI service and web UI for demonstration

**Requirements:**

- **Knowledge Base**: 10 curated markdown documents with provenance blocks (source URL, title, verification date, VERIFY flags for prices/timelines) covering company overview, 6 service lines, pricing/process, FAQ, contact
- **Crawler**: Sitemap fetcher + SPA-aware HTML extraction (documents limitations — taxkraft.com is a React SPA)
- **Chunking**: Section-aware markdown chunking (350 chars, 60 overlap) preserving headings
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2 (384-dim, normalized) with dummy hash embedder for tests/CI
- **Vector Store**: ChromaDB persistent, cosine similarity via L2 on normalized vectors (score = 1 - d²/2)
- **Retrieval**: Hybrid dense (0.70) + BM25 keyword (0.30) with rerank_k=8
- **Generation**: Grounded system prompt with hard rules; extractive fallback (top-k sentences from context); optional LLM via OpenAI-compatible client (Groq/OpenAI)
- **Guardrails (5 layers)**:
  1. Topic scope: keyword scoring + embedding centroid (threshold 0.42) — allows TaxKraft services, blocks competitors/general/off-topic
  2. PII: regex for Aadhaar, PAN, GSTIN, phone, email, UPI, bank, passport, DL + sensitive-intent phrases
  3. Prompt injection: signature detection (ignore instructions, system prompt reveal, role override, etc.)
  4. Retrieval confidence: minimum score threshold (0.30)
  5. Faithfulness: lexical token coverage + sentence coverage + optional embedding similarity (threshold 0.30)
- **Evaluation**: 3 suites — retrieval (Recall@1/3/5, MRR@5 on 32 golden queries keyed to KB topics), guardrails (in-scope acceptance, off-topic deflection, attack success=0 target, PII trip rate, confusion matrix), answers (answerability, faithfulness, citation coverage, latency) — auto-markdown report to `evaluation/reports/report.md`
- **API**: FastAPI with `/health`, `/guardrails/status`, `/chat`, `/eval` endpoints; CORS enabled; static web UI mounted
- **Tests**: 33 pytest tests (topic_scope, PII, injection, chunker, faithfulness, retriever, engine, API) — all passing
- **CLI**: `run.py` with `ingest`, `chat`, `eval`, `serve`, `test` commands

**Concepts Used:**

- Document ingestion with provenance tracking
- Section-aware chunking for markdown
- Embedding models and cosine similarity search
- ChromaDB persistent vector store
- BM25 keyword search (custom implementation)
- Hybrid retrieval with score fusion
- Extractive generation (faithful by construction)
- Guardrail engineering: scope, PII, injection, confidence, faithfulness
- RAG evaluation: retrieval metrics, guardrail metrics, answer quality metrics
- FastAPI service design
- Static web UI (vanilla JS)
- Pytest fixtures with dummy embedder for fast CI

**Suggested Architecture:**

```text
taxkraft.com sitemap → fetch → clean → seed KB (10 docs)
                                              ↓
                            chunk → embed → Chroma (132 chunks, cosine)
                                              ↓
User query → topic_scope → PII → injection → hybrid retrieve (dense + BM25)
                                              ↓
                                      confidence guard → extractive/LLM generate
                                              ↓
                                    faithfulness guard → response + citations
                                              ↓
                                           evaluation (3 suites) → report.md
```

**Milestones:**

| Milestone | Target | Deliverable |
|---|---|---|
| KB design & provenance | Day 1 | 10 markdown docs with VERIFY flags |
| Crawler + ingestion | Day 2 | Sitemap parser, chunker, embedder, Chroma store |
| Hybrid retrieval | Day 3 | Dense + BM25 fusion, reranking |
| Generation + guardrails | Day 4–5 | Extractive + 5-layer guard stack |
| Evaluation suite | Day 6 | 3 eval suites, 4 labeled datasets, markdown report |
| API + UI | Day 7 | FastAPI endpoints, web chat interface |
| Testing & verification | Day 8 | 33 pytest tests, ingest + eval run |

**Expected Output:**

- Complete runnable project at `projects/capstones/taxkraft-support-assistant/`
- 132-chunk vector index in `vectors/`
- Evaluation report at `evaluation/reports/report.md` (Recall@5=1.0, MRR@5≈0.9, attack_success=0.14, mean_faithfulness=1.0)
- 33 passing tests
- FastAPI service (`python run.py serve`) + web UI (`http://127.0.0.1:8000`)
- Architecture documented in `README.md`

**Evaluation Criteria:**

| Criterion | Excellent | Needs Work |
|---|---|---|
| KB provenance | Every chunk traces to source URL with VERIFY flags | No provenance or unverified claims |
| Retrieval quality | Recall@5 ≥ 0.9, MRR@5 ≥ 0.8 on golden set | No metrics or poor recall |
| Scope enforcement | Off-topic & competitor queries deflected | Answers from general knowledge |
| PII protection | Aadhaar/PAN/phone/email/UPI/bank all blocked | Any PII passes through |
| Injection defense | All 14 adversarial prompts deflected | Any injection succeeds |
| Faithfulness | Mean ≥ 0.9 on in-scope answers | Hallucinations unchecked |
| Evaluation | Auto-generated markdown with all 3 suites | Manual spot-check only |
| Engineering | Reproducible, tested, documented, CLI + API | Notebook-only, no tests |

**Common Mistakes:**

- ingesting without provenance (can't audit answers)
- using only semantic retrieval (misses exact keywords like GSTIN)
- skipping faithfulness guard (hallucinations reach user)
- no offline mode (mandatory LLM costs, no fallback)
- evaluation only on happy-path queries

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Irrelevant retrieval | Chunk size too large or embedding mismatch | Inspect top-k chunks | Reduce chunk_size, verify normalize |
| Off-topic answers accepted | Scope threshold too low or anchors incomplete | Check guardrails/anchors.py | Raise threshold, add off-topic terms |
| PII not caught | Regex gap for Indian formats | Test patterns in pii.py | Add missing patterns |
| Faithfulness false positive | Threshold too low, lexical only | Check embed similarity option | Raise threshold, enable embeddings |
| Evaluation crashes | Citation object vs dict mismatch | Check answer_eval.py | Use `.text` attribute |

**Possible Improvements / Advanced Extensions:**

- Live crawler with scheduled re-ingestion
- Cross-encoder reranker (e.g., ms-marco-MiniLM-L-6-v2)
- Multilingual support (Hindi queries → English KB)
- Streaming answers with real-time citation highlight
- Conversation memory with session-scoped retrieval
- Cost/latency dashboard in web UI
- A/B test extractive vs LLM generation quality

**Hands-On Practice (before the capstone):**

1. Build a basic RAG pipeline with FAISS and an LLM API (Unit 17.4).
2. Implement a BM25 retriever from scratch.
3. Write a topic classifier with keyword + embedding centroid.
4. Create a PII regex test suite for Indian identifiers.
5. Run a RAG evaluation with Recall@k and faithfulness on a small corpus.

**Knowledge Check:**

- Why does hybrid (dense + keyword) retrieval outperform either alone for company-scoped RAG?
- How does the faithfulness guard catch hallucinations that retrieval confidence misses?
- Why is provenance tracking critical for a company-scoped chatbot?
- What is the trade-off between extractive (offline) and LLM generation?
- How would you explain the guardrail stack to a non-technical stakeholder?

**Exit Criteria:**

- You can build a company-scoped RAG system with full guardrails and evaluation.
- You can measure and report retrieval quality, guardrail effectiveness, and answer faithfulness.
- You can operate the system fully offline with zero API costs.
- You can deploy as a FastAPI service with a working web UI.

---

## Capstone Requirements Checklist

Every substantial capstone project must contain:

- [ ] Problem definition with real-world context
- [ ] Measurable success criteria
- [ ] Requirements and constraints documented
- [ ] Data understanding (source, quality, limitations)
- [ ] Baseline implemented and compared
- [ ] Model or system selection with justification
- [ ] Experiment design documented
- [ ] Evaluation with task-appropriate metrics
- [ ] Error analysis with failure cases inspected
- [ ] Iteration documented (what was tried and why)
- [ ] Final architecture diagram
- [ ] Reproducible implementation
- [ ] Testing (unit, integration, or end-to-end)
- [ ] Documentation (README, setup, decisions)
- [ ] Limitations and future improvements
- [ ] Security and privacy review
- [ ] Deployment or serving instructions
- [ ] Monitoring and maintenance plan

---

## Required Capstone Rubric

| Area | Excellent | Satisfactory | Needs Work |
|---|---|---|---|
| Problem framing | Clear user/business problem, measurable success criteria | Problem defined but criteria vague | No clear problem or criteria |
| Data | Source documented, quality inspected, leakage audited | Data used with basic inspection | Data used blindly |
| Baseline | Simple baseline implemented and compared | Baseline mentioned but not implemented | No baseline |
| Modeling/system design | Architecture choices justified with trade-offs | Choices made but not fully justified | Choices unexplained |
| Evaluation | Metrics match real problem; failure cases tested | Basic metrics reported | Only demo examples |
| Error analysis | Weak segments and failure examples inspected | Some analysis done | No analysis |
| Engineering | Reproducible setup, tests, artifacts, docs | Mostly reproducible | Notebook-only, hard to rerun |
| Security/safety | Risks and controls fully documented | Basic security considered | Security ignored |
| Communication | README and presentation explain trade-offs clearly | Documentation exists but incomplete | Results cannot be understood |
| Deployment | Working deployment with health checks | Local serving works | No deployment path |
| Monitoring | Quality, cost, latency tracked | Some monitoring | No monitoring |

---

## Final Success Criterion

> **You should be able to leave the final capstone and independently design and build a new AI/ML system that was never explicitly taught in the curriculum.**

That is the actual measure of success.

---

## Independent Problem-Solving Test

Before calling the roadmap complete, choose a new problem not covered directly in the curriculum and answer:

1. What is the real objective?
2. What data is needed and how would you obtain it?
3. What is the simplest baseline?
4. What metric reflects success from the user's perspective?
5. What can go wrong (data, model, deployment, safety)?
6. What should be built first (MVP scope)?
7. What should be evaluated before deployment?
8. What must be monitored after deployment?
9. How would you communicate this system to a non-technical stakeholder?
10. What are the ethical considerations?

If you can answer all 10 and build a working prototype, the roadmap is complete.
