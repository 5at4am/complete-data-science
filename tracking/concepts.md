# Concept Graph

> **The knowledge network of this curriculum.**
> Every important concept, its dependencies, and its connections.

---

## How to Use

For every important concept, add an entry:

```text
### Concept Name

- **Prerequisites:** What must you know first?
- **Depends on:** What concepts build on this?
- **Used by:** What uses this concept?
- **Common confusion:** What is it often confused with?
- **Implementation:** How is it implemented?
- **Related concepts:** What is it related to?
- **Mastery level:** 0-6
- **Last reviewed:** YYYY-MM-DD
```

---

## Core Concepts

### Gradient Descent

- **Prerequisites:** Derivatives, partial derivatives
- **Depends on:** Optimization
- **Used by:** All ML/DL training
- **Common confusion:** Local vs global minimum
- **Implementation:** `θ ← θ - η·∇J(θ)`
- **Related concepts:** Learning rate, loss function, backpropagation
- **Mastery level:** —
- **Last reviewed:** —

### Overfitting

- **Prerequisites:** Model training, evaluation
- **Depends on:** Regularization
- **Used by:** Model selection, bias-variance
- **Common confusion:** Overfitting vs underfitting
- **Implementation:** Detect via train/val gap
- **Related concepts:** Regularization, cross-validation, bias-variance
- **Mastery level:** —
- **Last reviewed:** —

### Data Leakage

- **Prerequisites:** Train/test split
- **Depends on:** Proper evaluation
- **Used by:** All ML pipelines
- **Common confusion:** Leakage vs legitimate features
- **Implementation:** Split before preprocessing
- **Related concepts:** Train/test split, feature engineering
- **Mastery level:** —
- **Last reviewed:** —

### Attention

- **Prerequisites:** Sequence models, embeddings
- **Depends on:** Transformers
- **Used by:** LLMs, RAG, translation
- **Common confusion:** Self-attention vs cross-attention
- **Implementation:** `softmax(QKᵀ/√d)V`
- **Related concepts:** Transformers, embeddings, LLMs
- **Mastery level:** —
- **Last reviewed:** —

### Embeddings

- **Prerequisites:** Vectors, linear algebra
- **Depends on:** Vector search, RAG
- **Used by:** NLP, RAG, similarity
- **Common confusion:** Embeddings vs one-hot
- **Implementation:** Dense vector representation
- **Related concepts:** Vector search, RAG, transformers
- **Mastery level:** —
- **Last reviewed:** —

### RAG

- **Prerequisites:** Embeddings, vector search, LLMs
- **Depends on:** Advanced RAG, agentic RAG
- **Used by:** Production LLM systems
- **Common confusion:** RAG vs fine-tuning
- **Implementation:** Chunk → embed → retrieve → generate
- **Related concepts:** Vector databases, embeddings, LLMs
- **Mastery level:** —
- **Last reviewed:** —

---

## Concept Map (High Level)

```
Math → Statistics → ML → Deep Learning → NLP → Transformers → LLMs → RAG → Agents
  ↓        ↓          ↓        ↓            ↓        ↓           ↓      ↓       ↓
vectors  bias-var  regression  backprop   embeddings attention  tokens  vector  tools
dot prod  dists    trees       CNNs       TF-IDF    BERT        sampling search  planning
gradient  testing  boosting    RNNs       n-grams   GPT         context rerank   memory
eigen     sampling ensembling  attention  word2vec  fine-tune   prompt  hybrid   multi-agent
```

---

## Mastery Tracking

| Concept | Level | Last Reviewed |
|---------|-------|---------------|
| Gradient descent | — | — |
| Overfitting | — | — |
| Data leakage | — | — |
| Attention | — | — |
| Embeddings | — | — |
| RAG | — | — |
