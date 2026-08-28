# NLP Reference

> **A quick-reference guide to Natural Language Processing.**

---

## Text Preprocessing

### Tokenization
- Split text into tokens (words, subwords, characters)
- Word tokenization: split on whitespace/punctuation
- Subword tokenization: BPE, WordPiece, SentencePiece

### Normalization
- Lowercasing
- Removing punctuation
- Removing stop words (optional, context-dependent)
- Stemming: reduce to root (e.g., "running" → "run")
- Lemmatization: reduce to dictionary form (e.g., "better" → "good")

---

## Text Representation

### Bag of Words (BoW)
- Count of each word in a document
- Ignores order
- Sparse, high-dimensional

### TF-IDF
- Term Frequency × Inverse Document Frequency
- Weighs rare words higher
- Better than BoW for many tasks

### N-grams
- Sequences of n words
- Captures some local order

### Word Embeddings
- Dense vector representation of words
- Similar words have similar vectors
- Word2Vec, GloVe, FastText

### Contextual Embeddings
- Embeddings depend on context
- BERT, RoBERTa, etc.
- Same word can have different vectors in different contexts

---

## NLP Tasks

### Text Classification
- Spam detection, sentiment analysis, topic classification

### Named Entity Recognition (NER)
- Identify entities: people, places, organizations, dates

### Part-of-Speech Tagging
- Label each word with its grammatical role

### Machine Translation
- Translate between languages

### Text Summarization
- Abstractive (generate new text) vs extractive (select sentences)

### Question Answering
- Answer questions from a context

### Text Generation
- Generate coherent text

---

## Sequence Models

### RNN
- Process sequences step by step
- Hidden state carries context
- Vanishing gradient problem

### LSTM
- Long Short-Term Memory
- Gates control information flow
- Handles long-range dependencies

### GRU
- Simplified LSTM
- Fewer parameters

### Attention
- Weigh importance of different input parts
- Solves the bottleneck of fixed-size context

---

## Transformers

### Architecture
- Self-attention + feed-forward layers
- Positional encoding
- Encoder-decoder or encoder-only or decoder-only

### Self-Attention
- Query, Key, Value
- `Attention(Q,K,V) = softmax(QKᵀ/√d)V`
- Each token attends to all others

### BERT-style (Encoder-only)
- Bidirectional context
- Great for understanding tasks (classification, NER, QA)

### GPT-style (Decoder-only)
- Causal (left-to-right)
- Great for generation

---

## Hugging Face Ecosystem

### Transformers Library
```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Pipeline (easy)
classifier = pipeline("sentiment-analysis")
result = classifier("I love this!")

# Explicit (more control)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
```

### Datasets Library
```python
from datasets import load_dataset
dataset = load_dataset("imdb")
```

### Tokenizers Library
- Fast, efficient tokenization
- BPE, WordPiece, SentencePiece

---

## Evaluation Metrics

### Classification
- Accuracy, precision, recall, F1

### Generation
- BLEU: n-gram overlap with reference
- ROUGE: recall-oriented overlap
- Perplexity: how surprised the model is

### Embedding Quality
- Similarity to human judgments
- Downstream task performance

---

## NLP Pipeline

1. Text collection
2. Preprocessing (tokenization, normalization)
3. Representation (BoW, TF-IDF, embeddings)
4. Model (classical ML or deep learning)
5. Evaluation
6. Deployment

---

## When to Use What

| Approach | When to Use | When NOT to Use |
|----------|-------------|-----------------|
| BoW/TF-IDF | Simple, small data, baseline | Complex semantics, large data |
| Word2Vec | Static word similarity | Context-dependent meaning |
| BERT-style | Understanding tasks, classification | Generation |
| GPT-style | Generation | Understanding-only tasks |
| RNN/LSTM | Short sequences, low compute | Long sequences, parallelization |
| Transformer | Most modern NLP | Very small data |
