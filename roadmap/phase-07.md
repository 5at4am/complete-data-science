# Phase 07 — NLP

> **Goal:** Master natural language processing — from text preprocessing to embeddings, sequence models, and attention — so you can build systems that understand and generate human language.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced
**Priority:** Essential
**Prerequisites:** Phase 05 (Machine Learning), Phase 06 (Deep Learning)
**Mastery target:** Level 5 — decision making for NLP pipelines, embeddings, sequence models, and evaluation

---

## Why This Phase Exists

Text is the most common unstructured data type in the real world. Every chatbot, search engine, spam filter, translation system, sentiment analyzer, and LLM starts with NLP fundamentals. This phase bridges classical ML and deep learning by teaching you how to turn raw text into numbers, build models that understand sequences, and evaluate language tasks rigorously.

### Phase Mental Model

Text must be converted to numbers before any model can use it. The journey is:

```text
Raw text → cleaning & tokenization → numerical representation → model → evaluation
    ↓              ↓                        ↓                ↓          ↓
  words       BoW / TF-IDF              embeddings      RNN/LSTM    F1/BLEU
              (sparse, count-based)     (dense, meaning-based)    (attention)
```

Classical methods (BoW, TF-IDF) are fast and interpretable. Embeddings capture meaning. Sequence models capture order. Attention captures relevance across long distances. Each step builds on the last.

### What This Phase Prepares For

- Transformer architectures in Phase 08
- LLMs and prompt engineering in Phase 09–10
- RAG systems in Phase 11
- Text generation, summarization, and translation tasks
- Any application involving human language data

---

## Units

### Unit 07.1 — Text Preprocessing & Tokenization

**What is it?**
The process of cleaning raw text and splitting it into manageable units (tokens) — words, subwords, or characters — that can be converted to numerical features.

**Why does it matter?**
Raw text is noisy: inconsistent casing, punctuation, special characters, HTML tags, stopwords, and typos. Without cleaning, downstream models learn noise instead of signal. Tokenization determines the granularity of your vocabulary and directly affects model performance.

**Why learn it here?**
NLP begins with text. Before you can build any representation or model, you must know how to prepare text. This unit is the foundation every other NLP unit depends on.

**Prerequisites:** Phase 05 (ML basics), Phase 06 (PyTorch/TensorFlow basics for later units).

**Mental Model:**
Preprocessing is like cleaning a room before organizing it. You remove trash (HTML, stopwords), standardize items (lowercasing, normalization), then decide on storage containers (token granularity).

**Core Concepts:**

- **Tokenization:** splitting text into tokens (words, subwords, characters)
- **Lowercasing:** normalizing case to reduce vocabulary size
- **Punctuation removal:** stripping or preserving punctuation intentionally
- **Stopword removal:** removing common low-information words
- **Stemming:** crude suffix removal (Porter Stemmer, Snowball)
- **Lemmatization:** dictionary-based normalization (run → ran → run)
- **Subword tokenization:** BPE, WordPiece, SentencePiece (used by transformers)
- **Regex-based tokenization:** custom patterns for specialized text

**How It Works:**

1. Raw text arrives (from files, APIs, databases).
2. Apply cleaning: remove HTML tags, normalize unicode, fix encoding issues.
3. Tokenize: split into words or subwords.
4. Optional normalization: lowercase, stem, lemmatize.
5. Output: a list of clean tokens ready for numerical encoding.

**Syntax & Implementation:**

```python
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

text = "The cats were running <b>quickly</b> toward the mice!"

# 1. Remove HTML tags
clean = re.sub(r"<[^>]+>", "", text)

# 2. Lowercase
clean = clean.lower()

# 3. Tokenize
tokens = word_tokenize(clean)
# ['the', 'cats', 'were', 'running', 'quickly', 'toward', 'the', 'mice', '!']

# 4. Remove stopwords
stop_words = set(stopwords.words("english"))
filtered = [t for t in tokens if t not in stop_words and t.isalpha()]
# ['cats', 'running', 'quickly', 'toward', 'mice']

# 5. Stem
stemmer = PorterStemmer()
stemmed = [stemmer.stem(t) for t in filtered]
# ['cat', 'run', 'quick', 'toward', 'mice']

# 6. Lemmatize
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(t) for t in filtered]
# ['cat', 'running', 'quickly', 'toward', 'mouse']
```

**Simple Example:**

```python
def preprocess(text, lowercase=True, remove_stops=True):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    if lowercase:
        text = text.lower()
    tokens = text.split()
    if remove_stops:
        tokens = [t for t in tokens if t not in stopwords.words("english")]
    return tokens

print(preprocess("I love machine learning! It's amazing."))
# ['love', 'machine', 'learning', 'amazing']
```

**Real-World Example:**
Search engines preprocess queries before matching against indexed documents. E-commerce platforms normalize product descriptions for deduplication. Customer support systems tokenize incoming tickets for classification.

**Common Mistakes:**

- Aggressive stopword removal that removes negations ("not", "no") — breaks sentiment analysis
- Stemming when lemmatization is needed — "studies" → "studi" vs "study"
- Lowercasing proper nouns — "Apple" (company) vs "apple" (fruit)
- Not handling emojis/emoticons — loss of sentiment signal
- Tokenizing before cleaning — HTML tags become tokens

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Unknown words in vocabulary | Tokens not normalized the same way at train/test | Compare token examples | Apply identical preprocessing pipeline to both |
| Sentiment classifier fails on negations | Stopword removal killed "not" | Check filtered tokens | Keep negation words in stopword filter |
| Stemmer produces nonsense | Stemmer applied to already-stemmed text | Inspect intermediate output | Apply stemming once, at the end |
| Unicode errors | Mixed encoding in source text | `repr()` the raw text | Normalize with `unicodedata.normalize()` |
| Large vocabulary | No subword tokenization | Count unique tokens | Use BPE/WordPiece or limit vocabulary size |

**Alternatives:**

| Tool | Use When | Avoid When |
|---|---|---|
| NLTK | Learning, prototyping, classic NLP | Production-scale or speed-critical |
| spaCy | Production pipelines, large texts | You need fine-grained control over every step |
| Hugging Face Tokenizers | Transformer-based models, subword tokenization | Classical ML with simple word-level features |
| Regex alone | Custom tokenization rules | Natural language with complex morphology |

**Best Practices:**

- Always save your preprocessing pipeline and apply it identically at train and test time
- Keep negation words ("not", "no", "never") unless you have a specific reason
- Prefer lemmatization over stemming for interpretability
- Use subword tokenization for neural models to handle out-of-vocabulary words
- Inspect your tokens regularly — don't assume preprocessing is correct

**Hands-On Practice:**

1. **Basic:** Tokenize a sentence using NLTK and spaCy. Compare outputs.
2. **Guided:** Build a preprocessing pipeline that removes HTML, lowercases, tokenizes, removes stopwords, and lemmatizes.
3. **Independent:** Preprocess a dataset of product reviews. Count vocabulary before and after each step.
4. **Realistic:** Handle a dataset with mixed languages, emojis, and HTML tags. Decide what to keep and what to remove.
5. **Challenge:** Compare classification accuracy with and without each preprocessing step. Which steps help and which hurt?

**Exit Criteria:**

- You can tokenize text at word and subword levels
- You can choose between stemming and lemmatization with justification
- You can build a reproducible preprocessing pipeline

**Next Step:** Convert tokens into numerical features with Bag of Words and TF-IDF.

---

### Unit 07.2 — Bag of Words & TF-IDF

**What is it?**
Methods for converting a collection of text documents into numerical feature vectors. Bag of Words (BoW) counts word occurrences. TF-IDF weights words by their importance within a document relative to the corpus.

**Why does it matter?**
Machine learning models cannot operate on raw text — they need numbers. BoW and TF-IDF are the simplest, fastest, and most interpretable text-to-number representations. They remain strong baselines for many NLP tasks.

**Why learn it here?**
After preprocessing, the next step is numerical representation. BoW and TF-IDF are foundational: many modern techniques (embeddings, transformers) were developed to overcome their limitations. Understanding what they lack motivates everything that follows.

**Prerequisites:** Unit 07.1 (text preprocessing), Phase 05 (ML basics).

**Mental Model:**
BoW is a histogram of words across a document. TF-IDF is a BoW where rare, informative words get higher scores and common uninformative words get lower scores.

```text
Document = "the cat sat on the mat"
BoW      = {the:2, cat:1, sat:1, on:1, mat:1}
TF-IDF   = {the:0.1, cat:1.8, sat:2.0, on:0.3, mat:1.9}
                      ↑ rare    ↑ rare   ↑ common  ↑ rare
```

**Core Concepts:**

- **Vocabulary:** the set of all unique tokens across the corpus
- **Document-term matrix:** rows = documents, columns = vocabulary terms, values = counts or weights
- **Term Frequency (TF):** how often a word appears in one document
- **Inverse Document Frequency (IDF):** how rare a word is across the corpus
- **TF-IDF score:** TF × IDF — high when a word appears often in one document but rarely overall
- **Sparse representation:** most values are zero, stored efficiently with sparse matrices
- **Max features / min document frequency:** hyperparameters to control vocabulary size

**How It Works:**

1. Build vocabulary from all documents.
2. For each document, count occurrences of each vocabulary word (BoW).
3. Or compute TF-IDF: TF(t,d) = count(t in d) / len(d); IDF(t) = log(N / df(t)); TF-IDF = TF × IDF.
4. Output: a matrix where each row is a numerical vector representing one document.

**Syntax & Implementation:**

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = [
    "the cat sat on the mat",
    "the dog sat on the log",
    "cats and dogs are friends"
]

# Bag of Words
bow = CountVectorizer()
bow_matrix = bow.fit_transform(docs)
print(bow.get_feature_names_out())
# ['and' 'are' 'cat' 'cats' 'dog' 'dogs' 'friends' 'log' 'mat' 'on' 'sat' 'the']
print(bow_matrix.toarray())
# [[0 0 1 0 0 0 0 0 1 1 1 2]
#  [0 0 0 0 1 0 0 1 0 1 1 2]
#  [1 1 0 1 0 1 1 0 0 0 0 0]]

# TF-IDF
tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(docs)
print(tfidf_matrix.toarray().round(2))
```

**Simple Example:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "I love Python programming",
    "Python is great for data science",
    "data science uses Python"
]

vectorizer = TfidfVectorizer(stop_words="english")
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray().round(3))
```

**Real-World Example:**
Search engines use TF-IDF (or BM25, a TF-IDF variant) to rank documents by relevance to a query. Spam filters use BoW features to classify emails. Legal tech systems use TF-IDF to find relevant case law.

**Common Mistakes:**

- Fitting the vectorizer on the entire dataset including test data (data leakage)
- Ignoring vocabulary size — huge vocabularies cause memory issues and overfitting
- Using BoW for tasks requiring word order (it loses all sequence information)
- Not tuning `max_features`, `min_df`, `max_df` — default settings may include too many rare or too common terms

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Memory error on large corpus | Vocabulary too large | Check `len(vectorizer.vocabulary_)` | Set `max_features` or `min_df` |
| Test matrix has different shape than train | Vocabulary mismatch | Compare `get_feature_names_out()` | Use `transform()` not `fit_transform()` on test |
| Features are all zeros for a document | No vocabulary words matched | Inspect raw document | Check preprocessing consistency |
| IDF values are all 1.0 | Every word appears in every document | Check document count per word | Increase corpus diversity or adjust `min_df` |

**Alternatives:**

| Method | Use When | Avoid When |
|---|---|---|
| Bag of Words | Simple baseline, interpretable features | Word order matters, large vocabulary |
| TF-IDF | Weighted features, search, classification | You need semantic similarity or word meaning |
| BM25 | Information retrieval / search ranking | You need dense vectors for neural models |
| HashingVectorizer | Very large datasets, streaming | You need to inspect the vocabulary |

**Best Practices:**

- Always fit the vectorizer on training data only, then transform test data
- Start with `TfidfVectorizer` over `CountVectorizer` — it usually performs better
- Tune `max_features` (5000–50000), `min_df` (2–5), `max_df` (0.8–0.95) as hyperparameters
- Use the `analyzer` parameter for character n-grams or custom tokenization
- Save the fitted vectorizer alongside your model for consistent inference

**Hands-On Practice:**

1. **Basic:** Build a BoW matrix from 5 sentences. Inspect the vocabulary and feature values.
2. **Guided:** Compare BoW and TF-IDF on the same corpus. Which words get higher TF-IDF scores?
3. **Independent:** Build a TF-IDF + Logistic Regression pipeline for a spam dataset. Tune `max_features`.
4. **Realistic:** Handle a text classification problem with 10,000+ documents. Choose vectorization parameters and justify them.
5. **Challenge:** Compare character-level vs word-level TF-IDF for a task with many typos.

**Exit Criteria:**

- You can build BoW and TF-IDF representations from text
- You can explain TF-IDF and why it improves over raw counts
- You can tune vectorization hyperparameters and avoid data leakage

**Next Step:** Extend word-level features to capture local word order with N-grams.

---

### Unit 07.3 — N-grams

**What is it?**
N-grams are contiguous sequences of N tokens extracted from text. They extend Bag of Words by capturing local word order and multi-word phrases.

**Why does it matter?**
"not good" has opposite meaning from "good". A unigram model sees two separate words. A bigram model sees one meaningful phrase. N-grams let classical models capture simple context without neural networks.

**Why learn it here?**
After BoW/TF-IDF, the natural extension is to capture local context. N-grams are a simple, effective technique that bridges the gap between word-level features and sequence models.

**Prerequisites:** Unit 07.2 (BoW/TF-IDF).

**Mental Model:**
A sliding window of size N over a tokenized document. Each window position produces one N-gram.

```text
Text: "the cat sat on the mat"
Bigrams: ["the cat", "cat sat", "sat on", "on the", "the mat"]
Trigrams: ["the cat sat", "cat sat on", "sat on the", "on the mat"]
```

**Core Concepts:**

- **Unigram (N=1):** single words — same as BoW
- **Bigram (N=2):** pairs of adjacent words — captures phrases
- **Trigram (N=3):** triples — captures longer phrases
- **N-gram feature space:** the number of possible N-grams grows exponentially with N
- **N-gram TF-IDF:** applying TF-IDF weighting to N-gram features
- **Min/max N:** using a range of N-gram sizes simultaneously (e.g., unigrams + bigrams)

**How It Works:**

1. Tokenize text into words.
2. For each position in the token list, extract a window of N consecutive tokens.
3. Join tokens in each window to form N-gram strings.
4. Build a vocabulary of all unique N-grams.
5. Apply BoW or TF-IDF counting over the N-gram vocabulary.

**Syntax & Implementation:**

```python
from sklearn.feature_extraction.text import TfidfVectorizer

docs = [
    "I do not like spam",
    "I like good food",
    "spam is not good"
]

# Unigrams + Bigrams
vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
X = vectorizer.fit_transform(docs)

print(vectorizer.get_feature_names_out())
# Includes both single words and two-word phrases
print(X.toarray().round(3))
```

**Simple Example:**

```python
from sklearn.feature_extraction.text import CountVectorizer

docs = ["I love machine learning", "machine learning is fun"]

# Bigrams only
bigram_vec = CountVectorizer(ngram_range=(2, 2))
X = bigram_vec.fit_transform(docs)
print(bigram_vec.get_feature_names_out())
# ['is fun' 'learning is' 'love machine' 'machine learning']
```

**Real-World Example:**
Spam filters use bigrams like "free money" and "click here" as strong spam indicators. Sentiment analysis uses negation bigrams like "not good" and "never again". Autocomplete systems use N-grams to predict the next word.

**Common Mistakes:**

- Using N > 3 — vocabulary explodes and features become too sparse
- Not including unigrams when using bigrams — losing individual word signal
- Applying N-grams before proper tokenization — punctuation becomes part of phrases
- Ignoring memory usage — bigram/trigram matrices are much larger

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Vocabulary is huge (millions) | N too large or no `max_features` | Check `len(vectorizer.vocabulary_)` | Limit N to 2–3, set `max_features` |
| Model overfits with N-grams | Too many sparse features | Compare train/test performance | Reduce N-gram range or increase `min_df` |
| Key phrases missing | `ngram_range` too narrow | Check ngram_range setting | Use `(1, 2)` or `(1, 3)` |
| Memory error | Large N-gram matrix in dense format | Check matrix shape and type | Use sparse matrices throughout |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Unigrams only | Simple tasks, small datasets | Context and phrases matter |
| Bigrams / trigrams | Negation, phrases, short texts | Very large vocabularies, long documents |
| Subword tokenization | Neural models, handling OOV | Classical ML with BoW/TF-IDF |
| Word embeddings | Semantic similarity, analogy tasks | You need interpretable features |

**Best Practices:**

- Start with `ngram_range=(1, 2)` — unigrams + bigrams — as a strong default
- Keep N ≤ 3 for classical ML; higher N is rarely useful
- Always combine N-grams with `max_features` or `min_df` to control vocabulary
- Use sparse matrices — never convert to dense for large vocabularies
- Inspect the top N-gram features to understand what the model is learning

**Hands-On Practice:**

1. **Basic:** Extract bigrams and trigrams from a sentence manually and with sklearn.
2. **Guided:** Compare unigram-only vs unigram+bigram TF-IDF on a classification task.
3. **Independent:** Build a text classifier and tune the N-gram range as a hyperparameter.
4. **Realistic:** Analyze which N-grams are most predictive for a sentiment analysis dataset.
5. **Challenge:** Compare character-level N-grams (2–5) vs word-level N-grams for misspelled text classification.

**Exit Criteria:**

- You can explain what N-grams capture that unigrams miss
- You can choose appropriate N-gram ranges and tune vocabulary size
- You can combine N-grams with TF-IDF effectively

**Next Step:** Use these features to build a complete text classification system.

---

### Unit 07.4 — Text Classification

**What is it?**
The task of assigning a label (category, sentiment, topic) to a piece of text. This unit applies ML classifiers to text features built from previous units.

**Why does it matter?**
Text classification is one of the most common NLP applications: spam detection, sentiment analysis, topic labeling, intent detection, toxicity filtering, and document categorization.

**Why learn it here?**
You now have text preprocessing, numerical features (BoW/TF-IDF/N-grams). This unit brings them together into a complete ML pipeline for text.

**Prerequisites:** Unit 07.1–07.3, Phase 05 (ML classifiers).

**Mental Model:**
Text classification = preprocessing → feature extraction → classifier → prediction. The same pipeline you learned for tabular ML, but with text as input.

```text
Raw text → preprocess → vectorize (TF-IDF) → classifier → label
```

**Core Concepts:**

- **Pipeline construction:** chaining preprocessing, vectorization, and classification
- **Classical classifiers for text:** Naive Bayes, Logistic Regression, SVM, Random Forest
- **Naive Bayes:** the classic text classifier — fast, works well with TF-IDF
- **Logistic Regression:** strong baseline, interpretable coefficients
- **SVM:** effective in high-dimensional sparse spaces
- **Train/test splitting for text:** stratified splits, maintaining label distribution
- **Multi-class vs multi-label:** one label per document vs multiple labels

**How It Works:**

1. Split data into train/test sets (stratified by label).
2. Build preprocessing + vectorization pipeline on training data only.
3. Fit a classifier on the resulting features.
4. Predict on test data and evaluate.

**Syntax & Implementation:**

```python
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Sample data
texts = ["I love this product", "Terrible experience", "Great value", "Waste of money",
         "Highly recommend", "Very disappointing", "Amazing quality", "Not worth it"]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=positive, 0=negative

X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.25, random_state=42, stratify=labels
)

# Pipeline: vectorize + classify
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
    ("clf", LogisticRegression(max_iter=1000))
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
print(classification_report(y_test, predictions))
```

**Simple Example:**

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

# Minimal spam classifier
train_texts = ["win free money now", "meeting at 3pm", "click here for prize", "project update attached"]
train_labels = [1, 0, 1, 0]

model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("nb", MultinomialNB())
])
model.fit(train_texts, train_labels)
print(model.predict(["free prize waiting for you"]))  # [1]
```

**Real-World Example:**
Gmail uses text classification for spam filtering. Twitter/X classifies tweets for content moderation. Customer support systems route tickets by intent. News organizations auto-tag articles by topic.

**Common Mistakes:**

- Fitting the vectorizer on the entire dataset before splitting (data leakage)
- Using accuracy alone on imbalanced datasets
- Not trying a simple baseline before complex models
- Ignoring class imbalance — some categories dominate

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model always predicts majority class | Class imbalance | Check label distribution | Use `class_weight="balanced"` or resample |
| Perfect training accuracy, poor test | Overfitting | Compare train/test scores | Reduce features, increase regularization |
| Poor performance on short texts | Few features per document | Inspect document lengths | Use character n-grams, increase max_features |
| Classifier warning about convergence | Not enough iterations | Check max_iter | Increase max_iter or scale features |

**Alternatives:**

| Classifier | Use When | Avoid When |
|---|---|---|
| Naive Bayes | Fast baseline, small datasets, real-time | Complex feature interactions needed |
| Logistic Regression | Strong default, interpretable, well-understood | Highly non-linear decision boundaries |
| SVM | High-dimensional sparse data, good accuracy | Large datasets, slow training |
| Random Forest | Non-linear, robust to noise | Sparse high-dimensional text features |
| Neural network | Large datasets, complex patterns | Small data, need interpretability |

**Best Practices:**

- Always build a Pipeline to avoid data leakage
- Start with Logistic Regression or Naive Bayes as baselines
- Use `classification_report` — precision, recall, and F1 tell more than accuracy
- Handle class imbalance with `class_weight="balanced"` or SMOTE
- Tune the vectorizer and classifier as a joint pipeline

**Hands-On Practice:**

1. **Basic:** Build a Naive Bayes spam classifier with the SMS Spam dataset.
2. **Guided:** Compare Naive Bayes, Logistic Regression, and SVM on the same dataset.
3. **Independent:** Build a multi-class topic classifier with at least 5 categories.
4. **Realistic:** Handle a dataset with class imbalance, noisy labels, and varying document lengths.
5. **Challenge:** Build an ensemble of vectorizer + classifier combinations and compare to a single pipeline.

**Exit Criteria:**

- You can build a complete text classification pipeline from raw text to predictions
- You can choose appropriate classifiers and evaluate them properly
- You can handle class imbalance and data leakage

**Next Step:** Move beyond count-based features to dense semantic representations with word embeddings.

---

### Unit 07.5 — Word Embeddings (Word2Vec Concepts)

**What is it?**
Word embeddings are dense, low-dimensional vector representations of words where similar words are close in vector space. Word2Vec is the foundational method that learns these representations from text context.

**Why does it matter?**
BoW and TF-IDF treat every word as an independent symbol. Embeddings capture meaning: "king" − "man" + "woman" ≈ "queen". This semantic understanding enables better performance on many NLP tasks.

**Why learn it here?**
After understanding sparse features and their limitations (no word similarity, no semantic meaning), embeddings are the natural next step. They are the foundation for all modern NLP.

**Prerequisites:** Unit 07.1–07.4, Phase 06 (neural network basics).

**Mental Model:**
Imagine every word as a point in a high-dimensional space. Words used in similar contexts end up close together. The "dimensions" don't have human-readable names, but they capture properties like gender, plurality, tense, formality, and topic.

```text
Sparse features:  cat = [0,0,0,1,0,0,0,0,0,0]  (one-hot, no meaning)
Embeddings:       cat = [0.2, -0.5, 0.8, 0.1]  (dense, carries meaning)
```

**Core Concepts:**

- **Dense vectors:** low-dimensional (50–300) continuous representations
- **Distributional hypothesis:** "you shall know a word by the company it keeps"
- **Word2Vec architectures:** Skip-gram (predict context from word) and CBOW (predict word from context)
- **Training objective:** learn vectors so that words appearing in similar contexts have similar vectors
- **Cosine similarity:** measuring distance between embedding vectors
- **Pre-trained embeddings:** Word2Vec, GloVe, FastText trained on large corpora
- **Fine-tuning embeddings:** updating pre-trained vectors for your specific task

**How It Works:**

1. Build a vocabulary from the corpus.
2. For each word, create a context window (e.g., ±5 words).
3. Train a shallow neural network to predict either the context words (Skip-gram) or the target word (CBOW).
4. The learned hidden layer weights become the word embeddings.
5. Use pre-trained vectors (trained on billions of words) or train from scratch.

**Syntax & Implementation:**

```python
from gensim.models import Word2Vec
import numpy as np

# Training data: list of tokenized sentences
sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "sat", "on", "the", "log"],
    ["cats", "and", "dogs", "are", "friends"],
]

# Train Word2Vec
model = Word2Vec(
    sentences,
    vector_size=100,   # embedding dimension
    window=5,          # context window size
    min_count=1,       # ignore words with frequency < 1
    sg=1,              # 1=Skip-gram, 0=CBOW
    epochs=10
)

# Get embedding for a word
print(model.wv["cat"].shape)  # (100,)

# Find similar words
print(model.wv.most_similar("cat", topn=3))

# Cosine similarity between two words
print(model.wv.similarity("cat", "dog"))
```

**Simple Example:**

```python
from gensim.models import Word2Vec

sentences = [
    ["I", "love", "deep", "learning"],
    ["I", "love", "machine", "learning"],
    ["deep", "learning", "is", "powerful"],
    ["machine", "learning", "is", "useful"]
]

model = Word2Vec(sentences, vector_size=50, window=3, min_count=1, sg=1, epochs=20)

# Words used in similar contexts should be close
print(model.wv.similarity("deep", "machine"))  # Should be moderate
print(model.wv.similarity("love", "is"))       # Should be lower
```

**Real-World Example:**
Google News Word2Vec (trained on 100B words) famously captures analogies: king − man + woman = queen. Embeddings power search engines, recommendation systems, and as input features for downstream NLP models.

**Common Mistakes:**

- Using very small corpora — embeddings need millions of words to learn good representations
- Setting `vector_size` too high for small data — causes overfitting
- Ignoring preprocessing — embeddings learn from context, so noise hurts
- Confusing similarity with relatedness — "cat" and "dog" are similar, but "cat" and "feline" are related
- Not using pre-trained embeddings when your corpus is small

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Similar words seem random | Training data too small | Check corpus size | Use pre-trained embeddings or larger corpus |
| All similarities are ~1.0 | Vectors not normalized or degenerate | Check vector norms | Normalize or retrain with better parameters |
| OOV words return zero | Word not in vocabulary | Check `word in model.wv` | Use subword embeddings (FastText) or average known tokens |
| Embeddings don't capture meaning | Context window too small or too large | Tune window parameter | Try window=5 for general, window=2 for syntax |

**Alternatives:**

| Method | Use When | Avoid When |
|---|---|---|
| Word2Vec | General-purpose embeddings, fast training | Need subword information |
| GloVe | Global corpus statistics matter | Need to train on your own domain |
| FastText | OOV words, morphologically rich languages | Memory is limited |
| Pre-trained (GloVe/Word2Vec) | Small corpus, general domain | Domain-specific terminology |

**Best Practices:**

- Use pre-trained embeddings unless your corpus has millions of words
- Choose `vector_size` based on vocabulary size: 100–300 for most tasks
- Set `window=5` as a starting point; smaller for syntax, larger for semantics
- Always evaluate embeddings qualitatively (check similar words) and quantitatively (downstream task)
- Consider FastText if OOV words are a problem

**Hands-On Practice:**

1. **Basic:** Train Word2Vec on a small corpus and inspect similar words.
2. **Guided:** Load pre-trained GloVe vectors and compute word analogies.
3. **Independent:** Train Word2Vec on a domain-specific corpus (e.g., medical texts). Compare to general embeddings.
4. **Realistic:** Use word embeddings as features for a text classification task. Compare to TF-IDF.
5. **Challenge:** Implement a simple word analogy solver: king − man + woman = ?.

**Exit Criteria:**

- You can train Word2Vec and explain Skip-gram vs CBOW
- You can use pre-trained embeddings and compute cosine similarity
- You can evaluate embedding quality both qualitatively and quantitatively

**Next Step:** Learn sequence models (RNN/LSTM) that process word embeddings in order.

---

### Unit 07.6 — Sequence Models for NLP

**What is it?**
Recurrent Neural Networks (RNNs) and Long Short-Term Memory (LSTM) networks are neural architectures designed to process sequences of data — like sentences — by maintaining a hidden state that carries information from previous tokens.

**Why does it matter?**
Word order matters. "The cat chased the dog" and "The dog chased the cat" have the same words but different meanings. BoW and TF-IDF lose word order entirely. Sequence models preserve it, enabling the model to understand how words relate across a sentence.

**Why learn it here?**
After embeddings (which represent individual words), sequence models are the next step: they process word embeddings in order to understand context, grammar, and long-range dependencies. This is the bridge between classical NLP and transformers.

**Prerequisites:** Unit 07.5 (word embeddings), Phase 06 (neural networks, PyTorch/TF basics).

**Mental Model:**
An RNN reads a sentence one word at a time, maintaining a "memory" (hidden state) that summarizes everything it has seen so far. An LSTM enhances this memory with gates that decide what to keep, what to forget, and what to add.

```text
Input:    The   cat   sat   on   the   mat
Hidden:   h_0 → h_1 → h_2 → h_3 → h_4 → h_5 → h_6
                       (carries context forward)
```

**Core Concepts:**

- **Recurrent structure:** the same weights process each time step
- **Hidden state:** a vector that summarizes the sequence seen so far
- **Vanishing gradients:** RNNs struggle to learn long-range dependencies because gradients shrink exponentially
- **LSTM gates:** forget gate (what to drop), input gate (what to store), output gate (what to pass on)
- **GRU (Gated Recurrent Unit):** simplified LSTM with fewer parameters
- **Bidirectional RNN:** reads the sequence forward and backward for richer context
- **Sequence-to-sequence:** encoder reads input, decoder generates output (translation, summarization)
- **Many-to-one:** classify a sequence (sentiment analysis)
- **One-to-many:** generate a sequence from one input (captioning)

**How It Works:**

1. Convert each word to its embedding vector.
2. Feed embeddings sequentially into the RNN/LSTM.
3. At each step, update the hidden state based on current input and previous state.
4. Use the final hidden state (or all hidden states) for classification or generation.
5. Train with backpropagation through time (BPTT).

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # *2 for bidirectional

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)            # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)  # output: (batch, seq_len, hidden*2)

        # Concatenate final forward and backward hidden states
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden*2)
        return self.fc(hidden_cat)               # (batch, output_dim)

# Example usage
model = SentimentLSTM(vocab_size=10000, embed_dim=128, hidden_dim=256, output_dim=2)
dummy_input = torch.randint(0, 10000, (32, 50))  # batch=32, seq_len=50
print(model(dummy_input).shape)  # torch.Size([32, 2])
```

**Simple Example:**

```python
import torch
import torch.nn as nn

# Simple RNN for sequence classification
rnn = nn.RNN(input_size=10, hidden_size=20, num_layers=2, batch_first=True, bidirectional=True)

x = torch.randn(5, 15, 10)  # batch=5, seq_len=15, features=10
output, hidden = rnn(x)
print(output.shape)   # torch.Size([5, 15, 40])  # 40 = 20*2 (bidirectional)
print(hidden.shape)   # torch.Size([4, 5, 20])   # 4 = num_layers*2
```

**Real-World Example:**
Machine translation (encoder-decoder RNNs), speech recognition, music generation, named entity recognition, and sentiment analysis all use sequence models. Before transformers, LSTMs were the dominant architecture for most NLP tasks.

**Common Mistakes:**

- Using plain RNNs for long sequences — vanishing gradients kill long-range learning
- Not using bidirectional models when full context is available
- Feeding padded sequences without masking — padding tokens corrupt the hidden state
- Using too many LSTM layers — diminishing returns and harder training
- Not clipping gradients — RNNs are prone to exploding gradients

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Training loss doesn't decrease | Learning rate too high or vanishing gradients | Check gradient norms | Lower LR, use LSTM/GRU, add gradient clipping |
| Model ignores long-range context | Vanishing gradients in plain RNN | Compare to LSTM baseline | Switch to LSTM or GRU |
| Out of memory on long sequences | Hidden state storing full sequence | Check sequence lengths | Truncate, use gradient checkpointing |
| Poor performance on short texts | Hidden state underutilized | Analyze by text length | Use attention or CNN-based models instead |
| Slow training | Sequential processing can't parallelize | Profile training loop | Use teacher forcing, reduce sequence length |

**Alternatives:**

| Model | Use When | Avoid When |
|---|---|---|
| RNN | Simple tasks, short sequences | Long-range dependencies needed |
| LSTM/GRU | Most sequence tasks, moderate sequences | Very long sequences (use Transformer) |
| CNN for text | Fast inference, local patterns (n-grams) | Long-range dependencies matter |
| Transformer | Long sequences, parallel training needed | Small datasets, limited compute |

**Best Practices:**

- Always use LSTM or GRU over plain RNN — the improvement is almost free
- Use bidirectional models when you have the full input sequence (classification)
- Unidirectional models for autoregressive generation (you can't see the future)
- Apply dropout between LSTM layers to prevent overfitting
- Use gradient clipping (`torch.nn.utils.clip_grad_norm_`) to prevent exploding gradients

**Hands-On Practice:**

1. **Basic:** Build a simple RNN and process a batch of sequences. Inspect output shapes.
2. **Guided:** Implement a bidirectional LSTM sentiment classifier on IMDB data.
3. **Independent:** Build a many-to-many sequence model for part-of-speech tagging.
4. **Realistic:** Train an encoder-decoder model for a simple translation task.
5. **Challenge:** Compare RNN, LSTM, and GRU on the same classification task. Report accuracy and training time.

**Exit Criteria:**

- You can implement an LSTM classifier in PyTorch
- You can explain bidirectional vs unidirectional processing
- You can debug vanishing/exploding gradient issues

**Next Step:** Add attention to sequence models to let the model focus on relevant parts of the input.

---

### Unit 07.7 — Attention for NLP

**What is it?**
Attention is a mechanism that lets each token in a sequence dynamically focus on other tokens when computing its representation. Instead of compressing the entire input into one fixed vector, attention creates a weighted connection between every pair of positions.

**Why does it matter?**
In sequence models, the bottleneck is the fixed-length hidden state — it must compress an entire sentence into one vector. Attention fixes this by letting the decoder directly access all encoder states, weighted by relevance. This is the key innovation behind transformers and modern NLP.

**Why learn it here?**
After sequence models, you understand the limitation: the fixed hidden state. Attention is the solution. It completes the conceptual bridge from RNNs to transformers, which you'll study in Phase 08.

**Prerequisites:** Unit 07.6 (sequence models).

**Mental Model:**
When you translate a sentence, you don't memorize the entire source before starting. You glance back at relevant source words as you produce each target word. Attention formalizes this "glancing back" — it computes a relevance score between each target word and every source word.

```text
Without attention: encoder → single vector → decoder
With attention:    encoder → all states → weighted access → decoder
                   (decoder decides which states matter at each step)
```

**Core Concepts:**

- **Query, Key, Value:** the three vectors that compute attention scores
- **Attention score:** similarity between query and key vectors
- **Attention weights:** softmax-normalized scores (sum to 1)
- **Weighted sum:** the context vector is a weighted combination of value vectors
- **Bahdanau (additive) attention:** uses a feed-forward network to compute scores
- **Luong (multiplicative) attention:** uses dot product between query and key
- **Self-attention:** queries, keys, and values all come from the same sequence
- **Cross-attention:** queries from one sequence, keys/values from another (encoder-decoder)

**How It Works:**

1. For each decoder step, compute a query vector from the current decoder state.
2. Compute attention scores: score(q, k_i) for each encoder state k_i.
3. Normalize scores with softmax to get attention weights.
4. Compute context vector: weighted sum of encoder value vectors using attention weights.
5. Combine context vector with decoder state to produce the output.

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.encoder_attn = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_attn = nn.Linear(decoder_dim, attention_dim, bias=False)
        self.v = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch, src_len, encoder_dim)
        # decoder_hidden: (batch, decoder_dim)

        src_len = encoder_outputs.shape[1]

        # Repeat decoder hidden for each source position
        decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)

        # Compute attention scores
        energy = torch.tanh(self.encoder_attn(encoder_outputs) + self.decoder_attn(decoder_hidden))
        attention = self.v(energy).squeeze(2)  # (batch, src_len)

        # Normalize
        weights = F.softmax(attention, dim=1)  # (batch, src_len)

        # Weighted sum
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs)  # (batch, 1, encoder_dim)

        return context.squeeze(1), weights

# Example usage
attn = Attention(encoder_dim=256, decoder_dim=256, attention_dim=128)
enc_out = torch.randn(4, 20, 256)   # batch=4, src_len=20
dec_hid = torch.randn(4, 256)       # batch=4, decoder_dim=256
context, weights = attn(enc_out, dec_hid)
print(context.shape)   # torch.Size([4, 256])
print(weights.shape)   # torch.Size([4, 20])
```

**Simple Example:**

```python
import torch
import torch.nn.functional as F

def simple_dot_attention(query, keys):
    """Compute dot-product attention scores."""
    scores = torch.bmm(keys, query.unsqueeze(2)).squeeze(2)  # (batch, seq_len)
    weights = F.softmax(scores, dim=1)
    context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
    return context, weights

keys = torch.randn(2, 10, 64)  # batch=2, seq_len=10, dim=64
query = torch.randn(2, 64)
context, weights = simple_dot_attention(query, keys)
print(context.shape)   # (2, 64)
print(weights.sum(dim=1))  # Should be ~1.0 for each batch
```

**Real-World Example:**
Machine translation uses attention to align source and target words. Text summarization uses attention to identify important sentences. Question answering uses self-attention to understand relationships between question and context tokens.

**Common Mistakes:**

- Not scaling attention scores by sqrt(d_k) — causes softmax to saturate
- Confusing self-attention with cross-attention — they serve different purposes
- Assuming attention weights are always interpretable — they aren't always meaningful
- Using attention without masking in decoder — allows peeking at future tokens

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Attention weights are uniform | Not enough training or wrong scaling | Check weight distribution | Train longer, add scaling factor |
| Softmax outputs are all 0 or 1 | Scores too large, no scaling | Check score magnitude | Divide scores by sqrt(d_k) |
| Decoder attends to padding tokens | No attention masking | Check masking logic | Mask padding positions before softmax |
| Training is very slow | Computing full attention matrix | Check sequence lengths | Use efficient attention or truncate sequences |

**Alternatives:**

| Mechanism | Use When | Avoid When |
|---|---|---|
| Bahdanau attention | Seq2seq with alignment learning | Need fast decoding |
| Luong attention | Simpler, faster dot-product scoring | Complex alignment patterns |
| Self-attention | Understanding intra-sequence relationships | Very long sequences without efficient impl |
| Sparse attention | Long sequences, limited compute | Full context is needed |

**Best Practices:**

- Always scale attention scores by sqrt(d_k) for stable training
- Use masking to prevent attending to padding or future tokens
- Visualize attention weights to debug and interpret model behavior
- Start with dot-product attention — it's simple and effective
- Use multi-head attention for richer representation (covered in transformers)

**Hands-On Practice:**

1. **Basic:** Implement dot-product attention from scratch. Verify weights sum to 1.
2. **Guided:** Add attention to an LSTM encoder-decoder for a toy translation task.
3. **Independent:** Implement Bahdanau attention and compare to Luong attention.
4. **Realistic:** Visualize attention alignment for a real translation pair. Does it make linguistic sense?
5. **Challenge:** Implement multi-head attention and explain why multiple heads help.

**Exit Criteria:**

- You can implement attention from scratch (dot-product, Bahdanau)
- You can explain query/key/value and attention weights
- You can visualize and interpret attention patterns
- You understand why attention leads to transformers

**Next Step:** Learn how to properly evaluate NLP models with task-specific metrics.

---

### Unit 07.8 — NLP Evaluation

**What is it?**
The methods and metrics for measuring the quality of NLP systems. Different NLP tasks require different evaluation strategies: classification uses accuracy/F1, generation uses BLEU/ROUGE, and ranking uses precision@k/MAP.

**Why does it matter?**
Without proper evaluation, you cannot tell if your model is improving, overfitting, or producing useful outputs. NLP evaluation is trickier than classification accuracy — language is subjective, and there are often multiple valid answers.

**Why learn it here?**
After building models (units 07.1–07.7), you need to measure them. This unit teaches you which metrics to use for which tasks and how to interpret them correctly.

**Prerequisites:** Unit 07.4 (classification), Unit 07.6–07.7 (sequence/attention models).

**Mental Model:**
Different NLP tasks ask different questions. Classification asks "is this label right?" (accuracy/F1). Generation asks "does this output look like human text?" (BLEU/ROUGE). Retrieval asks "did we find the right things?" (precision@k/MAP). Choose the metric that matches the question.

**Core Concepts:**

- **Accuracy:** fraction correct — simple but misleading on imbalanced data
- **Precision:** of all items predicted positive, how many are actually positive
- **Recall:** of all actual positives, how many were predicted positive
- **F1 score:** harmonic mean of precision and recall — balances both
- **Confusion matrix:** shows where the model confuses classes
- **BLEU:** measures n-gram overlap between generated and reference text (translation)
- **ROUGE:** measures n-gram overlap for recall-focused tasks (summarization)
- **METEOR:** improves BLEU with synonyms, stemming, and alignment
- **Perplexity:** measures how surprised the model is by test text (language modeling)
- **Word Error Rate (WER):** for speech recognition — edit distance between reference and hypothesis
- **Human evaluation:** the gold standard for open-ended generation tasks

**How It Works:**

1. Choose the metric that matches your task type.
2. Compute the metric on a held-out test set (never training data).
3. Compare against baselines and human performance.
4. Analyze errors qualitatively (look at what the model gets wrong).
5. Use error analysis to guide improvements.

**Syntax & Implementation:**

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from rouge_score import rouge_scorer

# === Classification Metrics ===
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 0, 1, 0, 1, 1, 0, 1, 0]

print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall: {recall_score(y_true, y_pred):.3f}")
print(f"F1: {f1_score(y_true, y_pred):.3f}")
print(classification_report(y_true, y_pred))

# === BLEU (for translation/generation) ===
references = [["the", "cat", "is", "on", "the", "mat"]]
hypothesis = ["the", "cat", "sat", "on", "the", "mat"]
bleu = corpus_bleu([references], [hypothesis], weights=(0.25, 0.25, 0.25, 0.25))
print(f"BLEU: {bleu:.3f}")

# === ROUGE (for summarization) ===
scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
reference = "the cat is on the mat"
hypothesis = "the cat sat on the mat"
scores = scorer.score(reference, hypothesis)
for key, score in scores.items():
    print(f"{key}: P={score.precision:.3f} R={score.recall:.3f} F1={score.fmeasure:.3f}")
```

**Simple Example:**

```python
from sklearn.metrics import f1_score, classification_report

# Multi-class sentiment
y_true = ["positive", "negative", "positive", "neutral", "negative"]
y_pred = ["positive", "negative", "neutral", "neutral", "negative"]

print(f"Macro F1: {f1_score(y_true, y_pred, average='macro'):.3f}")
print(classification_report(y_true, y_pred))
```

**Real-World Example:**
Google evaluates search with MAP and NDCG. Machine translation benchmarks use BLEU on WMT datasets. Summarization uses ROUGE. Chatbots require human evaluation for open-ended quality. Alexa uses intent accuracy + slot F1 for dialogue.

**Common Mistakes:**

- Using accuracy on imbalanced data — 95% accuracy means nothing if 95% of data is one class
- Reporting BLEU without human evaluation — BLEU correlates imperfectly with quality
- Evaluating on training data — always use held-out test sets
- Ignoring confidence intervals — single-run metrics are noisy
- Using ROUGE for translation and BLEU for summarization — wrong task fit

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| High accuracy but model is useless | Class imbalance | Check confusion matrix | Use F1, PR-AUC, per-class metrics |
| BLEU score is 0 | No n-gram overlap with reference | Inspect outputs vs references | Check preprocessing, try smoothing |
| ROUGE scores don't match human judgment | Metric captures overlap, not meaning | Compare ROUGE to human ratings | Supplement with human evaluation |
| Metrics differ between runs | No fixed random seed | Check seed setting | Fix random_state, report std dev |

**Decision Guidance:**

| Task | Primary Metric | Secondary Metrics | Avoid |
|---|---|---|---|
| Binary classification | F1 | Precision, Recall, AUC-ROC | Accuracy alone on imbalanced data |
| Multi-class classification | Macro F1 | Per-class F1, confusion matrix | Micro F1 on imbalanced data |
| Machine translation | BLEU | METEOR, chrF, human eval | BLEU alone as final quality measure |
| Summarization | ROUGE-1, ROUGE-2, ROUGE-L | Human eval, factual consistency | ROUGE without considering coherence |
| Language modeling | Perplexity | Human eval for open-ended text | Perplexity across different vocabularies |
| Named entity recognition | Entity-level F1 | Partial matching F1 | Token-level accuracy |

**Best Practices:**

- Always report multiple metrics, not just one
- Use macro averaging for imbalanced multi-class problems
- Pair automatic metrics with human evaluation for generation tasks
- Report confidence intervals or standard deviations across runs
- Use a fixed test set and never evaluate on training data

**Hands-On Practice:**

1. **Basic:** Compute accuracy, precision, recall, and F1 for a binary classifier.
2. **Guided:** Build a confusion matrix and identify which classes are most confused.
3. **Independent:** Evaluate a translation model with BLEU and compare to human judgments.
4. **Realistic:** Evaluate a summarization system with ROUGE and supplement with human evaluation.
5. **Challenge:** Design an evaluation suite for a chatbot that includes automatic metrics, human ratings, and safety checks.

**Exit Criteria:**

- You can choose appropriate metrics for different NLP tasks
- You can compute and interpret BLEU, ROUGE, and F1 scores
- You can identify when automatic metrics disagree with human judgment

**Next Step:** Synthesize everything into a complete NLP project.

---

### Unit 07.9 — NLP Synthesis & Review

**What is it?**
A cumulative integration unit that combines all NLP concepts — preprocessing, feature engineering, embeddings, sequence models, attention, and evaluation — into a complete, independent NLP project.

**Why does it matter?**
Knowing individual techniques is not enough. You must be able to architect an end-to-end NLP system: choose preprocessing steps, select representations, build and train a model, evaluate it properly, and analyze errors.

**Prerequisites:** All previous units in Phase 07.

**Mini Project: Sentiment Analysis System with Multiple Approaches**

**Objective:**
Build a complete sentiment analysis system that compares classical ML (TF-IDF + classifier) with deep learning (LSTM with embeddings), evaluates both properly, and produces an error analysis report.

**Requirements:**

- Load and explore a real dataset (e.g., IMDB reviews, Amazon reviews)
- Implement a full preprocessing pipeline
- Build two models:
  1. Classical: TF-IDF + Logistic Regression (or Naive Bayes)
  2. Deep: Word Embeddings + Bidirectional LSTM
- Train both models with proper train/validation/test splits
- Evaluate with appropriate metrics (F1, confusion matrix, per-class analysis)
- Visualize attention weights (for the LSTM model)
- Compare model performance and explain trade-offs
- Write an error analysis: what types of reviews does each model get wrong?
- Save all artifacts and write a README

**Suggested Architecture:**

```text
Raw data → EDA → preprocessing pipeline
                     ↓
              ┌──────┴──────┐
              ↓              ↓
        TF-IDF features   Token sequences
              ↓              ↓
    Logistic Regression   Embedding + LSTM
              ↓              ↓
          Predictions     Predictions
              └──────┬──────┘
                     ↓
              Evaluation & Comparison
                     ↓
              Error Analysis & Report
```

**Dataset Options:**

| Dataset | Size | Task | Source |
|---|---|---|---|
| IMDB Reviews | 50K | Binary sentiment | `keras.datasets` or Hugging Face |
| Amazon Reviews | 3.6M+ | Multi-class sentiment | Hugging Face datasets |
| SST-2 | 67K | Binary sentiment | GLUE benchmark |
| Yelp Reviews | 560K | Multi-class sentiment | Hugging Face datasets |
| Twitter Sentiment | 1.6M | Binary sentiment | Kaggle |

**Milestones:**

1. Data loading and EDA (2 hours)
2. Preprocessing pipeline (1 hour)
3. Classical ML baseline (1 hour)
4. LSTM model (2 hours)
5. Evaluation and comparison (1 hour)
6. Error analysis and report (1 hour)

**Expected Output:**

- Cleaned dataset or data loader
- Two trained models (classical and deep learning)
- Evaluation metrics for both models
- Confusion matrices and per-class analysis
- Error analysis document
- README with setup instructions, results, and limitations
- Saved model artifacts

**Evaluation Criteria:**

| Criterion | Points |
|---|---|
| Preprocessing is correct and reproducible | 15 |
| Classical ML baseline is well-tuned | 15 |
| Deep learning model is correctly implemented | 20 |
| Evaluation uses appropriate metrics | 15 |
| Error analysis is insightful and specific | 15 |
| Code is clean and well-organized | 10 |
| README explains decisions and limitations | 10 |

**Failure Cases to Test:**

- Empty or very short reviews
- Reviews with sarcasm or negation
- Reviews in different languages
- Adversarial inputs (random characters, all caps)
- Class imbalance in test set

**Possible Improvements:**

- Add a third model (e.g., CNN for text, or transformer-based)
- Use pre-trained BERT embeddings as input to the LSTM
- Add data augmentation (synonym replacement, back-translation)
- Implement early stopping and learning rate scheduling
- Add production-ready inference code

**Advanced Extensions:**

- Deploy the best model as a simple API
- Add real-time prediction capability
- Build a simple web interface for user input
- Compare to a pre-trained transformer model (BERT)

**Knowledge Check:**

- Why did you choose the preprocessing steps you did?
- Which model performs better and why?
- What are the failure modes of each approach?
- How would you improve performance on the hardest cases?
- What would happen if you deployed this model to production?
- How would you monitor for model degradation over time?

**Exit Criteria:**

- You can build a complete NLP system from raw text to evaluation
- You can compare classical and deep learning approaches with justification
- You can identify and explain model failure modes
- You can produce a professional project report

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| BoW/TF-IDF vs embeddings | Interpretability, speed, small data | Semantic meaning, large data, neural models | Simplicity vs meaning |
| LSTM vs Transformer | Small/medium datasets, limited compute | Long sequences, parallel training, state-of-the-art | Architecture complexity vs performance |
| Pre-trained vs trained from scratch | Small domain-specific corpus | Large in-domain corpus available | Transfer learning vs domain specificity |
| BLEU vs human eval | Fast iteration, translation tasks | Final quality assessment, open-ended generation | Speed vs accuracy |
| Word-level vs subword tokenization | Classical ML, clean text | Neural models, morphologically rich languages | Vocabulary size vs coverage |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model performs well in validation but fails in production | Data leakage or bad split | Inspect features and split logic | Re-split before preprocessing; remove leaked fields |
| High training error and high validation error | Underfitting | Learning curves, simple baseline comparison | Add features/model capacity or fix preprocessing |
| Low training error and high validation error | Overfitting | Train vs validation curves | Regularize, simplify model, add data |
| Accuracy looks high but users complain | Wrong metric for imbalanced data | Confusion matrix, PR-AUC, segment analysis | Use task-relevant metric and threshold tuning |
| BLEU score is high but translation is bad | Metric doesn't capture fluency or meaning | Human evaluation | Use human eval + METEOR + chrF |
| LSTM ignores long-range dependencies | Vanishing gradients or insufficient training | Gradient norms, train longer | Use LSTM/GRU, add attention, check learning rate |
| Attention weights look random | Insufficient training or wrong masking | Visualize weights | Train longer, add masking, check preprocessing |

---

## Phase Review Checklist

- [ ] All 9 units complete.
- [ ] Text preprocessing pipeline built and tested.
- [ ] BoW and TF-IDF representations created and compared.
- [ ] N-grams implemented and evaluated.
- [ ] Text classification pipeline built with multiple classifiers.
- [ ] Word embeddings trained or loaded and evaluated.
- [ ] LSTM sequence model implemented and trained.
- [ ] Attention mechanism implemented and visualized.
- [ ] NLP evaluation metrics computed and interpreted.
- [ ] Mini project completed with both classical and deep learning approaches.
- [ ] Error analysis conducted and documented.
- [ ] Code is clean, reproducible, and well-documented.

---

## Mastery Check

Without following a tutorial, you should be able to:

1. Preprocess and tokenize text for any NLP task.
2. Build BoW, TF-IDF, and N-gram features from text.
3. Train a text classifier with a proper ML pipeline.
4. Explain word embeddings and compute word similarity.
5. Implement an LSTM for sequence classification.
6. Add attention to a sequence model.
7. Choose and compute appropriate evaluation metrics for NLP tasks.
8. Compare classical and deep learning approaches for NLP.
9. Build an end-to-end NLP system from scratch.
10. Conduct error analysis and propose improvements.

---

## Interview / Explain-Back Questions

- What is the difference between stemming and lemmatization? When would you choose one over the other?
- Explain TF-IDF. Why is it better than raw word counts?
- Why do N-grams help with sentiment analysis but can hurt with small datasets?
- What is the vanishing gradient problem and how does LSTM solve it?
- Explain attention in your own words. Why is it important for sequence models?
- When would you use BLEU vs ROUGE? Give an example for each.
- How do you handle class imbalance in text classification?
- What are word embeddings and why are they better than one-hot encoding?
- Compare bidirectional and unidirectional LSTMs. When would you use each?
- How would you evaluate a chatbot that generates open-ended responses?

---

## Exit Criteria

Move to Phase 08 only when you can:

1. Build a complete NLP preprocessing pipeline from raw text.
2. Choose between classical (TF-IDF) and deep learning (embeddings + LSTM) approaches with justification.
3. Implement and train LSTM models for text classification.
4. Understand and implement attention mechanisms.
5. Evaluate NLP models with task-appropriate metrics.
6. Conduct error analysis and propose improvements.
7. Build an end-to-end NLP system independently.
8. Explain all major NLP concepts clearly and confidently.
