# Phase 11 — RAG Systems

> **Goal:** Master Retrieval-Augmented Generation — from keyword search to agentic RAG, building production-quality retrieval pipelines that ground LLM answers in real documents.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced
**Priority:** Essential
**Prerequisites:** Phase 09 (GenAI), Phase 10 (LLMs)
**Mastery target:** Level 5 — build, evaluate, and optimize production RAG systems independently

---

## Why This Phase Exists

LLMs are powerful but fundamentally limited: they cannot reliably know private, current, or source-specific facts. They hallucinate. They have knowledge cutoffs. RAG solves this by retrieving relevant evidence before generating an answer. Every enterprise LLM application — customer support, legal research, medical Q&A, code documentation, internal knowledge bases — relies on retrieval-augmented generation. This phase teaches the full stack: from the simplest keyword search to agentic retrieval systems.

### Phase Mental Model

RAG is a two-part system: **find the right evidence**, then **give it to the model**. If retrieval is bad, generation cannot reliably fix it.

```text
Documents → Ingest → Chunk → Embed → Index
                                        ↓
User Query → Retrieve → Rerank → Build Context → Generate → Evaluate
                                        ↓
                              Retrieval Quality ← → Answer Quality
```

RAG exists because LLMs cannot reliably know every private, current, or source-specific fact.

### What This Phase Prepares For

- Phase 12–13 (Frameworks) — LangChain/LangGraph RAG abstractions
- Phase 14 (Agents) — agentic retrieval and tool-augmented reasoning
- Phase 15 (Evaluation) — formal RAG evaluation metrics and benchmarks
- Phase 17 (Capstones) — production RAG applications with monitoring

---

## Units

### Unit 11.1 — Why RAG Exists

**What is it?**  
The motivation for Retrieval-Augmented Generation: why LLMs alone are insufficient for many real-world tasks.

**Why does it matter?**  
Without understanding the problem, you cannot design the solution. RAG exists because LLMs have three fundamental limitations: knowledge cutoff (they stop knowing things after training), hallucination (they invent plausible-sounding facts), and inability to access private or real-time data.

**Why learn it here?**  
You just completed LLM fundamentals. You know how generation works. Now you need to understand what generation *cannot* do, so the rest of this phase has clear purpose.

**Prerequisites:** Phase 09 (GenAI), Phase 10 (LLMs)

**Mental Model:**  
An LLM is a brilliant consultant who read everything up to a certain date but has no internet access and sometimes makes things up. RAG gives that consultant a library card and a search engine.

**Core Concepts:**

- Knowledge cutoff and staleness
- Hallucination types (factual, confabulation, ungrounded reasoning)
- Private/corporate data the model never saw
- The cost of fine-tuning vs. retrieval
- When RAG beats fine-tuning, long-context prompting, or manual search

**How It Works:**

Without RAG, the LLM answers from parametric memory (weights). With RAG, a retrieval system finds relevant documents, injects them into the prompt context, and the LLM generates an answer grounded in that evidence.

```text
Without RAG:
  User Question → LLM → Answer (may be wrong)

With RAG:
  User Question → Retriever → Relevant Docs → LLM + Context → Answer (grounded)
```

**Simple Example:**

```python
# Without RAG — model guesses
prompt_no_rag = "What is the return policy for Product X?"

# With RAG — model answers from retrieved docs
retrieved_docs = retrieve(query="return policy Product X", k=3)
context = "\n".join([doc.text for doc in retrieved_docs])
prompt_rag = f"""Based on the following documents, answer the question.
Documents:
{context}
Question: What is the return policy for Product X?"""
```

**Real-World Example:**  
A law firm uses RAG to answer legal research questions. The LLM alone might hallucinate case law. With RAG, the system retrieves actual case documents, statutes, and memos, then generates answers with citations. The lawyer can verify every claim against the source.

**Decision Guidance:**

| Use RAG When | Use Fine-Tuning When | Use Long-Context When |
|---|---|---|
| Knowledge changes frequently | Style/format/behavior must change | Corpus fits in context window |
| Source attribution is required | Domain-specific reasoning patterns | Query is simple and infrequent |
| Documents are private/proprietary | The model needs to *learn* a skill | Cost/latency are acceptable |
| You need to cite sources | You cannot expose documents at inference | The corpus is small (<100K tokens) |

**Common Mistakes:**

- Building RAG when the data fits in the context window and queries are simple
- Assuming RAG eliminates hallucination entirely
- Ignoring retrieval quality and blaming the LLM
- Over-engineering retrieval before validating chunking and embeddings
- Not evaluating retrieval and generation separately

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model still hallucinates despite RAG | Retrieved docs are irrelevant | Print top-k retrieved chunks | Improve retrieval before improving prompts |
| Answers are generic/unhelpful | Context is too large or noisy | Count tokens in context window | Reduce k, improve reranking, filter by relevance |
| Model ignores retrieved context | Prompt doesn't instruct grounding | Test with and without context | Add explicit grounding instructions |
| User says "this isn't in our docs" | Data not indexed or wrong chunking | Check document ingestion pipeline | Verify all docs are chunked and indexed |

**Best Practices:**

- Start with the simplest possible RAG (keyword search + basic prompting) and iterate
- Always evaluate retrieval quality separately from answer quality
- Log what was retrieved alongside the answer for debugging
- Design for citations from day one
- Measure latency and cost, not just accuracy

**Hands-On Practice:**

1. **Basic:** Ask an LLM a question about your company's internal policy. Observe the hallucination.
2. **Guided:** Copy-paste relevant policy text into the prompt. Observe how the answer improves.
3. **Independent:** Build a simple file-based retrieval system that finds relevant paragraphs.
4. **Realistic:** Compare RAG vs. no-RAG on 10 questions from real documentation.
5. **Challenge:** Design a decision tree: when should the system use RAG, fine-tuning, or long-context?

**Knowledge Check:**

- What are the three main limitations of LLMs that RAG addresses?
- When is fine-tuning a better choice than RAG?
- Why should you evaluate retrieval and generation separately?
- What happens if your retrieved documents are irrelevant?

**Exit Criteria:**

- You can explain why RAG exists and what problems it solves.
- You can compare RAG with fine-tuning and long-context prompting.
- You can identify when RAG is the right architecture.

**Next Step:** Learn keyword search — the oldest and most reliable retrieval baseline.

---

### Unit 11.2 — Keyword Search

**What is it?**  
Text retrieval based on word matching: BM25, TF-IDF, and boolean search.

**Why does it matter?**  
Keyword search is the foundation of all retrieval systems. It is fast, interpretable, requires no embeddings, and often outperforms semantic search for exact-match queries. Every production RAG system benefits from including keyword search as a component.

**Why learn it here?**  
Before you learn semantic (vector) search, you need the baseline. Keyword search teaches you what retrieval *is* and gives you a comparison point for vector search.

**Prerequisites:** Phase 09 (GenAI), Phase 10 (LLMs), basic Python

**Mental Model:**  
Keyword search is a librarian who finds books by matching words in the title/index. Fast and reliable for exact terms, but misses synonyms and context.

**Core Concepts:**

- Term frequency (TF): how often a word appears in a document
- Inverse document frequency (IDF): how rare a word is across all documents
- BM25: a scoring function that combines TF and IDF with length normalization
- TF-IDF: a simpler predecessor to BM25
- Tokenization, stemming, stop words
- Index structures (inverted index)

**How It Works:**

1. **Index time:** For each document, tokenize the text, compute term frequencies, and build an inverted index (word → list of documents containing it).
2. **Query time:** Tokenize the query, look up each term in the inverted index, compute BM25/TF-IDF scores, rank documents by score, return top-k.

```text
Index: { "term": [(doc_id, tf, doc_length), ...] }
Query: tokenize → score each doc → rank → top-k
```

**Syntax & Implementation:**

```python
from rank_bm25 import BM25Okapi
import nltk

documents = [
    "The return policy allows returns within 30 days of purchase.",
    "Shipping takes 5-7 business days for standard delivery.",
    "Returns are accepted only for unopened items in original packaging.",
]

tokenized_docs = [nltk.word_tokenize(doc.lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_docs)

query = "return policy"
tokenized_query = nltk.word_tokenize(query.lower())
scores = bm25.get_scores(tokenized_query)
top_docs = sorted(enumerate(scores), key=lambda x: -x[1])

for doc_idx, score in top_docs[:3]:
    print(f"Score: {score:.3f} | {documents[doc_idx]}")
```

**Real-World Example:**  
Elasticsearch powers thousands of production search systems. It uses BM25 as its default scoring function. When you search a product catalog, keyword search finds exact product names, model numbers, and specifications.

**Alternatives:**

| Tool | Use When | Avoid When |
|---|---|---|
| BM25 (rank_bm25) | Small-to-medium corpus, exact-match queries | Synonym understanding needed |
| Elasticsearch | Large-scale production search | Simple prototype or small dataset |
| Whoosh | Python-native full-text search | Need distributed search |
| SQLite FTS | Embedded search in small apps | Complex ranking requirements |

**Common Mistakes:**

- Not handling case sensitivity
- Ignoring stop words when they matter (e.g., "not" in "not returning")
- Using TF-IDF when BM25 is available (BM25 is generally better)
- Not normalizing text (lowercasing, punctuation handling)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Expected document not retrieved | Query terms don't match doc tokens | Print tokenized query and doc | Add preprocessing (lowercasing, stemming) |
| Irrelevant documents scored highly | Rare words in query match noise | Check IDF scores | Use stop word removal, test on more queries |
| Slow search on large corpus | In-memory index too large | Measure index size | Use Elasticsearch or disk-based index |
| BM25 scores are all near zero | Query is too short or all stop words | Print tokenized query | Expand query or use phrase matching |

**Best Practices:**

- Always lowercase and tokenize consistently
- Use BM25 over raw TF-IDF
- Include keyword search in hybrid retrieval systems
- Benchmark on real queries before switching to vector search
- Log retrieval scores for debugging

**Hands-On Practice:**

1. **Basic:** Build a BM25 search over 10 documents.
2. **Guided:** Add preprocessing (lowercasing, stop word removal) and compare results.
3. **Independent:** Search a set of FAQ documents and measure precision@3.
4. **Realistic:** Compare BM25 with naive cosine similarity on embeddings for the same corpus.
5. **Challenge:** Implement phrase matching ("return policy" as exact phrase vs. separate words).

**Knowledge Check:**

- How does BM25 differ from simple term frequency matching?
- When does keyword search outperform semantic search?
- What is an inverted index and why is it fast?
- What preprocessing steps improve keyword search quality?

**Exit Criteria:**

- You can implement BM25 search over a document collection.
- You can explain when keyword search is preferable to vector search.
- You can preprocess text to improve retrieval quality.

**Next Step:** Learn embeddings — how to represent meaning as numbers.

---

### Unit 11.3 — Embeddings

**What is it?**  
Text embeddings are dense vector representations that capture semantic meaning. Similar texts produce similar vectors.

**Why does it matter?**  
Embeddings enable semantic search: finding documents by meaning rather than exact word match. "How do I return a product?" and "What is the refund process?" will have similar embeddings even though they share almost no words.

**Why learn it here?**  
You just learned keyword search. Embeddings are the alternative representation that powers vector search. You need to understand what embeddings *are* before you can store, search, or evaluate them.

**Prerequisites:** Phase 09 (GenAI), Phase 10 (LLMs), basic linear algebra awareness

**Mental Model:**  
An embedding is a coordinate in meaning-space. Words and sentences that are semantically similar are nearby points. "King" and "queen" are close. "King" and "airplane" are far apart.

**Core Concepts:**

- Dense vs sparse vectors
- Embedding dimensions (384, 768, 1536, etc.)
- Pre-trained embedding models (sentence-transformers, OpenAI, Cohere)
- Token-level vs sentence-level embeddings
- Mean pooling vs CLS token
- Embedding normalization (L2)

**How It Works:**

An embedding model takes text and produces a fixed-size vector of floating-point numbers. The model was trained on vast amounts of text so that semantically similar sentences end up close together in vector space.

```text
"Hello world" → [0.02, -0.15, 0.43, ..., 0.08]  (768 dimensions)
"Hi there"    → [0.03, -0.14, 0.41, ..., 0.09]  (close in space)
"Car engine"  → [-0.32, 0.11, -0.05, ..., 0.21] (far in space)
```

**Syntax & Implementation:**

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I return a product?",
    "What is your refund policy?",
    "How to cook pasta",
]

embeddings = model.encode(sentences)
print(embeddings.shape)  # (3, 384)
print(embeddings[0][:5])  # first 5 dimensions of first embedding
```

**Real-World Example:**  
OpenAI's `text-embedding-3-small` produces 1536-dimensional vectors. Companies use these to build semantic search over support tickets, legal documents, and codebases. A customer typing "I can't log in" finds relevant articles even if those articles use the phrase "authentication failure."

**Alternatives:**

| Model | Dimensions | Use When | Trade-off |
|---|---|---|---|
| all-MiniLM-L6-v2 | 384 | Fast, local, good general quality | Lower quality than large models |
| all-mpnet-base-v2 | 768 | Better quality, still local | Slower than MiniLM |
| text-embedding-3-small | 1536 | OpenAI ecosystem, high quality | API cost, data sent to OpenAI |
| Cohere embed-v3 | 1024 | Multilingual, long documents | API dependency |
| BGE-large | 1024 | Strong retrieval benchmarks | Larger model, more memory |

**Common Mistakes:**

- Using a general-purpose model for specialized domains (medical, legal)
- Not normalizing embeddings before similarity computation
- Mixing embedding models in the same index
- Ignoring embedding dimension when designing vector databases
- Assuming larger dimensions always mean better quality

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Similar sentences have low similarity | Model not suited for task | Test with known similar pairs | Switch to a retrieval-optimized model |
| Embeddings take too long to generate | Model too large for hardware | Check GPU/CPU usage | Use a smaller model or batch processing |
| Mixing models breaks search | Different embedding spaces | Check model names in index | Re-embed entire corpus with one model |
| All similarities near 0.5 | Embeddings not normalized | Check vector norms | Apply L2 normalization |

**Best Practices:**

- Use the same embedding model for indexing and querying
- Normalize embeddings for cosine similarity
- Choose a model appropriate for your domain and language
- Batch embedding generation for efficiency
- Store embedding model name alongside vectors for reproducibility

**Hands-On Practice:**

1. **Basic:** Embed 5 sentences and print their shapes.
2. **Guided:** Compute cosine similarity between all pairs and verify similar sentences score higher.
3. **Independent:** Embed a set of FAQ questions and find the most similar pair.
4. **Realistic:** Compare two embedding models on the same 20-query dataset.
5. **Challenge:** Embed documents of varying lengths (1 sentence to 500 words) and analyze quality degradation.

**Knowledge Check:**

- What does an embedding vector represent?
- Why must you use the same model for indexing and querying?
- What is the difference between dense and sparse embeddings?
- When would you choose a local embedding model over an API?

**Exit Criteria:**

- You can generate embeddings for text using a pre-trained model.
- You can explain how embedding space captures semantic similarity.
- You can choose an appropriate embedding model for a use case.

**Next Step:** Learn how to measure similarity between embedding vectors.

---

### Unit 11.4 — Vector Similarity

**What is it?**  
Methods for measuring how close two embedding vectors are in vector space: cosine similarity, dot product, and Euclidean distance.

**Why does it matters?**  
Retrieval quality depends on choosing the right similarity metric. Cosine similarity is most common, but dot product and Euclidean distance behave differently and have different normalization requirements.

**Why learn it here?**  
You now understand embeddings. Before you store them in a vector database, you need to understand how similarity is computed so you can choose the right metric and debug retrieval failures.

**Prerequisites:** Unit 11.3 (Embeddings)

**Mental Model:**  
Similarity metrics are different rulers for measuring distance in meaning-space. Cosine measures angle (direction), dot product measures alignment × magnitude, Euclidean measures straight-line distance.

**Core Concepts:**

- Cosine similarity: angle between vectors (range -1 to 1)
- Dot product: sum of element-wise products (unbounded)
- Euclidean distance: straight-line distance (≥ 0)
- Normalization: making vectors unit-length so dot product = cosine similarity
- When to use which metric

**How It Works:**

```text
Cosine:  sim(A, B) = (A · B) / (‖A‖ × ‖B‖)     → angle between vectors
Dot:     sim(A, B) = A · B                         → alignment × magnitude
Euclid:  dist(A, B) = ‖A - B‖                     → straight-line distance
```

If vectors are L2-normalized (unit length), cosine similarity equals dot product, and Euclidean distance is a monotonic function of both.

**Syntax & Implementation:**

```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def dot_product_similarity(a, b):
    return np.dot(a, b)

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

# Example
vec_a = np.array([0.1, 0.8, 0.3, 0.5])
vec_b = np.array([0.2, 0.7, 0.4, 0.4])
vec_c = np.array([0.9, 0.1, 0.1, 0.0])

print(f"Cosine(A,B): {cosine_similarity(vec_a, vec_b):.4f}")
print(f"Cosine(A,C): {cosine_similarity(vec_a, vec_c):.4f}")
print(f"Euclidean(A,B): {euclidean_distance(vec_a, vec_b):.4f}")
print(f"Euclidean(A,C): {euclidean_distance(vec_a, vec_c):.4f}")
```

**Decision Guidance:**

| Metric | Use When | Avoid When | Notes |
|---|---|---|---|
| Cosine similarity | Text retrieval, normalized embeddings | Magnitude matters (e.g., popularity) | Default choice for most RAG systems |
| Dot product | Embeddings are normalized, or magnitude is meaningful | Embeddings have different scales | Faster than cosine if normalized |
| Euclidean distance | Geometric clustering, k-means | High-dimensional spaces (curse of dimensionality) | Less common for text retrieval |

**Real-World Example:**  
A search engine uses cosine similarity with normalized embeddings. Two articles about "machine learning" have vectors pointing in similar directions (high cosine), even if one article is longer (higher magnitude). Without normalization, longer documents would dominate.

**Common Mistakes:**

- Using dot product on unnormalized embeddings (longer documents score higher)
- Mixing cosine similarity and Euclidean distance in the same system
- Not normalizing before using inner product search in vector databases
- Assuming cosine similarity is always in [0, 1] (it ranges -1 to 1 for unnormalized vectors)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Irrelevant long documents rank highest | Using dot product without normalization | Check if vectors are normalized | Normalize embeddings or switch to cosine |
| Similar documents have low similarity | Wrong metric for embedding model | Test with known pairs | Check model docs for recommended metric |
| All similarities are negative | Embeddings not normalized, opposite directions | Print vector norms | Apply L2 normalization |

**Best Practices:**

- Default to cosine similarity for text retrieval
- L2-normalize embeddings before indexing if using dot product search
- Test similarity metrics on real query-document pairs
- Document which metric your system uses (it affects reranking and thresholds)

**Hands-On Practice:**

1. **Basic:** Compute all three metrics for 3 vector pairs.
2. **Guided:** Embed 10 sentences, compute pairwise cosine similarity, find most/least similar pairs.
3. **Independent:** Compare retrieval quality with cosine vs. dot product on the same index.
4. **Realistic:** Debug a system where dot product is returning unexpected results.
5. **Challenge:** Implement a similarity search function that accepts any metric as a parameter.

**Knowledge Check:**

- When does cosine similarity equal dot product?
- Why is Euclidean distance less common for text retrieval?
- What happens if you use dot product on unnormalized embeddings?
- How does normalization affect retrieval?

**Exit Criteria:**

- You can compute and compare cosine, dot product, and Euclidean similarity.
- You can explain when to use each metric.
- You can debug normalization-related retrieval issues.

**Next Step:** Learn vector databases — how to store and search millions of embeddings efficiently.

---

### Unit 11.5 — Vector Databases

**What is it?**  
Specialized databases designed to store, index, and search high-dimensional vectors efficiently using approximate nearest neighbor (ANN) algorithms.

**Why does it matter?**  
Naive linear search over millions of embeddings is too slow. Vector databases use indexing structures (HNSW, IVF, LSH) to find approximate nearest neighbors in milliseconds, not seconds.

**Why learn it here?**  
You understand embeddings and similarity. Now you need a scalable way to store and search them. This is the infrastructure layer of RAG.

**Prerequisites:** Unit 11.3 (Embeddings), Unit 11.4 (Vector Similarity)

**Mental Model:**  
A vector database is a search engine for meaning. Instead of indexing words, it indexes vectors and finds the closest ones to a query vector.

**Core Concepts:**

- Approximate nearest neighbor (ANN) vs exact search
- HNSW (Hierarchical Navigable Small World) — graph-based index
- IVF (Inverted File Index) — partition-based index
- LSH (Locality-Sensitive Hashing) — hash-based index
- Index parameters: n_lists, n_probe, ef_search, M
- Persistence and metadata filtering

**How It Works:**

1. **Index time:** Vectors are inserted into an index structure that partitions or connects them for fast traversal.
2. **Query time:** The query vector is placed into the index, and the algorithm traverses the structure to find approximate nearest neighbors.

```text
Naive: O(n × d) per query — scan every vector
HNSW:  O(log n) per query — navigate a graph
IVF:   O(n_lists × d) per query — search only relevant partitions
```

**Syntax & Implementation:**

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection(
    name="documents",
    metadata={"hnsw:space": "cosine"}
)

# Add documents
collection.add(
    documents=["How to return a product", "Shipping policy details", "Refund process"],
    embeddings=[[0.1, 0.8, 0.3], [0.2, 0.7, 0.4], [0.9, 0.1, 0.1]],
    ids=["doc1", "doc2", "doc3"],
    metadatas=[{"category": "returns"}, {"category": "shipping"}, {"category": "returns"}]
)

# Query
results = collection.query(
    query_embeddings=[[0.15, 0.75, 0.35]],
    n_results=2,
    where={"category": "returns"}
)
print(results["documents"])
```

**Alternatives:**

| Database | Use When | Avoid When |
|---|---|---|
| Chroma | Quick prototyping, local, simple API | Large-scale production (>10M vectors) |
| FAISS | High-performance local search, research | Need metadata filtering or persistence out of box |
| Pinecone | Managed cloud vector DB, production | Budget constraints or data sovereignty requirements |
| Weaviate | Hybrid search, GraphQL API, schema-based | Simple use case doesn't need features |
| Qdrant | High performance, filtering, Rust-based | Team prefers Python-native solutions |
| pgvector | Already using PostgreSQL | Need ANN at massive scale |

**Common Mistakes:**

- Not tuning index parameters (default HNSW may be slow for your data size)
- Ignoring the trade-off between search quality and speed (ef_search, n_probe)
- Not persisting the index after building
- Mixing different embedding dimensions in the same collection
- Not using metadata filtering to narrow search space

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Search is slow | Index not built or too small for HNSW | Check collection stats | Build index, tune ef_search |
| Recall is low | Index parameters too aggressive | Compare with brute-force | Increase n_probe or ef_search |
| Out of memory | Vector dimension × count too large | Check vector count and dimension | Use quantization or disk-based index |
| Metadata filter returns empty | Filter syntax wrong or values missing | Print metadata | Verify metadata matches filter |

**Best Practices:**

- Start with Chroma for prototyping, migrate to production DB later
- Always persist your index
- Tune index parameters based on your speed/quality requirements
- Use metadata filtering to reduce search space
- Benchmark recall@k against brute-force for your data

**Hands-On Practice:**

1. **Basic:** Create a Chroma collection, add 10 vectors, query for top-3.
2. **Guided:** Add metadata, filter queries by metadata, compare filtered vs unfiltered results.
3. **Independent:** Build an index of 1000+ vectors, measure query latency.
4. **Realistic:** Compare Chroma and FAISS on the same dataset for speed and recall.
5. **Challenge:** Tune HNSW parameters and measure the speed-recall trade-off.

**Knowledge Check:**

- What is ANN and why is it used instead of exact search?
- What is the trade-off when increasing HNSW's ef_search parameter?
- When would you use metadata filtering instead of increasing k?
- How do you choose between Chroma, FAISS, and a managed vector DB?

**Exit Criteria:**

- You can store, index, and query vectors in a vector database.
- You can explain the trade-off between search speed and quality.
- You can choose an appropriate vector database for a use case.

**Next Step:** Learn how to get documents into the system — document ingestion.

---

### Unit 11.6 — Document Ingestion

**What is it?**  
Loading documents from various formats (PDF, text, HTML, DOCX, markdown) into a format suitable for chunking and embedding.

**Why does it matter?**  
Real-world data comes in messy formats. PDFs have tables, images, headers, and footers. Web pages have navigation and ads. DOCX has formatting. If ingestion is bad, everything downstream — chunking, embedding, retrieval — inherits the garbage.

**Why learn it here?**  
You understand the retrieval components. Now you need to feed real documents into the pipeline. Ingestion is the first stage of the data pipeline.

**Prerequisites:** Phase 01 (Python), Unit 11.3 (Embeddings)

**Mental Model:**  
Ingestion is like a document preprocessing pipeline: raw file → clean text → structured chunks. Each format requires a different parser, but the goal is the same: extract the meaningful text.

**Core Concepts:**

- Format-specific parsers (PyPDF2, python-docx, BeautifulSoup)
- Text extraction vs metadata extraction
- Handling tables, images, and special content
- Encoding issues (UTF-8, Latin-1)
- Document metadata (title, author, date, source)
- Incremental ingestion (new/updated documents)

**How It Works:**

```text
Raw File → Parser → Clean Text + Metadata → (next: Chunking)
```

Each format needs its own parser. The goal is to extract the text content and useful metadata while discarding formatting noise.

**Syntax & Implementation:**

```python
from pathlib import Path
import PyPDF2
from docx import Document

def load_pdf(path):
    reader = PyPDF2.PdfReader(str(path))
    texts = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            texts.append({"text": text, "page": i + 1, "source": str(path)})
    return texts

def load_txt(path):
    text = Path(path).read_text(encoding="utf-8")
    return [{"text": text, "source": str(path)}]

def load_docx(path):
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    return [{"text": full_text, "source": str(path)}]

# Usage
docs = load_pdf("policy.pdf")
print(f"Loaded {len(docs)} pages from {docs[0]['source']}")
```

**Real-World Example:**  
A company ingests 10,000 PDF policy documents. Each PDF has a title page, table of contents, body text, tables, and appendices. The ingestion pipeline extracts text page by page, preserves page numbers and document titles as metadata, and skips blank pages.

**Alternatives:**

| Tool | Use When | Avoid When |
|---|---|---|
| PyPDF2 | Basic PDF text extraction | PDFs with complex layouts/tables |
| Unstructured | Multi-format, production ingestion | Simple prototyping |
| LangChain loaders | Quick prototyping with LangChain | Need fine-grained control |
| LlamaIndex readers | Document-heavy RAG pipelines | Simple use cases |
| Custom parsers | Unique formats or strict requirements | Standard formats are sufficient |

**Common Mistakes:**

- Not handling encoding errors (UnicodeDecodeError)
- Extracting headers/footers as content
- Losing table structure by flattening to text
- Not tracking page numbers or section titles
- Ingesting the same document twice without deduplication

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Empty documents after extraction | Parser failed silently | Check page count vs extracted text | Try different parser, check PDF type |
| Garbled text | Encoding issue | Print raw bytes | Specify encoding, try different parser |
| Tables are unreadable | Table structure lost in extraction | Visually compare PDF and text | Use table-aware parser (tabula-py) |
| Duplicate chunks in results | Documents ingested multiple times | Check document IDs | Hash documents for deduplication |

**Best Practices:**

- Always store source filename and page/section numbers as metadata
- Handle encoding explicitly (UTF-8 default)
- Deduplicate documents before ingestion
- Test ingestion on a few documents before processing thousands
- Log ingestion stats (documents loaded, pages, characters)

**Hands-On Practice:**

1. **Basic:** Load a text file and a PDF, print extracted text.
2. **Guided:** Extract metadata (title, page count, file size) alongside text.
3. **Independent:** Build an ingestion pipeline that handles TXT, PDF, and DOCX.
4. **Realistic:** Ingest a PDF with tables and compare text extraction quality across parsers.
5. **Challenge:** Build deduplication into your ingestion pipeline.

**Knowledge Check:**

- Why is page number metadata important for RAG?
- How do you handle encoding errors in document loading?
- When should you use a table-aware parser?
- What happens if you ingest the same document twice?

**Exit Criteria:**

- You can load documents from at least 3 formats.
- You can extract and preserve metadata during ingestion.
- You can handle common ingestion errors.

**Next Step:** Learn chunking — how to split documents into retrieval-sized pieces.

---

### Unit 11.7 — Chunking

**What is it?**  
Splitting documents into smaller, retrieval-friendly segments that preserve meaning and context.

**Why does it matter?**  
LLMs have context windows. Embedding models have token limits. And most importantly: a 10-page document rarely answers a specific question — a paragraph does. Chunking determines what information your system can retrieve and how precisely.

**Why learn it here?**  
Ingestion gets raw text. Chunking turns it into the unit of retrieval. Bad chunking is the #1 cause of poor RAG quality. This is arguably the most impactful unit in this phase.

**Prerequisites:** Unit 11.6 (Document Ingestion)

**Mental Model:**  
Chunking is like cutting a book into flashcards. Each flashcard should contain one complete idea, enough context to understand it, and a reference back to where it came from.

**Core Concepts:**

- Fixed-size chunking (character/token count)
- Recursive character splitting
- Semantic chunking (paragraph, section, heading-based)
- Overlap and context preservation
- Chunk size and the precision vs context trade-off
- Metadata propagation (source, page, section)

**How It Works:**

```text
Document → Split Strategy → Chunks + Metadata → (next: Embedding)
```

The chunking strategy determines:
- How much context each chunk carries
- Whether related information stays together or gets split
- How many chunks your index will contain (affects cost and speed)

**Syntax & Implementation:**

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text = """The return policy allows returns within 30 days of purchase.
Items must be in original packaging. Shipping costs are non-refundable.
Contact support@example.com to initiate a return.
Refunds are processed within 5-7 business days."""

# Fixed-size with overlap
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars): {chunk[:80]}...")
```

**Decision Guidance:**

| Strategy | Use When | Avoid When | Chunk Size |
|---|---|---|---|
| Fixed-size | Quick baseline, uniform documents | Semantic boundaries matter | 200-500 tokens |
| Recursive | General-purpose, respects structure | Documents have unusual structure | 200-500 tokens |
| Paragraph/section | Well-structured documents (markdown, HTML) | Plain text with no structure | Varies by document |
| Semantic | High-quality retrieval, diverse content | Prototyping or budget constraints | Varies by content |
| Document-level | Small documents, full-context queries | Large documents or long-context limits | Entire document |

**Real-World Example:**  
A legal document RAG system chunks by section and subsection, not by character count. Each chunk preserves the section heading, the text, and the subsection number as metadata. When a user asks "What is the liability clause?", the system retrieves the exact section.

**Common Mistakes:**

- Chunking by arbitrary character count without preserving meaning
- No overlap — losing context at chunk boundaries
- Chunks too small (lose context) or too large (dilute relevance)
- Not propagating metadata (source, page, section)
- Splitting tables or code blocks across chunks

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Relevant answer split across chunks | Chunks too small, no overlap | Inspect chunk boundaries | Increase chunk size and overlap |
| Chunks contain unrelated content | Chunks too large | Check average chunk size | Reduce chunk size, use semantic chunking |
| Table data is garbled | Table split across chunks | Inspect chunk content | Detect tables and chunk them separately |
| Same sentence appears in 3 chunks | Overlap too large | Count duplicate chunks | Reduce overlap ratio |

**Best Practices:**

- Start with recursive character splitting (good default)
- Use 10-20% overlap between chunks
- Always propagate source metadata to each chunk
- Test chunking on 10+ real documents before scaling
- Measure retrieval quality at different chunk sizes

**Hands-On Practice:**

1. **Basic:** Chunk a paragraph using fixed-size splitting.
2. **Guided:** Compare fixed-size vs recursive splitting on the same document.
3. **Independent:** Chunk a markdown document, preserving section headers as metadata.
4. **Realistic:** Test 3 chunk sizes (100, 300, 500 tokens) and measure retrieval precision.
5. **Challenge:** Implement semantic chunking that splits at paragraph boundaries.

**Knowledge Check:**

- Why is overlap important in chunking?
- What happens when chunks are too small? Too large?
- How does chunking affect embedding quality?
- Why should you propagate source metadata?

**Exit Criteria:**

- You can chunk documents using at least 2 strategies.
- You can explain the trade-off between chunk size and retrieval precision.
- You can propagate metadata through the chunking process.

**Next Step:** Generate embeddings for your chunks and build the vector index.

---

### Unit 11.8 — Embedding Generation

**What is it?**  
Transforming chunks into embedding vectors and building a searchable index.

**Why does it matters?**  
This is the bridge between text and vector search. You have chunks; now you need vectors. The quality and efficiency of embedding generation directly affects index build time, cost, and retrieval quality.

**Why learn it here?**  
You understand embeddings (11.3) and chunks (11.7). Now you combine them: embed each chunk and store it in a vector database.

**Prerequisites:** Unit 11.3 (Embeddings), Unit 11.7 (Chunking), Unit 11.5 (Vector Databases)

**Mental Model:**  
Embedding generation is a factory assembly line: chunks go in one end, vectors come out the other, and the vector database stores them for fast lookup.

**Core Concepts:**

- Batch embedding generation
- Embedding dimension and token limits
- Cost estimation (tokens × price per million)
- Index build time and memory
- Incremental indexing (adding new documents)
- Embedding model versioning

**How It Works:**

```text
Chunks → Batch → Embedding Model → Vectors + Metadata → Vector DB Index
```

**Syntax & Implementation:**

```python
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

chunks = [
    "The return policy allows returns within 30 days.",
    "Items must be in original packaging.",
    "Contact support@example.com to initiate a return.",
]
chunk_ids = ["chunk_0", "chunk_1", "chunk_2"]
metadata = [
    {"source": "policy.pdf", "page": 1},
    {"source": "policy.pdf", "page": 1},
    {"source": "policy.pdf", "page": 2},
]

# Generate embeddings
embeddings = model.encode(chunks).tolist()

# Store in vector DB
client = chromadb.Client()
collection = client.create_collection("returns_policy")
collection.add(
    documents=chunks,
    embeddings=embeddings,
    ids=chunk_ids,
    metadatas=metadata
)
print(f"Indexed {collection.count()} chunks")
```

**Real-World Example:**  
A company indexes 50,000 support articles. Each article produces ~5 chunks. Total: 250,000 chunks. Using `text-embedding-3-small`, embedding generation takes ~20 minutes and costs ~$0.03. The index is persisted and updated nightly as articles change.

**Common Mistakes:**

- Embedding one chunk at a time instead of batching (slow)
- Not tracking which model version was used
- Exceeding token limits for long chunks
- Not handling API rate limits for hosted embedding services
- Not persisting the index after building

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Embedding generation is very slow | Not batching, or model too large | Check batch size | Use batch encoding, smaller model |
| API rate limit errors | Too many requests per minute | Check API logs | Add rate limiting, retry with backoff |
| Token count errors | Chunks exceed model limit | Tokenize chunks | Truncate or split long chunks |
| Index not found after restart | Not persisted | Check storage path | Persist to disk after building |

**Best Practices:**

- Batch embeddings (100-500 chunks per batch)
- Record embedding model name and version with the index
- Estimate cost before bulk embedding
- Test with a small subset before embedding entire corpus
- Persist index after building

**Hands-On Practice:**

1. **Basic:** Embed 5 chunks and store in Chroma.
2. **Guided:** Embed 100 chunks, measure time and check index count.
3. **Independent:** Build an incremental embedding pipeline that adds new chunks without re-embedding everything.
4. **Realistic:** Compare embedding generation time across 3 models.
5. **Challenge:** Estimate embedding cost for a 10,000-document corpus.

**Knowledge Check:**

- Why should you batch embedding generation?
- How do you estimate embedding cost?
- What happens if you change the embedding model after building an index?
- How do you handle incremental document updates?

**Exit Criteria:**

- You can generate embeddings and build a searchable index.
- You can estimate cost and time for embedding generation.
- You can handle incremental updates to an index.

**Next Step:** Retrieve relevant chunks for a user query.

---

### Unit 11.9 — Retrieval

**What is it?**  
The process of finding the most relevant chunks for a user query using vector search, keyword search, or hybrid approaches.

**Why does it matter?**  
Retrieval is the core of RAG. If retrieval fails — if the right chunks are not found — the LLM generates answers from irrelevant or missing context. Retrieval quality is the ceiling of RAG quality.

**Why learn it here?**  
You have embeddings and an index. Now you need to query it. This unit teaches the retrieval step and its parameters.

**Prerequisites:** Unit 11.4 (Vector Similarity), Unit 11.5 (Vector Databases), Unit 11.8 (Embedding Generation)

**Mental Model:**  
Retrieval is a librarian fetching the most relevant books for your question. The quality depends on how well the librarian understands your question and how well the library is organized.

**Core Concepts:**

- Query embedding (encode the query the same way as documents)
- Top-k retrieval
- Similarity threshold (minimum score)
- Recall@k and precision@k
- Metadata filtering
- Query expansion / HyDE

**How It Works:**

```text
User Query → Embed Query → Search Vector DB → Return Top-k Chunks
```

**Syntax & Implementation:**

```python
from sentence_transformers import SentenceTransformer
import chromadb

model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.Client()
collection = client.get_collection("returns_policy")

query = "How do I get a refund?"
query_embedding = model.encode([query]).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)

for doc, score, meta in zip(
    results["documents"][0],
    results["distances"][0],
    results["metadatas"][0]
):
    print(f"Score: {score:.3f} | Source: {meta['source']}")
    print(f"  {doc[:100]}...\n")
```

**Real-World Example:**  
A customer asks "Can I return an opened item?" The system embeds the query, retrieves the top-5 chunks by cosine similarity, and finds that chunk 2 discusses "unopened items" and chunk 4 discusses "defective items." The LLM uses both to answer accurately.

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Pure vector search | Semantic understanding needed | Exact keyword matching needed |
| Pure keyword search | Exact matches (codes, IDs, names) | Synonym/paraphrase queries |
| Hybrid (vector + keyword) | Best of both worlds | Simple use case, prototyping |

**Common Mistakes:**

- Using a different embedding model for queries than for indexing
- Retrieving too few chunks (low recall) or too many (noise)
- Not filtering by metadata (e.g., searching irrelevant document types)
- Ignoring retrieval latency

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Relevant doc not retrieved | k too small or wrong embedding model | Test with known queries | Increase k, verify model consistency |
| Too many irrelevant results | k too large or weak embedding model | Check precision@k | Reduce k, add reranking |
| Query is slow | Large index without proper index | Measure query time | Build proper ANN index |
| Metadata filter returns empty | Filter values don't match | Print metadata values | Debug filter syntax and values |

**Best Practices:**

- Use the same embedding model for indexing and querying
- Start with k=5 and tune based on evaluation
- Log retrieved chunks with scores for debugging
- Combine with metadata filtering when applicable
- Evaluate retrieval before evaluating generation

**Hands-On Practice:**

1. **Basic:** Query the index from 11.8 with 3 different queries.
2. **Guided:** Vary k (1, 3, 5, 10) and observe how results change.
3. **Independent:** Add metadata filtering and combine with vector search.
4. **Realistic:** Build a test set of 20 queries with known relevant docs. Measure recall@5.
5. **Challenge:** Implement hybrid search combining BM25 and vector results.

**Knowledge Check:**

- Why must you use the same embedding model for queries and indexing?
- What is the effect of increasing k on retrieval quality?
- How does metadata filtering improve retrieval?
- What is recall@k and why does it matter?

**Exit Criteria:**

- You can retrieve relevant chunks for a query.
- You can evaluate retrieval quality with recall@k and precision@k.
- You can combine vector search with metadata filtering.

**Next Step:** Improve retrieval with reranking.

---

### Unit 11.10 — Reranking

**What is it?**  
A second-stage retrieval step that takes initial retrieval results and reorders them using a more powerful (but slower) model.

**Why does it matter?**  
Initial retrieval (vector or keyword) is fast but approximate. Reranking uses a cross-encoder that reads the query and each document together, producing much more accurate relevance scores. This significantly improves precision in the top results.

**Why learn it here?**  
Retrieval (11.9) gets you候选结果. Reranking refines them. This is a key technique for moving from naive to production-quality RAG.

**Prerequisites:** Unit 11.9 (Retrieval)

**Mental Model:**  
Reranking is like having a senior librarian review the initial search results and reorder them based on deeper reading. The first search finds candidates; the senior librarian picks the best ones.

**Core Concepts:**

- Cross-encoder vs bi-encoder
- Two-stage retrieval: retrieve then rerank
- Reranking models (Cohere, BGE-reranker, cross-encoder/ms-marco)
- Number of candidates to rerank
- Latency vs quality trade-off

**How It Works:**

```text
Query → Retrieve top-50 (fast, approximate) → Rerank top-50 → Return top-5 (slow, precise)
```

**Syntax & Implementation:**

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

query = "How do I return a product?"
candidates = [
    "The return policy allows returns within 30 days.",
    "Shipping takes 5-7 business days.",
    "Items must be in original packaging for return.",
    "Contact support to initiate a return.",
    "We offer free standard shipping.",
]

# Rerank
pairs = [(query, doc) for doc in candidates]
scores = reranker.predict(pairs)

ranked = sorted(zip(candidates, scores), key=lambda x: -x[1])
for doc, score in ranked:
    print(f"Score: {score:.3f} | {doc[:60]}")
```

**Real-World Example:**  
A RAG system retrieves 50 candidates using vector search (10ms). It then reranks all 50 using a cross-encoder (200ms). The top-5 from reranking are much more relevant than the top-5 from vector search alone. Total latency: ~210ms — acceptable for most applications.

**Alternatives:**

| Reranker | Use When | Trade-off |
|---|---|---|
| Cross-encoder (local) | Low latency, data stays local | Model size, CPU/GPU needed |
| Cohere rerank | Quick setup, cloud-based | API cost, data sent to cloud |
| ColBERT | Token-level reranking, high quality | More complex to set up |
| LLM-based reranking | Best quality, reasoning ability | High cost and latency |

**Common Mistakes:**

- Reranking too many candidates (slow, diminishing returns)
- Not using reranking at all (leaving precision on the table)
- Using the same model for retrieval and reranking (wastes compute)
- Ignoring the latency impact on user experience

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Reranking is slow | Too many candidates | Measure per-candidate time | Reduce candidates to 20-50 |
| No quality improvement | Cross-encoder not suited for domain | Compare reranked vs original | Try different reranker |
| Memory errors | Too many candidates or model too large | Check memory usage | Reduce batch size or candidates |

**Best Practices:**

- Rerank 20-50 candidates (not thousands)
- Use reranking as a second stage, not primary retrieval
- Measure quality improvement vs latency cost
- Consider Cohere for quick setup, local models for data sovereignty

**Hands-On Practice:**

1. **Basic:** Rerank 5 candidates with a cross-encoder.
2. **Guided:** Compare top-5 from vector search vs top-5 from reranking.
3. **Independent:** Build a two-stage pipeline: retrieve 50, rerank to top-5.
4. **Realistic:** Measure latency and quality improvement of reranking.
5. **Challenge:** Test 3 different rerankers on the same query set.

**Knowledge Check:**

- What is the difference between a bi-encoder and a cross-encoder?
- Why is reranking slower than initial retrieval?
- How many candidates should you rerank?
- When is reranking not worth the cost?

**Exit Criteria:**

- You can implement two-stage retrieval with reranking.
- You can explain the quality-latency trade-off.
- You can choose an appropriate reranker.

**Next Step:** Build the prompt context from retrieved chunks.

---

### Unit 11.11 — Context Construction

**What is it?**  
Assembling retrieved chunks into a well-structured prompt that the LLM can use to generate a grounded answer.

**Why does it matters?**  
Even with perfect retrieval, a badly constructed prompt leads to poor answers. Context construction determines how the LLM interprets, prioritizes, and cites the retrieved information.

**Why learn it here?**  
You can retrieve and rerank. Now you need to format the context so the LLM uses it effectively. This is the bridge between retrieval and generation.

**Prerequisites:** Unit 11.9 (Retrieval), Unit 11.10 (Reranking)

**Mental Model:**  
Context construction is like writing a briefing document for an expert: organize the evidence clearly, label sources, and tell the expert exactly what you need from them.

**Core Concepts:**

- Prompt templates for RAG
- Source labeling and citation instructions
- Context window management (fitting chunks within token limits)
- Handling contradictory sources
- Instruction tuning ("answer only from the provided context")
- System prompt vs user prompt design

**How It Works:**

```text
Retrieved Chunks → Format with Sources → Manage Token Budget → Assemble Final Prompt
```

**Syntax & Implementation:**

```python>
def build_context(chunks, max_tokens=3000):
    context_parts = []
    token_count = 0
    for i, chunk in enumerate(chunks):
        source = chunk["metadata"]["source"]
        page = chunk["metadata"].get("page", "?")
        text = chunk["text"]
        part = f"[Source {i+1}: {source}, page {page}]\n{text}"
        token_estimate = len(part.split()) * 1.3  # rough token estimate
        if token_count + token_estimate > max_tokens:
            break
        context_parts.append(part)
        token_count += token_estimate
    return "\n\n".join(context_parts)

def build_rag_prompt(query, context):
    return f"""You are a helpful assistant. Answer the question using ONLY the provided context.
If the context does not contain enough information, say "I don't have enough information to answer this."
Cite your sources using [Source N] notation.

Context:
{context}

Question: {query}

Answer:"""
```

**Real-World Example:**  
A customer support RAG system retrieves 5 chunks. The context construction step formats them with source labels, checks total token count, and drops the least relevant chunk if over budget. The system prompt instructs: "If multiple sources conflict, note the discrepancy and cite both."

**Common Mistakes:**

- Dumping all chunks into the prompt without structure
- Not citing sources (user cannot verify claims)
- Exceeding the LLM's context window
- Including contradictory information without noting it
- Not telling the model to abstain when context is insufficient

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| LLM ignores context | Prompt doesn't instruct grounding | Test with/without context | Add explicit grounding instructions |
| Answer cites wrong source | Source labels unclear | Inspect prompt format | Improve source labeling |
| Token limit exceeded | Too many chunks or long chunks | Count tokens | Reduce k or truncate chunks |
| LLM makes up info despite context | Model not instructed to abstain | Test with irrelevant context | Add "say I don't know" instruction |

**Best Practices:**

- Always label sources with consistent numbering
- Include explicit grounding instructions
- Set a token budget and manage context size
- Handle contradictions explicitly in the prompt
- Test context construction with edge cases (no relevant docs, contradictory docs)

**Hands-On Practice:**

1. **Basic:** Format 3 chunks with source labels.
2. **Guided:** Build a RAG prompt with grounding instructions and test with an LLM.
3. **Independent:** Implement token budget management for context construction.
4. **Realistic:** Test with contradictory sources and observe LLM behavior.
5. **Challenge:** Design a context construction system that handles 1, 10, and 100 chunks gracefully.

**Knowledge Check:**

- Why must you tell the LLM to answer from context only?
- How do you handle token limits when many chunks are retrieved?
- What happens when retrieved sources contradict each other?
- How does source labeling affect answer quality?

**Exit Criteria:**

- You can construct well-structured RAG prompts.
- You can manage context within token limits.
- You can instruct the LLM to cite sources and abstain when appropriate.

**Next Step:** Generate grounded answers.

---

### Unit 11.12 — Generation & Grounding

**What is it?**  
The final step: using the LLM to generate an answer that is faithful to the retrieved context.

**Why does it matters?**  
This is where RAG produces its output. The generation step must balance helpfulness with faithfulness — answering the question while only using information from the provided context.

**Why learn it here?**  
All the retrieval and context work leads to this moment. Understanding generation in RAG means understanding how to control LLM output for accuracy and citation.

**Prerequisites:** Unit 11.11 (Context Construction), Phase 10 (LLMs)

**Mental Model:**  
Generation in RAG is like a well-briefed consultant: they read the evidence, synthesize an answer, and cite their sources. They don't add information from memory.

**Core Concepts:**

- Temperature and sampling parameters for RAG
- Groundedness vs helpfulness trade-off
- Citation generation
- Abstention (saying "I don't know")
- Streaming responses
- Post-processing (extracting citations, formatting)

**How It Works:**

```text
Context + Query → LLM → Answer + Citations
```

**Syntax & Implementation:**

```python
from openai import OpenAI

client = OpenAI()

def generate_answer(query, context, model="gpt-4o-mini"):
    prompt = build_rag_prompt(query, context)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You answer questions from provided context only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,  # deterministic for RAG
        max_tokens=500
    )
    return response.choices[0].message.content

# Full RAG pipeline
chunks = retrieve(query, k=5)
context = build_context(chunks)
answer = generate_answer(query, context)
print(answer)
```

**Real-World Example:**  
A legal research assistant generates answers with inline citations. For each claim, it includes [Source 1] or [Source 2]. If the context doesn't cover the question, it responds: "Based on the available documents, I don't have sufficient information to answer this question about [topic]."

**Decision Guidance:**

| Parameter | RAG Setting | Why |
|---|---|---|
| Temperature | 0.0 - 0.2 | Deterministic, faithful answers |
| Top-p | 0.9 - 1.0 | Keep some diversity for complex questions |
| Max tokens | 500 - 1000 | Enough for detailed answers, not too long |
| Model | gpt-4o-mini or equivalent | Balance quality and cost |

**Common Mistakes:**

- Using high temperature (creative but unfaithful answers)
- Not instructing the model to cite sources
- Allowing the model to use parametric knowledge instead of context
- Not handling the "I don't know" case
- Post-processing that strips citations

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Answer is creative but wrong | High temperature | Check temperature setting | Set temperature to 0.0 |
| Model uses external knowledge | Prompt doesn't restrict to context | Test with irrelevant context | Strengthen grounding instructions |
| Citations are fabricated | Model invents source references | Verify citations against chunks | Improve source labeling format |
| Answer is too brief | Max tokens too low | Check token limit | Increase max_tokens |

**Best Practices:**

- Use temperature 0.0 for RAG (deterministic answers)
- Always instruct: "Answer only from the provided context"
- Include abstention instructions: "If context is insufficient, say so"
- Verify citations programmatically against retrieved chunks
- Log both the prompt and the answer for auditing

**Hands-On Practice:**

1. **Basic:** Generate an answer from context using an LLM.
2. **Guided:** Compare answers with temperature 0.0 vs 0.8.
3. **Independent:** Build a full pipeline: retrieve → construct context → generate → extract citations.
4. **Realistic:** Test with queries where context is insufficient and verify abstention.
5. **Challenge:** Implement citation verification (check that cited sources actually exist in context).

**Knowledge Check:**

- Why is temperature 0.0 recommended for RAG?
- How do you prevent the model from using parametric knowledge?
- What is abstention and why is it important?
- How do you verify that citations are real?

**Exit Criteria:**

- You can generate grounded answers from retrieved context.
- You can control generation parameters for RAG quality.
- You can implement citation and abstention behavior.

**Next Step:** Evaluate the quality of your RAG system.

---

### Unit 11.13 — RAG Evaluation

**What is it?**  
Measuring the quality of both retrieval and generation in a RAG system using standardized metrics.

**Why does it matters?**  
You cannot improve what you cannot measure. RAG evaluation separates retrieval quality from generation quality, allowing you to identify which part of the pipeline needs improvement.

**Why learn it here?**  
You have built a complete RAG pipeline (11.1-11.12). Now you need to measure its quality systematically. This is the foundation for iteration and improvement.

**Prerequisites:** Units 11.1-11.12 (complete RAG pipeline)

**Mental Model:**  
RAG evaluation is like a two-part exam: (1) Did the system find the right books? (2) Did it use those books to write a good answer? You grade each part separately.

**Core Concepts:**

- Retrieval metrics: recall@k, precision@k, MRR, NDCG
- Generation metrics: faithfulness, answer relevance, context relevance
- Evaluation datasets: queries + ground truth answers + ground truth documents
- RAGAS framework
- Human evaluation vs automated evaluation
- Latency and cost metrics

**How It Works:**

```text
Evaluation Dataset → Run RAG Pipeline → Measure Retrieval → Measure Generation → Report
```

**Syntax & Implementation:**

```python
# Simple evaluation framework
def evaluate_retrieval(queries, ground_truth_docs, retriever, k=5):
    recalls = []
    precisions = []
    for query, true_docs in zip(queries, ground_truth_docs):
        retrieved = retriever.search(query, k=k)
        retrieved_ids = {doc["id"] for doc in retrieved}
        true_ids = set(true_docs)
        recall = len(retrieved_ids & true_ids) / len(true_ids)
        precision = len(retrieved_ids & true_ids) / k
        recalls.append(recall)
        precisions.append(precision)
    return {
        "recall@k": sum(recalls) / len(recalls),
        "precision@k": sum(precisions) / len(precisions)
    }

# Test set
queries = ["How do I return a product?", "What is the shipping time?"]
ground_truth = [
    ["doc_1", "doc_3"],  # docs about returns
    ["doc_2"],           # docs about shipping
]

results = evaluate_retrieval(queries, ground_truth, retriever, k=5)
print(f"Recall@5: {results['recall@k']:.2f}")
print(f"Precision@5: {results['precision@k']:.2f}")
```

**Decision Guidance:**

| Metric | What It Measures | Use When |
|---|---|---|
| Recall@k | Did we retrieve all relevant docs? | Source completeness matters |
| Precision@k | Are retrieved docs relevant? | Noise reduction matters |
| MRR | How high is the first relevant result? | Users only read the top result |
| Faithfulness | Is the answer grounded in context? | Hallucination prevention |
| Answer relevance | Does the answer address the question? | User satisfaction |

**Real-World Example:**  
A company evaluates its RAG system quarterly. They maintain 500 test queries with ground truth answers. Each quarter they measure: recall@5 (target: >0.85), faithfulness (target: >0.90), and answer relevance (target: >0.80). When recall drops, they investigate chunking. When faithfulness drops, they review prompts.

**Common Mistakes:**

- Only measuring answer quality, not retrieval quality
- Not having a ground truth evaluation set
- Evaluating on easy queries only
- Not tracking latency and cost alongside quality
- Using the same data for evaluation and testing

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Recall is low | Retrieval is missing relevant docs | Inspect top-k results | Increase k, improve chunking/embedding |
| Precision is low | Too many irrelevant results | Check retrieved docs | Add reranking, improve metadata filtering |
| Faithfulness is low | LLM is hallucinating despite context | Compare answer to context | Strengthen grounding instructions |
| Evaluation is slow | Large test set | Measure eval time | Sample test set for faster iteration |

**Best Practices:**

- Create a ground truth evaluation set early (even 50 queries helps)
- Evaluate retrieval and generation separately
- Track metrics over time (not just absolute values)
- Include latency and cost in evaluation reports
- Test with adversarial queries (unanswerable, ambiguous, contradictory)

**Hands-On Practice:**

1. **Basic:** Compute recall@3 for 5 queries.
2. **Guided:** Build a test set of 20 queries with ground truth answers.
3. **Independent:** Evaluate both retrieval and generation quality.
4. **Realistic:** Compare two retrieval strategies (keyword vs vector) on the same test set.
5. **Challenge:** Use RAGAS to compute faithfulness and answer relevance.

**Knowledge Check:**

- What is the difference between recall and precision in retrieval?
- Why must you evaluate retrieval and generation separately?
- What makes a good evaluation dataset?
- How does evaluation inform pipeline improvements?

**Exit Criteria:**

- You can compute retrieval and generation metrics.
- You can build and maintain an evaluation dataset.
- You can use evaluation results to guide pipeline improvements.

**Next Step:** Build a complete naive RAG system end-to-end.

---

### Unit 11.14 — Naive RAG (Build)

**What is it?**  
A complete, end-to-end RAG system using basic components: simple chunking, basic embedding, vector search, and direct generation.

**Why does it matters?**  
Building a naive RAG system from scratch solidifies all the concepts from 11.1-11.13 into a working system. It also establishes a baseline you can improve.

**Why learn it here?**  
You have learned every component. Now you assemble them. This is the integration unit — the most important learning moment in the phase.

**Prerequisites:** Units 11.1-11.13

**Mental Model:**  
Naive RAG is the simplest possible pipeline: documents in, answers out. It works surprisingly well for many use cases and serves as the baseline for all improvements.

**Core Concepts:**

- End-to-end pipeline assembly
- Configuration management
- Error handling at each stage
- Baseline establishment
- Minimum viable RAG

**Syntax & Implementation:**

```python
from sentence_transformers import SentenceTransformer
import chromadb

class NaiveRAG:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.Client()
        self.collection = self.client.create_collection("documents")

    def ingest(self, documents, chunk_size=500, overlap=50):
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=overlap
        )
        all_chunks, all_ids, all_metas = [], [], []
        for i, doc in enumerate(documents):
            chunks = splitter.split_text(doc["text"])
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"doc{i}_chunk{j}")
                all_metas.append({"source": doc.get("source", f"doc{i}")})

        embeddings = self.model.encode(all_chunks).tolist()
        self.collection.add(
            documents=all_chunks, embeddings=embeddings,
            ids=all_ids, metadatas=all_metas
        )
        return len(all_chunks)

    def retrieve(self, query, k=5):
        q_emb = self.model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=q_emb, n_results=k)
        return results

    def generate(self, query, k=5):
        results = self.retrieve(query, k)
        context = "\n\n".join([
            f"[{meta['source']}]: {doc}"
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ])
        prompt = f"""Answer using ONLY the context below.
Context: {context}
Question: {query}
Answer:"""
        return prompt  # Send to LLM

# Usage
rag = NaiveRAG()
rag.ingest([{"text": "Policy text...", "source": "policy.pdf"}])
prompt = rag.generate("How do I return?")
```

**Real-World Example:**  
A startup builds a naive RAG system for internal docs in one afternoon. It handles 80% of queries well. They iterate from there: better chunking, reranking, hybrid search. The naive system was the foundation.

**Evaluation Criteria:**

| Metric | Target | Notes |
|---|---|---|
| Recall@5 | > 0.70 | Baseline — improve with advanced techniques |
| End-to-end latency | < 5 seconds | For interactive use |
| Answer relevance | > 0.70 | Users find answers helpful |
| Code runs without errors | 100% | Basic requirement |

**Common Mistakes:**

- Skipping evaluation (building without measuring)
- Over-engineering before establishing a baseline
- Not handling edge cases (empty documents, failed queries)
- Not persisting the index

**Hands-On Practice:**

1. **Basic:** Implement the NaiveRAG class above.
2. **Guided:** Ingest 10 documents, test with 5 queries.
3. **Independent:** Build from scratch without the template.
4. **Realistic:** Ingest a real document set (company docs, Wikipedia subset).
5. **Challenge:** Add error handling, logging, and a simple evaluation.

**Knowledge Check:**

- What are the minimum components of a RAG system?
- Why is a naive baseline important?
- What would you improve first if retrieval quality is poor?
- How do you measure if the naive system is "good enough"?

**Exit Criteria:**

- You can build a complete RAG system from scratch.
- You can measure its retrieval and generation quality.
- You have a working baseline for improvement.

**Next Step:** Improve with advanced techniques.

---

### Unit 11.15 — Advanced RAG

**What is it?**  
Techniques that go beyond naive RAG: hybrid search, metadata filtering, multi-query retrieval, parent-child chunking, and query transformation.

**Why does it matters?**  
Naive RAG works for simple cases. Production systems need hybrid search (combining keyword and semantic), metadata filtering (narrowing by document type), and query expansion (reformulating queries for better recall).

**Why learn it here?**  
You have a working baseline. Now you need to know what improvements are available and when each one helps.

**Prerequisites:** Unit 11.14 (Naive RAG)

**Mental Model:**  
Advanced RAG is like upgrading a basic search engine: combine multiple search methods, use metadata to narrow results, and rewrite queries to find what the user really means.

**Core Concepts:**

- Hybrid search (BM25 + vector, reciprocal rank fusion)
- Metadata filtering (by date, source, category)
- Multi-query retrieval (expand one query into several)
- Parent-child chunking (small chunks for retrieval, large chunks for context)
- Query transformation (HyDE, sub-questions, step-back prompting)
- Sentence window retrieval

**How It Works:**

```text
Hybrid: BM25 score + Vector score → Fusion → Top-k
Multi-query: Original query → N reformulated queries → Merge results
Parent-child: Small chunk retrieved → Return parent chunk for context
```

**Syntax & Implementation:**

```python
def hybrid_search(query, bm25_index, vector_collection, k=5, alpha=0.5):
    # BM25 search
    bm25_scores = bm25_index.search(query, k=k*2)
    # Vector search
    vector_results = vector_collection.query(
        query_embeddings=encode(query), n_results=k*2
    )
    # Reciprocal Rank Fusion
    fused = {}
    for rank, (doc_id, _) in enumerate(bm25_scores):
        fused[doc_id] = fused.get(doc_id, 0) + alpha / (rank + 1)
    for rank, doc_id in enumerate(vector_results["ids"][0]):
        fused[doc_id] = fused.get(doc_id, 0) + (1-alpha) / (rank + 1)
    ranked = sorted(fused.items(), key=lambda x: -x[1])
    return ranked[:k]

def multi_query_retrieval(query, retriever, llm, n=3):
    expansion_prompt = f"Generate {n} different search queries for: {query}"
    queries = llm.generate(expansion_prompt).split("\n")
    all_results = []
    for q in queries:
        all_results.extend(retriever.search(q, k=3))
    return deduplicate_and_rank(all_results, k=5)
```

**Alternatives:**

| Technique | Use When | Avoid When | Complexity |
|---|---|---|---|
| Hybrid search | Mixed exact/semantic queries | Simple use case | Medium |
| Metadata filtering | Documents have useful categories | No useful metadata available | Low |
| Multi-query | Queries are ambiguous | Queries are precise | Medium |
| Parent-child | Small chunks needed for precision | Simple documents | High |
| HyDE | Query-document gap is large | Queries already match document style | Medium |

**Common Mistakes:**

- Applying all techniques at once (hard to isolate what helps)
- Not measuring improvement over naive baseline
- Hybrid search weights not tuned for the dataset
- Multi-query generating redundant queries

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Hybrid search is worse than vector alone | Fusion weights not tuned | Test different alpha values | Tune alpha on evaluation set |
| Multi-query returns same results | Queries are too similar | Inspect generated queries | Improve query diversity prompt |
| Metadata filter removes relevant docs | Filter too restrictive | Check filter values | Broaden filter criteria |
| Parent-child is slow | Retrieving full parent chunks | Measure chunk sizes | Limit parent chunk size |

**Best Practices:**

- Add one technique at a time and measure improvement
- Keep the naive baseline for comparison
- Tune fusion weights on an evaluation set
- Log which techniques contribute to each answer

**Hands-On Practice:**

1. **Basic:** Implement hybrid search with reciprocal rank fusion.
2. **Guided:** Add metadata filtering to your retrieval pipeline.
3. **Independent:** Implement multi-query retrieval.
4. **Realistic:** Compare naive vs advanced RAG on 30 test queries.
5. **Challenge:** Combine hybrid search, metadata filtering, and reranking.

**Knowledge Check:**

- When is hybrid search better than pure vector search?
- How does multi-query retrieval improve recall?
- What is parent-child chunking and when is it useful?
- How do you measure if an advanced technique actually helps?

**Exit Criteria:**

- You can implement at least 2 advanced RAG techniques.
- You can measure improvement over the naive baseline.
- You can choose appropriate techniques for a use case.

**Next Step:** Learn agentic RAG — letting the agent decide when and how to retrieve.

---

### Unit 11.16 — Agentic RAG

**What is it?**  
A RAG system where an LLM agent decides when to retrieve, what to retrieve, and how to use the results — rather than always retrieving for every query.

**Why does it matters?**  
Not every query needs retrieval. Some questions are factual (retrieve once), some are multi-hop (retrieve, reason, retrieve again), and some don't need retrieval at all. Agentic RAG adapts the retrieval strategy to the query.

**Why learn it here?**  
You have built static RAG pipelines. Agentic RAG introduces dynamic control: the agent decides the retrieval strategy. This bridges RAG and agents (Phase 14).

**Prerequisites:** Unit 11.15 (Advanced RAG), Phase 10 (LLMs)

**Mental Model:**  
Agentic RAG is like a researcher who decides: "Do I need to look this up?" If yes, "Where should I look?" After looking, "Do I need more information?" They don't always go to the library for every question.

**Core Concepts:**

- Retrieval decision (retrieve or not)
- Adaptive retrieval (single-hop vs multi-hop)
- Tool-augmented retrieval (use different tools for different queries)
- Self-RAG (self-reflective retrieval)
- Corrective RAG (CRAG)
- Iterative retrieval (retrieve → reason → retrieve again)

**How It Works:**

```text
User Query → Agent decides: retrieve? → If yes: which tool?
                                      → Retrieve
                                      → Agent evaluates: sufficient?
                                      → If no: retrieve again or use different tool
                                      → Generate answer
```

**Syntax & Implementation:**

```python
from openai import OpenAI

client = OpenAI()

def agentic_rag(query, retriever, llm):
    # Step 1: Decide if retrieval is needed
    decision_prompt = f"""Should this query use document retrieval?
Query: {query}
Answer with RETRIEVE or NO_RETRIEVE."""

    decision = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": decision_prompt}],
        temperature=0.0
    ).choices[0].message.content.strip()

    if "NO_RETRIEVE" in decision:
        return llm.generate(query)

    # Step 2: Retrieve
    context = retriever.search(query, k=5)

    # Step 3: Evaluate sufficiency
    eval_prompt = f"""Is this context sufficient to answer the question?
Context: {context}
Question: {query}
Answer with SUFFICIENT or INSUFFICIENT."""

    evaluation = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": eval_prompt}],
        temperature=0.0
    ).choices[0].message.content.strip()

    if "INSUFFICIENT" in evaluation:
        # Reformulate and retrieve again
        reformulated = reformulate_query(query, context)
        context = retriever.search(reformulated, k=5)

    return llm.generate_with_context(query, context)
```

**Real-World Example:**  
A customer support agent receives: "What's your return policy?" (retrieve), "Thanks!" (no retrieval needed), "How does your return policy compare to Amazon's?" (retrieve + external knowledge). The agent routes each query differently.

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Fixed retrieval (always retrieve) | Simple, predictable queries | Many queries don't need retrieval |
| Self-RAG | Quality is critical, cost is acceptable | Low-latency requirements |
| CRAG | Retrieval quality is variable | Retrieval is consistently good |
| Multi-step retrieval | Complex questions need multiple sources | Simple factual queries |

**Common Mistakes:**

- Over-engineering when naive RAG works fine
- Agent adds latency for every query (including ones that don't need retrieval)
- Not evaluating whether agent decisions improve quality
- Agent loops indefinitely without retrieval

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Agent never retrieves | Decision threshold too high | Log decisions | Adjust decision prompt |
| Agent always retrieves | Decision prompt too aggressive | Log decisions | Add NO_RETRIEVE examples |
| Agent loops | Insufficient check has no exit | Add step limit | Limit iterations to 3 |
| Latency is too high | Agent makes too many LLM calls | Trace call count | Cache decisions, reduce steps |

**Best Practices:**

- Start with fixed retrieval, add agentic behavior only when needed
- Log agent decisions for debugging
- Set hard limits on retrieval iterations
- Evaluate whether agent decisions actually improve quality
- Consider latency cost of each decision

**Hands-On Practice:**

1. **Basic:** Implement a retrieve-or-not decision for 10 queries.
2. **Guided:** Add a sufficiency check after retrieval.
3. **Independent:** Implement multi-step retrieval for complex queries.
4. **Realistic:** Compare fixed retrieval vs agentic RAG on 30 queries.
5. **Challenge:** Build a self-RAG system with reflection.

**Knowledge Check:**

- When is agentic RAG better than fixed retrieval?
- What decisions does an agentic RAG system make?
- How do you prevent agent loops?
- How do you evaluate whether agent decisions help?

**Exit Criteria:**

- You can implement retrieval decision logic.
- You can build a multi-step retrieval pipeline.
- You can evaluate whether agentic behavior improves quality.

**Next Step:** Synthesize all RAG knowledge into a comprehensive project.

---

### Unit 11.17 — RAG Synthesis & Review

**What is it?**  
A cumulative integration unit combining all RAG concepts: ingestion, chunking, embedding, retrieval, reranking, context construction, generation, evaluation, and advanced techniques.

**Why does it matters?**  
Knowing individual components is not enough. You must build a production-quality RAG system independently, evaluate it rigorously, and justify your design decisions.

**Prerequisites:** Units 11.1-11.16

---

## Mini Project — Production RAG Application

**Objective:** Build a document Q&A system that answers questions with grounded citations and can handle real-world edge cases.

**Problem Statement:**  
Organizations have thousands of documents (policies, manuals, reports, FAQs). Employees and customers need fast, accurate answers grounded in those documents. Build a RAG system that retrieves relevant documents, generates accurate answers with citations, and handles edge cases like unanswerable questions.

**Requirements:**

- Ingest documents from at least 2 formats (PDF + TXT or DOCX)
- Chunk with metadata propagation (source, page/section)
- Embed using a pre-trained model
- Build a vector index (Chroma or similar)
- Implement retrieval with configurable k
- Implement reranking (any reranker)
- Build context with source labeling and token budget management
- Generate answers with grounding instructions and citation support
- Handle unanswerable questions (abstention)
- Build an evaluation set (minimum 20 queries)
- Measure retrieval recall@5 and precision@5
- Measure answer faithfulness
- Include error handling and logging

**Suggested Architecture:**

```text
Documents → Ingest (PDF/TXT) → Chunk + Metadata → Embed → Vector Index
                                                                   ↓
User Query → Retrieve (top-50) → Rerank (top-5) → Context Builder → LLM → Answer + Citations
                                                                   ↓
                                                          Evaluation Pipeline
```

**Milestones:**

1. Ingestion + chunking working (Day 1)
2. Embedding + index built (Day 1)
3. Retrieval working with evaluation metrics (Day 2)
4. Reranking added (Day 2)
5. Context construction + generation (Day 3)
6. Evaluation set created, full pipeline tested (Day 3)
7. Edge cases handled, logging added (Day 4)
8. README and documentation (Day 4)

**Expected Output:**

- Working RAG pipeline (Python script or notebook)
- Vector index with embedded documents
- Evaluation report (retrieval + generation metrics)
- Error analysis (10 failure cases with root causes)
- README.md explaining architecture and decisions

**Evaluation Criteria:**

| Criterion | Weight | Target |
|---|---|---|
| Retrieval recall@5 | 25% | > 0.75 |
| Answer faithfulness | 25% | > 0.80 |
| Citation correctness | 15% | > 0.85 |
| Code quality and structure | 15% | Clean, documented, tested |
| Edge case handling | 10% | Handles 5+ failure modes |
| Documentation | 10% | Clear README, architecture explained |

**Failure Cases to Test:**

- Query with no relevant documents (should abstain)
- Query requiring information from multiple documents
- Contradictory sources
- Very long document vs very short document
- Query in different language than documents
- Query asking about specific numbers/dates
- Ambiguous query (could match multiple topics)

**Advanced Extensions:**

- Hybrid search (BM25 + vector)
- Multi-query retrieval
- Incremental indexing (add/update documents)
- Access control (filter by user permissions)
- Feedback loop (user ratings improve ranking)
- Dashboard for monitoring query quality
- Streaming responses
- Cost tracking per query

**Knowledge Check:**

- Why is retrieval quality the ceiling of RAG quality?
- How do you decide between naive and advanced RAG?
- What metrics matter most for a production RAG system?
- How would you handle a sudden drop in answer quality?
- When should you use agentic RAG vs fixed retrieval?
- How do you balance latency, cost, and quality?

---

## RAG Debugging Playbook

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Answer is wrong but retrieved docs are relevant | Prompt/context construction issue | Inspect final prompt | Improve instructions, cite sources, reduce context noise |
| Retrieved docs are irrelevant | Query, embedding, or chunking issue | Inspect top-k chunks | Rewrite query, adjust chunking, use hybrid search/reranking |
| Correct source exists but is not retrieved | Low recall | Test gold query set | Increase k, improve metadata filters, add reranker |
| Answer invents facts | Poor grounding | Compare answer claims to sources | Require citations and abstention behavior |
| System is slow/expensive | Too many chunks/model calls | Trace latency/cost by component | Cache, reduce k, batch embeddings, use smaller model |
| Answer is too generic | Context too large or noisy | Check context token count | Reduce k, add reranking, improve chunking |
| Duplicate answers for different queries | Chunks are too similar | Check chunk uniqueness | Deduplicate, improve chunking strategy |
| Citations don't match claims | Source labeling is unclear | Verify citation extraction | Improve prompt format for citations |

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Keyword vs Vector search | Exact matches needed (IDs, codes) | Semantic understanding needed | Precision vs recall |
| Fixed chunking vs Semantic chunking | Prototyping, uniform documents | Diverse content, quality critical | Simplicity vs accuracy |
| Naive vs Advanced RAG | Simple use case, quick MVP | Production system, quality matters | Speed to build vs quality |
| Cross-encoder vs Bi-encoder | Reranking (small set) | Initial retrieval (large set) | Quality vs speed |
| RAG vs Fine-tuning | Fresh/private/source-bound knowledge | Style/behavior adaptation | Flexibility vs behavior change |
| Fixed vs Agentic RAG | Predictable query patterns | Variable query complexity | Simplicity vs adaptability |

---

## Phase Review Checklist

- [ ] All 17 units complete
- [ ] Why RAG exists — can explain motivation and alternatives
- [ ] Keyword search — implemented BM25 baseline
- [ ] Embeddings — generated and understood vector representations
- [ ] Vector similarity — compared cosine, dot product, Euclidean
- [ ] Vector databases — stored and queried vectors in Chroma/FAISS
- [ ] Document ingestion — loaded PDF, TXT, DOCX
- [ ] Chunking — implemented and compared chunking strategies
- [ ] Embedding generation — built index with metadata
- [ ] Retrieval — measured recall@k and precision@k
- [ ] Reranking — implemented two-stage retrieval
- [ ] Context construction — built prompts with source citations
- [ ] Generation — produced grounded answers with abstention
- [ ] RAG evaluation — measured retrieval and generation quality separately
- [ ] Naive RAG — built complete end-to-end system
- [ ] Advanced RAG — implemented at least 2 advanced techniques
- [ ] Agentic RAG — implemented retrieval decision logic
- [ ] Mini project completed with evaluation report
- [ ] At least one bad chunking strategy tested and improved
- [ ] Cumulative review passed

## Mastery Check

Without following a tutorial, you should be able to:

1. Explain why RAG exists and when to use it vs alternatives.
2. Implement keyword search (BM25) and vector search.
3. Choose and justify an embedding model.
4. Chunk documents with metadata propagation.
5. Build and query a vector database.
6. Implement two-stage retrieval with reranking.
7. Construct context with source citations and token budget.
8. Generate grounded answers with abstention behavior.
9. Evaluate retrieval and generation quality separately.
10. Build a complete RAG system from scratch.
11. Choose appropriate advanced techniques.
12. Decide when agentic RAG is warranted.
13. Debug common RAG failure modes.
14. Explain trade-offs between RAG approaches.

## Interview / Explain-Back Questions

- **Basic:** What is RAG and why does it exist?
- **Conceptual:** How does retrieval quality affect generation quality?
- **Scenario:** You have 10,000 PDFs and users ask questions. Design the RAG pipeline.
- **Practical:** Debug a RAG system that retrieves irrelevant chunks.
- **Comparison:** When would you use RAG instead of fine-tuning?
- **Advanced:** How would you evaluate a RAG system for a production deployment?
- **Debugging:** A RAG system works for simple queries but fails for complex ones. What do you investigate?
- **Design:** How do you handle contradictory information from different sources?
- **Cost:** How do you balance retrieval quality with latency and cost?
- **Edge cases:** What should the system do when it cannot answer a question?

## Exit Criteria

Move to Phase 12 only when you can:

1. Build a complete RAG system from scratch without a tutorial.
2. Evaluate retrieval and generation quality independently.
3. Choose appropriate chunking, embedding, and retrieval strategies.
4. Debug common RAG failure modes.
5. Explain the trade-offs between RAG approaches.
6. Build a production-style RAG application with citations and evaluation.
