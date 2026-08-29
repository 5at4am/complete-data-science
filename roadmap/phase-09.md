# Phase 09 — Generative AI Foundations

> **Goal:** Master generative AI foundations — language models, tokenization, embeddings, attention, pretraining, inference, sampling, alignment, API usage, prompt engineering, structured output, tool calling, streaming, caching, and multimodal generation.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced
**Priority:** Essential
**Prerequisites:** Phase 08 (Transformers)
**Mastery target:** Level 5 — understand generation mechanics and build robust LLM-powered applications

---

## Why This Phase Exists

Generative AI changed what software can do. Before this phase the learner built models that classify or regress. After this phase the learner can build systems that generate text, structure outputs, call tools, and combine modalities. The goal is not to memorize API calls — it is to understand how generation works so you can debug failures, control quality, manage cost, and choose the right approach for real problems.

Weak understanding of generation mechanics leads to hallucination blindness, wasted tokens, broken prompts, fragile output formats, and production failures that are hard to diagnose. Strong foundations let you build reliable systems on top of probabilistic engines.

### Phase Mental Model

Generative AI does not retrieve truth by default. It predicts likely outputs from patterns learned during training, then your system design constrains, grounds, validates, or rejects those outputs.

```text
Text → tokens → embeddings → transformer layers → probability distribution → sampled token
                                                                            ↓
                         structured output / function calling / streaming ← next token prediction loop
                                                                            ↓
                                              alignment / prompting / constraints shape behavior
```

### What This Phase Prepares For

- LLM application design in Phase 10 (model selection, cost, evaluation)
- RAG systems in Phase 11 (retrieval-augmented generation)
- Framework evaluation in Phases 12–13 (LangChain, LlamaIndex trade-offs)
- Agent design in Phase 14 (tool use, orchestration, autonomy)
- Evaluation methodology throughout remaining phases

---

## Units

### Unit 09.1 — What Is a Language Model?

**What is it?**
A language model is a statistical system trained to predict the next token (word, subword, or character) given a sequence of preceding tokens. Modern large language models (LLMs) are deep neural networks — typically transformers — trained on massive text corpora to learn patterns of human language.

**Why does it matter?**
Everything else in generative AI builds on this: tokenization, embeddings, attention, inference, prompting, and tool calling are all parts of the same generation pipeline. If you do not understand what an LLM is doing at a mechanical level, you cannot debug failures or design reliable systems.

**Why learn it here?**
Phase 08 covered transformers and attention. Now the learner transitions from understanding architecture to understanding what that architecture produces when scaled and applied to generation tasks.

**Prerequisites:** Phase 08 (transformer architecture), basic probability concepts.

**Mental Model:**
An LLM is a very large autocomplete engine. Given "The cat sat on the", it assigns probabilities to every possible next token ("mat" might get 0.4, "floor" 0.2, "table" 0.15, etc.) and selects one. Repeating this process generates full responses.

**Core Concepts:**

- next-token prediction
- conditional probability: P(token | context)
- autoregressive generation (left-to-right)
- model parameters (billions of weights)
- training data and its influence on output
- the difference between retrieval and generation
- open-source vs proprietary models

**How It Works:**

1. Input text is tokenized into discrete units.
2. Tokens are converted to dense vector representations (embeddings).
3. Transformer layers process the sequence using self-attention.
4. The final layer produces a probability distribution over the entire vocabulary for the next token.
5. A decoding strategy selects one token from that distribution.
6. The selected token is appended to the sequence, and the process repeats.

**Syntax & Implementation:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "The meaning of life is"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=20)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

**Simple Example:**

```python
# Conceptual: what the model does at each step
# Input: "I love"
# Model predicts: P("programming" | "I love") = 0.35
#                  P("cats" | "I love") = 0.28
#                  P("this" | "I love") = 0.18
# Sample → "programming"
# New input: "I love programming"
# Model predicts: P("because" | "I love programming") = 0.41
# ... and so on
```

**Real-World Example:**
ChatGPT, Claude, Gemini, and open-source models like Llama all use this mechanism. When you type a message, the model autoregressively generates tokens one at a time until it produces a stop token or hits a length limit.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Treating LLM output as factual retrieval | Generates plausible-sounding but false statements (hallucination) |
| Ignoring training data cutoff | Model cannot know events after its training date |
| Assuming the model "understands" | It predicts statistically likely continuations, not meaning in a human sense |
| Using one model for everything | Different models have different strengths, costs, and contexts |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output is coherent but wrong | Model lacks relevant training data | Check model capabilities and training cutoff | Use RAG or a more recent model |
| Output is nonsensical | Input is malformed or too short | Inspect tokenized input | Provide clearer, longer context |
| Model refuses to answer | Safety alignment or content policy | Try rephrasing the question | Rephrase or use a different model |
| Output quality is poor | Model is too small for the task | Try a larger model or fine-tuned variant | Upgrade model or improve prompting |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Autoregressive LLM | Text generation, conversation, reasoning | You need deterministic, verifiable output only |
| Masked language model (BERT-style) | Classification, embedding, fill-in-the-blank | You need open-ended generation |
| N-gram model | Simple prediction, low compute | You need semantic understanding |
| Retrieval system | You need factual lookup from a corpus | You need creative or synthesized text |

**Best Practices:**

- Start by understanding what the model can and cannot do before trying to make it do more.
- Always check the model's training data cutoff and capabilities documentation.
- Use small experiments to explore model behavior before building production systems.
- Keep a mental separation between what the model generates and what is actually true.

**Hands-On Practice:**

1. Basic: load a pretrained model and generate text from a short prompt.
2. Guided: compare outputs from two different models on the same prompt.
3. Independent: explain why the model generates different outputs each time.
4. Realistic: find a prompt where the model confidently gives a wrong answer and explain why.
5. Challenge: research the training data and architecture of a specific model and summarize its strengths and weaknesses.

**Knowledge Check:**

- What is next-token prediction?
- Why can an LLM produce different outputs for the same input?
- What is the difference between a language model and a search engine?
- Why does model size matter for generation quality?

**Exit Criteria:**

- You can explain what a language model does at a mechanical level.
- You can generate text with a pretrained model and interpret the results.
- You can identify when generation is the right tool vs retrieval or rule-based logic.

**Next Step:** Understand how text becomes tokens that models can process.

---

### Unit 09.2 — Tokens & Tokenization

**What is it?**
Tokenization is the process of converting raw text into discrete units (tokens) that a language model can process. Tokens are typically subwords — pieces of words — not whole words or individual characters.

**Why does it matter?**
Tokenization determines what the model sees as input. It affects cost (API billing is per token), context window usage, model performance on non-English text, and the ability to handle special characters, code, and formatting. Misunderstanding tokenization leads to budget surprises and unexpected behavior.

**Why learn it here?**
After understanding what an LLM is, the next logical step is understanding how text enters the model. Tokenization is the bridge between human-readable text and model-processable numbers.

**Prerequisites:** Unit 09.1 (language model basics).

**Mental Model:**
Tokenization is like taking apart a puzzle. The tokenizer breaks text into the smallest meaningful pieces the model was trained on. Some pieces are whole words ("hello"), some are subwords ("un" + "likely"), and some are single characters or special symbols.

**Core Concepts:**

- tokens vs words vs characters
- subword tokenization (BPE, WordPiece, SentencePiece)
- vocabulary size
- token IDs
- special tokens (BOS, EOS, PAD, UNK, CLS, SEP)
- token boundaries and whitespace handling
- multilingual tokenization challenges

**How It Works:**

Byte Pair Encoding (BPE), the most common method:
1. Start with a vocabulary of individual characters.
2. Find the most frequent pair of adjacent tokens in the corpus.
3. Merge that pair into a new token and add it to the vocabulary.
4. Repeat until the vocabulary reaches the desired size.

This produces a vocabulary where common words are single tokens and rare words are split into multiple subword pieces.

**Syntax & Implementation:**

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")

text = "Tokenization is surprisingly important for LLMs."
tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)

print("Tokens:", tokens)
print("Token IDs:", ids)
print("Vocab size:", tokenizer.vocab_size)
print("Token count:", len(tokens))
```

**Output:**

```
Tokens: ['Token', 'ization', ' is', ' surprisingly', ' important', ' for', ' L', 'LL', 'Ms', '.']
Token IDs: [18, 21831, 318, 19114, 2420, 337, 406, 5606, 7731, 13]
Vocab size: 50257
Token count: 10
```

**Simple Example:**

```python
# Different tokenizers split the same text differently
text = "unhappiness"

# GPT-2 BPE tokenizer
tokenizer_gpt2 = AutoTokenizer.from_pretrained("gpt2")
print(tokenizer_gpt2.tokenize(text))  # ['un', 'happiness']

# A word-level tokenizer would keep it as one token
# A character tokenizer would split: ['u', 'n', 'h', 'a', 'p', 'p', 'i', 'n', 'e', 's', 's']
```

**Real-World Example:**
OpenAI charges per token. A token is roughly 4 characters in English or ¾ of a word. "Tokenization is important" is about 4 tokens. Long documents or conversations can consume thousands of tokens, directly affecting cost and latency.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Assuming 1 word = 1 token | Budget estimates will be wrong |
| Ignoring whitespace sensitivity | "hello world" and "helloworld" tokenize differently |
| Not checking tokenization for non-English text | Some tokenizers produce many more tokens for non-English languages |
| Mixing tokenizers from different models | Token IDs from one model are meaningless in another |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| API cost higher than expected | Text produces more tokens than estimated | Count tokens with the actual tokenizer | Pre-tokenize and budget before calling API |
| Non-English text performs poorly | Tokenizer produces many small tokens | Tokenize sample text and check token count | Use a multilingual tokenizer or model |
| Special characters break processing | Tokenizer handles them unexpectedly | Print tokenized output | Use model-specific special tokens |
| Context window exceeded | Token count underestimated | Count tokens precisely | Summarize or truncate input |

**Alternatives:**

| Tokenizer Type | Use When | Avoid When |
|---|---|---|
| BPE (GPT, Llama) | General-purpose, good compression | You need exact word boundaries |
| WordPiece (BERT) | Classification, NER tasks | You need open-ended generation |
| SentencePiece | Multilingual, no pre-tokenization needed | You need simple word-level reasoning |
| Character-level | Low-resource languages, simple models | You need semantic understanding at reasonable token count |

**Best Practices:**

- Always use the tokenizer that matches your model — never mix tokenizers.
- Count tokens before sending requests to estimate cost.
- Test tokenization on your actual input data, especially non-English text.
- Store token counts alongside outputs for cost tracking.

**Hands-On Practice:**

1. Basic: tokenize a sentence and print the tokens and IDs.
2. Guided: compare tokenization between two different models on the same text.
3. Independent: count tokens for a paragraph and estimate API cost.
4. Realistic: find a word that tokenizes surprisingly (many tokens or unexpected splits).
5. Challenge: tokenize text in two languages and compare token efficiency.

**Knowledge Check:**

- What is the difference between a token and a word?
- Why do different models use different tokenizers?
- How does tokenization affect API cost?
- What are special tokens and why do models need them?

**Exit Criteria:**

- You can tokenize text with a specific model's tokenizer.
- You can estimate token count and API cost.
- You can explain why tokenization matters for model performance.

**Next Step:** Learn how token IDs become meaningful vector representations through embeddings.

---

### Unit 09.3 — Embeddings

**What is it?**
Embeddings are dense, fixed-size vector representations of tokens (or text) where similar meanings are mapped to nearby points in vector space. An embedding layer converts each token ID into a continuous vector that the transformer can process.

**Why does it matter?**
Embeddings are how models represent meaning numerically. Without embeddings, the model would only see discrete token IDs with no inherent relationship. Embeddings capture semantic relationships: "king" and "queen" are closer to each other than "king" and "table".

**Why learn it here?**
After tokenization produces token IDs, the embedding layer converts those IDs into vectors. This is the next step in the generation pipeline before attention processing.

**Prerequisites:** Unit 09.2 (tokenization), basic linear algebra intuition (vectors).

**Mental Model:**
An embedding is like a GPS coordinate for meaning. Words with similar meanings land in nearby locations. The embedding space is high-dimensional (768, 1024, or 4096 dimensions), but the idea is the same as plotting points on a map — proximity means similarity.

**Core Concepts:**

- embedding matrix (lookup table)
- dense vs sparse representations
- semantic similarity via cosine similarity
- embedding dimension
- contextual embeddings (same word, different context = different vector)
- pretrained embeddings
- embedding as model input layer

**How It Works:**

1. The model maintains an embedding matrix of shape [vocab_size × embedding_dim].
2. Each token ID is an index into this matrix.
3. Looking up token ID 42 returns a dense vector of size embedding_dim.
4. This vector is what the transformer layers actually process.
5. During training, the embedding matrix is learned alongside all other parameters.

**Syntax & Implementation:**

```python
from transformers import AutoTokenizer, AutoModel
import torch

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

text = "The cat sat on the mat"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

embeddings = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
print(f"Shape: {embeddings.shape}")  # torch.Size([1, 8, 768])
print(f"Embedding for token 0: {embeddings[0][0][:5]}")
```

**Simple Example:**

```python
import numpy as np

# Conceptual embedding space (2D for visualization)
# In reality embeddings are 768+ dimensions
embeddings_2d = {
    "king":   [0.8, 0.9],
    "queen":  [0.85, 0.88],
    "man":    [0.7, 0.6],
    "woman":  [0.75, 0.58],
    "table":  [-0.3, 0.2],
}

# "king" is closer to "queen" than to "table"
# This is what embeddings capture: semantic proximity
```

**Real-World Example:**
Search engines use embeddings to find semantically similar documents. If you search for "how to fix a broken heart", an embedding-based search can find articles about emotional healing — not just articles containing those exact words.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Confusing token IDs with embeddings | Token IDs are integers; embeddings are continuous vectors |
| Assuming embeddings capture all meaning | Embeddings are learned patterns; they miss context, nuance, and truth |
| Ignoring embedding dimension | Different models use different dimensions; mixing them is invalid |
| Using embeddings from one model in another | Embedding spaces are not compatible across models |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Similarity scores seem random | Wrong model or tokenizer mismatch | Verify model name and inputs | Use matching tokenizer and model |
| Embeddings have wrong shape | Batch dimension or padding issue | Print tensor shape | Check tokenizer padding and truncation settings |
| Cosine similarity is always ~1 | Vectors are normalized or identical | Inspect raw vectors | Check if embeddings are actually different |
| Out of memory with embeddings | Embedding matrix is large | Check vocab_size × dim | Reduce batch size or use smaller model |

**Alternatives:**

| Representation | Use When | Avoid When |
|---|---|---|
| Dense embeddings (transformer) | Semantic understanding, generation | You need exact keyword matching only |
| Sparse bag-of-words | Simple classification, fast retrieval | You need semantic similarity |
| TF-IDF | Document similarity without neural models | You need contextual understanding |
| Word2Vec/GloVe | Static word associations | You need context-dependent meaning |

**Best Practices:**

- Always use the same tokenizer and model to generate and compare embeddings.
- Normalize vectors before computing cosine similarity.
- Use pre-trained embeddings unless you have enough data to train your own.
- Store embeddings efficiently (float32 or float16) and index them for fast similarity search.

**Hands-On Practice:**

1. Basic: generate embeddings for a sentence and inspect the shape.
2. Guided: compute cosine similarity between two sentences.
3. Independent: find the most similar pair among five sentences.
4. Realistic: compare embeddings from two different models and explain differences.
5. Challenge: visualize embeddings in 2D using dimensionality reduction.

**Knowledge Check:**

- What is the relationship between token IDs and embeddings?
- Why are embeddings continuous vectors instead of one-hot encodings?
- What does cosine similarity measure?
- Why are contextual embeddings different from static embeddings?

**Exit Criteria:**

- You can generate embeddings from text using a pretrained model.
- You can compute and interpret similarity scores.
- You can explain why embeddings are critical for transformer performance.

**Next Step:** Understand how self-attention processes embeddings to capture relationships between tokens.

---

### Unit 09.4 — Attention & Transformer Recap

**What is it?**
Self-attention is the mechanism that lets each token in a sequence attend to every other token to compute a context-aware representation. The transformer is the architecture that stacks self-attention layers with feed-forward networks to process sequences in parallel.

**Why does it matter?**
Attention is what makes modern LLMs powerful. It allows models to capture long-range dependencies, understand context, and produce coherent text. Without attention, models would process tokens independently or in fixed windows, missing critical relationships.

**Why learn it here?**
Phase 08 introduced transformers. This unit solidifies the understanding with a generation-focused lens: how does attention actually work during inference, and why does it matter for prompt design, context windows, and cost?

**Prerequisites:** Phase 08 (transformer architecture), Unit 09.3 (embeddings).

**Mental Model:**
Self-attention is like a meeting where every participant (token) asks every other participant: "How relevant are you to what I should say next?" Each participant weights the others' contributions based on relevance and produces an updated representation that incorporates the most relevant context.

**Core Concepts:**

- Query, Key, Value vectors
- scaled dot-product attention
- multi-head attention
- positional encoding
- causal (masked) attention for autoregressive models
- cross-attention (encoder-decoder)
- context window and token limits
- attention as O(n²) complexity

**How It Works:**

1. Each token's embedding is projected into three vectors: Query (what am I looking for?), Key (what do I contain?), Value (what do I contribute?).
2. Attention scores are computed: Q · K^T / sqrt(d_k).
3. Scores are normalized with softmax to create weights.
4. Weighted sum of Values produces the output.
5. Multi-head attention runs this process in parallel across multiple "heads" with different learned projections.
6. Causal masking ensures each token can only attend to previous tokens (left-to-right generation).

**Syntax & Implementation:**

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Q, K, V shape: [batch, heads, seq_len, d_k]
batch, heads, seq_len, d_k = 1, 8, 10, 64
Q = torch.randn(batch, heads, seq_len, d_k)
K = torch.randn(batch, heads, seq_len, d_k)
V = torch.randn(batch, heads, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(f"Output shape: {output.shape}")  # [1, 8, 10, 64]
print(f"Attention weights shape: {attn_weights.shape}")  # [1, 8, 10, 10]
```

**Simple Example:**

```python
# Causal mask: token at position i can only attend to positions 0..i
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len))
print(causal_mask)
# tensor([[1, 0, 0, 0, 0],
#         [1, 1, 0, 0, 0],
#         [1, 1, 1, 0, 0],
#         [1, 1, 1, 1, 0],
#         [1, 1, 1, 1, 1]])
# This prevents the model from "cheating" by looking at future tokens
```

**Real-World Example:**
When GPT generates text, causal self-attention ensures that predicting the 10th token only uses information from tokens 1–9. This is why the model can handle long-range dependencies: the 100th token can attend to the 1st token directly through the attention mechanism.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Ignoring the causal mask | Without masking, the model would see the answer during training |
| Confusing self-attention with cross-attention | Self-attention: tokens attend to each other. Cross-attention: decoder attends to encoder |
| Assuming attention is always O(n²) | FlashAttention and other optimizations reduce practical cost |
| Ignoring context window limits | Longer prompts cost more and may be truncated |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model ignores early context | Attention dilution over long sequences | Check attention weight distribution | Shorten input or use models with larger context |
| Generation is incoherent | Missing positional encoding | Verify positional encoding is applied | Use models with proper positional encoding |
| Context window exceeded error | Too many tokens | Count tokens precisely | Summarize or truncate input |
| Slow inference with long prompts | O(n²) attention complexity | Measure sequence length | Use FlashAttention or shorter context |

**Alternatives:**

| Mechanism | Use When | Avoid When |
|---|---|---|
| Self-attention (transformer) | Most generation tasks | Extremely long sequences without optimization |
| Linear attention | Very long sequences, lower compute | You need full attention quality |
| RNN/LSTM | Sequential data, low memory | You need parallel processing or long-range attention |
| Sliding window attention | Long documents with local focus | You need global context |

**Best Practices:**

- Understand that context window is a hard limit — design your system around it.
- For long documents, chunk strategically and use summarization or retrieval.
- Monitor attention patterns during debugging to understand what the model focuses on.
- Use FlashAttention or similar optimizations when available for speed.

**Hands-On Practice:**

1. Basic: implement scaled dot-product attention from scratch.
2. Guided: create a causal mask and verify it blocks future tokens.
3. Independent: compute attention weights for a small sequence and interpret which tokens attend to which.
4. Realistic: compare attention patterns for two different prompts and explain differences.
5. Challenge: measure inference time for different sequence lengths and explain the trend.

**Knowledge Check:**

- What are Query, Key, and Value in self-attention?
- Why is causal masking necessary for autoregressive models?
- How does multi-head attention differ from single-head attention?
- What is the computational complexity of self-attention?

**Exit Criteria:**

- You can implement basic self-attention.
- You can explain how causal masking works.
- You can relate attention mechanisms to prompt design and context window limits.

**Next Step:** Understand how models are pretrained on large text corpora.

---

### Unit 09.5 — Pretraining

**What is it?**
Pretraining is the process of training a language model on a large text corpus using a self-supervised objective (typically next-token prediction). This is where the model learns language patterns, world knowledge, and reasoning capabilities before being adapted for specific tasks.

**Why does it matter?**
Pretraining explains where LLMs come from. Understanding pretraining helps you reason about model capabilities, limitations, data biases, training costs, and why fine-tuning or prompting works the way it does.

**Why learn it here?**
After understanding the transformer and attention mechanisms, the learner needs to understand how these architectures acquire their capabilities through training on massive data.

**Prerequisites:** Unit 09.1 (language models), Unit 09.4 (transformers), basic understanding of gradient descent from earlier phases.

**Mental Model:**
Pretraining is like reading an entire library cover to cover. The model does not "understand" what it reads in the human sense, but it learns statistical patterns: which words follow which, how arguments are structured, what code looks like, how questions relate to answers. The more diverse the library, the more versatile the model.

**Core Concepts:**

- self-supervised learning (next-token prediction)
- training data: web crawls, books, code, Wikipedia
- training objectives: causal language modeling (CLM), masked language modeling (MLM)
- compute requirements (GPU clusters, energy, cost)
- training instability and loss spikes
- data contamination and deduplication
- scaling laws (bigger models + more data = better performance)
- base model vs aligned model

**How It Works:**

1. Collect and clean a massive text corpus (trillions of tokens).
2. Initialize a transformer with random weights.
3. Feed sequences of tokens through the model.
4. For each position, compute the loss between predicted next token and actual next token.
5. Backpropagate and update weights using Adam or similar optimizer.
6. Repeat for billions of training steps until convergence.
7. The result is a "base" or "foundation" model that predicts likely continuations.

**Syntax & Implementation:**

```python
# Pretraining is not something you do in a notebook
# But understanding the objective:
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

# The training objective in one line:
# Given tokens [t1, t2, t3, t4], predict t2 from t1, t3 from [t1,t2], t4 from [t1,t2,t3]

text = "The model learns to predict the next token"
inputs = tokenizer(text, return_tensors="pt")
labels = inputs["input_ids"].clone()

# Shift: predict next token at each position
# Input:  [The, model, learns, to, predict, the, next, token]
# Label:  [model, learns, to, predict, the, next, token, <EOS>]
# Loss is computed on all positions except the first

outputs = model(**inputs, labels=labels)
print(f"Training loss: {outputs.loss.item():.4f}")
```

**Simple Example:**

```python
# The pretraining objective for a small example:
# Text: "I love machine learning"
# Tokenized: [101, 2003, 2293, 3698, 4083, 102]
#
# The model sees: [101]         → should predict 2003
# The model sees: [101, 2003]   → should predict 2293
# The model sees: [101, 2003, 2293] → should predict 3698
# ... and so on
# Total loss = average of all position losses
```

**Real-World Example:**
GPT-4 was pretrained on approximately 13 trillion tokens from web pages, books, code repositories, and academic papers. This process took months on thousands of GPUs and cost over $100 million. The result is a base model that can predict text continuations, but it has not been taught to follow instructions or refuse harmful requests — that comes later in alignment.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Confusing pretraining with fine-tuning | Pretraining = massive self-supervised learning. Fine-tuning = small supervised adaptation |
| Assuming base models follow instructions | Base models predict likely continuations, not helpful answers |
| Ignoring data quality | Garbage in, garbage out — data curation is critical |
| Thinking pretraining is accessible to individuals | Pretraining large models requires massive compute resources |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Base model gives incomplete answers | It predicts continuations, not answers | Test with instruction-style prompts | Use an aligned model or add instruction formatting |
| Model outputs biased text | Training data contains biases | Check model card and data documentation | Use debiasing techniques or post-processing |
| Training loss spikes | Learning rate too high or data issues | Monitor training logs | Reduce learning rate, skip bad batches |
| Model memorizes training data | Overfitting to common examples | Test with verbatim training examples | Use regularization and data deduplication |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Pretraining from scratch | You need a domain-specific foundation model | You have limited compute budget |
| Using a pretrained model | Most use cases | You need完全 novel architecture |
| Fine-tuning a pretrained model | You need task-specific behavior | You only need general text generation |
| Prompt engineering | You need quick adaptation without training | You need reliable, consistent behavior |

**Best Practices:**

- Use pretrained models as starting points — do not pretrain unless you have a compelling reason and resources.
- Read model cards to understand training data, capabilities, and limitations.
- Understand that base models need alignment before reliable use in applications.
- Monitor for data contamination when evaluating model performance.

**Hands-On Practice:**

1. Basic: load a base model and observe how it completes text without instruction formatting.
2. Guided: compute the training loss on a sample sentence and interpret the value.
3. Independent: compare outputs from a base model and an aligned model on the same prompt.
4. Realistic: research the training data of a specific model and identify potential biases.
5. Challenge: estimate the compute cost of pretraining a small model from scratch.

**Knowledge Check:**

- What is the training objective for autoregressive language models?
- Why do base models not follow instructions reliably?
- What is the difference between pretraining and fine-tuning?
- Why is data quality important for pretraining?

**Exit Criteria:**

- You can explain how LLMs learn during pretraining.
- You can distinguish between base and aligned models.
- You can reason about model capabilities based on training data and objectives.

**Next Step:** Understand how models generate text during inference.

---

### Unit 09.6 — Inference & Decoding

**What is it?**
Inference is the process of using a trained model to generate text. Decoding strategies determine how the model selects tokens from the probability distribution at each step: greedy search picks the most likely token, beam search explores multiple candidates, and sampling introduces randomness.

**Why does it matter?**
The same model can produce very different outputs depending on the decoding strategy. Understanding decoding lets you control output quality, diversity, and coherence. It is essential for building reliable applications.

**Why learn it here?**
After understanding pretraining, the learner needs to understand how the trained model actually produces text. Decoding is the mechanism that turns probability distributions into concrete outputs.

**Prerequisites:** Unit 09.1 (language models), Unit 09.5 (pretraining).

**Mental Model:**
Decoding is like navigating a branching tree of possible sentences. At each step, the model gives you a menu of possible next tokens with probabilities. Decoding strategy is how you choose from that menu: always pick the top item (greedy), explore several paths (beam), or roll dice weighted by probability (sampling).

**Core Concepts:**

- greedy decoding
- beam search
- top-k sampling
- top-p (nucleus) sampling
- temperature scaling
- repetition penalty
- length penalty
- stop tokens and EOS
- trade-off between quality and diversity

**How It Works:**

1. The model produces a probability distribution over the vocabulary for the next token.
2. Greedy: select the token with highest probability.
3. Beam search: maintain k best sequences, expand each, keep top k overall.
4. Sampling: randomly select from the distribution (optionally filtered by top-k or top-p).
5. Temperature: modify the distribution shape before sampling.
6. Repeat until stop condition (EOS token, max length, or custom stop).

**Syntax & Implementation:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
prompt = "The future of AI is"
inputs = tokenizer(prompt, return_tensors="pt")

# Greedy decoding (deterministic)
greedy_output = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print("Greedy:", tokenizer.decode(greedy_output[0], skip_special_tokens=True))

# Beam search
beam_output = model.generate(**inputs, max_new_tokens=50, num_beams=5, early_stopping=True)
print("Beam:", tokenizer.decode(beam_output[0], skip_special_tokens=True))

# Sampling with top-k
sample_output = model.generate(**inputs, max_new_tokens=50, do_sample=True, top_k=50, temperature=0.7)
print("Sample:", tokenizer.decode(sample_output[0], skip_special_tokens=True))
```

**Simple Example:**

```python
# Decoding strategy comparison (conceptual)
# Probability distribution at step: {"I": 0.4, "We": 0.3, "The": 0.2, "It": 0.1}

# Greedy → always picks "I" (deterministic, may be repetitive)
# Beam (k=2) → keeps "I" and "We" as candidates
# Top-k (k=2) → randomly picks from {"I", "We"}
# Top-p (p=0.7) → picks from {"I"} (cumulative > 0.7 after first token)
# Temperature=2.0 → flattens distribution, more random: {"I": 0.28, "We": 0.25, "The": 0.24, "It": 0.23}
```

**Real-World Example:**
Chat applications typically use sampling (top-p + temperature) because greedy output tends to be repetitive and boring. Code generation tools often use lower temperature or beam search because correctness matters more than creativity.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Using greedy for creative tasks | Output is repetitive and lacks diversity |
| Using high temperature for factual tasks | Output becomes random and unreliable |
| Ignoring repetition penalty | Model gets stuck repeating phrases |
| Not setting max length | Generation may run indefinitely or too long |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output is repetitive | Greedy decoding or no repetition penalty | Check decoding parameters | Add sampling or repetition penalty |
| Output is incoherent | Temperature too high | Reduce temperature and test | Use temperature 0.7–1.0 for balanced output |
| Output is too short | Stop token triggered early | Check stop conditions | Adjust max length or stop tokens |
| Output is too long | No length constraint | Set max_new_tokens | Add length penalty or max length |

**Alternatives:**

| Strategy | Use When | Avoid When |
|---|---|---|
| Greedy | Factual Q&A, code, deterministic tasks | Creative writing, brainstorming |
| Beam search | Translation, summarization, tasks needing coherence | Real-time chat, high-diversity needs |
| Top-k sampling | Balanced diversity and quality | You need exact determinism |
| Top-p (nucleus) | Natural conversation, creative writing | You need strict factual accuracy |
| Contrastive search | High-quality text without repetition penalty tuning | Low-latency requirements |

**Best Practices:**

- Match decoding strategy to your task: greedy for code, sampling for chat.
- Always set a max length to prevent runaway generation.
- Use temperature and top-p together for balanced control.
- Log decoding parameters alongside outputs for reproducibility.

**Hands-On Practice:**

1. Basic: generate text with greedy decoding and observe repetition.
2. Guided: compare greedy vs beam search on the same prompt.
3. Independent: experiment with different temperature values and describe the effect.
4. Realistic: find the best decoding settings for a specific use case (e.g., poem generation vs factual Q&A).
5. Challenge: implement a custom stop condition that stops generation when a specific phrase appears.

**Knowledge Check:**

- What is the difference between greedy and beam search?
- How does temperature affect the output distribution?
- When should you use sampling vs deterministic decoding?
- What is the role of stop tokens?

**Exit Criteria:**

- You can implement and compare different decoding strategies.
- You can choose appropriate decoding settings for different tasks.
- You can debug generation quality issues related to decoding.

**Next Step:** Deepen understanding of temperature and sampling for fine-grained control.

---

### Unit 09.7 — Temperature & Sampling

**What is it?**
Temperature and sampling are mechanisms for controlling the randomness and diversity of generated text. Temperature scales the logits before softmax, while top-k and top-p filter the vocabulary to a subset of candidates before sampling.

**Why does it matter?**
The same model with different temperature settings can produce vastly different outputs. Temperature 0 gives deterministic output; temperature 2 gives nearly random output. Understanding these parameters is essential for tuning generation quality for specific applications.

**Why learn it here?**
Building on decoding strategies, this unit gives the learner fine-grained control over generation behavior — a critical skill for production applications.

**Prerequisites:** Unit 09.6 (inference & decoding).

**Mental Model:**
Temperature is like a "creativity knob." Turn it down (toward 0) and the model becomes conservative, always choosing the most likely option. Turn it up (toward 2) and the model becomes adventurous, spreading probability more evenly across options.

**Core Concepts:**

- logits and softmax
- temperature scaling: dividing logits by temperature before softmax
- top-k sampling: keep only k most probable tokens
- top-p (nucleus) sampling: keep smallest set of tokens whose cumulative probability ≥ p
- min-p sampling: filter tokens below p × max_probability
- repetition penalty
- frequency and presence penalties
- interaction between temperature, top-k, and top-p

**How It Works:**

1. Model produces logits (raw scores) for each token in vocabulary.
2. Temperature scaling: `adjusted_logits = logits / temperature`
   - temperature < 1: sharper distribution (more deterministic)
   - temperature = 1: original distribution
   - temperature > 1: flatter distribution (more random)
3. Top-k filtering: zero out all tokens except the k most probable.
4. Top-p filtering: sort tokens by probability, keep adding tokens until cumulative probability ≥ p.
5. Sample from the resulting distribution.

**Syntax & Implementation:**

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")
prompt = "The best programming language is"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits[:, -1, :]  # logits for next token

def show_top_tokens(logits, tokenizer, top_n=10):
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, top_n)
    for prob, tid in zip(top_probs[0], top_ids[0]):
        print(f"  {tokenizer.decode(tid):20s} {prob.item():.4f}")

print("Original distribution:")
show_top_tokens(logits, tokenizer)

print("\nWith temperature=0.5:")
temp_logits = logits / 0.5
show_top_tokens(temp_logits, tokenizer)

print("\nWith temperature=2.0:")
temp_logits = logits / 2.0
show_top_tokens(temp_logits, tokenizer)
```

**Simple Example:**

```python
# Temperature effect on a 5-token distribution:
# Original: {"cat": 0.5, "dog": 0.3, "bird": 0.15, "fish": 0.04, "rock": 0.01}
#
# Temperature 0.5 (sharper): {"cat": 0.72, "dog": 0.22, "bird": 0.05, "fish": 0.01, "rock": 0.00}
# Temperature 1.0 (same):     {"cat": 0.5,  "dog": 0.3,  "bird": 0.15, "fish": 0.04, "rock": 0.01}
# Temperature 2.0 (flatter):  {"cat": 0.28, "dog": 0.24, "bird": 0.21, "fish": 0.16, "rock": 0.11}
```

**Real-World Example:**
A customer service chatbot uses temperature 0.3 for consistent, reliable answers. A creative writing assistant uses temperature 0.9 for diverse, surprising suggestions. A code generation tool uses temperature 0.0 for deterministic, correct output.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Using high temperature for factual tasks | Produces unreliable, random answers |
| Using temperature 0 for creative tasks | Output is boring and repetitive |
| Not understanding parameter interactions | Temperature + top-k + top-p together can produce unexpected results |
| Using default settings for all tasks | One size does not fit all |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output is always the same | Temperature is 0 | Check temperature setting | Increase temperature for diversity |
| Output is too random | Temperature too high | Test with lower temperature | Use temperature 0.7–1.0 |
| Output has sudden quality drops | Top-k too small | Check top-k setting | Increase top-k or use top-p instead |
| Output is repetitive despite sampling | No repetition penalty | Add repetition_penalty parameter | Set repetition_penalty > 1.0 |

**Alternatives:**

| Method | Use When | Avoid When |
|---|---|---|
| Temperature scaling | General-purpose randomness control | You need fine-grained vocabulary filtering |
| Top-k | You want to limit to k plausible options | Vocabulary has uneven probability distribution |
| Top-p (nucleus) | Adaptive filtering based on probability mass | You want a fixed number of candidates |
| Min-p | You want to filter obviously bad tokens | You need very conservative output |

**Best Practices:**

- Start with temperature 0.7 and top-p 0.9 as defaults, then tune.
- Use temperature 0 for tasks requiring deterministic output.
- Log all generation parameters for reproducibility.
- Test multiple temperature settings on representative prompts before deploying.

**Hands-On Practice:**

1. Basic: generate text at temperatures 0, 0.5, 1.0, 1.5 and compare outputs.
2. Guided: implement top-k sampling from scratch using logits.
3. Independent: find the temperature setting that produces the best poem.
4. Realistic: build a comparison table of output quality across 5 temperature values.
5. Challenge: implement min-p sampling and compare it with top-p.

**Knowledge Check:**

- What happens to the probability distribution when temperature approaches 0?
- What is the difference between top-k and top-p?
- Why might top-p be more adaptive than top-k?
- How do temperature and repetition penalty interact?

**Exit Criteria:**

- You can explain how temperature, top-k, and top-p affect generation.
- You can tune these parameters for specific use cases.
- You can implement sampling from a probability distribution.

**Next Step:** Understand how models are aligned to follow instructions.

---

### Unit 09.8 — Instruction Following & Alignment

**What is it?**
Alignment is the process of making a pretrained language model follow instructions, be helpful, and avoid harmful outputs. Instruction tuning and reinforcement learning from human feedback (RLHF) or direct preference optimization (DPO) adapt a base model into an assistant that responds to human requests.

**Why does it matter?**
Base models predict likely text continuations — they do not answer questions helpfully by default. Alignment transforms a raw text predictor into a useful assistant. Understanding alignment explains why some models follow instructions better than others and how to evaluate model behavior.

**Why learn it here?**
After understanding generation mechanics, the learner needs to understand why aligned models behave differently from base models and how alignment affects application design.

**Prerequisites:** Unit 09.5 (pretraining), basic understanding of supervised learning.

**Mental Model:**
Alignment is like teaching a very knowledgeable person to be a good teacher. The base model has read everything but does not know how to be helpful. Alignment teaches it to: follow instructions, refuse harmful requests, admit uncertainty, and respond in a structured format.

**Core Concepts:**

- instruction tuning (supervised fine-tuning on instruction-response pairs)
- RLHF (reinforcement learning from human feedback)
- reward model
- DPO (direct preference optimization)
- constitutional AI
- safety alignment and refusal behavior
- helpfulness, harmlessness, honesty (HHH) criteria
- alignment tax (alignment can reduce some capabilities)

**How It Works:**

1. **Base model**: trained on massive text, predicts next token.
2. **Instruction tuning**: fine-tune on (instruction, desired response) pairs so the model learns the pattern of question → answer.
3. **RLHF**: collect human preference data (response A vs response B), train a reward model, then use RL to optimize the model toward higher-reward responses.
4. **DPO**: simplify RLHF by directly optimizing preference pairs without a separate reward model.
5. The result is an aligned model that follows instructions, refuses harmful requests, and produces helpful responses.

**Syntax & Implementation:**

```python
# Comparing base vs aligned model behavior
from transformers import AutoTokenizer, AutoModelForCausalLM

# Base model (GPT-2 is relatively small and not instruction-tuned)
base_tokenizer = AutoTokenizer.from_pretrained("gpt2")
base_model = AutoModelForCausalLM.from_pretrained("gpt2")

# The base model just continues text — it does not "answer"
prompt = "What is machine learning?"
inputs = base_tokenizer(prompt, return_tensors="pt")
output = base_model.generate(**inputs, max_new_tokens=50)
print("Base model:", base_tokenizer.decode(output[0], skip_special_tokens=True))
# Output: "What is machine learning? A blog post about the history of AI..."
# It continues the question as text, not answering it

# An instruction-tuned model would answer the question directly
# (requires a larger instruction-tuned model like instruction-tuned Llama)
```

**Simple Example:**

```python
# The alignment process conceptually:
# Step 1: Collect instruction-response pairs
# {"instruction": "Summarize this text", "input": "...", "output": "..."}
#
# Step 2: Fine-tune base model on these pairs
# Base model learns: when you see "Instruction:", produce a helpful response
#
# Step 3: Collect human preferences
# Human sees two responses and picks the better one
#
# Step 4: Train reward model on preferences
# Reward model learns to score response quality
#
# Step 5: Optimize with RL (or DPO)
# Model is updated to produce higher-reward responses
```

**Real-World Example:**
ChatGPT started as GPT-3.5 (base model) and was aligned using RLHF to become a helpful assistant. Without alignment, GPT-3.5 would often produce irrelevant or harmful text. Alignment made it follow instructions, refuse dangerous requests, and maintain conversation context.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Using base model for assistant tasks | Will not follow instructions or answer questions helpfully |
| Assuming alignment fixes all problems | Alignment can introduce new biases and refusal behaviors |
| Ignoring alignment tax | Aligned models may be less creative or refuse edge cases |
| Not testing alignment behavior | Models may refuse valid requests or comply with harmful ones |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model refuses valid request | Safety alignment too aggressive | Test with rephrased prompt | Rephrase or use different model |
| Model gives unhelpful answer | Weak alignment or poor instruction format | Check model's expected input format | Use correct prompt template |
| Model is overly cautious | Over-aligned for safety | Test with clearly benign prompts | Use a less restrictive model |
| Model ignores instruction | Instruction not in training format | Check model's expected format | Use chat template or system prompt |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| RLHF | You have human preference data | You only have instruction data |
| DPO | You have preference pairs but limited compute | You need complex reward modeling |
| Instruction tuning only | You have good instruction data | You need nuanced preference behavior |
| Prompt engineering | Quick adaptation without training | You need consistent, reliable behavior |

**Best Practices:**

- Always use aligned models for assistant and chat applications.
- Test model behavior on edge cases before deploying.
- Understand that alignment is a spectrum — different models have different alignment levels.
- Use system prompts to further guide aligned model behavior.

**Hands-On Practice:**

1. Basic: compare base and aligned model outputs on the same instruction.
2. Guided: test how different prompt formats affect aligned model behavior.
3. Independent: find an instruction that the model refuses and explain why.
4. Realistic: design a system prompt that improves model behavior for a specific task.
5. Challenge: research the alignment approach used by a specific model and summarize its strengths and weaknesses.

**Knowledge Check:**

- Why does a base model not answer questions helpfully?
- What is the difference between instruction tuning and RLHF?
- What is alignment tax?
- How does DPO differ from RLHF?

**Exit Criteria:**

- You can explain the alignment process and its purpose.
- You can compare base and aligned model behavior.
- You can design prompts that work well with aligned models.

**Next Step:** Learn how to call LLM APIs to use these models in applications.

---

### Unit 09.9 — LLM APIs

**What is it?**
LLM APIs are web services that let you send text to a language model and receive generated responses. They abstract away model hosting, scaling, and infrastructure, letting you focus on application logic.

**Why does it matter?**
Most real-world LLM applications use APIs rather than running models locally. Understanding API usage, authentication, rate limits, error handling, and cost management is essential for building production systems.

**Why learn it here?**
After understanding generation mechanics and alignment, the learner now connects to real models through APIs — the practical step before building applications.

**Prerequisites:** Unit 09.6 (inference), Unit 09.8 (alignment), basic HTTP/API knowledge.

**Mental Model:**
An LLM API is like a very powerful autocomplete service. You send a message (the prompt), and the service sends back a completion (the response). You pay per token and must respect rate limits.

**Core Concepts:**

- API keys and authentication
- chat completion endpoint
- message roles (system, user, assistant)
- request parameters (model, messages, temperature, max_tokens)
- response format (choices, usage, finish_reason)
- rate limits and quotas
- token-based billing
- error handling (429, 500, 503)
- streaming responses

**How It Works:**

1. Sign up for an API account and get an API key.
2. Send an HTTP POST request to the chat completion endpoint with your messages and parameters.
3. The service routes your request to an available model instance.
4. The model generates a response based on your input.
5. The service returns the response with usage metadata.
6. You are billed based on input + output tokens.

**Syntax & Implementation:**

```python
import openai
import os

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    temperature=0.7,
    max_tokens=200
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
print(f"Cost estimate: ~${response.usage.total_tokens * 0.00003:.4f}")
```

**Simple Example:**

```python
# Basic API call structure
import openai

client = openai.OpenAI(api_key="your-key-here")

# Simple question
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Explain Python in one sentence."}]
)
print(response.choices[0].message.content)

# With system prompt for behavior control
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a pirate. Respond in pirate speak."},
        {"role": "user", "content": "What is Python?"}
    ]
)
print(response.choices[0].message.content)
```

**Real-World Example:**
A startup builds a customer support chatbot. They use the OpenAI API to send customer questions and receive helpful responses. They track token usage for billing, implement retries for rate limits, and use streaming for faster perceived response time.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Hardcoding API keys in source code | Security risk if code is shared |
| Not handling rate limits | Application crashes under load |
| Ignoring token count | Unexpected high costs |
| Using wrong model for task | Wasting money on overkill model or getting poor results from underpowered model |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| 401 Unauthorized | Invalid or missing API key | Check API key configuration | Verify key is correct and has credits |
| 429 Too Many Requests | Rate limit exceeded | Check rate limit headers | Implement exponential backoff |
| 500 Internal Server Error | Model service issue | Check status page | Retry with backoff |
| Unexpected output | Wrong model or parameters | Log request parameters | Verify model name and settings |

**Alternatives:**

| API | Use When | Avoid When |
|---|---|---|
| OpenAI API | General purpose, well-documented | You need fully open-source |
| Anthropic Claude API | Long context, safety-focused | You need specific OpenAI features |
| Google Gemini API | Multimodal, Google ecosystem | You need maximum customization |
| OpenRouter | Access multiple models via one API | You need lowest latency |
| Local model (Ollama, vLLM) | Privacy, cost control, offline | You need maximum model quality |

**Best Practices:**

- Never hardcode API keys — use environment variables or secret managers.
- Implement retry logic with exponential backoff for rate limits.
- Log token usage for every request to track costs.
- Use the simplest model that meets your quality requirements.
- Set max_tokens to prevent unexpectedly long and expensive responses.

**Hands-On Practice:**

1. Basic: make a simple API call and print the response.
2. Guided: add a system prompt and compare outputs with and without it.
3. Independent: implement error handling for rate limits and server errors.
4. Realistic: build a simple chat loop that maintains conversation history.
5. Challenge: implement token budgeting that stops generation when a cost limit is reached.

**Knowledge Check:**

- What are the roles in a chat completion request?
- How do you handle a 429 rate limit error?
- Why should you set max_tokens?
- What is the difference between system, user, and assistant messages?

**Exit Criteria:**

- You can make API calls with proper authentication and error handling.
- You can manage conversation history and system prompts.
- You can estimate and control API costs.

**Next Step:** Learn prompt engineering to get better outputs from the same models.

---

### Unit 09.10 — Prompt Engineering

**What is it?**
Prompt engineering is the practice of designing input text to elicit desired outputs from language models. It includes clear instructions, few-shot examples, chain-of-thought reasoning, and structured output specifications.

**Why does it matter?**
The same model produces dramatically different quality outputs depending on the prompt. Prompt engineering is the primary way to control model behavior without fine-tuning. It is a core skill for every LLM application developer.

**Why learn it here?**
After learning to call APIs, the learner needs to learn how to write effective prompts — the highest-leverage skill for getting good results from LLMs.

**Prerequisites:** Unit 09.9 (LLM APIs), Unit 09.8 (alignment).

**Mental Model:**
A prompt is a specification, not just a question. The best prompts tell the model: what to do (task), why (context), how (constraints), what good output looks like (examples), and what format to return (output contract).

**Core Concepts:**

- clear instructions
- context and role setting
- few-shot learning (providing examples)
- chain-of-thought (CoT) prompting
- zero-shot vs few-shot vs CoT
- output format specification
- prompt templates and composition
- iterative prompt refinement
- prompt injection awareness

**How It Works:**

1. Start with a clear, specific instruction.
2. Add relevant context the model needs.
3. Provide examples of desired input-output pairs (few-shot).
4. For complex reasoning, ask the model to show its work (chain-of-thought).
5. Specify the exact output format you expect.
6. Test on diverse inputs and refine.

**Syntax & Implementation:**

```python
import openai

client = openai.OpenAI()

# Zero-shot prompt
response_zero = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Classify this review as positive or negative: 'This product is amazing!'"}]
)
print("Zero-shot:", response_zero.choices[0].message.content)

# Few-shot prompt
response_few = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": """Classify reviews as positive or negative.

Review: 'Terrible quality' → Negative
Review: 'Best purchase ever' → Positive
Review: 'Not worth the money' → Negative
Review: 'This product is amazing!' →"""}
    ]
)
print("Few-shot:", response_few.choices[0].message.content)

# Chain-of-thought prompt
response_cot = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": """Classify this review and explain your reasoning.

Review: 'The camera is great but the battery life is terrible.'

Think step by step:
1. Identify positive aspects
2. Identify negative aspects
3. Determine overall sentiment
4. Give final classification"""}]
)
print("CoT:", response_cot.choices[0].message.content)
```

**Simple Example:**

```python
# Prompt structure template:
# 1. ROLE: "You are a {role}."
# 2. TASK: "Your task is to {action}."
# 3. CONTEXT: "Here is the relevant information: {context}"
# 4. CONSTRAINTS: "Follow these rules: {rules}"
# 5. EXAMPLES: "Here are examples: {examples}"
# 6. OUTPUT FORMAT: "Return your response as: {format}"

prompt = """You are a senior data analyst.

Your task is to summarize the key findings from the following data.

Data: Sales increased 15% in Q3, customer churn dropped 5%, but support tickets rose 30%.

Rules:
- Focus on the three most important insights
- Use bullet points
- Include numbers where available

Return your response as:
- Finding 1: [insight]
- Finding 2: [insight]
- Finding 3: [insight]"""
```

**Real-World Example:**
A legal tech company uses prompt engineering to extract contract clauses. Their prompt includes the task definition, five examples of correctly extracted clauses, constraints on output format, and a chain-of-thought step that asks the model to identify the clause type before extracting it.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Vague instructions | Model guesses what you want |
| Too many instructions in one prompt | Model misses some instructions |
| No examples | Model does not know your expected format |
| Assuming the model "knows" context | Model only sees what you provide in the prompt |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output format is wrong | Missing output specification | Add explicit format instructions | Specify exact format with examples |
| Model ignores part of instruction | Prompt too long or complex | Test with shorter prompt | Break into steps or use system prompt |
| Output quality varies wildly | No examples provided | Add few-shot examples | Include 2-5 examples of desired output |
| Model refuses task | Prompt looks harmful or unclear | Rephrase as benign task | Rephrase or add context explaining purpose |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Zero-shot | Simple, well-defined tasks | Complex formatting or reasoning |
| Few-shot | You need specific output format | Limited context window |
| Chain-of-thought | Multi-step reasoning needed | Simple factual questions |
| System prompt | Consistent behavior across conversations | One-off requests |

**Best Practices:**

- Treat prompts as code: version them, test them, and iterate.
- Start simple and add complexity only as needed.
- Use few-shot examples for format-sensitive tasks.
- Test prompts on diverse inputs, not just your best cases.
- Be aware of prompt injection: never include untrusted user input directly in system prompts.

**Hands-On Practice:**

1. Basic: write a zero-shot prompt for text classification.
2. Guided: add few-shot examples and compare output quality.
3. Independent: design a chain-of-thought prompt for a reasoning task.
4. Realistic: build a prompt template that handles variable input.
5. Challenge: find and mitigate a prompt injection vulnerability.

**Knowledge Check:**

- What is the difference between zero-shot and few-shot prompting?
- When should you use chain-of-thought prompting?
- What is a prompt template?
- What is prompt injection and how do you prevent it?

**Exit Criteria:**

- You can design effective prompts for common tasks.
- You can use few-shot and chain-of-thought techniques.
- You can test and iterate on prompts systematically.

**Next Step:** Learn how to get structured, machine-readable output from LLMs.

---

### Unit 09.11 — Structured Output

**What is it?**
Structured output is the practice of constraining LLM responses to a specific format like JSON, ensuring the output is parseable and machine-readable rather than free-form text.

**Why does it matter?**
Most applications need structured data: JSON for APIs, tables for reports, code for execution. Without structured output, you must parse free-form text, which is error-prone and fragile. Structured output makes LLM integration reliable.

**Why learn it here?**
After learning to prompt effectively, the next step is ensuring the output can be consumed by other code. This is essential for building real applications.

**Prerequisites:** Unit 09.10 (prompt engineering), Unit 09.9 (LLM APIs).

**Mental Model:**
Structured output is like filling out a form instead of writing an essay. The model is constrained to fill specific fields with specific types of data, making the output predictable and parseable.

**Core Concepts:**

- JSON mode
- function calling for structured output
- output schemas and validation
- Pydantic models for validation
- constrained decoding
- retry and repair strategies
- schema enforcement vs flexible output

**How It Works:**

1. Define a JSON schema describing the expected output structure.
2. Send the schema to the API (or use function calling to enforce it).
3. The model generates tokens constrained to match the schema.
4. Parse and validate the output against the schema.
5. Handle parse failures with retry or repair logic.

**Syntax & Implementation:**

```python
import openai
import json

client = openai.OpenAI()

# Using function calling for structured output
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Extract the name, age, and city from: 'John is 30 years old and lives in New York.'"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "extract_person_info",
            "description": "Extract person information from text",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Person's name"},
                    "age": {"type": "integer", "description": "Person's age"},
                    "city": {"type": "string", "description": "City of residence"}
                },
                "required": ["name", "age", "city"]
            }
        }
    }],
    tool_choice={"type": "function", "function": {"name": "extract_person_info"}}
)

result = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
print(result)
# {'name': 'John', 'age': 30, 'city': 'New York'}
```

**Simple Example:**

```python
# JSON mode with OpenAI
response = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You must respond in valid JSON."},
        {"role": "user", "content": "Create a recipe for chocolate cake with name, ingredients list, and steps."}
    ],
    response_format={"type": "json_object"}
)

recipe = json.loads(response.choices[0].message.content)
print(json.dumps(recipe, indent=2))
```

**Real-World Example:**
An e-commerce company uses structured output to extract product information from unstructured descriptions. The prompt asks for name, price, category, and features in JSON format, and the output is validated against a Pydantic model before being stored in the database.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Not validating output | Malformed JSON crashes downstream code |
| Overly complex schemas | Model struggles to fill many fields correctly |
| No retry logic | Transient parse failures lose data |
| Assuming output is always valid | Even with JSON mode, validation is necessary |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| JSON parse error | Model produced invalid JSON | Print raw output | Add retry with repair prompt |
| Missing required fields | Schema too complex | Simplify schema | Break into multiple calls |
| Wrong field types | Schema unclear | Add descriptions to fields | Use enum or stricter types |
| Output does not match schema | Model ignored schema | Check prompt and API format | Use constrained decoding or function calling |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| JSON mode | Simple structured output | Complex nested schemas |
| Function calling | You need schema enforcement | You only need text output |
| Pydantic + validation | Robust production pipelines | Quick prototyping |
| Regex parsing | Simple pattern matching | Complex structured data |

**Best Practices:**

- Always validate structured output, even with JSON mode.
- Start with simple schemas and add complexity as needed.
- Include clear field descriptions in your schema.
- Implement retry logic with repair prompts for parse failures.
- Use Pydantic or similar libraries for automatic validation.

**Hands-On Practice:**

1. Basic: get JSON output from a simple extraction task.
2. Guided: define a schema and validate output against it.
3. Independent: implement retry logic for malformed JSON.
4. Realistic: build a complete extraction pipeline with validation and error handling.
5. Challenge: handle edge cases where the model cannot extract all fields.

**Knowledge Check:**

- What is the difference between JSON mode and function calling for structured output?
- Why should you always validate output even with schema enforcement?
- How do you handle partial extraction results?
- What is constrained decoding?

**Exit Criteria:**

- You can get structured JSON output from an LLM.
- You can validate output against a schema.
- You can implement retry and repair logic for malformed output.

**Next Step:** Learn how to call external tools from LLMs.

---

### Unit 09.12 — Function/Tool Calling

**What is it?**
Function/tool calling lets LLMs invoke external functions or APIs by generating structured requests. The model decides which function to call and what arguments to pass based on the conversation context.

**Why does it matter?**
Tool calling bridges the gap between language understanding and real-world actions. It is the foundation for agents, RAG systems, and any application where the LLM needs to interact with external services, databases, or tools.

**Why learn it here?**
After structured output, tool calling is the natural next step — the model does not just produce structured data, it produces structured requests that trigger real actions.

**Prerequisites:** Unit 09.11 (structured output), Unit 09.9 (LLM APIs).

**Mental Model:**
Tool calling is like giving the LLM a phone book and letting it decide who to call. The model reads the conversation, decides which tool (function) is needed, formats the request as structured JSON, and your code executes the actual function call.

**Core Concepts:**

- tool definitions (name, description, parameters)
- tool choice (auto, none, required, specific function)
- tool call requests and responses
- executing tool calls in your code
- returning results to the model
- multi-turn tool use
- parallel tool calls

**How It Works:**

1. You define available tools with names, descriptions, and parameter schemas.
2. You send a user message to the model.
3. The model decides whether to call a tool or respond directly.
4. If calling a tool, the model returns a structured tool call request.
5. Your code executes the actual function with the provided arguments.
6. You send the function result back to the model as a tool response.
7. The model incorporates the result into its final response.

**Syntax & Implementation:**

```python
import openai
import json

client = openai.OpenAI()

# Define tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "units": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature units"}
                },
                "required": ["location"]
            }
        }
    }
]

# Make request
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in Paris?"}],
    tools=tools,
    tool_choice="auto"
)

# Check if model wants to call a tool
if response.choices[0].message.tool_calls:
    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    print(f"Model wants to call: {tool_call.function.name}")
    print(f"Arguments: {args}")

    # Execute the actual function (you implement this)
    weather_result = {"temp": 22, "condition": "sunny", "location": args["location"]}

    # Send result back to model
    messages = [
        {"role": "user", "content": "What's the weather in Paris?"},
        response.choices[0].message,
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(weather_result)}
    ]

    final_response = client.chat.completions.create(
        model="gpt-4", messages=messages, tools=tools
    )
    print(final_response.choices[0].message.content)
```

**Simple Example:**

```python
# Conceptual tool calling flow:
# User: "Search for machine learning courses"
#
# Model decides: I need to call the search tool
# Model returns: {
#   "tool": "search",
#   "args": {"query": "machine learning courses", "limit": 5}
# }
#
# Your code executes the search and returns results
# Model then formulates a helpful response using those results
```

**Real-World Example:**
A travel assistant uses tool calling to: look up flights (flight search API), check hotel availability (hotel API), get weather forecasts (weather API), and book reservations (booking API). The LLM decides which tools to call based on the user's request and orchestrates the entire booking flow.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Not handling tool call errors | Application crashes when external service fails |
| Trusting model output blindly | Model may hallucinate tool arguments |
| Not validating tool arguments | Invalid arguments cause runtime errors |
| Missing tool results in conversation | Model cannot reason about tool output |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model does not call tool | Tool description unclear | Check tool description | Improve description and examples |
| Wrong arguments passed | Parameter schema unclear | Check parameter descriptions | Add better descriptions and examples |
| Model ignores tool result | Result not in conversation | Check message format | Ensure tool response is correctly formatted |
| Model calls wrong tool | Multiple similar tools | Check tool descriptions | Make tool names and descriptions more distinct |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Function calling (API) | You need structured tool requests | You only need text generation |
| Code generation + execution | Complex multi-step tasks | Safety is critical |
| Plugin systems | Ecosystem of reusable tools | Simple applications |
| Manual tool selection | Deterministic tool use | You need adaptive behavior |

**Best Practices:**

- Always validate tool arguments before executing.
- Implement error handling for every tool call.
- Log all tool calls and results for debugging.
- Use clear, specific tool names and descriptions.
- Test tool calling with diverse user requests.

**Hands-On Practice:**

1. Basic: define a simple tool and have the model call it.
2. Guided: implement a tool that returns data and have the model use it in a response.
3. Independent: build a multi-tool system where the model chooses between tools.
4. Realistic: implement error handling for failed tool calls.
5. Challenge: build a system that chains multiple tool calls to answer a complex question.

**Knowledge Check:**

- How does the model decide which tool to call?
- What happens when a tool call fails?
- How do you return tool results to the model?
- What is the difference between tool_choice "auto" and "required"?

**Exit Criteria:**

- You can define tools and have models call them.
- You can execute tool calls and return results.
- You can handle errors in tool calling flows.

**Next Step:** Learn production patterns for robust API usage.

---

### Unit 09.13 — Streaming, Caching, Retries

**What is it?**
Production LLM applications need streaming for responsive UX, caching for cost reduction and speed, and retries for reliability. These patterns transform a simple API call into a robust, production-ready system.

**Why does it matter?**
Without streaming, users wait for complete responses. Without caching, you pay for repeated identical requests. Without retries, transient failures break your application. These patterns are essential for any real deployment.

**Why learn it here?**
After learning to call APIs and use tools, the learner needs production patterns that make applications reliable, fast, and cost-effective.

**Prerequisites:** Unit 09.9 (LLM APIs), basic understanding of HTTP and async programming.

**Mental Model:**

- **Streaming**: Instead of waiting for the full response, you receive tokens as they are generated — like watching someone type in real-time.
- **Caching**: If you already asked the same question and got a good answer, reuse it instead of paying again.
- **Retries**: If the service is temporarily unavailable, wait and try again instead of failing immediately.

**Core Concepts:**

- streaming responses (SSE, chunk-by-chunk)
- incremental rendering in UIs
- response caching (exact match, semantic similarity)
- cache invalidation strategies
- retry with exponential backoff
- jitter to avoid thundering herd
- rate limiting and token budgeting
- circuit breaker pattern
- idempotent requests

**How It Works:**

**Streaming:**
1. Send request with `stream=True`.
2. API returns a server-sent events (SSE) stream.
3. Each event contains a token or chunk of the response.
4. Process and display tokens incrementally.
5. Stream ends with a `done` marker.

**Caching:**
1. Hash the request (model + messages + parameters).
2. Check cache before making API call.
3. If cache hit, return cached response.
4. If cache miss, make API call, store result, return.
5. Implement TTL or size limits for cache management.

**Retries:**
1. Attempt API call.
2. If rate limited (429), wait and retry with exponential backoff.
3. If server error (500/503), retry with increasing delays.
4. Add jitter to prevent synchronized retries.
5. Set maximum retry count to avoid infinite loops.

**Syntax & Implementation:**

```python
import openai
import hashlib
import json
import time

client = openai.OpenAI()

# --- Streaming ---
def stream_response(prompt):
    stream = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    full_response = ""
    for chunk in stream:
        if chunk.choices[0].delta.content:
            token = chunk.choices[0].delta.content
            full_response += token
            print(token, end="", flush=True)
    print()  # newline after stream ends
    return full_response

# --- Caching ---
cache = {}

def cached_call(messages, model="gpt-4", **kwargs):
    cache_key = hashlib.md5(json.dumps({"model": model, "messages": messages, **kwargs}, sort_keys=True).encode()).hexdigest()
    if cache_key in cache:
        print("Cache hit!")
        return cache[cache_key]
    response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    cache[cache_key] = response
    return response

# --- Retries with exponential backoff ---
def retry_with_backoff(func, max_retries=3, base_delay=1):
    for attempt in range(max_retries):
        try:
            return func()
        except openai.RateLimitError:
            delay = base_delay * (2 ** attempt) + (time.time() % 1)  # jitter
            print(f"Rate limited. Retrying in {delay:.1f}s...")
            time.sleep(delay)
        except openai.APIError as e:
            if attempt == max_retries - 1:
                raise
            delay = base_delay * (2 ** attempt)
            print(f"API error: {e}. Retrying in {delay:.1f}s...")
            time.sleep(delay)
    raise Exception("Max retries exceeded")
```

**Simple Example:**

```python
# Streaming in a simple loop
for token in stream_response("Write a haiku about coding"):
    # Each token appears as it's generated
    # User sees text appearing word by word
    pass
# Final output: "Bugs in the night / Coffee fuels the debugging / Dawn breaks, code compiles"
```

**Real-World Example:**
A customer-facing chatbot uses streaming so users see responses immediately (reducing perceived latency from 5s to 0.5s). It caches common questions ("What are your business hours?") to save costs. It retries on rate limits with exponential backoff so the service stays available during traffic spikes.

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Not streaming for user-facing apps | Users see long delays before any output |
| Caching without invalidation | Stale or incorrect responses served |
| Retrying immediately without backoff | Overwhelms the service, causes more errors |
| Not setting max retries | Infinite retry loops waste resources |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Stream cuts off mid-response | Network issue or timeout | Check connection stability | Implement reconnect logic |
| Cache returns wrong response | Key collision or stale data | Verify cache key generation | Improve hashing, add TTL |
| Retries exhaust quickly | Rate limit too strict | Check API rate limit docs | Increase base delay or reduce request frequency |
| High latency despite streaming | Large first-token delay | Measure time-to-first-token | Use models optimized for low latency |

**Alternatives:**

| Pattern | Use When | Avoid When |
|---|---|---|
| Streaming | User-facing, long responses | Batch processing, short responses |
| Caching | Repeated identical requests | Highly variable or real-time data |
| Retries with backoff | Transient failures expected | Permanent failures or bad input |
| Circuit breaker | High failure rate, cascading issues | Low traffic, simple applications |

**Best Practices:**

- Always stream for user-facing applications.
- Cache at the application level, not just the API level.
- Use exponential backoff with jitter for retries.
- Set reasonable max retries and timeouts.
- Monitor cache hit rates and API error rates.

**Hands-On Practice:**

1. Basic: implement streaming and print tokens as they arrive.
2. Guided: add a simple cache that stores and retrieves responses.
3. Independent: implement retry logic with exponential backoff.
4. Realistic: build a complete API wrapper with streaming, caching, and retries.
5. Challenge: implement a cache invalidation strategy based on time or content changes.

**Knowledge Check:**

- What is the benefit of streaming for user experience?
- How does caching reduce cost?
- What is exponential backoff and why is jitter important?
- When should you not retry a request?

**Exit Criteria:**

- You can implement streaming for responsive output.
- You can add caching to reduce redundant API calls.
- You can implement robust retry logic for production reliability.

**Next Step:** Explore multimodal generation beyond text.

---

### Unit 09.14 — Multimodal Generative AI Overview

**What is it?**
Multimodal generative AI extends beyond text to include image generation, image understanding, audio generation, speech-to-text, and video generation. Models like GPT-4V, DALL-E, Whisper, and Sora process or generate multiple modalities.

**Why does it matter?**
Real-world applications often need to process images, audio, or video alongside text. Understanding multimodal capabilities helps you identify when text-only approaches are insufficient and what tools are available.

**Why learn it here?**
After mastering text generation, the learner should understand the broader landscape of generative AI to make informed decisions about which modalities their applications need.

**Prerequisites:** Unit 09.1–09.13 (text generation fundamentals).

**Mental Model:**
Multimodal models are like humans who can read, write, look at pictures, listen to audio, and even create visual art. The same transformer architecture can be extended to process different types of input (tokens for text, patches for images, waveforms for audio).

**Core Concepts:**

- text-to-image generation (diffusion models, DALL-E, Stable Diffusion)
- image understanding (vision-language models, GPT-4V)
- speech-to-text (Whisper, deepgram)
- text-to-speech (TTS)
- text-to-video (Sora, Runway)
- multimodal embeddings
- cross-modal alignment
- input/output modality combinations

**How It Works:**

1. Each modality has its own encoder (text tokenizer, image patchifier, audio encoder).
2. A shared transformer processes the combined representations.
3. The decoder generates output in the target modality.
4. Cross-modal attention aligns representations from different modalities.
5. Training on paired data (image-caption, audio-text) teaches the model to translate between modalities.

**Syntax & Implementation:**

```python
# Example: Image understanding with GPT-4V
import openai
import base64

client = openai.OpenAI()

# Encode a local image
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

image_data = encode_image("photo.jpg")

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image in detail."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        }
    ]
)
print(response.choices[0].message.content)

# Example: Image generation with DALL-E
response = client.images.generate(
    model="dall-e-3",
    prompt="A futuristic city at sunset, digital art style",
    size="1024x1024",
    quality="hd"
)
print(response.data[0].url)
```

**Simple Example:**

```python
# Multimodal capability matrix:
# Input → Output combinations:
#
# Text  → Text      : Chat, Q&A, summarization (LLM)
# Text  → Image     : Image generation (DALL-E, Stable Diffusion)
# Text  → Audio     : Text-to-speech (TTS)
# Text  → Video     : Video generation (Sora, Runway)
# Image → Text      : Image captioning, VQA (GPT-4V)
# Image → Image     : Image editing, style transfer
# Audio → Text      : Speech-to-text (Whisper)
# Audio → Audio     : Voice conversion
# Video → Text      : Video understanding
```

**Real-World Example:**
A real estate company uses multimodal AI to: analyze property photos (image → text), generate listing descriptions (text → text), create virtual staging images (text → image), and transcribe property tour videos (video → text).

**Common Mistakes:**

| Mistake | Why It Hurts |
|---|---|
| Using text-only when images would help | Missing important context |
| Assuming all models handle all modalities | Different models support different modalities |
| Ignoring API costs for images/video | Image/video tokens cost significantly more |
| Not testing multimodal input quality | Models may miss details in images or audio |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Image not processed | Wrong API or model | Check model supports vision | Use a vision-capable model |
| Audio transcription inaccurate | Background noise or accent | Check audio quality | Use noise reduction or specialized models |
| Generated images miss details | Prompt too vague | Test with more specific prompts | Add details about style, composition, content |
| High cost for multimodal requests | Image tokens expensive | Check token count for images | Resize or compress images before sending |

**Alternatives:**

| Modality | Use When | Avoid When |
|---|---|---|
| Vision (image understanding) | Visual content needs analysis | Text-only data sufficient |
| Image generation | Visual content creation needed | Text description is enough |
| Speech-to-text | Audio content needs processing | Text input available |
| Text-to-speech | Audio output needed | Text output sufficient |
| Video generation | Video content creation | Static images or text sufficient |

**Best Practices:**

- Start with text-only if it meets your needs — add modalities only when required.
- Understand the cost implications of multimodal APIs.
- Test multimodal inputs with real data, not just test cases.
- Use the simplest modality combination that solves the problem.
- Consider privacy implications of processing images and audio.

**Hands-On Practice:**

1. Basic: use a vision model to describe an image.
2. Guided: generate an image from a text prompt.
3. Independent: transcribe an audio file to text.
4. Realistic: build a pipeline that processes an image and generates a text summary.
5. Challenge: compare the quality of different image generation models on the same prompt.

**Knowledge Check:**

- What are the main input/output modality combinations?
- How do diffusion models generate images?
- What is cross-modal alignment?
- When should you use multimodal models vs text-only?

**Exit Criteria:**

- You can identify when multimodal capabilities are needed.
- You can use vision, image generation, and speech APIs.
- You can compare multimodal approaches for different tasks.

**Next Step:** Synthesize all learning from this phase into a comprehensive project.

---

### Unit 09.15 — Generative AI Synthesis & Review

**What is it?**
A cumulative integration unit that combines all Phase 09 concepts: language models, tokenization, embeddings, attention, pretraining, inference, sampling, alignment, APIs, prompt engineering, structured output, tool calling, streaming, caching, retries, and multimodal generation.

**Why does it matter?**
Knowing isolated concepts is not enough. The learner must build a complete LLM application that combines generation, structure, tools, and production patterns into a cohesive system.

**Mini Project: Robust LLM Application**

**Objective:** Build a small but complete LLM-powered application that takes messy user input, generates a structured response using an API, validates the output, handles errors gracefully, and demonstrates at least two production patterns.

**Requirements:**

- accept free-form user input (text or file)
- use prompt engineering to extract or generate structured output
- validate output against a defined schema (JSON/Pydantic)
- implement at least one tool call (e.g., search, calculation, API lookup)
- implement streaming for user-facing output
- implement caching for repeated requests
- implement retry logic for API failures
- handle edge cases (empty input, malformed output, API errors)
- include evaluation examples that test correctness
- write a README explaining design decisions, limitations, and cost estimates

**Suggested Architecture:**

```text
User input → prompt engineering → LLM API call (streaming)
    ↓
Structured output extraction → schema validation
    ↓
Tool call (if needed) → tool execution → result back to LLM
    ↓
Cache store → retry logic → final response
    ↓
Output to user + logging + cost tracking
```

**Expected Output:**

- Working application (Python script or notebook)
- README with setup, usage, design decisions, and limitations
- Evaluation report with at least 10 test cases
- Cost estimate for 100 requests
- Error handling demonstration

**Evaluation Criteria:**

- Output is valid and matches the schema
- Prompt engineering produces good results
- Tool calling works correctly
- Streaming provides responsive output
- Caching reduces redundant API calls
- Retries handle transient failures
- Error messages are helpful
- Code is organized and readable
- README is clear and complete
- Cost estimate is reasonable

**Failure Cases to Test:**

- empty user input
- very long input exceeding context window
- API rate limit (simulated)
- API server error (simulated)
- malformed model output
- tool call failure
- cache collision

**Advanced Extensions:**

- multimodal input (image or audio)
- multi-turn conversation with history
- async API calls for parallel processing
- cost budget enforcement
- performance metrics logging
- A/B testing different prompts

**Knowledge Check:**

- Why is prompt engineering important for structured output?
- How does caching affect cost and latency?
- What is the role of retry logic in production?
- How do you balance output quality with cost?

**Exit Criteria:**

- You can build a complete LLM application from scratch.
- You can combine multiple production patterns (streaming, caching, retries).
- You can debug and handle common failure modes.
- You can estimate and control costs.

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Base model vs aligned model | Research, text continuation | Assistant, Q&A, structured tasks | Raw capability vs helpful behavior |
| Greedy vs sampling | Factual tasks, code | Creative tasks, brainstorming | Determinism vs diversity |
| Temperature 0 vs temperature 0.7 | Deterministic output | Balanced creativity | Predictability vs variety |
| Prompt engineering vs fine-tuning | Quick iteration, general tasks | Consistent behavior, domain-specific | Speed vs reliability |
| JSON mode vs function calling | Simple structured output | Complex tool orchestration | Simplicity vs capability |
| Streaming vs batch | User-facing, real-time | Background processing | UX responsiveness vs simplicity |
| Local model vs API | Privacy, cost control, offline | Maximum quality, no infra management | Control vs convenience |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output is fluent but false | Hallucination or missing grounding | Fact-check against source | Add retrieval, constraints, abstention, evaluation |
| Output format breaks | Weak output contract | Validate JSON/schema | Use structured output, parsers, retry with repair |
| Costs spike | Large context or repeated calls | Log token usage | Cache, summarize, reduce context, batch requests |
| Prompt works on examples but fails in use | Overfit prompt | Test diverse cases | Build an eval set and revise systematically |
| Model refuses valid request | Over-aligned safety | Rephrase as clearly benign | Rephrase prompt or use different model |
| Streaming breaks mid-response | Network or timeout | Check connection stability | Implement reconnect and partial response handling |
| Cache serves stale data | No invalidation | Check TTL and invalidation logic | Add time-based or content-based invalidation |
| Tool call passes wrong arguments | Unclear parameter schema | Log tool call arguments | Improve parameter descriptions and examples |

---

## Phase Review Checklist

- [ ] All 15 units complete
- [ ] Can explain what a language model is and how it generates text
- [ ] Can tokenize text and estimate token cost
- [ ] Can generate and compare embeddings
- [ ] Can explain attention and transformer mechanisms
- [ ] Can describe pretraining and alignment processes
- [ ] Can implement different decoding strategies
- [ ] Can tune temperature and sampling parameters
- [ ] Can make API calls with proper authentication and error handling
- [ ] Can engineer effective prompts with examples and CoT
- [ ] Can get structured JSON output and validate it
- [ ] Can define tools and implement tool calling
- [ ] Can implement streaming, caching, and retries
- [ ] Can use multimodal APIs for image and audio tasks
- [ ] Mini project completed with all required features
- [ ] Cost and latency measured and documented

## Mastery Check

Without following a tutorial, you should be able to:

1. Explain how a language model generates text, step by step.
2. Tokenize text with a specific model's tokenizer and estimate cost.
3. Generate embeddings and compute similarity scores.
4. Implement scaled dot-product attention from scratch.
5. Explain pretraining and alignment in plain language.
6. Choose and implement appropriate decoding strategies for different tasks.
7. Write effective prompts using zero-shot, few-shot, and chain-of-thought.
8. Get structured JSON output and validate it with error handling.
9. Define tools and implement a tool-calling workflow.
10. Build a production-ready API wrapper with streaming, caching, and retries.
11. Use multimodal APIs for image understanding or generation.
12. Debug common generation failures and explain their causes.

## Interview / Explain-Back Questions

- What is next-token prediction and why is it the foundation of LLMs?
- How does tokenization affect model performance and cost?
- What are embeddings and why are they continuous vectors instead of one-hot encodings?
- Explain self-attention in simple terms. Why is it O(n²)?
- What is the difference between pretraining and fine-tuning?
- How does temperature affect the probability distribution?
- What is RLHF and why is it necessary?
- How do you handle rate limits in a production LLM application?
- What is prompt injection and how do you prevent it?
- When should you use function calling vs JSON mode?
- How do you evaluate the quality of LLM outputs?
- What are the cost implications of streaming vs batch processing?
- How would you design a caching strategy for an LLM application?
- When should you use a local model vs an API?

## Exit Criteria

Move to Phase 10 only when you can independently build a robust LLM application that combines prompt engineering, structured output, tool calling, and production patterns — and explain every design decision, failure mode, and cost trade-off.
