# Generative AI Reference

> **A quick-reference guide to Generative AI, LLMs, and RAG.**

---

## Language Models

### What is a Language Model?
A model that predicts the next token given previous tokens:
`P(tokenₙ | token₁, ..., tokenₙ₋₁)`

### Tokens
- The basic unit of text for LLMs
- Words, subwords, or characters
- Tokenization converts text ↔ tokens

### Embeddings
- Dense vector representation of tokens
- Capture semantic meaning
- Similar meanings → similar vectors

### Context Window
- Maximum number of tokens the model can consider at once
- Limits how much text can be processed

---

## Transformer Architecture (for LLMs)

### Decoder-only (GPT-style)
- Causal attention (left-to-right)
- Predicts next token
- The basis of most modern LLMs

### Key Components
- Self-attention
- Feed-forward layers
- Layer normalization
- Positional encoding

---

## Pretraining and Fine-tuning

### Pretraining
- Train on massive text corpus
- Learn language patterns
- Next-token prediction objective

### Fine-tuning
- Adapt pretrained model to a specific task
- Supervised fine-tuning (SFT)
- Instruction tuning

### Alignment
- RLHF (Reinforcement Learning from Human Feedback)
- DPO (Direct Preference Optimization)
- Make models helpful, harmless, honest

---

## Inference

### Sampling
- **Greedy**: always pick highest probability token
- **Temperature**: controls randomness
  - Low (0.1): deterministic
  - High (1.0+): creative
- **Top-k**: sample from top k tokens
- **Top-p (nucleus)**: sample from smallest set with cumulative prob > p

### Decoding
- Beam search: keep top k sequences
- Sampling: stochastic generation

---

## Prompt Engineering

### Basic Techniques
- Clear instructions
- Few-shot examples
- Chain-of-thought
- Role prompting

### Structured Output
- JSON mode
- Function/tool calling
- Output parsers

---

## LLM APIs

### Key Concepts
- **API key**: authentication (never hard-code, use env vars)
- **Rate limits**: requests per minute
- **Tokens**: billing unit
- **Streaming**: receive output incrementally
- **Retries**: handle transient failures
- **Caching**: avoid repeated identical calls

### Common Operations
- Chat completion
- Structured output
- Tool/function calling
- Embeddings

---

## RAG (Retrieval-Augmented Generation)

### Why RAG Exists
- LLMs have a knowledge cutoff
- LLMs hallucinate
- LLMs don't know private/domain data
- RAG grounds generation in retrieved documents

### RAG Pipeline
1. **Ingestion**: chunk documents, embed, store in vector DB
2. **Retrieval**: embed query, find similar chunks
3. **Generation**: LLM generates answer grounded in retrieved context

### Chunking
- Split documents into meaningful pieces
- Overlap to preserve context
- Chunk size matters (too small = lost context, too large = noise)

### Embeddings
- Convert text to vectors
- Similar text → similar vectors
- Sentence-transformers, OpenAI embeddings, etc.

### Vector Search
- Find vectors closest to query vector
- Cosine similarity, dot product, Euclidean distance
- Approximate nearest neighbor (ANN) for scale

### Vector Databases
- Chroma, FAISS, Pinecone, Weaviate, Qdrant
- Store and search embeddings efficiently

### RAG Evaluation
- **Retrieval**: context precision, context recall
- **Generation**: groundedness, faithfulness, relevance, answer correctness

---

## RAG Variants

### Naive RAG
- Simple: chunk → embed → retrieve → generate

### Advanced RAG
- Query rewriting
- Hybrid search (keyword + vector)
- Metadata filtering
- Reranking
- Parent-child retrieval

### Agentic RAG
- Agent decides when/how to retrieve
- Multi-hop retrieval
- Iterative refinement

---

## LangChain

### What It Is
- A framework for building LLM applications
- Abstractions: models, prompts, parsers, tools, chains

### Key Components
- **Models**: LLM wrappers
- **Prompts**: prompt templates
- **Output parsers**: structured output
- **Tools**: functions the model can call
- **Chains**: sequences of operations
- **Memory**: conversation history

### When to Use
- Rapid prototyping
- Standard patterns (RAG, agents)
- When you want abstractions

### When to Avoid
- Need fine-grained control
- Simple use cases (direct API is fine)
- Debugging complex issues

---

## LangGraph

### What It Is
- A framework for building stateful, graph-based agent workflows
- Nodes, edges, state, conditional routing

### Key Concepts
- **State**: shared data across nodes
- **Nodes**: functions that process state
- **Edges**: connections between nodes
- **Conditional edges**: routing based on state
- **Loops**: cycles for iterative processes

### When to Use
- Complex agent workflows
- Multi-step reasoning
- Human-in-the-loop
- Stateful agents

---

## AI Agents

### What Is an Agent?
An LLM that can:
- Use tools
- Plan
- Execute actions
- Observe results
- Iterate

### Agent Loop
1. LLM decides next action
2. Call tool
3. Observe result
4. Repeat until done

### Why Agents Fail
- Hallucinated tools
- Wrong tool selection
- Infinite loops
- Bad planning
- Context explosion
- Uncontrolled costs
- Prompt injection
- Tool injection

---

## Evaluation of LLM Systems

### Retrieval Quality
- Context precision: are retrieved docs relevant?
- Context recall: are all relevant docs retrieved?

### Generation Quality
- Groundedness: is output supported by context?
- Faithfulness: does output match context?
- Relevance: does output answer the question?
- Answer correctness: is it factually correct?

### System Metrics
- Latency
- Cost
- Tool success rate
- Agent trajectory quality

---

## Security for GenAI

### Prompt Injection
- Malicious instructions in user input
- Indirect injection via retrieved documents

### Data Leakage
- Sensitive data in prompts
- PII in training data

### Mitigations
- Input validation
- Output validation
- Least-privilege tool permissions
- Rate limiting
- Authentication/authorization
- Guardrails
