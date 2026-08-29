# Phase 08 — Transformers

> **Goal:** Master the transformer architecture — the foundation of modern NLP, LLMs, and nearly all state-of-the-art sequence models.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced  
**Priority:** Essential  
**Prerequisites:** Phase 06 (Deep Learning), Phase 07 (NLP Fundamentals)  
**Mastery target:** Level 5 — independent design decisions for transformer-based systems

---

## Why This Phase Exists

Transformers replaced RNNs and LSTMs as the dominant architecture for sequence modeling. Every major model in modern AI — BERT, GPT, T5, ViT, Whisper — is a transformer variant. Understanding transformers from the ground up lets you debug model behavior, choose the right architecture family, fine-tune effectively, and reason about trade-offs between encoder-only, decoder-only, and encoder-decoder designs.

### Phase Mental Model

A transformer is a stack of attention and feed-forward layers that lets every token in a sequence attend to every other token in parallel. Attention decides what to focus on; position decides where things are; feed-forward layers transform what attention selects.

```text
Input tokens → embeddings + positional encoding
        ↓
    Self-attention (Q, K, V) → context vector per token
        ↓
    Feed-forward network → transformed representation
        ↓
    Stack N layers → output representations
        ↓
    Task head → classification / generation / extraction
```

### What This Phase Prepares For

- LLM inference and generation (Phase 09)
- Prompt engineering and instruction tuning (Phase 09)
- Fine-tuning for specific tasks (Phase 09)
- RAG systems that rely on embeddings and attention (Phase 11)
- Model selection and architecture decisions (Phase 09–10)
- Vision transformers and multimodal models (Phase 09)

---

## Units

### Unit 08.1 — Transformer Architecture (From Scratch)

**What is it?**  
The transformer is a neural network architecture that processes sequences by computing attention over all positions simultaneously, rather than step-by-step like RNNs.

**Why does it matter?**  
It is the backbone of virtually every modern NLP and vision model. Understanding it from scratch lets you debug, modify, and design model architectures rather than treating them as black boxes.

**Why learn it here?**  
After deep learning fundamentals (Phase 06) and NLP basics (Phase 07), you have the gradient, embedding, and sequence knowledge needed to understand why transformers work and what they replaced.

**Prerequisites:** Backpropagation, embeddings, sequence models (RNN/LSTM basics), PyTorch fundamentals.

**Mental Model:**

Think of a transformer as a processing factory. Raw tokens enter as embeddings. Attention is the quality-control station where each token inspects every other token to decide what matters. The feed-forward layer is the transformation station that applies learned operations. LayerNorm keeps the signal stable. The output is a rich representation stack.

```text
Tokens → Embedding Layer
    → Positional Encoding added
    → Repeat N times:
        → Multi-Head Self-Attention
        → Add & LayerNorm (residual)
        → Feed-Forward Network
        → Add & LayerNorm (residual)
    → Output representation
```

**Core Concepts:**

- Encoder vs decoder vs encoder-decoder
- Self-attention mechanism (detailed in 08.2)
- Positional encoding (detailed in 08.3)
- Multi-head attention
- Residual connections
- Layer normalization
- Masked attention (causal mask)
- Cross-attention (encoder-decoder)

**How It Works:**

1. Input tokens are converted to dense vectors via an embedding layer.
2. Positional encodings are added to inject sequence order.
3. Each layer computes self-attention: every token builds a query, and every token provides a key-value pair. Attention scores determine how much each token attends to every other.
4. The attended output passes through a position-wise feed-forward network.
5. Residual connections and layer normalization stabilize training.
6. In decoders, a causal mask prevents attending to future tokens.
7. In encoder-decoder models, cross-attention lets decoder tokens attend to encoder outputs.

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

# Example: stack 3 layers
block = TransformerBlock(d_model=512, n_heads=8, d_ff=2048)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
out = block(x)
print(out.shape)  # torch.Size([2, 10, 512])
```

**Simple Example:**

```python
import torch.nn as nn

d_model = 512
n_heads = 8
x = torch.randn(1, 20, d_model)  # 1 batch, 20 tokens, 512 dims

block = TransformerBlock(d_model, n_heads, d_ff=2048)
output = block(x)
print(f"Input:  {x.shape}")   # [1, 20, 512]
print(f"Output: {output.shape}")  # [1, 20, 512]
# Same shape — each token now has context-aware representation
```

**Real-World Example:**

GPT-4, Claude, and Llama all use transformer decoder blocks. BERT uses transformer encoder blocks. T5 uses full encoder-decoder. Vision Transformers (ViT) treat image patches as tokens and feed them through encoder blocks. The same building block powers all of these.

**Common Mistakes:**

- Confusing encoder-only, decoder-only, and encoder-decoder architectures
- Forgetting positional encoding (transformers have no inherent order sense)
- Ignoring the causal mask in decoder models (causes information leakage)
- Not understanding residual connections (vanishing gradients in deep stacks)
- Assuming attention replaces the need for positional information

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output is all zeros | Uninitialized weights or NaN gradients | Check `model.named_parameters()` for NaN | Use proper initialization, check learning rate |
| Attention weights are uniform | Model not learning, too much regularization | Print attention weights | Reduce dropout, check data |
| Loss doesn't decrease | Missing mask, wrong loss function, bad data | Inspect loss per epoch, check shapes | Add causal mask, verify label alignment |
| CUDA out of memory | Batch too large or model too deep | `torch.cuda.max_memory_allocated()` | Reduce batch size, use gradient checkpointing |
| Training NaN after a few epochs | Exploding gradients | Monitor gradient norms | Use gradient clipping, lower learning rate |

**Alternatives:**

| Architecture | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Transformer | Most sequence tasks, parallelizable | Very long sequences (>100K tokens) without efficient attention | Excellent performance but quadratic memory |
| RNN/LSTM | Very long sequences, streaming | Need parallelization or long-range attention | Sequential but can handle unbounded length |
| State Space Models (Mamba) | Long sequences, fast inference | Need explicit attention-based retrieval | Linear scaling but less interpretable |
| Linear Attention | Resource-constrained, long sequences | Need high-accuracy attention | Faster but lower quality |

**Best Practices:**

- Start with small models to verify training works before scaling
- Use pre-norm (LayerNorm before attention) for deeper models — it trains more stably
- Monitor attention patterns as a debugging tool
- Use gradient clipping for stable training
- Always check the masking behavior matches your task

**Hands-On Practice:**

1. **Basic:** Implement a single TransformerBlock and verify input/output shapes.
2. **Guided:** Stack 4 blocks, pass a sequence, print output shapes at each layer.
3. **Independent:** Build a small encoder-only model for sequence classification.
4. **Realistic:** Train it on a text classification dataset and debug training issues.
5. **Challenge:** Add a causal mask and verify that position i cannot attend to position i+1.

**Knowledge Check:**

- Why are residual connections essential in deep transformers?
- What happens if you remove positional encoding?
- How does a decoder block differ from an encoder block?
- Why is pre-norm preferred over post-norm in deep models?

**Exit Criteria:**

- You can implement a transformer block from scratch.
- You can explain every sub-layer and its purpose.
- You can debug shape mismatches and training instability.

**Next Step:** Deep dive into self-attention — the core mechanism.

---

### Unit 08.2 — Self-Attention Deep Dive

**What is it?**  
Self-attention is the mechanism that lets each token in a sequence compute a weighted sum of all other tokens' representations, where the weights are learned based on relevance.

**Why does it matter?**  
Self-attention is the single most important operation in transformers. Every architectural variant (BERT, GPT, T5, ViT) revolves around different flavors of attention. Understanding it deeply lets you debug models, interpret attention patterns, and design variants.

**Why learn it here?**  
You just built a transformer block. Now you need to understand exactly what happens inside the attention sub-layer — the math, the shapes, and the intuition.

**Prerequisites:** Matrix multiplication, dot products, softmax, the transformer overview from 08.1.

**Mental Model:**

Self-attention is like a library lookup. Each token writes a "query" (what am I looking for?), a "key" (what do I contain?), and a "value" (what do I provide?). The query compares against every key via dot product. High similarity means high attention weight. The output is a weighted blend of all values.

```text
Q = X @ W_q    (what each token is looking for)
K = X @ W_k    (what each token offers)
V = X @ W_v    (what each token contributes)

Attention(Q,K,V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

**Core Concepts:**

- Query, Key, Value projections
- Scaled dot-product attention
- Attention scores and attention weights
- Multi-head attention
- Causal masking
- Cross-attention
- Attention complexity: O(n²) in sequence length

**How It Works:**

1. Each input token has a d_model-dimensional vector.
2. Three learned weight matrices project X into Q, K, and V (each d_model → d_k, d_k, d_v).
3. Compute attention scores: Q @ K^T produces an n×n matrix of dot products.
4. Scale by sqrt(d_k) to prevent dot products from growing too large (which saturates softmax).
5. Apply softmax to get weights (each row sums to 1).
6. Multiply weights by V to get the output.
7. In multi-head attention, this happens in parallel for h heads, each with smaller d_k = d_model / h, and results are concatenated.

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    return torch.matmul(weights, V), weights

# Example
batch, seq_len, d_model = 1, 5, 64
n_heads = 8
d_k = d_model // n_heads  # 8

Q = torch.randn(batch, n_heads, seq_len, d_k)
K = torch.randn(batch, n_heads, seq_len, d_k)
V = torch.randn(batch, n_heads, seq_len, d_k)

output, attn_weights = scaled_dot_product_attention(Q, K, V)
print(output.shape)       # [1, 8, 5, 8]
print(attn_weights.shape) # [1, 8, 5, 5]
```

**Simple Example:**

```python
# Causal mask: position 0 can only attend to position 0
# Position 1 can attend to 0 and 1, etc.
seq_len = 5
causal_mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
print(causal_mask)
# tensor([[[[1, 0, 0, 0, 0],
#           [1, 1, 0, 0, 0],
#           [1, 1, 1, 0, 0],
#           [1, 1, 1, 1, 0],
#           [1, 1, 1, 1, 1]]]])

Q = torch.randn(1, 8, 5, 8)
K = torch.randn(1, 8, 5, 8)
V = torch.randn(1, 8, 5, 8)

output, weights = scaled_dot_product_attention(Q, K, V, mask=causal_mask)
# Each position can only attend to itself and previous positions
```

**Real-World Example:**

In BERT, self-attention is bidirectional — every token attends to every other token. This is why BERT excels at understanding tasks: the representation of "bank" in "river bank" benefits from seeing "river." In GPT, causal masking ensures each token only attends to previous tokens, which is essential for autoregressive generation.

**Common Mistakes:**

- Forgetting to scale by sqrt(d_k) — causes softmax to be too peaked
- Not applying causal mask in decoder — leads to information leakage
- Using the wrong dimension for Q/K/V projections
- Confusing attention weights with attention output
- Ignoring multi-head concatenation (concat then project, not average)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Attention weights all near-equal | Dot products too small, softmax saturated | Check Q/K magnitude | Scale by sqrt(d_k), check initialization |
| Attention weights all 0/1 | Scores too large, softmax peaked | Check raw scores before softmax | Reduce learning rate, add scaling |
| Decoder leaks future info | Missing causal mask | Print attention weights — should be upper-triangular zeros | Apply `torch.tril` mask |
| Gradient explodes in attention | Large d_k, no scaling | Print gradient norms | Use scaled attention, gradient clipping |
| Multi-head output wrong shape | Forgot concatenation | Check concat dim | `concat(dim=-1)` then linear projection |

**Alternatives:**

| Mechanism | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Scaled dot-product attention | Most tasks, well-understood | Very long sequences | Quadratic but reliable |
| FlashAttention | Long sequences, GPU training | Need full attention matrix | Faster and memory-efficient |
| Linear attention | Very long sequences, limited memory | Need exact attention | Approximate but linear scaling |
| Sparse attention | Specific patterns (local + global) | Need dense attention | Efficient but pattern-dependent |

**Best Practices:**

- Always use scaled dot-product attention (the scaling is not optional)
- Visualize attention weights as a debugging tool
- For classification, use the [CLS] token output, not average pooling, unless you benchmark both
- Use `torch.nn.MultiheadAttention` for production code — it handles the math correctly
- When interpreting attention, remember that attention weights ≠ feature importance

**Hands-On Practice:**

1. **Basic:** Implement scaled dot-product attention from scratch and verify output shapes.
2. **Guided:** Add a causal mask and show that future positions are blocked.
3. **Independent:** Implement multi-head attention with learned Q/K/V projections.
4. **Realistic:** Compare attention patterns between a trained model on easy vs hard examples.
5. **Challenge:** Implement efficient attention with FlashAttention or chunked attention.

**Knowledge Check:**

- Why do we scale by sqrt(d_k) instead of d_k?
- How does causal masking change the attention computation?
- What is the difference between self-attention and cross-attention?
- Why do we use multiple attention heads instead of one large head?

**Exit Criteria:**

- You can implement self-attention from scratch.
- You can explain Q, K, V, scaling, and masking.
- You can interpret attention weight patterns.

**Next Step:** Learn positional encoding — how transformers know sequence order.

---

### Unit 08.3 — Positional Encoding

**What is it?**  
Positional encoding adds information about token position to the input embeddings, since transformers process all tokens simultaneously and have no inherent sense of order.

**Why does it matter?**  
Without positional information, "dog bites man" and "man bites dog" are identical to a transformer. Positional encoding is what gives the model sequence awareness.

**Why learn it here?**  
You understand attention now — attention computes over all positions equally. The next logical question is: how does the model know which position is which?

**Prerequisites:** Self-attention (08.2), sine/cosine functions, embedding vectors.

**Mental Model:**

Positional encoding is like adding GPS coordinates to each token's location. The attention mechanism can see all tokens, but positional encoding tells it where each token is in the sequence. Sinusoidal encodings create a unique "fingerprint" for each position using waves of different frequencies.

**Core Concepts:**

- Why transformers need positional information
- Absolute positional encoding (sinusoidal)
- Learned positional embeddings
- Relative positional encoding (RoPE, ALiBi)
- RoPE (Rotary Position Embeddings) — used in modern LLMs
- Position encoding vs positional embedding

**How It Works:**

Sinusoidal encoding uses sine and cosine functions at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

Each position gets a unique vector. Low-frequency dimensions change slowly (capture coarse position), high-frequency dimensions change quickly (capture fine position). The encoding is added to the embedding, not concatenated.

**Syntax & Implementation:**

```python
import torch
import math

def sinusoidal_positional_encoding(max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len).unsqueeze(1).float()
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]

# Generate encodings
pe = sinusoidal_positional_encoding(max_len=100, d_model=512)
print(pe.shape)  # torch.Size([1, 100, 512])

# Add to embeddings
embedding = torch.randn(1, 20, 512)  # batch=1, seq_len=20, d_model=512
x = embedding + pe[:, :20, :]
print(x.shape)  # [1, 20, 512]
```

**Simple Example:**

```python
pe = sinusoidal_positional_encoding(max_len=10, d_model=8)
# Position 0 and Position 1 have different encodings
print("Position 0:", pe[0, 0])
print("Position 1:", pe[0, 1])
# Similar positions have similar encodings (nearby = close in vector space)
sim_0_1 = torch.cosine_similarity(pe[0, 0].unsqueeze(0), pe[0, 1].unsqueeze(0))
sim_0_5 = torch.cosine_similarity(pe[0, 0].unsqueeze(0), pe[0, 5].unsqueeze(0))
print(f"Similarity(0,1): {sim_0_1:.3f}")  # Higher
print(f"Similarity(0,5): {sim_0_5:.3f}")  # Lower
```

**Real-World Example:**

GPT-2 used learned positional embeddings. LLaMA and modern LLMs use RoPE (Rotary Position Embeddings), which rotates query and key vectors based on position — this naturally captures relative distances and generalizes to longer sequences than seen during training.

**Common Mistakes:**

- Concatenating positional encoding instead of adding it
- Using positional encoding after the first layer (should be at input)
- Not scaling embeddings after adding positional encoding
- Using learned embeddings when generalization to longer sequences is needed
- Forgetting to truncate positional encoding to match sequence length

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model ignores position info | Positional encoding too small relative to embeddings | Check magnitude of PE vs embedding | Scale PE or use learnable scaling |
| Model fails on longer sequences | Learned embeddings not trained on long sequences | Test with sequences > training length | Use sinusoidal or RoPE |
| Performance degrades with sequence length | Quadratic attention + long positions | Profile memory and time | Use FlashAttention or sliding window |

**Alternatives:**

| Method | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Sinusoidal (absolute) | General purpose, no training needed | Need relative distance explicitly | Simple, generalizes well |
| Learned embeddings | Fixed max length, task-specific | Need to generalize to longer sequences | Flexible but limited |
| RoPE (rotary) | Long-context LLMs, relative position | Simple models where absolute is enough | Best generalization, widely used now |
| ALiBi | Long sequences, no extra parameters | Need fine-grained position encoding | Simple, no added parameters |

**Best Practices:**

- Always add positional encoding at the input, before any transformer layers
- For modern LLMs, prefer RoPE — it's the standard for long-context models
- When fine-tuning, check if the model's positional encoding supports your sequence length
- Visualize positional encoding similarity matrices to understand the encoding

**Hands-On Practice:**

1. **Basic:** Implement sinusoidal positional encoding and print the similarity matrix.
2. **Guided:** Add positional encoding to embeddings and verify the model can distinguish token order.
3. **Independent:** Train two small models — one with and one without positional encoding — on a word-order-sensitive task.
4. **Realistic:** Compare learned vs sinusoidal encoding on a classification task.
5. **Challenge:** Implement RoPE and verify it generalizes to sequences longer than training.

**Knowledge Check:**

- Why must positional encoding be added, not concatenated?
- How does RoPE differ from sinusoidal encoding?
- What happens if you use learned positional embeddings on sequences longer than training?
- Why do low-frequency sine waves capture coarse position?

**Exit Criteria:**

- You can implement positional encoding from scratch.
- You can explain why transformers need positional information.
- You can choose between positional encoding methods for different scenarios.

**Next Step:** Combine encoder and decoder into the full seq2seq architecture.

---

### Unit 08.4 — Encoder-Decoder Architecture

**What is it?**  
The encoder-decoder architecture processes an input sequence through an encoder stack, then generates an output sequence through a decoder stack using cross-attention to the encoder's representations.

**Why does it matter?**  
This is the original transformer design from "Attention Is All You Need" (2017). It powers translation, summarization, and any task where input and output have different lengths or structures.

**Why learn it here?**  
You understand encoder blocks and decoder blocks separately. Now you need to see how they connect — particularly cross-attention and teacher forcing.

**Prerequisites:** Self-attention (08.2), positional encoding (08.3), transformer blocks (08.1).

**Mental Model:**

The encoder reads the entire input and builds a rich representation (like reading a whole sentence). The decoder generates output one token at a time, looking back at what it already generated (causal mask) and referencing the encoder's representation (cross-attention) for each new token.

```text
Input → [Encoder Stack] → Memory (encoder output)
                                   ↓
                            [Decoder Stack] ← previously generated tokens
                                   ↓
                              Output token
```

**Core Concepts:**

- Encoder stack: self-attention + feed-forward
- Decoder stack: masked self-attention + cross-attention + feed-forward
- Cross-attention: decoder queries attend to encoder keys/values
- Teacher forcing: training with gold previous tokens
- Teacher forcing ratio: mixing gold and predicted tokens
- Autoregressive generation: one token at a time

**How It Works:**

1. **Encoder:** Processes the full input sequence. Each layer has self-attention (bidirectional) and a feed-forward network. Output is a sequence of context-rich vectors.
2. **Cross-attention:** In the decoder, Q comes from the decoder, K and V come from the encoder. This lets each decoder position attend to all encoder positions.
3. **Decoder:** Generates tokens autoregressively. Masked self-attention prevents attending to future positions. Cross-attention references the encoder.
4. **Training:** Teacher forcing — the decoder receives the correct previous token, not its own prediction.
5. **Inference:** The decoder generates token by token, feeding each prediction as input to the next step.

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 512, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, n_heads, d_ff, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model, n_heads, d_ff, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers)
        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_emb = self.embedding(src) + self.pos_enc[:, :src.size(1), :]
        tgt_emb = self.embedding(tgt) + self.pos_enc[:, :tgt.size(1), :]

        memory = self.encoder(src_emb, mask=src_mask)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        return self.output_proj(output)

# Usage
model = EncoderDecoderTransformer(vocab_size=10000)
src = torch.randint(0, 10000, (2, 30))  # batch=2, src_len=30
tgt = torch.randint(0, 10000, (2, 20))  # batch=2, tgt_len=20
output = model(src, tgt)
print(output.shape)  # [2, 20, 10000] — logits over vocab for each target position
```

**Simple Example:**

```python
# Translation: English → French
# Encoder sees: "The cat sat on the mat"
# Decoder generates: "Le" → "chat" → "assis" → "sur" → "le" → "tapis"
# Each decoder step uses cross-attention to the encoder
```

**Real-World Example:**

T5 (Text-to-Text Transfer Transformer) treats every NLP task as encoder-decoder: classification becomes "classify: <text> → <label>", translation becomes "translate English to French: <text> → <translation>". BART also uses encoder-decoder for summarization and generation tasks.

**Common Mistakes:**

- Forgetting the causal mask in the decoder (allows cheating during training)
- Using bidirectional attention in the decoder (breaks autoregressive generation)
- Not using teacher forcing correctly (feeding model's own predictions during training)
- Confusing cross-attention with self-attention
- Not sharing the embedding matrix between encoder and decoder

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Decoder generates repetitive tokens | Causal mask missing or too permissive | Check mask shape | Verify lower-triangular mask |
| Training loss doesn't decrease | Cross-attention not connected | Check memory connection | Pass encoder output to decoder |
| Model performs well in training, poorly in inference | Teacher forcing dependency | Compare teacher forcing vs autoregressive | Use scheduled sampling |
| Output length always matches input | No end-of-sequence handling | Check if model learns EOS | Add EOS token, use greedy/beam search |

**Alternatives:**

| Architecture | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Encoder-decoder (T5, BART) | Seq2seq: different input/output lengths | Simple classification or generation | Flexible but more parameters |
| Decoder-only (GPT) | Generation, instruction following | Tasks needing deep input understanding | Simpler, but no explicit cross-attention |
| Encoder-only (BERT) | Classification, extraction | Generation tasks | Fast inference, no generation capability |

**Best Practices:**

- Share the embedding matrix between encoder and decoder (reduces parameters)
- Use scheduled sampling to reduce teacher forcing bias
- For translation, use subword tokenization (BPE) to handle unknown words
- Always include a special EOS token and train the model to predict it

**Hands-On Practice:**

1. **Basic:** Connect an encoder and decoder with cross-attention and verify shapes.
2. **Guided:** Train a small seq2seq model on a simple task (e.g., reverse a sequence).
3. **Independent:** Build a translation model with BPE tokenization.
4. **Realistic:** Compare teacher forcing vs scheduled sampling on generation quality.
5. **Challenge:** Implement beam search and compare with greedy decoding.

**Knowledge Check:**

- How does cross-attention differ from self-attention in the decoder?
- What is teacher forcing and what problem does it cause?
- Why does the decoder need a causal mask but the encoder does not?
- How do you generate sequences at inference time?

**Exit Criteria:**

- You can implement an encoder-decoder transformer.
- You can explain cross-attention and teacher forcing.
- You can build a seq2seq model for a translation-like task.

**Next Step:** Explore BERT — the encoder-only variant for understanding tasks.

---

### Unit 08.5 — BERT-style Models

**What is it?**  
BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only transformer pre-trained on masked language modeling and next-sentence prediction. It produces contextual embeddings for understanding tasks.

**Why does it matter?**  
BERT revolutionized NLP by showing that pre-training a large transformer encoder on massive text, then fine-tuning on specific tasks, dramatically improves performance. It's the foundation for understanding-focused models.

**Why learn it here?**  
After understanding encoder-decoder architecture, BERT is the natural first specialized variant — it uses only the encoder stack and introduces masked language modeling as a pre-training objective.

**Prerequisites:** Transformer architecture (08.1), self-attention (08.2), positional encoding (08.3).

**Mental Model:**

BERT is like a reading comprehension expert. It reads the entire sentence at once (bidirectional) and fills in blanks to learn language. For downstream tasks, it produces a rich representation of each token that captures context from both left and right.

```text
Pre-training (masked LM):
Input:  "The cat [MASK] on the mat"
Target: "sat"
→ Model learns to predict missing words using full context

Fine-tuning (classification):
Input:  "[CLS] This movie is great [SEP]"
Output: Positive
→ Model uses [CLS] token representation for classification
```

**Core Concepts:**

- Bidirectional attention (no causal mask)
- Masked Language Modeling (MLM)
- Next Sentence Prediction (NSP) — now mostly obsolete
- [CLS] and [SEP] special tokens
- Fine-tuning for classification, NER, QA, similarity
- BERT variants: RoBERTa, ALBERT, DistilBERT, DeBERTa

**How It Works:**

1. Pre-training: Randomly mask 15% of tokens. The model predicts the original token from context. This forces deep bidirectional understanding.
2. Fine-tuning: Add a task-specific head (e.g., linear layer for classification) and train on labeled data.
3. Inference: The encoder produces contextual representations. Use [CLS] for sentence-level tasks, token outputs for token-level tasks.

**Syntax & Implementation:**

```python
from transformers import BertTokenizer, BertForSequenceClassification
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize
inputs = tokenizer("This movie is fantastic!", return_tensors="pt")
print(inputs.keys())  # input_ids, token_type_ids, attention_mask

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    print(logits.shape)  # [1, 2] — binary classification

# Predicted class
predicted = torch.argmax(logits, dim=-1)
print(f"Predicted class: {predicted.item()}")
```

**Simple Example:**

```python
# Masked Language Modeling with BERT
from transformers import BertForMaskedLM, BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')

text = "The capital of France is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
mask_idx = (inputs.input_ids == tokenizer.mask_token_id).nonzero()[0, 1]

with torch.no_grad():
    outputs = model(**inputs)
    predictions = outputs.logits[0, mask_idx].argmax(dim=-1)
    print(tokenizer.decode([predictions]))  # "paris"
```

**Real-World Example:**

Search engines use BERT embeddings to understand query intent. Sentiment analysis systems fine-tune BERT on labeled reviews. Named entity recognition uses BERT token outputs to label each word. Question answering systems use BERT to extract answer spans from context.

**Common Mistakes:**

- Using BERT for generation (it's not designed for that — use GPT)
- Fine-tuning with too high a learning rate (destroys pre-trained representations)
- Not using [CLS] token correctly for classification
- Ignoring the attention mask during fine-tuning
- Not freezing layers when data is small

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Fine-tuning overfits quickly | Learning rate too high, small dataset | Check train/val loss curves | Lower LR (2e-5 to 5e-5), freeze lower layers |
| Poor performance on short texts | BERT has max 512 tokens | Check token count | Truncate or use Longformer |
| [CLS] output is poor | Not trained with [CLS] in mind | Check tokenizer output | Fine-tune with proper classification head |
| Memory error on long sequences | 512 token limit × batch size | Check sequence length | Use sliding window or Longformer |

**Alternatives:**

| Model | Use When | Avoid When | Trade-off |
|---|---|---|---|
| BERT | Classification, NER, QA, similarity | Generation tasks | Great for understanding, no generation |
| RoBERTa | Better BERT performance | Need smaller model | Same architecture, better pre-training |
| DistilBERT | Latency-constrained deployment | Need maximum accuracy | 40% smaller, 97% of BERT's performance |
| DeBERTa | Need disentangled attention | Simple tasks | State-of-the-art on many benchmarks |

**Best Practices:**

- Start with `bert-base-uncased` unless you have a reason to go larger
- Use learning rates between 2e-5 and 5e-5 for fine-tuning
- Freeze lower layers for small datasets
- Use the Hugging Face `Trainer` API for standard fine-tuning workflows
- Always set a random seed for reproducibility

**Hands-On Practice:**

1. **Basic:** Load BERT and extract embeddings for a sentence.
2. **Guided:** Fine-tune BERT for sentiment classification on a small dataset.
3. **Independent:** Fine-tune BERT for NER using token-level outputs.
4. **Realistic:** Compare BERT-base vs DistilBERT on accuracy and speed.
5. **Challenge:** Fine-tune BERT for extractive QA and evaluate with exact match/F1.

**Knowledge Check:**

- Why is BERT bidirectional but GPT is not?
- What does the [CLS] token represent?
- How does MLM pre-training help with downstream tasks?
- When would you choose BERT over GPT?

**Exit Criteria:**

- You can use BERT for classification, NER, and QA tasks.
- You can fine-tune BERT with proper hyperparameters.
- You can choose between BERT variants for different constraints.

**Next Step:** Explore GPT-style causal language models.

---

### Unit 08.6 — Causal Language Models (GPT-style)

**What is it?**  
Causal language models are decoder-only transformers that predict the next token in a sequence. They use causal masking to ensure each position can only attend to previous positions.

**Why does it matter?**  
GPT-style models power ChatGPT, Claude, Llama, and virtually all modern chat and generation systems. Understanding causal language modeling is essential for working with LLMs.

**Why learn it here?**  
After BERT (encoder-only), the decoder-only variant completes the picture. You now understand both directions of the encoder-decoder spectrum.

**Prerequisites:** Transformer architecture (08.1), self-attention with causal masking (08.2), positional encoding (08.3).

**Mental Model:**

A causal LM is like a writer who can only look at what they've already written. Each new word is predicted based on the entire previous context, but never on future words. This makes it naturally suited for generation.

```text
Context: "The capital of France is"
Model sees: [The] [capital] [of] [France] [is]
Model predicts: Paris

At inference: generate one token at a time
"The" → "capital" → "of" → "France" → "is" → "Paris"
```

**Core Concepts:**

- Causal (autoregressive) attention masking
- Next-token prediction objective
- Teacher forcing during training
- Greedy decoding, beam search, sampling
- Temperature and top-k/top-p sampling
- KV-cache for efficient generation
- Scaling laws and emergent abilities

**How It Works:**

1. Training: Feed the model a sequence and predict each token from previous tokens. Loss is cross-entropy on each position.
2. Inference: Start with a prompt, generate one token at a time, append it to the context, repeat.
3. Causal mask ensures token i can only attend to tokens 0..i-1.
4. Generation strategies control the trade-off between quality and diversity.

**Syntax & Implementation:**

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Generate text
prompt = "The future of AI is"
input_ids = tokenizer.encode(prompt, return_tensors='pt')

# Greedy decoding
output = model.generate(input_ids, max_length=50, do_sample=False)
print(tokenizer.decode(output[0]))

# Sampling with temperature
output = model.generate(
    input_ids,
    max_length=50,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
print(tokenizer.decode(output[0]))
```

**Simple Example:**

```python
# Compare temperature effects
import torch

logits = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

# Temperature = 1.0 (standard)
probs_t1 = torch.softmax(logits / 1.0, dim=0)
# Temperature = 0.5 (more confident)
probs_t05 = torch.softmax(logits / 0.5, dim=0)
# Temperature = 2.0 (more random)
probs_t2 = torch.softmax(logits / 2.0, dim=0)

print("T=1.0:", probs_t1)  # [0.0117, 0.0317, 0.0861, 0.2341, 0.6364]
print("T=0.5:", probs_t05)  # [0.0003, 0.0026, 0.0217, 0.1835, 0.7919]
print("T=2.0:", probs_t2)   # [0.0895, 0.1452, 0.2109, 0.2511, 0.3032]
# Lower temperature → more peaked (confident)
# Higher temperature → more uniform (creative)
```

**Real-World Example:**

GPT-4 generates responses by predicting one token at a time. When you see streaming output, each word appears as the model generates it. Chat applications use temperature and sampling to balance between factual responses (low temperature) and creative writing (high temperature).

**Common Mistakes:**

- Using BERT-style bidirectional attention for generation
- Not using the causal mask (causes information leakage)
- Setting temperature too low (repetitive) or too high (incoherent)
- Not using KV-cache (slow generation)
- Ignoring repetition penalty (causes loops)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Repetitive output | Temperature too low, no repetition penalty | Check generation params | Increase temperature, add repetition_penalty |
| Incoherent output | Temperature too high | Print probabilities | Lower temperature, use top-p |
| Very slow generation | Not using KV-cache | Profile generation time | Enable `use_cache=True` |
| Output truncates early | EOS token generated prematurely | Check EOS handling | Adjust `eos_token_id` or `min_length` |
| Model generates same output every time | `do_sample=False` | Check sampling settings | Set `do_sample=True` for variation |

**Alternatives:**

| Approach | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Greedy decoding | Factual/extractive tasks | Need diversity | Fast but repetitive |
| Beam search | Translation, summarization | Open-ended generation | Better quality, more compute |
| Top-k sampling | Creative generation | Need deterministic output | Diverse but less controlled |
| Top-p (nucleus) sampling | Balanced creativity and quality | Need strict control | Adaptive vocabulary pruning |
| Temperature scaling | Control randomness | Need fixed behavior | Simple diversity lever |

**Best Practices:**

- Use `do_sample=True` with temperature 0.7–1.0 for chat applications
- Always set a `max_length` to prevent infinite generation
- Use KV-cache for efficient autoregressive generation
- Monitor for repetition loops and add penalties if needed
- For production, batch generation requests for throughput

**Hands-On Practice:**

1. **Basic:** Generate text with GPT-2 using greedy decoding.
2. **Guided:** Compare different temperatures and sampling strategies.
3. **Independent:** Build a text completion system with custom prompts.
4. **Realistic:** Implement beam search and compare with greedy/sampling.
5. **Challenge:** Build a simple chatbot with context window management.

**Knowledge Check:**

- Why can't BERT do what GPT does (and vice versa)?
- What is the purpose of temperature in generation?
- How does KV-cache speed up autoregressive generation?
- What is the difference between top-k and top-p sampling?

**Exit Criteria:**

- You can use causal LMs for text generation.
- You understand generation strategies and their trade-offs.
- You can build a basic text generation pipeline.

**Next Step:** Understand tokenizers — how text becomes numbers.

---

### Unit 08.7 — Tokenizers (BPE, WordPiece)

**What is it?**  
Tokenizers convert raw text into numerical tokens that models can process. Subword tokenization methods like BPE and WordPiece split rare words into meaningful pieces while keeping common words as single tokens.

**Why does it matter?**  
Tokenization is the bridge between raw text and model input. A bad tokenizer causes vocabulary mismatches, poor generation, and wasted capacity. Every transformer model depends on its tokenizer.

**Why learn it here?**  
You've used tokenizers implicitly in 08.5 and 08.6. Now you need to understand how they work, because tokenizer choice affects model performance, vocabulary size, and handling of rare words.

**Prerequisites:** String processing, vocabulary concepts, basic probability.

**Mental Model:**

A tokenizer is like a smart compression algorithm for text. It keeps frequent words whole ("the", "and") and breaks rare words into pieces ("transform" + "##er" + "##s"). BPE builds this by iteratively merging the most common character pairs. WordPiece does the same but uses a likelihood-based merge criterion.

```text
Raw text: "Tokenization is fascinating"
BPE tokens: ["Token", "ization", " is", " fasc", "inating"]
WordPiece tokens: ["Token", "ization", " is", "fasc", "##inating"]
```

**Core Concepts:**

- Character-level vs word-level vs subword tokenization
- Byte Pair Encoding (BPE)
- WordPiece (used by BERT)
- SentencePiece (language-agnostic)
- Unigram tokenization (used by T5, Llama)
- Special tokens: [CLS], [SEP], [PAD], [MASK], <s>, </s>
- Vocabulary size trade-offs

**How It Works:**

**BPE Training:**
1. Start with individual characters as the vocabulary.
2. Count all adjacent pairs in the corpus.
3. Merge the most frequent pair into a new token.
4. Repeat until desired vocabulary size is reached.

**WordPiece:**
Same idea, but merges are chosen based on likelihood gain rather than raw frequency.

**Syntax & Implementation:**

```python
from transformers import AutoTokenizer

# BERT tokenizer (WordPiece)
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = bert_tokenizer.tokenize("Tokenization is fascinating")
print(tokens)  # ['token', '##ization', 'is', 'fasc', '##inating']

# GPT-2 tokenizer (BPE)
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokens = gpt2_tokenizer.tokenize("Tokenization is fascinating")
print(tokens)  # ['Token', 'ization', ' is', ' fasc', 'inating']

# Compare vocabulary sizes
print(f"BERT vocab: {bert_tokenizer.vocab_size}")   # 30522
print(f"GPT-2 vocab: {gpt2_tokenizer.vocab_size}")  # 50257

# Encode and decode
encoded = gpt2_tokenizer.encode("Hello world!")
print(encoded)        # [15496, 995, 0]
print(gpt2_tokenizer.decode(encoded))  # "Hello world!"
```

**Simple Example:**

```python
# How BPE builds vocabulary
corpus = "low low low low low lower lower newest newest newest widest"
# Step 1: Character vocab: l, o, w, e, r, n, s, t, i, d, ' '
# Step 2: Most frequent pair: l+o → "lo" (appears 5 times)
# Step 3: Most frequent pair: lo+w → "low" (appears 5 times)
# Step 4: Most frequent pair: e+s → "es"
# Step 5: Continue until vocab size reached
# Result: ["low", " low", " low", ...] efficiently encoded
```

**Real-World Example:**

When GPT-4 processes your prompt, it first tokenizes your text into subword tokens. If you type "antidisestablishmentarianism", BPE splits it into ["anti", "dis", "establish", "ment", "arian", "ism"]. This lets the model handle rare words without a massive vocabulary.

**Common Mistakes:**

- Using the wrong tokenizer for a model (BERT tokenizer for GPT model)
- Not handling special tokens correctly
- Assuming token count = word count
- Ignoring how tokenization affects cost (API pricing is per token)
- Not tokenizing input the same way during training and inference

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model outputs garbage | Wrong tokenizer used | Check tokenizer model name | Use matching tokenizer |
| Unknown tokens in output | Input outside vocabulary | Check `[UNK]` tokens | Use BPE/SentencePiece for open vocab |
| Token count different from expected | Different tokenizers, different merging | Compare tokenizers side by side | Standardize on one tokenizer |
| Poor performance on non-English text | Tokenizer trained on English | Check training data | Use multilingual tokenizer |

**Alternatives:**

| Tokenizer | Use When | Avoid When | Trade-off |
|---|---|---|---|
| BPE (GPT, RoBERTa) | Most tasks, well-tested | Need language-agnostic | Widely supported, simple |
| WordPiece (BERT) | BERT models, established pipelines | Need newer features | Slightly different merge strategy |
| SentencePiece (T5, Llama) | Multilingual, language-agnostic | Need pre-tokenization | Works on raw text, no pre-processing |
| Unigram | Probabilistic, flexible vocab | Need deterministic tokenization | Can sample multiple tokenizations |

**Best Practices:**

- Always use the tokenizer that matches your model
- Pay attention to special tokens ([CLS], [SEP], etc.) — they affect model behavior
- For API-based models, be aware that token count directly affects cost
- Test your tokenizer on edge cases: emoji, numbers, URLs, non-English text
- Save and version your tokenizer alongside your model

**Hands-On Practice:**

1. **Basic:** Load BPE and WordPiece tokenizers, compare their output on the same text.
2. **Guided:** Train a simple BPE tokenizer on a small corpus.
3. **Independent:** Compare tokenization of rare words, URLs, and code across tokenizers.
4. **Realistic:** Calculate how tokenization affects API cost for a real use case.
5. **Challenge:** Implement a minimal BPE tokenizer from scratch.

**Knowledge Check:**

- Why is subword tokenization preferred over word-level?
- How does BPE decide which pairs to merge?
- What special tokens does BERT need vs GPT?
- How does vocabulary size affect model performance and cost?

**Exit Criteria:**

- You can explain BPE and WordPiece algorithms.
- You can use tokenizers correctly with different models.
- You can debug tokenization issues.

**Next Step:** Use the Hugging Face ecosystem to work with pre-trained models.

---

### Unit 08.8 — Hugging Face Ecosystem

**What is it?**  
The Hugging Face ecosystem provides libraries (Transformers, Datasets, Tokenizers, Accelerate) and a model hub for working with pre-trained models, datasets, and tokenizers.

**Why does it matter?**  
Hugging Face is the standard platform for NLP and transformer work. It provides pre-trained models, easy-to-use APIs, and a community hub. Most ML practitioners use it daily.

**Why learn it here?**  
After understanding transformers, tokenizers, and model variants, you need a practical toolkit to use pre-trained models efficiently. Hugging Face is that toolkit.

**Prerequisites:** Python, PyTorch basics, understanding of transformers (08.1–08.7).

**Mental Model:**

Hugging Face is like a library system. The model hub is the catalog (thousands of pre-trained models). The `transformers` library is the checkout system (load any model with one line). The `datasets` library is the reference section (standard datasets ready to use). The `tokenizers` library is the translation desk (convert text to tokens).

**Core Concepts:**

- `transformers` library: `pipeline`, `AutoModel`, `AutoTokenizer`
- `datasets` library: loading, processing, streaming datasets
- Model Hub: browsing, downloading, uploading models
- Pipelines: high-level API for common tasks
- `Trainer` API: standard training loop
- `Accelerate`: multi-GPU and mixed-precision training
- Model cards and documentation

**How It Works:**

1. Install: `pip install transformers datasets accelerate`
2. Use `pipeline()` for quick inference on standard tasks
3. Use `AutoModel` and `AutoTokenizer` for more control
4. Use `Trainer` for fine-tuning with minimal boilerplate
5. Use `datasets` for efficient data loading and processing
6. Browse and share on the Hub at huggingface.co

**Syntax & Implementation:**

```python
# Quick inference with pipeline
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
result = classifier("I love this product!")
print(result)  # [{'label': 'POSITIVE', 'score': 0.9998}]

# More control with AutoModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english"
)

inputs = tokenizer("This is great!", return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    print(probs)  # [[0.001, 0.999]]
```

**Simple Example:**

```python
# Load and use a dataset
from datasets import load_dataset

dataset = load_dataset("imdb", split="train[:1000]")
print(dataset[0])  # {'text': '...', 'label': 1}
print(dataset.features)  # {'text': Value(dtype='string'), 'label': ClassLabel}

# Tokenize the dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True, max_length=512)

tokenized = dataset.map(tokenize, batched=True)
print(tokenized[0].keys())  # text, label, input_ids, token_type_ids, attention_mask
```

**Real-World Example:**

A data scientist at a company uses `pipeline("text-classification")` to prototype a sentiment model in minutes. They then use `Trainer` to fine-tune on company data. They push the model to the Hub for the team to use. The deployment team loads it with `pipeline(model="my-org/sentiment-v1")`.

**Common Mistakes:**

- Not matching tokenizer to model
- Forgetting to set `padding=True` and `truncation=True` during batch processing
- Not using `map()` for efficient dataset processing (loop is slower)
- Ignoring model cards (may have unexpected behavior or limitations)
- Not checking GPU memory before loading large models

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| `tokenizer` and `model` mismatch | Different model names | Check `from_pretrained` arguments | Use same model name for both |
| OOM on large model | Model too big for GPU | Check model parameters | Use `device_map="auto"` or smaller model |
| Slow dataset processing | Not using `batched=True` | Check processing time | Set `batched=True` in `map()` |
| Pipeline returns wrong labels | Wrong task specified | Check pipeline task | Use correct task name |

**Alternatives:**

| Tool | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Hugging Face | NLP, transformers, model sharing | Simple classical ML | Standard ecosystem but dependency-heavy |
| OpenAI API | Quick prototyping, no GPU | Need local control, fine-tuning | Easy but vendor lock-in |
| spaCy | Production NLP, smaller models | Need transformer-level accuracy | Faster but less flexible |
| LangChain | LLM application pipelines | Need direct model control | Abstraction overhead |

**Best Practices:**

- Always use `AutoModel` and `AutoTokenizer` for portability
- Use `datasets` `map()` for efficient preprocessing
- Check model cards before using a model from the Hub
- Use `device_map="auto"` for easy GPU distribution
- Set random seeds for reproducibility in `Trainer`

**Hands-On Practice:**

1. **Basic:** Use `pipeline()` for sentiment analysis, NER, and summarization.
2. **Guided:** Load a dataset, tokenize it, and prepare for training.
3. **Independent:** Fine-tune a model using `Trainer` on a custom dataset.
4. **Realistic:** Push a fine-tuned model to the Hub and load it from another script.
5. **Challenge:** Use `Accelerate` for multi-GPU training.

**Knowledge Check:**

- What does `AutoModel` do differently from loading a specific model class?
- How does `Trainer` simplify the training loop?
- When should you use `datasets` instead of loading data manually?
- What is the benefit of model cards on the Hub?

**Exit Criteria:**

- You can use Hugging Face pipelines for common tasks.
- You can load and process datasets with the `datasets` library.
- You can fine-tune models using `Trainer`.
- You can navigate the Model Hub.

**Next Step:** Apply these skills to fine-tune transformers for specific tasks.

---

### Unit 08.9 — Fine-tuning Transformers

**What is it?**  
Fine-tuning is the process of taking a pre-trained transformer and continuing training on task-specific data to adapt it to your use case.

**Why does it matter?**  
Pre-trained models have general language understanding. Fine-tuning specializes that understanding for your specific task — classification, generation, extraction, etc. — with far less data than training from scratch.

**Why learn it here?**  
After understanding the architecture, tokenizers, and Hugging Face ecosystem, you can now combine everything to adapt pre-trained models to real tasks.

**Prerequisites:** Hugging Face ecosystem (08.8), understanding of BERT/GPT (08.5, 08.6), training basics (Phase 06).

**Mental Model:**

Fine-tuning is like hiring a well-educated generalist and training them for a specific job. The pre-trained model already knows language. Fine-tuning teaches it the task's specific patterns with task-specific data.

**Core Concepts:**

- When to fine-tune vs use as-is
- Learning rate scheduling for fine-tuning
- Freezing layers and gradual unfreezing
- LoRA and parameter-efficient fine-tuning (PEFT)
- Evaluation metrics for NLP tasks
- Early stopping and checkpointing
- Handling class imbalance

**How It Works:**

1. Load a pre-trained model and tokenizer.
2. Prepare your dataset (tokenize, split, format).
3. Add a task-specific head (classification, generation, etc.).
4. Train with a small learning rate (2e-5 to 5e-5) to preserve pre-trained knowledge.
5. Evaluate on a validation set.
6. Save the best checkpoint.

**Syntax & Implementation:**

```python
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
import numpy as np

# Load data
dataset = load_dataset("imdb", split={"train": "train[:2000]", "test": "test[:500]"})

# Load model and tokenizer
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def preprocess(examples):
    return tokenizer(examples["text"], truncation=True, padding=True, max_length=256)

tokenized = dataset.map(preprocess, batched=True)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}

# Training
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    compute_metrics=compute_metrics,
)

trainer.train()
print(trainer.evaluate())
```

**Simple Example:**

```python
# LoRA fine-tuning (parameter-efficient)
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,                  # rank
    lora_alpha=32,
    target_modules=["q_lin", "v_lin"],  # attention layers
    lora_dropout=0.05,
    task_type="SEQ_CLS",
)

peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# trainable params: 600,000 || all params: 66,000,000 || 0.9%
# Only 0.9% of parameters are trained!
```

**Real-World Example:**

A healthcare company fine-tunes ClinicalBERT on medical records for diagnosis coding. They use LoRA to reduce training costs and only fine-tune 1% of parameters. A customer support team fine-tunes DistilBERT on support tickets for intent classification. Both achieve 90%+ accuracy with thousands of labeled examples.

**Common Mistakes:**

- Using too high a learning rate (destroys pre-trained knowledge)
- Not evaluating during training (overfitting without knowing)
- Not saving checkpoints (losing best model)
- Ignoring class imbalance in classification
- Fine-tuning all layers when data is small (overfitting)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Overfits quickly | Learning rate too high, data too small | Train/val loss divergence | Lower LR, freeze layers, augment data |
| Underfits | Learning rate too low, model too small | Both train and val loss high | Increase LR, unfreeze layers |
| Class imbalance in predictions | Imbalanced training data | Check label distribution | Use weighted loss, oversampling |
| Fine-tuning degrades performance | Catastrophic forgetting | Compare pre vs post fine-tune | Use smaller LR, freeze more layers |
| OOM during fine-tuning | Batch size too large | Check GPU memory | Reduce batch size, use gradient accumulation |

**Best Practices:**

- Start with 2e-5 to 5e-5 learning rate for BERT-like models
- Use early stopping based on validation loss
- For small datasets (<10K), freeze lower layers
- Use LoRA for large models to reduce compute costs
- Always evaluate with task-appropriate metrics (not just accuracy)

**Hands-On Practice:**

1. **Basic:** Fine-tune DistilBERT for sentiment classification using `Trainer`.
2. **Guided:** Experiment with learning rates and batch sizes.
3. **Independent:** Fine-tune for a different task (NER or QA).
4. **Realistic:** Compare full fine-tuning vs LoRA on accuracy and speed.
5. **Challenge:** Build a complete fine-tuning pipeline with evaluation, error analysis, and model card.

**Knowledge Check:**

- Why should you use a smaller learning rate for fine-tuning?
- When should you use LoRA instead of full fine-tuning?
- How do you handle class imbalance during fine-tuning?
- What metrics should you use beyond accuracy?

**Exit Criteria:**

- You can fine-tune transformers for classification and other tasks.
- You understand learning rate, freezing, and LoRA trade-offs.
- You can evaluate fine-tuned models properly.

**Next Step:** Synthesize everything into a mini project.

---

### Unit 08.10 — Transformers Synthesis & Review

**What is it?**  
A cumulative integration unit combining transformer architecture, attention, positional encoding, tokenizers, Hugging Face, and fine-tuning into a complete project.

**Why does it matter?**  
Knowing individual components is not enough. The learner must build an end-to-end transformer application independently, making design decisions and debugging real issues.

**Prerequisites:** All previous units in Phase 08.

**Mini Project: Sentiment Analysis with Fine-tuned Transformer**

**Objective:** Build a complete sentiment analysis system — from raw text to fine-tuned model to evaluation — using transformers.

**Problem Statement:** Given movie reviews, classify each as positive or negative. Build a system that can be retrained on new data and evaluated rigorously.

**Requirements:**

- Load and explore a sentiment dataset
- Tokenize with the correct tokenizer
- Fine-tune a pre-trained transformer (BERT, DistilBERT, or RoBERTa)
- Implement proper train/validation/test splitting
- Evaluate with accuracy, precision, recall, F1, and confusion matrix
- Perform error analysis on misclassified examples
- Save the model and tokenizer for later use
- Write a README explaining your choices

**Suggested Architecture:**

```text
Raw text → Exploratory analysis
    → Tokenization (match tokenizer to model)
    → Train/val/test split
    → Fine-tuning with Trainer API
    → Evaluation (metrics + confusion matrix)
    → Error analysis (examine misclassified examples)
    → Save model + tokenizer
    → README with decisions and limitations
```

**Expected Output:**

- Fine-tuned model saved locally
- Evaluation metrics (accuracy, F1, precision, recall)
- Confusion matrix visualization
- Error analysis report (10+ misclassified examples with analysis)
- README with model choice, hyperparameters, and limitations
- Script or notebook that runs end-to-end

**Evaluation Criteria:**

- Code runs from a clean environment
- Correct tokenizer-model pairing
- Proper evaluation with multiple metrics
- Error analysis identifies patterns in failures
- Model choice is justified
- README explains all decisions
- Results are reproducible (random seeds set)

**Advanced Extensions:**

- Compare 2–3 different models (BERT vs DistilBERT vs RoBERTa)
- Implement LoRA fine-tuning and compare with full fine-tuning
- Add data augmentation and measure its effect
- Build a simple inference API with FastAPI
- Add monitoring for prediction drift

**Knowledge Check:**

- Why did you choose this particular model?
- What would happen if you used the wrong tokenizer?
- How would you handle a dataset with 100K reviews vs 1K?
- What patterns do you see in the misclassified examples?
- How would you improve the system with more time?

**Hands-On Practice:**

1. **Basic:** Load a dataset, tokenize it, and verify shapes.
2. **Guided:** Fine-tune a model and evaluate with basic metrics.
3. **Independent:** Complete the full project with error analysis.
4. **Realistic:** Compare multiple models and justify the best choice.
5. **Challenge:** Add LoRA, data augmentation, and a simple API.

**Exit Criteria:**

- You can build a complete fine-tuning pipeline from scratch.
- You can evaluate models with appropriate metrics.
- You can perform error analysis and identify improvement directions.
- You can make and justify model/hyperparameter choices.

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Encoder-only (BERT) vs Decoder-only (GPT) | Understanding/classification tasks | Generation/creation tasks | Bidirectional context vs autoregressive generation |
| Encoder-decoder (T5) vs Decoder-only (GPT) | Different input/output structures | Same-direction tasks | Flexibility vs simplicity |
| BERT vs RoBERTa | Baseline understanding | Better performance needed | Simpler vs more optimized pre-training |
| Full fine-tuning vs LoRA | Large dataset, unlimited compute | Small dataset, limited compute | Full control vs efficiency |
| BPE vs WordPiece | Most modern models | BERT-family models | Slightly different merge criteria |
| Greedy vs beam search | Simple generation | Translation/summarization | Speed vs quality |
| Top-k vs top-p sampling | Fixed vocabulary size | Adaptive vocabulary size | Simpler vs more flexible |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model outputs garbage text | Wrong tokenizer for model | Check tokenizer-model pairing | Use matching tokenizer |
| Fine-tuning loss spikes | Learning rate too high | Monitor loss per step | Reduce LR, add warmup |
| Model memorizes training data | Overfitting, small dataset | Compare train vs val metrics | Add regularization, freeze layers, augment data |
| Attention weights are all uniform | Model not learning | Check training loss | Reduce dropout, increase model size |
| Generation is repetitive | Temperature too low | Check sampling params | Increase temperature, add repetition penalty |
| CUDA OOM during fine-tuning | Batch too large | Check GPU memory | Reduce batch size, use gradient accumulation |
| Model ignores certain features | Tokenization drops information | Check tokenized output | Adjust preprocessing, use different tokenizer |

---

## Phase Review Checklist

- [ ] Transformer block implemented from scratch
- [ ] Self-attention with Q, K, V, scaling, and masking understood
- [ ] Positional encoding implemented and tested
- [ ] Encoder-decoder architecture with cross-attention understood
- [ ] BERT used for classification, NER, or QA
- [ ] GPT used for text generation with sampling strategies
- [ ] Tokenizers (BPE, WordPiece) explained and compared
- [ ] Hugging Face ecosystem used (pipeline, datasets, Trainer)
- [ ] Fine-tuning completed with proper evaluation
- [ ] Mini project completed with error analysis
- [ ] All units passed knowledge checks

## Mastery Check

Without following a tutorial, you should be able to:

1. Implement a transformer block from scratch.
2. Explain self-attention mechanics including Q, K, V, scaling, and masking.
3. Choose between encoder-only, decoder-only, and encoder-decoder for a given task.
4. Use BERT for understanding tasks and GPT for generation tasks.
5. Explain BPE and WordPiece tokenization algorithms.
6. Navigate the Hugging Face ecosystem for model loading, training, and deployment.
7. Fine-tune a transformer with proper hyperparameters and evaluation.
8. Debug common transformer training and inference issues.
9. Make informed architectural decisions for NLP tasks.

## Interview / Explain-Back Questions

- What is self-attention and why is it the core of transformers?
- Why do transformers need positional encoding?
- How does causal masking differ from bidirectional attention?
- What is the difference between BERT and GPT at the architectural level?
- Why is fine-tuning more efficient than training from scratch?
- What is LoRA and when should you use it?
- How does BPE tokenization work?
- Why do transformers use residual connections and layer normalization?
- What are the trade-offs between encoder-only, decoder-only, and encoder-decoder?
- How would you debug a model that overfits during fine-tuning?

## Exit Criteria

Move to Phase 09 only when you can independently build, fine-tune, and evaluate a transformer model for a specific task, explain your architectural and hyperparameter choices, debug training issues, and articulate the differences between major transformer variants.
