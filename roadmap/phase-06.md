# Phase 06 — Deep Learning

> **Goal:** Master deep learning — from-scratch backpropagation through PyTorch and modern architectures.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced  
**Priority:** Essential  
**Prerequisites:** Phase 02 (Math), Phase 05 (Machine Learning)  
**Mastery target:** Level 5 — independent deep learning model building, debugging, and architectural decisions

---

## Why This Phase Exists

Deep learning is the engine behind modern computer vision, NLP, speech, and generative AI. This phase takes you from understanding *what* a neural network is to *building and debugging* them in PyTorch. You will implement backpropagation from scratch before using frameworks, so you understand what the tools hide. You will also learn training discipline — regularization, validation, checkpointing, transfer learning — that separates working demos from reliable systems.

### Phase Mental Model

A neural network is a sequence of linear transformations followed by nonlinear activations. Training is just gradient-based optimization: forward pass → compute loss → backward pass → update parameters. Every advanced architecture (CNN, RNN, Transformer) is a structured arrangement of these same building blocks.

```text
Perceptron (single neuron)
    ↓
Activation functions → introduce nonlinearity
    ↓
Loss functions → quantify error
    ↓
Gradient descent + backprop → learn from error
    ↓
MLP (stacked layers)
    ↓
Regularization + optimizers + data loading → train well
    ↓
CNNs → spatial data | RNNs → sequential data | Attention → all-pairs relationships
```

### What This Phase Prepares For

- Transformer architectures in Phase 07
- NLP and tokenization in Phase 08
- Generative models (diffusion, GANs) in Phase 09
- Fine-tuning and transfer learning in Phase 10
- Production deployment of trained models in Phase 16

---

## Units

### Unit 06.1 — Perceptron & Activation Functions

**What is it?**  
The perceptron is the smallest unit of a neural network — a single neuron that computes a weighted sum of inputs, adds a bias, and passes the result through an activation function. Activation functions introduce nonlinearity so networks can learn complex mappings.

**Why does it matter?**  
Every deep network is built from these atomic units. Without understanding the perceptron, layers and architectures are black boxes. Without understanding activation functions, you cannot diagnose training failures like dead neurons or vanishing gradients.

**Why learn it here?**  
You have the linear algebra and ML foundations from Phase 02 and 05. This is where those math concepts become executable neural network code.

**Prerequisites:** Phase 02 (linear algebra, calculus basics), Phase 05 (gradient-based optimization concepts).

**Mental Model:**  
A perceptron is a decision machine: it takes in signals, weighs them by importance, adds a threshold, and decides how strongly to fire. The activation function controls the shape of that firing behavior.

**Core Concepts:**

- weighted sum: `z = w₁x₁ + w₂x₂ + ... + b`
- activation functions: sigmoid, tanh, ReLU, Leaky ReLU, softmax
- binary classification with a single neuron
- decision boundaries
- vectorized computation across multiple inputs

**How It Works:**

1. Compute the dot product of inputs and weights: `z = X @ w + b`
2. Apply an activation function: `a = σ(z)`
3. For classification, threshold the output or use it as a probability.

**Syntax & Implementation:**

```python
import numpy as np

# Perceptron forward pass
def perceptron(X, w, b):
    z = X @ w + b
    return sigmoid(z)

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# Training loop
def train_perceptron(X, y, lr=0.1, epochs=100):
    w = np.zeros(X.shape[1])
    b = 0.0
    for epoch in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        w -= lr * (X.T @ error) / len(y)
        b -= lr * error.mean()
        if epoch % 20 == 0:
            loss = -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
            print(f"Epoch {epoch}: loss={loss:.4f}")
    return w, b

# Example: AND gate
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])
w, b = train_perceptron(X, y, lr=0.5, epochs=200)
print("Predictions:", sigmoid(X @ w + b).round())
```

**Real-World Example:**  
A spam filter perceptron takes word-frequency features and outputs a probability of spam. Each word weight reflects how much that word signals spam (e.g., "free" has a high positive weight).

**Activation Function Comparison:**

| Function | Formula | Range | Use When | Avoid When |
|---|---|---|---|---|
| Sigmoid | `1 / (1 + e⁻ᶻ)` | (0, 1) | Binary output / probability | Hidden layers (vanishing gradients) |
| Tanh | `(eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)` | (-1, 1) | Hidden layers, zero-centered preferred | Very deep networks (still saturates) |
| ReLU | `max(0, z)` | [0, ∞) | Default hidden layer choice | Dead neurons (negative inputs) |
| Leaky ReLU | `max(0.01z, z)` | (-∞, ∞) | When ReLU dies frequently | Rarely worse than ReLU |
| Softmax | `eᶻⁱ / Σeᶻʲ` | (0, 1), sums to 1 | Multi-class output | Regression / binary output |

**Common Mistakes:**

- Using sigmoid in hidden layers (causes vanishing gradients)
- Forgetting to clip sigmoid input to avoid overflow
- Using softmax for binary classification (use single sigmoid instead)
- Not initializing weights properly (all zeros = no learning)

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Output stuck at 0.5 | Sigmoid saturating | Print pre-activation values | Use ReLU in hidden layers, check weight scale |
| Loss is NaN | Sigmoid overflow | Check `z` values before activation | Clip inputs or use stable log-sum-exp |
| No learning at all | All weights zero | Print weights after training | Use random initialization |
| Dead neurons | ReLU outputs always 0 | Count neurons with zero activations | Use Leaky ReLU or lower learning rate |

**Hands-On Practice:**

1. Basic: implement sigmoid and ReLU from scratch. Plot them.
2. Guided: build a perceptron that learns AND and OR gates.
3. Independent: extend to XOR (hint: you need two layers).
4. Realistic: compare sigmoid vs ReLU on a toy classification dataset.
5. Challenge: implement softmax and verify outputs sum to 1 across classes.

**Knowledge Check:**

- Why can a single perceptron not solve XOR?
- What happens to gradients when sigmoid input is very large or very small?
- Why is ReLU preferred over sigmoid in hidden layers?
- What does the bias term allow a neuron to do that weights alone cannot?

**Exit Criteria:**

- You can implement a single-neuron classifier from scratch.
- You can explain the effect of each activation function on gradients and outputs.
- You can diagnose dead neurons and vanishing activations.

**Next Step:** Learn how to quantify prediction error with loss functions.

---

### Unit 06.2 — Loss Functions

**What is it?**  
A loss function measures the gap between a model's prediction and the true target. It converts "how wrong am I?" into a single scalar that gradient descent can minimize.

**Why does it matter?**  
Choosing the wrong loss function means optimizing for the wrong thing. MSE on classification or cross-entropy on regression will produce nonsense. Loss functions are the contract between your model and your objective.

**Why learn it here?**  
You need loss functions before you can implement backpropagation (Unit 06.3). The choice of loss determines the gradient landscape your optimizer navigates.

**Prerequisites:** Unit 06.1 (perceptron, activations), basic probability (softmax outputs probabilities).

**Mental Model:**  
A loss function is a scoring rubric. The lower the score, the better the model performs. Different tasks need different rubrics: "how far off?" (regression) vs "how confident in the wrong answer?" (classification).

**Core Concepts:**

- regression losses: MSE, MAE, Huber loss
- classification losses: binary cross-entropy, categorical cross-entropy
- maximum likelihood interpretation of loss functions
- log-sum-exp trick for numerical stability
- loss weighting for imbalanced classes

**How It Works:**

- MSE: `L = (1/n) Σ(ŷ - y)²` — penalizes large errors quadratically
- BCE: `L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]` — rewards confident correct predictions
- Cross-entropy: `L = -Σ yᵢ·log(ŷᵢ)` — generalizes BCE to multiple classes

**Syntax & Implementation:**

```python
import numpy as np

# MSE Loss (regression)
def mse_loss(y_pred, y_true):
    return np.mean((y_pred - y_true) ** 2)

# Binary Cross-Entropy (binary classification)
def bce_loss(y_pred, y_true):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred + eps) + (1 - y_true) * np.log(1 - y_pred + eps))

# Categorical Cross-Entropy (multi-class)
def cross_entropy_loss(y_pred, y_true):
    """
    y_pred: (n_samples, n_classes) — probabilities from softmax
    y_true: (n_samples, n_classes) — one-hot encoded
    """
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))

# Example usage
y_true_binary = np.array([0, 1, 1, 0])
y_pred_binary = np.array([0.1, 0.9, 0.8, 0.2])
print(f"BCE: {bce_loss(y_pred_binary, y_true_binary):.4f}")

y_true_multi = np.array([[1,0,0], [0,1,0], [0,0,1]])
y_pred_multi = np.array([[0.7,0.2,0.1], [0.1,0.8,0.1], [0.2,0.1,0.7]])
print(f"CE:  {cross_entropy_loss(y_pred_multi, y_true_multi):.4f}")
```

**Real-World Example:**  
An image classifier uses cross-entropy because outputs are class probabilities. A house price predictor uses MSE because errors should be penalized quadratically (a $50K error is more than twice as bad as a $25K error). A robust regression task with outliers might use Huber loss to avoid extreme gradients.

**Decision Guidance:**

| Task | Recommended Loss | Why | Avoid |
|---|---|---|---|
| Binary classification | Binary cross-entropy | Probabilistic, differentiable | MSE (gradients vanish for confident wrong predictions) |
| Multi-class classification | Cross-entropy + softmax | Proper scoring rule | MSE on one-hot outputs |
| Regression | MSE | Smooth gradients, penalizes large errors | MAE when outliers dominate |
| Regression with outliers | Huber loss | Combines MSE and MAE benefits | MSE when extreme outliers exist |
| Imbalanced classification | Weighted cross-entropy | Downweights majority class | Unweighted cross-entropy (model ignores minority) |

**Common Mistakes:**

- Using MSE for classification (gradient issues near confident predictions)
- Forgetting numerical stability (log of zero = NaN)
- Using softmax + cross-entropy without understanding they are mathematically linked
- Not normalizing loss by batch size (makes learning rate sensitive to batch size)

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Loss is NaN | Log of zero or very small prediction | Print predictions before loss | Add epsilon, clip predictions |
| Loss stops decreasing | Saturated softmax or poor loss choice | Inspect prediction distribution | Check loss function, use label smoothing |
| Loss decreases but accuracy doesn't | Mismatched loss and metric | Plot loss and accuracy together | Verify loss matches the actual task |
| Loss varies wildly between batches | Batch size too small or high learning rate | Monitor per-batch loss variance | Increase batch size or reduce LR |

**Hands-On Practice:**

1. Basic: implement MSE, BCE, and cross-entropy from scratch.
2. Guided: compute gradients of MSE and BCE analytically and verify numerically.
3. Independent: compare MSE vs BCE on a binary classification task — which converges faster?
4. Realistic: implement class-weighted cross-entropy for an imbalanced dataset.
5. Challenge: implement Huber loss and compare it to MSE on data with outliers.

**Knowledge Check:**

- Why does cross-entropy pair naturally with softmax?
- What problem does MSE have for binary classification?
- How does adding epsilon to log prevent NaN?
- Why does weighted cross-entropy help with imbalanced classes?

**Exit Criteria:**

- You can implement the three most common loss functions from scratch.
- You can choose the right loss for a given task and explain why.
- You can debug NaN losses and mismatched loss/metric behavior.

**Next Step:** Learn how gradient descent and backpropagation use loss to update weights.

---

### Unit 06.3 — Gradient Descent & Backpropagation (from scratch)

**What is it?**  
Gradient descent is the optimization algorithm that minimizes loss by iteratively adjusting parameters in the direction that reduces error. Backpropagation is the efficient algorithm that computes gradients through the chain rule, allowing credit assignment for each parameter's contribution to the error.

**Why does it matter?**  
This is *the* mechanism by which neural networks learn. Without understanding backpropagation, you cannot debug training failures, understand why architectures fail, or reason about gradient flow in deep networks.

**Why learn it here?**  
You have the perceptron, activations, and loss functions. Now you need to connect them into a learning system. Building from scratch cements understanding before PyTorch hides the details.

**Prerequisites:** Unit 06.1 (perceptron, activations), Unit 06.2 (loss functions), calculus (chain rule).

**Mental Model:**  
Backpropagation is a blame game. After the forward pass produces a loss, backward pass asks: "which parameter caused how much of this error?" The chain rule traces the error backward layer by layer, assigning each parameter a gradient proportional to its responsibility.

```text
Forward:  input → [layer1] → [layer2] → output → loss
Backward: loss → [layer2 grad] → [layer1 grad] → weight updates
```

**Core Concepts:**

- forward pass: compute outputs and loss
- backward pass: compute gradients via chain rule
- chain rule: `dL/dw = dL/da · da/dz · dz/dw`
- computational graph
- gradient accumulation
- numerical vs analytical gradients

**How It Works:**

1. Forward pass: store intermediate values (pre-activations, activations)
2. Compute loss from output and target
3. Backward pass: starting from `dL/dL = 1`, propagate gradients backward
4. Update: `w = w - lr * dL/dw`

**Syntax & Implementation:**

```python
import numpy as np

np.random.seed(42)

# A 2-layer MLP from scratch
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_grad(a):
    return a * (1 - a)

# Initialize
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)  # XOR

W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

lr = 1.0
epochs = 5000

for epoch in range(epochs):
    # Forward pass
    z1 = X @ W1 + b1          # (4, 4)
    a1 = sigmoid(z1)           # (4, 4)
    z2 = a1 @ W2 + b2         # (4, 1)
    a2 = sigmoid(z2)           # (4, 1) — output

    # Loss (BCE)
    loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

    # Backward pass
    dz2 = a2 - y                          # (4, 1)
    dW2 = a1.T @ dz2 / len(y)             # (4, 1)
    db2 = dz2.mean(axis=0, keepdims=True) # (1, 1)

    da1 = dz2 @ W2.T                      # (4, 4)
    dz1 = da1 * sigmoid_grad(a1)          # (4, 4)
    dW1 = X.T @ dz1 / len(y)             # (2, 4)
    db1 = dz1.mean(axis=0, keepdims=True) # (1, 4)

    # Update
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: loss={loss:.4f}")

# Test
pred = sigmoid(sigmoid(X @ W1 + b1) @ W2 + b2)
print("Predictions:\n", pred.round(2))
```

**Real-World Example:**  
When training a network to recognize handwritten digits, the forward pass runs an image through layers to get class probabilities. Backpropagation traces back from the cross-entropy loss to find which weights in which layers contributed to misclassifying a "3" as an "8", then adjusts those weights slightly.

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Gradients are all zero | Dead neurons or saturated activation | Print gradients at each layer | Use ReLU, check initialization |
| Gradients explode (very large values) | Deep network or high LR | Monitor gradient norms | Gradient clipping, lower LR |
| Network doesn't learn | Weights not updating | Print weights before/after update | Check gradient flow, increase LR |
| Loss oscillates | LR too high | Plot loss curve | Reduce learning rate |
| Numerical vs analytical gradient mismatch | Implementation bug | Compare finite-difference gradient | Debug backward pass line by line |

**Hands-On Practice:**

1. Basic: implement forward pass for a 2-layer network. Verify output shapes.
2. Guided: implement the full backward pass. Verify gradients with numerical differentiation.
3. Independent: extend to a 3-layer network. What changes in the backward pass?
4. Realistic: train on XOR and a spiral dataset. Plot decision boundaries.
5. Challenge: implement gradient clipping and compare training stability.

**Knowledge Check:**

- Why does backpropagation use the chain rule?
- What is the difference between a forward pass and a backward pass?
- How do you verify your backward pass is correct?
- Why can gradients vanish in deep networks with sigmoid activations?

**Exit Criteria:**

- You can implement a multi-layer network with backpropagation from scratch in NumPy.
- You can verify gradients using numerical differentiation.
- You can diagnose vanishing/exploding gradients and dead neurons.

**Next Step:** Use PyTorch's autograd to build the same networks with less boilerplate.

---

### Unit 06.4 — MLP with PyTorch

**What is it?**  
PyTorch is a deep learning framework that provides automatic differentiation (autograd), GPU acceleration, and high-level building blocks for networks, datasets, and training loops. An MLP (multi-layer perceptron) is a stack of fully connected layers with nonlinear activations.

**Why does it matter?**  
Building from scratch is educational but impractical. PyTorch lets you build, train, and debug real models efficiently. Nearly all modern deep learning work uses a framework.

**Why learn it here?**  
After implementing backpropagation manually, you understand exactly what PyTorch automates. This prevents the "framework dependency" trap — you know what `loss.backward()` does because you built it yourself.

**Prerequisites:** Unit 06.3 (backpropagation from scratch), basic Python.

**Mental Model:**  
PyTorch is a computational graph that remembers how to compute gradients. You define the forward pass, and PyTorch builds the backward pass automatically. `model(x)` computes the output; `loss.backward()` computes all gradients; `optimizer.step()` applies the updates.

**Core Concepts:**

- `torch.Tensor` and `torch.nn.Module`
- `nn.Linear`, `nn.Sigmoid`, `nn.ReLU`
- `nn.Sequential` for stacking layers
- `torch.optim.SGD` and `torch.optim.Adam`
- `loss.backward()` and autograd
- `model.parameters()` and `model.train()`

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# Data
X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

# Training setup
model = MLP()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Test
with torch.no_grad():
    print("Predictions:", model(X).round())
```

**Real-World Example:**  
A credit scoring model uses an MLP with 3 hidden layers (64, 32, 16 neurons) to predict default probability from 20 financial features. PyTorch handles the forward pass, loss computation, and gradient updates while the team focuses on feature engineering and evaluation.

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Loss doesn't decrease | Forgot `optimizer.zero_grad()` | Check gradient accumulation | Add `optimizer.zero_grad()` before each backward pass |
| CUDA out of memory | Batch too large or model too big | Check `torch.cuda.memory_summary()` | Reduce batch size, use gradient accumulation |
| Model trains on CPU unexpectedly | No `.to(device)` call | Check tensor device | Use `model.to(device)` and `x.to(device)` |
| Results not reproducible | Random seeds not set | Set seeds at start | `torch.manual_seed(42)` |
| NaN loss | Learning rate too high or bad data | Print pre-activation values | Lower LR, check data for NaN/inf |

**Alternatives:**

| Framework | Use When | Avoid When |
|---|---|---|
| PyTorch | Research, flexibility, dynamic graphs | Production simplicity without customization |
| TensorFlow/Keras | Production pipelines, TFLite deployment | Research requiring dynamic architecture changes |
| JAX | Functional optimization, custom research | Standard production workflows |

**Hands-On Practice:**

1. Basic: build a 2-layer MLP and train on the AND gate.
2. Guided: experiment with hidden layer size (4, 16, 64). How does it affect convergence?
3. Independent: train on a spiral dataset with 3 classes (use softmax + cross-entropy).
4. Realistic: build an MLP for MNIST digit classification. Achieve 95%+ accuracy.
5. Challenge: compare Adam vs SGD with momentum. Plot learning curves for both.

**Knowledge Check:**

- What does `optimizer.zero_grad()` do and why is it necessary?
- How does PyTorch's autograd differ from your from-scratch implementation?
- When would you use `nn.Sequential` vs a custom `forward()` method?
- What is the difference between `model.train()` and `model.eval()`?

**Exit Criteria:**

- You can build and train an MLP in PyTorch for classification and regression.
- You can explain every line of a PyTorch training loop.
- You can debug common training issues (no learning, NaN loss, memory errors).

**Next Step:** Learn regularization techniques to prevent overfitting.

---

### Unit 06.5 — Regularization

**What is it?**  
Regularization techniques prevent a model from memorizing training data by adding constraints or noise during training. The goal is to improve generalization — performance on unseen data.

**Why does it matter?**  
Deep networks have millions of parameters. Without regularization, they will perfectly memorize training data while failing on test data. Overfitting is the default failure mode of deep learning.

**Why learn it here?**  
You can now build and train networks. Before moving to complex architectures, you need the discipline to prevent overfitting — this is a skill that separates working models from reliable ones.

**Prerequisites:** Unit 06.4 (PyTorch MLP), basic probability.

**Mental Model:**  
Regularization is like a teacher who doesn't let students memorize the textbook. By adding noise, restricting capacity, or forcing early stopping, the student (model) must learn general principles instead of specific examples.

**Core Concepts:**

- L1 regularization (Lasso): adds `λΣ|w|` to loss, promotes sparsity
- L2 regularization (Ridge/Weight Decay): adds `λΣw²` to loss, prevents large weights
- Dropout: randomly zeros neurons during training
- Batch normalization: normalizes activations per batch
- Early stopping: halt training when validation loss stops improving
- Data augmentation: artificially increase training data diversity
- Label smoothing: soften hard labels (0/1 → 0.1/0.9)

**How It Works:**

- **L2 in PyTorch:** `weight_decay` parameter in optimizer adds `λ/2 * ||w||²` to loss
- **Dropout:** During training, each neuron has probability `p` of being zeroed. At inference, all neurons active (scaled by `1-p`).
- **Batch Norm:** Normalizes layer inputs to mean=0, std=1, then applies learnable scale/shift.

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class RegularizedMLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, output_dim=1, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# L2 regularization via weight_decay
model = RegularizedMLP(dropout_rate=0.3)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

# Early stopping setup
best_val_loss = float('inf')
patience = 10
patience_counter = 0

# Training with early stopping
for epoch in range(500):
    model.train()
    train_pred = model(X_train)
    train_loss = criterion(train_pred, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = criterion(val_pred, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_state = model.state_dict().copy()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            model.load_state_dict(best_state)
            break
```

**Real-World Example:**  
A medical image classifier uses dropout (0.5), batch normalization, and early stopping to prevent overfitting on a small dataset of 5,000 images. Without these techniques, the model achieves 99% training accuracy but 62% test accuracy. With regularization, it reaches 92% test accuracy.

**Decision Guidance:**

| Technique | Best For | Effort | Risk |
|---|---|---|---|
| L2 weight decay | General-purpose, always use | Low | None (just a hyperparameter) |
| Dropout | Large networks, high overfitting risk | Low | Can slow convergence |
| Batch Normalization | Deep networks, training stability | Medium | Adds complexity at inference |
| Early stopping | Universal, simple safety net | Low | May stop too early without monitoring |
| Data augmentation | Vision, NLP (when data is limited) | Medium | Domain-specific knowledge needed |
| Label smoothing | Classification with noisy labels | Low | May reduce calibration |

**Common Mistakes:**

- Using dropout during evaluation (always use `model.eval()`)
- Forgetting weight decay when switching optimizers
- Setting dropout too high (model can't learn) or too low (no effect)
- Not using early stopping because "I'll just train for more epochs"
- Applying batch norm to very small batch sizes (statistics become noisy)

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Training accuracy low with dropout | Dropout rate too high | Reduce p gradually | Use 0.1–0.3 for hidden layers |
| Validation loss unstable | Batch norm with small batch size | Check batch size | Increase batch size or use LayerNorm |
| Model performs same with/ without regularization | Underfitting (not enough capacity) | Check train vs val accuracy | Increase model size first |
| Early stopping triggers too early | Patience too low or no improvement signal | Plot val loss curve | Increase patience, check data |

**Hands-On Practice:**

1. Basic: add L2 weight decay to an MLP and observe the effect on weight magnitudes.
2. Guided: add dropout and batch norm. Compare convergence speed and final accuracy.
3. Independent: implement early stopping with model checkpointing.
4. Realistic: train a model with no regularization, then add each technique one by one. Record the effect.
5. Challenge: apply data augmentation (random flips/rotations) to MNIST and measure test accuracy improvement.

**Knowledge Check:**

- Why does L2 regularization promote small (not zero) weights while L1 promotes sparsity?
- Why must dropout be disabled during evaluation?
- How does batch normalization help both training speed and regularization?
- When is early stopping insufficient as a regularization strategy?

**Exit Criteria:**

- You can apply L1, L2, dropout, batch norm, and early stopping.
- You can diagnose overfitting and choose appropriate regularization.
- You understand the interaction between regularization and model capacity.

**Next Step:** Learn optimizers and learning rate schedules for efficient training.

---

### Unit 06.6 — Optimizers & Learning Rate Schedules

**What is it?**  
Optimizers determine how parameter updates are computed from gradients. Learning rate schedules adjust the learning rate during training to improve convergence speed and final performance.

**Why does it matter?**  
The optimizer and learning rate are the most impactful hyperparameters in deep learning. A poor optimizer or fixed learning rate causes slow convergence, oscillation, or divergence.

**Why learn it here?**  
You have built networks and applied regularization. Now you need to train them efficiently. This unit gives you the tools to make training fast and stable.

**Prerequisites:** Unit 06.3 (gradient descent), Unit 06.4 (PyTorch training loops).

**Mental Model:**  
SGD is a hiker walking downhill in fog — they only see the local slope. Momentum adds inertia so they don't get stuck in small valleys. Adam is like a hiker with a GPS that remembers both the terrain slope and how bumpy it has been, adjusting step size for each dimension independently.

**Core Concepts:**

- SGD with and without momentum
- Adaptive methods: Adam, AdamW, RMSProp
- Learning rate: the single most important hyperparameter
- Learning rate schedules: step decay, cosine annealing, warmup
- Warmup: start with small LR, increase to target

**How It Works:**

- **SGD:** `w = w - lr * grad`
- **SGD + Momentum:** `v = β·v + grad; w = w - lr·v`
- **Adam:** Combines momentum (first moment) with RMSProp (second moment), with bias correction
- **AdamW:** Adam with decoupled weight decay (more principled than L2)

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(784, 256), nn.ReLU(),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 10)
)

# Different optimizers
optimizers = {
    'sgd': optim.SGD(model.parameters(), lr=0.01),
    'sgd_momentum': optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
    'adam': optim.Adam(model.parameters(), lr=0.001),
    'adamw': optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01),
}

# Learning rate schedule example
optimizer = optim.AdamW(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# Training with scheduler
for epoch in range(50):
    # ... training code ...
    scheduler.step()
    print(f"Epoch {epoch}: LR = {scheduler.get_last_lr()[0]:.6f}")

# Warmup + cosine schedule
def warmup_cosine(epoch, warmup_epochs=5, total_epochs=50, min_lr=1e-6, max_lr=1e-3):
    if epoch < warmup_epochs:
        return max_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + 0.5 * (max_lr - min_lr) * (1 + __import__('math').cos(__import__('math').pi * progress))
```

**Optimizer Comparison:**

| Optimizer | Use When | Avoid When | Key Hyperparameters |
|---|---|---|---|
| SGD | Simple problems, want maximum control | Complex architectures, fast convergence needed | lr, momentum |
| SGD + Momentum | Default for vision models | Very sparse gradients | lr, momentum (0.9 typical) |
| Adam | General-purpose, fast convergence | When weight decay matters most | lr, betas, eps |
| AdamW | When L2 regularization is needed | Simple problems where SGD suffices | lr, weight_decay |

**Learning Rate Schedule Comparison:**

| Schedule | Behavior | Use When |
|---|---|---|
| Constant | Fixed LR | Baseline comparison only |
| Step decay | Reduce LR by factor every N epochs | When you know when to slow down |
| Cosine annealing | Smooth decrease following cosine curve | Default choice, works well generally |
| Warmup + cosine | Gradual increase then smooth decrease | Large batch training, transformers |

**Common Mistakes:**

- Using the same LR for all layers (different layers learn at different speeds)
- Not resetting optimizer state when changing LR mid-training
- Using Adam without weight decay and expecting good generalization
- Choosing LR based on final accuracy without checking convergence speed

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Loss oscillates wildly | LR too high | Check LR value vs typical ranges | Reduce LR by 10x |
| Loss plateaus immediately | LR too low | Plot loss curve | Increase LR by 10x |
| Training is slow but stable | LR too conservative | Compare to known baselines | Use warmup schedule |
| Adam diverges | LR too high or eps too small | Check gradient norms | Lower LR, increase eps |

**Hands-On Practice:**

1. Basic: train the same model with SGD, Adam, and AdamW. Compare convergence speed.
2. Guided: implement cosine annealing and plot the learning rate schedule.
3. Independent: train with constant LR vs cosine annealing. Which gives better final accuracy?
4. Realistic: implement warmup + cosine and compare to constant LR on a real dataset.
5. Challenge: find the optimal initial LR using a learning rate range test.

**Knowledge Check:**

- Why does Adam adapt the learning rate per parameter?
- What is the difference between weight decay in Adam and L2 regularization in the loss?
- When would SGD outperform Adam?
- Why is learning rate warmup important for large-batch training?

**Exit Criteria:**

- You can compare SGD, Adam, and AdamW and explain when to use each.
- You can implement learning rate schedules in PyTorch.
- You can diagnose LR-related training failures.

**Next Step:** Learn PyTorch data loading for real-world datasets.

---

### Unit 06.7 — PyTorch Datasets & DataLoaders

**What is it?**  
The `Dataset` and `DataLoader` classes in PyTorch handle data loading, batching, shuffling, and parallel preprocessing. They are the bridge between raw data and the training loop.

**Why does it matter?**  
Real datasets don't fit in memory or come as clean tensors. You need efficient data pipelines to feed GPUs with batches on demand. Poor data loading is a common bottleneck that is invisible from the model side.

**Why learn it here?**  
You have the model and optimizer. Now you need real data pipelines to train on anything beyond toy examples.

**Prerequisites:** Unit 06.4 (PyTorch basics), basic Python file I/O.

**Mental Model:**  
A `Dataset` is a library card catalog — it knows where each data point lives and how to fetch it. A `DataLoader` is the librarian — it grabs books in batches, shuffles the shelf, and hands them to you in parallel.

**Core Concepts:**

- `torch.utils.data.Dataset`: abstract class with `__len__` and `__getitem__`
- `torch.utils.data.DataLoader`: batching, shuffling, num_workers
- `torchvision.transforms` for image preprocessing
- Custom datasets vs `TensorDataset`
- Train/validation/test splits
- Data augmentation transforms

**Syntax & Implementation:**

```python
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Simple Dataset from tensors
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000, 1))
dataset = TensorDataset(X, y)

# Custom Dataset
class CSVDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,  # set >0 for parallel loading
    drop_last=True  # drop incomplete last batch
)

# Training with DataLoader
model = torch.nn.Linear(20, 2)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for batch_X, batch_y in dataloader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

**Real-World Example:**  
A medical imaging pipeline uses a custom `Dataset` that loads DICOM files, applies random rotation and flipping (augmentation), normalizes pixel values, and returns (image, label) pairs. The `DataLoader` batches 32 images with 4 parallel workers, keeping the GPU fed without bottlenecking.

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| DataLoader is slow | num_workers=0 (single process) | Check worker count | Set num_workers=2-4 |
| GPU utilization low | Data loading bottleneck | Monitor GPU usage vs CPU | Increase workers, use pin_memory |
| Memory error during loading | Loading all data into memory | Check Dataset.__getitem__ | Load lazily (one sample at a time) |
| Inconsistent batch sizes | drop_last=False with uneven data | Check last batch size | Use drop_last=True for training |

**Hands-On Practice:**

1. Basic: create a TensorDataset and iterate with a DataLoader.
2. Guided: build a custom Dataset that loads from a list of dictionaries.
3. Independent: implement train/val/test splits with different DataLoaders.
4. Realistic: add data augmentation transforms to an image Dataset.
5. Challenge: profile data loading time and optimize num_workers and pin_memory.

**Knowledge Check:**

- What does `shuffle=True` do and why is it important for training?
- Why use `pin_memory=True` when training on GPU?
- What is the difference between `Dataset` and `DataLoader`?
- When would you use `drop_last=True`?

**Exit Criteria:**

- You can build custom Datasets and efficient DataLoaders.
- You can implement train/val/test splits with proper batching.
- You can identify and fix data loading bottlenecks.

**Next Step:** Build complete training and validation loops.

---

### Unit 06.8 — Training Loops & Validation

**What is it?**  
A training loop orchestrates the forward pass, loss computation, backward pass, and parameter updates over multiple epochs. A validation loop measures performance on held-out data to detect overfitting.

**Why does it matter?**  
Many beginners only check training loss. Without validation monitoring, you cannot tell if the model is generalizing or memorizing. Proper training loops are the difference between a notebook experiment and a reliable model.

**Why learn it here?**  
You have all the pieces — model, loss, optimizer, data loading. Now you need the discipline to assemble them into a proper training pipeline with monitoring.

**Prerequisites:** Unit 06.4 (PyTorch), Unit 06.5 (regularization), Unit 06.7 (DataLoader).

**Mental Model:**  
Training is like a sports practice: you practice (train), then scrim (validate), then adjust strategy based on scrim results. Never judge a team solely by practice performance.

**Core Concepts:**

- train mode vs eval mode (`model.train()`, `model.eval()`)
- gradient computation only during training
- epoch-level vs batch-level metrics
- tracking train/val loss and metrics
- metric computation (accuracy, precision, recall, F1)
- logging and visualization

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def accuracy(pred, target):
    return (pred.argmax(dim=1) == target).float().mean().item()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_acc = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y_batch)
        total_acc += accuracy(pred, y_batch) * len(y_batch)

    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_acc = 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item() * len(y_batch)
        total_acc += accuracy(pred, y_batch) * len(y_batch)
    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)

# Full training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = nn.Sequential(nn.Linear(20, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

for epoch in range(20):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
          f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
```

**Real-World Example:**  
A fraud detection system trains for 50 epochs. Training accuracy reaches 99.8% but validation accuracy plateaus at 94.2% after epoch 15. The gap signals overfitting. The team applies early stopping at epoch 18 and achieves 94.0% test accuracy.

**Common Mistakes:**

- Not switching between train/eval mode (dropout and batch norm behave differently)
- Computing validation metrics inside `torch.no_grad()` (saves memory and computation)
- Not tracking validation loss (only accuracy is not enough)
- Logging only the last batch's loss instead of epoch average

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Val accuracy much lower than train | Overfitting or data leakage | Check train/val split | Add regularization, fix data leakage |
| Val loss increasing while train loss decreases | Overfitting | Plot both curves | Early stopping, add regularization |
| Both losses high | Underfitting | Check model capacity | Increase model size, reduce regularization |
| Training is very slow | No GPU or data bottleneck | Check device utilization | Move to GPU, optimize data loading |

**Hands-On Practice:**

1. Basic: implement train and validate functions.
2. Guided: add accuracy tracking and print epoch summaries.
3. Independent: implement early stopping based on validation loss.
4. Realistic: train on MNIST, plot train/val curves, identify overfitting point.
5. Challenge: add learning rate scheduling and compare to fixed LR.

**Knowledge Check:**

- Why must you call `model.eval()` before validation?
- What does `@torch.no_grad()` do and why use it during validation?
- How do you detect overfitting from train/val curves?
- What is the difference between epoch loss and batch loss?

**Exit Criteria:**

- You can implement complete training loops with validation monitoring.
- You can diagnose overfitting and underfitting from learning curves.
- You can track and log metrics properly.

**Next Step:** Learn checkpointing and transfer learning for practical model reuse.

---

### Unit 06.9 — Checkpointing & Transfer Learning

**What is it?**  
Checkpointing saves model parameters, optimizer state, and training progress so training can be resumed or the best model recovered. Transfer learning reuses a pretrained model on a new task, dramatically reducing data and training time requirements.

**Why does it matter?**  
Training large models takes hours to days. Without checkpointing, a crash means starting over. Without transfer learning, most real-world tasks require impractical amounts of data and compute.

**Why learn it here?**  
After mastering training loops, checkpointing and transfer learning are the natural next steps for practical model building.

**Prerequisites:** Unit 06.8 (training loops), basic file I/O.

**Mental Model:**  
Checkpointing is saving your game progress. Transfer learning is hiring someone who already knows most of the job and just needs to learn your company's specific procedures.

**Core Concepts:**

- `torch.save()` and `torch.load()`
- Saving/loading model state_dict
- Saving optimizer state for resuming training
- Using pretrained models from `torchvision`
- Freezing layers
- Fine-tuning vs feature extraction

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
from torchvision import models

# --- Checkpointing ---
# Save
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pt')

# Load
checkpoint = torch.load('checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch'] + 1

# --- Transfer Learning ---
# Load pretrained ResNet18
model = models.resnet18(pretrained=True)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace final layer for new task
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 5)  # 5 classes

# Only train the new layer
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)

# Fine-tuning: unfreeze some layers after initial training
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.0001},
], lr=0.001)
```

**Real-World Example:**  
A team fine-tunes a pretrained ResNet50 for plant disease detection with only 2,000 images. They freeze all layers except the last block and classifier, achieving 94% accuracy in 10 minutes. Training from scratch with the same architecture would need 50,000+ images and hours of training.

**Decision Guidance:**

| Strategy | Use When | Data Required | Compute Required |
|---|---|---|---|
| Feature extraction | Small dataset, similar domain | Hundreds | Low |
| Fine-tune top layers | Medium dataset, similar domain | Thousands | Medium |
| Full fine-tuning | Large dataset or different domain | Tens of thousands | High |
| Train from scratch | Very different domain or sufficient data | Very large | Very high |

**Common Mistakes:**

- Forgetting to freeze layers (overwrites pretrained features immediately)
- Using too high a learning rate for fine-tuning (destroys pretrained knowledge)
- Not saving optimizer state (can't resume training properly)
- Loading model on GPU but checkpoint was saved on CPU

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Fine-tuning accuracy lower than feature extraction | LR too high destroying pretrained features | Reduce LR by 10-100x | Use discriminative LR (lower for earlier layers) |
| Cannot load checkpoint | Device mismatch | Check saved device vs current | Use `map_location=device` |
| Memory error loading large model | Model too large for GPU | Check model size | Use `torch.load(..., map_location='cpu')` |
| Fine-tuning overfits quickly | Dataset too small | Check dataset size | Freeze more layers, use augmentation |

**Hands-On Practice:**

1. Basic: save and load a model checkpoint. Verify predictions match.
2. Guided: implement fine-tuning on a pretrained ResNet for CIFAR-10.
3. Independent: compare feature extraction vs fine-tuning vs full training.
4. Realistic: implement training resumption from a checkpoint with optimizer state.
5. Challenge: use discriminative learning rates for different layer groups.

**Knowledge Check:**

- Why save `state_dict()` instead of the entire model?
- What happens if you don't freeze pretrained layers before fine-tuning?
- How do you choose between feature extraction and fine-tuning?
- Why use lower learning rates for pretrained layers?

**Exit Criteria:**

- You can save and load model checkpoints correctly.
- You can implement transfer learning with feature extraction and fine-tuning.
- You can resume training from a saved checkpoint.

**Next Step:** Learn CNNs for spatial data like images.

---

### Unit 06.10 — CNNs for Computer Vision

**What is it?**  
Convolutional Neural Networks (CNNs) use convolutional filters to detect spatial patterns (edges, textures, shapes) in grid-structured data like images. They are the foundation of computer vision.

**Why does it matter?**  
Fully connected layers ignore spatial structure — they treat each pixel independently. CNNs exploit spatial locality and translation invariance, making them dramatically more efficient and effective for images.

**Why learn it here?**  
After mastering MLPs and training loops, CNNs extend those concepts to spatial data. The building blocks (convolution, pooling) are new, but the training machinery is the same.

**Prerequisites:** Unit 06.4 (PyTorch), Unit 06.8 (training loops), basic linear algebra.

**Mental Model:**  
A CNN is a feature detector hierarchy. Early layers detect edges and colors. Middle layers combine edges into textures and patterns. Late layers combine patterns into object parts. The final layer maps these features to class probabilities.

```text
Input image → [Conv + ReLU] → edge detectors
           → [Conv + ReLU] → texture detectors
           → [Conv + ReLU] → part detectors
           → [Flatten] → [Linear] → class scores
```

**Core Concepts:**

- convolution: sliding filter over input to produce feature maps
- kernels/filters: small weight matrices that detect patterns
- stride, padding, output size
- pooling: downsampling (max pooling, average pooling)
- feature maps: activated filter outputs
- CNN architectures: LeNet, VGG, ResNet

**How It Works:**

- Convolution: `output[i,j] = Σ(input[i+m, j+n] * kernel[m,n])`
- Each filter produces one feature map
- Multiple filters → multiple feature maps (depth)
- Pooling reduces spatial dimensions, increases receptive field

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # 28x28x1 → 28x28x32
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 28x28x32 → 14x14x32
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 14x14x32 → 14x14x64
            nn.ReLU(),
            nn.MaxPool2d(2),                               # 14x14x64 → 7x7x64
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = SimpleCNN(num_classes=10)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

**Real-World Example:**  
A self-driving car system uses CNNs to detect lanes, pedestrians, and traffic signs. The CNN processes a 640x480 image in 30ms on a GPU, detecting multiple object classes simultaneously through feature map hierarchies.

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Shape mismatch error | Incorrect conv/pool output size | Calculate output size manually | Use `nn.Conv2d` padding='same' |
| CNN performs worse than MLP | Not enough data or wrong architecture | Check dataset size | Use transfer learning with pretrained CNN |
| Very slow training | Large input size or no pooling | Check input dimensions | Add pooling layers, reduce input size |
| Overfitting on small image dataset | Model too large for data | Check parameter count | Use data augmentation, pretrained models |

**Hands-On Practice:**

1. Basic: build a simple CNN and print output shapes at each layer.
2. Guided: train on MNIST. Compare to MLP performance.
3. Independent: add data augmentation (random crop, flip) and measure improvement.
4. Realistic: use a pretrained ResNet for CIFAR-10 with fine-tuning.
5. Challenge: implement a CNN that achieves >99% on MNIST.

**Knowledge Check:**

- What does a convolutional filter detect?
- Why does pooling help with translation invariance?
- How do you calculate the output size of a convolutional layer?
- Why are CNNs more parameter-efficient than MLPs for images?

**Exit Criteria:**

- You can build and train CNNs for image classification.
- You can explain convolution, pooling, and feature map hierarchies.
- You can use transfer learning with pretrained CNNs.

**Next Step:** Learn RNNs and LSTMs for sequential data.

---

### Unit 06.11 — RNNs, LSTMs, GRUs

**What is it?**  
Recurrent Neural Networks (RNNs) process sequential data by maintaining a hidden state that carries information from previous time steps. LSTMs and GRUs are improved variants that solve the vanishing gradient problem.

**Why does it matter?**  
Many real-world problems involve sequences: text, time series, speech, music, and genomics. CNNs and MLPs treat each input independently, but sequences have temporal dependencies that RNNs capture.

**Why learn it here?**  
After CNNs (spatial data), RNNs extend deep learning to temporal/sequential data. Understanding the limitations of vanilla RNNs motivates the LSTM/GRU designs.

**Prerequisites:** Unit 06.4 (PyTorch), Unit 06.8 (training loops), basic linear algebra.

**Mental Model:**  
An RNN is a person reading a sentence word by word. At each word, they update their understanding (hidden state) based on what they just read and what they already knew. An LSTM adds a notebook (cell state) that lets them remember important information and forget irrelevant details across long passages.

**Core Concepts:**

- recurrent connection: `h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b)`
- hidden state: memory carried across time steps
- vanishing/exploding gradients through time
- LSTM: cell state, forget gate, input gate, output gate
- GRU: simplified gating (update gate, reset gate)
- bidirectional RNNs
- sequence-to-sequence and many-to-one patterns

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn

# Simple RNN
rnn = nn.RNN(input_size=10, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)

# LSTM
lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)

# GRU
gru = nn.GRU(input_size=10, hidden_size=32, num_layers=2, batch_first=True, dropout=0.2)

# Example: sequence classification
class TextClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2,
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # *2 for bidirectional

    def forward(self, x):
        embedded = self.embedding(x)                   # (batch, seq_len, embed_dim)
        output, (hidden, cell) = self.lstm(embedded)   # output: (batch, seq_len, hidden*2)
        # Concatenate final hidden states from both directions
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden*2)
        return self.fc(hidden_cat)

# Example usage
model = TextClassifier(vocab_size=10000, embed_dim=128, hidden_dim=64, num_classes=3)
x = torch.randint(0, 10000, (32, 50))  # batch of 32 sequences, length 50
pred = model(x)
print(f"Output shape: {pred.shape}")  # (32, 3)
```

**Real-World Example:**  
A stock price predictor uses an LSTM with 2 layers and 128 hidden units to process 60-day windows of historical prices and technical indicators. The model outputs next-day price direction (up/down) with 62% accuracy — modest but profitable when combined with risk management.

**Gating Mechanism Comparison:**

| Architecture | Parameters | Memory | Speed | Use When |
|---|---|---|---|---|
| Vanilla RNN | Fewest | Lowest | Fastest | Short sequences, simple tasks |
| GRU | Few | Low | Fast | Medium sequences, general purpose |
| LSTM | Most | High | Moderate | Long sequences, complex dependencies |
| Bidirectional | 2x unidirectional | 2x | Slower | When full sequence context is available |

**Common Mistakes:**

- Using vanilla RNN for long sequences (vanishing gradients)
- Not packing padded sequences (wasting computation on padding)
- Forgetting to detach hidden states between batches
- Using bidirectional RNN for real-time prediction (needs future context)

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Loss is NaN | Exploding gradients in RNN | Check gradient norms | Clip gradients, reduce LR |
| Model can't capture long dependencies | Vanilla RNN vanishing gradients | Switch to LSTM/GRU | Use LSTM with sufficient hidden size |
| Memory error on long sequences | Sequence too long for GPU | Check sequence length | Reduce max length, use packing |
| Training very slow | No batching or very long sequences | Check DataLoader setup | Batch sequences, use shorter windows |

**Hands-On Practice:**

1. Basic: build an RNN and process a sequence. Inspect hidden state evolution.
2. Guided: train LSTM for sequence classification (e.g., sentiment).
3. Independent: compare RNN vs LSTM vs GRU on a long-sequence task.
4. Realistic: implement a time series forecaster with LSTM.
5. Challenge: use bidirectional LSTM and compare to unidirectional.

**Knowledge Check:**

- How does the LSTM cell state solve the vanishing gradient problem?
- What is the difference between GRU and LSTM gates?
- When would you use bidirectional vs unidirectional RNN?
- Why do you need to detach hidden states during training?

**Exit Criteria:**

- You can build and train RNNs, LSTMs, and GRUs in PyTorch.
- You can explain gating mechanisms and the vanishing gradient problem.
- You can choose the right recurrent architecture for a given task.

**Next Step:** Learn the attention mechanism, the foundation of transformers.

---

### Unit 06.12 — Attention Mechanism

**What is it?**  
Attention allows a model to dynamically focus on different parts of the input when producing each part of the output. Instead of compressing all information into a fixed-size hidden state, attention creates direct connections between relevant positions.

**Why does it matter?**  
Attention solved the information bottleneck problem of RNNs. It is the foundation of transformers, which now dominate NLP, vision, and beyond. Understanding attention is prerequisite to understanding modern AI.

**Why learn it here?**  
After RNNs, you have experienced the limitation of fixed-size hidden states. Attention is the natural solution, and understanding it from scratch prepares you for transformers.

**Prerequisites:** Unit 06.11 (RNNs/LSTMs), basic linear algebra (matrix multiplication, softmax).

**Mental Model:**  
Attention is like a search engine inside the model. For each output position, the model creates a query ("what am I looking for?") and matches it against keys ("what is available?") to decide which values ("what information to use") to attend to. The match scores determine how much each input contributes.

```text
Query (what I want) × Keys (what's available) = attention scores
Scores → softmax → attention weights
Weights × Values (content) = context vector
```

**Core Concepts:**

- queries, keys, values (Q, K, V)
- scaled dot-product attention: `Attention(Q, K, V) = softmax(QKᵀ / √d_k) V`
- self-attention: Q, K, V all come from the same sequence
- multi-head attention: parallel attention with different learned projections
- positional encoding: adding sequence order information
- causal masking: preventing attention to future positions

**Syntax & Implementation:**

```python
import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    query: (batch, seq_len_q, d_k)
    key:   (batch, seq_len_k, d_k)
    value: (batch, seq_len_k, d_v)
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, value), weights

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        attn_output, weights = scaled_dot_product_attention(Q, K, V, mask)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return self.W_o(attn_output)

# Example usage
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 10, 512)  # batch=2, seq_len=10, d_model=512
output = mha(x, x, x)  # self-attention
print(f"Output shape: {output.shape}")  # (2, 10, 512)
```

**Real-World Example:**  
In machine translation, attention allows the decoder to look at relevant parts of the input sentence when generating each output word. When translating "The cat sat on the mat" to French, generating "le" attends to "The", generating "chat" attends to "cat", and so on.

**Common Mistakes:**

- Forgetting to scale by `√d_k` (causes softmax saturation)
- Not implementing causal masking for autoregressive generation
- Using too many heads with too-small d_k (each head learns too little)
- Confusing self-attention with cross-attention (different Q/K/V sources)

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Attention weights nearly uniform | Not scaling, or d_k too large | Check scores before softmax | Add scaling factor |
| Model attends to wrong positions | No positional encoding | Add position information | Use sinusoidal or learned positional encoding |
| Memory error on long sequences | O(n²) attention complexity | Check sequence length | Use sparse attention or chunking |
| Causal model generates from future | Missing causal mask | Check mask implementation | Apply lower-triangular mask |

**Hands-On Practice:**

1. Basic: implement scaled dot-product attention from scratch.
2. Guided: build multi-head attention and verify output shapes.
3. Independent: implement positional encoding and add it to self-attention.
4. Realistic: build a simple attention-based sequence-to-sequence model.
5. Challenge: visualize attention weights and interpret what the model learns.

**Knowledge Check:**

- Why scale dot-product attention by `√d_k`?
- What is the difference between self-attention and cross-attention?
- Why is positional encoding necessary?
- What is the computational complexity of standard attention and why is it a problem?

**Exit Criteria:**

- You can implement scaled dot-product and multi-head attention from scratch.
- You can explain Q/K/V, masking, and positional encoding.
- You can diagnose attention-related training issues.

**Next Step:** Complete the phase with a synthesis project combining all deep learning concepts.

---

### Unit 06.13 — Deep Learning Synthesis & Review

**What is it?**  
A cumulative integration unit that combines all deep learning concepts — perceptrons, loss functions, backpropagation, PyTorch, regularization, optimizers, data loading, training loops, CNNs, RNNs, and attention — into a cohesive practice project.

**Why does it matter?**  
Knowing individual concepts is not enough. The learner must build an end-to-end deep learning system independently, making architectural and training decisions without a tutorial.

**Prerequisites:** All previous units in Phase 06.

**Mini Project: End-to-End Image Classifier**

**Objective:** Build, train, and evaluate a deep learning model for image classification using PyTorch. The project should demonstrate mastery of all Phase 06 concepts.

**Requirements:**

- Choose a dataset (CIFAR-10, Flowers, or a custom dataset)
- Implement a custom CNN architecture (not just using a pretrained model)
- Include proper data augmentation
- Implement a complete training loop with validation
- Apply at least 2 regularization techniques
- Use learning rate scheduling
- Save and load checkpoints
- Implement early stopping
- Compare at least 2 architectures or configurations
- Generate evaluation metrics (accuracy, per-class precision/recall, confusion matrix)
- Write a README explaining architecture choices and results

**Suggested Architecture:**

```text
Data loading → augmentation → model definition → training loop
    ↓                                                        ↓
DataLoaders with transforms                          validation monitoring
    ↓                                                        ↓
Train/val/test split                             checkpointing + early stopping
    ↓                                                        ↓
Evaluation metrics                               saved best model + results
```

**Evaluation Criteria:**

- Code runs from a clean environment
- Training loop is correctly structured (train/eval modes, grad management)
- Overfitting is identified and addressed
- Results are reproducible (random seeds)
- README explains every major decision
- Multiple configurations are compared with fair evaluation

**Advanced Extensions:**

- Transfer learning comparison (pretrained vs from scratch)
- Implement a simple RNN or attention module for a side task
- Deploy the model as a simple web API
- Add a confusion matrix and error analysis

**Knowledge Check:**

- Why did you choose your architecture over alternatives?
- What regularization techniques did you use and why?
- How did you verify your model is not overfitting?
- What would you change with more time or data?

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| MLP vs CNN | Tabular/non-spatial data | Image/spatial data | Simplicity vs spatial awareness |
| RNN vs Transformer | Short sequences, simple tasks | Long sequences, complex dependencies | Speed vs expressiveness |
| Dropout vs Batch Norm | Simple models, small data | Deep networks, training stability | Different regularization effects |
| Adam vs SGD | Rapid prototyping | Final model training | Speed vs possible generalization gap |
| Feature extraction vs fine-tuning | Small dataset, similar domain | Large dataset or domain shift | Data efficiency vs maximum performance |
| From scratch vs pretrained | Learning, very different domain | Similar domain, limited data | Understanding vs practicality |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model overfits immediately | Too many parameters, no regularization | Check train/val curves | Add dropout, reduce model size, add augmentation |
| Training loss NaN | Learning rate too high, bad data, exploding gradients | Print pre-activation values | Lower LR, gradient clipping, check data |
| Model trains but accuracy stuck | Underfitting or wrong loss function | Check baseline performance | Increase model capacity, verify loss choice |
| GPU memory error | Batch too large or model too big | Check `nvidia-smi` | Reduce batch size, use gradient accumulation |
| Transfer learning performs worse | Pretrained features destroyed | Check layer freezing | Freeze early layers, reduce LR |
| RNN can't learn long patterns | Vanishing gradients | Switch architecture | Use LSTM/GRU, add skip connections |
| Attention weights uninformative | Missing positional encoding | Check attention visualization | Add positional encoding, verify mask |

---

## Phase Review Checklist

- [ ] All 13 units completed
- [ ] Perceptron and activation functions implemented from scratch
- [ ] Loss functions implemented and compared
- [ ] Backpropagation implemented from scratch with numerical verification
- [ ] MLPs built and trained in PyTorch
- [ ] Regularization techniques applied and compared
- [ ] Optimizers and LR schedules compared
- [ ] Custom Datasets and DataLoaders built
- [ ] Complete training loops with validation implemented
- [ ] Checkpointing and transfer learning practiced
- [ ] CNNs built for image classification
- [ ] RNNs/LSTMs/GRUs built for sequence tasks
- [ ] Attention mechanism implemented from scratch
- [ ] Mini project completed and documented
- [ ] All debugging tables reviewed

## Mastery Check

Without following a tutorial, you should be able to:

1. Implement backpropagation from scratch in NumPy
2. Build and train MLPs, CNNs, and RNNs in PyTorch
3. Choose and implement appropriate loss functions
4. Apply regularization to prevent overfitting
5. Select and configure optimizers and LR schedules
6. Build efficient data pipelines with Datasets and DataLoaders
7. Implement complete training loops with proper monitoring
8. Save and load checkpoints
9. Use transfer learning effectively
10. Implement attention mechanisms from scratch
11. Debug common training failures (NaN loss, overfitting, vanishing gradients)
12. Build an end-to-end deep learning system independently

## Interview / Explain-Back Questions

- Explain backpropagation step by step. What is the chain rule and why does it matter?
- Why is ReLU preferred over sigmoid in hidden layers?
- What is the difference between L1 and L2 regularization?
- How does Adam optimizer differ from SGD?
- Why do we need `model.eval()` and `torch.no_grad()` during validation?
- Explain the difference between feature extraction and fine-tuning in transfer learning.
- How does a convolutional filter detect features? What do early vs late layers detect?
- Why do LSTMs solve the vanishing gradient problem better than vanilla RNNs?
- Explain scaled dot-product attention. Why scale by `√d_k`?
- How would you debug a model that has low training loss but high validation loss?
- When would you use a CNN vs an RNN vs an MLP?
- What are the trade-offs between fine-tuning a pretrained model and training from scratch?

## Exit Criteria

Move to Phase 07 only when you can independently build, train, debug, and evaluate deep learning models in PyTorch — including choosing architectures, loss functions, optimizers, regularization strategies, and data pipelines — and explain every decision you make.