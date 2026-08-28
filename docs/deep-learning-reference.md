# Deep Learning Reference

> **A quick-reference guide to deep learning.**

---

## Fundamentals

### The Neuron
- `output = activation(w·x + b)`
- Weighted sum of inputs + bias, passed through activation

### Activation Functions
- **Sigmoid**: `σ(x) = 1/(1+e⁻ˣ)` — squashes to (0,1)
- **Tanh**: squashes to (-1,1)
- **ReLU**: `max(0, x)` — most common for hidden layers
- **Leaky ReLU**: `max(0.01x, x)` — avoids dead neurons
- **Softmax**: converts logits to probabilities (for classification)

### Loss Functions
- **MSE**: regression
- **Cross-entropy**: classification
- **Binary cross-entropy**: binary classification

---

## Training

### Forward Pass
- Input → hidden layers → output
- Compute predictions

### Loss
- Compare predictions to ground truth

### Backpropagation
- Compute gradient of loss w.r.t. each weight
- Uses the chain rule
- Propagate error backward through the network

### Gradient Descent
- `θ ← θ - η·∇J(θ)`
- η = learning rate

### Optimizers
- **SGD**: basic gradient descent
- **Momentum**: adds velocity
- **Adam**: adaptive learning rates per parameter (most common)
- **AdamW**: Adam with weight decay

### Learning Rate Schedules
- Step decay, exponential decay, cosine annealing
- Warmup then decay

---

## Regularization

### L1/L2 Regularization
- Penalize large weights
- L2 (weight decay): `λ||w||²`
- L1: `λ||w||₁` (sparse)

### Dropout
- Randomly zero out neurons during training
- Prevents co-adaptation

### Batch Normalization
- Normalize activations within a batch
- Stabilizes and speeds up training

### Early Stopping
- Stop when validation loss stops improving

### Data Augmentation
- Create variations of training data

---

## Architectures

### MLP (Multi-Layer Perceptron)
- Fully connected layers
- For tabular data, simple patterns

### CNN (Convolutional Neural Network)
- For images
- Convolution, pooling, fully connected layers
- Learns spatial features hierarchically

### RNN (Recurrent Neural Network)
- For sequences
- Hidden state carries information across time steps
- Suffers from vanishing gradients

### LSTM / GRU
- RNNs with gating mechanisms
- Handle long-range dependencies

### Attention
- Weigh importance of different input parts
- The foundation of transformers

### Transformer
- Self-attention + feed-forward layers
- Parallelizable, handles long sequences
- The basis of modern LLMs

---

## PyTorch Workflow

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define model
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

# Training loop
for epoch in range(num_epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
```

---

## Key Concepts

### Transfer Learning
- Use a pretrained model, fine-tune on your data
- Saves time and data

### Checkpointing
- Save model state periodically
- Resume training from checkpoint

### Vanishing/Exploding Gradients
- Gradients become too small/large in deep networks
- Solutions: ReLU, batch norm, residual connections, proper initialization

### Overfitting in Deep Learning
- Model memorizes training data
- Solutions: regularization, dropout, more data, early stopping

---

## When to Use Deep Learning

**Use when:**
- Large amounts of data
- Complex patterns (images, text, audio)
- Traditional methods underperform

**Avoid when:**
- Small data
- Need interpretability
- Simple linear relationships
- Limited compute
