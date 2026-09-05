# 06. Policy Gradient (REINFORCE)

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | REINFORCE (Monte-Carlo Policy Gradient) |
| Category | Reinforcement Learning |
| Type | Policy-based, Model-free, On-policy, Monte-Carlo |
| Parametric / Non-parametric | Parametric (policy π_θ(a\|s), usually a neural network) |
| Generative / Discriminative | N/A (decision-making / stochastic policy) |
| Main Objective | Maximize expected discounted return J(θ) directly by moving the policy parameters along the gradient of the expected return |
| Input | State s (high- or low-dimensional); returns G_t computed from episodes |
| Output | Action probabilities π_θ(a\|s) (and a sampled action) |
| Core Idea | Sample episodes, compute full returns G_t, then update θ += α·G_t·∇log π_θ(a_t\|s_t) — increase probability of actions that led to high return |
| Typical Use Cases | Continuous/discrete action spaces, policies that should be stochastic (games, rock-paper-scissors), robotics with continuous control |

---

## 02. One-Line Definition

### Beginner Definition
REINFORCE is a "trial, remember, reinforce" method: it plays whole games, then makes the moves that earned more reward more likely and the moves that earned less less likely.

### Technical Definition
REINFORCE directly maximizes the expected return J(θ) = E_τ~π_θ[G_0] by estimating its gradient via the log-derivative trick: ∇J(θ) = E[Σ_t G_t ∇log π_θ(a_t\|s_t)], approximated with Monte-Carlo sample episodes, giving the update θ ← θ + α·G_t·∇log π_θ(a_t|s_t) per visited step.

---

## 03. Intuition

Imagine a juggler practicing. Every routine (episode) ends with a score (return). After each routine the juggler doesn't know which throw caused what — but he tries to **reinforce** the throws that appeared in high-scoring routines and dampen the throws from low-scoring ones.

"Which throw appeared" is the probability his policy assigned to that throw. The trick `∇log π(a|s)` answers: "which direction in parameter space makes this exact throw *more* likely?" Multiply by the total score G_t and update — high score → make that throw more probable; low/negative score → make it less probable.

Because he uses the *whole episode's* score, REINFORCE is a Monte-Carlo method: high variance — one lucky routine can strongly boost an unrelated throw.

Step-by-step reasoning:
1. Roll out a whole episode with the current policy, recording states, actions, and rewards.
2. Work backward to compute the return G_t for every step.
3. For each step, compute the log-probability gradient of the chosen action.
4. Increasing-score actions get their probabilities raised; decreasing ones lowered.
5. Repeat, and the policy's probability mass concentrates on good actions.

---

## 04. Problem It Solves

**Problem:** Q-learning needs an action-value function; DQN needs discrete actions (argmax); and *greedy maximization* over actions doesn't exist for continuous action spaces. We also often want a **stochastic** policy (e.g., bluff strategies), which greedy policies can't express.

**Example:** A robot arm's joint torques are continuous — you can't run argmax over uncountably many torques. Also, in a game of scissors-paper-stone, a deterministic policy can be exploited; you want probabilities.

**What we want:** A method that:
- works directly on continuous actions,
- produces explicit action probabilities,
- optimizes expected return from **first principles** (no value function needed for the update).

**Why policy gradients help:** the policy π_θ(a|s) is itself a parametrized distribution (e.g., Gaussian over torques). The REINFORCE gradient simply points "in the direction that makes the good actions from this episode more likely" — no max, no discretization, stochastic by construction.

**Small example:** In a 2-action bandit, the policy outputs P(action 1). If action-1 episodes carry returns +1 and action-0 episodes −1, the gradient nudges P(action 1) upward — a pure "reinforce the good choice" loop.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning
├── Unsupervised Learning
└── Reinforcement Learning
    ├── Model-Free
    │   ├── Value-Based (Q-learning, SARSA, DQN)  — files 01,02,05
    │   └── Policy-Based
    │       ├── REINFORCE (this file) — MC policy gradient
    │       ├── Policy-gradient with baseline
    │       └── Actor-Critic — file 07 (gradient + learned baseline/critic)
    └── Model-Based
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Policy π_θ | "The plan", parametrized | Probability distribution over actions given state |
| Trajectory τ | One whole episode's path | Sequence (s₀,a₀,r₁,...,s_T) |
| Return G_t | Total future reward | G_t = Σ γᵏ r_{t+k+1} |
| Expected return J(θ) | Average score of policy | J(θ) = E_τ~π_θ[G_0] |
| Log-derivative trick | math identity for gradients | ∇log π = ∇π/π |
| Score function | ∇log π_θ(a\|s) | Direction to make action more likely |
| Gradient ascent | Move params up the hill | θ ← θ + α·∇J |
| MC estimate | Sample-based gradient | Uses whole-episode returns |
| High variance | Noisy gradient estimates | Each episode is one correlated sample |
| Baseline | Subtract a constant from G_t | Reduces variance without bias (added later; critic in file 07) |
| On-policy | Learn from your own policy | REINFORCE is on-policy (uses π_θ's own episodes) |
| Stochastic policy | Randomness in actions | π_θ(a\|s) probabilities |

---

## 07. Input and Output

**Input (training):**
- Episodes sampled from the environment under the current policy π_θ:
  lists of (s_t, a_t, r_{t+1}).

**Input (hyperparameters):**
- γ, learning rate α, network architecture, (later) baseline trick, number of episodes.

**Output (training):**
- Policy parameters θ (e.g., neural weights mapping state → action distribution).

**Output (prediction / use):**
- π_θ(a|s) probabilities; behavior: sample an action from them (or take argmax of the mean for deterministic deployment in continuous tasks).

**Parameters learned:** θ — policy network weights.

**Hyperparameters:** γ, α, architecture, episodes, (for Gaussian policies: initial log-std), optional baseline.

---

## 08. Mathematical Foundation

The objective is the expected return of the policy:

```text
J(θ) = E_τ~π_θ[ G_0 ] = E_τ~π_θ[ Σ_{t} γ^t r_{t+1} ]
```

**The discount factor γ intuitively:** a reward now is worth more than the same reward later — γ decides how many future steps the policy "cares" about.

We want ∇_θ J(θ). The chain of dependencies is circular (actions depend on θ; states depend on actions), so we can't naively push ∇ through. The **log-derivative trick** breaks the loop:

```text
∇_θ π_θ(τ) = π_θ(τ)·∇_θ log π_θ(τ)
```

Expanding log π_θ(τ) = Σ_t log π_θ(a_t|s_t) (dynamics terms don't depend on θ) yields the REINFORCE gradient:

```text
∇J(θ) = E_τ~π_θ[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t|s_t) ]
```

The **score function** ∇log π_θ(a|s) is the direction that increases the probability of the sampled action.

**Required math concepts:** expectation, probability density functions, gradients, the chain rule, Gaussian/softmax parameterization.

---

## 09. Core Formula

### The policy-gradient (REINFORCE) update

```text
θ ← θ + α · G_t · ∇_θ log π_θ(a_t | s_t)
```

#### Meaning
After observing episode-return G_t for the action a_t taken in state s_t, nudge the policy parameters so that the probability of a_t is pushed up (larger G_t) or down (smaller G_t).

#### Symbols
- `θ` = policy parameters
- `α` = learning rate
- `G_t` = return (total discounted future reward) after time t
- `π_θ(a_t|s_t)` = probability the policy assigns to a_t in s_t
- `∇_θ log π_θ(a_t|s_t)` = score function (gradient direction)

#### Intuition
"Good episode + that action → make it more likely." The multiplication by G_t is the credit assignment: actions with high return get promoted.

#### Example (one REINFORCE step — hand-computed below in Section 15):
Policy has two actions; after an episode, G_t = 2.0 and the score function for the taken action is valued, giving a positive step.

---

### The gradient (full form)

```text
∇_θ J(θ) = E_τ~π_θ[ Σ_{t} G_t ∇_θ log π_θ(a_t|s_t) ]
```

#### Meaning
The expected-gradient form; in practice we approximate with the sampled episode sum.

#### Symbols
- `E_τ~π_θ` = expectation over episodes sampled with π_θ
- `Σ_t` = sum over the episode's timesteps
- `G_t` = return at step t

#### Intuition
Unbiased — the sample sum's expectation is exactly the true gradient. That unbiasedness is why REINFORCE is the canonical policy-gradient method — but it comes with high variance.

---

### Baseline variant (anticipation of actor-critic)

```text
∇J(θ) = E[ Σ_t (G_t − b(s_t)) ∇log π_θ(a_t|s_t) ]
```

#### Meaning
Subtracting any state-dependent baseline b(s_t) (which must not depend on a_t) keeps the gradient unbiased while reducing variance.

#### Symbols
- `b(s_t)` = baseline (e.g., the value estimate V(s_t))

#### Intuition
The term (G_t − V(s_t)) is the **advantage**: "how much better than average was this action?" — this is the bridge to Actor-Critic (file 07).

---

## 10. Derivation

**Step 1 — Objective:**
```text
J(θ) = E_τ[ G_0 ] = ∫ π_θ(τ) G_0 dτ
```

**Step 2 — Take the gradient through the reward:**
```text
∇_θ J(θ) = ∫ ∇_θ π_θ(τ) G_0 dτ
```

**Step 3 — Log-derivative trick:**
```text
∇_θ π_θ(τ) = π_θ(τ) ∇_θ log π_θ(τ)
⇒ ∇_θ J(θ) = ∫ π_θ(τ) ∇_θ log π_θ(τ) G_0 dτ = E_τ[ ∇_θ log π_θ(τ) G_0 ]
```

**Step 4 — Expand the log-probability:**
```text
π_θ(τ) = p(s_0) Π_t π_θ(a_t|s_t) p(s_{t+1}|s_t,a_t)
log π_θ(τ) = log p(s_0) + Σ_t log π_θ(a_t|s_t) + Σ_t log p(s_{t+1}|s_t,a_t)
```
Only the middle term depends on θ, so:
```text
∇_θ log π_θ(τ) = Σ_t ∇_θ log π_θ(a_t|s_t)
```

**Step 5 — Per-step greediness due to causality:**
The return at t only uses rewards from t onward, so the per-step form uses **G_t** (not G_0) for step t's score:
```text
∇_θ J(θ) = E_τ[ Σ_t G_t ∇_θ log π_θ(a_t|s_t) ]
```

**Step 6 — Monte-Carlo estimate:**
Replace the expectation by one sampled episode:
```text
∇J(θ) ≈ Σ_t G_t ∇_θ log π_θ(a_t|s_t)
θ ← θ + α·∇J(θ)
```
This is REINFORCE. Subtracting a baseline b(s_t) (any action-independent function) steps to the advantage form:
```text
∇J(θ) ≈ Σ_t (G_t − b(s_t)) ∇log π_θ(a_t|s_t)
```
with E[b(s_t)∇log π]=0 (because Σ_a ∇π = 0) — hence unbiased, but lower variance. Choosing b=V(s_t) yields Actor-Critic (file 07).

---

## 11. How the Algorithm Works

```text
Initialize policy π_θ
    ↓
Loop:
    Sample an episode: record (s_t, a_t, r_{t+1}) for all t
    ↓
    Walk backward, computing returns: G_t = r_{t+1} + γ·G_{t+1}
    ↓
    Compute per-step loss contribution: −G_t·log π_θ(a_t|s_t)
    (sign flipped because optimizers minimize)
    ↓
    Backprop: θ ← θ + α·Σ_t G_t ∇log π_θ(a_t|s_t)
    ↓
    (optional) update baseline b(s)
Repeat
    ↓
Final model: policy π_θ
    ↓
Deployment: sample actions from π_θ (or argmax of distribution mean)
```

---

## 12. Training Process

**Pre-training:** build policy network (softmax for discrete; Gaussian head for continuous), choose γ, α.

**During training:**
- Generate one (or a batch of) complete episode(s).
- Convert the episode into returns G_t (backward recursion).
- One gradient ascent update per episode using all steps' (G_t, score) pairs.
- REINFORCE updates only AFTER the episode ends (Monte Carlo property).

**What is learned:** policy parameters θ shaping action probabilities across states.

**Changes per update (one episode):**
```text
G ← 0
loss ← 0
for t from last step back to 0:
    G = γ·G + r_{t+1}
    loss += −G·log π_θ(a_t|s_t)
θ ← θ − α·∇_θ loss      (minimizer; equivalent to gradient ascent on J)
```

**Stopping:** fixed budget of episodes or evaluation return plateau.

**Final model contents:** policy weights θ producing π_θ(a|s).

---

## 13. Objective Function / Loss Function

REINFORCE maximizes expected return; implemented with optimizers, it minimizes the negated sample estimate:

```text
J(θ) = E_τ[ G_0 ]          (the true objective)
L(θ) = −(1/N) Σ_episodes Σ_t G_t log π_θ(a_t|s_t)   (surrogate loss for descent)
```

**Why chosen:** the log-prob with return weighting is the *unbiased* gradient estimate of J; negating turns ascent into familiar descent.

**Interpretation:** "False if the action was improbable but earned high returns." Low loss (in the surrogate) = policy likely under high-return actions.

**Training objective ≠ evaluation metric:** the surrogate loss is not comparable to "task score"; evaluation uses mean episode return of the learned policy. Do not tune solely on surrogate loss.

---

## 14. Optimization

**Definition:** perform gradient ascent on J(θ).

**Method:** Monte-Carlo estimate of ∇J from full episodes; standard deep optimizers (SGD/Adam) descend the negated surrogate.

**Gradient:** Σ_t G_t ∇log π_θ(a_t|s_t) — each term scales the score direction by the episode return.

**Learning rate α:** too high → policy entropy collapses (deterministic and exploitable, or unstable); too low → glacial progress.

**Convergence:** unbiased gradient estimates → converges to local optima of J for smooth policies with step sizes satisfying Σα=∞, Σα²<∞; but variance makes progress painful.

**Local/global:** J is non-concave (network); the method finds local optima of the return landscape.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified, scalar-parameter policy)

A 2-action bandit (state-free). Policy: π_θ(action 1) = σ(θ) (sigmoid), so P(action 0) = 1 − σ(θ). We collect one episode (single step): action 1 sampled; reward +2.

**Step 1 — compute the return:**
```text
G_0 = r = 2
```

**Step 2 — the score function:**
For a Bernoulli policy p = σ(θ), score = ∇log P(a).
```text
If a=1:  ∇log σ(θ) = (1−p)
If a=0:  ∇log (1−σ(θ)) = −p
Here a=1 ⇒ score = 1 − p.
With θ=0: p = σ(0) = 0.5 ⇒ score = 0.5.
```

**Step 3 — the REINFORCE update (α = 0.1):**
```text
θ ← θ + α·G·score = 0 + 0.1·2·0.5 = 0.1
```

**Step 4 — new probability:**
```text
p = σ(0.1) = 1/(1+e^{−0.1}) ≈ 0.525
```
Action 1 (which produced G=2) became more likely: 0.5 → 0.525. ✓

**Negative-reward counter-example:** if the sampled action had G=−2:
```text
θ ← 0 + 0.1·(−2)·0.5 = −0.1  ⇒  p = 0.475  (action 1 now less likely) ✓
```

**One more step to verify direction:** with G=2 again, score now = 1−0.525=0.475:
```text
θ ← 0.1 + 0.1·2·0.475 = 0.1 + 0.095 = 0.195
p = σ(0.195) ≈ 0.549   (keeps rising) ✓
```

| update | G | score (1−p) | θ before | θ after | p after |
|---|---|---|---|---|---|
| 1 | +2 | 0.500 | 0 | 0.100 | 0.525 |
| 2 | +2 | 0.475 | 0.100 | 0.195 | 0.549 |
| 1′ | −2 | 0.500 | 0 | −0.100 | 0.475 |

Hand-verified: sigmoid-gradient arithmetic confirmed; the direction (reinforce good, dampen bad) behaves exactly as intended.

---

## 16. Visual Explanation

### Agent–policy loop

```text
   π_θ(a|s)   ── sample action a ──▶  environment
      ▲                                  │
      │                                  ▼
   θ += α·G·∇log π(a|s)  ◀───  episode return G, reward r
      (after full episode)
```

### Policy-network output heads

```text
Discrete:  s → net → logits → softmax → [p(a₀), p(a₁), p(a₂)]
Continuous: s → net → mean μ(s), log-std → Gaussian N(μ,σ) → sample a
```

### The reinforce rule (visual)

```text
episode return G > 0  ──▶  raise p(action)   (good moves get more likely)
episode return G < 0  ──▶  lower p(action)   (bad moves get less likely)
```

---

## 17. Algorithm / Pseudocode

```text
1. Init policy π_θ (θ), γ, learning rate α
2. Repeat until convergence:
     a. sample an episode: (s₀,a₀,r₁), (s₁,a₁,r₂), ..., (s_{T-1},a_{T-1},r_T)
     b. G ← 0 ;  accumulate = 0
     c. for t = T-1 down to 0:
          G ← γ·G + r_{t+1}
          accumulate += G · ∇_θ log π_θ(a_t|s_t)
     d. θ ← θ + α·accumulate
3. Return π_θ
```

---

## 18. From-Scratch Implementation

A minimal-policy REINFORCE on a small gridworld, using numpy for the policy and manual score computation.

```python
import numpy as np

class ReinforceGrid:
    def __init__(self, n_actions, gamma=0.9, lr=0.1):
        self.gamma = gamma
        self.lr = lr
        self.theta = np.zeros(n_actions)      # state-free softmax policy
        self.type = "simple"

    def probs(self):
        e = np.exp(self.theta - self.theta.max())
        return e / e.sum()

    def sample_action(self, rng):
        p = self.probs()
        return rng.choice(len(p), p=p)

    def score(self, a):
        p = self.probs()
        s = -p.copy()
        s[a] += 1.0                            # ∇log softmax = onehot − p
        return s

    def update(self, actions, returns):
        grad = np.zeros_like(self.theta)
        for a, G in zip(actions, returns):
            grad += G * self.score(a)
        self.theta += self.lr * grad

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    agent = ReinforceGrid(2, gamma=0.9, lr=0.1)
    for ep in range(300):
        # placeholder: a real env would provide (action, reward) sequences.
        a = agent.sample_action(rng)
        r = 1.0 if a == 1 else -0.5            # action 1 is better
        returns = [r]
        agent.update([a], returns)
        if ep in (0, 99, 299):
            print(f"ep {ep}: p(a1) = {round(agent.probs()[1],3)}")
```

(For a full state-conditioned REINFORCE with a neural network, see the PyTorch implementation below.)

---

## 19. Code Explanation

```text
Line:  def probs(self): softmax over theta
   What: converts policy parameters into action probabilities
   Why: defines π_θ(a|s) (state-free bandit case)
   Math: π(a) = exp(θ_a)/Σ exp(θ_i)

Line:  def score(self, a): return onehot−p
   What: the score function ∇log π for a softmax policy
   Why: REINFORCE's update direction
   Math: ∇log π(a) = e_a − p

Line:  grad += G * self.score(a)
   What: accumulate the episode's reinforce sum
   Why: Monte-Carlo gradient estimate
   Math: Σ_t G_t ∇log π(a_t|s_t)

Line:  self.theta += self.lr * grad
   What: gradient ascent on J
   Why: maximize expected return
   Math: θ ← θ + α·∇J
```

---

## 20. Library Implementation

PyTorch REINFORCE on CartPole (discrete policy), a standard clean example:

```python
import torch
import torch.nn as nn
import gymnasium as gym

class Policy(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(n_s, 128), nn.ReLU(),
                                nn.Linear(128, n_a))
    def forward(self, s):
        return torch.softmax(self.fc(s), dim=-1)

env = gym.make("CartPole-v1")
policy = Policy(env.observation_space.shape[0], env.action_space.n)
opt = torch.optim.Adam(policy.parameters(), lr=1e-2)
gamma = 0.99

for ep in range(1000):
    states, actions, rewards = [], [], []
    s, _ = env.reset()
    done = False
    while not done:
        logits = policy(torch.FloatTensor(s))
        dist = torch.distributions.Categorical(logits)
        a = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(a))
        states.append(s); actions.append(log_prob); rewards.append(0.0)
        s, r, term, trunc, _ = env.step(a)
        rewards[-1] = r if r != 0 else 0.0
        done = term or trunc
    G = 0.0
    loss = torch.tensor(0.0)
    for log_prob, r in zip(reversed(actions), reversed(rewards)):
        G = gamma*G + r
        loss = loss + (-G)*log_prob     # negated REINFORCE surrogate
    opt.zero_grad(); loss.backward(); opt.step()
    if ep % 100 == 0:
        print(f"ep {ep}, surrogate-loss {loss.item():.2f}")
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Learning rate α | Step size | High → unstable/entropy collapse; low → slow | 1e-3–1e-2 (Adam) |
| Discount γ | Future-reward weight | Low → myopic | 0.99 |
| Episodes per update | Batch of rollouts | Larger batch → lower variance | 10–1000 |
| Architecture | Policy net size | Capacity | state-size dependent |
| Action distribution | softmax/Gaussian | Discrete/continuous choice | — |
| Initial log-std (continuous) | Exploration scale | Noise in sampled actions | tuned per task |
| Baseline (later) | Value baseline | Variance reduction | see file 07 |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- θ — all policy-network weights/biases.

### Hyperparameters (chosen)
- γ, α, batch size (episodes), architecture, distribution family, (optional) baseline network structure, training-budget episodes.

---

## 23. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Differentiable policy | π_θ smooth in θ | Gradient ascent needs it | Verify softmax/Gaussian | Use policy gradient variants assuming differentiability needs it |
| Finiteness of returns | G_t bounded/integrable | Objective defined | ε-clip rewards | Normalize returns |
| Full episodes available | MC needs them | G_t ends at episode end | Check termination | Use TD-style critics (actor-critic) |
| Exploration through stochasticity | Policy must sample | Needed to see outcomes | Track entropy | Add entropy bonus |
| On-policy data | Episodes from current π_θ | Gradient is for THIS policy | Freeze policy while sampling | Off-policy importance sampling |

---

## 24. Data Requirements

- **Data type:** full episodes of (s, a, r), generated by the current policy (on-policy).
- **State space:** arbitrary (low-dim vectors or high-dim inputs).
- **Action space:** discrete (softmax) or continuous (Gaussian) — REINFORCE handles both.
- **Rewards:** scalar; normalize/whiten returns for stability.
- **Sample budget:** needs many episodes (high variance).
- **No missing values/outliers/scaling:** preprocess observations as usual; scale returns.

---

## 25. Feature Scaling

**Recommended:** normalize observations (StandardScaler or pixel rescaling) for stable neural training. Also recommended: normalize/standardize returns across a batch (whitening) to tame the raw G_t magnitudes — this is a *sample-level* scaling, not feature scaling, but it stabilizes REINFORCE substantially.

---

## 26. Evaluation Metrics

| Metric | Definition | Interpretation | Use When | Avoid When |
|---|---|---|---|---|
| Mean episode return (eval) | Avg Σ rewards across fresh episodes | True policy quality | All cases | — |
| Success rate | Fraction solving the task | Task completion | Goal tasks | Non-goal |
| Policy entropy | −Σp log p over actions | Exploration health | Diagnosing collapse | Terminal reporting |
| Return variance across episodes | Spread of outcomes | Confidence | Reporting stability | Tiny sample sizes |

**Important:** the **surrogate loss** (minus return-weighted log-likelihood) is NOT an evaluation metric — it can grow/shrink while true returns behave differently. Evaluate the policy by resetting exploration to deterministic (take the distribution mean / argmax) and averaging episode returns.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Works on continuous actions | No argmax needed over actions |
| Explicit stochastic policies | Models mixed strategies / exploration |
| Directly optimizes the return | Objective = the real score |
| Unbiased gradient estimator | Correct expected direction of progress |
| Simple core idea | Easy to implement and reason about |
| Flexible policy parameterization | Any differentiable distribution |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| High variance | Needs huge numbers of episodes |
| Slow learning | Far slower than value-based methods on easy tasks |
| On-policy sample waste | Can't reuse old episodes efficiently |
| Credit-assignment coarseness | Whole-episode return blamed on every step |
| Premature entropy collapse | Policy can lock in and never explore |
| No theoretical global guarantee | Converges to local optima only |
| Episode-bound | Can't update mid-episode |

---

## 29. When to Use

✓ Continuous action spaces (no argmax possible).
✓ You want an explicitly stochastic policy.
✓ Episodes are cheap to simulate.
✓ The task's optimal policy is genuinely stochastic.
✓ You want to optimize the return directly, without a value function.
✓ As the actor component that actor-critic methods extend.

---

## 30. When NOT to Use

✗ You have few environment calls available (variance kills it).
✗ Long-horizon/rare-reward tasks (credit assignment hopeless).
✗ Small discrete tasks where Q-learning converges faster.
✗ You need sample efficiency / offline data.
✗ When a deterministic policy is obviously optimal and data is scarce.

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Robot arm continuous control | joint states | REINFORCE (continuous head) | Torque/angle commands |
| Poker / bluffing agents | game state | REINFORCE (stochastic) | Betting probabilities |
| Portfolio allocation | market features | REINFORCE | Allocation weights |
| Dialogue policies | dialogue state | REINFORCE | Utterance-type choice |
| Power grid dispatch | grid state | REINFORCE | Generator setpoints |

---

## 32. Failure Cases

- **Variance blow-up:** stochastic env + long episodes → gradient directions dominated by noise.
- **Entropy collapse:** policy becomes near-deterministic early → no further exploration → stuck.
- **Reward-scale pathology:** huge outliers in G_t → giant updates → instability.
- **Exploration starvation:** Gaussian std shrinks to ~0 → deterministic-but-arbitrary policy.
- **Batch-noise sensitivity:** single-episode updates are worst; very noisy.
- **Non-stationary eval:** on-policy samples become stale the instant θ changes.

---

## 33. Overfitting and Underfitting

- **Underfitting:** too small a policy network → can't represent the optimal distribution.
- **Overfitting (memory of episodic luck):** the gradient trusts the specific sampled episodes; a lucky-events batch can concentrate probability on bad actions. Large batch sizes average this out. In tabular-format or high-capacity settings, over-reliance on a few episodes is the standing risk.
- **Balance:** right-size network to state complexity; use batches of episodes; watch policy entropy as a health check.

---

## 34. Bias-Variance Perspective

- **Bias:** REINFORCE's gradient estimate is **unbiased** — sampling exactly estimates the true ∇J (thanks to the log-derivative trick).
- **Variance:** extremely high — G_t includes the whole episode's randomness, and actions under the *same state* can pull in opposite directions.
- **Reduction path:** a baseline b(s) reduces variance without bias (subtract an action-independent function); the learned baseline V(s) gives the advantage and leads directly to Actor-Critic (file 07).

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| REINFORCE | MC policy gradient | Unbiased, simple, continuous actions | High variance, slow | Proof-of-concept policy gradient |
| Q-Learning/DQN | Value-based | Efficient for small discrete worlds | Requires argmax, no stochastic policy | Discrete control |
| REINFORCE + baseline | Subtract b(s) | Lower variance, still unbiased | Needs a good baseline | Intermediate step |
| Actor-Critic | Critic + actor | Much lower variance | Two networks, added bias | Practical deep RL |
| A2C/A3C | Synchronous/asynchronous AC | Stable & parallel | More machinery | Part 2 coverage |

---

## 36. Algorithm Selection Guide

```text
Need a policy (not just values)?
├── Continuous actions?
│   ├── YES → Policy gradient family
│   │         Variance OK? → REINFORCE
│   │         Need lower variance → Actor-Critic (file 07)
│   └── NO → also fine; DQN may be cheaper if discrete
├── Want stochastic deterministic mix → Actor-Critic / entropy bonus
└── Rare interaction budget → value-based or model-based instead
```

---

## 37. Common Mistakes

```text
❌ Using θ += α·G·∇log π but forgetting G is the WHOLE-Episode return
Why wrong: only full returns make REINFORCE unbiased.
Correct: compute G_t backward; never use single-step rewards.

❌ Backpropagating through the sampled action
Why wrong: the action is data; gradients go to policy log-probs only.
Correct: loss = −Σ G_t · log π(a_t|s_t); no grad through samples.

❌ No entropy control → premature collapse
Why wrong: all probability mass on one action → no exploration.
Correct: track entropy, add entropy bonus, decay it.

❌ Forgetting to normalize returns (huge G values)
Why wrong: giant updates, unstable θ.
Correct: standardize returns per batch (mean/std) as a variance-reduction trick.

❌ Judging training by the surrogate loss
Why wrong: surrogate ≠ task score.
Correct: evaluate fresh deterministic episodes; average returns.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is a policy gradient?**
A: Optimizing the policy π_θ directly by moving θ along the gradient of expected return.

**Q2. What is REINFORCE's update?**
A: θ ← θ + α·G_t·∇log π_θ(a_t|s_t), using full-episode returns.

**Q3. What is the score function?**
A: ∇log π_θ(a|s) — the direction making action a more likely.

### Intermediate
**Q4. Why is REINFORCE a Monte-Carlo method?**
A: It uses complete episode returns G_t as unbiased gradient estimates — no bootstrapping.

**Q5. Why is the gradient unbiased but high-variance?**
A: The log-derivative trick gives an exact expectation; but each sampled return carries the whole episode's randomness.

**Q6. How do you handle continuous actions?**
A: Parameterize π_θ as a Gaussian (mean μ_θ(s), std σ_θ(s)) and score ∝ (a−μ)/σ².

### Advanced
**Q7. What does a baseline do and why is it unbiased?**
A: b(s) subtracts an action-independent function: E[b(s)∇log π] = 0, so no bias, but variance drops.

**Q8. What advantage term does the value baseline produce?**
A: G_t − V(s_t) — "how much better than average" — the bridge to actor-critic.

**Q9. How is entropy used to prevent collapse?**
A: Add +β·H(π_θ(·|s)) to the objective to keep probabilities spread during training.

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
```text
J(θ) = E[ G_0 ]
∇J(θ) = E[ Σ_t G_t ∇log π_θ(a_t|s_t) ]
REINFORCE step: θ ← θ + α G_t ∇log π_θ(a_t|s_t)
Policy Gradient Theorem (general): ∇J ∝ E[ Q^π(s,a) ∇log π(a|s) ]
```

**Concepts likely tested:**
- Log-derivative trick
- Why MC (full returns) ⇒ high variance
- On-policy nature
- Softmax/Gaussian score functions
- Baseline indifference property (unbiasedness)

> **Representative pattern question (NOT a past GATE PYQ):** "Policy σ(θ)=0.5 takes action a=1, gets G=+2, score=0.5, α=0.1. What are the new θ and probability?" → θ=0.1, p≈0.525 (Section 15 arithmetic).

**Common traps:**
- Using per-step rewards instead of full returns.
- Treating the log of the episode probability as needing the environment dynamics (they cancel in the gradient).
- Confusing policy gradient with value iteration.
- Forgetting that REINFORCE is on-policy (samples must come from current π_θ).

---

## 40. Coding Practice

**Level 1 — Basic:** Implement softmax policy score (onehot − p) in numpy.
**Level 2 — Bandit:** REINFORCE on the 2-action bandit; verify p rises for the good arm.
**Level 3 — CartPole:** PyTorch REINFORCE on CartPole; plot mean return.
**Level 4 — Variance:** run with batch size 1 vs 32 episodes; compare return curves.
**Level 5 — Continuous:** Gaussian-policy REINFORCE on a simple continuous task (Pendulum-v1).
**Level 6 — Entropy:** add entropy bonus; show collapse prevention.
**Level 7 — Real-world case study:** a custom continuous-control gym (e.g., lunar lander variant), fingerprint the three failure modes (variance, entropy collapse, reward-scale), and fix each with batch normalization, entropy bonus, and return whitening; report greedy eval returns.

---

## 41. Practical ML Workflow

```text
Problem → continuous? stochastic-policy need? define reward
   ↓
Environment → gymnasium / custom simulator (episodic)
   ↓
Preprocess → normalize obs, define action head (softmax/Gaussian)
   ↓
Policy design → net, γ, α, batch size, entropy/β
   ↓
Train → sample episodes → returns → REINFORCE step
   ↓
Evaluate → deterministic eval returns (take mean/argmax)
   ↓
Tune → batch size, α, entropy β, return normalization
   ↓
Error analysis → entropy curve, gradient-norm curve, return variance
   ↓
Deploy → deterministic policy for most control tasks
   ↓
Monitor → re-train on env drift
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Episode sampling | O(T·env) | T episode length |
| Return computation | O(T) | Backward recursion |
| Gradient step | O(batch·network) | Per episode/batch |
| Space | O(parameters) | Policy weights |
| Total training | O(episodes·T·network) | Linear-ish in experience |
| Convergence speed | Slow (variance-limited) | Often needs 10⁵–10⁷ steps |

---

## 43. Advanced Concepts

- **Policy Gradient Theorem:** the general form ∇J ∝ E[Q^π(s,a)∇log π] — REINFORCE samples Q^π via G_t; the theorem is the backbone of actor-critic.
- **Baselines and advantage:** value-function baseline turns G_t into the advantage A_t = G_t − V(s_t) (file 07).
- **Entropy regularization:** +βH(π) trades exploration for expected return.
- **Natural policy gradient (TRPO/PPO):** constrain parameter moves by KL (Part 2).
- **Gaussian policies:** reparameterization a = μ + σ·ε eases gradients through continuous actions.
- **Return normalization / variance reduction:** whitening G across a batch.

---

## 44. Connections to Other Algorithms

```text
REINFORCE (Policy Gradient)
   │
   ├── variance fix → Actor-Critic (file 07): critic V(s) as baseline
   ├── value complement → DQN (file 05) (value-based, discrete)
   ├── sampling base → Monte Carlo (file 03)
   ├── estimator kin → TD learning (file 04) (critic side)
   └── modern family → TRPO/PPO/A2C/A3C (Part 2)
```

---

## 45. If You Remember Only 5 Things

1. REINFORCE directly maximizes expected return via θ ← θ + α·G_t·∇log π_θ(a_t|s_t).
2. ∇J(θ) = E[Σ G_t ∇log π(a_t|s_t)] — derived via the log-derivative trick.
3. It is a Monte-Carlo, on-policy method: unbiased but very high variance.
4. A baseline b(s) (e.g., V(s)) reduces variance without bias → advantage → actor-critic.
5. Use it when actions are continuous or you need an explicit stochastic policy.

---

## 46. Cheat Sheet

```text
Algorithm   : REINFORCE (MC Policy Gradient)
Category    : RL, policy-based, model-free, on-policy
Goal        : Maximize expected return J(θ)
Input       : full episodes (s,a,r)
Output      : policy π_θ(a|s) (discrete or continuous)
Core Formula: θ ← θ + α G_t ∇log π_θ(a_t|s_t)
Loss        : surrogate −Σ G_t log π(a_t|s_t)
Optimization: gradient ascent (ADAM/SGD)
Parameters  : policy weights θ
Hyperparams : γ, α, batch size, entropy β, architecture
Assumptions : differentiable policy, episodic, on-policy samples
Advantages  : continuous actions, stochastic policy, unbiased
Disadvantages: high variance, slow, sample-inefficient
Use When    : continuous/stochastic-policy tasks, cheap episodes
Avoid When  : scarce interactions, long-horizon sparse reward
Related     : Actor-Critic, PPO/TRPO, A2C (Part 2), DQN
Key Exam    : log-derivative trick, REINFORCE step, on-policy
Key Interv  : variance, baseline, entropy, continuous actions
```

---

## 47. Final Mental Model

```text
Sample episode under π_θ
   ↓
Compute returns G_t (backward)
   ↓
Accumulate Σ G_t ∇log π_θ(a_t|s_t)
   ↓
θ ← θ + α·accum       (up-weight good actions)
   ↓
Policy probabilities shift toward high-return actions
   ↓
Repeat → local optimum of expected return
```

---

## 48. Knowledge Check

### Recall (5)
1. Write the REINFORCE update.
2. Define J(θ).
3. What is the score function?
4. Why is REINFORCE Monte-Carlo?
5. What is the log-derivative trick?

### Understanding (5)
6. Why is the estimate unbiased?
7. Why is variance so high?
8. Why is REINFORCE on-policy?
9. How does a baseline reduce variance without bias?
10. What does "entropy collapse" mean?

### Application (5)
11. Compute one update for a softmax/Bernoulli policy.
12. Choose a head for continuous actions.
13. Set batch size to cut variance.
14. Add an entropy term.
15. Build a deterministic eval loop.

### Mathematical (5)
16. Derive ∇J(θ) = E[Σ G_t ∇log π(a_t|s_t)].
17. Show environment dynamics cancel in the gradient.
18. Write the score for a Gaussian policy.
19. Prove baseline unbiasedness (E[b∇log π]=0).
20. State the Policy Gradient Theorem.

### Interview (5)
21. REINFORCE vs Q-learning — when each?
22. How would you make REINFORCE less noisy?
23. What's the advantage expression from a value baseline?
24. Why might the policy collapse deterministically?
25. How does actor-critic fix REINFORCE's problems?

### Problem Solving (5)
26. Returns vary enormously episode to episode — what to do?
27. Policy becomes deterministic too early — diagnosis + fix?
28. Training curves flat for 1k episodes — hypothesis?
29. Discrete env, DQN worked, but you want a stochastic policy — options?
30. Continuous control, REINFORCE too slow — recommend next step?

## Answers (explained)

1. θ ← θ + α·G_t·∇log π_θ(a_t|s_t).
2. J(θ) = E_τ~π_θ[G_0], the expected discounted return.
3. ∇log π_θ(a|s) — parameter direction increasing that action's probability.
4. Its gradient uses full episode returns G_t from sampled trajectories.
5. ∇π_θ(τ) = π_θ(τ)·∇log π_θ(τ), turning the intractable gradient into an expectation.
6. The expectation of the sampled sum equals the true ∇J by the identity.
7. Each return aggregates the whole trajectory's reward noise into one gradient term.
8. Its episodes must be generated by the current θ (the gradient is for that policy).
9. E[b(s)∇log π] = Σ_a b(s)∇π = b(s)·∇(Σπ)=0 → zero-mean subtraction, variance falls.
10. All probability mass concentrates on one action early, halting exploration.
11. Follow Section 15: p=σ(0)=0.5, G=2, score=0.5, θ→0.1, p≈0.525.
12. Gaussian head: output μ(s), σ(s); sample a~N(μ,σ²); score ∝ (a−μ)/σ².
13. Average multiple episodes per update — gradient noise ~ 1/√N.
14. Append −β·Σπ log π to the surrogate loss; tune β to balance.
15. Take distribution mean/argmax, run fresh episodes, average returns.
16. Steps 1–6 of Section 10: objective → log-derivative → expansion → causality → MC sample.
17. p(s_{t+1}|s_t,a_t) appears in log π_θ(τ) without θ, so its gradient is 0.
18. ∂log N(a|μ,σ²)/∂θ = ((a−μ)/σ²)·∂μ/∂θ + ((a−μ)²/σ² − 1)/σ·∂σ/∂θ.
19. E[b(s)∇log π] = b(s)∇Σ_a π_θ(a|s) = b(s)·∇1 = 0.
20. ∇J(θ) ∝ E_π[Q^π(s,a)∇log π(a|s)].
21. Q-learning for small discrete/cheap tasks; REINFORCE for continuous or stochastic needs.
22. Larger batches, baselines, entropy bonuses, return normalization.
23. A_t = G_t − V(s_t) — the advantage.
24. Variance + greedy sampling can concentrate mass early; entropy regularization prevents it.
25. The critic (V) as a baseline removes the return's baseline drift → lower variance with unbiased advantage.
26. Increase batch size, standardize returns, add a baseline.
27. Entropy collapse: add/re-tune entropy bonus; raise action noise.
28. Learning rate too low, variance too high (batch), or reward scaling off.
29. Use REINFORCE/Actor-Critic instead of the argmax policy from Q.
30. Move to Actor-Critic (file 07), then PPO/SAC (Part 2), or use baseline + batch.

---

## 49. Final Learning Checklist

- [ ] I can define REINFORCE in one sentence
- [ ] I can write the update θ ← θ + α·G·∇log π
- [ ] I understand J(θ) = E[G_0]
- [ ] I can derive the log-derivative trick
- [ ] I know why dynamics cancel in the gradient
- [ ] I understand MC / full-return property
- [ ] I know why it's unbiased
- [ ] I know why variance is high
- [ ] I can compute a softmax/Bernoulli score
- [ ] I can compute a Gaussian score
- [ ] I understand the on-policy requirement
- [ ] I can implement REINFORCE from scratch
- [ ] I can implement in PyTorch on CartPole
- [ ] I know how a baseline works
- [ ] I understand advantage = G − V(s)
- [ ] I know entropy regularization
- [ ] I can evaluate properly (not by surrogate loss)
- [ ] I can diagnose entropy collapse
- [ ] I can connect REINFORCE to actor-critic
- [ ] I can explain REINFORCE to a beginner

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** log-derivative derivation, causality step, baseline unbiasedness proof all verified; Bernoulli-softmax numeric example hand-computed (θ→0.1, p→0.525) and the negative-reward branch checked.
- **Beginner-friendliness:** juggler analogy, step-by-step reasoning, short paragraphs.
- **Math depth:** full derivation, Policy Gradient Theorem reference, Gaussian score.
- **Practical depth:** from-scratch numpy agent + PyTorch CartPole, hyperparameter table, failure-mode fingerprinting practice.
- **Exam depth:** log-derivative trick and on-policy traps, representative (non-PYQ) pattern question.
- **Structure:** all 50 sections present in order.

**Verified:** numerically recomputed updates (both positive and negative return branches) consistent with the sigmoid-gradient rule.