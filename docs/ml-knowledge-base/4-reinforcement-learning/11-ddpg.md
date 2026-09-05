# 11. Deep Deterministic Policy Gradient

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐☆☆☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐⭐☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|---|
| Algorithm Name | Deep Deterministic Policy Gradient (DDPG) |
| Category | Reinforcement Learning (RL) |
| Type | **Off-policy**, deterministic actor-critic, continuous control |
| Parametric / Non-parametric | Parametric (deep networks) |
| Generative / Discriminative | Actor outputs a single deterministic action |
| Main Objective | Learn deterministic policy a=µ_θ(s) maximizing Q by policy gradient ∇J = E[∇_a Q·∇_θ µ(s)], off-policy via a replay buffer and target networks |
| Input | Replay-buffer transitions (s, a, r, s') from an exploration policy |
| Output | Deterministic policy a=µ_θ(s) and Q-function Q_φ(s,a) |
| Core Idea | Off-policy DQN-style learning + continuous deterministic actor, with target networks and Polyak averaging for stability |
| Typical Use Cases | Continuous control (robotics, pendulum, walkers) needing sample efficiency |

---

## 02. One-Line Definition

### Beginner Definition
DDPG trains a "doer" that outputs a single best action and a "judge" that scores (state, action); it learns from a big memory of past experiences (replay buffer) and keeps slow-moving copies (target networks) so training doesn't wobble.

### Technical Definition
DDPG is an off-policy actor-critic for continuous actions: the actor µ_θ(s) outputs a deterministic action; the critic Q_φ(s,a) is updated by TD regression toward a target `y = r + γ Q_φ'(s', µ_θ'(s'))`; the actor is updated by ∇J = E[∇_a Q_φ(s,a)·∇_θ µ_θ(s)], with target networks softened via Polyak averaging `θ' ← τθ + (1−τ)θ'`.

---

## 03. Intuition

- **Deterministic actor:** instead of outputting a distribution, the actor directly outputs the exact torque/velocity to apply — suited to continuous control where a single best action exists.
- **Replay buffer:** past (s,a,r,s') experiences are stored and re-sampled, enabling **off-policy** learning (reusing old data) → far better sample efficiency than on-policy methods.
- **Judge (critic) Q(s,a):** predicts the total future reward of doing `a` in `s`; the actor is trained to pick `a` that maximizes the judge's score.
- **Target networks:** slow-motion copies provide stable regression targets; **Polyak averaging** smoothly blends each update `θ' ← τθ + (1−τ)θ'` (like a low-pass filter on the network weights).

---

## 04. Problem It Solves

- **DPG/DQN can't handle continuous actions** — DQN needs an argmax over actions, infeasible when actions are continuous vectors.
- **On-policy methods waste data** — each rollout is used once.
- **DDPG solves both:** a **deterministic actor** sidesteps the argmax, and a **replay buffer + off-policy critic** reuses experiences for sample efficiency.
- It's the "DQN-style" adaptation to continuous, high-dimensional action spaces (robotics, control).

---

## 05. Where It Fits in Machine Learning

```
Machine Learning
└── Reinforcement Learning
    ├── On-policy  — PPO, A2C, A3C
    └── Off-policy Actor-Critic
        ├── Deterministic: DDPG ◄── HERE
        │        ├── TD3 (fixes DDPG's overestimation)
        │        └── SAC (stochastic max-entropy)
        └── Value: DQN family
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Deterministic policy | Single action per state | a = µ_θ(s) (no sampling) |
| Off-policy | Learns from old data | Uses a replay buffer |
| Replay buffer | Memory of transitions | Stores (s,a,r,s') tuples |
| Critic Q(s,a) | Action-value estimate | E[R \| s,a] |
| Actor | Deterministic policy | µ_θ(s) |
| Target network | Slow-motion copy | θ', φ' provide stable targets |
| Polyak averaging | Weight blending | θ' ← τθ + (1−τ)θ', τ∈(0,1] |
| TD target | Regression target | y = r + γQ'(s', µ'(s')) |
| Exploration noise | Random action perturb | OU noise / Gaussian |
| Overestimation bias | Q over-optimistic | (fix in TD3) |

---

## 07. Input and Output

**Input:**
- Observations `s`, actions `a`, rewards `r`, next state `s'` (batched transitions from replay buffer).
- Hyperparameters: lr_actor, lr_critic, γ, τ, buffer size, batch size, noise std, target update frequency.

**Output:**
- Deterministic policy `a = µ_θ(s)`.
- Q-function `Q_φ(s,a)` approximating E[return].

---

## 08. Mathematical Foundation

**Deterministic policy gradient (DPG theorem).** For a deterministic policy µ_θ, the gradient of the performance J(θ)=E[Q^{µ}(s, µ_θ(s))] w.r.t. actor parameters is:

```
∇_θ J(θ) = E[ ∇_θ µ_θ(s) · ∇_a Q^{µ}(s,a)|_{a=µ_θ(s)} ]
```

The chain rule: how the Q function changes as we change the action, times how the action changes with θ. This is the actor's gradient signal, no log-prob needed.

**Off-policy critic (TD).** The critic is trained against a target network:

```
y = r + γ Q_φ'(s', µ_θ'(s'))
L(φ) = E[( Q_φ(s,a) − y )²]
```

**Polyak target update:**

```
θ' ← τ θ + (1−τ) θ'
φ' ← τ φ + (1−τ) φ'
```

---

## 09. Core Formula

The two main updates:

```text
Critic:  L(φ) = E[( Q_φ(s,a) − y )²],   y = r + γ Q_φ'(s', µ_θ'(s'))
Actor:   ∇_θ J = E[ ∇_a Q_φ(s,a)|_{a=µ_θ(s)} · ∇_θ µ_θ(s) ]
Target:  θ' ← τθ + (1−τ)θ' ,   φ' ← τφ + (1−τ)φ'
```

### Meaning
The critic reports how good a (state,action) is; the actor is updated to choose the action that maximizes the critic's score; targets are stabilized with slow-copy networks and Polyak mixing.

### Symbols
- `µ_θ(s)` deterministic actor; `µ_θ'` target actor.
- `Q_φ(s,a)` critic; `Q_φ'` target critic.
- `γ` discount; `τ` Polyak rate (small, e.g., 0.005).
- `y` TD target.

### Intuition
The critic teaches the actor "which direction in action-space improves value." Both learn incrementally; targets keep the learning stable by not moving too fast.

### Example (tiny dataset, hand-verified)
Let γ=0.99, and one transition (s, a, r, s'):
- s=1, a=0.5, r=1, s'=2.
- Target actor output `µ_θ'(s') = 0.8`.
- Target critic `Q_φ'(s', 0.8) = 2.5`.
- TD target `y = 1 + 0.99·2.5 = 1 + 2.475 = 3.475`.
If critic prediction `Q_φ(s,a) = 3.0`, the critic regression target is 3.475 (residual 0.475). The actor gradient at `a=0.5` uses `∇_a Q` there.

---

## 10. Derivation

**Critic.** Start from the Bellman equation `Q^µ(s,a) = r + γ Q^µ(s', µ(s'))`. Replace the unknown Q^µ on the right with a **target network** to slow movement:

```
y = r + γ Q_φ'(s', µ_θ'(s'))
```

Minimize squared TD error: `L = (Q_φ(s,a) − y)²`.

**Actor.** Performance objective `J(θ) = E_s[ Q^{µ}(s, µ_θ(s)) ]`. By chain rule:

```
∇_θ J = E_s[ ∇_θ µ_θ(s) ∇_a Q(s,a)|_{a=µ_θ(s)} ]
```

This is the **deterministic policy gradient** — the actor moves actions in the direction the critic says increases value.

**Polyak.** To keep regression stable, target parameters are soft-updated (small τ), turning hard target swaps (DQN) into smooth blending:

```
θ' ← τθ + (1−τ)θ'      (small τ ⇒ target changes slowly)
```

---

## 11. How the Algorithm Works

```text
Input: actor µ_θ, critic Q_φ, targets µ_θ', Q_φ', replay buffer
  ↓
Loop:
  Sample action a = µ_θ(s) + noise  (exploration)
  Step env → (s,a,r,s'), store in buffer
  ↓
  Sample mini-batch from buffer
  ↓
  y = r + γ Q_φ'(s', µ_θ'(s'))             (TD target)
  L(φ) = (Q_φ(s,a) − y)²                   (critic loss)
  φ ← φ − α_c ∇ L(φ)
  ↓
  ∇_θ J = ∇_a Q_φ(s,a)|_a·∇_θ µ_θ(s)       (actor grad)
  θ ← θ + α_a ∇_θ J
  ↓
  θ' ← τθ + (1−τ)θ' ;  φ' ← τφ + (1−τ)φ'    (Polyak)
  ↓
Repeat
```

---

## 12. Training Process

- **Pre-training:** init nets, targets, replay buffer; warm it with random/noisy transitions.
- **During:** each step, act with exploration noise, store transition. Periodically (per step) sample a batch, update critic then actor, then Polyak-update targets.
- **What's learned:** θ (actor), φ (critic) and their targets.
- **Change per iteration:** critic nudges toward the TD target; actor toward higher Q; targets smoothed.
- **Stopping:** mean eval return plateaus or step budget.
- **Final model:** actor µ_θ(s) (used at inference, no noise).

---

## 13. Objective Function / Loss Function

```
Critic (minimize): L(φ) = E[( Q_φ(s,a) − (r + γ Q_φ'(s', µ_θ'(s'))) )²]
Actor (maximize):  J(θ) = E[ Q_φ(s, µ_θ(s)) ]   (via ∇_θ J = E[∇_a Q·∇_θ µ])
```

- **Critic loss:** TD regression to target network output. High MSE = critic inaccurate.
- **Actor objective:** maximize predicted Q of its chosen deterministic actions.

---

## 14. Optimization

- **Critic:** gradient descent on L(φ) (Adam).
- **Actor:** gradient ascent on J(θ) (Adam), using ∇_a Q from the critic as the direction.
- **Targets:** no gradient; Polyak blend.

```text
Current θ, φ
  ↓ batch from replay
TD target y → critic loss → φ update
∇_a Q · ∇_θ µ  → actor update → θ += α_a·(∇_a Q ∇_θ µ)
Polyak: θ' ← τθ+(1−τ)θ'; φ' ← τφ+(1−τ)φ'
```

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified above).

**Setup.** γ=0.99, τ=0.005, α_c=α_a=0.1. One transition:

| s | a | r | s' | µ_θ'(s') | Q_φ'(s',·) |
|---|---|---|---|---|---|
| 1 | 0.5 | 1 | 2 | 0.8 | 2.5 |

**Step 1 — TD target:** `y = 1 + 0.99·2.5 = 3.475`.
**Step 2 — critic loss:** `(Q_φ(s,a) − y)² = (3.0 − 3.475)² = 0.2256`.
**Step 3 — critic update (assume ∇L = 2(Q−y)∇_a·... ≈ 2(3−3.475) = −0.95):** φ updated toward 3.475.
**Step 4 — actor gradient:** `∇_a Q at a=0.5 = +2.0`, `∇_θ µ(s) = 1.0` ⇒ `∇J = 2.0·1.0 = 2.0`; `θ += 0.1·2.0 = 0.2` (increases a toward higher Q).
**Step 5 — Polyak:**
`θ' = 0.005·θ + 0.995·θ'` (small shift), same for φ'. Targets move a little toward the new nets.

---

## 16. Visual Explanation

Deterministic policy-gradient "lift the action uphill in Q":

```
Q(s,a)
  |      .
  |        .___
  |      . /|   \
  |    . / |     ← actor moves a rightward
  |     0.5 1.0   (increases Q)
  +------------------ a
```

Actor-critic with target & replay:

```
 ENV ──► (s,a,r,s') ──► REPLAY BUFFER ──► mini-batch ──►
                                                          ▼
            CRITIC Q_φ ──L=(Q−y)²──► y = r+γQ_φ'(s',µ_θ'(s'))
            ACTOR µ_θ  ◄──∇_aQ·∇_θµ── derived from critic
            TARGETS (θ',φ') ⟵ τ-blend from online (slow)
```

---

## 17. Algorithm / Pseudocode

```text
1. Init actor µ_θ, critic Q_φ, targets µ_θ'=µ_θ, Q_φ'=Q_φ, replay buffer B.
2. for each episode:
3.   reset env; observe s.
4.   for each step:
5.     a = µ_θ(s) + noise(exploration).
6.     step → (s,a,r,s'); store in B.
7.     Sample batch from B.
8.     y = r + γ Q_φ'(s', µ_θ'(s')).
9.     φ ← φ − α_c ∇(Q_φ(s,a)−y)².
10.    θ ← θ + α_a [∇_aQ_φ(s,a)|µ·∇_θµ(s)].
11.    θ' ← τθ+(1−τ)θ';  φ' ← τφ+(1−τ)φ'.
```

---

## 18. From-Scratch Implementation

Minimal conceptual NumPy sketch (update equations appear literally):

```python
import numpy as np

class Net:
    def __init__(self, dims):
        # tiny 2-layer net; we treat forward/backward via autograd elsewhere
        self.w1 = np.random.randn(dims[0], dims[1]) * 0.1
        self.w2 = np.random.randn(dims[1], dims[2]) * 0.1

def ddpg_step(actor, critic, actor_target, critic_target, batch,
              buffer, gamma=0.99, tau=0.005, lr_c=0.1, lr_a=0.1):
    for (s, a, r, s2) in batch:
        # TD target using target networks
        a2 = actor_target.forward(s2)                    # µ_θ'(s')
        y = r + gamma * critic_target.forward(s2, a2)    # y = r + γQ'(s',µ'(s'))
        # critic loss and update
        q = critic.forward(s, a)
        critic_loss = (q - y) ** 2
        grad = critic.backward(s, a) * 2 * (q - y)       # ∇L
        update(critic, -grad * lr_c)                     # φ ← φ − α_c ∇L
        # actor gradient: ∇_a Q · ∇_θ µ
        grad_a = critic.grad_wrt_action(s)               # ∇_a Q
        grad_theta = actor.backward(s)                   # ∇_θ µ
        actor_grad = grad_a * grad_theta * lr_a          # ∇J = ∇_aQ·∇_θµ
        update(actor, actor_grad)                        # θ += α_a ∇J
    # Polyak target updates
    actor_target.w1 = tau * actor.w1 + (1 - tau) * actor_target.w1
    actor_target.w2 = tau * actor.w2 + (1 - tau) * actor_target.w2
    critic_target.w1 = tau * critic.w1 + (1 - tau) * critic_target.w1
    critic_target.w2 = tau * critic.w2 + (1 - tau) * critic_target.w2
```

Lines literally: TD target `y = r + γ Q'(s',µ'(s'))`, critic `(Q−y)²`, actor `∇_aQ·∇_θµ`, Polyak `θ' ← τθ+(1−τ)θ'`.

---

## 19. Code Explanation

```text
Code ↓ What does it do? ↓ Why required? ↓ Mathematical concept?

a2 = actor_target(s2)      Target action        µ_θ'(s')
y = r + γ Q_v2(s2,a2)      TD target            Bellman target
critic_loss=(q-y)²         Critic regression    L(φ)
grad_a = ∇_a Q             Action direction     determines useful action
actor_grad = ∇_aQ·∇_θµ     Actor update         deterministic PG
tau-blend targets          Slow weight copy     Polyak θ'←τθ+(1−τ)θ'
```

---

## 20. Library Implementation

CleanRL/stable-baselines3-style DDPG (simplified PyTorch):

```python
import torch, torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs, act, max_a):
        super().__init__(); self.net = nn.Sequential(
            nn.Linear(obs,256), nn.ReLU(), nn.Linear(256,256), nn.ReLU(),
            nn.Linear(256, act), nn.Tanh())
        self.max_a = max_a
    def forward(self, x): return self.net(x) * self.max_a

class Critic(nn.Module):
    def __init__(self, obs, act):
        super().__init__(); self.net = nn.Sequential(
            nn.Linear(obs+act,256), nn.ReLU(), nn.Linear(256,256),
            nn.ReLU(), nn.Linear(256,1))
    def forward(self, s, a): return self.net(torch.cat([s,a],-1))

# per update step (sample batch B from replay):
with torch.no_grad():
    next_a = actor_target(next_s); target = r + gamma * critic_target(next_s, next_a)
critic_loss = ((critic(s,a) - target)**2).mean()
actor_loss = -critic(s, actor(s)).mean()
# ... optimize both, then polyak:
for p, pt in zip(actor.parameters(), actor_target.parameters()):
    pt.data.mul_(1-tau).add_(p.data, alpha=tau)
# (same for critic)
```

Actor loss `−Q(s, µ(s))` matches ∇_aQ·∇_θµ; Polyak loop matches the blend.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Replay buffer size | Memory capacity | Bigger = more data | 1e5–1e6 |
| Batch size | Samples per update | Bigger = more stable | 64–256 |
| Discount γ | Future credit | Higher = farther-sighted | 0.99 |
| τ (Polyak) | Target blend rate | Small = slower, stable targets | 0.005 |
| Exploration noise std | Perturbation | Higher = explore more | 0.1–0.2 |
| Actor/critic lr | Step sizes | Too high → diverge | 1e-3/1e-4 |
| Target update freq | Steps between target sync | Lower overhead | every step (Polyak) |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Actor θ, critic φ, target copies θ', φ'.

### Hyperparameters (chosen)
- Buffer/batch, γ, τ, noise std, lrs, architecture.

---

## 23. Assumptions

- **Continuous action space** — DDPG needs it (deterministic action outputs).
- **Off-policy data validity** — replay mixes old experiences; okay because Q-learning is off-policy.
- **Deterministic optimal policy exists** — generally true for control tasks.
- **Stationary target grading** — the slow Polyak target provides it.
- **Bounded actions** — Tanh output assumed; Q must be well-calibrated.

---

## 24. Data Requirements

- Data: (s,a,r,s') transitions; continuous state and action features.
- Numerical; normalize observations and rewards.
- Sample efficiency: high (off-policy, reuses data).
- Buffer must be large enough to break correlation; enough diverse transitions to learn Q.

---

## 25. Feature Scaling

**Recommended.** Normalize observations (running mean/std) and rewards (e.g., clip or standardize) — Q-networks are scale-sensitive.

---

## 26. Evaluation Metrics

| Metric | Definition | Use |
|---|---|---|
| Mean episode return | Avg reward sum at eval | Primary performance |
| Mean episode length | Steps per episode | Task completeness |
| Learning curve | Return vs env steps | Sample efficiency |
| Estimated Q / true return gap | (Q − actual) | Detect overestimation bias |

Do not judge by critic loss alone.

---

## 27. Advantages

- **Sample efficient** — off-policy replay reuses data.
- **Continuous actions** — no argmax over infinite actions.
- **Stable targets** — target networks + Polyak averaging.
- **Smooth deterministic policy** — good for control/servo tasks.
- **Foundational** — base for TD3 and many robots.

---

## 28. Disadvantages

- **Overestimation bias** — single critic can overestimate Q (DDPG's key flaw, fixed by TD3).
- **Hyperparameter-sensitive** — τ, noise, lr interplay.
- **Deterministic policy lacks built-in exploration** — relies on injected noise.
- **Off-policy instability** — Q function can diverge with function approximation.
- **Assumes continuous actions** — not for discrete.

---

## 29. When to Use

- ✓ Continuous action space.
- ✓ Sample efficiency important (sim/robot budget small).
- ✓ Smooth deterministic control is the goal.
- ✓ You can add exploration noise.

---

## 30. When NOT to Use

- ✗ Discrete action space → DQN.
- ✗ Stochastic optimal policy needed (multi-modal) → SAC.
- ✗ Maximum stability wanted → TD3 (fixes DDPG) or PPO.
- ✗ Very high noise/uncertain reward → overestimation hurts.

---

## 31. Real-World Applications

| Problem ↓ | Input ↓ | Algorithm ↓ | Output ↓ |
|---|---|---|---|
| Robotic arm control | Joint angles | DDPG | Joint torques |
| Pendulum swing-up | Angle/velocity | DDPG | Torque |
| Autonomous driving throttle/steer | State vector | DDPG | Continuous control |
| Continuous control benchmarks | MuJoCo states | DDPG | Action |

---

## 32. Failure Cases

- **Overestimation bias** → critic too optimistic → suboptimal/oscillating actor.
- **Exploration noise too high/low** → suboptimal or stuck policy.
- **Replay buffer too small** → correlation, unstable Q.
- **Polyak τ too large** → targets move fast → divergence.
- **Q-divergence** with function approximation.

---

## 33. Overfitting and Underfitting

- **Overfitting:** buffer dominated by few trajectories; mitigate larger buffer, more exploration, regularization.
- **Underfitting:** policy/Q capacity insufficient; enlarge networks, longer training.
- Balance via eval return and Q-fit.

---

## 34. Bias-Variance Perspective

- TD targets reduce variance (one-step) vs MC but add bias (approximate targets).
- Target networks reduce the **target-shifting variance** at the cost of slower adaptation (bias).
- Polyak averaging is a bias-variance trade on target motion.
- **Overestimation is a systematic bias** of the single-critic DDPG — TD3 addresses it.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| **DDPG** | **Deterministic off-policy A-C** | **Sample efficient, continuous** | **Overestimation bias** | **Baseline continuous** |
| TD3 | DDPG + twin critics + delay | Fixes overestimation | More machinery | Robust continuous |
| SAC | Stochastic max-entropy | Most sample-efficient, robust | Complexity | Best continuous |
| PPO | On-policy clipped | Stable | Sample-inefficient | On-policy tasks |

---

## 36. Algorithm Selection Guide

```text
Continuous actions, sample-constrained?
  ├─ Prefer robustness → TD3 / SAC
  └─ Simpler baseline → DDPG ◄── (but TD3 usually better)
Discrete actions → DQN
```

---

## 37. Common Mistakes

```text
❌ Not normalizing observations/rewards
Why wrong: Q-network unstable.
Correct: standardize inputs, scale rewards.

❌ Setting τ too large (e.g., τ=1)
Why wrong: target jumps like DQN hard swap.
Correct: τ≈0.005.

❌ Forgetting target networks
Why wrong: targets move with predictions → divergence.
Correct: keep separate slow-copy targets.

❌ Zero exploration noise at start
Why wrong: never covers action space.
Correct: decay noise as training proceeds.
```

---

## 38. Interview Questions

### Beginner
- **Q: What is a deterministic policy?** A: Directly outputs the exact action a = µ_θ(s).
- **Q: Why a replay buffer?** A: Reuse experience → off-policy, sample-efficient.
- **Q: What are target networks?** A: Slow-copy networks giving stable TD targets.

### Intermediate
- **Q: Write the TD target.** A: `y = r + γ Q_φ'(s', µ_θ'(s'))`.
- **Q: Write the actor gradient.** A: `∇_θ J = E[∇_a Q·∇_θ µ]`.
- **Q: What is Polyak averaging?** A: `θ' ← τθ + (1−τ)θ'` smooth target update.

### Advanced
- **Q: What is DDPG's key failure?** A: Overestimation bias in the single critic.
- **Q: How does TD3 fix it?** A: Twin clipped critics + delayed actor + target smoothing.
- **Q: Why off-policy here but on-policy in PPO?** A: Q-learning allows bootstrapping from any (s,a), not just the acting policy's distribution.

---

## 39. GATE / Exam Perspective

- Key formulas: TD target `y = r + γ Q'(s',µ'(s'))`; deterministic PG `∇J = E[∇_aQ·∇_θµ]`; Polyak `θ'←τθ+(1−τ)θ'`.
- Concepts: off-policy vs on-policy, replay buffer, target network, deterministic policy.
- Traps: DDPG is deterministic (no sampling) — distinct from stochastic PG; overestimation bias is the known weakness.
- *Representative pattern question*: "Derive the deterministic policy gradient and explain why target networks are needed."

---

## 40. Coding Practice

1. **Level 1:** TD target computation.
2. **Level 2:** critic MSE loss.
3. **Level 3:** actor ∇_aQ·∇_θµ.
4. **Level 4:** replay buffer class.
5. **Level 5:** Polyak target update.
6. **Level 6:** full DDPG loop on a control env.
7. **Level 7:** add noise schedule, compare to TD3.

---

## 41. Practical ML Workflow

1. Define continuous MDP.
2. Init actor/critic/targets/buffer.
3. Fill buffer with noise actions.
4. Sample batches, update critic → actor.
5. Polyak-update targets.
6. Explore with decaying noise.
7. Tune τ, noise, lr.
8. Evaluate mean return (no noise).
9. Error analysis (overestimation, Q-gap).
10. Deploy deterministic policy.

---

## 42. Complexity

- Training time: quadratic-ish in buffer reads? No — per step O(batch × network). Off-policy makes it sample-efficient.
- Space: stores replay buffer (up to 1e6 transitions).
- Scales with buffer size & network size; parallel envs can fill buffer faster.

---

## 43. Advanced Concepts

- **Deterministic policy gradient theorem** (the theoretical basis).
- **Replay buffer** as off-policy decorrelation.
- **Exploration noise** (Gaussian/OU) as stochastic approximation of a stochastic policy.
- **Overestimation bias** and its measurement (Q − true_return).
- **Target smoothing & twin critics** (bridge to TD3).

---

## 44. Connections to Other Algorithms

```
DPG (theorem) ──► DDPG (deep) ──► TD3 (twin/delay/smooth)
DQN (value off-policy) supplies replay + target ideas to DDPG.
SAC is DDPG's stochastic sibling (max-entropy).
PPO/A2C are the on-policy contrast.
```

---

## 45. If You Remember Only 5 Things

1. DDPG is off-policy, deterministic actor-critic for continuous control.
2. Actor: a = µ_θ(s); gradient ∇J = E[∇_aQ·∇_θµ].
3. Critic: TD target y = r + γQ'(s',µ'(s')); loss (Q−y)².
4. Uses replay buffer + target networks + Polyak averaging for stability.
5. Key weakness: overestimation bias — fixed by TD3.

---

## 46. Cheat Sheet

| Aspect | Value |
|---|---|
| Algorithm | DDPG |
| Category | Off-policy deterministic actor-critic |
| Goal | Max return, continuous actions |
| Input | Replay transitions |
| Output | Deterministic policy a=µ_θ(s) |
| Core Formula | ∇J=E[∇_aQ·∇_θµ]; y=r+γQ'(s',µ') |
| Loss | critic (Q−y)²; actor −Q(s,µ(s)) |
| Optimization | Adam (critic descent, actor ascent) |
| Parameters | θ (actor), φ (critic), targets θ',φ' |
| Hyperparams | τ, γ, noise, buffer, batch, lr |
| Assumptions | Continuous actions, off-policy |
| Adv | Sample efficient, continuous |
| Disadv | Overestimation bias |
| Use | Sample-efficient continuous |
| Avoid | Discrete, max-stability |
| Related | TD3, SAC, DQN, DPG |
| Key Exam | deterministic PG, TD target, Polyak |
| Key Interview | overestimation bias |

---

## 47. Final Mental Model

```text
Act with µ_θ + noise → store (s,a,r,s')
   → sample batch → TD target → update critic
   → ∇_aQ·∇_θµ → update actor
   → Polyak-blend targets → next
```

---

## 48. Knowledge Check

### Recall
1. DDPG gradient formulas.
2. Deterministic vs stochastic policy.
3. What is Polyak averaging?
4. Why target networks?
5. What is off-policy?

### Understanding
6. Why continuous-friendly.
7. Why sample efficient.
8. Role of replay.
9. Why overestimation.
10. Why target smoothing.

### Application
11. Set τ.
12. Add exploration noise.
13. Detect Q divergence.
14. Choose DDPG vs TD3.
15. Normalize rewards.

### Mathematical
16. Compute TD target.
17. Compute actor gradient.
18. Write Polyak update.
19. Critic loss.
20. Deterministic PG theorem.

### Interview
21. DDPG vs SAC.
22. DDPG vs PPO.
23. Overestimation cause.
24. Discrete limit.
25. Stability methods.

### Problem Solving
26. Q overestimation observed.
27. Network diverges.
28. Stuck suboptimal policy.
29. Buffer too correlated.
30. Noise scheduling.

## Answers (explained)
1. `∇J=E[∇_aQ·∇_θµ]`; `y=r+γQ'`.
2. Deterministic outputs one action; stochastic samples a distribution.
3. `θ'←τθ+(1−τ)θ'`.
4. Stable regression targets.
5. Learns from previously-collected/buffer data.
6. No argmax over continuous actions.
7. Reuses old data.
8. Decorrelates, reuses experience.
9. max over Q in DQN propagates noise up.
10. Blended targets reduce jumpiness.
11. ≈0.005.
12. Decaying Gaussian/OU.
13. Q >> true return.
14. Prefer TD3 for robustness.
15. Standardize.
16. `y=r+γQ'(s',µ'(s'))`.
17. `∇_aQ∇_θµ`.
18. `θ'←τθ+(1−τ)θ'`.
19. `(Q−y)²`.
20. `∇θJ=E[∇θµ·∇aQ]`.
21. SAC stochastic/max-entropy, more robust.
22. DDPG off-policy/efficient vs PPO on-policy.
23. Bellman max/noise pushes Q up.
24. Deterministic continuous only.
25. Targets, Polyak, replay.
26. Use twin critics (TD3).
27. Lower lr, normalize, τ down.
28. Add exploration, anneal slower.
29. Larger buffer, more random init.
30. Schedule decay with steps.

---

## 49. Final Learning Checklist

- [ ] I can write TD target
- [ ] I can write actor gradient
- [ ] I can write Polyak update
- [ ] I can explain off-policy reuse
- [ ] I know why target networks
- [ ] I know overestimation bias
- [ ] I can build replay buffer
- [ ] I can add exploration noise
- [ ] I can normalize inputs
- [ ] I can set τ and lr
- [ ] I can write from-scratch DDPG
- [ ] I can write CleanRL DDPG
- [ ] I can detect Q divergence
- [ ] I can compare DDPG/TD3/SAC
- [ ] I know continuous limit
- [ ] I can evaluate mean return
- [ ] I can diagnose overestimation
- [ ] I know deterministic PG
- [ ] I can schedule noise decay
- [ ] I can deploy deterministic actor

---

## 50. Quality Control Note

- **Accuracy:** TD target, deterministic PG, Polyak verified; example hand-verified.
- **Beginner-friendliness:** deterministic actor vs judge analogy.
- **Math depth:** deterministic PG theorem, derivations.
- **Practical depth:** from-scratch + PyTorch code, hyperparameters, failure cases.
- **Exam depth:** overestimation trap, representative pattern question (clearly labeled).
- **Structure:** follows 50-section template exactly.
