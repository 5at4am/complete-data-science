# 08. Proximal Policy Optimization

> Difficulty: ⭐⭐⭐⭐☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Proximal Policy Optimization (PPO) |
| Category | Reinforcement Learning (RL) |
| Type | On-policy, stochastic policy-gradient, actor-critic |
| Parametric / Non-parametric | Parametric (deep neural network policies) |
| Generative / Discriminative | Actor part is generative over actions |
| Main Objective | Maximize expected return while keeping each policy update small ("proximal") via a clipped surrogate objective |
| Input | State observations s (and rewards) collected by the current policy |
| Output | A stochastic policy π_θ(a\|s) (and a value function V_φ(s)) |
| Core Idea | Take multiple small gradient steps on data collected by the *old* policy, clipped so the new policy never diverges too far |
| Typical Use Cases | Continuous/robotic control, games, simulated environments, any on-policy task needing stability |

---

## 02. One-Line Definition

### Beginner Definition
PPO updates a policy with a clipped rule: if the new policy wants to get much more likely on a good action, it is capped; if it wants to get much less likely on a bad action, it is floored — so it improves a little at a time without big jumps that wreck the progress.

### Technical Definition
PPO maximizes a clipped surrogate objective
`L(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ), 1−ε, 1+ε)Â_t)]`
where `r_t(θ) = π_θ(a_t|s_t)/π_old(a_t|s_t)` is the probability ratio and `Â_t` is the advantage estimate, optionally including a value-loss term and an entropy bonus.

---

## 03. Intuition

- Imagine teaching someone a dance move. Raw policy gradient is like shouting "try to be 10× more likely to step left" — the person lunges wildly and falls. PPO is like saying "improve, but only nudge each move by a few percent this round."
- The **clip** creates a "trust region without the computational cost" of TRPO (Trust Region Policy Optimization).
- If a new probability ratio `r_t` rises past `1 + ε`, the objective stops rewarding further increases — the update is capped. If it drops below `1 − ε`, the penalty is stopped so the ratio can be pulled back.
- Two modes per sample:
  - **Positive advantage** (good action): benefit rises until ratio hits `1+ε`, then flattens (don't exploit one state too much).
  - **Negative advantage** (bad action): the negative benefit grows until the ratio falls below `1−ε` and the policy "releases the brake", letting it snap back.

---

## 04. Problem It Solves

- Vanilla **policy gradient (REINFORCE)** has high variance: a single weight update uses many trajectory samples and can change the policy drastically in one step, or barely move at all.
- **TRPO** solved the stability problem but required solving a constrained optimization with a Fisher-information-matrix/KKT step — expensive and complex per iteration.
- PPO keeps TRPO-style trust-region *stability* (don't let the policy jump far) using a **cheap first-order clipped objective**, dropping the expensive constraint machinery.
- This makes PPO the default stable **on-policy** method: it is simple, robust, and works on both discrete and continuous action spaces.

---

## 05. Where It Fits in Machine Learning

```
Machine Learning
├── Supervised / Unsupervised / Semi-supervised
└── Reinforcement Learning
    ├── Value-based         — Q-learning, DQN (off-policy)
    ├── Policy Gradient (on-policy)
    │   ├── REINFORCE
    │   ├── Actor-Critic   — A2C, A3C
    │   └── Proximal       — PPO ◄── HERE, TRPO
    └── Actor-Critic (off-policy deterministic)
        ├── DDPG, TD3
        └── SAC
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Policy π_θ | Strategy choosing actions | Distribution over actions given state, parameterized by θ |
| On-policy | Learn only from current policy's data | Each update needs fresh rollouts from π_θ |
| Return (R_t) | Sum of future rewards | R_t = Σ_{k≥0} γ^k r_{t+k} |
| Value function V(s) | Expected return from s | V(s) = E[Σ γ^k r | s] |
| Advantage Â_t | Is this action better than average? | Â_t ≈ R_t − V(s_t) (GAE extension) |
| Probability ratio r_t | New vs old likelihood | r_t(θ) = π_θ(a_t\|s_t)/π_old(a_t\|s_t) |
| Clip (ε) | Trust region width | Keeps r_t within [1−ε, 1+ε] |
| Surrogate objective | Proxy to optimize | Maximizes advantage-scaled ratio |
| Entropy bonus | Encourages exploration | Adds β·H(π(·\|s)) to loss |
| GAE | Advantage smoothing | Generalized Advantage Estimation with λ |
| Actor | Policy network | π_θ(a\|s) |
| Critic | Value network | V_φ(s) |

---

## 07. Input and Output

**Input:**
- Observations `s_t` (e.g., joint angles, pixels, velocities) from environment rollouts.
- Rewards `r_t` collected under the current policy.
- Hyperparameters: learning rate, clip `ε`, GAE `λ`, discount `γ`, batch size, mini-batch size, number of epochs, entropy coefficient.

**Output:**
- Updated policy parameters `θ` (actor).
- Updated value parameters `φ` (critic).
- During use: action samples `a ~ π_θ(·|s)`.

---

## 08. Mathematical Foundation

**Setup.** An MDP: states s, actions a, transition P(s'|s,a), reward r(s,a), discount γ ∈ [0,1). The policy gradient theorem says the gradient of the expected return J(θ) can be written as an expectation over the on-policy distribution:

```
∇_θ J(θ) = E_π[ ∇_θ log π_θ(a|s) · Q^π(s,a) ]
```

Replacing Q^π with the advantage `Â_t = R_t − V(s_t)` (using a baseline) reduces variance without biasing the expectation:

```
∇_θ J(θ) = E_π[ ∇_θ log π_θ(a|s) · Â_t ]
```

**GAE** estimates advantage as a discounted sum of TD errors `δ_t = r_t + γV(s_{t+1}) − V(s_t)`:

```
Â_t^GAE = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}
```

λ=0 gives pure one-step advantage; λ=1 gives full Monte-Carlo return.

---

## 09. Core Formula

The clipped surrogate objective is the heart of PPO:

```text
L^CLIP(θ) = E_π_old [ min( r_t(θ)·Â_t , clip(r_t(θ), 1−ε, 1+ε)·Â_t ) ]

where  r_t(θ) = π_θ(a_t|s_t) / π_old(a_t|s_t)
```

### Meaning
We maximize the expected advantage-weighted probability ratio, but we **clip the ratio** so a single update cannot change probabilities too far from the old policy.

### Symbols
- `θ` current policy params; `θ_old` params that collected the data.
- `π_θ(a|s)`, `π_old(a|s)` new and old action likelihoods.
- `r_t(θ)` probability ratio (1 when θ = θ_old).
- `Â_t` advantage estimate (GAE). Positive = better than average.
- `ε` clip width, typically 0.1–0.3 (0.2 common).

### Intuition
The `min` acts as a lower bound on the *unclipped* objective. If the ratio is clipped, the gradient is zero for that sample — so improvements that would move the ratio beyond the clip are ignored.

### Example (tiny dataset, calculated — hand-verified)
Let ε = 0.2, and three samples with ratios and advantages:

| t | r_t(θ) | Â_t | r_t·Â_t | clip(r_t,0.8,1.2) | clipped·Â_t | min |
|---|---|---|---|---|---|---|
| 1 | 1.25 | +1.0 | 1.25 | 1.20 | 1.20 | **1.20** |
| 2 | 0.60 | −1.0 | −0.60 | 0.80 | −0.80 | **−0.80** |
| 3 | 1.00 | +0.5 | 0.50 | 1.00 | 0.50 | **0.50** |

- Row 1: r=1.25 > 1.2, clipped → term 1.20 (we stop rewarding the extra likelihood).
- Row 2: r=0.60 < 0.8, clipped → −0.80 (the floor pulls it back because the action is bad, A < 0).
- Row 3: within range, unchanged.
- Mean surrogate ≈ (1.20 − 0.80 + 0.50)/3 = 0.90/3 = **0.30**. This is the value PPO maximizes (with the other terms).

---

## 10. Derivation

Start from the policy-gradient objective with importance sampling from the old policy:

```
J(θ) = E_{a~π_old}[ π_θ(a|s)/π_old(a|s) · Â_t ]
     = E_{a~π_old}[ r_t(θ) · Â_t ]
```

TRPO maximizes this subject to a hard constraint on the KL divergence between π_θ and π_old. That constrained problem needs an expensive second-order solve. PPO's insight: **replace the hard KL constraint with a simple clipping penalty** that bounds how far r_t(θ) can move:

```
L^CLIP(θ) = E[ min( r_t(θ)Â_t , clip(r_t(θ),1−ε,1+ε)Â_t ) ]
```

The `min` guarantees L^CLIP ≤ the unclipped objective, so it is a conservative (lower-bound) surrogate. When θ = θ_old, r=1, both branches agree and the gradient equals the standard policy gradient around the current policy.

**Full PPO loss** (common form):

```
L(θ) = L^CLIP(θ) − c1 · L^VF(φ) + c2 · S[π_θ](s)

where L^VF = E[(V_φ(s_t) − R_t)²]  and  S is the policy entropy
```

- `−c1 L^VF`: critic regression to the observed returns (minus: we minimize it).
- `+c2 S`: entropy bonus encourages exploration; keeps probability mass spread rather than collapsing.

---

## 11. How the Algorithm Works

```text
Input: env, policy π_θ, value V_φ
  ↓
Collect N trajectories with π_θ (on-policy rollouts)
  ↓
Compute returns R_t and advantages Â_t (GAE)
  ↓
Loop K epochs:
   Shuffle data into mini-batches
    ↓
   Compute ratio r_t = π_θ/π_old
    ↓
   Loss = L^CLIP − c1·L^VF + c2·H
    ↓
   Gradient ascent on θ, gradient descent on φ (Adam)
  ↓
Update θ_old ← θ (next rollout uses the new policy)
  ↓
Repeat collect → optimize until convergence
```

---

## 12. Training Process

- **Data collection:** rollout K timesteps with the current policy into a buffer with full state, action, reward, value trajectories.
- **Advantage estimation:** compute GAE advantages using the critic; returns R_t for critic targets.
- **Optimization:** run several epochs (e.g., 3–10) of mini-batch gradient ascent over this *fixed* dataset. Because the data is from π_old, we use the ratio r_t to correct for the difference.
- **What's learned:** actor weights θ and critic weights φ.
- **Per iteration change:** θ is nudged toward actions with higher advantage, clipped at ε.
- **Stopping:** when mean return plateaus or a step/timestep budget is exhausted.
- **Final model:** a policy π_θ(·|s) plus a value estimate V_φ(s).

---

## 13. Objective Function / Loss Function

Optimizes `L(θ) = L^CLIP − c1·L^VF + c2·H`:

| Term | Role | High value means |
|---|---|---|
| L^CLIP | Improves policy toward high advantage | Larger ratio·advantage (capped) |
| L^VF | Trains critic to predict return | (Always minimized; we subtract c1·L^VF) |
| H (entropy) | Exploration bonus | More spread-out policy |

- **Why chosen:** clip gives a stable lower-bound surrogate; entropy prevents premature convergence; value loss keeps the critic accurate.
- **High/low loss meaning:** L^CLIP high = exploiting good actions; the objective is *not* the evaluation metric (you should watch mean episode return instead).

---

## 14. Optimization

**Why:** maximize expected return without destabilizing the policy.

**Method:** stochastic gradient ascent (Adam) on θ, gradient descent on φ, over mini-batches re-used K epochs within one rollout.

```text
Current θ, φ
  ↓
Mini-batch ratios r_t and advantages Â_t
  ↓
Compute ∇L^CLIP, ∇L^VF, ∇H
  ↓
θ ← θ + α·∇_θ(L^CLIP + c2·H)      (clip zeroes gradient where clipped)
φ ← φ − α·∇_φ(c1·L^VF)
  ↓
New θ, φ
  ↓
Repeat epochs / new rollout
```

**Convergence:** PPO doesn't strictly guarantee monotonic improvement but in practice is very stable. Local vs global optimum: like all deep RL, it optimizes a non-convex objective; clip acts as an implicit regularizer.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified above).

**Setup.** Old policy π_old, new (partially updated) policy π_θ over two discrete actions. Three samples:

| t | π_old | π_θ | a_t | Â_t |
|---|---|---|---|---|
| 1 | 0.40 | 0.50 | left | +1.0 |
| 2 | 0.50 | 0.30 | right | −1.0 |
| 3 | 0.20 | 0.20 | left | +0.5 |

**Step 1 — ratios:** `r_1 = 0.50/0.40 = 1.25`, `r_2 = 0.30/0.50 = 0.60`, `r_3 = 0.20/0.20 = 1.00`.

**Step 2 — clip (ε=0.2):** `clip(r, 0.8, 1.2)`: r1→1.20, r2→0.80, r3→1.00.

**Step 3 — clipped objective per sample:** 1.20·1.0 = 1.20; 0.80·(−1.0) = −0.80; 1.00·0.5 = 0.50.

**Step 4 — min with unclipped** (shown in §09): rows equal the min values above.

**Step 5 — surrogate mean:** (1.20 − 0.80 + 0.50)/3 = 0.90/3 = **0.30**.

Gradient ascent then raises θ in the direction that increases this mean over the mini-batch, subject to clipping.

---

## 16. Visual Explanation

Clipped objective shape for a positive advantage (good action):

```
gain
  |         ________  ← flat past 1+ε (no more reward for ratio)
  |        /
  |       /
  |      /  slope = A
  |     /
  |____/__________ ratio r
       1  1+ε
```

For a negative advantage (bad action), the curve falls then flattens below 1−ε (the "brake release").

Actor-critic loop:

```
        ┌────────────────────────────┐
        │          ENVIRONMENT       │
        │  s_t ──────►  r_t, s_{t+1} │
        └──────▲─────────────────────┘
               │
      action a_t
               │
   ┌───────────┴──────────┐
   │        ACTOR π_θ      │
   │  policy log-probs      │
   └───────────▲──────────┘
               │  gradient signal (advantage)
   ┌───────────┴──────────┐
   │        CRITIC V_φ     │
   │  estimates return      │
   └───────────────────────┘
```

---

## 17. Algorithm / Pseudocode

```text
1. Initialize actor π_θ and critic V_φ.
2. for each iteration:
3.   Collect T timesteps with π_θ (store s, a, r, V(s), log π(a|s)).
4.   Compute returns R_t and GAE advantages Â_t.
5.   For K epochs:
6.     For each mini-batch:
7.       r_t ← π_θ(a|s) / π_old(a|s)
8.       L ← L^CLIP − c1·L^VF + c2·H
9.       θ ← θ + α∇_θ L ;  φ ← φ − α∇_φ L
10.  Set π_old ← π_θ (discard buffer).
```

---

## 18. From-Scratch Implementation

Minimal but genuine NumPy sketch (update equations appear literally in code):

```python
import numpy as np

def ppo_update(theta, phi, buffer, lr=3e-4, eps=0.2, gamma=0.99, lam=0.95,
               epochs=3, c1=0.5, c2=0.01):
    states  = np.array(buffer["s"])
    actions = np.array(buffer["a"])
    old_logp = np.array(buffer["logp"])
    rewards = np.array(buffer["r"])

    # Compute returns and GAE advantages
    returns = np.zeros_like(rewards)
    adv = np.zeros_like(rewards)
    gae = 0.0
    G = 0.0
    # values from critic V_phi (simulated here)
    values = np.zeros_like(rewards)
    for t in reversed(range(len(rewards))):
        G = rewards[t] + gamma * G
        returns[t] = G
        delta = rewards[t] + gamma * values[t+1] - values[t] if t+1 < len(rewards) \
                else rewards[t] - values[t]
        gae = delta + gamma * lam * gae
        adv[t] = gae
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    for _ in range(epochs):
        for i in range(len(states)):
            # new log-prob from current theta (torch-style; here fake forward pass)
            new_logp = log_pi(theta, states[i], actions[i])   # placeholder
            ratio = np.exp(new_logp - old_logp[i])
            A = adv[i]
            clip_adv = np.clip(ratio, 1 - eps, 1 + eps) * A
            loss_clip = -np.minimum(ratio * A, clip_adv)      # negate for grad-descent
            # (theta update uses autograd on loss_clip - c1*vf_loss + c2*H)
```

This mirrors the exact update math: ratio, clip, min; returns/GAE; the `−` sign because we minimize.

---

## 19. Code Explanation

```text
Code ↓ What does it do? ↓ Why required? ↓ Mathematical concept?

states/actions/logp   Store rollout          On-policy: data from old policy
returns = G           Discounted sum reward  R_t = Σ γ^k r_{t+k}
delta = r + γV' - V   TD error                Basis of GAE
gae = δ + γλ·gae       Recursive smoothing    Â_t^GAE = Σ(γλ)^l δ_{t+l}
ratio = exp(logp-...)  New/old probability    r_t = π_θ/π_old
clip_adv = clip·A      Cap the update        min(rA, clip(r)A)
loss = -min(...)       Negate for descent    Maximize L^CLIP
```

---

## 20. Library Implementation

CleanRL-style PPO (single-file, PyTorch):

```python
import torch, torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(),
                                 nn.Linear(64, 64), nn.Tanh())
        self.mean = nn.Linear(64, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.net(x)
        return self.mean(h), self.log_std, self.critic(h)

    def get_action(self, x):
        m, ls, _ = self.forward(x)
        dist = torch.distributions.Normal(m, ls.exp())
        a = dist.sample(); return a, dist.log_prob(a).sum(-1)

# per update:
ratio = (new_logp - old_logp).exp()
pg_loss = -(torch.min(ratio * adv, torch.clamp(ratio, 1-eps, 1+eps) * adv)).mean()
vf_loss = 0.5 * ((value - returns)**2).mean()
entropy = dist.entropy().mean()
loss = pg_loss + 0.5 * vf_loss - 0.01 * entropy
```

Note how the clipped objective, value loss, entropy bonus, and the `+`/`−` signs match §09/§13 literally.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Clip ε | Trust-region width | Larger = more aggressive updates | 0.1–0.3; 0.2 default |
| GAE λ | Advantage smoothing | λ→1 more bias-free, λ→0 lower variance | 0.95 |
| Discount γ | Credit for future reward | Higher = farther-sighted | 0.99 |
| Rollout T | Buffer length | Longer = more data per update | 1024–2048 |
| Epochs K | Reuse of buffer | Higher = more updates per batch | 3–10 |
| Mini-batch size | SGD granularity | Affects variance & speed | 32–256 |
| Entropy coef c2 | Exploration | Higher = more random policy | 0.0–0.02 |
| Value coef c1 | Critic weight | Higher = value dominates | 0.5 |
| Learning rate | Step size | Too high → instability | 3e-4 |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Actor weights θ, critic weights φ, log_std of policy.

### Hyperparameters (chosen)
- ε, γ, λ, rollout length, epochs, mini-batch, c1, c2, learning rate, hidden sizes.

---

## 23. Assumptions

- **Markov property** — state summarizes history; check: env without hidden memory. If violated: use RNN policies / memory.
- **Stationary environment** — transition/reward distribution stable. If violated: recency weighting.
- **On-policy data validity** — buffer valid only for π_old; keep updates within the clip. Violated → biased gradients.
- **Bounded rewards** — unbounded rewards inflate advantage variance; clip/normalize rewards.

---

## 24. Data Requirements

- Data type: state-action-reward rollouts (no labels).
- Numerical state features; scale/standardize observations when using NN.
- No missing-value handling needed (RL collects its own data).
- Sample efficiency: low/medium; needs many interactions (on-policy).
- No class imbalance concept, but action distribution can collapse → entropy bonus.

---

## 25. Feature Scaling

**Recommended.** Normalize state observations (running mean/std) and advantages (per-batch normalization) — improves stability and learning speed. Reward normalization optional.

---

## 26. Evaluation Metrics

**Training objective ≠ evaluation metric.**

| Metric | Definition | Use |
|---|---|---|
| Mean episode return | Average discounted/undiscounted sum of rewards over eval episodes | Primary measure of performance |
| Mean episode length | Steps per episode | Task progress (timeouts) |
| Return over training timesteps | Learning curve | Sample efficiency, stability |
| Policy entropy | H(π(·\|s)) | Exploration health (should not collapse too early) |

Do **not** judge by actor/critic loss values (loss can rise with better exploration; losses are means during optimization, not final performance).

---

## 27. Advantages

- **Stable & sample-safe** — clip bounds each update; rarely catastrophic.
- **Simple first-order** — no second-order (Fisher/KKT) machinery like TRPO.
- **On-policy correctness** — clean gradient, good for continuous & discrete.
- **Single-file friendly** — easy to implement, tune, debug.
- **Proven default** — one of the most widely used RL algorithms in research/industry.

---

## 28. Disadvantages

- **Sample inefficient** — on-policy; each update discards its data (throws away rollouts).
- **Hyperparameter sensitive** — clip, entropy, epochs interact.
- **No guaranteed monotonic improvement** — clip approximates TRPO's guarantee.
- **Local optima** — entropy annealing can under-explore on sparse reward tasks.
- **CPU/GPU cost** — many short rollouts; parallelism tuning needed.

---

## 29. When to Use

- ✓ Stable on-policy updates are critical.
- ✓ Continuous or discrete control with a simulator.
- ✓ You can generate fresh rollouts cheaply.
- ✓ You want a robust, well-understood baseline.
- ✓ Task rewards are dense enough to learn a value baseline.

---

## 30. When NOT to Use

- ✗ Very sample-constrained (real robots/hardware) → prefer off-policy (SAC/TD3).
- ✗ Massive parallel continuous control → possibly use PPO anyway, but SAC often better.
- ✗ Determininistic action needs with millions of env steps → off-policy deterministic methods.
- ✗ Your reward is extremely sparse and coverage is the bottleneck → explore harder methods / curriculum.

---

## 31. Real-World Applications

| Problem ↓ | Input ↓ | Algorithm ↓ | Output ↓ |
|---|---|---|---|
| Robot locomotion | Joint states/velocities | PPO | Torque commands per joint |
| Game playing | Screen/pixels or state | PPO | Action per frame |
| Continuous control benchmarks | MuJoCo/Atari states | PPO | Optimized policy |
| Sim-to-real (quadrotor, manipulation) | Sim state | PPO + domain random | Transferable policy |

---

## 32. Failure Cases

- **Entropy collapse** on sparse reward → premature deterministic policy, no improvement.
- **Value (critic) inaccuracy** → wrong advantages → wrong gradient direction.
- **Clip too wide** → policy jumps, destabilizes; **too narrow** → slow learning.
- **Reward scale mismatch** → advantage magnitude dominates clip behavior.
- **Non-stationary env** → old-policy data invalid.

---

## 33. Overfitting and Underfitting

- **Overfitting:** policy memorizes a narrow set of trajectories; on-policy data reuse across many epochs worsens this. Mitigate: fewer epochs, larger rollout, more exploration.
- **Underfitting:** too little capacity/short rollout → policy cannot represent useful behaviors. Mitigate: larger networks, longer rollouts.
- Balance via watch on policy/value loss and return curves.

---

## 34. Bias-Variance Perspective

- Policy gradient estimate is **unbiased in θ** (using the score function) but **high-variance**.
- Advantage baseline (critic) and GAE reduce **variance at the cost of bias** (critic is approximate).
- λ trades bias (λ→0, more TD trimming) vs variance (λ→1, Monte-Carlo).
- Clip adds mild **bias** (a lower-bound surrogate) to reduce instability (variance of updates).
- The art: choose γ, λ, clip to balance.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| REINFORCE | Pure MC policy gradient | Simple | High variance | Toy/teaching |
| A2C | Synced actor-critic | Lower variance | Sync overhead | Medium tasks |
| A3C | Async workers | Parallel speed | Non-deterministic sync | Older infra |
| TRPO | Hard KL constraint | Monotonic guarantee | Expensive 2nd-order | Inherited by PPO |
| **PPO** | **Clipped surrogate** | **Stable, simple** | **Sample-inefficient** | **Default on-policy** |
| SAC/TD3 | Off-policy | Sample efficient | More machinery | Continuous sample-poor |

---

## 36. Algorithm Selection Guide

```text
Stable on-policy updates needed?
  ├─ Yes, and plenty of fresh data → PPO ◄──
  ├─ Need guaranteed monotonic → TRPO
  └─ Very sample-constrained → SAC / TD3
```

---

## 37. Common Mistakes

```text
❌ Setting clip too large (ε>0.5) thinking it speeds learning
Why wrong: destroys trust region → policy instability.
Correct: start ε=0.2, tune carefully.

❌ Judging the run by actor loss going "down"
Why wrong: loss is a proxy, not performance.
Correct: watch mean episode return.

❌ Forgetting to zero advantage gradient where ratio is clipped
Why wrong: unclipped term lets update exceed bounds.
Correct: use the min() inside the loss (auto-zeroed).

❌ Reusing the buffer after updating policy
Why wrong: on-policy assumption broken.
Correct: fresh rollouts each iteration.
```

---

## 38. Interview Questions

### Beginner
- **Q: What does the clip in PPO do?** A: It caps the probability ratio to [1−ε, 1+ε] so each update is small and stable.
- **Q: Why a ratio instead of raw probabilities?** A: ratio r = π_θ/π_old measures relative change; stays 1 at start, making gradient well-conditioned.
- **Q: Is PPO on-policy or off-policy?** A: On-policy — it learns only from the current policy's rollouts.

### Intermediate
- **Q: Why is PPO more stable than REINFORCE?** A: Advantage baseline (critic/GAE) reduces variance, and the clip bounds each update.
- **Q: Name the three terms in the full PPO loss.** A: Clipped surrogate L^CLIP, value loss L^VF, entropy bonus c2·H.
- **Q: What does GAE λ=0 vs λ=1 mean?** A: λ=0 one-step TD (lower variance, more bias); λ=1 full return (unbiased, higher variance).

### Advanced
- **Q: How does PPO approximate TRPO's trust region?** A: It replaces the hard KL constraint with a per-ratio clip, a cheaper first-order lower-bound surrogate.
- **Q: Why does clipping zero out gradient for out-of-range samples?** A: `min(rA, clip(r)A)` is flat (constant) in r beyond the clip, so its derivative is 0 there.
- **Q: When is PPO's entropy bonus critical and how to anneal it?** A: On sparse-reward tasks to keep exploration; anneal from high to low while monitoring entropy.

---

## 39. GATE / Exam Perspective

- Key formula to reproduce: `L^CLIP(θ) = E[min(r_t(θ)Â_t, clip(r_t(θ),1−ε,1+ε)Â_t)]` with `r_t = π_θ/π_old`.
- Know: on-policy, actor-critic, trust-region-by-clipping, GAE advantage.
- Traps: distinguishing PPO (clipped, first-order) from TRPO (hard KL, second-order); clip applies to the ratio, not to the advantage.
- *Representative pattern question* (not a verified PYQ): "Explain why clipping the surrogate objective stabilizes policy gradient training." Order: define ratio → clip → why low variance.

---

## 40. Coding Practice

1. **Level 1 — Basic:** Write a function computing `clip_ratio(r, eps)`.
2. **Level 2 — Core:** Implement the full L^CLIP for a mini-batch in NumPy.
3. **Level 3 — GAE:** Implement GAE recursion over a trajectory.
4. **Level 4 — Networks:** Build actor/critic MLP in PyTorch.
5. **Level 5 — Loop:** Couple rollout collection + update loop.
6. **Level 6 — Env:** Train PPO on a continuous-control env (e.g., pendulum).
7. **Level 7 — Real-world:** Tune clip/entropy, compare curves; implement entropy-annealing variant.

---

## 41. Practical ML Workflow

1. Problem → define MDP: state, action, reward.
2. Choose environment/rollout budget.
3. Implement actor & critic networks.
4. Collect initial rollout, normalize observations.
5. Compute GAE advantages + returns.
6. Optimize with clipped objective over multiple epochs.
7. Tune clip/entropy/lr on return curves.
8. Evaluate mean return over fixed eval seeds.
9. Error analysis: entropy collapse, critic drift, reward scale.
10. Deploy policy; monitor drift/retrain.

---

## 42. Complexity

- Training time: dominated by environment interaction (on-policy, sample-inefficient).
- Per-update cost: O(batch × network forward/backward); low.
- Space: buffer of rollout T (not whole experience pool).
- Scaling: parallel workers (multiple envs) improve wall-clock; more envs ≠ more samples beyond wall-clock speedup.
- Complexity grows with network size and rollout length, not with historical experience.

---

## 43. Advanced Concepts

- **TRPO connection:** PPO is the first-order relaxation of TRPO's constrained optimization.
- **GAE as control-variate family:** interpolates bias-variance.
- **Entropy annealing:** schedule c2 to trade exploration vs exploitation.
- **Clipping + old-policy importance sampling:** jointly define valid surrogate.
- **Vectorized parallel envs** (multiple environments in one step) for better utilization.

---

## 44. Connections to Other Algorithms

```
REINFORCE ──(add critic)──► A2C/A3C ──(clip step)──► PPO ◄──(relaxation)── TRPO
                                            │
                                            └──(off-policy counterpart)──► SAC, TD3
```

Policy-gradient family: REINFORCE → actor-critic → PPO. Value-based siblings: DQN, TD3, SAC.

---

## 45. If You Remember Only 5 Things

1. PPO is a stable, on-policy, first-order actor-critic.
2. Core: clipped surrogate `L = E[min(r·Â, clip(r,1±ε)·Â)]`.
3. `r = π_θ/π_old`; clip bounds each probability update (trust region without TRPO's cost).
4. Full loss also includes value loss and an entropy bonus.
5. Sample-inefficient but robust — the default stable on-policy baseline.

---

## 46. Cheat Sheet

| Aspect | Value |
|---|---|
| Algorithm | PPO |
| Category | On-policy actor-critic RL |
| Goal | Maximize return with proximal updates |
| Input | State-action-reward rollouts |
| Output | Policy π_θ(·\|s) |
| Core Formula | L=min(rÂ, clip(r,1±ε)Â); r=π_θ/π_old |
| Loss | L^CLIP − c1·L^VF + c2·H |
| Optimization | Adam, multi-epoch mini-batches |
| Parameters | θ (actor), φ (critic) |
| Hyperparameters | ε, γ, λ, epochs, mini-batch, c1, c2, lr |
| Assumptions | Markov, stationary, on-policy data |
| Advantages | Stable, simple, robust |
| Disadvantages | Sample-inefficient |
| Use When | Fresh data cheap, stability vital |
| Avoid When | Extremely sample-constrained |
| Related | TRPO, A2C, A3C, SAC, TD3 |
| Key Exam | Clipped surrogate, ratio, GAE |
| Key Interview | Trust region via clip; three loss terms |

---

## 47. Final Mental Model

```text
Rollout with π_θ
   → returns + GAE advantages
   → clipped surrogate (min ratio·Â)
   → grad ascent θ, grad descent φ
   → new rollout
```

PPO incrementally nudges the policy uphill, never stepping too far.

---

## 48. Knowledge Check

### Recall
1. Full PPO clipped objective formula.
2. What is the probability ratio?
3. Three terms of the PPO loss.
4. Meaning of ε.
5. Is PPO on-policy or off-policy?

### Understanding
6. Why clip instead of hard constraint?
7. How does the min() produce the lower bound?
8. Why an advantage baseline?
9. Role of entropy bonus.
10. Why is clip applied to ratio, not advantage?

### Application
11. Choose ε for an unstable environment.
12. Detect entropy collapse.
13. Decide PPO vs SAC for a robot.
14. Tune GAE λ.
15. Set up parallel rollouts.

### Mathematical
16. Compute r_t given new/old probs.
17. Evaluate min(rÂ, clip(r)Â) for a sample.
18. Write the GAE recursion.
19. Explain why gradient is 0 when clipped.
20. State the value loss formula.

### Interview
21. PPO vs TRPO.
22. Why stable.
23. When sample-efficiency hurts.
24. Name loss terms.
25. Anneal entropy why.

### Problem Solving
26. Reward scale 10⁶ — what breaks?
27. Policy stuck at low entropy — how to fix.
28. Value function diverges — how to detect.
29. Non-stationary env — adaptation.
30. Batch reuse too high — diagnosis.

## Answers (explained)
1. `L=min(rÂ, clip(r,1±ε)Â)`; ratio `r=π_θ/π_old`.
2. Ratio of new to old action likelihood.
3. Clipped surrogate, value loss, entropy bonus.
4. Trust-region width on the ratio.
5. On-policy.
6. Cheaper than TRPO's Fisher/KKT, still stable.
7. min is ≤ unclipped term, so it's a conservative bound; flat where clipped → zero grad.
8. Baseline V reduces variance without bias.
9. Prevents overconfident/under-exploring policies.
10. Clip limits change in policy probabilities, not the scale of advantage.
11. Lower ε (e.g., 0.1).
12. Entropy → 0 early, return stalls.
13. Sample-constrained → SAC.
14. λ low = less variance, more bias.
15. Multiple vectorized envs feeding one buffer.
16. r = π_θ/π_old.
17. See §09 table.
18. `Â=Σ(γλ)^l δ_{t+l}`, δ = r+γV'−V.
19. Piecewise constant ⇒ derivative 0.
20. `(V_φ(s)−R)²`.
21. Clip vs hard KL constraint.
22. Bounded updates + variance reduction.
23. Fresh rollouts discarded each step.
24. L^CLIP, L^VF, H.
25. Balance late-training exploitation vs exploration.
26. Clip becomes meaningless, advantage dominates; normalize rewards.
27. Raise entropy coef / anneal slower.
28. Watch TD error and value loss blow up.
29. Weight recent data, shorter rollout.
30. Overfitting to stale data; reduce epochs.

---

## 49. Final Learning Checklist

- [ ] I can write L^CLIP from memory
- [ ] I can define the probability ratio r_t
- [ ] I can explain why clip ≈ trust region
- [ ] I can compute one clipped-ratio example
- [ ] I can measure GAE advantages
- [ ] I can list all three loss terms
- [ ] I know ε, γ, λ defaults
- [ ] I can explain on-policy data requirement
- [ ] I can decide PPO vs SAC/TD3
- [ ] I know the actor-critic loop
- [ ] I can write the from-scratch update
- [ ] I can write a CleanRL-style PPO
- [ ] I can detect entropy collapse
- [ ] I can normalize advantages
- [ ] I can evaluate with mean return, not loss
- [ ] I can compare PPO/TRPO/A2C
- [ ] I can explain GAE bias-variance
- [ ] I know when clip gradient is zero
- [ ] I can set up vectorized rollouts
- [ ] I can diagnose unstable updates

---

## 50. Quality Control Note

- **Accuracy:** clipped surrogate, ratio, GAE, entropy verified against standard formulations; numerical example hand-verified.
- **Beginner-friendliness:** analogy + step-by-step numerical example.
- **Math depth:** core formulas, derivation, GAE recursion, verified example.
- **Practical depth:** from-scratch + CleanRL code, hyperparameters, failure cases.
- **Exam depth:** trap analysis, representative pattern question (clearly labeled, not a real PYQ).
- **Structure:** follows the 50-section template exactly.
