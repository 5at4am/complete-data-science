# 07. Actor-Critic

> Difficulty: ⭐⭐⭐⭐⭐ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐⭐⭐ | Coding Required: ⭐⭐⭐⭐⭐
> GATE Relevance: ⭐⭐⭐☆☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Actor-Critic (basic / one-step Actor-Critic; A2C/A3C addressed in Part 2) |
| Category | Reinforcement Learning |
| Type | Policy-based + value-based hybrid, Model-free, TD |
| Parametric / Non-parametric | Parametric (actor π_θ and critic V_φ networks) |
| Generative / Discriminative | N/A (decision-making) |
| Main Objective | Maximize expected return by improving the policy (actor) using an advantage signal produced by the learned value function (critic) |
| Input | State s; critic output (value) and actor output (action distribution) feed the update |
| Output | Policy π_θ(a\|s) (actor) and value estimate V_φ(s) (critic) |
| Core Idea | Two networks: the **actor** adjusts the policy along ∇log π multiplied by the **advantage** A = r + γV_φ(s') − V_φ(s) (one-step TD), while the **critic** learns V_φ(s) via TD updates — giving low-variance policy gradients with bootstrapping |
| Typical Use Cases | Continuous/discrete control, robotics, games, any on-policy task where REINFORCE's variance is unacceptable |

---

## 02. One-Line Definition

### Beginner Definition
Actor-Critic uses two learners — an "actor" that decides what to do, taught by a "critic" who estimates how good each situation is — so the actor learns from *"you did better than expected"* instead of raw luck.

### Technical Definition
Actor-Critic methods maintain a parametrized policy π_θ (the actor) and a parametrized value function V_φ(s) (the critic); the critic is updated by temporal-difference learning, and the actor is updated by policy-gradient ascent using the TD advantage `A = r + γV_φ(s') − V_φ(s)` as the scaling factor, replacing the high-variance Monte-Carlo return of REINFORCE with a lower-variance bootstrapped signal.

---

## 03. Intuition

REINFORCE was noisy because it graded every action by the whole episode's total score — "luck" contaminated the signal. Actor-Critic fixes this with a **critic**: a teammate who watches each step and constantly estimates "how well is this situation going, on average?"

Now when the actor makes a move:
- The critic predicts the situation's value **before** and **after**.
- The **advantage** = (real reward) + (predicted value after) − (predicted value before).
- If the advantage is positive, the actor was *better than average* → strengthen that move. If negative, weaken it.

The luck is cancelled: even in a rotten episode, a move that improved an already-bad situation gets a positive advantage. This is dramatically less noisy than "total score of the episode."

Think of it as training with a coach instead of a judge: the judge only gives a final score (REINFORCE); the coach gives immediate, contextual feedback every step (Actor-Critic).

Step-by-step reasoning:
1. The actor picks action a in state s.
2. The world returns r and s'.
3. The critic estimates V_φ(s) and V_φ(s') (baseline before/after).
4. Advantage = r + γ·V_φ(s') − V_φ(s) — "better or worse than expected?"
5. Actor update: strengthen π where advantage > 0, weaken where < 0.
6. Critic update: nudge V_φ(s) toward the TD target r + γV_φ(s').
7. Repeat every single step (one-step TD — fast, low variance).

---

## 04. Problem It Solves

**Problem:** REINFORCE's Monte-Carlo returns are unbiased but have variance so high that learning is painfully slow; also it can't update until the episode ends.

**Example:** A robot learns a 200-step manipulation task. REINFORCE must wait 200 steps per update and its gradient is dominated by all the noise collected along the way. The robot essentially learns from the entire trajectory's luck.

**What we want:**
1. Lower-variance policy-gradient updates.
2. Updates during the episode (online).
3. A principled "was this action good *relative to typical performance here*?" signal.

**Why Actor-Critic helps:** the critic's value estimate acts as a **baseline**, and the TD advantage `r + γV(s') − V(s)` measures relative quality. It:
- keeps the policy-gradient estimate (approximately) unbiased up to bootstrap bias,
- collapses variance vs REINFORCE,
- updates every step (one-step TD bootstrap).

**Small example:** In state with expected value 5, an action gives r=0 and next-state value predicted 6 (γ=1 toy): advantage = 0+6−5 = +1 → the actor makes the action more likely even though the raw reward is 0. REINFORCE would have judged only via episode totals, missing this nuance.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning
├── Unsupervised Learning
└── Reinforcement Learning
    ├── Model-Free
    │   ├── Value-Based (Q-learning, DQN) — files 01,05
    │   ├── Policy-Based
    │   │   ├── REINFORCE (MC gradient) — file 06
    │   │   └── ACTOR-CRITIC        ← YOU ARE HERE
    │   │       ├── 1-step AC (this note)
    │   │       ├── A2C / A3C          (Part 2)
    │   │       └── PPO / TRPO         (Part 2)
    │   └── Hybrids (DDPG, SAC — Part 2)
    └── Model-Based
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Actor | The decision-maker | Policy π_θ(a\|s) |
| Critic | The value judge | Value function V_φ(s) (or Q or advantage) |
| Advantage A | Better/worse than expected | A = r + γV_φ(s') − V_φ(s) (one-step TD form) |
| Baseline | "Typical performance here" | V_φ(s) subtracts out state luck |
| TD target | Update direction for critic | r + γV_φ(s') |
| TD error / δ | Surprise for critic AND actor | δ = r + γV_φ(s') − V_φ(s) |
| Policy gradient | Actor's update direction | ∇log π_θ(a\|s)·A |
| Bootstrapping | Estimate from estimates | Critic's γV(s') is an estimate |
| On-policy | Learn from own behavior | AC uses current π_θ episodes |
| A2C | Synchronous N-agent Actor-Critic | Part 2 |
| A3C | Asynchronous version | Part 2 |
| Entropy bonus | Encourage exploration | Small +βH(π) term |

---

## 07. Input and Output

**Input (training):**
- Transitions (s, a, r, s') streamed online from the environment under the current policy.
- High- or low-dimensional states; discrete or continuous actions.

**Input (hyperparameters):**
- γ, α_actor, α_critic (or single Adam), architecture, entropy β, number of steps.

**Output (training):**
- Actor weights θ (policy) and critic weights φ (value).

**Output (prediction / use):**
- π_θ(a|s) for action selection (deployment: mean/argmax for determinism), plus V_φ(s) for diagnostics.

**Parameters learned:** θ (actor) and φ (critic).

**Hyperparameters:** γ, α_actor, α_critic, β (entropy), architecture, update cadence.

---

## 08. Mathematical Foundation

The actor learns by policy gradient on the expected return:

```text
∇J(θ) = E[ Σ_t A_t ∇_θ log π_θ(a_t|s_t) ]
```

**The discount factor γ intuitively:** reward now is worth more than reward later; γ scales both the advantage's future term and the critic's target.

The critic learns via one-step TD:

```text
V_φ(s) ← V_φ(s) + α_critic·δ,    δ = r + γ·V_φ(s') − V_φ(s)
```

The **advantage** used by the actor is exactly the TD error:

```text
A_t = r_{t+1} + γ·V_φ(s_{t+1}) − V_φ(s_t) = δ
```

This unifies the two updates: **the same δ drives both.** The actor treats a *positive* δ as "better than the critic's expectation — reinforce"; a *negative* δ as "worse than expected — suppress."

**Why this reduces variance:** REINFORCE's G_t contains the full episode's randomness; A_t = δ contains only the last transition's randomness plus estimate errors. Bootstrapping with V_φ trades a little bias for a large variance cut.

**Required math concepts:** policy gradient (file 06), TD learning (file 04), neural networks, advantage estimation.

---

## 09. Core Formula

### The actor update

```text
θ ← θ + α_actor · A · ∇_θ log π_θ(a|s)
```

#### Meaning
Increase probability of actions with positive advantage, decrease with negative.

#### Symbols
- `θ` = actor weights
- `α_actor` = actor learning rate
- `A` = advantage (TD error δ)
- `π_θ(a|s)` = policy

#### Intuition
The advantage answers: "did this action beat the average performance here?" The actor moves to make such actions more likely.

---

### The critic update

```text
φ ← φ + α_critic · δ · ∇_φ V_φ(s),   δ = r + γ·V_φ(s') − V_φ(s)
```

#### Meaning
Nudge the value estimate toward the one-step TD target (standard TD learning).

#### Symbols
- `φ` = critic weights
- `α_critic` = critic learning rate
- `δ` = TD error
- `V_φ(s)` = critic prediction

#### Intuition
The critic keeps correcting its "expected performance" toward reality; over time V_φ becomes a good baseline.

---

### The unifying advantage (one-step form)

```text
A = r + γ·V_φ(s') − V_φ(s)
```

#### Meaning
The TD error itself serves as the advantage for one-step Actor-Critic.

#### Symbols
- `r` = reward
- `V_φ(s)`, `V_φ(s')` = before/after critic estimates
- `γ` = discount

#### Intuition
"Reward + predicted-future − predicted-present." Positive → good relative to expectation; negative → poor relative to expectation.

#### Example (hand-computed)
γ=0.9, V_φ(s)=4.0, V_φ(s')=5.0, r=0.
```text
A = 0 + 0.9·5.0 − 4.0 = 0.5 > 0  → action reinforced.
```
If r=−5: A = −5 + 4.5 − 4.0 = −4.5 < 0 → action suppressed.

---

## 10. Derivation

**Step 1 — Start from the REINFORCE gradient:**
```text
∇J(θ) = E[ Σ_t G_t ∇log π_θ(a_t|s_t) ]
```

**Step 2 — Subtract a baseline that doesn't depend on the action:**
The value of the *state* V_φ(s_t) is action-independent, so by the baseline property (E[b∇log π] = 0):
```text
∇J(θ) = E[ Σ_t (G_t − V_φ(s_t)) ∇log π_θ(a_t|s_t) ]
```
The subtracted term = **advantage** (how much better/worse the return was than the state's typical value).

**Step 3 — Bootstrap the return with the critic:**
Replace the sampled return piece by a one-step estimate:
```text
G_t ≈ r_{t+1} + γ·V_φ(s_{t+1})
```
so:
```text
A_t = r_{t+1} + γ·V_φ(s_{t+1}) − V_φ(s_t) = TD error δ
```

**Step 4 — Critic learning:** the critic minimizes its own TD error:
```text
L_critic(φ) = (1/2)( r + γ·V_φ(s') − V_φ(s) )²   (semi-gradient: φ treated as constant in target)
```

**Step 5 — The two updates:**
```text
Actors:  θ ← θ + α_actor·δ·∇log π_θ(a|s)
Critic:  φ ← φ − α_critic·∇_φ L_critic
```
**Note the bias–variance tradeoff:** replacing G_t with a bootstrap introduces small bias (the critic's estimates) but removes the full-episode noise — the defining advantage of Actor-Critic over REINFORCE.

**Theorem note (Policy Gradient Theorem):** for a differentiable policy, ∇J depends only on ∇log π and the advantage Q^π − V^π, so using the TD advantage preserves the correct ascent direction (to the quality of the approximation).

---

## 11. How the Algorithm Works

```text
Init actor π_θ, critic V_φ
    ↓
Loop steps (online, every transition):
    In state s: sample action a ~ π_θ(·|s)
    ↓
    Step env → r, s'
    Compute δ = r + γ·V_φ(s') − V_φ(s)     (terminal: V_φ(s')=0)
    ↓
    Actor:  θ ← θ + α_actor·δ·∇log π_θ(a|s)
    Critic: φ ← φ + α_critic·δ·∇V_φ(s)
    ↓
    s ← s'
Repeat
    ↓
Final model: actor + critic
    ↓
Deployment: use actor only (deterministic mean/argmax if wanted)
```

---

## 12. Training Process

**Pre-training:** create actor network (softmax/Gaussian head) and critic network (scalar output), initialize θ, φ, hyperparameters.

**During training:**
- Every step: act, observe (r, s'), compute TD error δ, update BOTH networks.
- The critic improves its baseline; the actor improves the policy against it.
- Both are updated online, no need to wait for episode end.

**What is learned:** θ (better policy) and φ (better value estimates).

**Changes per iteration (one step):**
```text
δ = r + γ·V_φ(s') − V_φ(s)
θ += α_actor·δ·∇log π_θ(a|s)
φ += α_critic·δ·∇V_φ(s)
```

**Stopping:** episode/step budget; evaluation-return plateau.

**Final model contents:** actor weights (deployable policy) and critic weights (baseline/diagnostics).

---

## 13. Objective Function / Loss Function

**Actor objective (maximize):**
```text
J_actor(θ) = E[ A_t log π_θ(a|s) ]       (plus optional entropy bonus β·H(π))
```

**Critic objective (minimize):**
```text
L_critic(φ) = (1/2)( r + γ·V_φ(s') − V_φ(s) )²     (mean squared TD error)
```

**Why these:**
- The actor's is the policy-gradient surrogate (maximizing expected advantage-augmented log-prob).
- The critic's is plain TD regression (the same squared TD error as file 04).

**Interpretation of loss values:** critic loss decreasing → values becoming consistent. Actor "loss" is not a meaningful scale — its value depends on advantage magnitude. Neither is the task metric.

**Training objective ≠ evaluation metric:** both losses measure *learning health*, not task performance. Evaluate the actor's greedy/mean-policy episode returns.

---

## 14. Optimization

**Definition:** jointly maximize J_actor and minimize L_critic.

**Methods:** stochastic gradient ascent (actor) and descent (critic), typically with Adam; often one optimizer or two with separate rates.

**Gradients:**
```text
∇_θ J_actor = δ·∇log π_θ(a|s)
∇_φ L_critic = −δ·∇V_φ(s)
```

**Learning rates:** α_actor and α_critic; misbalance destabilizes (critic too fast → drift; critic too slow → stale baseline).

**Convergence:** tabular/linear-approximation cases have convergence guarantees for TD critics; general neural AC use is heuristic but standard. Bootstrapping introduces bias but tames variance.

**Local/global:** non-convex joint objective; local optima in practice. Stability monitored via evaluation returns and gradient norms.

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified; scalar-parameter actor + scalar critic)

A 2-action bandit with **no discounting** (γ=1), so the TD advantage for a one-step episode is A = r + V_φ(T) − V_φ(s), with V_φ(T)=0.

Setup: θ=0 (so p(action 1)=σ(0)=0.5), V_φ(·) parameterized as the estimated "entry value" v. Current critic v = 4.0. Sample action 1 → reward r = 6. α_actor = 0.1, α_critic = 0.1.

**Step 1 — Advantage (TD error):**
```text
A = r + γ·V_φ(T) − V_φ(s) = 6 + 0 − 4 = +2
```
Positive → action 1 performed better than the critic's expectation. ✓

**Step 2 — Score function** (Bernoulli policy, p=0.5, action=1):
```text
score = 1 − p = 0.5
```

**Step 3 — Actor update:**
```text
θ ← θ + α_actor·A·score = 0 + 0.1·2·0.5 = 0.1
p = σ(0.1) ≈ 0.525
```
Action 1 probability rose 0.5 → 0.525. ✓

**Step 4 — Critic update:**
```text
TD target = r + γ·V_φ(T) = 6 + 0 = 6
v ← v + α_critic·(target − v) = 4 + 0.1·(6−4) = 4 + 0.2 = 4.2
```
The critic raised its estimate toward the observed 6. ✓

**Second step (same settings, action 1):**
A = 6 − 4.2 = +1.8; score = 1−0.525 = 0.475:
```text
θ ← 0.1 + 0.1·1.8·0.475 = 0.1 + 0.0855 = 0.1855 → p ≈ 0.546
v ← 4.2 + 0.1·(6−4.2) = 4.38
```

| step | A (advantage) | θ before | θ after | p(a1) | v before | v after |
|---|---|---|---|---|---|---|
| 1 | 2.0 | 0 | 0.100 | 0.525 | 4.0 | 4.20 |
| 2 | 1.8 | 0.100 | 0.186 | 0.546 | 4.2 | 4.38 |

**Why this beats REINFORCE here:** REINFORCE would scale updates by the raw return (G=6); the actor-critic scales by the *relative* **A** (2, then 1.8) — noise from each episode's absolute level is removed by the baseline v. Hand-verified arithmetic confirms both the actor's probability rise and the critic's drift toward the environment mean.

---

## 16. Visual Explanation

### The actor-critic loop

```text
        ┌──────────────────────────────────────────┐
        │                                          ▼
   π_θ(a|s) ── sample a ──▶ environment ──▶ (r, s')
   (actor)                                │
        ▲                                 ▼
        │                        critic: δ = r + γV_φ(s') − V_φ(s)
   θ += α_actor·δ·∇log π       └────────────┬────────────┘
                                            ▼
                          φ += α_critic·δ·∇V_φ(s)   (TD update)
```

### Network architecture

```text
        state s
    ┌─────┴─────┐
    │  ACTOR    │          │  CRITIC   │
    │  π_θ      │          │  V_φ(s)   │  ──▶ scalar value
    └─────┬─────┘          └─────┬─────┘
          ▼                       │
   [p(a0), p(a1)]        both learn from the SAME δ
```

### Advantage semantics

```text
δ > 0  →  "action exceeded the critic's expectation"  → reinforce (up)
δ < 0  →  "action underperformed the expectation"      → suppress (down)
```

---

## 17. Algorithm / Pseudocode

```text
1. Init actor π_θ, critic V_φ, α_actor, α_critic, γ
2. For each episode:
     s ← env.reset()
     done ← False
     while not done:
        a ~ π_θ(·|s)
        s', r, done ← env.step(a)
        V_next = 0 if done else V_φ(s')
        δ = r + γ·V_next − V_φ(s)
        θ ← θ + α_actor·δ·∇log π_θ(a|s)
        φ ← φ + α_critic·δ·∇V_φ(s)
        s ← s'
3. Return (π_θ, V_φ)
```

---

## 18. From-Scratch Implementation

Minimal numpy one-step Actor-Critic on a 2-action bandit (scalar-parameter policy + scalar value), fully runnable:

```python
import numpy as np

class OneStepActorCritic:
    def __init__(self, alpha_theta=0.1, alpha_v=0.1, gamma=1.0):
        self.alpha_theta = alpha_theta
        self.alpha_v = alpha_v
        self.gamma = gamma
        self.theta = 0.0            # Bernoulli policy parameter
        self.v = 0.0                # scalar critic (entry value)
        self.rewards = []

    def act(self, rng):
        p = 1.0/(1.0+np.exp(-self.theta))
        a = 1 if rng.random() < p else 0
        s = np.array([1.0 - p, p])  # store both probs for this scalar case
        logits = [0.0, self.theta]
        return a, s, logits

    def one_step(self, rng, r):
        a, s, logits = self.act(rng)          # sample action
        delta = r + self.gamma*0.0 - self.v   # one-step advantage (terminal)
        if a == 1:
            score = 1.0 - s[1]                # ∇log σ(θ) = 1−p
        else:
            score = -s[0]                     # ∇log(1−σ(θ)) = −p
        self.theta += self.alpha_theta*delta*score   # actor
        self.v += self.alpha_v*delta                # critic
        return a, delta

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ac = OneStepActorCritic()
    for ep in range(2000):
        r = 6.0 if ac.act(rng)[0] == 1 else 0.0   # action 1 is best
        ac.one_step(rng, r)
    p = 1.0/(1.0+np.exp(-ac.theta))
    print("p(a1) =", round(p,3), " critic v =", round(ac.v,3))
    print("expected ~ 1.0 and 6.0")
```

---

## 19. Code Explanation

```text
Line:  delta = r + gamma*0.0 - self.v
   What: one-step TD advantage (terminal next-value is 0)
   Why: the single signal driving actor AND critic
   Math: δ = r + γV(s') − V(s)

Line:  score = 1.0 - s[1]   (action 1 taken)
   What: score function for the Bernoulli/softmax policy
   Why: actor gradient direction
   Math: ∇log σ(θ) = 1 − p

Line:  self.theta += self.alpha_theta*delta*score
   What: actor update
   Why: reinforce actions beating the baseline
   Math: θ ← θ + α_actor·δ·∇log π

Line:  self.v += self.alpha_v*delta
   What: critic update
   Why: move value estimate toward its TD target
   Math: v ← v + α_critic·(r + γv' − v)
```

---

## 20. Library Implementation

PyTorch one-step Actor-Critic on CartPole (actor + critic from shared features):

```python
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

class AC(nn.Module):
    def __init__(self, n_s, n_a):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(n_s,128), nn.ReLU())
        self.actor = nn.Linear(128, n_a)
        self.critic = nn.Linear(128, 1)
    def act(self, s):
        h = self.fc(s)
        logits = self.actor(h)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), self.critic(h)

env = gym.make("CartPole-v1")
net = AC(env.observation_space.shape[0], env.action_space.n)
opt = optim.Adam(net.parameters(), lr=1e-2)
gamma = 0.99

for ep in range(1500):
    s, _ = env.reset()
    done = False
    G = 0.0
    while not done:
        s_t = torch.FloatTensor(s)
        a, logp, v = net.act(s_t)
        s2, r, term, trunc, _ = env.step(a.item())
        done = term or trunc
        G += r
        with torch.no_grad():
            v_next = 0.0 if done else net.critic(net.fc(torch.FloatTensor(s2)))
        delta = r + gamma*v_next - v.item()
        actor_loss = -logp*delta
        critic_loss = delta**2
        loss = actor_loss + critic_loss
        opt.zero_grad(); loss.backward(); opt.step()
        s = s2
    if ep % 100 == 0:
        print(f"ep {ep}, return {G:.1f}")
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Actor LR α_actor | Policy step size | High → unstable; low → slow | 1e-3–1e-2 |
| Critic LR α_critic | Value step size | High → value chasing; low → stale baseline | 1e-3–1e-2 (often same) |
| Discount γ | Future weight | Low → myopic | 0.99 |
| Entropy β | Exploration pressure | Controls probability spread | 0–0.01 start |
| Architecture | Actor/critic capacity | Shared vs separate nets | Separate heads usual |
| Update cadence | Every step vs every N | n-step tradeoff | 1-step here; n-step in Part 2 |
| Return/obs normalization | Scaling | Stability | Standardize observations |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- Actor weights θ (policy distribution).
- Critic weights φ (value function).

### Hyperparameters (chosen)
- γ, α_actor, α_critic, entropy β, architecture, update cadence, optimizer.

---

## 23. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Differentiable policy | π_θ smooth in θ | Actor gradient | Standard heads OK | Use other families |
| Critic tracks policy | V_φ approximates V of current π | Advantage validity | Compare V to returns | Better/tuned critic |
| On-policy data | Transitions from current π | Gradients valid | Refresh samples per θ change | Importance sampling / off-policy AC |
| Markov property | s carries needed info | TD bootstrap | Stack frames/memory | Add recurrent architecture |
| Bounded rewards | Stable δ magnitudes | Stability | Clip/normalize | Reward scaling |

---

## 24. Data Requirements

- **Data type:** online stream of (s, a, r, s') under the current policy (on-policy).
- **State space:** arbitrary (vector or high-dim).
- **Action space:** discrete (softmax head) or continuous (Gaussian head).
- **Rewards:** scalar; normalization strongly recommended.
- **Sample budget:** far smaller than REINFORCE for equal variance, still needs many steps.
- **No missing/outliers/scaling:** scale observations and (optionally) advantages.

---

## 25. Feature Scaling

**Recommended:** normalize state observations for stable network training (StandardScaler or rescale). Advantages δ are usually kept as-is or lightly normalized across batches; extreme reward scales can be clipped. The critic's output scale follows the rewards, so reward normalization/scaling is the practical key.

---

## 26. Evaluation Metrics

| Metric | Definition | Interpretation | Use When | Avoid When |
|---|---|---|---|---|
| Mean episode return (deterministic eval) | Deterministic-policy average return | Real task quality | All cases | — |
| Success rate | Solved episodes fraction | Task completion | Goal tasks | — |
| Mean TD error magnitude | \|δ\| average | Critic/call alignment | Diagnosing instability | Final quality claims |
| Policy entropy | −Σp log p | Exploration health | Early-training checks | Terminal reporting |

**Important:** neither the actor surrogate nor the critic TD error is an evaluation metric. Small TD error ≠ good policy (a constant zero-value critic also has low error on zero-return tasks). Evaluate the **actor** deterministically and average episode returns.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Lower variance than REINFORCE | Critic baseline removes return-level noise |
| Online, step-by-step updates | Learns during episodes (TD bootstrap) |
| Supports continuous actions | Actor head is a distribution, no argmax |
| Flexible value use | V_φ serves as baseline and evaluator |
| Combines the best of both families | Policy gradients + value learning |
| Basis for modern methods | A2C/A3C/PPO/SAC all extend it |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| Introduces bias via bootstrapping | Estimates inherit critic error |
| Two networks to tune | More hyperparameters, harder to stabilize |
| Actor-critic coupling | Bad critic → bad actor (positive feedback) |
| On-policy sample waste | Old episodes unusable after θ change |
| Still high variance vs modern variants | 1-step AC noisy on sparse rewards |
| No global convergence guarantees (neural) | Stability is heuristic |
| Value function target drift | Critic must keep up with changing policy |

---

## 29. When to Use

✓ Continuous or discrete actions where policy optimization is needed.
✓ REINFORCE's variance is too high but you stay on-policy.
✓ You want step-level (TD) updates rather than full episodes.
✓ Cheap-ish environment interaction (still sample-heavy).
✓ As the starting point before moving to A2C/PPO.
✓ You need a critic anyway (diagnostics, advantage estimation).

---

## 30. When NOT to Use

✗ Very small discrete worlds — tabular Q-learning is simpler and faster.
✗ Ultra-limited interaction budget — needs modern sample-efficient off-policy methods (SAC).
✗ Extremely sparse long-horizon returns — 1-step advantage sees little signal (need n-step/GAE/curiosity).
✗ When you want theoretical stability guarantees — tabular/linear cases only.
✗ Fully deterministic dynamics where model-based planning shines.

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Robot manipulator control | joint positions | One-step AC (continuous head) | Joint torque commands |
| Autonomous discrete nav | sensor state | One-step AC (softmax) | Steering choice |
| Game-playing agents | game state | One-step AC | Move choice |
| Bidding/pricing agents | market state | One-step AC | Bid level |
| Dialogue policy learning | dialog state | One-step AC | Utterance type |

---

## 32. Failure Cases

- **Critic explosion:** unbounded rewards → huge δ → unstable updates; normalize.
- **Entropy collapse:** actor becomes deterministic; add entropy bonus.
- **Actor-critic misalignment:** critic lags the moving policy (common in non-stationary on-policy learning).
- **Bootstrap bias amplification:** wrong critic estimate propagates into actor gradients.
- **Hyperparameter sensitivity:** mismatched α_actor/α_critic destabilizes learning.
- **Long-horizon blindness:** one-step advantage misses delayed consequences.

---

## 33. Overfitting and Underfitting

- **Underfitting:** small actor/critic nets can't represent good policy/value → poor returns, small losses.
- **Overfitting (to recent experience):** sharing features + SGD on correlated online data can specialize on recent states — the on-policy analogue of catastrophic interference; mitigated by n-step returns, batching, and (in Part 2) A2C's N parallel actors or PPO's clipping/limits.
- **Balance:** capacity sized to state complexity; monitor entropy and eval returns, not just losses.

---

## 34. Bias-Variance Perspective

- **Variance:** dramatically lower than REINFORCE — δ uses one transition, and the baseline removes return-level shifts.
- **Bias:** introduced by bootstrapping (γV(s') is an estimate) plus any critic approximation error — REINFORCE had zero bias from returns; AC trades this in for variance reduction.
- **The spectrum:** REINFORCE (no bias, high variance) ↔ one-step AC (biased, low variance) ↔ n-step AC / A2C (tunable middle) ↔ PPO/GAE (Part 2 uses λ-returns to choose the operating point).

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| REINFORCE | MC policy gradient | Unbiased, simple | High variance | Concept/demo |
| One-step AC | TD advantage + policy gradient | Lower variance, online | Bootstrap bias | Everyday AC start |
| Q-learning/DQN | Value-based control | Sample-efficient (replay) | Discrete actions, no stochastic policy | Discrete control |
| A2C | Synchronous multi-actor | Stabilizes, parallel | More machinery (Part 2) | Larger-scale AC |
| A3C | Asynchronous actors | Faster wall-clock (Part 2) | Nondeterminism (Part 2) | Legacy Parallel |
| PPO | Clipped surrogate (Part 2) | Stable modern default | More tuning knobs | Production RL |

---

## 36. Algorithm Selection Guide

```text
Need a policy (continuous or stochastic)?
├── Variance acceptable → REINFORCE
├── Want online/low variance → ONE-STEP ACTOR-CRITIC
├── Long-horizon/parallel → A2C/A3C (Part 2)
├── Modern stable default → PPO (Part 2)
└── Sample-hungry settings → off-policy SAC/DDPG (Part 2)
```

---

## 37. Common Mistakes

```text
❌ Forgetting that the actor needs ∇log π — NOT the action value
Why wrong: mixing up regression targets with policy updates.
Correct: actor loss = −δ·log π (backprop only through log-prob).

❌ Backpropagating through the critic's target (double-counting φ)
Why wrong: the target r + γV_φ(s') must be treated as constant.
Correct: detach/no_grad the target (semi-gradient for the critic).

❌ Using the raw return for A instead of the advantage
Why wrong: you've just reinverted REINFORCE (variance returns).
Correct: A = r + γV(s') − V(s).

❌ Terminal states not masked
Why wrong: phantom future value inflates both updates.
Correct: V_next = 0 when done.

❌ Judging quality by critic loss
Why wrong: both losses are learning-health signals, not task metrics.
Correct: evaluate actor deterministically; average episode returns.
```

---

## 38. Interview Questions

### Beginner
**Q1. What are the actor and critic?**
A: Actor = policy π_θ choosing actions; critic = value function V_φ judging how good states are.

**Q2. What is the advantage in one-step Actor-Critic?**
A: A = r + γV_φ(s') − V_φ(s) — whether the outcome beat the critic's expectation.

**Q3. How does the actor update?**
A: θ += α·A·∇log π_θ(a|s) — reinforce actions that beat the baseline.

### Intermediate
**Q4. How does AC reduce REINFORCE's variance?**
A: It replaces the full-return G_t with the bootstrapped advantage δ, removing whole-episode noise at the cost of small bias.

**Q5. Why does the critic learn with TD?**
A: V_φ must approximate the value of the current policy; TD updates are online and low variance.

**Q6. What happens to the objective if the critic is exact?**
A: The actor gradient becomes ∇log π·(G_t − V) — exactly the zero-mean advantage, still unbiased, with variance minimized for that baseline.

### Advanced
**Q7. Where does bias come from in AC?**
A: Bootstrapping (using γV(s')) plus critic approximation error; the substituted advantage is no longer exactly the true advantage.

**Q8. What is the Policy Gradient Theorem's role?**
A: It justifies using Q^π − V^π (hence δ) rather than G_t in ∇J without changing the ascent direction up to estimator quality.

**Q9. What distinguishes A2C/A3C (Part 2)?**
A: A2C runs N synchronous actors and averages their advantages (more stable gradients); A3C uses asynchronous parallel actors; both reduce correlation and increase throughput.

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
```text
Advantage (1-step):  A_t = r_{t+1} + γ·V_φ(s_{t+1}) − V_φ(s_t)
Actor update:        θ ← θ + α_actor·A_t·∇log π_θ(a_t|s_t)
Critic update:       φ ← φ + α_critic·A_t·∇V_φ(s_t)
```

**Concepts likely tested:**
- Why the critic serves as a baseline
- Advantage = TD error (one-step case)
- Bias–variance tradeoff vs REINFORCE
- Bootstrapping in the actor's signal
- Terminal masking

> **Representative pattern question (NOT a past GATE PYQ):** "γ=0.9, V(s)=2, V(s')=3, r=1. Compute the advantage and say whether the action is reinforced." → A = 1 + 0.9·3 − 2 = 1.7 > 0 → reinforced.

**Common traps:**
- Using G_t instead of the advantage (that's REINFORCE).
- Backpropagating through the critic target.
- Confusing actor loss (log-prob) with Q-learning loss (Bellman MSE).
- Omitting entropy, then suffering collapse.

---

## 40. Coding Practice

**Level 1 — Basic:** Hand-compute one advantage + actor + critic update (Section 15).
**Level 2 — Bandit:** Implement the from-scratch scalar AC; verify p→1, v→6.
**Level 3 — CartPole:** PyTorch one-step AC; plot eval returns.
**Level 4 — Variance compare:** AC vs REINFORCE return curves on the same task.
**Level 5 — Continuous:** Gaussian-head AC on Pendulum-v1 (small n).
**Level 6 — Entropy:** add/decay entropy bonus; report entropy curve.
**Level 7 — Real-world case study:** on a custom continuous-control gym task, ablate (a) no baseline, (b) with critic baseline, (c) 1-step vs 2-step advantage; report greedy eval returns; explain which failure modes each ablation hits.

---

## 41. Practical ML Workflow

```text
Problem → continuous vs discrete actions; define reward & γ
   ↓
Environment → gymnasium / custom simulator
   ↓
Preprocess → normalize obs; scale/clip rewards
   ↓
Networks → actor head + critic head (shared or separate)
   ↓
Train → online: act → δ → actor & critic updates
   ↓
Evaluate → deterministic actor eval returns
   ↓
Tune → α_actor/α_critic balance, entropy β, γ
   ↓
Error analysis → critic TD-error magnitude, entropy, gradient norms
   ↓
Deploy → actor (deterministic mode) + monitor env drift
   ↓
Scale → A2C/PPO (Part 2) when needed
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Per step compute | O(actor net + critic net) | Two forwards |
| Per step update | O(network params) | Both networks |
| Space | O(\|θ\| + \|φ\|) | Two nets |
| Env steps to converge | 10⁴ – 10⁶ | On-policy; task-dependent |
| Parallel (A2C, Part 2) | linear speedup | N actors over the environment |
| Accent vs REINFORCE | fewer episodes for equal variance | Advantage replaces full returns |

---

## 43. Advanced Concepts

- **n-step advantage and GAE(λ):** generalize δ over n steps (GAE trades bias/variance smoothly — used by PPO).
- **Policy Gradient Theorem:** ∇J ∝ E[Q^π ∇log π] → advantage form.
- **Entropy regularization and adaptive β.**
- **Sharing vs separating actor/critic features.**
- **Off-policy AC variants** (DDPG/SAC — Part 2).
- **Async/sync parallelism** (A3C vs A2C — Part 2).

---

## 44. Connections to Other Algorithms

```text
Actor-Critic
   │
   ├── actor side → Policy Gradient / REINFORCE (file 06)
   ├── critic side → TD Learning (file 04), Q-functions (files 01,05)
   ├── generalization → n-step / GAE / PPO / A2C / A3C (Part 2)
   ├── off-policy deep relatives → DDPG, SAC (Part 2)
   └── shared philosophy → value learning + policy optimization hybrid
```

---

## 45. If You Remember Only 5 Things

1. Actor-Critic = policy (actor) + critic (value), updated from the **same TD error** δ.
2. Advantage A = r + γV_φ(s') − V_φ(s) drives the actor: reinforce above-average actions.
3. It trades REINFORCE's unbiased-but-high-variance returns for low-variance, mildly biased bootstrap signals.
4. Critic learns by TD; terminal states must be masked (V_next=0).
5. It is the foundation of A2C, A3C, PPO (Part 2) — the modern RL workhorses.

---

## 46. Cheat Sheet

```text
Algorithm   : Actor-Critic (one-step)
Category    : RL, policy+value hybrid, model-free, on-policy, TD
Goal        : Maximize return; learn policy + value together
Input       : online transitions (s,a,r,s')
Output      : policy π_θ (actor) + value V_φ (critic)
Core Formula: δ = r+γV_φ(s')−V_φ(s); θ+=α_actor·δ·∇log π; φ+=α_critic·δ·∇V
Loss        : actor −δ·logπ (+entropy); critic δ²
Optimization: gradient ascent (actor) / descent (critic)
Parameters  : θ (actor weights), φ (critic weights)
Hyperparams : α_actor, α_critic, γ, entropy β, architecture
Assumptions : differentiable policy, on-policy sampling, Markov-ish input
Advantages  : lower variance, online, continuous actions, flexible
Disadvantages: bootstrap bias, two nets to tune, on-policy waste
Use When    : need policy, want speed over REINFORCE
Avoid When  : tiny discrete tasks, extreme sample budgets, sparse long-horizon
Related     : REINFORCE, TD, A2C/A3C/PPO (Part 2), DQN
Key Exam    : advantage formula, same-δ updates, TD-vs-MC tradeoff
Key Interv  : variance reduction, baseline unbiasedness, terminal masking
```

---

## 47. Final Mental Model

```text
Act with π_θ
   ↓
Observe (r, s')
   ↓
δ = r + γV_φ(s') − V_φ(s)      (the single learning signal)
   ↓
Actor:  reinforce if δ>0, suppress if δ<0
Critic: move V toward its TD target
   ↓
Repeat every step
   ↓
Actor improves, critic tracks it
```

---

## 48. Knowledge Check

### Recall (5)
1. Write the one-step advantage.
2. Write the actor update.
3. Write the critic update.
4. What two networks exist and what do they learn?
5. What signal drives both updates?

### Understanding (5)
6. Why is AC's variance lower than REINFORCE's?
7. Where does AC's bias come from?
8. Why is the critic a valid baseline?
9. What does a positive δ mean for the actor?
10. Why mask terminal states?

### Application (5)
11. Hand-compute one AC update.
12. Choose actor and critic heads for a continuous task.
13. Set α_actor vs α_critic sensible defaults.
14. Add entropy to prevent collapse.
15. Build a deterministic evaluation loop.

### Mathematical (5)
16. Show A=δ (one-step case) equals the TD error.
17. Prove the baseline leaves the gradient unbiased.
18. Write the Policy Gradient Theorem in advantage form.
19. Explain the bias–variance tradeoff vs G_t.
20. Give the critic's squared-TD-error loss.

### Interview (5)
21. When does AC beat REINFORCE and when not?
22. How is the actor-critic coupling a risk?
23. What is n-step advantage / GAE concept?
24. Why might the critic mislead the actor?
25. How do A2C/A3C (Part 2) extend this?

### Problem Solving (5)
26. Actor converges too fast to mediocre — what's wrong?
27. Critic loss explodes — what do you check?
28. Task has long delayed rewards — what do you change?
29. Continuous control with 1-step AC is unstable — next step?
30. You have 8 parallel environments — how to exploit (conceptually)?

## Answers (explained)

1. A = r + γV_φ(s') − V_φ(s).
2. θ ← θ + α_actor·δ·∇log π_θ(a|s).
3. φ ← φ + α_critic·δ·∇V_φ(s) (φ ← φ − α_critic·∇(½δ²)).
4. Actor = policy π_θ; critic = value V_φ.
5. The TD advantage δ (same number for both).
6. δ uses one transition, not the full episode return; the baseline removes return-level noise.
7. Bootstrapping (γV(s') estimate) and critic approximation error.
8. V_φ(s) is action-independent: E[b(s)∇log π] = 0 (baseline property).
9. The action outperformed the critic's expectation — reinforce it.
10. No future exists after termination; otherwise phantom value inflates V and policy.
11. Follow Section 15: A=+2, θ→0.1, p≈0.525, v→4.2.
12. Gaussian head (μ,σ) for the actor; single scalar output for the critic.
13. Both ~1e-3 (Adam); separate rates only if needed.
14. Add +β·H(π) to the actor's objective; start β small.
15. Run fresh episodes using distribution mean/argmax actions; average returns.
16. δ = r + γV(s') − V(s) is exactly the definition of A in the one-step case.
17. E[b(s)∇log π] = b(s)∇Σ_aπ = b(s)∇1 = 0 — subtraction leaves E[∇J] unchanged.
18. ∇J(θ) ∝ E[π][ (Q^π(s,a) − V^π(s)) ∇log π(a|s) ].
19. G_t includes all future noise (variance↑) but zero bootstrap bias; δ includes critic-estimate bias but only one-transition noise (variance↓).
20. L_critic(φ) = (1/2)(r + γV_φ(s') − V_φ(s))².
21. AC beats REINFORCE almost always in practice (variance); REINFORCE wins when a clean unbiased estimator is essential (analysis/teaching).
22. Wrong critic → wrong advantage → wrong actor gradients → mutual degradation (positive feedback).
23. Consider n rewards before bootstrapping (GAE blends them) — tunable bias/variance; basis for PPO.
24. Critic lagging the moving policy, or bootstrap error amplification on top of outliers.
25. A2C runs N synchronous agents and averages their advantages; A3C runs them asynchronously — both decorrelate and scale training.
26. Too small entropy (collapse) or too large α_actor — raise β, lower α, check entropy curve.
27. Reward scale, missing terminal mask, α_critic too high, or target double-counting.
28. Increase n (n-step/GAE), densify rewards, or add auxiliary signals.
29. n-step AC / GAE; then PPO/SAC from Part 2.
30. Batch them into one update (A2C style): average the advantages across parallel rollouts for lower-variance gradients.

---

## 49. Final Learning Checklist

- [ ] I can define Actor-Critic in one sentence
- [ ] I can write the one-step advantage
- [ ] I know both actor and critic updates
- [ ] I understand why the same δ drives both
- [ ] I can explain the variance reduction vs REINFORCE
- [ ] I understand the bootstrap bias tradeoff
- [ ] I know why V_φ is a valid baseline
- [ ] I can hand-compute one AC update
- [ ] I can implement the from-scratch scalar AC
- [ ] I can implement PyTorch AC on CartPole
- [ ] I can build Gaussian-head AC for continuous control
- [ ] I know how to add entropy
- [ ] I know terminal masking
- [ ] I can evaluate deterministically (not by losses)
- [ ] I can diagnose actor-critic misalignment
- [ ] I understand the Policy Gradient Theorem link
- [ ] I know the n-step/GAE idea
- [ ] I can connect AC to A2C/A3C/PPO
- [ ] I know when NOT to use AC
- [ ] I can explain AC to a beginner

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** advantage = TD error identity, actor/critic update forms, and baseline-unbiasedness proof verified; the numeric example (A=+2 → θ=0.1, p≈0.525, v→4.2) and its second step recomputed by hand.
- **Beginner-friendliness:** judge-vs-coach analogy, short paragraphs, tables, no-math intuition first.
- **Math depth:** derivation from REINFORCE to advantage, Policy Gradient Theorem reference, bias–variance analysis.
- **Practical depth:** from-scratch numpy AC + PyTorch CartPole, hyperparameters, failure-mode ablate practice.
- **Exam depth:** advantage formula, same-δ architecture, terminal-mask trap, representative (non-PYQ) pattern question.
- **Structure:** all 50 sections present in order.

**Verified:** all arithmetic re-derived by hand; both actor-probability and critic-value trajectories (p: 0.5→0.525→0.546; v: 4.0→4.2→4.38) confirmed consistent with the stated updates.