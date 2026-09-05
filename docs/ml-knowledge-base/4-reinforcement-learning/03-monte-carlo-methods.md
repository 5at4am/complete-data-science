# 03. Monte Carlo Methods

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Monte Carlo Methods (for RL — prediction & control) |
| Category | Reinforcement Learning |
| Type | Value-based, Model-free, On-policy or Off-policy (MC with exploring starts / importance sampling) |
| Parametric / Non-parametric | Non-parametric (tabular value estimates) |
| Generative / Discriminative | N/A (decision-making) |
| Main Objective | Estimate V(s) or Q(s,a) by averaging the **full-episode returns** G_t — no bootstrapping |
| Input | Complete episodes: sequences of (state, action, reward) |
| Output | Value estimates V(s)/Q(s,a), and a policy derived from them |
| Core Idea | Sample complete episodes, compute the total discounted return per state (or state–action), then average — the mean of many samples estimates the expectation |
| Typical Use Cases | Episodic tasks, games with clear end states, evaluation of a known policy, first-visit vs every-visit estimation, policy construction from average returns |

---

## 02. One-Line Definition

### Beginner Definition
Monte Carlo methods learn the value of a situation by playing many complete games and simply **averaging** how much total reward was collected from that situation.

### Technical Definition
Monte Carlo (MC) methods estimate value functions by averaging the complete, discounted returns observed at the end of full episodes; because they use only real returns (never a bootstrapped estimate), they are **unbiased** estimators of the true value under the sampling policy, at the cost of **high variance**.

---

## 03. Intuition

A student wants to know how much money she makes on average per trip when she drives to work. Every day she notes the route she took, then at the end of the trip she records the day's **total expenses** and attributes it to that route.

Over a month, she has many trips for each route. Her estimate for any route is simply: *total money across all trips that used that route ÷ number of such trips* — the average.

That is the whole idea. In RL:
- A "trip" is an **episode** (a complete run of the environment).
- The "total expenses" is the **return** G_t (sum of discounted rewards until the end).
- The route is the state, or the (state, action) pair.

Nothing is predicted mid-way; you wait until the episode ends, compute how good things actually turned out, then update the estimate. No guessing, no bootstrapping — only **measured outcomes**.

Step-by-step reasoning:
1. Play a full episode, remembering every (state, action, reward).
2. At the end, walk backward through the episode and compute the return for each step (this is efficient via recursion).
3. For each visited state (or state–action pair), add the return to a running average.
4. Repeat for many episodes. The average converges to the true value by the Law of Large Numbers.

---

## 04. Problem It Solves

**Problem:** TD methods (Q-learning, SARSA) update using an *estimate within an estimate* (bootstrapping), which is biased. Sometimes you want the true, measured value of a policy — no assumptions about future estimates.

**Example:** Evaluate a fixed playing policy for a card game. Each hand (episode) has a known final result (win/loss). We want, per game state, "how much do I win on average from here?" The honest answer is: play thousands of hands, and average the eventual outcomes for each state.

**What we want:** Estimation of V(s) or Q(s,a) using only real experienced returns, so estimates are unbiased and simple.

**Why MC helps:**
- No model needed (only episodes).
- No bootstrapping → unbiased target.
- Update is trivial (incremental average).
- Works even when transitions are highly stochastic, because it averages many full outcomes.

**Small example:** A 2-step problem: start A, go to B (reward based), then terminal with reward. To value state A, MC waits for the end of each episode, computes G = r₁ + γ·r₂, and averages across episodes.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning
├── Unsupervised Learning
└── Reinforcement Learning
    ├── Model-Based
    └── Model-Free
        ├── Value-Based
        │   ├── MONTE CARLO              ← YOU ARE HERE (full episodes, no bootstrap)
        │   ├── Temporal Difference (TD) (one-step bootstrap)
        │   ├── SARSA / Q-Learning       (TD control)
        │   └── DQN
        └── Policy-Based (REINFORCE is itself a Monte-Carlo policy-gradient method)
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Episode | One complete run | S₀,A₀,R₁,S₁,...,S_T with terminal S_T |
| Return G_t | Total future reward from t | Gₜ = Σγᵏ rₜ₊ₖ₊₁ over the rest of the episode |
| Discount γ | How much future reward counts | γ ∈ [0,1] |
| Value V(s) | Average return from state s | E[Gₜ \| Sₜ=s] |
| Action-value Q(s,a) | Average return from (s,a) | E[Gₜ \| Sₜ=s, Aₜ=a] |
| First-visit MC | Update using first visit per episode | Only first occurrence of s counts in averaging |
| Every-visit MC | Update on every visit per episode | All occurrences of s count |
| Law of Large Numbers | Averages converge | As n→∞, sample mean → expectation |
| Unbiased estimate | Average targets the truth | E[MC estimate] = true value |
| High variance | Each sample differs a lot | Full-episode returns include all randomness |
| Exploring starts | Start each episode from random (s,a) | Ensures every pair visited for control |
| Policy evaluation | Estimating V/Q for a fixed policy | MC prediction |
| Policy improvement | Making the policy greedier | Change π to argmax of Q |
| Off-policy MC | Learn target policy from another behavior policy | Uses importance sampling weighting |

---

## 07. Input and Output

**Input (training):**
- A complete set of episodes, each a list of (sₜ, aₜ, rₜ₊₁, sₜ₊₁) ending at terminal state.
- Discount factor γ.

**Input (hyperparameters):**
- γ; whether first-visit or every-visit; (for control) ε schedule / exploring-starts; number of episodes.

**Output (training):**
- Estimated V(s) for all states (prediction), or Q(s,a) for all pairs (for control), plus the derived greedy/ε-greedy policy.

**Output (prediction / use):**
- For evaluation: the value estimates.
- For control: policy π(s) = argmax_a Q(s,a).

**Parameters learned:** the tabular value estimates.

**Hyperparameters:** γ, first-vs-every visit, ε (exploration), number of episodes.

---

## 08. Mathematical Foundation

The return for a complete episode:

```text
G_t = Σ_{k=0}^{T-t-1} γ^k · r_{t+k+1}
```

**The discount factor γ intuitively:** a reward now is worth more than the same reward later. In finite episodes T is finite; γ further down-weights far-future rewards and controls how "far-sighted" the agent is.

The value function is the expected return:

```text
V_π(s) = E_π[ G_t | S_t = s ]
```

**Incremental averaging formula** (equivalent to the mean but usable online):

```text
V(s) ← V(s) + (1/N(s)) · (G_t − V(s))
```
or, with step size α:
```text
V(s) ← V(s) + α·(G_t − V(s))
```

**Key property — no bootstrapping:** the target G_t is a *real observed* return, not a function of other estimates. That makes MC estimates **unbiased**.

**Why variance is high:** G_t sums reward over the entire random trajectory, so every stochastic transition and every stochastic reward contributes noise to one update.

**Required math concepts:** expectation, sampling, law of large numbers, variance of sums.

---

## 09. Core Formula

### The return (target)

```text
G_t = Σ_{k=0}^{T-t-1} γ^k · r_{t+k+1}
```

#### Meaning
Total discounted reward collected from time t until the episode ends.

#### Symbols
- `G_t` = return at time t
- `r_{t+k+1}` = reward k+1 steps after t
- `γ` = discount factor
- `T` = final time step of the episode

#### Intuition
"Add up everything earned from now until the end, shrinking future rewards by γ."

---

### The MC update

```text
V(s) ← V(s) + α · (G_t − V(s))        (or divide by visit count: 1/N(s))
```

#### Meaning
Move the value estimate toward the observed return by a fraction α (or by the exact average increment).

#### Symbols
- `V(s)` = current estimate for state s
- `G_t` = observed return from the episode
- `α` = step size (0 < α ≤ 1); for exact averaging use 1/N(s) where N(s) = visits to s

#### Intuition
Each episode provides one noisy "measurement" G_t of the true value; we average measurements across episodes.

#### Example (one update)
V(A)=0, α=0.5, an episode where G=2 from state A.
```text
V(A) = 0 + 0.5·(2 − 0) = 1.0
```
After a second episode with G=4:
```text
V(A) = 1.0 + 0.5·(4 − 1.0) = 2.5   (both-sample average would be 3; α-averaging approaches it)
```

---

### First-visit vs every-visit

```text
First-visit MC:  average G_t only at the FIRST time s appears in an episode
Every-visit MC:  average G_t at EVERY appearance of s in an episode
```

#### Meaning
Both estimate E[G|s]; first-visit uses independent samples, every-visit uses all samples (biased but usually fine).

#### Intuition
First-visit gives clean i.i.d. samples of the "first-encounter return"; every-visit squeezes more data from each episode.

---

## 10. Derivation

**Step 1 — Definition of the value:**

```text
V_π(s) = E_π[ G_t | S_t = s ]
```

**Step 2 — Estimate the expectation by sampling:**
We cannot integrate over trajectories; instead we sample N episodes and average:

```text
V_N(s) = (1/N) Σ_{i=1}^{N} G_t^{(i)}
```

**Step 3 — Why unbiased:**
E[V_N(s)] = (1/N)·Σ E[G_t^{(i)}] = E[G_t | S_t=s] = V_π(s) — the estimate is exactly V_π(s) in expectation.

**Step 4 — Convergence:**
By the **Law of Large Numbers**, V_N(s) → V_π(s) as N→∞. The update version (α=1/N(s)) implements this mean online without storing all returns.

**Note — no bootstrapping anywhere:** the target is a real episode return, never composed of other estimates. This is the defining property of Monte Carlo RL and the source of both its unbiasedness (good) and its variance (bad).

---

## 11. How the Algorithm Works

```text
Choose γ, α (or track visit counts)
    ↓
Initialize V (or Q) = 0, visits = 0
    ↓
Loop episodes (many):
    Generate a full episode under policy π: S₀,A₀,R₁,...,S_T
    ↓
    Walk backward (t = T-1 ... 0), accumulating G = G·γ + r_{t+1}
    ↓
    For each visited state (first-visit or every-visit):
        V(s) += α·(G − V(s))      [or increment running average]
    ↓
    (Control): update policy to ε-greedy over Q
    ↓
Repeat until values stabilize
    ↓
Final model: value estimates / Q-table
```

---

## 12. Training Process

**Pre-training:** Initialize V(s)=0 (evaluation) or Q(s,a)=0 (control).

**During training:**
- Generate complete episodes under the current policy.
- After the episode ends, compute returns backwards (efficient recursion).
- Update each visited state's estimate toward its observed return.

**What is learned:** expected return per state (V) or per state–action (Q) under the sampling policy.

**Changes per iteration (per episode):**
```text
G ← 0
for t from last-step back to 0:
    G ← γ·G + r_{t+1}
    V(s_t) ← V(s_t) + α·(G − V(s_t))     # (first-visit: only if s_t not seen earlier this episode)
```

**Stopping:** fixed number of episodes or value changes small.

**Final model contents:** table of value estimates (and the greedy/ε-greedy policy for control).

---

## 13. Objective Function / Loss Function

For **prediction**, MC minimizes (in expectation) the **mean squared error** between the estimate and the observed returns:

```text
L = E[ (G_t − V(s))² ]
```

**Interpretation:** across episodes, we want the estimate to be as close as possible to the real measured returns. Each episode's update is a noisy gradient step on this squared error.

For **control**, the objective is the same for Q(s,a), and the policy is then chosen greedily over Q — i.e., maximize the estimated action-values.

**Low loss** = estimates close to real returns (good). **High loss** = estimates far from measured outcomes (bad — possibly because few episodes).

**Training objective ≠ evaluation metric:** MC minimizes squared error to the *observed returns*; the evaluation metric is downstream performance (e.g., value-accuracy against a known model, or episode return of the derived policy).

---

## 14. Optimization

**Definition:** Find value estimates that equal (in expectation) the true expected returns.

**Method:** stochastic incremental averaging — each episode supplies one unbiased sample G_t; we move the estimate a little toward it.

**Step size α (or 1/N(s)):** controls how fast averages converge. α=1/N(s) gives exact mean; constant α adapts faster but never fully settles.

**Gradient view:** the update V(s) ← V(s) + α(G_t − V(s)) is stochastic gradient descent on the squared error (G_t − V(s))² with gradient −2(G_t − V(s)).

**Convergence:** by the Law of Large Numbers, averages converge to the true expectation provided enough episodes.

**Local/global:** there is a single true value; MC converges to it (no harmful local optima in the tabular case).

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified)

Three states A, B, C; terminal T. Deterministic chain: A → B → C → T. Rewards: A→B gives +1, B→C gives +2, C→T gives +3. γ = 0.5. (Small γ deliberately used to make the arithmetic clear.)

**One episode:** (A, a1, +1, B), (B, a2, +2, C), (C, a3, +3, T).

**Return at each state (backwards recursion):**
```text
G at C: return = Σ of remaining rewards.
    From C: r₃ = 3.     G_C = 3
    From B: G_B = r₂ + γ·G_C = 2 + 0.5·3 = 2 + 1.5 = 3.5
    From A: G_A = r₁ + γ·G_B = 1 + 0.5·3.5 = 1 + 1.75 = 2.75
```

Alternatively, directly:
```text
G_A = 1 + 0.5·2 + 0.5²·3 = 1 + 1 + 0.75 = 2.75  ✓
G_B = 2 + 0.5·3 = 3.5  ✓
G_C = 3  ✓
```

**Update with α = 0.5** (values initialized at 0):
```text
V(A) = 0 + 0.5·(2.75 − 0) = 1.375
V(B) = 0 + 0.5·(3.5 − 0)  = 1.75
V(C) = 0 + 0.5·(3 − 0)    = 1.5
```

**Second episode (same rewards):**
Returns identical: G_A=2.75, G_B=3.5, G_C=3.
```text
V(A) = 1.375 + 0.5·(2.75 − 1.375) = 1.375 + 0.5·1.375 = 2.0625
V(B) = 1.75 + 0.5·(3.5 − 1.75)    = 1.75 + 0.875 = 2.625
V(C) = 1.5 + 0.5·(3 − 1.5)        = 1.5 + 0.75 = 2.25
```

**True values** (deterministic chain):
```text
V(C) = 3
V(B) = 2 + 0.5·3 = 3.5
V(A) = 1 + 0.5·3.5 = 2.75
```

| Episode | V(A) | V(B) | V(C) |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 1 | 1.375 | 1.75 | 1.5 |
| 2 | 2.0625 | 2.625 | 2.25 |
| ∞ | 2.75 | 3.5 | 3.0 |

Hand-verified: recursion and direct summation agree; with exact averaging (α=1/N) the estimates converge exactly to the true values. MC is unbiased — that is visible here.

---

## 16. Visual Explanation

### Generating one episode and computing returns backward

```text
time   s₀ ──a₁──▶ s₁ ──a₂──▶ s₂ ──a₃──▶ T(terminal)
reward        r₁=+1      r₂=+2      r₃=+3

G₃(C) = 3
G₂(B) = 2 + γ·3 = 3.5        (γ = 0.5)
G₁(A) = 1 + γ·3.5 = 2.75

= compute from right to left, then update V for each visited state
```

### Averaging mechanism (value estimate stabilizes)

```text
V(A):  0 ─▶ 1.375 ─▶ 2.0625 ─▶ 2.3 ... ─▶ 2.75   (true value)
            ▲         ▲         ▲
          ep1       ep2       ep3 ...  (each: V ← V + α(G − V))
```

---

## 17. Algorithm / Pseudocode

```text
Monte Carlo prediction (first-visit):
1. Input: policy π, γ, episodes N, step size α
2. V(s) ← 0 for all s
3. For each episode:
     a. generate S₀,A₀,R₁,S₁,...,S_T under π
     b. G ← 0
     c. for t = T-1 down to 0:
          G ← γ·G + R_{t+1}
          if S_t not visited earlier in this episode (first-visit):
             V(S_t) ← V(S_t) + α·(G − V(S_t))
4. Return V

Monte Carlo control (ε-greedy, exploring starts):
1. Q(s,a) ← 0 for all (s,a)
2. For each episode:
     random start (exploring starts) → generate episode under ε-greedy policy
     compute returns backward
     update Q(s_t, a_t) with its return
     for each state: π(s) ← argmax_a Q(s,a)   (or set ε-greedy)
3. Return Q
```

---

## 18. From-Scratch Implementation

```python
import random

class MonteCarloAgent:
    def __init__(self, rows, cols, gamma=0.9, epsilon=0.1):
        self.rows, self.cols = rows, cols
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)]
        self.n_actions = len(self.actions)
        self.q = {}
        self.returns_sum = {}
        self.returns_count = {}
        for r in range(rows):
            for c in range(cols):
                self.q[(r,c)] = [0.0]*self.n_actions
                self.returns_sum[(r,c)] = [0.0]*self.n_actions
                self.returns_count[(r,c)] = [0]*self.n_actions

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return max(range(self.n_actions), key=lambda a: self.q[state][a])

    def step(self, state, action):
        dr, dc = self.actions[action]
        nr, nc = state[0]+dr, state[1]+dc
        nxt = (nr, nc) if self.in_bounds(nr, nc) else state
        reward, done = 0.0, False
        if nxt == self.goal:
            reward, done = 1.0, True
        elif nxt == self.pit:
            reward, done = -1.0, True
        return nxt, reward, done

    def play_episode(self):
        episode = []
        s = (0, 0)
        done = False
        while not done:
            a = self.choose_action(s)
            s2, r, done = self.step(s, a)
            episode.append((s, a, r))
            s = s2
        return episode

    def train(self, episodes):
        for _ in range(episodes):
            episode = self.play_episode()
            G = 0.0
            seen = set()
            for s, a, r in reversed(episode):
                G = self.gamma*G + r
                if (s, a) in seen:
                    continue
                seen.add((s, a))
                self.returns_sum[s][a] += G
                self.returns_count[s][a] += 1
                self.q[s][a] = self.returns_sum[s][a] / self.returns_count[s][a]

if __name__ == "__main__":
    agent = MonteCarloAgent(3, 3)
    agent.goal = (2, 2)
    agent.pit = (2, 1)
    agent.train(2000)
    for state in agent.q:
        print(state, [round(v,2) for v in agent.q[state]])
```

---

## 19. Code Explanation

```text
Line:  episode.append((s, a, r))
   What: records the full trajectory
   Why: MC needs the whole episode before any update
   Math: defines the episode for return computation

Line:  for s, a, r in reversed(episode):
   What: walks the episode backward
   Why: enables O(1) return recursion G = γG + r
   Math: G_t = r_{t+1} + γ·G_{t+1}

Line:  if (s, a) in seen: continue
   What: first-visit MC for state–action pairs
   Why: clean independent samples
   Math: only the first occurrence per episode counts

Line:  self.q[s][a] = sum/count
   What: exact incremental average
   Why: by LLN the mean converges to the expectation
   Math: Q(s,a) ← mean of observed returns
```

---

## 20. Library Implementation

No single library implements "MC RL" for you; the pattern is: run episodes, then average. Gymnasium supplies the episodes; numpy the arithmetic.

```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma = 0.9

returns_sum = np.zeros((n_states, n_actions))
returns_count = np.zeros((n_states, n_actions), dtype=int)
Q = np.zeros((n_states, n_actions))

def run_episode(epsilon=0.1):
    episode = []
    s, _ = env.reset()
    done = False
    while not done:
        if np.random.random() < epsilon:
            a = env.action_space.sample()
        else:
            a = np.argmax(Q[s])
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        episode.append((s, a, r))
        s = s2
    return episode

for ep in range(2000):
    episode = run_episode()
    G = 0.0
    seen = set()
    for s, a, r in reversed(episode):
        G = gamma*G + r
        if (s, a) in seen:
            continue
        seen.add((s, a))
        returns_sum[s, a] += G
        returns_count[s, a] += 1
        Q[s, a] = returns_sum[s, a] / returns_count[s, a]

# Evaluate greedy policy
success = 0
for _ in range(500):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.argmax(Q[s])
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        if r > 0:
            success += 1
print("MC success rate (greedy):", success/500)
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Discount γ | Future reward weight | Low → myopic; high → far-sighted | 0.9–0.99 |
| Step size α | Update weight (or exact 1/N) | α=1/N exact mean; α<1 adapts | 1/N or small constant |
| First vs every visit | Which appearances update | First-visit = i.i.d. samples | First-visit preferred |
| Episodes N | Number of full runs | More → lower variance | Until estimates stabilize |
| ε (control) | Exploration probability | Governs (s,a) coverage | 0.1–0.3, or exploring starts |
| Exploring starts | Random-start episodes | Guarantees pair coverage | When ε alone insufficient |

**Too high / too low:** too few episodes → high variance estimates; constant α too large → never settle; α too small → slow.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- V(s) estimates, Q(s,a) estimates — from averaging returns.

### Hyperparameters (chosen)
- γ, α (or exact averaging), first-vs-every-visit, episodes, ε (control).

---

## 23. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Episodic tasks | Episodes must end | MC needs the full return | Check tasks terminate | Use TD/λ methods for continuing tasks (discounted/avg reward) |
| Full observability of episode | Can record (s,a,r) | Updates need them | Logging | Use off-policy/other estimators |
| Ergodicity / sufficient sampling | Every state reachable | Averages need samples | Track visits | Increase exploration / exploring starts |
| Policy fixed during evaluation | π constant | Estimates are for π | Verify | Re-run per policy |

---

## 24. Data Requirements

- **Data type:** episodes — sequences of (state, action, reward); generated by the agent.
- **State/action spaces:** discrete for tabular MC; continuous possible only with function approximation (e.g., MC policy gradient).
- **Rewards:** scalar; episodic return must be finite (or γ<1).
- **Termination requirement:** must be able to complete episodes.
- **Sample count:** needs many episodes per state to reduce variance.
- **Missing/outliers/scaling:** N/A (decision-making setting, no feature matrix).

---

## 25. Feature Scaling

**Unnecessary** for tabular MC — no features, only state indices and reward sums. (Function-approximation variants with neural nets would need input standardization, as usual for deep learning.)

---

## 26. Evaluation Metrics

| Metric | Definition | Interpretation | Use When | Avoid When |
|---|---|---|---|---|
| Mean squared error vs true V | (1/n)Σ(V̂−V)² | Estimation accuracy | Known-environment sanity checks | Model unknown |
| Episode return | Total (discounted) reward | Policy performance | Comparing policies | Want fine-grained credit |
| Success rate | Fraction of episodes solved | Task-completion probability | Goal tasks | Non-goal tasks |
| Variability of returns | Variance/VAR(G) | Confidence in estimate | Reporting precisions | — |

**Important:** MC's training objective is minimizing error vs *observed returns*; the *evaluation metric* is end performance (return/success). A low MC "loss" doesn't by itself guarantee a good decision policy.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Unbiased estimates | Averages target the true value with no bootstrap bias |
| Simple & intuitive | "play, measure, average" is trivial to implement |
| Model-free | No dynamics table needed |
| Works on any episodic task | No bootstrapping constraints |
| Clean samples | First-visit returns are independent |
| Good baseline | TD methods are often validated against MC |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| High variance | Needs many episodes to average out trajectory noise |
| Requires full episodes | Cannot update mid-episode; slow feedback |
| Episodic tasks only | Continuing tasks need TD/λ |
| Inefficient with long episodes | Return sums many noisy rewards |
| Tabular scaling issues | Big spaces need other methods |
| Off-policy corrections are complex | Importance sampling adds variance |

---

## 29. When to Use

✓ Episodic tasks with clear ends.
✓ You want unbiased value estimates, not bootstrapped ones.
✓ A full episode is cheap to simulate (sample-inefficiency acceptable).
✓ Evaluating the value of a **fixed** policy accurately.
✓ Teaching/testing baseline concepts before TD.
✓ As the inner mechanism of policy gradients (REINFORCE uses MC returns).

---

## 30. When NOT to Use

✗ Continuing / non-terminating tasks.
✗ Limited environment calls (MC consumes full episodes per update).
✗ You need fast updates mid-episode (use TD).
✗ Long episodes where most updates add little.
✗ When bootstrap bias is acceptable and you want speed (TD wins).

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Evaluating a game-playing policy | game episodes | MC prediction | Win-rate estimate per position |
| Backgammon gameplay training | dice/board states | MC control | Piece-move policy |
| Poker-hand evaluation | dealt-hand episodes | MC value estimation | Hand strength estimate |
| Fleet routing evaluation | route episodes | MC evaluation | Expected route cost per state |
| Policy comparison studies | batch of episodes | MC prediction | Ranked policy values |

---

## 32. Failure Cases

- **Variance blow-up:** stochastic environments → returns vary hugely → estimates noisy for many episodes.
- **Termination failure:** non-terminating episodes → G_t never computed.
- **Rare-state failure:** states seen once or never → unreliable or missing values.
- **Exploration collapse (control):** greedy policy stops exploring → never updates uncovered pairs.
- **Importance-sampling explosion:** off-policy MC weights can blow up ratio variance.

---

## 33. Overfitting and Underfitting

- **Underfitting (too few samples):** few episodes → estimates far from truth (high variance residuals). Equivalent to "underfitting" the true value function.
- **Overfitting (by sample mean quirks):** with very few episodes, averages can reflect peculiar outcomes (e.g., one lucky long-lived path) — the estimate "overfits" a small batch of returns.

**Balance:** tabular MC converges to the true mean with enough episodes; watch estimate-vs-sample-size plots. In function-approximation MC, overfitting is a real generalization concern (too many parameters per state).

---

## 34. Bias-Variance Perspective

- **Bias:** MC estimates are **unbiased** — E[G_t] = V_π(s). No bootstrap contamination.
- **Variance:** **high** — each return includes every stochastic event in the episode (the most variance of any RL estimator here).
- **Tradeoff:** MC sits at one end of the bias–variance axis (zero bias, max variance). TD sits at the other end (small bias from bootstrapping, low variance). SARSA/Q-learning are in the middle. This axis is THE key way to connect MC, TD, and DQN.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| MC prediction | Average full-episode returns | Unbiased, simple | High variance, episodic only | Offline policy evaluation |
| TD(0) | One-step bootstrap | Low variance, online | Biased | Online prediction |
| Q-Learning | TD + max | Off-policy optimum | Bias, max-bias | Discrete MDP control |
| SARSA | TD + actual action | On-policy safety | Suboptimal w/ exploration | Hazard-adjacent tasks |
| DQN | Deep TD | Scales | Divergence risk | High-dim states |

---

## 36. Algorithm Selection Guide

```text
Want to estimate values?
├── Episodic, can afford full runs, want unbiased → MONTE CARLO
├── Online, need quick updates → TD(0)
└── Mix of both → TD(λ) / eligibility traces

Want control?
├── Episodic, small discrete → MC control (exploring starts)
└── Continuous/huge → Deep RL (DQN / policy gradient)
```

---

## 37. Common Mistakes

```text
❌ Updating V mid-episode with a partial return
Why wrong: MC target must be the COMPLETE return.
Correct: wait until the terminal state, then compute returns backward.

❌ Forgetting to discount (γ=0 implicitly)
Why wrong: gamma shapes what "total reward" means.
Correct: G = γ·G + r, walking backward.

❌ Using every-visit but claiming independence
Why wrong: every-visit samples within an episode are correlated.
Correct: prefer first-visit; know why you use every-visit.

❌ Handling terminal as a state (V(T)≠0)
Why wrong: terminal has no future return by definition.
Correct: never update V for terminal states; G stops there.

❌ Expecting quick convergence on a stochastic task
Why wrong: MC variance is high.
Correct: generate MANY episodes; monitor return variance.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is a Monte Carlo method in RL?**
A: Estimating value functions by averaging complete, observed episode returns.

**Q2. What is the return G_t?**
A: The total discounted reward from time t until the episode ends: Σγᵏr₍ₜ₊ₖ₊₁₎.

**Q3. First-visit vs every-visit MC?**
A: First-visit updates only the first occurrence of a state per episode; every-visit updates every occurrence.

### Intermediate
**Q4. Why is MC unbiased but high variance?**
A: It uses the true realized return (no bootstrap → unbiased), but that return contains the noise of the whole trajectory (→ high variance).

**Q5. Why must MC wait for the episode to end?**
A: The update target is the complete return; only known at termination.

**Q6. How do MC and TD differ?**
A: MC = full-episode real returns, no bootstrap, unbiased/high variance; TD = one-step bootstrap, biased/low variance, can update online.

### Advanced
**Q7. Can MC be used for control directly?**
A: Yes — MC control with exploring starts and ε-greedy improvement (policy evaluation via MC + policy improvement).

**Q8. What are the theoretical convergence guarantees?**
A: Law of Large Numbers for prediction; for control, guaranteed (under GAE-style conditions of exploring starts + stochastic approximation) for tabular methods.

**Q9. How do you do off-policy MC?**
A: Weight each return by the importance-sampling ratio Π(π_target/π_behavior) over the episode — correct but variance-explosive.

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
```text
G_t = Σ_{k=0}^{T-t-1} γ^k r_{t+k+1}
V(s) ← V(s) + α(G_t − V(s))      (or 1/N(s))
unbiased: E[V_avg(s)] = V_π(s)
convergence: Law of Large Numbers
```

**Concepts likely tested:**
- Definition of return and discounting
- First-visit vs every-visit
- Why MC is unbiased; why variance is high
- MC vs TD bootstrap difference
- Requirement that episodes terminate

> **Representative pattern question (NOT a past GATE PYQ):** "An episode is S₁→(r=2)→S₂→(r=3)→T with γ=0.5. What is G at S₁?" → G = 2 + 0.5·3 = 3.5; at S₂, G = 3.

**Common traps:**
- Treating terminal state as having a value.
- Thinking MC updates happen mid-episode.
- Missing γ in the return sum.
- Claiming MC is biased.

---

## 40. Coding Practice

**Level 1 — Basic:** Implement return computation by backward recursion on a hard-coded episode.
**Level 2 — Prediction:** First-visit MC prediction on a 3-state chain; compare to true values.
**Level 3 — Control:** MC control with exploring starts on a gridworld; print Q.
**Level 4 — Every-visit:** Re-run every-visit; compare estimates.
**Level 5 — Variance study:** Plot estimate-vs-episodes for a stochastic world; show noise decays slowly.
**Level 6 — FrozenLake:** MC control on FrozenLake from scratch.
**Level 7 — Real-world case study:** Evaluate two policies for a navigation simulator with MC, report value estimates and confidence, then choose the better policy.

---

## 41. Practical ML Workflow

```text
Problem → episodic? define γ, policy to evaluate
   ↓
Environment → simulator / gymnasium episodic task
   ↓
Collect → generate MANY episodes under policy π
   ↓
Compute returns → backward recursion, apply discount
   ↓
Average → first-visit (or every-visit) means
   ↓
Evaluate → error vs true values (if known), return stats
   ↓
Tune → more episodes, γ, first/visit choice
   ↓
Deploy → value estimates / policy for control
   ↓
Monitor → re-estimate when environment changes
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Episode generation | O(T) time | T = episode length |
| Return computation | O(T) | One backward pass |
| Update per episode | O(\|visited states\|) | First-visit subset |
| Space | O(\|S\|) or O(\|S\|·\|A\|) | V or Q tables |
| Total training | O(number of episodes × T) | Linear in experience |
| Variance reduction | Slower than linear in episodes | Needs many runs in noisy worlds |

---

## 43. Advanced Concepts

- **Importance sampling** for off-policy MC (ratio weighting) and its variance problems.
- **Weighted importance sampling** (bounded variance correction).
- **Eligibility traces (TD(λ))** blend MC and TD — MC is the λ=1 extreme.
- **MC policy gradient (REINFORCE)** — file 06 uses MC returns as the gradient direction.
- **Monte Carlo in planning:** model-free MC can be paired with a learned model (MC tree search).

---

## 44. Connections to Other Algorithms

```text
Monte Carlo Methods
   │
   ├── prediction baseline → TD(0) (file 04) (bias–variance tradeoff)
   ├── control counterpart → Q-learning / SARSA (TD control versions)
   ├── deep relative → Value-based DQN (file 05) uses TD, not MC
   ├── policy version → REINFORCE (file 06) uses MC returns
   ├── generalization → Eligibility traces (TD(λ))
   └── shares philosophy → Actor-Critic baselines reduce MC variance (file 07)
```

---

## 45. If You Remember Only 5 Things

1. MC estimates values by averaging **complete episode returns** — no bootstrapping.
2. G_t = Σ γᵏ rₜ₊ₖ₊₁, computed efficiently backward as G ← γ·G + r.
3. MC is **unbiased** (targets the true expectation) but **high variance**.
4. First-visit keeps samples independent; every-visit reuses data.
5. MC requires terminating episodes; for control use exploring starts / ε-greedy.

---

## 46. Cheat Sheet

```text
Algorithm   : Monte Carlo (RL)
Category    : Reinforcement Learning (model-free, episodic)
Goal        : Estimate V(s)/Q(s,a) by averaging full returns
Input       : complete episodes (s,a,r) sequences
Output      : value estimates, optional greedy policy
Core Formula: V(s) ← V(s) + α(G_t − V(s)), G_t = Σγᵏ r
Loss        : MSE vs observed returns
Optimization: incremental averaging
Parameters  : value table entries
Hyperparams : γ, α(1/N), first/every-visit, episodes, ε
Assumptions : episodic, policy-fixed for evaluation, sufficient sampling
Advantages  : unbiased, simple, model-free
Disadvantages: high variance, episode-bound, tabular scaling
Use When    : episodic evaluation, cheap simulations
Avoid When  : continuing tasks, limited env calls
Related     : TD(0), Q-learning, SARSA, REINFORCE, TD(λ)
Key Exam    : G_t, unbiased vs variance, first/every visit
Key Interv  : MC vs TD, why unbiased, control via exploring starts
```

---

## 47. Final Mental Model

```text
Run full episodes under π
   ↓
Compute returns backward: G ← γ·G + r
   ↓
Average returns per state/(s,a)
   ↓
(Control) act ε-greedily over averages
   ↓
Re-run many times → estimates → true values (LLN)
```

---

## 48. Knowledge Check

### Recall (5)
1. Define the MC return G_t.
2. First-visit vs every-visit.
3. Write the incremental average update.
4. Why is MC unbiased?
5. Why does MC need terminating episodes?

### Understanding (5)
6. Why is MC variance high?
7. Contrast MC with TD bootstrapping.
8. What is the role of γ in MC?
9. How does MC compute returns in O(T)?
10. Why exploring starts for MC control?

### Application (5)
11. Compute returns and update V for a 3-state episode.
12. Set up MC evaluation of a fixed policy.
13. Choose first-visit vs every-visit.
14. Plan MC control with ε-greedy on a grid.
15. Diagnose slow convergence — what to change?

### Mathematical (5)
16. Prove unbiasedness (E[mean]=V_π).
17. Explain LLN convergence.
18. Show G recursion equals direct sum.
19. Relate MC variance to episode length.
20. Derive α=1/N equivalent to the sample mean.

### Interview (5)
21. When MC beats TD?
22. When TD beats MC?
23. What is importance sampling for off-policy MC?
24. How does λ blend MC and TD?
25. Why is REINFORCE called a "Monte Carlo" policy-gradient?

### Problem Solving (5)
26. Environment returns vary wildly — what to do?
27. States visited once — reliable estimates?
28. Episode won't end — what now?
29. Control never visits a dangerous state — danger of bad Q.
30. Comparing two policies on fixed budget — which metric?

## Answers (explained)

1. G_t = Σ_{k=0}^{T−t−1} γᵏ·rₜ₊ₖ₊₁.
2. First-visit uses only the first occurrence; every-visit uses all.
3. V(s) ← V(s) + α(G_t − V(s)), exactly 1/N for the mean.
4. E[G_t] = V_π(s), because the mean target is the expectation.
5. The target G_t is defined only at episode end.
6. Each return includes the whole trajectory's randomness at once.
7. TD updates mid-episode with a bootstrap; MC waits for the full real return.
8. γ down-weights far-future rewards and makes far returns finite.
9. Backward recursion G ← γ·G + r reuses the next state's G.
10. Guarantees every (s,a) has some chance to appear for its Q estimate.
11. Use recursion (Section 15): G_C=3→G_B=3.5→G_A=2.75; then V += α(G−V).
12. Fix π, run many episodes, average returns per state only (no policy change).
13. First-visit for clean i.i.d. samples; every-visit to reuse scarce data.
14. Random or forced starts + ε-greedy action selection, average returns per (s,a), then π=argmax.
15. Add episodes; possibly lower α/use exact averaging; reduce γ-lag effects.
16. Linearity of expectation over independent episodes.
17. Sample mean → expectation a.s. as N→∞ for i.i.d. samples.
18. G_{t} = r_{t+1}+γG_{t+1} expands to the direct sum by induction.
19. Var(G_t) = Σγ²ᵏVar(r) grows with episode length T → longer episodes, more variance.
20. Recurrence V ← V + (1/N)(G−V) reconstructs (ΣG)/N after N samples.
21. Unbiased estimates needed, episodic, cheap simulations.
22. Online updates, continuing tasks, low-variance requirement.
23. Weight returns by Π π_target(a|s)/π_behavior(a|s) across the episode.
24. TD(λ) interpolates: λ=0 → TD(0); λ=1 → MC.
25. Its gradient estimate uses the full MC return G_t of an episode.
26. Run more episodes; reduce γ if appropriate; use baseline subtraction conceptually.
27. No — small-sample estimates are unreliable; gather more visits or pool states.
28. Use γ and TD-style estimates, or truncate with finite-horizon return.
29. Danger: optimistic Q for unvisited pairs → aggressive policy near hazards; force exploring starts.
30. Use mean return with standard error, or success rate; report confidence intervals.

---

## 49. Final Learning Checklist

- [ ] I can define MC RL in one sentence
- [ ] I can compute G_t backward-recursively
- [ ] I know first-visit vs every-visit
- [ ] I understand unbiasedness
- [ ] I understand the high-variance issue
- [ ] I know why episodes must terminate
- [ ] I can implement MC prediction from scratch
- [ ] I can implement MC control (exploring starts)
- [ ] I can run MC on FrozenLake
- [ ] I can evaluate a fixed policy with MC
- [ ] I know the LLN convergence argument
- [ ] I can contrast MC with TD(0)
- [ ] I know how γ affects G
- [ ] I know the incremental-average trick
- [ ] I understand importance sampling basics
- [ ] I know TD(λ) interpolation
- [ ] I can choose when MC beats TD
- [ ] I can report value estimates with uncertainty
- [ ] I can connect MC to REINFORCE
- [ ] I can explain MC to a beginner

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** return recursion verified against direct summation (hand-computed); true-value comparison table confirmed; unbiasedness argument checked.
- **Beginner-friendliness:** driving-to-work analogy, short paragraphs, tables.
- **Math depth:** expectation, LLN, bias–variance analysis of MC, worked numerical example.
- **Practical depth:** full from-scratch MC agent + library-style FrozenLake, workflow, complexity table.
- **Exam depth:** return/γ traps, representative (non-PYQ) pattern question.
- **Structure:** all 50 sections present in order.

**Verified:** numerical example re-derived by hand (G_A=2.75, G_B=3.5, G_C=3 under γ=0.5); convergence to true values confirmed.