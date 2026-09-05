# 02. SARSA

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | SARSA (State-Action-Reward-State-Action) |
| Category | Reinforcement Learning |
| Type | Value-based, Model-free, On-policy, Temporal-Difference (TD) control |
| Parametric / Non-parametric | Non-parametric (tabular Q-table) |
| Generative / Discriminative | N/A (decision-making) |
| Main Objective | Learn the action-value function Q(s,a) for the policy actually being followed |
| Input | State / observation S and action A |
| Output | Action to take, and a learned Q-table for the behavior policy |
| Core Idea | Update Q(s,a) toward the TD target r + γ·Q(s',a') where a' is the actual next action chosen — making it strictly on-policy |
| Typical Use Cases | Discrete control with risky exploration, cliff-walking, safe navigation, on-policy safety tasks (e.g., driving, robots near hazards) |

---

## 02. One-Line Definition

### Beginner Definition
SARSA is like Q-learning but the agent updates its score using the move it **actually** makes next, not the best move it could have made — so it learns the value of its own real, possibly clumsy, behavior.

### Technical Definition
SARSA is a model-free, on-policy, value-based TD control algorithm that learns Q(s,a) by updating toward the bootstrap target `r + γ·Q(s',a')`, where `a'` is the action actually taken in the next state under the current ε-greedy policy — therefore it learns the value of the behavior policy, not the optimal policy directly.

---

## 03. Intuition

Think of two students marking their own practice tests.

**Q-learning student** updates each answer by imagining: "I'll be perfect from here on." So when it reaches a state it asks *"what's the best I could ever do here?"* and uses the max. It learns the score of a hypothetical ideal player.

**SARSA student** updates by asking *"what would I actually do next, with my current knowledge and my current randomness?"* It uses the value of the move it really intends to make. So SARSA learns the score of its *own actual* (imperfect) play.

Why does this matter? If the agent is still exploring (acting randomly with ε-greedy), its real behavior is imperfect. SARSA's values reflect that — it learns "how well my current, still-sloppy strategy performs." This makes SARSA **safer**: it never assumes perfect play, so it won't, say, walk right next to a cliff just because a perfect player could; it accounts for the chance of slipping (exploring) into the cliff.

Step-by-step reasoning:
1. Remember the whole 5-tuple: the state, action, reward, next state, **and the next action**.
2. Use reward + value-of-next-state-given-next-action as the target.
3. Update the current Q-value toward it.
4. Make the "next" become the new "current" and continue.

---

## 04. Problem It Solves

**Problem:** Q-learning's `max` over next states assumes the agent will always act optimally from the next state. But during training the agent is still exploring — it may well take a *random* (bad) action there. If near hazards, Q-learning can plan a path that hugs a cliff, because in the update it pictures an ideal player avoiding the danger; the real exploring agent might then slip into it.

**Example:** Cliff-walking gridworld — a path right along a cliff is *shorter* and gives optimal return, but any exploratory slip drops the agent to the bottom. A Q-learning agent is more likely to hug the cliff (optimal in theory); a SARSA agent, because its values bake in the possibility of exploratory slips, learns the *safer* longer path along the top.

**What we want:** A method that learns the value of the policy we are actually following, so behavior during exploration is taken into account.

**Why SARSA helps:** It is **on-policy** — the TD target uses the actual next action. Its learned Q-values reflect the true behavior policy (including exploration), producing policies that are robust to the agent's own imperfect execution.

**Small example:** Two states A→B with one action each, terminal reward +1 at B. With high exploration, SARSA's estimate of A's value includes the possibility that at B the agent explores into a bad action. Q-learning's estimate assumes best action at B. The two learn *different* values even on the same data.

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
        │   ├── Monte Carlo
        │   ├── TD(0)
        │   ├── SARSA                        ← YOU ARE HERE (on-policy TD)
        │   ├── Q-Learning                   (off-policy TD)
        │   └── DQN                          (Q-learning + neural net)
        └── Policy-Based
            ├── REINFORCE
            └── Actor-Critic
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Agent | The decision maker | The learner maximizing cumulative reward |
| Environment | The world responding to actions | Provides next state & reward |
| State S | Current situation | Sₜ ∈ set of states |
| Action A | A move the agent can make | Aₜ ∈ A(sₜ) |
| Reward R | Immediate feedback | rₜ₊₁ after action aₜ |
| Policy π | Decision rule | Map state → action (or probabilities) |
| Return G_t | Total discounted future reward | Gₜ = Σγᵏrₜ₊ₖ₊₁ |
| Discount γ | Future reward weight | γ ∈ [0,1] |
| Q(s,a) | Value of action at state | Expected return from (s,a) |
| Episode | One full run | S₀A₀R₁S₁...S_T |
| Temporal difference | Partial-step learning | reward + estimate − old estimate |
| Bootstrapping | Estimate updates estimate | Uses Q(s',a') in the target |
| ε-greedy | Exploration/exploitation policy | With prob ε act randomly else greedily |
| On-policy | Learns the policy being followed | Uses actual next action a' |
| Off-policy | Learns a different policy | Uses max/greedy next action |
| TD target | The value the update moves toward | r + γ·Q(s',a') for SARSA |

---

## 07. Input and Output

**Input (training):**
- The environment (states, actions, rewards, transitions — all unknown to agent).
- At each step the agent sees state sₜ and reward rₜ.

**Input (hyperparameters):**
- α (learning rate), γ (discount), ε (exploration) + decay, number of episodes, max steps.

**Output (training):**
- A learned Q-table for the behavior policy actually followed.

**Output (prediction / use):**
- Policy π(s) = argmax_a Q(s,a) (the greedy policy over learned values).

**Parameters learned:** each Q(s,a) entry in the table.

**Hyperparameters:** α, γ, ε, ε-decay, episodes, max steps.

---

## 08. Mathematical Foundation

Like Q-learning, SARSA lives in a **Markov Decision Process (MDP)** and aims to maximize expected discounted return:

```text
G_t = Σ_{k=0}^{∞} γ^k r_{t+k+1}
```

The **discount factor γ** intuitively: a reward now is worth more than the same reward later — the agent cares about the future, but not infinitely far-future equally.

Under a policy π, the action-value:

```text
Q_π(s,a) = E_π[ G_t | S_t = s, A_t = a ]
```

The **Bellman equation for Q_π** (expectation form):

```text
Q_π(s,a) = Σ_{s',r} P(s',r|s,a) [ r + γ·Σ_{a'} π(a'|s') Q_π(s',a') ]
```

**Key difference from Q-learning:** the inner sum over a' uses `π(a'|s')` (the policy being followed), *not* `max`. That inner expectation is exactly the "expected value of the next action under π" — which SARSA estimates with the single sampled next action a'.

**Notation:**
- `π(a'|s')` = probability the behavior policy takes a' in s'
- `P(s',r|s,a)` = environment dynamics
- `Q_π(s,a)` = action-value under policy π
- `γ`, `G_t`, `r` as before

**Required math concepts:** expectation, probability, geometric series for convergence of G_t.

---

## 09. Core Formula

### The SARSA update

```text
Q(s,a) ← Q(s,a) + α·[ r + γ·Q(s',a') − Q(s,a) ]
```

#### Meaning
Move the current Q-value toward the TD target formed by the immediate reward plus the value of the *actual next action* a' in the next state.

#### Symbols
- `Q(s,a)` = current estimate
- `α` = learning rate, 0<α≤1
- `r` = reward received after taking a in s
- `γ` = discount factor
- `s'` = next state
- `a'` = the action actually chosen in s' (by ε-greedy)
- `Q(s',a')` = value of that chosen next action
- The bracketed term = TD target; its difference from Q(s,a) = TD error

#### Intuition
`GDP = r + γ·Q(s',a')` is the *expected* value of the current policy's behavior, estimated with one sample. We nudge Q toward it. Because a' is the *actual* next action, SARSA converges to the value of the policy being followed — that is what makes it **on-policy**.

#### Example (one update)
α=0.5, γ=0.9. At A take action 0 → r=0, reach B; at B the ε-greedy policy picks action 1, and Q(B,1)=2.0.
```text
target = 0 + 0.9·2.0 = 1.8
Q(A,0) = 0 + 0.5·(1.8 − 0) = 0.9
```

---

### The Bellman expectation equation for Q_π (background)

```text
Q_π(s,a) = E_{s',a'} [ r + γ·Q_π(s',a') ]
```

#### Meaning
The value of (s,a) under π equals the expected reward plus the discounted expected value of the policy's next action.

#### Intuition
SARSA is the *sampled* version of this equation: it replaces the expectation with a single observed (r, s', a').

---

## 10. Derivation

**Step 1 — Start from the Bellman equation for Q_π:**

```text
Q_π(s,a) = E_π[ r_{t+1} + γ·Q_π(S_{t+1}, A_{t+1}) | S_t = s, A_t = a ]
```

**Step 2 — Replace expectations with samples.**
We don't know the dynamics or the policy probabilities; SARSA uses one observed transition (s, a, r, s', a'):

```text
target = r + γ·Q(s',a')
```

**Step 3 — Incremental update toward the target:**

```text
Q(s,a) ← Q(s,a) + α( target − Q(s,a) )
```

**Step 4 — Fixed-point interpretation.**
SARSA performs stochastic approximation to the **policy-evaluation** Bellman equation for the current ε-greedy behavior policy π. Between updates to Q, the policy π (from ε-greedy over Q) is improved implicitly. Under standard conditions (all pairs visited, α decays appropriately, and the policy converges), SARSA converges to Q_π for a policy that is in some sense optimal among those robust to exploration. This on-policy fixed point differs from Q-learning's off-policy fixed point whenever the agent explores.

---

## 11. How the Algorithm Works

```text
Initialize Q table
    ↓
Pick initial state s and initial action a (ε-greedy)
    ↓
Loop each step until terminal:
    Take action a → observe reward r, next state s'
    Choose next action a' from s' using ε-greedy over Q
    Update: Q(s,a) += α( r + γ·Q(s',a') − Q(s,a) )
    s ← s' ;  a ← a'
    ↓
Decay ε per episode
Convergence
    ↓
Final model: Q-table
    ↓
Policy: π(s) = argmax_a Q(s,a)
```

**Note the structural difference from Q-learning:** SARSA must select a' *before* performing the update, because the target uses a'.

---

## 12. Training Process

**Pre-training:** Build Q-table, initialize to 0.

**During training:**
- Each episode, pick the first action a with ε-greedy.
- Each step: take a, observe (r, s'), pick next a' with ε-greedy, then update Q(s,a) using γ·Q(s',a').
- Move (s ← s', a ← a').
- Because the target uses a', SARSA learns the value of its *own* ε-greedy behavior.

**What is learned:** Q-values for the behavior policy.

**Changes per iteration (one step):**
```text
Q(s,a) ← Q(s,a) + α( r + γ·Q(s',a') − Q(s,a) )
```

**Stopping:** fixed episodes or Q-stabilization.

**Final model:** the Q-table.

---

## 13. Objective Function / Loss Function

SARSA implicitly minimizes the squared TD error between the current estimate and the on-policy target:

```text
L = (1/2)·( r + γ·Q(s',a') − Q(s,a) )²
```

**Interpretation:** like Q-learning, but the target uses the actual next action a', not the max. So the "label" reflects the policy currently being followed.

**Why chosen:** encodes the policy-evaluation Bellman equation for the behavior policy; minimizes the mismatch between Q and the true value of the actual policy.

**Training objective ≠ evaluation metric:** TD error is the update objective; the *evaluation* is greedy episode return / success rate, measured separately.

---

## 14. Optimization

**Definition:** Find Q satisfying the on-policy Bellman equation for the behavior policy ε-greedy.

**Method:** incremental stochastic updates (stochastic approximation), no closed form.

**Update:** move Q(s,a) partway toward r + γ·Q(s',a') with step size α (like gradient descent on the squared TD error).

**Learning rate α:** too large → oscillation; too small → slow.

**Convergence:** guarantees require all pairs visited infinitely often, αₜ with Σα=∞, Σα²<∞, and that the policy converges (a refinement condition). SARSA converges to Q_π (optimal within the constraint that exploration matters at the convergence point).

**Local/global optimum:** converges to the on-policy value of the learned policy; with ε→0 it approaches the optimal policy but retains a small exploratory bias (e.g., towards safer actions near hazards).

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified)

Two states A, B and terminal T. From A, one action "go" → B reward 0. From B, one action "go" → T reward +1. Only one action per state, so the next action a' is always "go". γ=0.9, α=0.5.

Initialize Q(A,go)=0, Q(B,go)=0.

**Episode 1:**
- Step 1: at A, a="go" → r=0, s'=B, next a'="go". target = 0 + 0.9·Q(B,go)=0+0.9·0=0.
  ```text
  Q(A,go) = 0 + 0.5·(0 − 0) = 0
  ```
- Step 2: at B, a="go" → r=+1, s'=T (terminal), a' = none (Q(terminal)=0). target=1+0.9·0=1.
  ```text
  Q(B,go) = 0 + 0.5·(1−0) = 0.5
  ```

**Episode 2:**
- at A: r=0, s'=B, a'="go". target = 0 + 0.9·Q(B,go) = 0.9·0.5 = 0.45.
  ```text
  Q(A,go) = 0 + 0.5·(0.45−0) = 0.225
  ```
- at B: r=1, s'=T. target=1.
  ```text
  Q(B,go) = 0.5 + 0.5·(1−0.5) = 0.75
  ```

**Result table** — identical arithmetic to Q-learning here because with only one action per state, `max` equals the actual next action:

| Episode | Q(A,go) | Q(B,go) |
|---|---|---|
| 0 | 0 | 0 |
| 1 | 0 | 0.5 |
| 2 | 0.225 | 0.75 |
| →∞ | 0.9 | 1.0 |

**Why SARSA differs from Q-learning in general:** SARSA and Q-learning coincide whenever the actual next action a' is the maximizing action. They diverge when ε-greedy *explores* at s' (a' is random, possibly not the max). In that case SARSA's target `γ·Q(s',a')` is lower than Q-learning's `γ·max Q(s',·)`, so SARSA's Q-values (and safe policy) are lower — exactly the "safe exploration" property.

Hand-verified: arithmetic confirmed; the single-action case is also used to show SARSA ≡ Q-learning under determinism in the greedy case.

---

## 16. Visual Explanation

### Agent–environment loop for SARSA (note the extra arrow: first pick a', then update)

```text
      choose a' first
   ┌────────────────────┐
   │                    ▼
   │              ┌───────────┐   take a, observe r, s'
   │              │ AGENT     │ ─────────────────────────▶ ENVIRONMENT
   │              │ Q-table   │  observe s'  ◀────────────
   │              │           │
   │              └───────────┘
   │   update uses r + γQ(s',a')   (a' = actual next action)
   └────────────────────────────┘
```

### Cliff-walking illustration (why SARSA is safer)

```text
S . . . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . . . .
. . . C C C C C C C C C C C C C .   C = cliff (fatal)
. . . . . . . . . . . . . . . . . .
                            G = goal

Q-Learning: tends to hug the cliff (shorter optimal path)
SARSA:      stays one row back (safer; accounts for exploration slips)
```

---

## 17. Algorithm / Pseudocode

```text
Input: states S, actions A(s), γ, α, ε
Output: Q-table

1. Q(s,a) ← 0 for all s,a
2. For each episode:
     s ← initial state
     a ← ε-greedy(s)
     Repeat until terminal or max-steps:
       take action a, observe r, s'
       a' ← ε-greedy(s')              # choose NEXT action first (key!)
       Q(s,a) ← Q(s,a) + α( r + γ·Q(s',a') − Q(s,a) )
       s ← s' ; a ← a'
     Decay ε
3. Return Q
4. Greedy policy: π(s) = argmax_a Q(s,a)
```

---

## 18. From-Scratch Implementation

```python
import random

class SARSAAgent:
    def __init__(self, rows, cols, alpha=0.5, gamma=0.9, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995):
        self.rows, self.cols = rows, cols
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.actions = [(0,1),(1,0),(0,-1),(-1,0)]
        self.n_actions = len(self.actions)
        self.q = {}
        for r in range(rows):
            for c in range(cols):
                self.q[(r,c)] = [0.0]*self.n_actions

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

    def train(self, episodes):
        for ep in range(episodes):
            s = (0, 0)
            a = self.choose_action(s)
            done = False
            while not done:
                s2, r, done = self.step(s, a)
                a2 = self.choose_action(s2) if not done else None
                future = self.q[s2][a2] if a2 is not None else 0.0
                self.q[s][a] += self.alpha*(r + self.gamma*future - self.q[s][a])
                s, a = s2, a2 if a2 is not None else a
            self.epsilon = max(self.epsilon_min, self.epsilon*self.epsilon_decay)

if __name__ == "__main__":
    agent = SARSAAgent(3, 3)
    agent.goal = (2, 2)
    agent.pit = (2, 1)
    agent.train(2000)
    for state in agent.q:
        print(state, [round(v,2) for v in agent.q[state]])
```

---

## 19. Code Explanation

```text
Line:  a = self.choose_action(s)            # before the loop
   What: pick the first action for the episode
   Why: SARSA needs an action before its first update
   Math: the (S,A) of the tuple

Line:  a2 = self.choose_action(s2)
   What: pick the NEXT action a' BEFORE updating
   Why: the target uses Q(s',a'), so a' must be known first
   Math: completes the (S,A,R,S',A') tuple

Line:  future = self.q[s2][a2] if a2 is not None else 0.0
   What: value of the actual next action (0 if terminal)
   Why: on-policy bootstrapping
   Math: Q(s',a') term, max absent → on-policy

Line:  self.q[s][a] += self.alpha*(r + self.gamma*future - self.q[s][a])
   What: the SARSA update
   Why: move toward on-policy target
   Math: Q(s,a) += α(r + γQ(s',a') − Q(s,a))
```

---

## 20. Library Implementation

```python
import gymnasium as gym
import numpy as np

env = gym.make("CliffWalking-v0")
n_states = env.observation_space.n
n_actions = env.action_space.n

alpha, gamma, epsilon, decays = 0.5, 0.9, 1.0, 0.995
Q = np.zeros((n_states, n_actions))
episodes = 4000
returns = []

def eps_greedy(s):
    if np.random.random() < epsilon:
        return env.action_space.sample()
    return np.argmax(Q[s])

for ep in range(episodes):
    s, _ = env.reset()
    a = eps_greedy(s)
    done = False
    G = 0.0
    while not done:
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        a2 = eps_greedy(s2) if not done else None
        future = Q[s2][a2] if a2 is not None else 0.0
        Q[s, a] += alpha*(r + gamma*future - Q[s, a])
        s, a = s2, a2 if a2 is not None else a
        G += r
    returns.append(G)
    epsilon = max(0.01, epsilon*decays)

print("Avg return last 500 eps (SARSA):", np.mean(returns[-500:]))

# Evaluate greedy policy
success = 0
for _ in range(1000):
    s, _ = env.reset()
    done = False
    while not done:
        a = np.argmax(Q[s])
        s, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        if r == -100:   # fell into cliff
            break
print("Avg greedy return:", np.mean([eval_episode(Q, env) for _ in range(1000)]))
```

(For brevity, `eval_episode` runs a greedy episode and returns total reward.)

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Learning rate α | Step size per update | High → unstable; low → slow | 0.1–0.5 |
| Discount γ | Future reward weight | Low → myopic; high → future-aware | 0.9–0.99 |
| Exploration ε | Random-action probability | Governs safety/learning vs exploitation | Start 1.0 → 0.01 |
| ε-decay | How fast ε drops | Fast → less exploration; affects safety bias | 0.99–0.999 |
| Episodes | Number of runs | More → converged, slower | Until Q stable |
| Max steps/episode | Cap on path length | Prevents loops | World-size dependent |

**Note for SARSA:** because it learns the value of the *exploring* policy, a high ε keeps Q-values optimistic about exploration costs — the "safe path" bias. Tuning ε and its decay directly controls that safety–optimality tradeoff.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- All Q(s,a) entries — learned via on-policy TD updates.

### Hyperparameters (chosen)
- α, γ, ε and ε-decay, number of episodes, max steps per episode.

---

## 23. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Markov property | Next state depends on current (s,a) | Bellman equation relies on it | Compare histories | Add memory / recurrent nets |
| On-policy learning | Agent learns its own behavior | Target uses actual a' | Confirm a' chosen after s' | Use off-policy (Q-learning) instead |
| Tabular state/action | Finite discrete | Table indexing | Check space size | Use function approximation |
| Sufficient exploration | All pairs visited | Convergence guarantee | Track visits | Raise ε |
| Stationary environment | Dynamics don't change | Fixed-point assumption | Monitor | Re-train / adapt |

---

## 24. Data Requirements

- **Data type:** generated online by the agent; no pre-existing dataset.
- **State space:** discrete and finite for tabular SARSA.
- **Action space:** discrete.
- **Rewards:** scalar, possibly sparse or dense.
- **Missing/outliers/scaling:** N/A (no feature matrix).
- **Exploration requirement:** must visit all (s,a) pairs for convergence.
- **Dataset size:** number of episodic experiences; more exploration → more coverage.

---

## 25. Feature Scaling

**Unnecessary** for tabular SARSA — no features; states are indices and Q-values are dimensionless.

For deep variants, normalize observations; but that is beyond tabular SARSA.

---

## 26. Evaluation Metrics

| Metric | Definition | Interpretation | Use When | Avoid When |
|---|---|---|---|---|
| Episode return | Sum of rewards per episode | Total reward | Basic learning measure | Sparse muddy returns |
| Success rate | Fraction of episodes reaching goal | Probability of task success | Goal tasks | Non-goal tasks |
| Mean episode length | Average steps per episode | Efficiency / safety (short ≠ always good) | Comparing policies | Cliff tasks where longer = safer |
| Average greedy return | Mean return with ε=0 | True policy value | Final comparison | During training (ε≠0) |

**Important:** The **TD error** minimized in training is **not** the evaluation metric. Evaluate with ε=0 over a batch of episodes and look at return/success. For cliff tasks, also watch *safety* (e.g., how often the agent falls) — SARSA's main selling point.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| On-policy correctness | Values match the policy actually followed |
| Safer near hazards | Accounts for exploration slips (notably better on cliff tasks) |
| Model-free | No dynamics needed |
| Convergent (tabular) | Reaches the on-policy fixed point reliably |
| Rewards the behavior | Optimizes the real, exploratory policy |
| Simple & interpretable | Q-table easy to read |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| Has to choose a' before updating | Slightly more per-step bookkeeping |
| Does not directly learn the optimal policy | With persistent exploration, stays suboptimal-suboptimal (biased toward safety) |
| Tabular only | Fails on large/continuous spaces |
| Sample-inefficient | Needs many episodes |
| Exploration can lower learned values | Provides a slightly pessimistic estimate vs true optimum |
| Less standard in deep RL | On-policy deep variants usually use full policy-gradient methods |

---

## 29. When to Use

✓ You want the agent's learned values to match its actual (exploring) behavior.
✓ Exploration near hazards is dangerous (cliffs, driving, robot near obstacles).
✓ Small discrete MDP you control online.
✓ You specifically need on-policy safety guarantees.
✓ You want to compare on-policy vs off-policy behavior.

---

## 30. When NOT to Use

✗ You need the true optimal policy and can safely explore anywhere (use off-policy Q-learning/DQN).
✗ Continuous or large state/action spaces (function approximation / deep methods).
✗ You have a fixed offline dataset (off-policy methods are needed).
✗ Exploration is cheap and you want speed to optimality.
✗ You require sample efficiency from limited environment calls.

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Cliff-walking robot navigation | grid state, moves | SARSA | Move selection (safe path) |
| Autonomous driving near hazards | lane/state, actions | SARSA | Safe maneuver choice |
| Warehouse robot pathing | grid cells | SARSA | Step decisions avoiding pits |
| Maze solving with obstacles | cell state | SARSA | Path respecting exploration risk |
| Game training with risky moves | board state | SARSA | Move choice (conservative) |

---

## 32. Failure Cases

- **Over-exploration death spiral:** if ε stays high near hazards, SARSA's values stay pessimistic and it may never exploit a good path.
- **Terminal handling bug:** forgetting that after a terminal state there is no a' (must set Q(terminal)=0) — otherwise spurious values.
- **No exploration:** if ε=0 from the start, SARSA never covers unseen states → stuck in first-behavior policy.
- **Non-stationary environment:** learned values for the old dynamics become wrong.
- **Slow convergence:** on large grids, many episodes needed; may plateau.

---

## 33. Overfitting and Underfitting

- **Underfitting (too little learning):** few episodes / high ε → Q stays low, agent behaves near-randomly; values "underfit" the reward structure.
- **Overfitting (memorizing limited experience):** if exploration is cut off early, SARSA memorizes a narrow set of trajectories and locks a suboptimal-but-safe policy — it "overfits" the explored region rather than the whole state space.

**Balance:** anneal ε over enough episodes so all (s,a) are visited, then let the on-policy values settle.

---

## 34. Bias-Variance Perspective

- **Variance:** one-step TD → only one random transition per update → low variance, like Q-learning.
- **Bias:** uses a bootstrap Q(s',a') → biased toward the current policy's (imperfect) value. This bias is *deliberate* for SARSA: it encodes the exploratory policy's realism.
- **Tradeoff:** SARSA trades a small downward bias (from accounting for exploration) for safety. Compared to Monte Carlo (unbiased, high variance) it converges faster but to an on-policy value that may differ from the true optimal.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| SARSA | Update toward r + γ·Q(s',a') | On-policy, safe exploration | Suboptimal w/ persistence ε | Hazard-adjacent control |
| Q-Learning | Update toward r + γ·max Q | Learns optimal directly | Max-bias, unsafe near cliffs | Clean exploration |
| Monte Carlo | Full-episode return | Unbiased | High variance, episodic only | Known full returns |
| TD(0) | One-step bootstrap | Low variance | Bias | Prediction |
| DQN | Q-learning + neural net | Scales | Divergence risk | Big state spaces |

---

## 36. Algorithm Selection Guide

```text
Value-based control, small discrete MDP:
├── Is exploration near hazards dangerous?
│   ├── YES → SARSA (on-policy safety)
│   └── NO  → Q-Learning (optimal, off-policy)
└── Is state/action space continuous or huge?
        → DQN / deep methods (see file 05+)
```

---

## 37. Common Mistakes

```text
❌ Forgetting to choose a' before the update
Why wrong: you'd be using an action chosen after the update — breaks the on-policy tuple.
Correct: pick a' first, then update with Q(s',a').

❌ Using max instead of Q(s',a') and still calling it SARSA
Why wrong: that is Q-learning's off-policy target, not SARSA's.
Correct: use the actual next action a'.

❌ Treating a terminal state as having a valid a'
Why wrong: no future → must use 0.
Correct: if done, set future = 0.

❌ Expecting SARSA to reach the same values as Q-learning
Why wrong: SARSA values reflect exploration; they differ (usually lower) near hazards.
Correct: understand this is the intended safety property.

❌ Judging success by TD loss
Why wrong: TD loss isn't the goal metric.
Correct: evaluate greedy episode return / success rate.
```

---

## 38. Interview Questions

### Beginner
**Q1. What does SARSA stand for?**
A: State–Action–Reward–State–Action — the 5-tuple used in each update.

**Q2. What is SARSA's update rule?**
A: Q(s,a) ← Q(s,a) + α(r + γ·Q(s',a') − Q(s,a)).

**Q3. What is on-policy learning?**
A: Learning the value of the policy you are actually following (using the real next action).

### Intermediate
**Q4. How does SARSA differ from Q-learning?**
A: SARSA uses the actual next action a' (on-policy); Q-learning uses max over next actions (off-policy).

**Q5. Why is SARSA considered safer?**
A: Its values bake in the chance of exploratory slips, so it avoids paths right at the edge of hazards (e.g., cliff tasks).

**Q6. In what setting do SARSA and Q-learning give identical updates?**
A: When the actual next action a' is itself the maximizing action (deterministic greedy behavior).

### Advanced
**Q7. What does SARSA converge to?**
A: The action-value of the behavior policy (on-policy fixed point), which near hazards is suboptimal-but-safe relative to Q-learning's optimum.

**Q8. Does SARSA converge to the optimal policy?**
A: Only in the limit as exploration ε→0; with persistent exploration it stays at the value of the exploratory policy.

**Q9. When would you definitely choose SARSA over Q-learning?**
A: When exploratory slips are catastrophic near the optimal path (safe robot/driving tasks), where the "optimistic" Q-learning path is too risky.

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
```text
Q(s,a) ← Q(s,a) + α( r + γ·Q(s',a') − Q(s,a) )    [SARSA, on-policy]
Q(s,a) ← Q(s,a) + α( r + γ·max_{a'}Q(s',a') − Q(s,a) )   [Q-learning, off-policy]
```

**Concepts likely tested:**
- On-policy vs off-policy difference (SARSA vs Q-learning)
- Which uses the actual next action vs the max
- Why SARSA is safer on cliff-walking
- The SARSA 5-tuple (S,A,R,S',A')

> **Representative pattern question (NOT a past GATE PYQ):** "At state B the ε-greedy policy actually selects action with Q=1.5; the max over B is 2.0. In the SARSA update for the transition into B, which value is used?" → Answer: Q(B,a') = 1.5 (the actual next action), not 2.0.

**Common traps:**
- Confusing SARSA's on-policy target with Q-learning's max target.
- Forgetting SARSA picks a' before updating.
- Assuming SARSA always equals or beats Q-learning — it trades optimality for safety.

---

## 40. Coding Practice

**Level 1 — Basic:** Implement tabular SARSA on a 3×3 grid; print Q-table.
**Level 2 — On-policy vs off-policy:** Implement both SARSA and Q-learning; compare final Q-values.
**Level 3 — Cliff task:** Run both on CliffWalking and record episode returns; observe the safe vs optimal path difference.
**Level 4 — Safety metric:** Track how often each algorithm falls into the cliff during greedy evaluation.
**Level 5 — ε sensitivity:** Vary ε and report the safety-optimality tradeoff.
**Level 6 — Hyperparameters:** Sweep α, γ; plot return curves.
**Level 7 — Real-world case study:** Build a warehouse-robot simulator with hazards; use SARSA for collision-safe navigation and justify the choice vs Q-learning.

---

## 41. Practical ML Workflow

```text
Problem → define states, actions, rewards, hazards, γ
   ↓
Environment → simulator / gymnasium (CliffWalking, FrozenLake)
   ↓
Agent design → Q-table, ε-greedy, α, γ, ε schedule
   ↓
Train → run episodes (pick a' before each update)
   ↓
Evaluate → greedy (ε=0) return + safety metric
   ↓
Tune → α, γ, ε-decay for desired safety/optimality
   ↓
Error analysis → verify safe paths, check terminal handling
   ↓
Deploy → ε → 0 policy π(s)=argmax_a Q(s,a)
   ↓
Monitor → re-train on dynamics change
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Space (Q-table) | O(\|S\|·\|A\|) | One value per (s,a) |
| Update per step | O(1) after a' chosen | Just one Q entry |
| Episode | Linear in steps | Path length |
| Training total | O(episodes × steps) | Linear in experience |
| Prediction | O(\|A\|) | argmax scan |

---

## 43. Advanced Concepts

- **SARSA(λ) / eligibility traces:** mix one-step and full-return updates to trade bias–variance (TD(λ)).
- **Expected SARSA:** use the *expected* value over π(s') instead of a single sample → lower variance, still on-policy-like.
- **On-policy in deep RL:** modern policy-gradient / actor-critic methods (files 06, 07) are also on-policy, echoing SARSA's philosophy at scale.
- **Hard safety constraints:** tabular SARSA is a stepping stone; real safety uses constrained/barrier methods.

---

## 44. Connections to Other Algorithms

```text
SARSA
   │
   ├── sibling → Q-Learning (off-policy version)
   ├── foundation → TD(0) prediction
   ├── extension → Expected SARSA, SARSA(λ)
   ├── cousin → Monte Carlo (episodic return)
   ├── deep on-policy lineage → Actor-Critic / PPO (file 07 + Part 2)
   └── contrast → DQN (off-policy deep, file 05)
```

---

## 45. If You Remember Only 5 Things

1. SARSA is on-policy, value-based, model-free TD control.
2. Update: Q(s,a) ← Q(s,a) + α(r + γ·Q(s',a') − Q(s,a)).
3. It uses the **actual next action** a' (not the max) — the on-policy hallmark.
4. It is safer than Q-learning near hazards because its values reflect exploratory slips.
5. It converges to the value of the behavior policy, coinciding with Q-learning when a' is the max.

---

## 46. Cheat Sheet

```text
Algorithm   : SARSA
Category    : Reinforcement Learning (model-free, on-policy, value-based)
Goal        : Learn Q for the behavior policy
Input       : (S,A,R,S',A') tuple
Output      : greedy policy π(s)=argmax_a Q(s,a)
Core Formula: Q(s,a) += α(r + γQ(s',a') − Q(s,a))
Loss        : squared on-policy TD error
Optimization: incremental updates
Parameters  : Q-table entries
Hyperparams : α, γ, ε, ε-decay, episodes
Assumptions : Markov, tabular, sufficient exploration, stationary
Advantages  : on-policy correct, safe near hazards, model-free
Disadvantages: suboptimal w/ exploration, tabular only, sample-inefficient
Use When    : hazard-adjacent control, on-policy safety
Avoid When  : want optimal policy cheaply, continuous spaces
Related     : Q-Learning, Expected SARSA, SARSA(λ), TD(0)
Key Exam    : on-policy vs off-policy; actual next action a'
Key Interv  : why safer, when identical to Q-learning, convergence target
```

---

## 47. Final Mental Model

```text
Initialize Q-table
   ↓
Pick first action a (ε-greedy)
   ↓
Take a → observe (r, s')
   ↓
Pick next action a' (ε-greedy)
   ↓
Update Q(s,a) toward r + γ·Q(s',a')
   ↓
s ← s', a ← a'
   ↓
Repeat → learn value of own behavior
   ↓
Deploy greedy policy
```

---

## 48. Knowledge Check

### Recall (5)
1. Write the SARSA update.
2. What does SARSA stand for?
3. What is the on-policy target?
4. How does SARSA differ from Q-learning?
5. Define the SARSA 5-tuple.

### Understanding (5)
6. Why is SARSA safer near cliffs?
7. Why must a' be chosen before the update?
8. When do SARSA and Q-learning coincide?
9. What does SARSA converge to?
10. How does ε affect SARSA's learned values?

### Application (5)
11. Implement the a'-before-update ordering.
12. Choose SARSA vs Q-learning for a robot near a cliff.
13. Handle the terminal state in the update.
14. Evaluate greedy return with ε=0.
15. Set an ε schedule that preserves some safety.

### Mathematical (5)
16. Write the Bellman equation SARSA approximates.
17. Explain the fixed point of SARSA.
18. Show how γ·max vs γ·Q(s',a') differ qualitatively.
19. Verify a 2-state SARSA computation.
20. Explain why persistent ε creates downward bias in values.

### Interview (5)
21. When exactly is SARSA ≡ Q-learning?
22. What are the convergence conditions?
23. Why not always use SARSA?
24. What is Expected SARSA?
25. How does SARSA relate to on-policy deep RL?

### Problem Solving (5)
26. SARSA learns a path hugging the cliff despite ε>0 — is that wrong?
27. Agent's values too pessimistic; how to adjust?
28. Cliff task: SARSA returns are lower than Q-learning's — is something broken?
29. How to measure "safety" alongside return?
30. Environment gets a new obstacle — next steps?

## Answers (explained)

1. Q(s,a) ← Q(s,a) + α(r + γ·Q(s',a') − Q(s,a)).
2. State–Action–Reward–State–Action.
3. r + γ·Q(s',a') with a' = actual next action.
4. SARSA on-policy (actual a'), Q-learning off-policy (max).
5. The 5-tuple (s, a, r, s', a') used in each update.
6. Values account for exploratory slips into the cliff, so the greedy policy stays safer.
7. The target needs Q(s',a'), so a' must be known before updating.
8. When a' is itself the maximizing action (deterministic greedy next step).
9. The action-value of the behavior (exploring) policy — on-policy fixed point.
10. Higher ε keeps Q-values lower/pessimistic (exploration cost) → safer policy.
11. Pick a2 = choose_action(s2) before the Q update line.
12. SARSA (on-policy safety) — a cliff slip is catastrophic.
13. If done, future = 0 (no next action exists).
14. Loop episodes with a = argmax Q; sum rewards; average.
15. Start ε=1.0, decay slowly; keep a small floor ε to retain safety margin.
16. Q_π(s,a) = E[r + γQ_π(s',a')] with a'~π.
17. Iterating sampled updates approaches the on-policy Bellman fixed point.
18. γ·max assumes perfect continuation (higher); γ·Q(s',a') uses the real policy (lower when exploring).
19. Follow the 2-episode arithmetic in Section 15 → Q(A)→0.9, Q(B)→1.0 (single-action case).
20. Because targets include suboptimal exploratory actions, values are dragged below the optimal ones.
21. Whenever the actual next action equals the argmax (greedy next action).
22. All (s,a) visited infinitely often, α satisfying Σα=∞, Σα²<∞, stationary env.
23. It doesn't learn the pure optimal policy under persistent exploration and is tabular-only.
24. SARSA but using E[Q(s',·)] over π instead of sampled a' — lower variance.
25. It embodies on-policy principles carried into policy-gradient / actor-critic deep RL.
26. Q-learning would hug the cliff; SARSA hugging it despite ε suggests ε too low — but some safety margin expected.
27. Lower ε, or use Expected SARSA, or accept safety.
28. Not broken — that's the intended safety/optimality tradeoff. Compare success and safety too.
29. Track greedy path distance from cliff and fall frequency alongside return.
30. Re-train — the Q-table is stale for new dynamics.

---

## 49. Final Learning Checklist

- [ ] I can define SARSA in one sentence
- [ ] I can write the SARSA update
- [ ] I understand the 5-tuple (S,A,R,S',A')
- [ ] I know why SARSA is on-policy
- [ ] I can pick a' before the update
- [ ] I understand the safety property near cliffs
- [ ] I know when SARSA ≡ Q-learning
- [ ] I can handle terminal states correctly
- [ ] I can implement SARSA from scratch
- [ ] I can implement on CliffWalking
- [ ] I can evaluate greedy return and safety
- [ ] I know convergence conditions
- [ ] I understand the ε → value bias link
- [ ] I can contrast SARSA with Q-learning
- [ ] I know Expected SARSA
- [ ] I can tune α, γ, ε
- [ ] I understand TD vs MC variance/bias
- [ ] I know when NOT to use SARSA
- [ ] I can connect SARSA to actor-critic lineage
- [ ] I can explain SARSA to a beginner

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** On-policy update, off-policy contrast, and convergence target verified; single-action numerical example hand-checked; the Q-learning-coincidence property confirmed.
- **Beginner-friendliness:** two-student analogy, cliff-walking intuition, tables, short paragraphs.
- **Math depth:** Bellman equation, fixed point, on-policy bias explanation, hand-verified example.
- **Practical depth:** from-scratch SARSA + CliffWalking library code, safety evaluation, workflow.
- **Exam depth:** on-policy/off-policy trap, representative (non-PYQ) pattern question.
- **Structure:** all 50 sections present in order.

**Verified:** numerical example recomputed by hand; matches Q-learning in the single-action deterministic case, which is exactly the expected behavior.
