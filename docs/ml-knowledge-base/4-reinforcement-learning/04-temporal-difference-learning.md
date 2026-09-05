# 04. Temporal Difference (TD) Learning

> Difficulty: ⭐⭐⭐☆☆ | Importance: ⭐⭐⭐⭐⭐
> Math Required: ⭐⭐⭐☆☆ | Coding Required: ⭐⭐⭐⭐☆
> GATE Relevance: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐⭐ | Industry: ⭐⭐⭐⭐⭐

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | Temporal Difference (TD) Learning — TD(0) prediction, + TD control family (SARSA/Q-learning) |
| Category | Reinforcement Learning |
| Type | Value-based, Model-free, Prediction (TD(0)); control via SARSA/Q-learning |
| Parametric / Non-parametric | Non-parametric (tabular) |
| Generative / Discriminative | N/A (decision-making) |
| Main Objective | Estimate V(s) (or Q(s,a)) by learning from **one-step** differences between consecutive value estimates — a mix of Monte Carlo and dynamic programming |
| Input | Stream of transitions (s, a, r, s') |
| Output | Value estimates and/or a control policy |
| Core Idea | Update after a single step using target r + γ·V(s') — bootstrapping a one-step lookahead instead of waiting for the full episode return |
| Typical Use Cases | Online prediction, learning during an episode, foundation of Q-learning/SARSA/DQN, RL teaching |

---

## 02. One-Line Definition

### Beginner Definition
TD learning updates its beliefs after every single step, using the immediate reward plus its own (current) guess about the next state — never waiting for the end of the game.

### Technical Definition
Temporal-Difference learning is a model-free prediction method that updates the current value estimate V(s) toward the one-step target `r + γ·V(s')`, combining the sampling idea of Monte Carlo with the bootstrapping idea of dynamic programming; it is biased but low-variance and supports online, incremental, in-episode learning.

---

## 03. Intuition

You are estimating how long the morning commute takes. Instead of waiting for the whole trip, you learn to predict *point by point* — every time you reach a landmark you revise.

Suppose your estimate of "trip length from home" is 40 minutes. Today you reach the halfway landmark (usually a 20 min point) in 25 minutes, and you estimate the remaining "halfway→office" leg at 20 minutes. That's a new target: 25 + 20 = 45 minutes. You slide your old 40-minute estimate a little toward 45.

Notice: you used your *own* guess about the second half, not the measured truth. That is **bootstrapping**. But you did use one *real* measurement (the 25-minute first half), which is what makes this "temporal difference" — you shrunk the gap between the old estimate and the new (reward + next estimate) target.

Because you update every step, you learn *during* the trip, are never stuck waiting for the end, and each update has little noise.

Step-by-step reasoning:
1. Take one real step: observe reward r and next state s'.
2. Form target = r + γ·(your current estimate of s').
3. Notice the difference between the target and your old estimate of s.
4. Move your estimate of s a small step toward the target.
5. Repeat every step.

---

## 04. Problem It Solves

**Problem:** Monte Carlo learning is unbiased but needs a full episode before any update, and has high variance. We often want to learn online, update quickly, and reduce variance.

**Example:** A trading bot gets long sequences of rewards. Waiting for the end of the day (a "terminal" event) to update is slow. We want to react every tick with a small estimate correction.

**What we want:** An estimator that updates every single transition, uses only one real reward at a time, and has modest variance.

**Why TD helps:** It combines:
- MC's sampling (a real r is observed),
- Dynamic programming's bootstrapping (γ·V(s') reused estimate)
giving a **one-step lookahead** target. It updates instantly and has low variance.

**Small example:** A→terminal with reward 1. TD(0) update: target = 1 + γ·0 = 1; V(A) moves toward 1 after ONE step. MC would also give 1 (same here), but TD did not need to wait.

---

## 05. Where It Fits in Machine Learning

```text
MACHINE LEARNING
│
├── Supervised Learning
├── Unsupervised Learning
└── Reinforcement Learning
    ├── Model-Free
    │   ├── Prediction
    │   │   ├── Monte Carlo (full episodes)
    │   │   └── TEMPORAL DIFFERENCE (TD(0))   ← YOU ARE HERE
    │   └── Control
    │       ├── TD control
    │       │   ├── SARSA (on-policy)
    │       │   ├── Q-Learning (off-policy)
    │       │   └── DQN (deep)
    │       └── Policy-gradient methods
    └── Model-Based
```

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| TD target | The value to move toward | r + γ·V(s') for TD(0) |
| TD error | How wrong current estimate is | δ = r + γ·V(s') − V(s) |
| Bootstrapping | Estimate updating with estimate | Uses V(s') in the target |
| Sampling | Using real observed reward | Uses observed r, not expectation |
| V(s) | Value of a state (expected return) | E[G_t \| S_t = s] |
| γ | Discount factor | Future-reward weight |
| α | Learning rate / step size | How far to move toward target |
| Online learning | Learn instantly from each step | No waiting for episode end |
| TD(0) | One-step TD prediction | Single-step target |
| TD(λ) | Multi-step extension | λ mixes 1-step and full-return |
| Control | Producing a policy | SARSA/Q-learning from TD |
| δ | TD error (named "delta") | The corrective signal |

---

## 07. Input and Output

**Input (training):**
- A stream of transitions (sₜ, aₜ, rₜ₊₁, sₜ₊₁) from interaction with the environment.

**Input (hyperparameters):**
- γ (discount), α (learning rate), and for control ε/ε-decay.

**Output (training):**
- Updated value estimate V(s) (prediction) or Q(s,a) (control) after **every single step**.

**Output (prediction / use):**
- For prediction: V(s) estimates.
- For control: policy π(s) = argmax_a Q(s,a).

**Parameters learned:** tabular value estimates.

**Hyperparameters:** γ, α, (ε for control), number of steps/episodes.

---

## 08. Mathematical Foundation

TD(0) updates the state-value estimate one step at a time:

```text
V(s) ← V(s) + α·[ r + γ·V(s') − V(s) ]
```

The target is a **one-step bootstrap**: one real reward r plus the discounted estimate of the next state. This is the middle ground between MC (uses full return, no bootstrap) and Dynamic Programming (uses the model's expectation to bootstrap).

**The discount factor γ intuitively:** a reward now is worth more than the same reward later. TD's target literally combines "now" (r) and "later" (γ·V(s')), so γ controls how patient the learner is about delayed reward.

The update is the sampled version of the **Bellman expectation equation**:

```text
V_π(s) = E_π[ r + γ·V_π(s') ]
```

Replace the expectation with one observed transition (r, s') — that sampled target is exact TD(0) learning.

**Required math concepts:** expectation, sampling, recursion, geometric series.

---

## 09. Core Formula

### The TD(0) prediction update

```text
V(s) ← V(s) + α·( r + γ·V(s') − V(s) )
```

#### Meaning
Slide the estimate of s toward "reward now + discounted guess of next state".

#### Symbols
- `V(s)` = current estimate of state s's value
- `α` = learning rate (0 < α ≤ 1)
- `r` = observed reward after leaving s
- `γ` = discount factor
- `V(s')` = current estimate of next state
- `r + γ·V(s')` = **TD target**
- `δ = r + γ·V(s') − V(s)` = **TD error**

#### Intuition
"Now + predicted later, corrected a little." One real nudge (r) and one imagined continuation (γ·V(s')) — both cheap to obtain, and usable mid-episode.

#### Example (one update)
γ = 0.9, α = 0.5. V(A)=10. Step from A→B gives r=0; V(B)=12.
```text
target = 0 + 0.9·12 = 10.8
V(A) = 10 + 0.5·(10.8 − 10) = 10 + 0.4 = 10.4
```
Note: V(A) and V(B) were estimates, not real outcomes — bootstrapping at work.

---

### For control: the SARSA / Q-learning control updates

```text
SARSA:      V(s,a) ← V(s,a) + α·( r + γ·Q(s',a') − Q(s,a) )      [on-policy]
Q-learning: Q(s,a) ← Q(s,a) + α·( r + γ·max_{a'} Q(s',a') − Q(s,a) )  [off-policy]
```

#### Meaning
The same one-step TD idea applied to action-values, choosing how to form the next-state estimate (actual action vs max).

#### Symbols
- `Q(s,a)` = action value
- `a'` = actual next action (SARSA) or argmax action (Q-learning's max)

#### Intuition
Control = TD prediction + a policy. The policy question ("which next action's value?") is exactly what splits on-policy (SARSA) from off-policy (Q-learning).

---

## 10. Derivation

**Step 1 — Write the value recursively:**

```text
V_π(s) = E_π[ r_{t+1} + γ·V_π(S_{t+1}) | S_t = s ]
```

**Step 2 — Replace the expectation with a single sample:**
One observed (r, s') gives a noisy target:

```text
target = r + γ·V(s')
```

**Step 3 — Interpret as stochastic gradient descent on squared error:**
For L = (1/2)(target − V(s))², the negative gradient is (target − V(s)). The TD update:
```text
V(s) ← V(s) + α·(target − V(s))
```
is one gradient step with step size α (target treated as fixed).

**Step 4 — Relationship to MC and DP:**
- MC replaces the expectation with a *full* sample path (full return) — no bootstrap.
- DP replaces sampling with the true expectation using the model — full bootstrap.
- TD(0) samples AND bootstraps — a single step of each.

**Key result:** since the transition is sampled but s' value is bootstrapped, TD estimates are **biased** (bootstrap error bleeds through) but **low-variance** (only one random transition per update).

---

## 11. How the Algorithm Works

```text
Initialize V (or Q) = 0
    ↓
Loop over steps:
    Observe current state s
    Take action a (policy for control)
    Observe reward r and next state s'
    ↓
    Compute TD target:  r + γ·V(s')    (control: r + γ·Q(s',a') / max)
    Compute TD error:   δ = target − estimate
    Update: estimate += α·δ
    ↓
    s ← s'
Repeat until convergence
```

---

## 12. Training Process

**Pre-training:** Initialize value table to 0.

**During training:**
- Each step produces a (r, s') pair; the estimate of the previous state is adjusted immediately.
- No waiting for episode end (unlike MC).
- For control, a policy (ε-greedy) is derived from Q between updates.

**What is learned:** one value per state (or state–action).

**Changes per iteration (one step):**
```text
δ = r + γ·V(s') − V(s)
V(s) = V(s) + α·δ
```

**Stopping:** fixed number of transitions/episodes, or value changes small.

**Final model contents:** value table (and Q-table for control).

---

## 13. Objective Function / Loss Function

For prediction, TD(0) effectively minimizes (in expectation) the **mean squared TD error**:

```text
L = (1/2)·( r + γ·V(s') − V(s) )²
```

**Interpretation:** penalize mismatch between the current estimate and its one-step "now + predicted next" target. High TD error = current estimate inconsistent with reward and the next-state estimate.

**Why chosen:** the TD error is central — it is both the loss and the update signal. It measures how "surprised" the learner is by the step.

**Low TD error** = estimates consistent. **High TD error** = estimates inconsistent (early learning, noisy transitions, or bad α).

**Note:** The TD error being minimized is NOT the evaluation metric — evaluation uses downstream return / success.

---

## 14. Optimization

**Definition:** Find estimates satisfying the Bellman equation (fixed point δ=0 in expectation).

**Method:** incremental stochastic updates — a stochastic approximation to the Bellman expectation operator.

**Update direction:** each transition pushes V(s) toward target = r + γ·V(s'), i.e., gradient descent on the squared TD error.

**Learning rate α:** controls update size; needs to decay over time for convergence.

**Convergence:** for tabular TD(0) with appropriate α schedule (Σα=∞, Σα²<∞), V(s) converges to V_π(s) with probability 1 even on continuing tasks (where MC cannot be applied directly).

**Local/global:** the true V_π is the unique fixed point; TD converges to it (with proper schedules).

---

## 15. Complete Numerical Example

**VERIFIED EXAMPLE** (hand-verified)

Two states A, B; from A, action "go" leads to B with reward 0; from B, "go" leads to terminal T with reward +1. Deterministic. γ=0.9, α=0.5. True values (from section on Q-learning/MC): V(B)=1/(1−0/... ) — for a one-step return to terminal, V(B)=1.0, V(A)=γ·V(B)=0.9.

Initialize V(A)=0, V(B)=0.

**Transition 1: (A → B, r=0):**
```text
target = 0 + 0.9·V(B) = 0 + 0.9·0 = 0
V(A) = 0 + 0.5·(0−0) = 0
```

**Transition 2: (B → T, r=1):**
```text
target = 1 + 0.9·V(T) = 1 + 0.9·0 = 1
V(B) = 0 + 0.5·(1−0) = 0.5
```

**Transition 3: (A → B, r=0):** V(B) is now 0.5.
```text
target = 0 + 0.9·0.5 = 0.45
V(A) = 0 + 0.5·(0.45−0) = 0.225
```

**Transition 4: (B → T, r=1):**
```text
target = 1 + 0.9·0 = 1
V(B) = 0.5 + 0.5·(1−0.5) = 0.75
```

| Step | Transition | V(A) | V(B) |
|---|---|---|---|
| 0 | init | 0 | 0 |
| 1 | A→B | 0 | 0 |
| 2 | B→T | 0 | 0.5 |
| 3 | A→B | 0.225 | 0.5 |
| 4 | B→T | 0.225 | 0.75 |
| ∞ | — | 0.9 | 1.0 |

**Key insight vs MC:** TD updated V(A) already at **step 3**, well before the end of the episode — online learning. The bootstrap (V(B)=0.5) entered A's target even though B never reached its true value yet. That is the essence of TD.

Hand-verified: every arithmetic step recomputed; convergence toward the true values V(A)=0.9, V(B)=1.0 confirmed.

---

## 16. Visual Explanation

### The TD "one-step learning" loop

```text
   V(s)  ──take action──▶  r
    ▲                       │
    │                       ▼
   target = r + γ·V(s') ◀── s'
    │
    ▼
   V(s) ← V(s) + α·(target − V(s))   ← update on every single step
```

### TD versus MC versus DP

```text
Estimator        Target                    Uses        Updates
──────────────────────────────────────────────────────────────────
Monte Carlo      full return G_t          sample path  at episode end
TD(0)            r + γ·V(s')              sample +     every step
                                           bootstrap
DP (model)       E[r + γ·V(s')]           model        every step

bias:     MC = 0,   TD = small,   DP = depends on model
variance: MC = high, TD = low,    DP = none (uses expectation)
```

---

## 17. Algorithm / Pseudocode

```text
TD(0) prediction:
1. Input: policy π, γ, α, steps N
2. V(s) ← 0 for all s
3. For each step (or episode):
     s ← current state
     a ← π(s)
     observe r, s'
     V(s) ← V(s) + α·( r + γ·V(s') − V(s) )
     s ← s'
4. Return V

TD control (Q-learning / SARSA): as in files 01 and 02 —
     same update with Q and an action-selection rule.
```

---

## 18. From-Scratch Implementation

```python
import random

class TDPredictor:
    def __init__(self, states, alpha=0.5, gamma=0.9):
        self.v = {s: 0.0 for s in states}
        self.alpha = alpha
        self.gamma = gamma

    def update(self, s, r, s_next):
        target = r + self.gamma*self.v[s_next] if s_next is not None else r
        td_error = target - self.v[s]
        self.v[s] += self.alpha*td_error
        return td_error

    def train_on_chain(self, episodes):
        # chain A->B->Terminal, deterministic, rewards 0, +1
        for _ in range(episodes):
            s = "A"
            for (state, next_state, reward) in self.chain:
                _ = self.update(state, reward, next_state)
                s = state

if __name__ == "__main__":
    tp = TDPredictor(["A", "B"])
    tp.chain = [("A", "B", 0.0), ("B", None, 1.0)]
    tp.train_on_chain(5000)
    print("V(A) =", round(tp.v["A"], 3), " V(B) =", round(tp.v["B"], 3))
    print("expected ~ 0.9 and 1.0")
```

(For a gridworld TD control, apply the same update to Q as shown in files 01/02; the prediction machinery is identical.)

---

## 19. Code Explanation

```text
Line:  target = r + self.gamma*self.v[s_next]
   What: forms the one-step bootstrap target
   Why: TD's update direction
   Math: r + γ·V(s')

Line:  td_error = target - self.v[s]
   What: computes the surprise signal δ
   Why: drives the update magnitude
   Math: δ = r + γ·V(s') − V(s)

Line:  self.v[s] += self.alpha*td_error
   What: moves the estimate toward target
   Why: incremental learning
   Math: V(s) ← V(s) + α·δ
```

---

## 20. Library Implementation

Libraries do not ship "TD learning" as a class; gymnasium supplies transitions and numpy does the arithmetic:

```python
import gymnasium as gym
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)
n_states = env.observation_space.n
n_actions = env.action_space.n
gamma, alpha = 0.9, 0.5

# Q-learning (TD control): same one-step TD idea, with max next action
Q = np.zeros((n_states, n_actions))
epsilon = 1.0

for ep in range(4000):
    s, _ = env.reset()
    done = False
    while not done:
        a = env.action_space.sample() if np.random.random() < epsilon else np.argmax(Q[s])
        s2, r, terminated, truncated, _ = env.step(a)
        done = terminated or truncated
        Q[s, a] += alpha*(r + gamma*np.max(Q[s2]) - Q[s, a])
        s = s2
    epsilon = max(0.01, epsilon*0.995)

print("Learned Q via TD control (Q-learning):")
print(Q[:4])
```

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| Discount γ | Future-reward weight | Low → myopic; high → far-sighted | 0.9–0.99 |
| Learning rate α | Update step size | High → unstable; low → slow | 0.1–0.5, decay |
| ε (control) | Exploration probability | Governs coverage | 1.0 → 0.01 |
| ε-decay | ε shrinkage rate | Controls exploration schedule | 0.99–0.999 |
| Steps/episodes | Training budget | More → better convergence | Until stable |
| n-step (beyond TD(0)) | Target lookahead length | Blends TD↔MC | See TD(λ), n-step TD |

**Too high / too low:** α too high → TD oscillates; γ too low → ignores future; ε too low → no exploration.

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- V(s) / Q(s,a) table entries — learned by TD updates.

### Hyperparameters (chosen)
- γ, α, ε (control), episodes, TD order (1-step vs n-step).

---

## 23. Assumptions

| Assumption | What it means | Why | How to check | If violated |
|---|---|---|---|---|
| Markov property | s' depends only on (s,a) | The bootstrap target assumes it | Compare histories with same (s,a) | Add memory / recurrent methods |
| Policy fixed (prediction) | π stop changing during eval | Estimate refers to π | Freeze π during eval | Re-evaluate per policy |
| Stationary rewards | Distribution of r doesn't drift | Fixed point exists | Monitor reward stats | Use non-stationary/adaptive methods |
| Tabular | Finite discrete values | Table indexing | Space size check | Function approximation |

---

## 24. Data Requirements

- **Data type:** a stream of transitions (s, a, r, s').
- **State/action spaces:** discrete and finite (tabular TD).
- **Rewards:** scalar; finite.
- **No missing values/outliers/scaling:** N/A.
- **Termination:** not needed for TD(0) prediction (works on continuing tasks — unlike MC).
- **Amount of data:** moderate — TD is efficient because each step updates.

---

## 25. Feature Scaling

**Unnecessary** for tabular TD — states are indices, no features.

For **deep TD** (DQN and successors) input normalization helps, as with any neural network input — but that's a deep RL concern, not tabular TD.

---

## 26. Evaluation Metrics

| Metric | Definition | Interpretation | Use When | Avoid When |
|---|---|---|---|---|
| TD error (δ) | r + γ·V(s') − V(s) | Movement signal, NOT final score | Debugging learning dynamics | Judging final quality |
| MSE vs true V | (1/n)Σ(V̂−V)² | Estimation accuracy | Known-model checks | Model unknown |
| Episode return | Σ (discounted) rewards | Actual task performance | Policy comparison | Want per-step credit |
| Success rate | Fraction of successful episodes | Task completion | Goal tasks | — |

**Important:** The **TD error** being minimized is NOT an evaluation metric — small TD error can accompany a poor policy. Evaluate with greedy policies and measure return/success.

---

## 27. Advantages

| Advantage | Why it matters |
|---|---|
| Online / incremental | One update per step; learn before episode ends |
| Low variance | Only one random transition per update |
| Model-free | No dynamics knowledge needed |
| Works on continuing tasks | No termination requirement (unlike MC) |
| Foundation of control | SARSA, Q-learning, DQN are TD control |
| Computationally cheap | O(1) per update |

---

## 28. Disadvantages

| Disadvantage | Consequence |
|---|---|
| Biased estimates | Bootstrap uses imperfect estimates of s' |
| Slower to propagate rewards | Long-delay rewards need many noisy hops |
| Sensitive to α | Fast learning requires careful tuning |
| Tabular scaling limits | Large spaces need function approximation |
| Markov assumption reliance | Non-Markovian settings degrade the bootstrap |
| No full-return information | A single step misses long-range structure |

---

## 29. When to Use

✓ You need online updates mid-episode.
✓ Continuing (non-terminating) tasks.
✓ Low variance matters and slight bias is acceptable.
✓ You want a foundation for SARSA/Q-learning/DQN.
✓ You can tolerate many environment steps (TD is cheap per step).
✓ You want robustness to bootstrap-safety tradeoffs via n-step variants.

---

## 30. When NOT to Use

✗ You need unbiased estimates (use MC).
✗ Episodes are short/count-limited (MC returns more per episode).
✗ You have the environment model (DP/methods exploit it better).
✗ Non-Markovian tasks without memory augmentation.
✗ You want to rely on a fixed offline dataset with behavior mismatch (off-policy deep RL complexities apply anyway).

---

## 31. Real-World Applications

| Application | Input | Algorithm | Output |
|---|---|---|---|
| Online elevator dispatch | floor states, requests | TD + control | Floor-to-serve policy |
| Adaptive traffic signals | traffic states | TD prediction | Congestion forecast per state |
| Electricity load control | load states, price | TD control | Action to optimize cost |
| Robot arm joint control | joint states | TD + FA | Torque decisions |
| Predictive maintenance | machine states | TD prediction | Failure-risk value |

---

## 32. Failure Cases

- **Bootstrap garbage-in:** if V(s') is wildly wrong, TD propagates that error (bias grows early).
- **α mishandling:** too large → oscillation; no decay → never converges.
- **Non-Markovian model:** TD changes based on hidden history → biased estimates.
- **Slow reward propagation:** with sparse/late rewards, many steps needed to transfer value backward.
- **Exploration dead-ends (control):** ε too low means never discovering better states.

---

## 33. Overfitting and Underfitting

- **Underfitting:** with α too small, estimates never track reality; with too few updates, value "looks flat" (underfit).
- **Overfitting:** tabular TD can't really overfit (one value per state), but function-approximation TD can memorize recent transitions — the deep-RL "catastrophic interference". In the tabular case, "over-reacting" to a single noisy transition (α too high) is the analogous failure — estimates wiggle around the truth.

**Balance:** decay α, keep exploring, and (in deep TD) use replay + target networks (file 05).

---

## 34. Bias-Variance Perspective

- **Bias:** TD bootstraps, so estimates inherit error from V(s') — **biased** (small bias when V is close, larger early in learning).
- **Variance:** only one random transition per update — **low variance**.
- **The spectrum:** MC (λ=1) = no bias, max variance; TD(0) (λ=0) = max bootstrap bias, min variance; n-step TD and TD(λ) interpolate.
- **Practical note:** because of the bootstrap, TD learns faster in many tasks despite the bias — bias decays as estimates improve, variance does not.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| TD(0) prediction | One-step bootstrap | Online, low variance | Biased | Online value est. |
| Monte Carlo | Full returns | Unbiased | High variance, episodic only | Offline evaluation |
| DP | Model-based full backup | Exact | Needs model | Planning |
| TD control (Q-learning/SARSA) | TD + policy | Full decision-making | Policy design choices | Actual tasks |
| DQN | Deep TD + replay | Scales to big states | Instability risk | High-dim obs. |

---

## 36. Algorithm Selection Guide

```text
Value estimation?
├── Need online updates now → TD(0)
├── Episodic, unbiased more important → Monte Carlo
├── Want a blend → n-step TD / TD(λ)
└── Have model → Dynamic Programming

Control?
├── Discrete, small → Q-learning / SARSA (TD control)
├── Large/continuous states → DQN (file 05)
└── Continuous actions → Policy-gradient / Actor-Critic (files 06–07)
```

---

## 37. Common Mistakes

```text
❌ Using MC-style full-return target but updating every step
Why wrong: mixing paradigms; TD(0) uses ONE-step target.
Correct: keep target = r + γ·V(s'), or use n-step explicitly.

❌ Forgetting terminal states have no next value
Why wrong: V(terminal)=0 by definition; else value leaks.
Correct: set future value = 0 at terminations.

❌ Using too large α with no decay
Why wrong: TD keeps bouncing around the fixed point.
Correct: decay α or use adaptive schedule.

❌ Judging learning by TD error magnitude
Why wrong: TD error is a training signal, not a performance metric.
Correct: evaluate greedy return / success rate.

❌ Updating V(s) with the CURRENT V(s') inside the same loop is fine —
   but implementing an OLD and NEW table wrongly
Why wrong: online update must use the updated table consistently.
Correct: TD(0) online uses latest values; batch TD(1) uses old — know which you implement.
```

---

## 38. Interview Questions

### Beginner
**Q1. What is TD learning?**
A: Estimating values by updating toward r + γ·V(s'), a one-step bootstrap target sampled from experience.

**Q2. What is the TD error?**
A: δ = r + γ·V(s') − V(s) — the difference between target and current estimate.

**Q3. Difference between TD and MC?**
A: TD bootstraps one step ahead (biased, low variance, online); MC uses the full episode return (unbiased, high variance, offline-at-final).

### Intermediate
**Q4. Why does TD combine MC and DP?**
A: MC samples real rewards; DP bootstraps future estimates. TD does both — sample one reward and bootstrap the next state.

**Q5. Where do SARSA and Q-learning come from?**
A: Both are TD control: TD prediction updated on Q(s,a) instead of V(s), with a next-action selection rule (actual a' for SARSA; max for Q-learning).

**Q6. Can TD work when episodes never end?**
A: Yes — TD(0) can update forever on continuing tasks; MC cannot (needs an end to compute G_t).

### Advanced
**Q7. What is the bias–variance profile of TD?**
A: Bootstrap → biased; single transition → low variance. MC is opposite. n-step TD interpolates.

**Q8. What is TD(λ) / eligibility traces?**
A: A mechanism assigning credit across the last λ-window of steps, blending TD(0) (λ=0) and MC (λ=1), with backward views via traces.

**Q9. Why does deep TD need replay + target networks?**
A: Nonstationary targets and correlated samples destabilize SGD; replay decorrelates and the fixed target network stabilizes the target distribution (file 05).

---

## 39. GATE / Exam Perspective

**Key formulas to memorize:**
```text
TD(0):  V(s) ← V(s) + α( r + γ·V(s') − V(s) )
MC:     V(s) ← V(s) + α( G_t − V(s) ),   G_t = Σγᵏ r
DP:     V(s) ← Σ_a π(a|s) Σ P(s'|s,a)[ r + γ·V(s') ]
```

**Concepts likely tested:**
- TD vs MC vs DP contrast (bootstrap + sampling)
- What TD error measures
- Online, incremental nature of TD
- Continuing-task applicability
- Which control method is derived from which target

> **Representative pattern question (NOT a past GATE PYQ):** "With V(A)=5, take action to B (r=2), next-state estimate V(B)=9, γ=0.9, α=0.5. Compute the TD target and updated V(A)." → target = 2+0.9·9=10.1; V(A)=5+0.5(10.1−5)=5+2.55=7.55.

**Common traps:**
- Confusing δ (TD error) with the target (`r + γ·V(s')`).
- Thinking TD waits for episode end.
- Forgetting V(terminal)=0 in the target.
- Claiming TD is unbiased (it is biased; MC is unbiased).

---

## 40. Coding Practice

**Level 1 — Basic:** Implement TD(0) on a 3-state chain; print V after each step.
**Level 2 — Online pattern:** Show V updating BEFORE episode end.
**Level 3 — Compare:** Run MC vs TD(0) on a random-walk; plot convergence.
**Level 4 — Control:** Implement Q-learning (TD control) on a gridworld.
**Level 5 — SARSA:** Implement SARSA; compare with Q-learning Q-tables.
**Level 6 — n-step:** Implement 2-step TD; compare bias/variance behavior.
**Level 7 — Real-world case study:** Use TD-based Q-learning on Taxi-v3 or a custom warehouse-nav simulator; report return curves and success rate, and explain when TD updates beat MC there.

---

## 41. Practical ML Workflow

```text
Problem → episodic or continuing? pick evaluation metric
   ↓
Environment → simulator / gymnasium
   ↓
Estimator → tabular V or Q, choose γ, α, ε schedule
   ↓
Train → step-by-step TD updates
   ↓
Evaluate → greedy performance (return / success rate)
   ↓
Tune → α, γ, ε-decay; consider n-step
   ↓
Error analysis → track TD error to spot instability
   ↓
Deploy → policy from learned values
   ↓
Monitor → re-estimate if dynamics/rewards drift
```

---

## 42. Complexity

| Aspect | Complexity | Notes |
|---|---|---|
| Update per step | O(1) | One table cell |
| Space | O(\|S\|) or O(\|S\|·\|A\|) | Value/Q tables |
| Episode cost | O(T) | Linear in length |
| Total training | O(steps) | Linear in experience |
| Prediction | O(\|S\|) or O(\|A\|) | scan for argmax/greedy |
| n-step | O(n) per update | Extra lookahead memory |

---

## 43. Advanced Concepts

- **n-step TD / TD(λ):** intermediate lookahead trading bias for variance slowly.
- **Eligibility traces:** backward credit assignment to many past states from one step.
- **Off-policy TD learning:** importance sampling in the TD error (used in deep RL partially).
- **Value function approximation:** TD with function approximation is where deep RL lives (DQN).
- **Gradient TD methods:** fix convergence issues of off-policy semi-gradient TD.
- **True Online TD(λ):** computationally tidy TD(λ) variant.

---

## 44. Connections to Other Algorithms

```text
TD Learning
   │
   ├── prediction base → MC (file 03) (bias–variance counterpart)
   ├── control → SARSA (file 02), Q-Learning (file 01)
   ├── deep extension → DQN (file 05)
   ├── multi-step generalization → TD(λ), n-step TD
   └── contributes the critic → Actor-Critic (file 07) uses TD for the critic
```

---

## 45. If You Remember Only 5 Things

1. TD updates every step: V(s) ← V(s) + α(r + γ·V(s') − V(s)).
2. It blends MC sampling (real r) and DP bootstrapping (γ·V(s')).
3. Biased but low variance; works online and on continuing tasks.
4. TD error δ = r + γ·V(s') − V(s) is the learning signal.
5. All of SARSA, Q-learning, and DQN are TD mechanisms in disguise.

---

## 46. Cheat Sheet

```text
Algorithm   : Temporal Difference Learning (TD(0))
Category    : Reinforcement Learning (model-free, value-based prediction)
Goal        : Estimate V(s)/Q(s,a) online with one-step bootstrap
Input       : transitions (s, a, r, s')
Output      : value estimates (+ policy in control variants)
Core Formula: V(s) ← V(s) + α(r + γ·V(s') − V(s))
Loss        : squared TD error (1/2)δ²
Optimization: stochastic incremental updates
Parameters  : value table entries
Hyperparams : γ, α, ε (control), episodes
Assumptions : Markov, tabular, stationary rewards
Advantages  : online, low variance, model-free, continuing tasks
Disadvantages: biased, sensitive to α, limited tabular scale
Use When    : online, cheap updates, be careful with bias
Avoid When  : need unbiased estimates, have model, episodes rare
Related     : MC, DP, SARSA, Q-learning, DQN, TD(λ)
Key Exam    : target vs δ, bootstrap, TD-vs-MC-vs-DP, γ
Key Interv  : bias/variance, control derivations, online learning
```

---

## 47. Final Mental Model

```text
Step in environment
   ↓
Observe r, s'
   ↓
target = r + γ·V(s')
   ↓
δ = target − V(s)
   ↓
V(s) += α·δ
   ↓
s ← s'
Repeat → estimates satisfy Bellman equation (in expectation)
```

---

## 48. Knowledge Check

### Recall (5)
1. Write the TD(0) update.
2. Define the TD target.
3. Define the TD error δ.
4. What does bootstrapping mean here?
5. Name two TD control algorithms.

### Understanding (5)
6. Why is TD biased?
7. Why is TD low-variance?
8. Why does TD work on continuing tasks?
9. How does TD combine MC and DP ideas?
10. When do V(s) and the MC estimate coincide?

### Application (5)
11. Hand-compute one TD update.
12. Choose γ for a long-horizon task.
13. Convert TD prediction to Q-learning control.
14. Convert TD prediction to SARSA control.
15. Design an α schedule.

### Mathematical (5)
16. Show TD is (expected) gradient descent on squared TD error.
17. State tabular TD(0) convergence conditions.
18. Write δ for SARSA vs Q-learning.
19. Explain why MC target = full return, TD target = one-step.
20. Describe TD(λ) interpolation endpoint behavior.

### Interview (5)
21. When would you prefer MC over TD?
22. When TD over MC?
23. How do target networks stabilize deep TD?
24. What's the difference between online TD(0) and batch TD(0)?
25. How should you evaluate a TD-trained agent?

### Problem Solving (5)
26. Value oscillation with α=0.9 — fix?
27. Sparse reward — TD slow to learn; what helps?
28. Continuing task — which estimator family is valid?
29. Non-Markovian environment — what to do?
30. Reports of "TD error near zero but terrible policy" — explain.

## Answers (explained)

1. V(s) ← V(s) + α(r + γ·V(s') − V(s)).
2. target = r + γ·V(s') — a one-step bootstrap.
3. δ = target − V(s).
4. Using an estimate (V(s')) inside another estimate's update target.
5. SARSA (on-policy) and Q-learning (off-policy).
6. The target contains V(s'), itself an estimate — errors propagate.
7. Only ONE random transition feeds each update, not a full trajectory.
8. The target requires only the current step, so it never needs an episode end.
9. Samples one real reward (MC-like) and bootstraps next value (DP-like).
10. In deterministic environments after V(s') = true value, or when the one-step target equals the full return (e.g., at terminal-adjacent states).
11. target=10.1 example in Section 39 style → V(A)=7.55.
12. Choose higher γ (0.95–0.99) to value long-term credit.
13. Replace V(s)/V(s') with Q(s,a)/max_a Q(s',a); add ε-greedy action choice.
14. Replace with Q(s,a)/Q(s',a') where a' = actual next action chosen by policy.
15. Start ~0.3–0.5, decay each episode (e.g., ×0.99); or schedule α_t = 1/t.
16. ∇(1/2)δ² = −δ assuming fixed target → V update by +αδ is gradient descent.
17. γ∈[0,1), summable α with Σα=∞, Σα²<∞, and all states visited infinitely often → wp1 convergence to V_π.
18. SARSA δ = r + γQ(s',a') − Q(s,a); Q-learning δ = r + γ·max_a'Q(s',a') − Q(s,a).
19. Because bootstrapping length is the difference: MC sums everything (γ^k→γ^T), TD stops at one step (γ¹).
20. λ=0 → TD(0) (one-step); λ=1 → MC (full returns).
21. When unbiasedness matters and you can afford full episodes; when you want true sampled returns.
22. When online updates, low variance, or continuing tasks are important.
23. Fixed target network freezes the target distribution, reducing nonstationarity in the loss (DQN).
24. Online = use current estimates immediately; batch = hold old estimates for a full sweep (tabular batch TD, offline).
25. Fix ε=0, run greedy episodes, compute mean return / success rate — never TD error alone.
26. Reduce α, add decay schedule.
27. Dense/engineered rewards, n-step returns, or credit-assignment tricks.
28. TD-style estimators (one-step or n-step); MC full-return is undefined without termination.
29. Feed the agent a history/memory window that restores Markovianity.
30. TD error being small means estimates are self-consistent, NOT that policy quality is high — always check return/success.

---

## 49. Final Learning Checklist

- [ ] I can define TD learning in one sentence
- [ ] I can write the TD(0) update
- [ ] I know the difference between target and TD error
- [ ] I understand bootstrapping
- [ ] I know why TD is biased
- [ ] I know why TD is low variance
- [ ] I can contrast TD with MC and DP
- [ ] I know TD works on continuing tasks
- [ ] I can hand-compute a TD update
- [ ] I can implement TD(0) from scratch
- [ ] I can build Q-learning from TD
- [ ] I can build SARSA from TD
- [ ] I understand n-step TD / TD(λ) basics
- [ ] I know convergence conditions
- [ ] I understand α, γ effects
- [ ] I can design an α schedule
- [ ] I can evaluate an agent properly (not by TD error)
- [ ] I know when MC beats TD and vice-versa
- [ ] I can connect TD to DQN's replay/target ideas
- [ ] I can explain TD learning to a beginner

---

## 50. Quality Control Note

**Self-review:**
- **Accuracy:** TD(0) update and TD error verified; the bias/variance claims cross-checked against MC and DP; hand-computed example confirmed (V(A)→0.9, V(B)→1.0) and contrasted with MC timing (online update at step 3).
- **Beginner-friendliness:** commute analogy, short paragraphs, comparison tables.
- **Math depth:** derivation as stochastic-approximation / gradient descent, convergence conditions, n-step extension.
- **Practical depth:** from-scratch predictor + gymnasium control code, hyperparameters, workflow.
- **Exam depth:** target-vs-delta trap, TD-vs-MC-vs-DP contrasts, representative (non-PYQ) pattern question.
- **Structure:** all 50 sections present in order.

**Verified:** numerical example recomputed by hand; the online-learning contrast with MC (V(A) updated before episode end) demonstrated explicitly.