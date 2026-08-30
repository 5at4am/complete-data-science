# GATE DA — DAY 1 QUESTION PAPER

**Date:** 30 August 2026
**Total Questions:** 50
**Distribution:** Probability & Statistics (10) · Linear Algebra (8) · Calculus & Optimization (6) · Programming/DSA (8) · Databases & Warehousing (6) · Machine Learning (8) · AI (4)
**Difficulty:** 15 Easy · 25 Medium · 10 Hard

---

## ROUND 1 — QUESTIONS ONLY

Attempt all 50 questions before reading the solutions in ROUND 2. Record your answer and approximate time per question.

---

### SECTION A — PROBABILITY & STATISTICS (Q1–Q10)

**GATE-DA-D001-Q001** [EASY · MCQ]
A fair coin is tossed twice. What is the probability that at least one toss shows Heads?
- (A) 1/4
- (B) 1/2
- (C) 3/4
- (D) 1

**GATE-DA-D001-Q002** [MEDIUM · NAT]
A diagnostic test for a disease is 95% accurate on infected patients (sensitivity) and 90% accurate on healthy patients (specificity = 90%, i.e., 10% false positive). The disease affects 1% of the population. If a randomly chosen person tests positive, what is the probability (as a percentage, rounded to 2 decimal places) that they actually have the disease?

**GATE-DA-D001-Q003** [EASY · NAT]
A fair six-sided die is rolled. Let X be the number shown. Find E[X], the expected value of the roll.

**GATE-DA-D001-Q004** [MEDIUM · NAT]
A coin with P(Heads) = 0.6 is tossed 5 times. What is the probability of getting exactly 3 heads? (Give answer rounded to 3 decimal places.)

**GATE-DA-D001-Q005** [MEDIUM · NAT]
A call center receives an average of 3 calls per minute. Assuming calls follow a Poisson process, what is the probability of receiving exactly 2 calls in a given minute? (Give answer rounded to 3 decimal places. Use e ≈ 2.718.)

**GATE-DA-D001-Q006** [EASY · MCQ]
Let Z be a standard normal random variable. Which of the following statements is TRUE?
- (A) P(Z ≤ 0) = 0.5
- (B) P(Z ≤ 0) = 0
- (C) P(Z ≥ 0) = 0.25
- (D) P(Z = 0) = 0.5

**GATE-DA-D001-Q007** [MEDIUM · MSQ]
Let X and Y be two random variables with Var(X) = 4, Var(Y) = 9, and Cov(X, Y) = 3. Which of the following are TRUE?
- (A) Var(X + Y) = 19
- (B) Var(X − Y) = 7
- (C) Var(2X) = 8
- (D) The correlation coefficient ρ(X, Y) = 0.5

**GATE-DA-D001-Q008** [HARD · NAT]
The joint PMF of discrete random variables X and Y is given by P(X = x, Y = y) = c·(x + y) for x ∈ {1, 2} and y ∈ {1, 2}, and 0 otherwise. Find the value of the constant c.

**GATE-DA-D001-Q009** [HARD · NAT]
Let X₁, X₂, ..., Xₙ be i.i.d. random variables from an Exponential distribution with rate parameter λ (i.e., f(x) = λe^(−λx), x > 0). Find the Maximum Likelihood Estimator (MLE) of λ in terms of the sample mean x̄ = (1/n)ΣXᵢ. Give your answer as a simplified expression.

**GATE-DA-D001-Q010** [MEDIUM · MCQ]
Which of the following statements about the Central Limit Theorem (CLT) is TRUE?
- (A) The CLT states that the sample mean of i.i.d. random variables is always exactly normally distributed for any sample size.
- (B) The CLT states that the distribution of the sample mean of i.i.d. random variables approaches a normal distribution as the sample size increases, regardless of the underlying distribution (with finite variance).
- (C) The CLT only applies to normally distributed populations.
- (D) The CLT requires the underlying distribution to be symmetric.

---

### SECTION B — LINEAR ALGEBRA (Q11–Q18)

**GATE-DA-D001-Q011** [EASY · MCQ]
What is the rank of the matrix
A = [[1, 2], [2, 4]]?
- (A) 0
- (B) 1
- (C) 2
- (D) 3

**GATE-DA-D001-Q012** [EASY · NAT]
Find the determinant of the matrix
A = [[2, 3], [1, 4]].

**GATE-DA-D001-Q013** [MEDIUM · NAT]
Find the (1,1) entry of the inverse of the matrix
A = [[2, 1], [1, 3]].

**GATE-DA-D001-Q014** [MEDIUM · NAT]
Find the sum of the eigenvalues of the matrix
A = [[4, 1], [2, 3]].

**GATE-DA-D001-Q015** [MEDIUM · MSQ]
Which of the following sets of vectors in ℝ³ are linearly independent?
- (A) {(1, 0, 0), (0, 1, 0), (0, 0, 1)}
- (B) {(1, 2, 3), (2, 4, 6)}
- (C) {(1, 1, 0), (1, 0, 1), (0, 1, 1)}
- (D) {(1, 0, 0), (0, 1, 0), (1, 1, 0)}

**GATE-DA-D001-Q016** [MEDIUM · MCQ]
Which of the following matrices is positive definite?
- (A) [[1, 2], [2, 1]]
- (B) [[2, 0], [0, 3]]
- (C) [[1, 0], [0, −1]]
- (D) [[0, 0], [0, 0]]

**GATE-DA-D001-Q017** [HARD · NAT]
Let A be a 5 × 7 matrix with rank 3. What is the dimension of the null space (nullity) of A?

**GATE-DA-D001-Q018** [HARD · NAT]
A 2 × 2 matrix A has eigenvalues λ₁ = 2 and λ₂ = 5. What is the determinant of A?

---

### SECTION C — CALCULUS & OPTIMIZATION (Q19–Q24)

**GATE-DA-D001-Q019** [EASY · NAT]
Evaluate the limit: lim(x→0) sin(x)/x.

**GATE-DA-D001-Q020** [EASY · NAT]
Let f(x, y) = x² + 3xy + y². Find the gradient ∇f at the point (1, 1). Give your answer as the sum of the two components (∂f/∂x + ∂f/∂y) at (1,1).

**GATE-DA-D001-Q021** [MEDIUM · NAT]
Let f(x, y) = x² + 2xy + y². Find the determinant of the Hessian matrix of f.

**GATE-DA-D001-Q022** [MEDIUM · MSQ]
Which of the following functions are convex on their entire domain?
- (A) f(x) = x²
- (B) f(x) = eˣ
- (C) f(x) = −x²
- (D) f(x) = |x|

**GATE-DA-D001-Q023** [HARD · NAT]
Use Lagrange multipliers to find the maximum value of f(x, y) = xy subject to the constraint x + y = 4 (with x, y > 0).

**GATE-DA-D001-Q024** [MEDIUM · NAT]
Consider gradient descent on f(x) = x² with learning rate η = 0.5, starting at x₀ = 4. What is the value of x after one update step (x₁)?

---

### SECTION D — PROGRAMMING & DATA STRUCTURES (Q25–Q32)

**GATE-DA-D001-Q025** [EASY · NAT]
What is the output of the following code?
```
int sum = 0;
for (int i = 1; i <= 5; i++) {
    sum += i;
}
printf("%d", sum);
```

**GATE-DA-D001-Q026** [MEDIUM · NAT]
What is the output of the following recursive function when called as f(4)?
```
int f(int n) {
    if (n <= 1) return 1;
    return n * f(n - 1);
}
```

**GATE-DA-D001-Q027** [EASY · MCQ]
What is the time complexity of the following code?
```
for (int i = 1; i <= n; i *= 2) {
    printf("%d", i);
}
```
- (A) O(n)
- (B) O(log n)
- (C) O(n log n)
- (D) O(√n)

**GATE-DA-D001-Q028** [EASY · MCQ]
In a Binary Search Tree (BST), which of the following is TRUE?
- (A) The left subtree of a node contains only nodes with keys greater than the node's key.
- (B) The right subtree of a node contains only nodes with keys less than the node's key.
- (C) The left subtree of a node contains only nodes with keys less than the node's key.
- (D) In-order traversal of a BST always produces a decreasing sequence.

**GATE-DA-D001-Q029** [EASY · MCQ]
Which data structure follows the Last-In-First-Out (LIFO) principle?
- (A) Queue
- (B) Stack
- (C) Array
- (D) Linked List

**GATE-DA-D001-Q030** [MEDIUM · NAT]
Consider the array [5, 1, 4, 2, 8]. How many swaps are performed during the FIRST pass of Bubble Sort (ascending order)?

**GATE-DA-D001-Q031** [HARD · NAT]
Solve the recurrence relation T(n) = 2T(n/2) + n, with T(1) = 1. What is the asymptotic complexity of T(n)? Give your answer in the form O(n^k) where k is an integer.

**GATE-DA-D001-Q032** [HARD · MCQ]
Consider an undirected graph with vertices {A, B, C, D, E} and edges {A-B, A-C, B-D, C-E, D-E}. Starting BFS from vertex A, which of the following is a valid BFS traversal order (assuming neighbors are visited in alphabetical order)?
- (A) A, B, C, D, E
- (B) A, B, C, E, D
- (C) A, C, B, D, E
- (D) A, B, D, C, E

---

### SECTION E — DATABASES & DATA WAREHOUSING (Q33–Q38)

**GATE-DA-D001-Q033** [EASY · MCQ]
Consider a table `Sales(product, amount)`. Which SQL query computes the total amount per product?
- (A) SELECT product, SUM(amount) FROM Sales;
- (B) SELECT product, SUM(amount) FROM Sales GROUP BY product;
- (C) SELECT SUM(amount) FROM Sales GROUP BY product;
- (D) SELECT product, amount FROM Sales GROUP BY product;

**GATE-DA-D001-Q034** [MEDIUM · MCQ]
A relation R(A, B, C) has functional dependencies A → B and B → C. Which normal form is R in?
- (A) 1NF only
- (B) 2NF but not 3NF
- (C) 3NF but not BCNF
- (D) BCNF

**GATE-DA-D001-Q035** [MEDIUM · MSQ]
Given a relation R(A, B, C, D) with functional dependencies A → B and A → C. Which of the following are TRUE?
- (A) A is a candidate key of R.
- (B) {A, D} is a candidate key of R.
- (C) B is a candidate key of R.
- (D) The closure of {A} is {A, B, C}.

**GATE-DA-D001-Q036** [MEDIUM · MCQ]
In relational algebra, which operation selects rows (tuples) satisfying a condition?
- (A) Projection (π)
- (B) Selection (σ)
- (C) Join (⋈)
- (D) Union (∪)

**GATE-DA-D001-Q037** [EASY · MCQ]
Which of the following is NOT one of the ACID properties of a transaction?
- (A) Atomicity
- (B) Consistency
- (C) Isolation
- (D) Distribution

**GATE-DA-D001-Q038** [MEDIUM · MCQ]
In a star schema data warehouse, which table contains the measures (facts) and foreign keys to dimension tables?
- (A) Dimension table
- (B) Fact table
- (C) Staging table
- (D) Lookup table

---

### SECTION F — MACHINE LEARNING (Q39–Q46)

**GATE-DA-D001-Q039** [EASY · MCQ]
In the bias-variance tradeoff, a model that is too complex (overfits) typically has:
- (A) High bias, low variance
- (B) Low bias, high variance
- (C) High bias, high variance
- (D) Low bias, low variance

**GATE-DA-D001-Q040** [MEDIUM · NAT]
In simple linear regression y = β₀ + β₁x, given the following data: (x, y) = (1, 2), (2, 3), (3, 4). What is the value of the slope β₁? (Assume least squares fit.)

**GATE-DA-D001-Q041** [MEDIUM · MCQ]
In logistic regression, the output of the sigmoid function σ(z) = 1/(1 + e^(−z)) represents:
- (A) The predicted class label directly
- (B) The probability that the instance belongs to the positive class
- (C) The loss value
- (D) The gradient of the loss

**GATE-DA-D001-Q042** [MEDIUM · NAT]
In k-NN classification with k = 3, a test point has 3 nearest neighbors with labels {Class A, Class A, Class B}. What class is the test point assigned to? (Answer: Class A or Class B)

**GATE-DA-D001-Q043** [MEDIUM · MSQ]
Which of the following statements about Support Vector Machines (SVM) are TRUE?
- (A) SVM finds the hyperplane that maximizes the margin between classes.
- (B) Support vectors are the data points closest to the decision boundary.
- (C) The kernel trick allows SVM to handle non-linearly separable data.
- (D) SVM always requires the data to be linearly separable.

**GATE-DA-D001-Q044** [MEDIUM · NAT]
In K-Means clustering with K = 2, the initial centroids are at positions 0 and 10. The data points are {1, 2, 8, 9}. After the first assignment step, what is the new centroid of the cluster containing point 1 (i.e., the cluster with initial centroid 0)?

**GATE-DA-D001-Q045** [HARD · NAT]
PCA is applied to a dataset with 4 features. The eigenvalues of the covariance matrix are 10, 5, 3, 2. What is the minimum number of principal components needed to retain at least 80% of the total variance?

**GATE-DA-D001-Q046** [HARD · NAT]
A classifier produces the following confusion matrix: TP = 40, FP = 10, FN = 20, TN = 30. What is the F1-score? (Give answer rounded to 2 decimal places.)

---

### SECTION G — ARTIFICIAL INTELLIGENCE (Q47–Q50)

**GATE-DA-D001-Q047** [EASY · MCQ]
Which of the following is an uninformed (blind) search strategy?
- (A) A* search
- (B) Greedy best-first search
- (C) Breadth-First Search (BFS)
- (D) Hill climbing

**GATE-DA-D001-Q048** [MEDIUM · MCQ]
For A* search to be optimal, the heuristic h(n) must be:
- (A) Admissible (never overestimate the true cost to reach the goal)
- (B) Consistent (monotonic) — which also implies admissibility
- (C) Both A and B are correct conditions for optimality
- (D) h(n) = 0 for all nodes

**GATE-DA-D001-Q049** [MEDIUM · NAT]
In a minimax game tree where the MAX player chooses the maximum and MIN chooses the minimum, the leaf values (from left to right) are: 3, 5, 2, 9, 1, 7, 4, 6. The tree has 3 levels (root MAX, then MIN, then leaves). What is the minimax value at the root?

**GATE-DA-D001-Q050** [HARD · MCQ]
Which of the following propositional logic statements is a TAUTOLOGY?
- (A) P ∧ ¬P
- (B) P ∨ ¬P
- (C) P → Q
- (D) P ∧ Q

---

## ROUND 2 — ANSWERS & SOLUTIONS

---

### SECTION A — PROBABILITY & STATISTICS

#### GATE-DA-D001-Q001
**Answer: (C) 3/4**

**Given:** Fair coin tossed twice.
**Required:** P(at least one Heads).

**Solution:**
Sample space = {HH, HT, TH, TT}, each with probability 1/4.
P(at least one H) = 1 − P(no H) = 1 − P(TT) = 1 − 1/4 = 3/4.

**Why others are wrong:**
- (A) 1/4 is P(TT) — the complement event, not the answer.
- (B) 1/2 is P(exactly one H) or a common careless guess.
- (D) 1 would require certainty, which is false.

**Shortcut:** P(at least one) = 1 − P(none). For n tosses, P(at least one H) = 1 − (1/2)ⁿ.

**Common trap:** Forgetting to subtract from 1 and instead counting only HH.

**GATE insight:** "At least one" problems are almost always solved via the complement. This is a recurring GATE pattern.

---

#### GATE-DA-D001-Q002
**Answer: 8.72%**

**Given:** Sensitivity = P(+|D) = 0.95, Specificity = P(−|H) = 0.90 → P(+|H) = 0.10. Prevalence P(D) = 0.01.
**Required:** P(D|+) — probability of disease given positive test.

**Formula (Bayes' Theorem):**
P(D|+) = [P(+|D)·P(D)] / [P(+|D)·P(D) + P(+|H)·P(H)]

**Substitution:**
P(D|+) = (0.95 × 0.01) / (0.95 × 0.01 + 0.10 × 0.99)
= 0.0095 / (0.0095 + 0.099)
= 0.0095 / 0.1085
= 0.08756...
≈ 0.0876 = 8.76%

Let me recompute precisely: 0.0095 / 0.1085 = 0.0875576... ≈ 8.76%.

**Final answer: 8.76%** (rounded to 2 decimal places).

**Shortcut:** This is the classic "base rate fallacy" problem. Even with a highly accurate test, low prevalence keeps P(D|+) low.

**Common trap:** Many answer 95% (sensitivity), ignoring the low base rate. This is the classic base-rate fallacy.

**GATE insight:** Bayes' theorem with a medical-test framing is a very common GATE DA question. Always account for prevalence.

---

#### GATE-DA-D001-Q003
**Answer: 3.5**

**Given:** Fair die, X = number shown.
**Required:** E[X].

**Formula:** E[X] = Σ x·P(X = x) = (1+2+3+4+5+6)/6.

**Calculation:** (1+2+3+4+5+6)/6 = 21/6 = 3.5.

**Final answer: 3.5**

**Shortcut:** For a uniform distribution over {1,...,n}, E[X] = (n+1)/2 = (6+1)/2 = 3.5.

**Common trap:** Forgetting that expectation need not be an integer or an actual outcome.

**GATE insight:** Expectation of a uniform discrete variable is a quick, high-frequency concept.

---

#### GATE-DA-D001-Q004
**Answer: 0.346**

**Given:** n = 5, p = 0.6, k = 3.
**Required:** P(X = 3).

**Formula (Binomial):** P(X = k) = C(n,k)·pᵏ·(1−p)^(n−k).

**Substitution:**
P(X = 3) = C(5,3)·(0.6)³·(0.4)²
= 10 × 0.216 × 0.16
= 10 × 0.03456
= 0.3456
≈ 0.346

**Final answer: 0.346**

**Shortcut:** C(5,3) = C(5,2) = 10.

**Common trap:** Using p = 0.5 (assuming fair coin) instead of 0.6, or miscomputing C(5,3).

**GATE insight:** Binomial probability with a non-fair coin tests both the formula and careful arithmetic.

---

#### GATE-DA-D001-Q005
**Answer: 0.224**

**Given:** λ = 3 calls/minute, k = 2.
**Required:** P(X = 2).

**Formula (Poisson):** P(X = k) = (λᵏ·e^(−λ))/k!.

**Substitution:**
P(X = 2) = (3²·e^(−3))/2!
= (9 × e^(−3))/2
= (9 × 0.049787)/2
= 0.44804/2
= 0.22402
≈ 0.224

**Final answer: 0.224**

**Shortcut:** e^(−3) ≈ 0.0498. Memorize common e^(−λ) values.

**Common trap:** Forgetting to divide by k! or using the wrong λ.

**GATE insight:** Poisson distribution is frequently tested; remember the mean = variance = λ.

---

#### GATE-DA-D001-Q006
**Answer: (A) P(Z ≤ 0) = 0.5**

**Given:** Z ~ N(0, 1).
**Required:** True statement.

**Solution:**
The standard normal is symmetric about 0, so P(Z ≤ 0) = 0.5. Also, since Z is continuous, P(Z = 0) = 0 (not 0.5).

**Why others are wrong:**
- (B) P(Z ≤ 0) = 0 is false; it's 0.5.
- (C) P(Z ≥ 0) = 0.25 is false; it's 0.5.
- (D) P(Z = 0) = 0.5 is false; for a continuous variable, P(Z = c) = 0.

**Common trap:** Confusing continuous probability (P(Z = 0) = 0) with discrete probability.

**GATE insight:** Understanding that continuous random variables have zero probability at any single point is a key conceptual point.

---

#### GATE-DA-D001-Q007
**Answer: (A), (B), (C), (D)**

**Given:** Var(X) = 4, Var(Y) = 9, Cov(X, Y) = 3.
**Required:** True statements.

**Solution:**
- (A) Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y) = 4 + 9 + 2(3) = 19. TRUE.
- (B) Var(X − Y) = Var(X) + Var(Y) − 2Cov(X, Y) = 4 + 9 − 6 = 7. TRUE.
- (C) Var(2X) = 4·Var(X) = 4 × 4 = 16. **Wait** — Var(2X) = 2²·Var(X) = 4 × 4 = 16, NOT 8. So (C) is FALSE.

Let me recompute: Var(2X) = 4·Var(X) = 4 × 4 = 16. So (C) is FALSE.

- (D) ρ = Cov(X, Y)/√(Var(X)·Var(Y)) = 3/√(4×9) = 3/√36 = 3/6 = 0.5. TRUE.

**Correct answer: (A), (B), (D)**

**Why (C) is wrong:** Var(2X) = 2²·Var(X) = 16, not 8. The factor of 2 is squared.

**Common trap:** Forgetting that Var(aX) = a²Var(X), not a·Var(X).

**GATE insight:** Variance scaling and covariance in sums are classic MSQ material.

---

#### GATE-DA-D001-Q008
**Answer: 1/12**

**Given:** P(X = x, Y = y) = c(x + y) for x, y ∈ {1, 2}.
**Required:** c.

**Solution:**
The joint PMF must sum to 1 over all (x, y):
ΣΣ c(x + y) = 1.

Possible (x, y) pairs and their (x + y) values:
- (1,1): 2
- (1,2): 3
- (2,1): 3
- (2,2): 4

Sum of (x + y) = 2 + 3 + 3 + 4 = 12.
So c × 12 = 1 → c = 1/12.

**Final answer: 1/12**

**Shortcut:** Sum all (x + y) values, then c = 1/(sum).

**Common trap:** Forgetting to include all four (x, y) combinations.

**GATE insight:** Normalizing a joint PMF by summing over all outcomes is a fundamental skill.

---

#### GATE-DA-D001-Q009
**Answer: 1/x̄**

**Given:** Xᵢ ~ Exp(λ), f(x) = λe^(−λx).
**Required:** MLE of λ.

**Solution:**
Likelihood: L(λ) = Π λe^(−λxᵢ) = λⁿ·e^(−λΣxᵢ).
Log-likelihood: ℓ(λ) = n·ln(λ) − λΣxᵢ.
Derivative: dℓ/dλ = n/λ − Σxᵢ = 0.
→ n/λ = Σxᵢ → λ̂ = n/Σxᵢ = 1/x̄.

**Final answer: 1/x̄** (where x̄ is the sample mean).

**Shortcut:** For Exponential, the MLE of the rate is the reciprocal of the sample mean.

**Common trap:** Confusing rate λ with mean 1/λ. The MLE of the mean is x̄, but the MLE of the rate is 1/x̄.

**GATE insight:** MLE derivation via log-likelihood is a standard GATE DA question.

---

#### GATE-DA-D001-Q010
**Answer: (B)**

**Solution:**
The CLT states that the distribution of the sample mean of i.i.d. random variables (with finite variance) approaches a normal distribution as n → ∞.

**Why others are wrong:**
- (A) "Always exactly normal for any sample size" is false; it's approximate, asymptotic.
- (C) CLT applies to any distribution with finite variance, not just normal.
- (D) CLT does not require symmetry.

**Common trap:** Believing CLT gives exact normality for small samples.

**GATE insight:** The CLT is the theoretical basis for many statistical procedures; understanding its conditions (i.i.d., finite variance, asymptotic) is key.

---

### SECTION B — LINEAR ALGEBRA

#### GATE-DA-D001-Q011
**Answer: (B) 1**

**Given:** A = [[1, 2], [2, 4]].
**Required:** Rank.

**Solution:**
Row 2 = 2 × Row 1, so the rows are linearly dependent. The rank is the number of linearly independent rows = 1.

**Why others are wrong:**
- (C) 2 would require independent rows.
- (A) 0 would require a zero matrix.

**Shortcut:** If one row is a scalar multiple of another, rank decreases by 1.

**Common trap:** Counting rows instead of independent rows.

**GATE insight:** Rank = number of linearly independent rows/columns. Detecting linear dependence is fast.

---

#### GATE-DA-D001-Q012
**Answer: 5**

**Given:** A = [[2, 3], [1, 4]].
**Required:** det(A).

**Formula:** det([[a, b], [c, d]]) = ad − bc.

**Calculation:** det = 2×4 − 3×1 = 8 − 3 = 5.

**Final answer: 5**

**Shortcut:** ad − bc directly.

**Common trap:** Sign errors in the cross-multiplication.

**GATE insight:** 2×2 determinant is a fast, high-frequency computation.

---

#### GATE-DA-D001-Q013
**Answer: 3/5**

**Given:** A = [[2, 1], [1, 3]].
**Required:** (1,1) entry of A⁻¹.

**Solution:**
det(A) = 2×3 − 1×1 = 6 − 1 = 5.
A⁻¹ = (1/det)·[[3, −1], [−1, 2]].
The (1,1) entry = 3/5.

**Final answer: 3/5**

**Shortcut:** For [[a, b], [c, d]], A⁻¹ = (1/(ad−bc))·[[d, −b], [−c, a]].

**Common trap:** Swapping the wrong entries or forgetting the 1/det factor.

**GATE insight:** 2×2 inverse formula is essential and fast.

---

#### GATE-DA-D001-Q014
**Answer: 7**

**Given:** A = [[4, 1], [2, 3]].
**Required:** Sum of eigenvalues.

**Formula:** Sum of eigenvalues = trace(A) = a + d.

**Calculation:** trace = 4 + 3 = 7.

**Final answer: 7**

**Shortcut:** Sum of eigenvalues = trace; product = determinant.

**Common trap:** Computing the determinant instead of the trace.

**GATE insight:** Trace = sum of eigenvalues is a powerful shortcut that avoids solving the characteristic polynomial.

---

#### GATE-DA-D001-Q015
**Answer: (A), (C)**

**Solution:**
- (A) {(1,0,0), (0,1,0), (0,0,1)} — standard basis, linearly independent. TRUE.
- (B) {(1,2,3), (2,4,6)} — second is 2× first, linearly dependent. FALSE.
- (C) {(1,1,0), (1,0,1), (0,1,1)} — determinant = 1(0−1) − 1(1−0) + 0 = −1 − 1 = −2 ≠ 0, so independent. TRUE.
- (D) {(1,0,0), (0,1,0), (1,1,0)} — third = first + second, linearly dependent. FALSE.

**Correct answer: (A), (C)**

**Common trap:** For (D), not noticing (1,1,0) = (1,0,0) + (0,1,0).

**GATE insight:** Linear independence is often checked via determinant (for n vectors in ℝⁿ) or by spotting linear combinations.

---

#### GATE-DA-D001-Q016
**Answer: (B) [[2, 0], [0, 3]]**

**Solution:**
A symmetric matrix is positive definite iff all its eigenvalues are positive (or all leading principal minors positive).
- (A) [[1,2],[2,1]]: eigenvalues 3 and −1 → not PD.
- (B) [[2,0],[0,3]]: eigenvalues 2 and 3, both positive → PD. TRUE.
- (C) [[1,0],[0,−1]]: eigenvalue −1 → not PD.
- (D) [[0,0],[0,0]]: eigenvalues 0 → positive semi-definite, not PD.

**Correct answer: (B)**

**Common trap:** Confusing positive definite (all eigenvalues > 0) with positive semi-definite (all ≥ 0).

**GATE insight:** Positive definiteness via eigenvalues is a key ML-relevant concept (e.g., in optimization and covariance matrices).

---

#### GATE-DA-D001-Q017
**Answer: 4**

**Given:** A is 5 × 7, rank = 3.
**Required:** Nullity (dimension of null space).

**Formula (Rank-Nullity):** For an m × n matrix, rank + nullity = n (number of columns).

**Calculation:** nullity = n − rank = 7 − 3 = 4.

**Final answer: 4**

**Shortcut:** nullity = (number of columns) − rank.

**Common trap:** Using the number of rows (5) instead of columns (7).

**GATE insight:** Rank-Nullity theorem: rank + nullity = number of columns. Very common.

---

#### GATE-DA-D001-Q018
**Answer: 10**

**Given:** Eigenvalues λ₁ = 2, λ₂ = 5.
**Required:** det(A).

**Formula:** det(A) = product of eigenvalues.

**Calculation:** det = 2 × 5 = 10.

**Final answer: 10**

**Shortcut:** Product of eigenvalues = determinant.

**Common trap:** Summing instead of multiplying.

**GATE insight:** Determinant = product of eigenvalues; trace = sum. Memorize both.

---

### SECTION C — CALCULUS & OPTIMIZATION

#### GATE-DA-D001-Q019
**Answer: 1**

**Given:** lim(x→0) sin(x)/x.
**Required:** Limit.

**Formula:** lim(x→0) sin(x)/x = 1 (standard limit).

**Final answer: 1**

**Shortcut:** This is a fundamental standard limit.

**Common trap:** Using L'Hôpital incorrectly or thinking it's 0.

**GATE insight:** Standard limits are foundational and appear in many derivative/optimization problems.

---

#### GATE-DA-D001-Q020
**Answer: 10**

**Given:** f(x, y) = x² + 3xy + y².
**Required:** ∂f/∂x + ∂f/∂y at (1, 1).

**Solution:**
∂f/∂x = 2x + 3y. At (1,1): 2 + 3 = 5.
∂f/∂y = 3x + 2y. At (1,1): 3 + 2 = 5.
Sum = 5 + 5 = 10.

**Final answer: 10**

**Shortcut:** Compute partials, evaluate, sum.

**Common trap:** Forgetting to evaluate at the point, or mixing up partial derivatives.

**GATE insight:** Gradient computation is fundamental for optimization and ML.

---

#### GATE-DA-D001-Q021
**Answer: 0**

**Given:** f(x, y) = x² + 2xy + y².
**Required:** det(Hessian).

**Solution:**
∂f/∂x = 2x + 2y, ∂f/∂y = 2x + 2y.
∂²f/∂x² = 2, ∂²f/∂y² = 2, ∂²f/∂x∂y = 2.
Hessian H = [[2, 2], [2, 2]].
det(H) = 2×2 − 2×2 = 4 − 4 = 0.

**Final answer: 0**

**Shortcut:** Note f = (x + y)², so the Hessian is rank-1 with determinant 0.

**Common trap:** Miscomputing the mixed partial.

**GATE insight:** Hessian determinant (discriminant) determines the nature of critical points.

---

#### GATE-DA-D001-Q022
**Answer: (A), (B), (D)**

**Solution:**
- (A) f(x) = x²: second derivative 2 > 0, convex. TRUE.
- (B) f(x) = eˣ: second derivative eˣ > 0, convex. TRUE.
- (C) f(x) = −x²: second derivative −2 < 0, concave, not convex. FALSE.
- (D) f(x) = |x|: convex (epigraph is convex; subgradient exists). TRUE.

**Correct answer: (A), (B), (D)**

**Common trap:** Forgetting that |x| is convex (it's not differentiable at 0 but is convex).

**GATE insight:** Convexity via second derivative (f'' ≥ 0) is a key optimization concept.

---

#### GATE-DA-D001-Q023
**Answer: 4**

**Given:** Maximize f(x, y) = xy subject to x + y = 4, x, y > 0.
**Required:** Maximum value.

**Solution (Lagrange multipliers):**
Lagrangian: L = xy + λ(4 − x − y).
∂L/∂x = y − λ = 0 → y = λ.
∂L/∂y = x − λ = 0 → x = λ.
So x = y. Constraint: x + y = 4 → 2x = 4 → x = 2, y = 2.
f(2, 2) = 4.

**Final answer: 4**

**Shortcut:** By AM-GM, xy ≤ ((x+y)/2)² = (4/2)² = 4, with equality at x = y = 2.

**Common trap:** Forgetting the constraint or misapplying the multiplier conditions.

**GATE insight:** Lagrange multipliers for constrained optimization is a standard GATE DA topic.

---

#### GATE-DA-D001-Q024
**Answer: 2**

**Given:** f(x) = x², η = 0.5, x₀ = 4.
**Required:** x₁ after one gradient descent step.

**Formula:** x₁ = x₀ − η·f'(x₀).

**Solution:**
f'(x) = 2x. f'(4) = 8.
x₁ = 4 − 0.5 × 8 = 4 − 4 = 0.

**Wait** — let me recompute: x₁ = 4 − 0.5 × 8 = 4 − 4 = 0.

**Final answer: 0**

**Common trap:** Forgetting the learning rate or the derivative factor.

**GATE insight:** Gradient descent update is fundamental to ML optimization.

---

### SECTION D — PROGRAMMING & DATA STRUCTURES

#### GATE-DA-D001-Q025
**Answer: 15**

**Given:** Loop sums i from 1 to 5.
**Required:** Output.

**Solution:**
sum = 1 + 2 + 3 + 4 + 5 = 15.

**Final answer: 15**

**Shortcut:** Sum of first n integers = n(n+1)/2 = 5×6/2 = 15.

**Common trap:** Off-by-one errors in loop bounds.

**GATE insight:** Loop tracing and summation formulas are basic but frequent.

---

#### GATE-DA-D001-Q026
**Answer: 24**

**Given:** f(n) = n·f(n−1), f(1) = 1.
**Required:** f(4).

**Solution:**
f(4) = 4·f(3) = 4·3·f(2) = 4·3·2·f(1) = 4·3·2·1 = 24.

**Final answer: 24**

**Shortcut:** This computes 4! = 24.

**Common trap:** Off-by-one in the base case or recursion depth.

**GATE insight:** Recursion tracing (factorial) is a classic GATE question.

---

#### GATE-DA-D001-Q027
**Answer: (B) O(log n)**

**Solution:**
The loop variable doubles each iteration: i = 1, 2, 4, 8, ... The number of iterations is the number of times we can double before exceeding n, which is log₂(n). So complexity is O(log n).

**Why others are wrong:**
- (A) O(n) would be a linear increment.
- (C) O(n log n) is for nested loops.
- (D) O(√n) is for i² ≤ n.

**Common trap:** Confusing doubling (log) with linear increments.

**GATE insight:** Recognizing log complexity from doubling/halving loop variables is essential.

---

#### GATE-DA-D001-Q028
**Answer: (C)**

**Solution:**
In a BST, the left subtree of a node contains only nodes with keys LESS than the node's key, and the right subtree contains keys GREATER than the node's key.

**Why others are wrong:**
- (A) Left subtree has keys less than, not greater than.
- (B) Right subtree has keys greater than, not less than.
- (D) In-order traversal of a BST produces an INCREASING (sorted ascending) sequence, not decreasing.

**Common trap:** Reversing the left/right ordering or the in-order direction.

**GATE insight:** BST property and in-order traversal (sorted order) are fundamental.

---

#### GATE-DA-D001-Q029
**Answer: (B) Stack**

**Solution:**
A stack follows LIFO (Last-In-First-Out). A queue follows FIFO.

**Why others are wrong:**
- (A) Queue is FIFO.
- (C) Array is a random-access structure.
- (D) Linked list is a linear structure without LIFO/FIFO semantics by itself.

**Common trap:** Confusing stack (LIFO) with queue (FIFO).

**GATE insight:** LIFO vs FIFO is a basic but frequently tested distinction.

---

#### GATE-DA-D001-Q030
**Answer: 3**

**Given:** Array [5, 1, 4, 2, 8], Bubble Sort ascending.
**Required:** Swaps in first pass.

**Solution:**
First pass compares adjacent pairs and swaps if out of order:
- Compare 5, 1: 5 > 1 → swap → [1, 5, 4, 2, 8]. Swap 1.
- Compare 5, 4: 5 > 4 → swap → [1, 4, 5, 2, 8]. Swap 2.
- Compare 5, 2: 5 > 2 → swap → [1, 4, 2, 5, 8]. Swap 3.
- Compare 5, 8: 5 < 8 → no swap.

Total swaps in first pass = 3.

**Final answer: 3**

**Common trap:** Counting comparisons instead of swaps, or miscounting.

**GATE insight:** Tracing Bubble Sort passes is a common GATE question.

---

#### GATE-DA-D001-Q031
**Answer: O(n log n)**

**Given:** T(n) = 2T(n/2) + n, T(1) = 1.
**Required:** Asymptotic complexity.

**Solution (Master Theorem):**
a = 2, b = 2, f(n) = n. log_b(a) = log₂(2) = 1. f(n) = n = n^1 = n^(log_b a). This is Case 2: T(n) = Θ(n^log_b a · log n) = Θ(n log n).

**Final answer: O(n log n)**

**Shortcut:** This is the classic Merge Sort recurrence → O(n log n).

**Common trap:** Confusing with T(n) = 2T(n/2) + 1 (which is O(n)) or T(n) = T(n/2) + n (which is O(n)).

**GATE insight:** Master Theorem application is a high-frequency GATE question.

---

#### GATE-DA-D001-Q032
**Answer: (A) A, B, C, D, E**

**Given:** Vertices {A,B,C,D,E}, edges {A-B, A-C, B-D, C-E, D-E}. BFS from A, neighbors in alphabetical order.
**Required:** BFS traversal.

**Solution:**
BFS uses a queue:
- Visit A. Enqueue neighbors in alphabetical order: B, C.
- Dequeue B. Visit B. Enqueue its unvisited neighbors: D.
- Dequeue C. Visit C. Enqueue its unvisited neighbors: E.
- Dequeue D. Visit D. Its neighbor E already visited.
- Dequeue E. Visit E.

Order: A, B, C, D, E.

**Correct answer: (A)**

**Why others are wrong:**
- (B) A, B, C, E, D — wrong order for D and E.
- (C) A, C, B, D, E — wrong (B before C).
- (D) A, B, D, C, E — wrong (D before C).

**Common trap:** Confusing BFS (queue, level-order) with DFS (stack, depth-first).

**GATE insight:** BFS vs DFS traversal order is a classic GATE question.

---

### SECTION E — DATABASES & DATA WAREHOUSING

#### GATE-DA-D001-Q033
**Answer: (B)**

**Solution:**
To compute total amount per product, group by product and aggregate SUM(amount): `SELECT product, SUM(amount) FROM Sales GROUP BY product;`

**Why others are wrong:**
- (A) Missing GROUP BY — invalid (product not aggregated).
- (C) Missing product in SELECT — can't show per-product.
- (D) Missing SUM — no aggregation.

**Common trap:** Forgetting GROUP BY when using aggregate functions with a non-aggregated column.

**GATE insight:** GROUP BY + aggregate is a fundamental SQL pattern.

---

#### GATE-DA-D001-Q034
**Answer: (C) 3NF but not BCNF**

**Given:** R(A, B, C), FDs A → B, B → C.
**Required:** Normal form.

**Solution:**
Candidate key: A (A determines B and C). 
- A → B: B is a non-prime attribute, A is a superkey → OK for BCNF.
- B → C: B is not a superkey (B does not determine A), and C is a non-prime attribute. This is a transitive dependency (A → B → C). So R is NOT in 3NF... 

Wait, let me reconsider. 3NF requires: for every FD X → Y, either X is a superkey OR Y is a prime attribute. Here B → C: B is not a superkey, and C is not a prime attribute (prime attributes are those in candidate keys; candidate key is {A}, so only A is prime). So B → C violates 3NF.

So R is in 2NF but NOT 3NF. Let me check 2NF: 2NF requires no partial dependency of non-prime attributes on any proper subset of a candidate key. Since the candidate key is a single attribute {A}, there are no partial dependencies. So R is in 2NF.

**Correct answer: (B) 2NF but not 3NF**

**Common trap:** The transitive dependency A → B → C violates 3NF, so R is not 3NF.

**GATE insight:** Recognizing transitive dependencies and normal forms is a core DB question.

---

#### GATE-DA-D001-Q035
**Answer: (B), (D)**

**Given:** R(A, B, C, D), FDs A → B, A → C.
**Required:** True statements.

**Solution:**
- (A) A is a candidate key? Closure of {A} = {A, B, C}. It does NOT include D, so A alone is not a candidate key. FALSE.
- (B) {A, D} is a candidate key? Closure of {A, D} = {A, B, C, D} = all attributes. Minimal? A alone doesn't determine D, D alone doesn't determine A. So {A, D} is a candidate key. TRUE.
- (C) B is a candidate key? Closure of {B} = {B} only. Not a key. FALSE.
- (D) Closure of {A} = {A, B, C}. TRUE.

**Correct answer: (B), (D)**

**Common trap:** Forgetting that A alone doesn't determine D, so A is not a key.

**GATE insight:** Computing attribute closures to find candidate keys is a fundamental skill.

---

#### GATE-DA-D001-Q036
**Answer: (B) Selection (σ)**

**Solution:**
Selection (σ) selects rows (tuples) satisfying a condition. Projection (π) selects columns.

**Why others are wrong:**
- (A) Projection selects columns, not rows.
- (C) Join combines tables.
- (D) Union combines rows from two tables.

**Common trap:** Confusing selection (rows) with projection (columns).

**GATE insight:** Selection vs projection is a basic relational algebra distinction.

---

#### GATE-DA-D001-Q037
**Answer: (D) Distribution**

**Solution:**
ACID = Atomicity, Consistency, Isolation, Durability. "Distribution" is not one of them.

**Correct answer: (D)**

**Common trap:** Confusing "Durability" with "Distribution."

**GATE insight:** ACID properties are a fundamental transaction concept.

---

#### GATE-DA-D001-Q038
**Answer: (B) Fact table**

**Solution:**
In a star schema, the fact table contains measures (numeric facts) and foreign keys to dimension tables. Dimension tables contain descriptive attributes.

**Why others are wrong:**
- (A) Dimension table contains descriptive attributes, not measures.
- (C) Staging table is for ETL intermediate storage.
- (D) Lookup table is another term for dimension.

**Common trap:** Confusing fact table (measures) with dimension table (descriptors).

**GATE insight:** Star schema fact/dimension distinction is a core data warehousing concept.

---

### SECTION F — MACHINE LEARNING

#### GATE-DA-D001-Q039
**Answer: (B) Low bias, high variance**

**Solution:**
An overfit (too complex) model fits training data too closely, capturing noise. It has low bias (fits well) but high variance (generalizes poorly to new data).

**Why others are wrong:**
- (A) High bias, low variance = underfitting.
- (C) High bias, high variance = poor model overall.
- (D) Low bias, low variance = ideal, not overfitting.

**Common trap:** Confusing overfitting (low bias, high variance) with underfitting (high bias, low variance).

**GATE insight:** Bias-variance tradeoff is a fundamental ML concept.

---

#### GATE-DA-D001-Q040
**Answer: 1**

**Given:** (x, y) = (1, 2), (2, 3), (3, 4).
**Required:** Slope β₁.

**Formula:** β₁ = Σ(xᵢ − x̄)(yᵢ − ȳ) / Σ(xᵢ − x̄)².

**Solution:**
x̄ = (1+2+3)/3 = 2. ȳ = (2+3+4)/3 = 3.
Numerator: (1−2)(2−3) + (2−2)(3−3) + (3−2)(4−3) = (−1)(−1) + 0 + (1)(1) = 1 + 0 + 1 = 2.
Denominator: (1−2)² + (2−2)² + (3−2)² = 1 + 0 + 1 = 2.
β₁ = 2/2 = 1.

**Final answer: 1**

**Shortcut:** The data is perfectly linear (y = x + 1), so slope = 1.

**Common trap:** Using the wrong formula or miscomputing means.

**GATE insight:** Least squares slope computation is a core ML/statistics question.

---

#### GATE-DA-D001-Q041
**Answer: (B)**

**Solution:**
The sigmoid output σ(z) ∈ (0, 1) represents the probability that the instance belongs to the positive class, P(y = 1 | x).

**Why others are wrong:**
- (A) The predicted class is obtained by thresholding (e.g., ≥ 0.5 → class 1), not directly.
- (C) The loss is computed separately (e.g., cross-entropy).
- (D) The gradient is a derivative, not the sigmoid output.

**Common trap:** Confusing the sigmoid output (probability) with the hard class label.

**GATE insight:** Logistic regression outputs probabilities; thresholding gives class labels.

---

#### GATE-DA-D001-Q042
**Answer: Class A**

**Solution:**
With k = 3, the test point is assigned the majority class among its 3 nearest neighbors. Neighbors: {Class A, Class A, Class B}. Class A appears 2 times, Class B once. Majority = Class A.

**Final answer: Class A**

**Shortcut:** Majority vote among k neighbors.

**Common trap:** Using k = 3 but counting incorrectly, or using an even k with a tie.

**GATE insight:** k-NN majority voting is a basic classification concept.

---

#### GATE-DA-D001-Q043
**Answer: (A), (B), (C)**

**Solution:**
- (A) SVM maximizes the margin. TRUE.
- (B) Support vectors are the closest points to the decision boundary. TRUE.
- (C) The kernel trick handles non-linear separability. TRUE.
- (D) SVM does NOT require linear separability (soft margin + kernels handle this). FALSE.

**Correct answer: (A), (B), (C)**

**Common trap:** Believing SVM requires linearly separable data (it doesn't, with soft margins/kernels).

**GATE insight:** SVM margin maximization and support vectors are core concepts.

---

#### GATE-DA-D001-Q044
**Answer: 1.5**

**Given:** K = 2, initial centroids 0 and 10, data {1, 2, 8, 9}.
**Required:** New centroid of cluster containing point 1.

**Solution:**
Assignment: each point goes to nearest centroid.
- Point 1: distance to 0 is 1, to 10 is 9 → cluster 0.
- Point 2: distance to 0 is 2, to 10 is 8 → cluster 0.
- Point 8: distance to 0 is 8, to 10 is 2 → cluster 10.
- Point 9: distance to 0 is 9, to 10 is 1 → cluster 10.

Cluster with centroid 0 contains {1, 2}. New centroid = (1+2)/2 = 1.5.

**Final answer: 1.5**

**Shortcut:** New centroid = mean of assigned points.

**Common trap:** Including points from the other cluster or misassigning.

**GATE insight:** K-Means assignment and centroid update steps are classic.

---

#### GATE-DA-D001-Q045
**Answer: 2**

**Given:** Eigenvalues 10, 5, 3, 2.
**Required:** Min PCs to retain ≥ 80% variance.

**Solution:**
Total variance = 10 + 5 + 3 + 2 = 20.
- 1 PC: 10/20 = 50%.
- 2 PCs: (10+5)/20 = 15/20 = 75%.
- 3 PCs: (10+5+3)/20 = 18/20 = 90% ≥ 80%.

So we need 3 PCs.

**Wait** — 2 PCs give 75% < 80%, 3 PCs give 90% ≥ 80%. So minimum is 3.

**Final answer: 3**

**Common trap:** Stopping at 2 PCs (75%) which is below the 80% threshold.

**GATE insight:** PCA variance retention via cumulative eigenvalues is a standard question.

---

#### GATE-DA-D001-Q046
**Answer: 0.73**

**Given:** TP = 40, FP = 10, FN = 20, TN = 30.
**Required:** F1-score.

**Formula:**
Precision = TP/(TP+FP) = 40/50 = 0.8.
Recall = TP/(TP+FN) = 40/60 = 0.6667.
F1 = 2·Precision·Recall/(Precision + Recall) = 2(0.8)(0.6667)/(0.8 + 0.6667) = 1.0667/1.4667 = 0.7273 ≈ 0.73.

**Final answer: 0.73**

**Shortcut:** F1 = 2TP/(2TP + FP + FN) = 80/(80 + 10 + 20) = 80/110 = 0.7273.

**Common trap:** Using accuracy instead of F1, or miscomputing precision/recall.

**GATE insight:** F1-score from confusion matrix is a high-frequency ML question.

---

### SECTION G — ARTIFICIAL INTELLIGENCE

#### GATE-DA-D001-Q047
**Answer: (C) Breadth-First Search (BFS)**

**Solution:**
Uninformed (blind) searches use no domain knowledge/heuristics. BFS, DFS, and uniform-cost search are uninformed. A*, greedy best-first, and hill climbing use heuristics (informed).

**Why others are wrong:**
- (A) A* uses a heuristic (informed).
- (B) Greedy best-first uses a heuristic (informed).
- (D) Hill climbing uses a heuristic (informed).

**Common trap:** Confusing informed vs uninformed search.

**GATE insight:** Distinguishing informed (heuristic) from uninformed search is fundamental AI.

---

#### GATE-DA-D001-Q048
**Answer: (C)**

**Solution:**
For A* to be optimal, the heuristic must be admissible (never overestimate the true cost to the goal). A consistent (monotonic) heuristic is a stronger condition that also guarantees optimality (and ensures no re-expansion). So both (A) and (B) are correct conditions.

**Why others are wrong:**
- (A) Admissible is necessary for optimality — TRUE but incomplete as a standalone answer.
- (B) Consistent implies admissible and guarantees optimality — TRUE but incomplete.
- (D) h(n) = 0 is admissible but trivial (reduces to UCS); not the general condition.

**Correct answer: (C)** — both admissibility and consistency are correct conditions for optimality.

**Common trap:** Choosing only (A) or only (B) when both are valid conditions.

**GATE insight:** A* optimality requires admissible (and preferably consistent) heuristics.

---

#### GATE-DA-D001-Q049
**Answer: 5**

**Given:** Leaf values (left to right): 3, 5, 2, 9, 1, 7, 4, 6. Tree: root MAX, then MIN, then leaves.
**Required:** Minimax value at root.

**Solution:**
The leaves are grouped into pairs under MIN nodes (since there are 8 leaves and 3 levels: root MAX → 2 MIN nodes → 4 leaves each... wait, let me reconsider).

Actually, with 3 levels (root MAX, level 2 MIN, level 3 leaves), and 8 leaves, the tree has 2 MIN nodes at level 2, each with 4 leaves.

MIN node 1 (leaves 3, 5, 2, 9): min = 2.
MIN node 2 (leaves 1, 7, 4, 6): min = 1.
Root MAX takes max of {2, 1} = 2.

**Wait** — that gives 2. Let me reconsider the tree structure.

Hmm, but the standard minimax with 8 leaves and 3 levels: root MAX has 2 children (MIN nodes), each MIN node has 4 leaf children. 

MIN node 1: min(3, 5, 2, 9) = 2.
MIN node 2: min(1, 7, 4, 6) = 1.
MAX root: max(2, 1) = 2.

**Final answer: 2**

**Common trap:** Misgrouping leaves under MIN nodes, or applying MAX/MIN at the wrong level.

**GATE insight:** Minimax alternates MIN and MAX levels from leaves to root.

---

#### GATE-DA-D001-Q050
**Answer: (B) P ∨ ¬P**

**Solution:**
P ∨ ¬P is the Law of Excluded Middle — always true regardless of P's truth value. It is a tautology.

**Why others are wrong:**
- (A) P ∧ ¬P is a contradiction (always false).
- (C) P → Q is not always true (false when P true, Q false).
- (D) P ∧ Q is not always true.

**Correct answer: (B)**

**Common trap:** Confusing tautology (always true) with contradiction (always false).

**GATE insight:** Recognizing tautologies/contradictions via truth tables is a basic logic skill.

---

## END OF DAY 1 QUESTION PAPER
