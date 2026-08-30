# GATE DA — DAY 1 — ROUND 2: ANSWERS & SOLUTIONS (Q001–Q025)

---

## GATE-DA-D001-Q001 — Conditional Probability
**ANSWER: (B) 1/6**

**Direct explanation:** We condition on the event "sum = 7". Among the outcomes where the sum is 7, exactly one has first roll = 3.

**Step-by-step:**
1. Sample space for two rolls: 36 equally likely outcomes.
2. Outcomes with sum 7: (1,6), (2,5), (3,4), (4,3), (5,2), (6,1) → 6 outcomes.
3. Favorable: first roll = 3 → (3,4) → 1 outcome.
4. P(first = 3 | sum = 7) = 1/6.

**Formula:** P(A|B) = P(A ∩ B) / P(B) = (1/36) / (6/36) = 1/6.

**Why others are wrong:**
- (A) 1/36: P(first=3 AND sum=7) unconditioned — the trap of forgetting to condition.
- (C) 1/12: a guess based on miscounting.
- (D) 1/3: confusing with P(first=3) among 3 possible first-roll values.

**Shortcut:** Once you know the sum is 7, the first roll is uniformly one of {1,…,6} — each equally likely → 1/6.

**Common trap:** Forgetting to restrict the sample space to the conditioning event.

**GATE insight:** GATE DA loves conditional probability where the conditioning event is a sum/product of dice. Always enumerate the reduced sample space first.

---

## GATE-DA-D001-Q002 — Bayes Theorem
**ANSWER: 0.0876**

**Direct explanation:** This is a classic Bayes application: P(D|+) = P(+|D)P(D) / P(+).

**Step-by-step:**
1. Given: P(D) = 0.01, P(+|D) = 0.95, P(−|no D) = 0.90 → P(+|no D) = 0.10.
2. P(+) = P(+|D)P(D) + P(+|no D)P(no D) = 0.95(0.01) + 0.10(0.99) = 0.0095 + 0.099 = 0.1085.
3. P(D|+) = 0.0095 / 0.1085 = 95/1085 = 19/217 ≈ 0.08756.

**Formula:** P(D|+) = P(+|D)·P(D) / [P(+|D)·P(D) + P(+|no D)·P(no D)].

**Final answer:** 0.0876 (accept 0.087–0.088).

**Shortcut:** With low prevalence, even a "good" test gives a surprisingly low posterior — the denominator is dominated by false positives (0.099 vs 0.0095).

**Common trap:** Using specificity directly instead of converting to the false-positive rate 1 − specificity = 0.10.

**GATE insight:** Sensitivity/specificity/prevalence Bayes problems are a recurring GATE DA pattern. Memorize: P(+|no D) = 1 − specificity.

---

## GATE-DA-D001-Q003 — Expectation
**ANSWER: 2.1**

**Direct explanation:** E[X] = Σ x·P(X=x).

**Step-by-step:**
1. E[X] = 1(0.2) + 2(0.5) + 3(0.3)
2. = 0.2 + 1.0 + 0.9 = 2.1.

**Formula:** E[X] = Σᵢ xᵢ pᵢ.

**Shortcut:** Weighted average — the probabilities sum to 1, so it's just a weighted mean.

**Common trap:** Forgetting to multiply each value by its probability (averaging 1,2,3 → 2 is wrong).

**GATE insight:** Expectation is the single most-tested concept in DA statistics. Also know E[X²] and Var(X) = E[X²] − (E[X])².

---

## GATE-DA-D001-Q004 — Binomial Distribution
**ANSWER: 0.3456**

**Direct explanation:** X ~ Binomial(5, 0.6); P(X=3) = C(5,3)(0.6)³(0.4)².

**Step-by-step:**
1. C(5,3) = 10.
2. (0.6)³ = 0.216; (0.4)² = 0.16.
3. P = 10 × 0.216 × 0.16 = 0.3456.

**Formula:** P(X=k) = C(n,k) pᵏ (1−p)^(n−k).

**Final answer:** 0.3456.

**Shortcut:** For n=5, Pascal row 1 5 10 10 5 1 gives coefficients directly.

**Common trap:** Using (0.6)²(0.4)³ (swapping exponents) or forgetting the binomial coefficient.

**GATE insight:** Know when binomial applies: fixed n, independent trials, constant p, two outcomes.

---

## GATE-DA-D001-Q005 — Poisson Distribution
**ANSWER: 0.2240**

**Direct explanation:** X ~ Poisson(3); P(X=2) = e⁻³·3²/2!.

**Step-by-step:**
1. P(X=2) = e⁻³ × 9 / 2 = 4.5·e⁻³.
2. e⁻³ ≈ 0.049787.
3. P = 4.5 × 0.049787 ≈ 0.22404.

**Formula:** P(X=k) = e^(−λ)·λᵏ/k!.

**Final answer:** 0.2240 (accept 0.224).

**Shortcut:** For λ = 3, memorize e⁻³ ≈ 0.0498.

**Common trap:** Using λ = 3 for a different time window — Poisson scales with the interval (e.g., 2 hours → λ = 6).

**GATE insight:** Poisson is the limit of Binomial(n, p) as n→∞, p→0 with np = λ. GATE DA tests both the formula and this connection.

---

## GATE-DA-D001-Q006 — Standard Normal Distribution
**ANSWER: (B) 0.50**

**Direct explanation:** The standard normal is symmetric about 0, so exactly half the mass lies below 0.

**Step-by-step:**
1. N(0,1) PDF is symmetric: f(z) = f(−z).
2. P(Z ≤ 0) = P(Z ≥ 0) = 0.5.

**Why others are wrong:**
- (C) 0.68: P(|Z| ≤ 1) — the 68-95-99.7 rule for ±1σ.
- (D) 0.95: P(|Z| ≤ 1.96) ≈ 0.95.
- (A) 0.25: no standard meaning.

**Shortcut:** Any symmetric continuous distribution has median = mean; P(X ≤ mean) = 0.5.

**Common trap:** Confusing P(Z ≤ 0) with P(Z ≤ 1) ≈ 0.8413.

**GATE insight:** Symmetry arguments avoid table lookups. Know 68-95-99.7 and z = 1.96 ↔ 0.95.

---

## GATE-DA-D001-Q007 — Variance & Covariance Properties
**ANSWER: (A), (C), (D)**

**Direct explanation:** (B) is false — Var(X+Y) = Var(X) + Var(Y) only when X, Y are uncorrelated (e.g., independent).

**Step-by-step:**
- (A) Var(aX) = a²Var(X) ✓ — scaling by a scales deviations by a, so variance scales by a².
- (B) Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y) ✗ — the covariance term is missing.
- (C) Cov(X,X) = E[XX] − E[X]E[X] = E[X²] − (E[X])² = Var(X) ✓.
- (D) Cov(aX, bY) = ab·Cov(X,Y) ✓ — bilinearity of covariance.

**Common trap:** Treating variance as linear. It is NOT: Var(aX + bY) = a²Var(X) + b²Var(Y) + 2ab·Cov(X,Y).

**GATE insight:** MSQ trap — "always true" statements about variance/covariance. The covariance term is the classic omission.

---

## GATE-DA-D001-Q008 — Joint & Conditional Distributions
**ANSWER: 0.4286**

**Direct explanation:** Normalize the joint PMF, then apply the conditional formula.

**Step-by-step:**
1. Sum of c(x+y) over all 6 cells: x=1: (2+3+4) = 9; x=2: (3+4+5) = 12; total = 21 → c = 1/21.
2. P(X=1, Y=2) = c(1+2) = 3/21 = 1/7.
3. P(Y=2) = P(X=1,Y=2) + P(X=2,Y=2) = 3/21 + 4/21 = 7/21 = 1/3.
4. P(X=1|Y=2) = (1/7)/(1/3) = 3/7 ≈ 0.4286.

**Formula:** P(X=x|Y=y) = P(X=x, Y=y) / P(Y=y).

**Final answer:** 0.4286 (accept 3/7).

**Shortcut:** Build the 2×3 table of joint probabilities first; conditional = cell / row total.

**Common trap:** Forgetting to normalize c first, or dividing by the wrong marginal.

**GATE insight:** Joint PMF normalization + conditional extraction is a staple DA numerical. Always draw the table.

---

## GATE-DA-D001-Q009 — Maximum Likelihood Estimation
**ANSWER: 0.5**

**Direct explanation:** For Exponential(λ), the MLE is λ̂ = 1/x̄ = n/Σxᵢ.

**Step-by-step:**
1. Likelihood: L(λ) = Π λe^(−λxᵢ) = λⁿ e^(−λΣxᵢ).
2. Log-likelihood: ℓ(λ) = n·ln λ − λ·Σxᵢ.
3. dℓ/dλ = n/λ − Σxᵢ = 0 → λ̂ = n/Σxᵢ = 1/x̄.
4. λ̂ = 10/20 = 0.5.

**Formula:** λ̂_MLE = n / Σxᵢ.

**Final answer:** 0.5.

**Shortcut:** For exponential, MLE of rate = 1/sample mean. No need to re-derive in the exam.

**Common trap:** Confusing rate λ with mean 1/λ. The sample mean is 2, so the rate estimate is 1/2, not 2.

**GATE insight:** MLE for standard families (Bernoulli, Poisson, Normal, Exponential) is high-yield. Know the closed forms.

---

## GATE-DA-D001-Q010 — Central Limit Theorem
**ANSWER: (B) N(μ, σ²/n)**

**Direct explanation:** The sample mean has mean μ and variance σ²/n; CLT says its distribution is approximately normal for large n.

**Step-by-step:**
1. E[X̄ₙ] = μ (unbiased).
2. Var(X̄ₙ) = Var((1/n)ΣXᵢ) = (1/n²)·n·σ² = σ²/n (independence).
3. CLT: X̄ₙ ≈ N(μ, σ²/n).

**Why others are wrong:**
- (A) N(μ, σ²): variance of a single observation, not the mean.
- (C) N(0,1): the standardized version (X̄ₙ − μ)/(σ/√n), not X̄ₙ itself.
- (D) N(μ, σ²/√n): wrong variance scaling — variance divides by n, not √n (√n divides the SD).

**Shortcut:** Mean of n i.i.d. variables: variance shrinks by n. Standard deviation shrinks by √n.

**Common trap:** Mixing up σ/√n (SD of the mean) with σ²/n (variance of the mean).

**GATE insight:** CLT questions test the exact variance scaling. Remember: Var(X̄) = σ²/n.

---

## GATE-DA-D001-Q011 — Matrix Rank
**ANSWER: (B) 1**

**Direct explanation:** Row 2 = 2 × Row 1, so the rows are linearly dependent; only one independent row.

**Step-by-step:**
1. Rank = number of linearly independent rows (or columns).
2. Row 2 = 2·Row 1 → rank ≤ 1.
3. Row 1 ≠ 0 → rank = 1.

**Why others are wrong:**
- (C) 2: assumes full rank without checking dependence.
- (A) 0: only for the zero matrix.
- (D) nonsense — rank is defined for all matrices.

**Shortcut:** If one row/column is a scalar multiple of another, rank drops by at least 1.

**Common trap:** Counting rows instead of independent rows.

**GATE insight:** Rank questions pair with determinant = 0 (singular) for dependent rows. det([[1,2],[2,4]]) = 0 confirms rank < 2.

---

## GATE-DA-D001-Q012 — Determinant
**ANSWER: −2**

**Direct explanation:** For a 2×2 matrix, det = ad − bc.

**Step-by-step:**
1. det([[1,2],[3,4]]) = 1·4 − 2·3.
2. = 4 − 6 = −2.

**Formula:** det([[a,b],[c,d]]) = ad − bc.

**Final answer:** −2.

**Shortcut:** Cross-multiply diagonals: main diagonal minus anti-diagonal.

**Common trap:** Sign errors — the anti-diagonal product is subtracted.

**GATE insight:** Determinants feed into invertibility (det ≠ 0), eigenvalues, and area/volume interpretations.

---

## GATE-DA-D001-Q013 — Matrix Inverse
**ANSWER: −2**

**Direct explanation:** A⁻¹ = (1/det)·adj(A); for 2×2, swap diagonal, negate off-diagonal.

**Step-by-step:**
1. det(A) = 1·4 − 2·3 = −2.
2. adj(A) = [[4, −2], [−3, 1]].
3. A⁻¹ = (1/−2)·[[4,−2],[−3,1]] = [[−2, 1], [3/2, −1/2]].
4. (1,1) entry = −2.

**Formula:** A⁻¹ = (1/(ad−bc))·[[d, −b], [−c, a]].

**Final answer:** −2.

**Shortcut:** For 2×2: swap a↔d, negate b and c, divide by det.

**Common trap:** Forgetting to divide by the determinant, or negating the wrong entries.

**GATE insight:** Verify with AA⁻¹ = I. Also: (1,1) entry of A⁻¹ = d/det(A) — a fast formula.

---

## GATE-DA-D001-Q014 — Eigenvalues
**ANSWER: 3**

**Direct explanation:** Solve det(A − λI) = 0.

**Step-by-step:**
1. det([[2−λ, 1], [1, 2−λ]]) = (2−λ)² − 1 = 0.
2. (2−λ)² = 1 → 2−λ = ±1 → λ = 1 or λ = 3.
3. Larger eigenvalue = 3.

**Formula:** det(A − λI) = 0.

**Final answer:** 3.

**Shortcut:** For symmetric 2×2 [[a,b],[b,a]]: eigenvalues are a+b and a−b → 3 and 1.

**Common trap:** Sign errors when expanding (2−λ)².

**GATE insight:** Symmetric matrices have real eigenvalues; trace = sum of eigenvalues = 4 (check: 3+1 = 4 ✓).

---

## GATE-DA-D001-Q015 — Linear Independence
**ANSWER: (A), (C)**

**Direct explanation:** A set is independent iff no vector is a linear combination of the others (equivalently, the only solution to Σcᵢvᵢ = 0 is all cᵢ = 0).

**Step-by-step:**
- (A) Standard basis — clearly independent ✓.
- (B) (2,2,0) = 2·(1,1,0) → dependent ✗.
- (C) det of the 3×3 matrix with these as rows: 1(6−1) − 2(4−3) + 3(2−9) = 5 − 2 − 21 = −18 ≠ 0 → independent ✓.
- (D) (2,0,0) = 2·(1,0,0) → dependent ✗.

**Shortcut:** For n vectors in Rⁿ, compute the determinant — nonzero ⟺ independent.

**Common trap:** Assuming 3 vectors in R³ are automatically independent (they need not be).

**GATE insight:** Independence ↔ determinant ↔ rank are the same idea in different clothes.

---

## GATE-DA-D001-Q016 — Positive Definite Matrices
**ANSWER: (C) [[2, 0], [0, 3]]**

**Direct explanation:** A symmetric matrix is positive definite iff all eigenvalues are strictly positive.

**Step-by-step:**
- (A) [[1,0],[0,−1]]: eigenvalues 1, −1 → indefinite ✗.
- (B) [[1,2],[2,1]]: eigenvalues 3, −1 → indefinite ✗.
- (C) [[2,0],[0,3]]: eigenvalues 2, 3 → positive definite ✓.
- (D) [[0,0],[0,1]]: eigenvalue 0 → positive SEMI-definite, not definite ✗.

**Shortcut:** Diagonal matrix → PD iff all diagonal entries > 0.

**Common trap:** Confusing positive semi-definite (λ ≥ 0) with positive definite (λ > 0).

**GATE insight:** PD matrices ⟺ positive quadratic forms ⟺ convex quadratic objectives — the bridge between linear algebra and optimization/ML.

---

## GATE-DA-D001-Q017 — Rank-Nullity Theorem
**ANSWER: 2**

**Direct explanation:** For an m×n matrix, rank + nullity = n (number of columns).

**Step-by-step:**
1. A is 4×5 → n = 5 columns.
2. rank(A) = 3.
3. nullity = dim(null space) = 5 − 3 = 2.

**Formula:** rank(A) + nullity(A) = number of columns.

**Final answer:** 2.

**Shortcut:** Nullity = columns − rank. The number of rows is irrelevant here.

**Common trap:** Using rows (4) instead of columns (5).

**GATE insight:** Rank-nullity is the key to counting solutions of Ax = b: nullity = degrees of freedom of the solution set.

---

## GATE-DA-D001-Q018 — Eigenvalues & Trace
**ANSWER: 38**

**Direct explanation:** If A has eigenvalues λᵢ, then A² has eigenvalues λᵢ², and trace = sum of eigenvalues.

**Step-by-step:**
1. Eigenvalues of A²: 2², 3², 5² = 4, 9, 25.
2. trace(A²) = 4 + 9 + 25 = 38.

**Formula:** trace(Aᵏ) = Σ λᵢᵏ.

**Final answer:** 38.

**Shortcut:** trace(A²) = Σλᵢ² — no need to compute A² explicitly.

**Common trap:** Computing trace(A)² = (2+3+5)² = 100 instead of trace(A²) = 38.

**GATE insight:** Trace/determinant of matrix powers via eigenvalues is a fast, high-yield trick. Also: det(A) = Πλᵢ = 30 here.

---

## GATE-DA-D001-Q019 — Limits
**ANSWER: 3**

**Direct explanation:** Standard limit: lim_{x→0} sin(kx)/x = k.

**Step-by-step:**
1. lim_{x→0} sin(3x)/x = lim_{x→0} 3·[sin(3x)/(3x)].
2. Let u = 3x; as x→0, u→0, and lim_{u→0} sin(u)/u = 1.
3. Result: 3·1 = 3.

**Formula:** lim_{x→0} sin(ax)/x = a.

**Final answer:** 3.

**Shortcut:** Coefficient of x inside sin is the answer.

**Common trap:** Answering 1 (forgetting the 3) or 0 (thinking sin(0) = 0 kills the limit).

**GATE insight:** Standard limits (sin x/x, (eˣ−1)/x, (1+x)^(1/x)) are free marks — memorize them.

---

## GATE-DA-D001-Q020 — Gradient
**ANSWER: 7**

**Direct explanation:** ∂f/∂x treats y as constant.

**Step-by-step:**
1. f(x,y) = x²y + 3x.
2. ∂f/∂x = 2xy + 3.
3. At (1,2): 2(1)(2) + 3 = 4 + 3 = 7.

**Formula:** ∇f = (∂f/∂x, ∂f/∂y).

**Final answer:** 7.

**Shortcut:** Differentiate term by term; treat the other variable as a constant coefficient.

**Common trap:** Differentiating x²y as 2xy + x² (product rule confusion — y is constant, not a function of x).

**GATE insight:** Gradients are the engine of gradient descent — the single most important ML connection in calculus.

---

## GATE-DA-D001-Q021 — Hessian Matrix
**ANSWER: −5**

**Direct explanation:** Hessian H = [[f_xx, f_xy], [f_yx, f_yy]]; here it's constant, so det is constant.

**Step-by-step:**
1. f_x = 2x + 3y; f_y = 3x + 2y.
2. f_xx = 2, f_yy = 2, f_xy = 3, f_yx = 3.
3. H = [[2, 3], [3, 2]].
4. det(H) = 2·2 − 3·3 = 4 − 9 = −5.

**Formula:** det(H) = f_xx·f_yy − (f_xy)².

**Final answer:** −5.

**Shortcut:** For quadratic forms, the Hessian is the symmetric matrix of the quadratic coefficients (times 2 on diagonal).

**Common trap:** Forgetting the mixed partials or their sign in the determinant.

**GATE insight:** det(H) < 0 → saddle point (indefinite). This function has a saddle at (0,0), not a min/max.

---

## GATE-DA-D001-Q022 — Convex Functions
**ANSWER: (A), (B), (D)**

**Direct explanation:** A twice-differentiable function is convex iff f″(x) ≥ 0 everywhere.

**Step-by-step:**
- (A) f(x) = x²: f″ = 2 > 0 ✓.
- (B) f(x) = eˣ: f″ = eˣ > 0 ✓.
- (C) f(x) = −x²: f″ = −2 < 0 → strictly concave ✗.
- (D) f(x) = |x|: not differentiable at 0, but convex by the definition (or f″ ≥ 0 a.e. with the kink) ✓.

**Shortcut:** Second derivative ≥ 0 ⟺ convex (for twice-differentiable functions).

**Common trap:** Thinking |x| is non-convex because it's not differentiable — convexity doesn't require differentiability.

**GATE insight:** Convexity ⟹ every local minimum is global — the reason ML loss functions are designed convex.

---

## GATE-DA-D001-Q023 — Lagrange Multipliers
**ANSWER: 2**

**Direct explanation:** Minimize x² + y² on the line x + y = 2. The closest point to the origin on that line is (1,1).

**Step-by-step:**
1. L = x² + y² + λ(x + y − 2).
2. ∂L/∂x = 2x + λ = 0 → x = −λ/2.
3. ∂L/∂y = 2y + λ = 0 → y = −λ/2 → x = y.
4. Constraint: x + y = 2 → 2x = 2 → x = y = 1.
5. f(1,1) = 1 + 1 = 2.

**Formula:** ∇f = λ∇g with g(x,y) = x + y − 2 = 0.

**Final answer:** 2.

**Shortcut:** By symmetry, the optimum of a symmetric objective on a symmetric constraint lies at x = y.

**Common trap:** Forgetting to substitute back into f (answering x = 1 instead of f = 2).

**GATE insight:** Geometric view: min distance from origin to line x+y=2 is 2/√2 = √2, so min of squared distance = 2. Same answer, faster.

---

## GATE-DA-D001-Q024 — Gradient Descent
**ANSWER: 2.4**

**Direct explanation:** One step: x₁ = x₀ − η·f′(x₀).

**Step-by-step:**
1. f(x) = x² → f′(x) = 2x.
2. f′(3) = 6.
3. x₁ = 3 − 0.1·6 = 3 − 0.6 = 2.4.

**Formula:** x_{t+1} = x_t − η·∇f(x_t).

**Final answer:** 2.4.

**Shortcut:** For f(x) = x², the update is x_{t+1} = (1 − 2η)x_t — a geometric decay.

**Common trap:** Using f(x₀) = 9 instead of the derivative f′(x₀) = 6.

**GATE insight:** Gradient descent on quadratics converges linearly; step size η < 1/λ_max guarantees convergence. This is the core of training neural nets.

---

## GATE-DA-D001-Q025 — Loops & Output
**ANSWER: 15**

**Direct explanation:** The loop sums 1 + 2 + 3 + 4 + 5.

**Step-by-step:**
1. range(1, 6) yields 1, 2, 3, 4, 5 (stop value exclusive).
2. s accumulates: 1, 3, 6, 10, 15.
3. print(s) → 15.

**Formula:** Sum 1..n = n(n+1)/2 = 5·6/2 = 15.

**Final answer:** 15.

**Shortcut:** n(n+1)/2 for the sum of the first n integers.

**Common trap:** Including 6 (forgetting range's stop is exclusive) → would give 21.

**GATE insight:** Python range semantics (exclusive stop) is a favorite DA trap. Also watch range(1,6,2), negative steps, etc.

---

**Continue to `solutions-2.md` for Q026–Q050.**# GATE DA — DAY 1 — ROUND 2: ANSWERS & SOLUTIONS (Q026–Q050)

---

## GATE-DA-D001-Q026 — Recursion
**ANSWER: 8**

**Direct explanation:** f(n) computes a Fibonacci-like sequence with f(0) = f(1) = 1.

**Step-by-step:**
1. f(0) = 1, f(1) = 1 (base cases).
2. f(2) = f(1) + f(0) = 2.
3. f(3) = f(2) + f(1) = 3.
4. f(4) = f(3) + f(2) = 5.
5. f(5) = f(4) + f(3) = 8.

**Final answer:** 8.

**Shortcut:** This is the Fibonacci sequence shifted: 1, 1, 2, 3, 5, 8, …

**Common trap:** Using standard Fibonacci f(0) = 0 (giving f(5) = 5) — here the base case returns 1 for both n=0 and n=1.

**GATE insight:** Trace recursion bottom-up from base cases. Also know: this recursion has exponential time complexity O(2ⁿ) without memoization.

---

## GATE-DA-D001-Q027 — Complexity Analysis
**ANSWER: (B) O(log n)**

**Direct explanation:** Each comparison halves the search interval.

**Step-by-step:**
1. After k comparisons, the interval size is n/2ᵏ.
2. Stop when n/2ᵏ ≤ 1 → k = ⌈log₂ n⌉.
3. Worst case: O(log n).

**Why others are wrong:**
- (A) O(n): linear search.
- (C) O(n log n): sorting algorithms like merge sort.
- (D) O(1): hash table lookup (average case).

**Shortcut:** "Halving" algorithms → log n.

**Common trap:** Confusing binary search (O(log n)) with sorting (O(n log n)).

**GATE insight:** Complexity of standard operations (search, sort, insert) is guaranteed marks. Know: BST search O(h), hash O(1) avg, binary search O(log n).

---

## GATE-DA-D001-Q028 — Binary Search Trees
**ANSWER: (B) Ascending (sorted) order of keys**

**Direct explanation:** In-order traversal = left subtree, node, right subtree; the BST invariant (left < node < right) makes this sorted.

**Step-by-step:**
1. In-order: visit left subtree → root → right subtree.
2. All keys in the left subtree < root < all keys in the right subtree.
3. Recursively, the sequence is ascending.

**Why others are wrong:**
- (C) Descending: that's reverse in-order (right, node, left).
- (D) Level order: that's BFS.
- (A) Random: no.

**Shortcut:** In-order of BST = sorted output. This is why BSTs support sorted iteration.

**Common trap:** Confusing in-order with pre-order (node, left, right) or post-order (left, right, node).

**GATE insight:** Traversal orders and their uses (in-order → sorted; pre-order → serialize; post-order → delete) are classic DA questions.

---

## GATE-DA-D001-Q029 — Stack
**ANSWER: (A) 1**

**Direct explanation:** Stack is LIFO: last pushed is first popped.

**Step-by-step:**
1. Push 1 → [1]; push 2 → [1,2]; push 3 → [1,2,3] (top = 3).
2. Pop → removes 3 → [1,2] (top = 2).
3. Pop → removes 2 → [1] (top = 1).

**Why others are wrong:**
- (B) 2: would be the top after only ONE pop.
- (C) 3: the top before any pop.
- (D) Empty: would need three pops.

**Shortcut:** LIFO — the last element pushed is the first out.

**Common trap:** Applying FIFO (queue) behavior instead of LIFO.

**GATE insight:** Stack/queue behavior under push/pop sequences is a frequent DA NAT. Trace the stack state explicitly.

---

## GATE-DA-D001-Q030 — Bubble Sort
**ANSWER: 2**

**Direct explanation:** First pass compares adjacent pairs and swaps when out of order.

**Step-by-step:**
1. Compare (5,3): 5 > 3 → swap → [3,5,8,1]. Swap #1.
2. Compare (5,8): 5 < 8 → no swap.
3. Compare (8,1): 8 > 1 → swap → [3,5,1,8]. Swap #2.
4. End of pass 1: largest element 8 is in place. Total swaps = 2.

**Final answer:** 2.

**Shortcut:** Count inversions resolved at the boundary — but tracing adjacent comparisons is safest.

**Common trap:** Counting comparisons (3) instead of swaps (2).

**GATE insight:** Bubble sort: n−1 passes, worst case O(n²) swaps. Trace-based questions reward careful bookkeeping.

---

## GATE-DA-D001-Q031 — Recurrence Relations
**ANSWER: 80**

**Direct explanation:** Expand the recurrence (or use Master Theorem: T(n) = Θ(n log n)).

**Step-by-step:**
1. T(2) = 2T(1) + 2 = 2(1) + 2 = 4.
2. T(4) = 2T(2) + 4 = 2(4) + 4 = 12.
3. T(8) = 2T(4) + 8 = 2(12) + 8 = 32.
4. T(16) = 2T(8) + 16 = 2(32) + 16 = 80.

**Formula:** Master Theorem case 2: T(n) = 2T(n/2) + n → Θ(n log n); T(16) = 16·log₂16 = 64 only if T(1) = 0 — with T(1) = 1 the constant shifts the exact value to 80.

**Final answer:** 80.

**Shortcut:** Unroll: T(n) = n·log₂n + n·T(1) = 16·4 + 16 = 80.

**Common trap:** Using the asymptotic Θ(n log n) as if it were the exact value (64 ≠ 80).

**GATE insight:** Distinguish asymptotic complexity from exact values. GATE DA asks both — read the question carefully.

---

## GATE-DA-D001-Q032 — BFS on Graphs
**ANSWER: (A) The BFS tree contains exactly n − 1 edges.**

**Direct explanation:** A BFS tree on a connected graph with n vertices is a spanning tree.

**Step-by-step:**
- (A) ✓ — BFS visits every vertex exactly once; each vertex (except the root) is added via exactly one tree edge → n − 1 edges.
- (B) ✗ — BFS uses a QUEUE; DFS uses a stack.
- (C) ✗ — BFS finds shortest paths only in UNWEIGHTED graphs; for weighted graphs use Dijkstra.
- (D) ✗ — BFS visits vertices in NON-DECREASING order of distance (ties allowed), not strictly decreasing.

**Common trap:** Confusing BFS (queue) with DFS (stack), or extending BFS shortest-path guarantees to weighted graphs.

**GATE insight:** BFS/DFS properties (queue vs stack, tree edges, shortest paths in unweighted graphs) are core DA graph questions.

---

## GATE-DA-D001-Q033 — SQL Aggregation
**ANSWER: (A) The number of students whose grade is 'A'**

**Direct explanation:** COUNT(*) counts rows; WHERE filters rows before counting.

**Step-by-step:**
1. WHERE grade = 'A' selects only rows with grade 'A'.
2. COUNT(*) counts those rows.

**Why others are wrong:**
- (B) Total students: would be COUNT(*) without WHERE.
- (C) Distinct grades: would be COUNT(DISTINCT grade).
- (D) Sum of grades: would be SUM(grade) — and grades are strings here anyway.

**Shortcut:** WHERE filters first, then aggregation applies.

**Common trap:** Forgetting that COUNT(*) counts rows, not distinct values.

**GATE insight:** SQL execution order: FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY. Know it cold.

---

## GATE-DA-D001-Q034 — BCNF
**ANSWER: (C) For every non-trivial functional dependency X → A on R, X is a superkey of R.**

**Direct explanation:** BCNF requires every non-trivial FD's left side to be a superkey.

**Step-by-step:**
- (A) Atomic attributes → 1NF, not BCNF.
- (B) Full FD of non-prime attributes on the key → 2NF.
- (D) No transitive dependencies → 3NF.
- (C) ✓ — this is exactly the BCNF definition.

**Common trap:** Confusing BCNF with 3NF. BCNF is stricter: 3NF allows X → A when A is prime even if X is not a superkey; BCNF does not.

**GATE insight:** Normal form ladder: 1NF → 2NF → 3NF → BCNF. Know the exact FD-based definition of each.

---

## GATE-DA-D001-Q035 — Functional Dependencies
**ANSWER: (A), (B), (D)**

**Direct explanation:** Apply Armstrong's axioms to A → B and B → C.

**Step-by-step:**
- (A) A → C ✓ — transitivity: A → B, B → C ⟹ A → C.
- (B) AB → C ✓ — augmentation of B → C with A: B → C ⟹ AB → AC ⟹ AB → C.
- (C) B → A ✗ — no axiom derives this; FDs are not symmetric.
- (D) A → BC ✓ — union rule: A → B and A → C ⟹ A → BC.

**Shortcut:** Transitivity + augmentation + union cover most FD questions.

**Common trap:** Assuming FDs are reversible (B → A from A → B) — they are not.

**GATE insight:** Armstrong's axioms (reflexivity, augmentation, transitivity) and derived rules (union, decomposition, pseudotransitivity) are exam staples.

---

## GATE-DA-D001-Q036 — Relational Algebra
**ANSWER: (B) Selection (σ)**

**Direct explanation:** σ filters rows by a condition — exactly what WHERE does.

**Step-by-step:**
- (A) Projection (π) → SELECT column list.
- (B) Selection (σ) → WHERE ✓.
- (C) Join (⋈) → JOIN.
- (D) Union (∪) → UNION.

**Shortcut:** σ = rows (WHERE), π = columns (SELECT list).

**Common trap:** Confusing selection (rows) with projection (columns).

**GATE insight:** Relational algebra ↔ SQL mapping is guaranteed marks: σ↔WHERE, π↔SELECT-list, ⋈↔JOIN, ×↔CROSS JOIN.

---

## GATE-DA-D001-Q037 — ACID Properties
**ANSWER: (A) Atomicity**

**Direct explanation:** Atomicity = all-or-nothing execution.

**Step-by-step:**
- (A) Atomicity ✓ — transaction is indivisible: commit fully or roll back fully.
- (B) Consistency — DB moves from one valid state to another.
- (C) Isolation — concurrent transactions don't interfere.
- (D) Durability — committed changes survive failures.

**Shortcut:** A = All-or-nothing; C = Constraints preserved; I = Invisible to others; D = Data persists.

**Common trap:** Mixing up atomicity (all-or-nothing) with consistency (valid states).

**GATE insight:** ACID definitions and which property handles which failure scenario are frequent DA questions.

---

## GATE-DA-D001-Q038 — Star Schema
**ANSWER: (B) Fact table**

**Direct explanation:** The fact table sits at the center of a star schema holding measures.

**Step-by-step:**
- (B) Fact table ✓ — quantitative measures (sales amount, quantity), with foreign keys to dimensions.
- (A) Dimension tables — descriptive attributes (time, product, store), at the star's points.
- (C) Lookup table — a generic term, not the star-schema central table.
- (D) Staging table — ETL intermediate storage.

**Shortcut:** Facts = numbers/measures (center); Dimensions = descriptions (points of the star).

**Common trap:** Reversing fact and dimension roles.

**GATE insight:** OLTP vs OLAP, star vs snowflake, fact vs dimension — the data warehousing core. Snowflake = normalized dimensions.

---

## GATE-DA-D001-Q039 — Bias-Variance Tradeoff
**ANSWER: (B) Bias decreases, variance increases**

**Direct explanation:** More complex models fit training data better (lower bias) but become sensitive to training-set noise (higher variance).

**Step-by-step:**
1. Bias = error from simplifying assumptions.
2. Variance = error from sensitivity to training data.
3. Complexity ↑ → bias ↓, variance ↑.

**Why others are wrong:**
- (A) Bias increases: that's for simpler models.
- (C)/(D): both move in opposite directions — that's the tradeoff.

**Shortcut:** Simple model → high bias, low variance. Complex model → low bias, high variance.

**Common trap:** Forgetting the direction: overfitting = low bias + high variance.

**GATE insight:** Bias-variance decomposition of expected error: E[(y − f̂)²] = bias² + variance + irreducible noise. Core DA concept.

---

## GATE-DA-D001-Q040 — Linear Regression
**ANSWER: 1**

**Direct explanation:** w₁ = Cov(x,y)/Var(x) for simple linear regression.

**Step-by-step:**
1. x̄ = 2, ȳ = 3.
2. Σ(x−x̄)(y−ȳ) = (−1)(−1) + (0)(0) + (1)(1) = 2.
3. Σ(x−x̄)² = 1 + 0 + 1 = 2.
4. w₁ = 2/2 = 1.

**Formula:** w₁ = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)².

**Final answer:** 1.

**Shortcut:** The points are perfectly collinear on y = x + 1, so slope = 1 by inspection.

**Common trap:** Using the population formula with 1/n factors inconsistently — they cancel anyway.

**GATE insight:** Closed-form least squares (normal equations) is a DA staple. Also know w₀ = ȳ − w₁x̄ = 1 here.

---

## GATE-DA-D001-Q041 — Logistic Regression
**ANSWER: (A) A probability in (0, 1) obtained via the sigmoid function**

**Direct explanation:** Logistic regression applies sigmoid σ(z) = 1/(1+e^(−z)) to a linear score, yielding a probability.

**Step-by-step:**
1. Linear score: z = w·x + b.
2. σ(z) ∈ (0, 1) — always a valid probability.
3. Threshold (e.g., 0.5) converts to a label at decision time.

**Why others are wrong:**
- (B) Hard label: the model outputs a probability; thresholding is a separate step.
- (C) Unbounded score: that's linear regression.
- (D) Ranking: not the model's output.

**Shortcut:** "Logistic = sigmoid = probability."

**Common trap:** Thinking logistic regression outputs 0/1 directly.

**GATE insight:** Logistic regression + cross-entropy loss + sigmoid is a high-frequency DA topic. Know σ′(z) = σ(z)(1−σ(z)).

---

## GATE-DA-D001-Q042 — k-NN
**ANSWER: 1**

**Direct explanation:** Find the 3 nearest neighbors of (1,0) and take the majority label.

**Step-by-step:**
1. Distances from (1,0): A(0,0): √1 = 1; B(1,1): √1 = 1; C(2,0): √1 = 1; D(3,1): √(4+1) = √5 ≈ 2.24.
2. 3 nearest: A (+), B (+), C (−) — tie broken by majority: 2 vs 1.
3. Predicted label: + → 1.

**Final answer:** 1.

**Shortcut:** With k = 3 and three points equidistant, majority vote decides.

**Common trap:** Including D (4th nearest) or miscounting the tie.

**GATE insight:** k-NN: distance metric + k choice + majority vote. Odd k avoids ties. Also know: k-NN is lazy (no training phase).

---

## GATE-DA-D001-Q043 — Support Vector Machines
**ANSWER: (A), (B), (D)**

**Direct explanation:** Hard-margin SVM maximizes margin with zero training error; the solution is determined by support vectors.

**Step-by-step:**
- (A) ✓ — objective: maximize the margin (minimize ‖w‖).
- (B) ✓ — support vectors are the closest points; they define the margin.
- (C) ✗ — allowing misclassification is the SOFT-margin SVM (slack variables). Hard margin requires perfect separation.
- (D) ✓ — removing non-support-vector points doesn't change the boundary.

**Common trap:** Confusing hard-margin (no errors allowed) with soft-margin (errors allowed with penalty C).

**GATE insight:** SVM: margin = 2/‖w‖, support vectors, kernel trick, hard vs soft margin — all exam favorites.

---

## GATE-DA-D001-Q044 — K-Means Clustering
**ANSWER: 2.33**

**Direct explanation:** Assign each point to the nearest centroid, then recompute centroids as cluster means.

**Step-by-step:**
1. Distances to c₁ = 1: |1−1|=0, |2−1|=1, |4−1|=3, |8−1|=7.
2. Distances to c₂ = 8: |1−8|=7, |2−8|=6, |4−8|=4, |8−8|=0.
3. Assignment: 1→c₁, 2→c₁ (1 < 6), 4→c₁ (3 < 4), 8→c₂.
4. Cluster 1 = {1, 2, 4}; new centroid = (1+2+4)/3 = 7/3 ≈ 2.33.

**Final answer:** 2.33 (accept 7/3).

**Shortcut:** The cluster containing point 2 also contains 1 and 4 — average them.

**Common trap:** Assigning 4 to c₂ (|4−8| = 4 vs |4−1| = 3 — 4 is closer to 1).

**GATE insight:** K-means: assignment step + update step, converges to local optimum, sensitive to initialization. Know SSE objective.

---

## GATE-DA-D001-Q045 — PCA
**ANSWER: 2**

**Direct explanation:** Covariance matrix = (1/n)Σ xxᵀ for centered data; trace = total variance.

**Step-by-step:**
1. Data is already centered (mean = (0,0)).
2. Σ xxᵀ over the 4 points: each point contributes [[x², xy], [xy, y²]].
   - (1,1): [[1,1],[1,1]]; (−1,−1): [[1,1],[1,1]]; (1,−1): [[1,−1],[−1,1]]; (−1,1): [[1,−1],[−1,1]].
3. Sum = [[4, 0], [0, 4]].
4. Cov = (1/4)·[[4,0],[0,4]] = [[1, 0], [0, 1]].
5. Trace = 1 + 1 = 2.

**Formula:** trace(Cov) = total variance = (1/n)Σ‖xᵢ‖².

**Final answer:** 2.

**Shortcut:** Total variance = average squared norm of centered points = (1/4)(2+2+2+2) = 2.

**Common trap:** Using n−1 (sample covariance) instead of n, or forgetting data must be centered.

**GATE insight:** PCA = eigen-decomposition of the covariance matrix; eigenvalues = variance explained per component. Here both eigenvalues are 1 (equal variance, no reduction possible).

---

## GATE-DA-D001-Q046 — Precision, Recall, F1
**ANSWER: 0.7273**

**Direct explanation:** F1 = harmonic mean of precision and recall.

**Step-by-step:**
1. Precision = TP/(TP+FP) = 40/50 = 0.8.
2. Recall = TP/(TP+FN) = 40/60 = 2/3 ≈ 0.6667.
3. F1 = 2·P·R/(P+R) = 2(0.8)(0.6667)/(0.8+0.6667) = 1.0667/1.4667 ≈ 0.7273.

**Formula:** F1 = 2PR/(P+R) = 2TP/(2TP + FP + FN).

**Final answer:** 0.7273 (accept 0.727).

**Shortcut:** F1 = 2TP/(2TP + FP + FN) = 80/(80+10+20) = 80/110 = 0.7273 — one formula, no intermediate rounding.

**Common trap:** Using accuracy (70/100 = 0.7) instead of F1, or rounding intermediates too early.

**GATE insight:** Confusion-matrix metrics (accuracy, precision, recall, F1, specificity, ROC-AUC) are guaranteed DA questions. Use the combined F1 formula to avoid rounding drift.

---

## GATE-DA-D001-Q047 — Uninformed Search
**ANSWER: (B) Breadth-First Search**

**Direct explanation:** BFS explores level by level, so the first goal found is at minimum depth — optimal for unweighted graphs.

**Step-by-step:**
- (B) ✓ — complete (finds a goal if one exists) and optimal for unweighted graphs (shallowest goal).
- (A) DFS — complete only with cycle checking; NOT optimal (may find a deep goal first).
- (C) Greedy best-first — informed, not guaranteed optimal.
- (D) Hill climbing — incomplete, gets stuck in local optima.

**Shortcut:** "Shallowest goal, unweighted" → BFS (queue). "Any goal, memory-light" → DFS (stack).

**Common trap:** Saying DFS is optimal — it is not.

**GATE insight:** Search properties table (complete? optimal? time? space?) for BFS/DFS/UCS/IDDFS is a DA favorite. BFS: O(b^d) time and space.

---

## GATE-DA-D001-Q048 — A* Heuristics
**ANSWER: (A) h(n) ≤ h*(n), where h*(n) is the true cost of the optimal path from n to the goal**

**Direct explanation:** Admissibility = never overestimate the remaining cost.

**Step-by-step:**
- (A) ✓ — h never overestimates → A* with tree search is optimal.
- (B) Overestimating → can prune the optimal path → not admissible.
- (C) Perfect heuristic — admissible but a special case, not the definition.
- (D) h = 0 — admissible (trivially) but useless; not the definition.

**Shortcut:** "Admissible = optimistic = never overestimate."

**Common trap:** Confusing admissibility (h ≤ h*) with consistency (h(n) ≤ c(n,n′) + h(n′)). Consistency implies admissibility (for non-negative costs), not vice versa.

**GATE insight:** A* optimality requires admissible heuristic (tree search) or consistent heuristic (graph search). Straight-line distance is the classic admissible heuristic.

---

## GATE-DA-D001-Q049 — Minimax
**ANSWER: 3**

**Direct explanation:** MIN nodes take the minimum of children; MAX nodes take the maximum.

**Step-by-step:**
1. Left MIN child: min(3, 9) = 3.
2. Right MIN child: min(5, 1) = 1.
3. Root MAX: max(3, 1) = 3.

**Formula:** Minimax value = MAX over MIN over leaves.

**Final answer:** 3.

**Shortcut:** Work bottom-up: collapse MIN nodes first, then MAX.

**Common trap:** Reversing the roles (taking max at MIN nodes) → would give 9.

**GATE insight:** Minimax + alpha-beta pruning (which prunes the 9 and the 1 here) is a core AI game-playing topic. Alpha-beta visits fewer nodes with the same result.

---

## GATE-DA-D001-Q050 — Propositional Logic
**ANSWER: (B) P ∨ ¬P**

**Direct explanation:** The law of excluded middle — true under every assignment.

**Step-by-step:**
- (A) P ∧ ¬P — contradiction: false under every assignment.
- (B) P ∨ ¬P — tautology ✓: if P is true, true; if P is false, ¬P is true.
- (C) P → ¬P ≡ ¬P ∨ ¬P ≡ ¬P — false when P is true.
- (D) P ∧ Q — false when either is false.

**Shortcut:** A formula is a tautology iff its negation is unsatisfiable. Or check the truth table.

**Common trap:** Thinking P → ¬P is a tautology — it's equivalent to ¬P, which is contingent.

**GATE insight:** Tautology/contradiction/contingency classification via truth tables or equivalences is a DA logic staple. Know De Morgan, implication rewrite P→Q ≡ ¬P∨Q.

---

**End of Day 1 solutions.** After attempting, report your answers + time per question so accuracy baselines, weak-topic detection, and the revision queue can be built.
