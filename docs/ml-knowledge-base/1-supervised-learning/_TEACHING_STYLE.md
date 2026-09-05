# 🧑‍🏫 SUPERVISED LEARNING — TEACHING & EXPLANATION SYSTEM v2

> Governs every note in this folder. Target: **"A great teacher + interactive textbook + coding lab + visual simulator + practice system."**
>
> Master principle: **Do not tell the learner everything — make them understand one thing, then build the next thing on top of it.**

---

## THE THREE GOLDEN RULES

1. **Never introduce a concept before creating the need for it.** (Problem → curiosity → concept name → formula.)
2. **Never introduce a formula before explaining what problem it solves.** (Problem → intuition → meaning → formula → symbols → number example → visual → code.)
3. **Make the learner THINK before you explain.** (Question → learner guesses → reveal → explain why.)

---

## THE LEARNING LOOP

```text
UNDERSTAND → VISUALIZE → CALCULATE → IMPLEMENT → EXPERIMENT
→ BREAK → ANALYZE → COMPARE → PRACTICE → EXPLAIN → APPLY → MASTER
```

Every note should push the learner around this loop at least once.

---

## THE 34-SECTION CONTENT HIERARCHY

Every algorithm note uses these sections **in this order** (progressive disclosure — Level 1 first, deep sections later):

```text
01. Start Here            — metadata, journey promise
02. The Problem           — FIRST SCREEN: a story + question the learner must answer
03. Let's Think           — let the learner guess before revealing
04. Intuition             — the idea in ordinary language
05. Visual                — picture BEFORE any formula
06. First Prediction      — use the intuition to predict something
07. Core Concept          — name the idea, define precisely
08. Terminology           — terms emerge here, each with simple + technical meaning
09. Mathematics           — gradual: line → error → squared error → objective
10. Numerical Example     — tiny dataset, every step shown
11. How It Works          — plain-language process flow
12. Internal Process      — the model's internal story + what fit() really does
13. From Scratch          — Version 1 (simplest) → Version 2 (vectorized) → Version 3 (clean)
14. Library Implementation — sklearn, with the abstraction EXPLAINED
15. Code Walkthrough      — why each important line exists
16. Interactive Experiment — sliders/buttons; what changes, why, what to notice
17. Break the Model       — one deliberate failure at a time
18. What If?              — a table or set of "what changes when…" prompts
19. Hyperparameters       — plain meaning, too-high/too-low, tuning
20. Assumptions           — what, why, how to check, what if violated
21. Data Requirements     — type, missing, outliers, scaling, size
22. Evaluation            — metrics with formulas; loss ≠ metric
23. Failure Cases         — when and why it fails
24. Debugging             — symptom → cause checklist
25. Compare               — conceptual difference FIRST, table as summary
26. Real-World Workflow   — business problem → data → model → deploy → monitor
27. Practice              — 8 levels: Recall → … → Build → Explain
28. Interview             — beginner / intermediate / advanced with answers
29. GATE / Exam           — formulas, traps, patterns (no invented PYQs)
30. Deep Dive             — derivation, matrix form, geometry, stats, complexity (gated)
31. Teach Back            — explain in 30s, to a 12-year-old, in an interview, with math
32. Mastery Test          — 10 "without looking" checks
33. Cheat Sheet           — compact revision block
34. What Next?            — connections + recommended next algorithm
```

**Do NOT display all 34 as a wall.** Sections 01–18 are the Level-1 learning path. Sections 19–26 are Level 2. Sections 27–34 are Level 3 / reference (gated behind "Want to go deeper?" where appropriate).

---

## SEMANTIC BLOCKS (for the interactive platform)

Each block below can become a UI component. Mark them as hidden HTML comments so a parser can find them without the reader seeing tags:

```html
<!-- [STORY] -->
<!-- [QUESTION] -->
<!-- [TRY_IT] -->
<!-- [INTUITION] -->
<!-- [VISUAL] -->
<!-- [CONCEPT] -->
<!-- [FORMULA] -->
<!-- [CALCULATION] -->
<!-- [THINK_ABOUT_IT] -->
<!-- [COMMON_MISTAKE] -->
<!-- [UNDER_THE_HOOD] -->
<!-- [CODE] -->
<!-- [CODE_WALKTHROUGH] -->
<!-- [EXPERIMENT] -->
<!-- [BREAK_IT] -->
<!-- [WHAT_IF] -->
<!-- [COMPARE] -->
<!-- [PRACTICE] -->
<!-- [INTERVIEW] -->
<!-- [GATE] -->
<!-- [DEEP_DIVE] -->
<!-- [TEACH_BACK] -->
<!-- [MASTERY] -->
```

Rule: if a section's content maps naturally to a component, put its tag on the line above the content. Do not over-tag plain prose.

---

## FIRST-SCREEN RULE

The very first thing a learner sees must be a problem + a question, NOT a definition.

```text
❌ "Linear Regression is a supervised parametric model..."

✅ "Ankit has 3.5 years of experience.
    Previous employees earned 5L, 5.8L, 6.5L, 7.2L, 8L.
    What salary would you predict for Ankit?"  [ enter guess ]
```

---

## ONE CORE STORY PER ALGORITHM

Pick one primary relatable story and reuse it while teaching every concept (feature, target, prediction, slope, intercept, residual, loss…). Introduce a **second example only after** the core idea is clear.

Examples pool: salary vs experience · house price · exam marks vs study hours · food delivery · cab fare · mobile recharge · electricity bill · cricket stats · attendance vs marks · placement prediction · rent prediction · ad spend vs sales.

---

## VOICE

Conversational + professional. Prefer "Let's see what's happening", "Now we have a problem", "Good question", "Let's calculate with tiny numbers". Avoid "Hence / Thus / Furthermore / It can be observed that".

Target: **simple + intelligent + conversational + technically correct.**

---

## EMOJI RULE

Sparingly. They help scanning, not decoration:

```
💡 Intuition · 🧮 Math · 💻 Code · 🧪 Experiment · ⚠️ Common mistake · 💥 Break it · 🎯 Practice · 🤔 Think about it · 📌 Remember
```

No emojis in every heading. No meaningless emoji spam.

---

## TECHNICAL ACCURACY

Simplified explanations must be clearly marked as "simple intuition", and the technically correct version must always be present. Never sacrifice correctness to sound simple. No invented GATE PYQs — sample questions are labeled "representative pattern question".

---

## FINAL QUALITY GATE — self-score before finishing a note

```text
Clarity /10 · Beginner friendliness /10 · Intuition /10 · Storytelling /10
Mathematical teaching /10 · Visual explanation /10 · Practical relevance /10
Code explanation /10 · Experiment quality /10 · Failure analysis /10
Progressive depth /10 · Technical accuracy /10
```

Any score below 9/10 → rewrite before finishing.