# Phase 15 — Evaluation

> **Goal:** Master evaluation across ML, LLMs, RAG, and agents — building the judgment to measure what matters, not just what's easy to compute.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced
**Priority:** Essential
**Prerequisites:** Phase 05 (ML), Phase 11 (RAG), Phase 14 (Agents)
**Mastery target:** Level 5 — independent evaluation design for any AI system

---

## Why This Phase Exists

Throughout earlier phases you built models, pipelines, and agents. But building is not enough — you must know whether they work, how they fail, and whether they are good enough for deployment. Evaluation is the discipline that turns "it seems to work" into "here is the evidence." Without it, every ML system is a black box you cannot trust, debug, or improve.

### Phase Mental Model

Evaluation is a funnel: define what success means → measure it with metrics → analyze failures → iterate.

```text
Define success criteria
       ↓
Select metrics (quantitative + qualitative)
       ↓
Build evaluation datasets
       ↓
Run experiments, track results
       ↓
Analyze errors and failures
       ↓
Iterate: improve model/data/prompt
       ↓
Deploy with confidence + monitoring
```

### What This Phase Prepares For

- Phase 16 (Deployment) — monitoring and production quality
- Phase 17 (Capstones) — rigorous project evaluation and reporting
- Professional practice — cost/quality trade-offs in real systems

---

## Units

### Unit 15.1 — ML Evaluation Deep Dive

**What is it?**
The systematic measurement of traditional ML model performance using metrics, error analysis, and calibration techniques.

**Why does it matter?**
A model that you cannot measure is a model you cannot improve. Choosing the wrong metric can make a bad model look good and a good model look bad.

**Why learn it here?**
Earlier phases introduced train/test splits and accuracy. Now we go deeper: precision, recall, F1, ROC-AUC, calibration, and when each matters.

**Prerequisites:** Phase 05 (ML basics), Phase 03 (statistics)

**Mental Model:**
Metrics are lenses. Accuracy is a wide-angle lens — useful but not specific. Precision, recall, and F1 are zoom lenses that let you focus on specific failure types.

**Core Concepts:**

- confusion matrix
- precision, recall, F1-score
- ROC curve and AUC
- precision-recall curve
- regression metrics (MAE, MSE, RMSE, R²)
- class imbalance and metric choice
- threshold tuning
- calibration curves (reliability diagrams)
- error analysis by segment

**How It Works:**

1. Split data into train/validation/test.
2. Train model, predict on held-out data.
3. Compute metrics that match the business cost of errors.
4. Examine errors by subgroup to find systematic failures.
5. Tune threshold if the model outputs probabilities.

**Syntax & Implementation:**

```python
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, precision_recall_curve
)
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 0])
y_prob = np.array([0.9, 0.1, 0.8, 0.3, 0.2, 0.7, 0.4, 0.05, 0.85, 0.15])

y_pred = (y_prob >= 0.5).astype(int)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.3f}")

precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
```

**Simple Example:**

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

X, y = make_classification(n_samples=1000, n_features=20, weights=[0.9, 0.1], random_state=42)
model = LogisticRegression().fit(X[:800], y[:800])
y_pred = model.predict(X[800:])

print(f"Accuracy: {accuracy_score(y[800:], y_pred):.3f}")
print(f"F1:       {f1_score(y[800:], y_pred):.3f}")
```

**Real-World Example:**
A fraud detection model with 99% accuracy but catches only 10% of frauds. Precision-recall analysis reveals the threshold is too high. Lowering it catches 80% of frauds with acceptable false-positive cost.

**Common Mistakes:**

- using accuracy on imbalanced data
- ignoring the cost difference between false positives and false negatives
- evaluating only on training data
- not analyzing errors by subgroup
- assuming high AUC means the model is production-ready

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| High accuracy, low F1 | Class imbalance | Check class distribution | Use F1, precision-recall, or weighted metrics |
| ROC-AUC high but model useless | Threshold not tuned for class distribution | Plot precision-recall curve | Tune threshold, use business-cost metric |
| Model scores differently each run | No random seed or different splits | Check `random_state`, split consistency | Fix seeds, use stratified splits |
| Metrics don't match business needs | Wrong metric chosen | Interview stakeholders | Define cost matrix, use cost-sensitive metrics |
| Overfitting invisible | No validation split | Check if train and test metrics diverge | Add proper train/val/test split |

**Alternatives:**

| Tool | Use When | Avoid When |
|---|---|---|
| scikit-learn metrics | Standard tabular ML | Need LLM/RAG-specific evaluation |
| MLflow tracking | Experiment comparison | Quick one-off evaluation |
| Evidently AI | Production monitoring | Offline evaluation only |

**Best Practices:**

- always define metrics before training
- use stratified splits for imbalanced data
- report confidence intervals when possible
- include error analysis, not just aggregate scores
- document metric choice and business rationale

**Hands-On Practice:**

1. **Basic:** Compute accuracy, precision, recall, F1 on a binary classification task.
2. **Guided:** Generate a confusion matrix and interpret each cell.
3. **Independent:** Evaluate a model on imbalanced data — compare accuracy vs F1 vs AUC-PR.
4. **Realistic:** Perform error analysis: find the 10% of samples the model fails on most, and explain why.
5. **Challenge:** Build a threshold-tuning pipeline that optimizes for a custom cost function.

**Knowledge Check:**

- When is accuracy misleading?
- What is the difference between ROC-AUC and precision-recall AUC?
- Why might you prefer recall over precision?
- How do you decide the probability threshold for a production classifier?

**Exit Criteria:**

- You can choose the right metric for a given problem.
- You can perform error analysis and find systematic failures.
- You can explain threshold tuning and calibration.

**Next Step:** Move to evaluating LLM outputs, where metrics are less standardized.

---

### Unit 15.2 — LLM Evaluation

**What is it?**
Measuring the quality of language model outputs across dimensions like correctness, relevance, faithfulness, and safety.

**Why does it matter?**
LLMs generate free-form text. Traditional metrics like accuracy don't apply. Without structured evaluation, you cannot distinguish a good answer from a confident-sounding wrong one.

**Why learn it here?**
You have built LLM-powered systems. Now you need the tools to know when they work and when they hallucinate.

**Prerequisites:** Phase 09 (GenAI), Phase 10 (LLMs), Phase 15.1

**Mental Model:**
LLM evaluation is multi-dimensional: correctness, helpfulness, safety, and cost are all different axes. A model can be correct but unsafe, or safe but unhelpful.

**Core Concepts:**

- reference-based metrics (BLEU, ROUGE)
- LLM-as-judge (using one model to evaluate another)
- human evaluation
- groundedness / faithfulness
- relevance and correctness
- safety and toxicity evaluation
- hallucination detection
- evaluation frameworks (RAGAS, DeepEval, TruLens)

**How It Works:**

1. Define evaluation dimensions: correctness, relevance, groundedness, safety.
2. Create evaluation datasets with input-output-reference triples.
3. Run model, collect outputs.
4. Score automatically (reference metrics or LLM-as-judge) or with human annotators.
5. Analyze failures by dimension.

**Syntax & Implementation:**

```python
# LLM-as-judge example using DeepEval
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

test_case = LLMTestCase(
    input="What is the capital of France?",
    actual_output="The capital of France is Paris.",
    retrieval_context=["France is a country in Europe. Its capital is Paris."]
)

faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)

faithfulness.measure(test_case)
print(f"Faithfulness: {faithfulness.score}")
print(f"Reason: {faithfulness.reason}")

relevancy.measure(test_case)
print(f"Relevancy: {relevancy.score}")
```

**Simple Example:**

```python
# Simple reference-based evaluation
def evaluate_answer(reference: str, generated: str) -> dict:
    ref_words = set(reference.lower().split())
    gen_words = set(generated.lower().split())
    overlap = ref_words & gen_words
    return {
        "word_overlap": len(overlap) / max(len(ref_words), 1),
        "exact_match": reference.lower().strip() == generated.lower().strip()
    }

result = evaluate_answer(
    reference="The capital of France is Paris.",
    generated="Paris is the capital of France."
)
print(result)  # {'word_overlap': 1.0, 'exact_match': False}
```

**Real-World Example:**
A customer support bot answers 1000 questions. Human review of 100 samples finds 15% are factually wrong. You build an automated evaluation pipeline using LLM-as-judge that detects 80% of these errors, enabling continuous monitoring without manual review of every response.

**Common Mistakes:**

- relying only on BLEU/ROUGE for factual questions
- using LLM-as-judge without validating it against human labels
- evaluating without a fixed test set (test set drift)
- ignoring safety and toxicity dimensions
- treating one metric as the full picture

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| BLEU score high but answer wrong | Reference and output share words but facts differ | Human review subset | Add groundedness and faithfulness metrics |
| LLM-as-judge gives inconsistent scores | Prompt or model temperature variation | Run same output twice, check variance | Use temperature=0, structured judge prompts |
| Evaluation doesn't match user satisfaction | Wrong dimension measured | Compare eval scores to user feedback | Add multi-dimensional evaluation |
| Hallucinations not detected | No grounding check | Sample outputs, check claims against sources | Add retrieval context and faithfulness metric |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Reference metrics (BLEU/ROUGE) | Translation, summarization with reference | Factual QA, open-ended generation |
| LLM-as-judge | Large-scale automated evaluation | Small datasets where human review is feasible |
| Human evaluation | Gold standard, safety-critical | Expensive, slow, not scalable alone |

**Best Practices:**

- always use a fixed, representative evaluation dataset
- evaluate multiple dimensions (correctness + relevance + safety)
- validate LLM-as-judge against human labels on a sample
- report failure modes, not just aggregate scores
- version your evaluation datasets alongside your code

**Hands-On Practice:**

1. **Basic:** Compute BLEU and ROUGE scores for a summarization task.
2. **Guided:** Set up an LLM-as-judge pipeline to rate answers on a 1-5 scale.
3. **Independent:** Build a multi-dimension evaluation for a Q&A system (correctness, relevance, groundedness).
4. **Realistic:** Evaluate 100 LLM outputs, find the 10 worst, and categorize failure modes.
5. **Challenge:** Design an evaluation pipeline that runs automatically on every prompt change.

**Knowledge Check:**

- Why is BLEU insufficient for factual QA?
- How do you validate that LLM-as-judge agrees with human judgment?
- What dimensions should you evaluate for a customer-facing chatbot?
- How do you handle evaluation when there is no single correct answer?

**Exit Criteria:**

- You can design multi-dimensional LLM evaluation.
- You can implement both reference-based and judge-based evaluation.
- You can analyze LLM failures systematically.

**Next Step:** Specialize evaluation for RAG systems.

---

### Unit 15.3 — RAG Evaluation

**What is it?**
Measuring both the retrieval quality and the generation quality of a Retrieval-Augmented Generation system.

**Why does it matter?**
RAG failures come from two sources: bad retrieval (wrong context) and bad generation (ignoring good context). If you only measure the final answer, you cannot tell which part failed.

**Why learn it here?**
You have built RAG pipelines. Now you need to diagnose whether failures come from chunking, embedding, retrieval, or generation.

**Prerequisites:** Phase 11 (RAG), Phase 15.2

**Mental Model:**
RAG evaluation is a two-stage pipeline:

```text
Query → Retrieval → Generation → Answer
         ↓              ↓
    Was the right      Did the model use it
    context found?     correctly and honestly?
```

**Core Concepts:**

- retrieval metrics (context precision, context recall, hit rate, MRR)
- generation metrics (faithfulness, answer relevancy, correctness)
- end-to-end metrics
- chunking quality assessment
- embedding quality assessment
- hallucination in RAG
- RAGAS framework
- reference-free vs reference-based evaluation

**How It Works:**

1. Create a QA evaluation dataset with questions, ground-truth answers, and ground-truth context.
2. Run the RAG pipeline, capture both retrieved context and generated answer.
3. Score retrieval quality (did we find the right chunks?).
4. Score generation quality (did the model use the context correctly?).
5. Identify whether failures come from retrieval or generation.

**Syntax & Implementation:**

```python
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
)

test_case = LLMTestCase(
    input="What are the side effects of aspirin?",
    actual_output="Common side effects include stomach upset and bleeding risk.",
    retrieval_context=[
        "Aspirin can cause stomach upset, nausea, and increased bleeding risk.",
        "Aspirin is used to reduce pain and fever."
    ],
    expected_output="Side effects include stomach upset, nausea, and increased bleeding risk."
)

faithfulness = FaithfulnessMetric(threshold=0.7)
relevancy = AnswerRelevancyMetric(threshold=0.7)
precision = ContextualPrecisionMetric(threshold=0.7)
recall = ContextualRecallMetric(threshold=0.7)

for metric in [faithfulness, relevancy, precision, recall]:
    metric.measure(test_case)
    print(f"{metric.__class__.__name__}: {metric.score:.2f}")
```

**Simple Example:**

```python
def rag_retrieval_quality(query: str, retrieved_docs: list[str], gold_docs: list[str]) -> dict:
    retrieved_set = set(d.lower() for d in retrieved_docs)
    gold_set = set(d.lower() for d in gold_docs)
    hits = retrieved_set & gold_set
    return {
        "precision": len(hits) / max(len(retrieved_set), 1),
        "recall": len(hits) / max(len(gold_set), 1),
        "hit_rate": 1.0 if hits else 0.0
    }

result = rag_retrieval_quality(
    query="aspirin side effects",
    retrieved_docs=["Aspirin causes stomach upset.", "Aspirin thins blood."],
    gold_docs=["Aspirin causes stomach upset.", "Aspirin increases bleeding risk."]
)
print(result)
```

**Real-World Example:**
A medical QA system retrieves 5 chunks per query. Precision@3 is 0.4 — two of the top 3 chunks are irrelevant. After switching from naive cosine similarity to hybrid search with reranking, precision@3 rises to 0.8. The generation quality metric (faithfulness) also improves because the model receives better context.

**Common Mistakes:**

- evaluating only the final answer without checking retrieval
- using the same chunks for evaluation that were used for training
- not having gold-standard context annotations
- ignoring chunk overlap and deduplication
- assuming better retrieval always means better answers

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Answer is wrong but retrieval was good | Generation problem | Check retrieved context vs answer | Improve prompt, add generation constraints |
| Retrieval has low precision | Bad embeddings or chunking | Inspect top-k chunks per query | Tune chunk size, use hybrid search, add reranking |
| Faithfulness score is low | Model ignores context | Compare answer to retrieved chunks | Strengthen prompt instructions, add citations |
| Recall is high but answer is still bad | Too many irrelevant chunks dilute context | Check chunk relevance distribution | Improve filtering, use smaller top-k with reranking |
| Evaluation scores don't match user feedback | Test set not representative | Compare test queries to real queries | Refresh eval set with production data |

**Best Practices:**

- always separate retrieval evaluation from generation evaluation
- maintain gold-standard QA pairs with expected context
- evaluate chunk size and overlap effects on retrieval quality
- use hybrid search and compare against pure vector search
- track both automatic metrics and human spot-checks

**Hands-On Practice:**

1. **Basic:** Compute retrieval precision and recall for 5 queries with known gold documents.
2. **Guided:** Set up a RAG pipeline and evaluate both retrieval and generation using RAGAS.
3. **Independent:** Compare two chunking strategies (fixed-size vs semantic) on retrieval quality.
4. **Realistic:** Diagnose a failing RAG pipeline — is the problem retrieval or generation?
5. **Challenge:** Build an automated RAG evaluation suite that runs on every pipeline change.

**Knowledge Check:**

- What is the difference between context precision and context recall?
- Why might high retrieval recall not lead to high answer quality?
- How do you decide between chunking strategies using evaluation?
- When should you use reranking vs improving embeddings?

**Exit Criteria:**

- You can separately evaluate retrieval and generation in a RAG system.
- You can diagnose whether a RAG failure is a retrieval or generation problem.
- You can design evaluation datasets for RAG.

**Next Step:** Extend evaluation to agents, which involve tool use and multi-step reasoning.

---

### Unit 15.4 — Agent Evaluation

**What is it?**
Measuring the performance of LLM agents that use tools, make decisions, and execute multi-step workflows.

**Why does it matter?**
Agents are non-deterministic and stateful. A single "correct output" metric is insufficient — you must evaluate tool selection, reasoning steps, final outcomes, and cost/safety.

**Why learn it here?**
You have built agents. Now you need to measure whether they complete tasks correctly, efficiently, and safely.

**Prerequisites:** Phase 14 (Agents), Phase 15.2

**Mental Model:**
Agent evaluation is about the trajectory, not just the destination:

```text
Task → Plan → Action → Observation → Action → ... → Result
         ↓      ↓         ↓
      Reasoning  Tool     Environment
      quality   choice    side effects
```

**Core Concepts:**

- task completion rate
- tool selection accuracy
- step efficiency (steps taken vs optimal)
- trajectory evaluation
- cost per task (tokens, API calls, time)
- safety and guardrail compliance
- human-in-the-loop evaluation
- benchmark suites (GAIA, AgentBench, SWE-bench)
- failure mode taxonomy

**How It Works:**

1. Define task scenarios with expected outcomes.
2. Run the agent on each scenario, logging all steps.
3. Evaluate: did it complete the task? How many steps? What tools did it use?
4. Compare against optimal trajectories or human baselines.
5. Categorize failures: wrong tool, infinite loop, safety violation, incomplete task.

**Syntax & Implementation:**

```python
from dataclasses import dataclass, field

@dataclass
class AgentTrajectory:
    task: str
    steps: list[dict] = field(default_factory=list)
    final_result: str = ""
    total_tokens: int = 0
    total_time_s: float = 0.0

    def add_step(self, tool: str, input_text: str, output_text: str):
        self.steps.append({
            "tool": tool,
            "input": input_text,
            "output": output_text,
        })

    def evaluate(self, expected_result: str) -> dict:
        return {
            "completed": bool(self.final_result),
            "steps_taken": len(self.steps),
            "exact_match": self.final_result.strip().lower() == expected_result.strip().lower(),
            "total_tokens": self.total_tokens,
            "time_seconds": self.total_time_s,
        }

trajectory = AgentTrajectory(task="Summarize file report.txt")
trajectory.add_step("read_file", "report.txt", "Q3 revenue grew 15%...")
trajectory.add_step("summarize", "Q3 revenue grew 15%...", "Revenue increased 15% in Q3.")
trajectory.final_result = "Revenue increased 15% in Q3."
trajectory.total_tokens = 450
trajectory.total_time_s = 2.3

print(trajectory.evaluate("Revenue grew 15% in Q3."))
```

**Simple Example:**

```python
def evaluate_agent_run(tasks: list[dict], agent_fn) -> dict:
    results = []
    for task in tasks:
        trajectory = agent_fn(task["query"])
        results.append({
            "task": task["query"],
            "completed": bool(trajectory.final_result),
            "steps": len(trajectory.steps),
            "correct": trajectory.final_result.strip().lower() == task["expected"].strip().lower(),
        })
    total = len(results)
    return {
        "completion_rate": sum(r["completed"] for r in results) / total,
        "accuracy": sum(r["correct"] for r in results) / total,
        "avg_steps": sum(r["steps"] for r in results) / total,
    }
```

**Real-World Example:**
An agent that researches competitors. Evaluation shows 90% task completion but average 12 steps per task. After adding a planning step and tool-use constraints, completion stays at 90% but average steps drop to 5 — halving token cost.

**Common Mistakes:**

- evaluating only the final answer
- not tracking token usage and cost
- ignoring intermediate reasoning errors
- testing only on easy tasks
- not having a baseline to compare against

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Agent loops indefinitely | Missing step limit or stop condition | Log step count per task | Add max_steps guard and timeout |
| High task completion but wrong answer | Agent returns partial or hallucinated result | Human review of outputs | Add verification step, confidence checks |
| Agent uses wrong tool | Tool description unclear or schema mismatch | Check tool schemas and prompts | Improve tool descriptions, add examples |
| Evaluation is slow | Running full trajectory for every test | Measure eval time | Use smaller test set for quick iteration |
| Agent succeeds in eval but fails in production | Test set not representative | Compare eval vs production distributions | Refresh eval set with real user tasks |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Task completion metrics | End-to-end evaluation | Need to diagnose intermediate failures |
| Trajectory comparison | Detailed debugging, research | Simple tasks with few steps |
| Human evaluation | Safety-critical or subjective tasks | Expensive, slow |
| Benchmark suites (SWE-bench) | Standardized comparison | Tasks differ from benchmark scope |

**Best Practices:**

- always log full agent trajectories (every tool call and response)
- define clear success criteria per task type
- evaluate cost (tokens, time) alongside correctness
- include safety and guardrail compliance in evaluation
- use both easy and hard test cases

**Hands-On Practice:**

1. **Basic:** Log and inspect the trajectory of a simple agent task.
2. **Guided:** Build a trajectory evaluator that measures steps, completion, and correctness.
3. **Independent:** Evaluate 20 agent tasks and categorize failure modes.
4. **Realistic:** Compare two agent designs on the same task set — which is more efficient?
5. **Challenge:** Build a benchmark suite that tests tool selection, reasoning, and safety.

**Knowledge Check:**

- Why is task completion rate insufficient for agent evaluation?
- How do you measure agent efficiency, not just correctness?
- What is trajectory evaluation and why does it matter?
- How do you detect when an agent is stuck in a loop?

**Exit Criteria:**

- You can evaluate agents on completion, efficiency, and cost.
- You can diagnose agent failures using trajectory logs.
- You can design evaluation suites for agent tasks.

**Next Step:** Learn how to build and maintain the evaluation datasets that power all these evaluations.

---

### Unit 15.5 — Evaluation Datasets

**What is it?**
Creating, maintaining, and versioning structured datasets used to measure system quality across ML, LLMs, RAG, and agents.

**Why does it matter?**
Evaluation is only as good as your test data. A biased, stale, or unrepresentative evaluation dataset produces misleading results.

**Why learn it here?**
You have seen evaluation metrics. Now you need to understand where the test data comes from and how to build it properly.

**Prerequisites:** Phase 15.1–15.4

**Mental Model:**
An evaluation dataset is a contract between your system and reality. It defines what "correct" means and must be maintained as the system evolves.

**Core Concepts:**

- dataset design (inputs, expected outputs, metadata)
- stratification and coverage
- adversarial and edge cases
- dataset versioning
- data leakage prevention
- gold annotations vs proxy labels
- synthetic data generation for evaluation
- dataset quality checks

**How It Works:**

1. Define evaluation dimensions (what are you testing?).
2. Collect or generate diverse inputs.
3. Annotate with expected outputs (gold labels).
4. Add metadata (category, difficulty, edge case flags).
5. Version the dataset and track changes.
6. Split into subsets for different evaluation purposes.

**Syntax & Implementation:**

```python
import json
from pathlib import Path

eval_dataset = {
    "version": "1.2",
    "tasks": [
        {
            "id": "qa-001",
            "input": "What is machine learning?",
            "expected_output": "ML is a subset of AI where systems learn from data.",
            "category": "definition",
            "difficulty": "easy",
            "is_edge_case": False,
        },
        {
            "id": "qa-002",
            "input": "Explain overfitting in neural networks.",
            "expected_output": "Overfitting occurs when a model memorizes training data...",
            "category": "concept",
            "difficulty": "medium",
            "is_edge_case": False,
        },
        {
            "id": "qa-003",
            "input": "",
            "expected_output": "I cannot answer an empty question.",
            "category": "edge_case",
            "difficulty": "hard",
            "is_edge_case": True,
        },
    ]
}

path = Path("eval_datasets/qa_v1.2.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(eval_dataset, indent=2))
print(f"Saved {len(eval_dataset['tasks'])} evaluation items to {path}")
```

**Simple Example:**

```python
def validate_eval_dataset(dataset: dict) -> dict:
    issues = []
    for item in dataset["tasks"]:
        if not item.get("input"):
            issues.append(f"{item['id']}: empty input")
        if not item.get("expected_output"):
            issues.append(f"{item['id']}: missing expected output")
        if item.get("difficulty") not in ("easy", "medium", "hard"):
            issues.append(f"{item['id']}: unknown difficulty level")
    return {"total": len(dataset["tasks"]), "issues": issues}

report = validate_eval_dataset(eval_dataset)
print(f"Total: {report['total']}, Issues: {len(report['issues'])}")
```

**Real-World Example:**
A team builds a customer support evaluation dataset with 500 queries. They categorize by intent (billing, technical, general), difficulty, and whether the query requires multi-turn context. They add 50 adversarial queries (prompt injection attempts, ambiguous questions). They version the dataset with each release and track evaluation scores over time.

**Common Mistakes:**

- building the eval set after training (leakage risk)
- not including edge cases and adversarial inputs
- using only easy examples (overestimates performance)
- not versioning the dataset
- having inconsistent annotation quality

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| High eval score but user complaints | Eval set not representative | Compare eval queries to real traffic | Add production-representative samples |
| Scores drop suddenly after update | Eval set changed or data leakage fixed | Check dataset version, review changes | Audit dataset changes, restore previous version |
| Annotation disagreements | Unclear guidelines or ambiguous questions | Check inter-annotator agreement | Improve guidelines, add examples, resolve ambiguity |
| Eval set too small for reliable results | Insufficient samples per category | Count items per category | Expand dataset, especially for minority categories |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Manual annotation | High-quality gold labels needed | Expensive, slow |
| Synthetic generation | Quick coverage, adversarial testing | Quality varies, needs validation |
| Production data sampling | Representative evaluation | May contain noise, privacy concerns |

**Best Practices:**

- version your evaluation datasets like code
- include adversarial and edge cases (at least 10%)
- stratify by category, difficulty, and input type
- validate annotation quality periodically
- refresh the dataset as the system and user base evolve

**Hands-On Practice:**

1. **Basic:** Create a 20-item evaluation dataset with inputs, expected outputs, and categories.
2. **Guided:** Add metadata (difficulty, edge case flag) and validate the dataset.
3. **Independent:** Generate 50 synthetic evaluation items and filter for quality.
4. **Realistic:** Build a 200-item eval set for a RAG system covering retrieval and generation.
5. **Challenge:** Design a dataset versioning system with change tracking and rollback.

**Knowledge Check:**

- Why must you build evaluation datasets before training?
- How do you handle evaluation when there is no single correct answer?
- What percentage of your eval set should be edge cases?
- How do you detect and fix annotation drift?

**Exit Criteria:**

- You can build evaluation datasets with proper structure and metadata.
- You can validate and version evaluation datasets.
- You can design eval sets that cover edge cases and adversarial inputs.

**Next Step:** Track experiments systematically so evaluation results are reproducible.

---

### Unit 15.6 — Experiment Tracking

**What is it?**
Recording and comparing the results of ML experiments — parameters, metrics, code versions, and artifacts — to enable reproducibility and informed iteration.

**Why does it matter?**
Without tracking, you cannot remember which configuration produced which result. You waste time repeating failed experiments and lose confidence in improvements.

**Why learn it here?**
You are now running multiple experiments with different models, prompts, and configurations. Tracking turns chaos into a searchable history.

**Prerequisites:** Phase 15.1–15.5

**Mental Model:**
An experiment tracker is a lab notebook for machines: every run records what you tried, what happened, and what changed.

```text
Experiment run = config + code version + data version + metrics + artifacts
```

**Core Concepts:**

- experiment vs run vs metric
- logging parameters, metrics, and artifacts
- MLflow tracking
- Weights & Biases
- DVC for data versioning
- experiment comparison and visualization
- reproducibility (seeds, environment snapshots)
- tagging and note-taking

**How It Works:**

1. Initialize a tracking run.
2. Log configuration parameters (model type, learning rate, chunk size).
3. Train or run the system.
4. Log metrics at each step or at completion.
5. Save artifacts (model weights, evaluation reports).
6. Compare runs to find the best configuration.

**Syntax & Implementation:**

```python
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("ml-evaluation-demo")

with mlflow.start_run(run_name="rf-baseline"):
    params = {"n_estimators": 100, "max_depth": 10, "random_state": 42}
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(model, "model")
    print(f"Run logged: F1={f1:.3f}")
```

**Simple Example:**

```python
import time

experiments = []

def log_experiment(name: str, params: dict, metrics: dict):
    experiments.append({
        "name": name,
        "params": params,
        "metrics": metrics,
        "timestamp": time.time(),
    })

log_experiment("baseline", {"model": "logreg", "lr": 0.01}, {"f1": 0.72, "accuracy": 0.85})
log_experiment("tuned", {"model": "rf", "n_estimators": 200}, {"f1": 0.81, "accuracy": 0.89})

best = max(experiments, key=lambda e: e["metrics"]["f1"])
print(f"Best: {best['name']} with F1={best['metrics']['f1']}")
```

**Real-World Example:**
A team runs 15 RAG experiments varying chunk size (256, 512, 1024), embedding model (ada, e5, bge), and top-k (3, 5, 10). MLflow tracks all 15 combinations. The team discovers that chunk size 512 + e5 embeddings + top-k 5 gives the best context precision. Without tracking, this comparison would require manual spreadsheets and re-running experiments.

**Common Mistakes:**

- not logging the random seed (reproducibility loss)
- forgetting to log the data version
- overwriting previous runs
- tracking too many things (noise)
- not comparing against a baseline

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Results not reproducible | Missing seed or data version logging | Check logged params | Always log random_state, data hash |
| Cannot find a previous run | Run not saved or wrong experiment name | Check MLflow/W&B dashboard | Verify experiment name, use tags |
| Comparison shows no clear winner | Metrics not relevant or variance too high | Check metric variance across runs | Increase test set size, use more relevant metrics |
| Tracking server errors | Server not running or network issue | Check server status | Start tracking server, check network config |

**Alternatives:**

| Tool | Use When | Avoid When |
|---|---|---|
| MLflow | Open-source, local or server, broad framework support | Need real-time collaboration features |
| Weights & Biases | Team collaboration, visualization, sweeps | Need fully offline/local setup |
| DVC | Data and model versioning alongside code | Need metric tracking (use with MLflow/W&B) |
| Plain CSV/JSON | Quick one-off tracking | Many experiments, need comparison UI |

**Best Practices:**

- always log parameters, metrics, and code version together
- set a random seed and log it
- tag runs with descriptive names and notes
- maintain a baseline experiment for comparison
- review experiments periodically and clean up stale runs

**Hands-On Practice:**

1. **Basic:** Log parameters and a metric for one experiment using MLflow.
2. **Guided:** Run 5 experiments with different hyperparameters and compare results in the MLflow UI.
3. **Independent:** Build a tracking workflow that logs model, metrics, and a data version hash.
4. **Realistic:** Compare 3 RAG configurations using experiment tracking and select the best.
5. **Challenge:** Set up an automated sweep that tracks all results and generates a comparison report.

**Knowledge Check:**

- Why must you log data version alongside model parameters?
- How do you choose between MLflow and Weights & Biases?
- What is the minimum information you need to reproduce an experiment?
- How do you maintain a baseline for fair comparison?

**Exit Criteria:**

- You can track experiments with parameters, metrics, and artifacts.
- You can compare multiple runs and identify the best configuration.
- You can reproduce a previous experiment from logs alone.

**Next Step:** Synthesize all evaluation skills into a comprehensive system.

---

### Unit 15.7 — Evaluation Synthesis & Review

**What is it?**
A cumulative integration unit that combines ML evaluation, LLM evaluation, RAG evaluation, agent evaluation, evaluation datasets, and experiment tracking into a coherent evaluation practice.

**Why does it matter?**
Knowing individual metrics is not enough. You must be able to design an evaluation strategy for any system — from a simple classifier to a complex multi-agent pipeline.

**Prerequisites:** Units 15.1–15.6

---

#### Mini Project: Evaluation System for a RAG Application

**Objective:** Build a complete evaluation pipeline for a RAG-based question-answering system.

**Problem Statement:** A legal QA system retrieves relevant case law and generates answers. You must evaluate whether it retrieves the right cases, generates faithful answers, and meets quality standards.

**Requirements:**

- build an evaluation dataset with 50+ items across 3 categories (factual, analytical, edge cases)
- implement retrieval evaluation (precision@k, recall@k, MRR)
- implement generation evaluation (faithfulness, relevancy)
- run experiments with at least 3 configurations
- track all experiments with MLflow or similar
- perform error analysis on the worst-performing configuration
- write an evaluation report with recommendations
- version the evaluation dataset

**Suggested Architecture:**

```text
Evaluation dataset → RAG pipeline → Collect retrieval + generation
                                          ↓
                               Retrieval metrics (precision, recall, MRR)
                               Generation metrics (faithfulness, relevancy)
                               Combined score
                                          ↓
                               MLflow tracking (params, metrics, artifacts)
                                          ↓
                               Error analysis → Recommendations
```

**Expected Output:**

- evaluation dataset (JSON, versioned)
- evaluation script with retrieval and generation metrics
- MLflow tracking dashboard with 3+ experiment runs
- error analysis report (top 10 failures with root causes)
- evaluation report with visualizations and recommendations
- README explaining setup and methodology

**Evaluation Criteria:**

- [ ] evaluation dataset covers multiple categories and edge cases
- [ ] retrieval and generation are evaluated separately
- [ ] experiments are tracked with parameters and metrics
- [ ] error analysis identifies specific failure modes
- [ ] recommendations are actionable and tied to evidence
- [ ] code is runnable and documented
- [ ] evaluation dataset is versioned

**Advanced Extensions:**

- add agent-based evaluation (multi-step retrieval and reasoning)
- build a threshold-tuning pipeline
- compare automated metrics against human evaluation on a sample
- implement monitoring hooks for production evaluation
- add safety and toxicity checks

---

#### Knowledge Check

- How would you design an evaluation strategy for a system that combines ML, LLM, and agent components?
- When do you use reference-based metrics vs human evaluation vs LLM-as-judge?
- How do you prevent evaluation dataset leakage?
- How do you balance evaluation thoroughness with cost and time?
- What makes an evaluation report useful for a non-technical stakeholder?

#### Exit Criteria

- You can design a multi-dimensional evaluation strategy for any AI system.
- You can build, maintain, and version evaluation datasets.
- You can track experiments and compare results systematically.
- You can perform error analysis and produce actionable recommendations.
- You can communicate evaluation results to both technical and non-technical audiences.

---

## Phase Review Checklist

- [ ] All 7 units completed.
- [ ] ML evaluation metrics practiced (accuracy, precision, recall, F1, AUC, calibration).
- [ ] LLM evaluation dimensions understood (correctness, relevance, faithfulness, safety).
- [ ] RAG evaluation separated into retrieval and generation.
- [ ] Agent evaluation covers task completion, efficiency, and cost.
- [ ] Evaluation dataset built with metadata and versioning.
- [ ] Experiment tracking implemented with MLflow or similar.
- [ ] Mini project completed with full evaluation pipeline.
- [ ] Error analysis performed on failed examples.
- [ ] Evaluation report written with recommendations.

## Mastery Check

Without following a tutorial, you should be able to:

1. Choose the right metric for any ML problem.
2. Design multi-dimensional evaluation for LLM outputs.
3. Separately evaluate retrieval and generation in RAG.
4. Evaluate agents on completion, efficiency, and cost.
5. Build and version evaluation datasets with edge cases.
6. Track experiments systematically and compare results.
7. Perform error analysis and produce actionable recommendations.
8. Communicate evaluation results to technical and non-technical stakeholders.

## Interview / Explain-Back Questions

- When is accuracy a misleading metric? Give an example.
- How do you evaluate an LLM when there is no single correct answer?
- What is the difference between context precision and context recall in RAG?
- How would you evaluate an agent that uses 5 different tools?
- Why must you version evaluation datasets?
- How do you prevent data leakage in evaluation?
- Describe your process for diagnosing a system with low evaluation scores.
- How do you balance evaluation cost with thoroughness?
- When would you use LLM-as-judge instead of human evaluation?
- How do you communicate evaluation results to a product manager?

## Exit Criteria

Move to Phase 16 only when you can independently design, implement, and communicate a complete evaluation strategy for any AI system — from a simple classifier to a multi-agent RAG pipeline — and defend your metric choices with clear reasoning.
