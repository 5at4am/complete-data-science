# Phase 14 — AI Agents

> **Goal:** Master AI agents — from tool calling to multi-agent systems, including failure modes and security.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced
**Priority:** Essential
**Prerequisites:** Phase 12 (Frameworks), Phase 13 (LLM Applications)
**Mastery target:** Level 5 — build, secure, and evaluate tool-using agents independently

---

## Why This Phase Exists

Agents extend LLMs from text generation to action-taking systems. The engineering challenge is not making an agent act — it is making it act safely, predictably, observably, and within limits. This phase teaches bounded autonomy: when to use deterministic workflows, when agents add value, and how to build systems that fail gracefully instead of catastrophically.

### Phase Mental Model

An agent is an LLM inside a controlled loop:

```text
Observe state → Decide next action → Call tool / respond → Update state → Stop or continue
```

The progression through this phase builds that loop layer by layer:

```text
Tool calling (14.1) → Single-tool agent loop (14.2)
       ↓
Planning (14.3) → Execution (14.4) → Reflection (14.5)
       ↓
Memory (14.6) → Multi-tool selection (14.7) → Persistent state (14.8)
       ↓
Multi-agent coordination (14.9) → Failure prevention (14.10) → Security (14.11)
       ↓
Synthesis and independent building (14.12)
```

### What This Phase Prepares For

- Phase 15: formal evaluation of agent quality and safety
- Phase 16: deployment of production agent systems
- Phase 17: capstone projects combining agents with RAG, evaluation, and monitoring
- Industry work on autonomous and semi-autonomous AI systems

---

## Units

### Unit 14.1 — LLM Call → Tool Calling

**What is it?**
Tool calling extends a plain LLM API call so the model can request structured function invocations instead of only generating text.

**Why does it matter?**
Without tool calling, an LLM can only produce text. With tool calling, it can interact with the outside world: read databases, call APIs, run code, search documents. Every agent capability depends on this foundation.

**Why learn it here?**
The learner already understands LLM APIs (Phase 09–10), RAG pipelines (Phase 11), and framework abstractions (Phase 12–13). Tool calling is the bridge between generating text and taking action.

**Prerequisites:** LLM API usage, function definitions in Python, JSON schema familiarity.

#### Mental Model

A tool schema is a contract: the LLM sees function names, descriptions, and parameter types, and outputs structured JSON when it decides a tool is needed. The application code then executes the real function.

```text
User prompt → LLM decides: text response OR tool call?
                  ↓ (tool call)
        Structured JSON (name + args)
                  ↓
        Application executes function
                  ↓
        Result fed back to LLM
```

#### Core Concepts

- Function/tool definition with name, description, and parameter schema
- Structured output: JSON arguments matching a schema
- Application-side execution: the LLM does not run the tool; your code does
- Return of tool results into the conversation context
- Deciding when tool calling is needed vs. plain text response
- Handling tool call errors and retry logic

#### How It Works

1. You define a tool schema (name, description, parameters with types).
2. You send the prompt plus tool schemas to the LLM API.
3. The LLM either returns a text response or a tool-call request with structured arguments.
4. Your code detects the tool call, executes the real function, and appends the result.
5. You send the updated conversation back to the LLM for further reasoning.

#### Syntax & Implementation

```python
from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "City name"
                    }
                },
                "required": ["city"]
            }
        }
    }
]

def get_weather(city: str) -> str:
    # In production, call a real weather API
    return f"Weather in {city}: 22°C, clear sky"

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is the weather in Tokyo?"}],
    tools=tools,
    tool_choice="auto"
)

message = response.choices[0].message

if message.tool_calls:
    tool_call = message.tool_calls[0]
    args = eval(tool_call.function.arguments)  # use json.loads in production
    result = get_weather(**args)
    print(f"Tool result: {result}")
else:
    print(message.content)
```

#### Simple Example

```python
import json

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

def calculate(expression: str) -> float:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        raise ValueError("Invalid characters in expression")
    return eval(expression)

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What is 42 * 17 + 3?"}],
    tools=tools
)

msg = response.choices[0].message
if msg.tool_calls:
    args = json.loads(msg.tool_calls[0].function.arguments)
    result = calculate(args["expression"])
    print(result)  # 717
```

#### Common Mistakes

- Letting the LLM execute tools directly instead of application-side execution
- Using `eval()` without input validation (security risk)
- Missing error handling when tool execution fails
- Providing vague tool descriptions that confuse the LLM
- Not returning tool results into the conversation context

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| LLM ignores tools entirely | Tool descriptions too vague or irrelevant | Check tool schema and description clarity | Improve descriptions; ensure they match the user's intent |
| LLM calls wrong tool | Multiple tools with overlapping descriptions | Compare tool names and descriptions | Make each tool's purpose distinct and narrow |
| Tool call arguments are malformed | Schema is ambiguous or incomplete | Inspect the returned JSON | Add required fields, enums, and clearer descriptions |
| Application crashes on tool call | No error handling for tool execution | Wrap tool calls in try/except | Handle failures gracefully and return error to LLM |
| LLM hallucinates tool names not in schema | Schema not provided or token limit exceeded | Check API request payload | Ensure all tools are in the schema and context fits |

#### Best Practices

- Define precise tool schemas with clear descriptions and constrained parameter types
- Always execute tools application-side; never let the LLM run code directly
- Validate tool arguments before execution
- Return structured results (success/failure) to the LLM
- Log every tool call and result for debugging
- Start with `tool_choice="auto"` and only constrain when needed

#### Hands-On Practice

1. **Basic:** Define a tool schema and parse the LLM's tool-call response.
2. **Guided:** Execute the tool and return the result to the LLM for a complete cycle.
3. **Independent:** Build a calculator agent that routes math vs. text queries.
4. **Realistic:** Handle tool execution errors and return meaningful messages to the LLM.
5. **Challenge:** Add logging to every tool call and build a simple audit trail.

#### Knowledge Check

- Who executes the tool — the LLM or your application code?
- What happens if the LLM returns a tool call with missing required arguments?
- Why should tool descriptions be specific rather than generic?
- How do you ensure the LLM knows what results to expect from a tool?

#### Exit Criteria

- You can define tool schemas and parse structured tool-call responses.
- You can execute tools application-side and feed results back.
- You can handle tool errors without crashing the agent.

#### Next Step → Unit 14.2

---

### Unit 14.2 — Single-Tool Agent

**What is it?**
A single-tool agent wraps an LLM in a loop that repeatedly calls one tool until a task is complete or a stop condition is met.

**Why does it matter?**
The agent loop is the fundamental architecture. Understanding a minimal loop — observe, decide, act, repeat — is prerequisite for every more complex agent pattern.

**Why learn it here?**
Tool calling (14.1) showed how to make one LLM call produce a tool invocation. Now the learner adds a loop, state tracking, and termination logic.

**Prerequisites:** Unit 14.1 (tool calling).

#### Mental Model

A single-tool agent is a thermostat for tasks: it checks the current state, decides if action is needed, takes action, and checks again.

```text
while not done:
    response = llm(messages, tools)
    if tool_call:
        result = execute_tool(tool_call)
        messages.append(tool_result)
    else:
        done = True
    steps += 1
    if steps > max_steps:
        break
```

#### Core Concepts

- The observe-decide-act loop
- Accumulating conversation history (messages list)
- Stop conditions: LLM returns text (no tool call), max steps reached, explicit done signal
- Step counting to prevent infinite loops
- State inspection between iterations

#### How It Works

1. Send initial prompt to LLM with tool schema.
2. If LLM requests a tool call, execute it and append the result to messages.
3. Send updated messages back to LLM.
4. Repeat until LLM returns a final text response or step limit is hit.
5. The LLM's text response is the agent's answer.

#### Syntax & Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

def search_knowledge_base(query: str) -> str:
    kb = {
        "python": "Python is a high-level programming language.",
        "agent": "An agent is an LLM in a loop with tools.",
        "rag": "RAG retrieves documents before generating answers."
    }
    for key, value in kb.items():
        if key in query.lower():
            return value
    return f"No results found for '{query}'"

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search internal knowledge base for information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"]
            }
        }
    }
]

def run_agent(user_question: str, max_steps: int = 5) -> str:
    messages = [{"role": "user", "content": user_question}]
    system = (
        "You are a helpful assistant. Use the search tool to find information. "
        "When you have enough information, provide a final answer without calling tools."
    )
    messages.insert(0, {"role": "system", "content": system})

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = search_knowledge_base(**args)
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })
            print(f"Step {step + 1}: called tool")
        else:
            print(f"Step {step + 1}: final answer")
            return msg.content

    return "Agent reached max steps without completing the task."

answer = run_agent("What is an agent?")
print(answer)
```

#### Real-World Example

A customer support agent that searches a knowledge base for relevant articles before answering a customer question, looping until it finds a good answer or reaches its step limit.

#### Common Mistakes

- No max step limit, causing infinite loops
- Not appending tool results to messages, so the LLM loses context
- Forgetting the system prompt that tells the LLM when to stop using tools
- Clearing conversation history between loop iterations

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Agent loops forever | No stop condition or step limit | Add print statements for each step | Add `max_steps` and a done condition |
| Agent gives final answer without searching | System prompt does not instruct tool use | Check system message | Add instructions to always search first |
| Agent ignores tool results | Results not appended to messages | Inspect messages list | Append tool results with role "tool" |
| Agent crashes after tool call | Missing tool_call_id in tool result | Check message format | Include `tool_call_id` from the tool call |

#### Best Practices

- Always set a maximum step limit (5–10 for simple tasks)
- Log each step for debugging
- Use a clear system prompt that tells the LLM when to use tools vs. when to answer
- Accumulate messages correctly: assistant message with tool_calls, then tool messages with results
- Test with trivial queries first to verify the loop works

#### Hands-On Practice

1. **Basic:** Trace through a 2-step agent interaction by hand.
2. **Guided:** Build the single-tool agent above and run it with a search query.
3. **Independent:** Add a second knowledge base topic and verify correct routing.
4. **Realistic:** Add error handling for a tool that sometimes returns no results.
5. **Challenge:** Log every step to a file and build a trace viewer.

#### Knowledge Check

- What signals the agent to stop looping?
- Why must tool results be appended to the messages list?
- What happens if you forget to add `tool_call_id` to the tool result message?
- How does the agent know it has enough information to stop?

#### Exit Criteria

- You can trace every step of a single-tool agent loop.
- You can explain why the loop stops and what prevents infinite execution.
- You can build a minimal agent from scratch without a tutorial.

#### Next Step → Unit 14.3

---

### Unit 14.3 — Planning

**What is it?**
Planning adds a phase where the LLM decomposes a complex task into steps before executing them, rather than acting reactively.

**Why does it matter?**
Without planning, agents react to each tool result without a strategy. Planning improves accuracy on multi-step tasks and makes agent behavior more predictable and inspectable.

**Why learn it here?**
The single-tool agent (14.2) showed reactive behavior. Planning adds proactive reasoning: the agent decides what to do before doing it.

**Prerequisites:** Unit 14.2 (agent loop).

#### Mental Model

Planning is the difference between navigating with turn-by-turn directions vs. looking at the full route first. A plan lets the agent allocate tools to steps and detect when the task is complete.

```text
User request → LLM generates plan (list of steps)
    ↓
For each step:
    Execute step → Verify result → Update plan status
    ↓
All steps done → Synthesize answer
```

#### Core Concepts

- Task decomposition: breaking a complex goal into ordered steps
- Plan representation: numbered list or structured JSON
- Step-by-step execution with plan status tracking
- Re-planning: updating the plan when a step fails or returns unexpected results
- Plan visibility: outputting the plan for human inspection

#### Syntax & Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

def plan_task(task: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Given a task, return a numbered list of concrete steps to complete it. "
                "Return ONLY the numbered list, nothing else."
            )},
            {"role": "user", "content": task}
        ]
    )
    text = response.choices[0].message.content
    steps = [line.strip() for line in text.split("\n") if line.strip()]
    return steps

def execute_step(step: str) -> str:
    # In production, each step might call a different tool
    return f"Completed: {step}"

def run_with_plan(task: str) -> dict:
    steps = plan_task(task)
    print("Plan:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    results = []
    for i, step in enumerate(steps, 1):
        result = execute_step(step)
        results.append({"step": i, "description": step, "result": result})
        print(f"  Step {i}: {result}")

    return {"steps": steps, "results": results}

output = run_with_plan("Research Python async, write example code, and explain it")
```

#### Simple Example

```python
# A planning agent for a data analysis task
def plan_analysis(dataset_path: str) -> list[str]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Given a data analysis task, return a numbered list of steps. "
                "Each step should be one concrete action."
            )},
            {"role": "user", "content": f"Analyze the dataset at {dataset_path}"}
        ]
    )
    text = response.choices[0].message.content
    return [line.strip() for line in text.split("\n") if line.strip()]

plan = plan_analysis("sales_2024.csv")
for i, step in enumerate(plan, 1):
    print(f"{i}. {step}")
# Output might be:
# 1. Load the CSV and inspect its structure
# 2. Check for missing values and data types
# 3. Compute summary statistics for key columns
# 4. Create visualizations for trends and distributions
# 5. Write a summary report
```

#### Common Mistakes

- Plan is too vague ("analyze the data") instead of concrete steps
- No re-planning when a step fails
- Executing all steps before checking intermediate results
- Not saving the plan for inspection

#### Best Practices

- Generate concrete, executable steps, not abstract goals
- Display the plan to the user before execution when possible
- Track step status: pending, in-progress, completed, failed
- Allow re-planning when results are unexpected
- Store the plan for debugging and audit

#### Hands-On Practice

1. **Basic:** Generate a plan for a simple multi-step task and print it.
2. **Guided:** Execute each step and print results alongside the plan.
3. **Independent:** Add status tracking (pending/done/failed) to each step.
4. **Realistic:** Implement re-planning when a step fails.
5. **Challenge:** Compare planned execution vs. reactive execution for the same task.

#### Knowledge Check

- Why is planning better than pure reactive execution for complex tasks?
- When should the agent re-plan instead of continuing with the original plan?
- What makes a plan step "concrete enough" to execute?

#### Exit Criteria

- You can decompose a task into concrete executable steps.
- You can track plan status through execution.
- You can explain when planning improves agent reliability.

#### Next Step → Unit 14.4

---

### Unit 14.4 — Execution

**What is it?**
Execution is the process of carrying out each planned step by invoking tools, collecting results, and updating the agent's state.

**Why does it matter?**
A plan without execution is just a list. Execution connects planning to real-world action: running code, calling APIs, writing files, and producing tangible outputs.

**Why learn it here?**
Planning (14.3) showed how to decompose tasks. Execution (14.4) shows how to carry them out reliably with proper error handling and result tracking.

**Prerequisites:** Unit 14.3 (planning), Unit 14.1 (tool calling).

#### Mental Model

Execution is a pipeline: each step consumes the previous step's output and produces input for the next. Failures at any stage must be caught, logged, and handled.

```text
Plan step → Select tool → Prepare arguments → Execute → Validate result → Store → Next step
                                                                    ↓ (failure)
                                                              Log error → Handle/retry/skip
```

#### Core Concepts

- Tool selection per step based on plan
- Argument preparation from plan and context
- Result validation: checking if output matches expectations
- Error handling: catching failures, retrying, or skipping
- Result accumulation: building a history of what was done
- Side effects: writing files, modifying state, sending messages

#### Syntax & Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

class ExecutionEngine:
    def __init__(self):
        self.tools = {}
        self.results = []

    def register_tool(self, name: str, func):
        self.tools[name] = func

    def execute_step(self, step: dict) -> dict:
        tool_name = step["tool"]
        args = step.get("args", {})

        if tool_name not in self.tools:
            return {"status": "error", "message": f"Unknown tool: {tool_name}"}

        try:
            result = self.tools[tool_name](**args)
            entry = {"step": step, "status": "success", "result": result}
        except Exception as e:
            entry = {"step": step, "status": "error", "message": str(e)}

        self.results.append(entry)
        return entry

engine = ExecutionEngine()
engine.register_tool("read_file", lambda path: f"Contents of {path}")
engine.register_tool("write_file", lambda path, content: f"Wrote to {path}")

step1 = {"tool": "read_file", "args": {"path": "data.csv"}}
step2 = {"tool": "write_file", "args": {"path": "output.txt", "content": "processed data"}}

r1 = engine.execute_step(step1)
r2 = engine.execute_step(step2)

for r in engine.results:
    print(f"{r['status']}: {r.get('result', r.get('message'))}")
```

#### Real-World Example

An agent that reads a CSV file, computes summary statistics using a Python function, writes results to a new file, and generates a summary email — each step building on the previous one.

#### Common Mistakes

- Not validating results between steps (a failed step poisons downstream)
- Ignoring exceptions and silently continuing
- Not recording what was actually executed
- Assuming side effects (file writes, API calls) are reversible

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Step succeeds but downstream fails | Result format unexpected | Inspect result type and shape | Validate result before passing to next step |
| Agent continues after error | Exception swallowed silently | Check error handling | Log errors and decide: retry, skip, or abort |
| Side effects happen twice | Step retried after transient failure | Check idempotency | Make tools idempotent or track execution state |
| Results are empty | Tool returned None or empty string | Inspect tool output | Ensure tools always return structured results |

#### Best Practices

- Validate each step's result before using it in the next step
- Make tools idempotent where possible (safe to retry)
- Log every execution with timestamps and inputs/outputs
- Track side effects explicitly (files written, APIs called)
- Use structured result formats (dict with status + data) instead of raw strings

#### Hands-On Practice

1. **Basic:** Build an execution engine that runs a list of steps sequentially.
2. **Guided:** Add error handling that catches and logs failures per step.
3. **Independent:** Add result validation between steps.
4. **Realistic:** Implement retry logic for transient failures.
5. **Challenge:** Add idempotency tracking so retried steps do not duplicate side effects.

#### Knowledge Check

- Why must you validate results between execution steps?
- What is idempotency and why does it matter for agents?
- How do you handle a step that fails but is critical for the plan?

#### Exit Criteria

- You can execute planned steps with proper error handling.
- You can validate results and handle failures gracefully.
- You can track what was executed and what side effects occurred.

#### Next Step → Unit 14.5

---

### Unit 14.5 — Reflection

**What is it?**
Reflection is a self-evaluation step where the agent reviews its own output or tool results before finalizing a response.

**Why does it matter?**
LLMs can produce confident but wrong outputs. Reflection catches errors, inconsistencies, and missing information before the agent returns a final answer. It is a lightweight quality gate.

**Why learn it here?**
Execution (14.4) showed how to carry out steps. Reflection adds a verification layer: the agent checks its own work.

**Prerequisites:** Unit 14.4 (execution), Unit 14.2 (agent loop).

#### Mental Model

Reflection is the agent looking in a mirror: it compares its output against the original task, checks for gaps, and decides whether to revise or finalize.

```text
Agent generates draft answer
    ↓
LLM reviews draft against task requirements
    ↓
If issues found → revise → re-review
If acceptable → return final answer
```

#### Core Concepts

- Self-critique: the LLM evaluates its own output
- Revision loop: generating improved versions based on critique
- Quality criteria: completeness, accuracy, relevance, format
- Bounded revision: a maximum number of revision rounds
- Critique prompts: structured evaluation questions

#### Syntax & Implementation

```python
from openai import OpenAI

client = OpenAI()

def reflect_and_revise(task: str, draft: str, max_revisions: int = 2) -> str:
    current = draft

    for revision in range(max_revisions):
        critique_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    "Review this draft response against the task. "
                    "List specific issues: missing information, errors, unclear parts. "
                    "If no issues remain, respond with just: APPROVED"
                )},
                {"role": "user", "content": f"Task: {task}\n\nDraft: {current}"}
            ]
        )

        critique = critique_response.choices[0].message.content

        if "APPROVED" in critique:
            print(f"Approved after {revision} revision(s)")
            return current

        revision_response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "Revise the response to fix the listed issues."},
                {"role": "user", "content": f"Original task: {task}\n\nCritique: {critique}\n\nCurrent draft: {current}"}
            ]
        )
        current = revision_response.choices[0].message.content
        print(f"Revision {revision + 1} complete")

    return current

final = reflect_and_revise(
    "Explain the difference between supervised and unsupervised learning",
    "Supervised learning uses labeled data. Unsupervised learning does not."
)
print(final)
```

#### Simple Example

```python
# Reflection as a post-processing step
def check_answer(task: str, answer: str) -> tuple[bool, list[str]]:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": (
                "Evaluate this answer. Return a JSON with:\n"
                "- 'issues': list of problems\n"
                "- 'score': 1-5 quality rating"
            )},
            {"role": "user", "content": f"Task: {task}\nAnswer: {answer}"}
        ],
        response_format={"type": "json_object"}
    )
    import json
    evaluation = json.loads(response.choices[0].message.content)
    issues = evaluation.get("issues", [])
    return len(issues) == 0, issues

approved, issues = check_answer(
    "List 3 benefits of RAG",
    "RAG reduces hallucination."
)
print(f"Approved: {approved}, Issues: {issues}")
```

#### Common Mistakes

- Reflecting too many times (wasting tokens and time)
- Using the same model to reflect and generate (limited self-correction)
- Not bounding revision loops (infinite improvement attempts)
- Reflection is too vague ("make it better") instead of specific

#### Best Practices

- Limit reflection to 1–2 rounds
- Use specific critique criteria (completeness, accuracy, format)
- Log each revision for debugging
- Consider a different model or prompt for reflection vs. generation
- Only reflect when the task has verifiable quality criteria

#### Hands-On Practice

1. **Basic:** Implement a one-round reflection that critiques and revises a draft.
2. **Guided:** Add specific evaluation criteria to the critique prompt.
3. **Independent:** Build a reflection loop with max_revisions and logging.
4. **Realistic:** Apply reflection to a tool-using agent's intermediate results.
5. **Challenge:** Compare output quality with and without reflection on 5 test cases.

#### Knowledge Check

- Why is bounded revision important?
- What happens when you use the same model for generation and reflection?
- When should you skip reflection entirely?

#### Exit Criteria

- You can implement reflection that catches specific issues.
- You can bound revision loops to prevent waste.
- You can explain when reflection improves quality vs. when it does not.

#### Next Step → Unit 14.6

---

### Unit 14.6 — Memory

**What is it?**
Memory gives agents the ability to retain information across turns and interactions: short-term (within a session) and long-term (across sessions).

**Why does it matter?**
Without memory, every interaction starts from zero. Memory enables personalization, context accumulation, learning from past mistakes, and continuity across long conversations.

**Why learn it here?**
Planning (14.3) and execution (14.4) handle single-task execution. Memory adds persistence, allowing agents to build on prior context and improve over time.

**Prerequisites:** Unit 14.2 (agent loop), understanding of conversation history.

#### Mental Model

```text
Short-term memory = conversation messages (current session)
Long-term memory = stored facts, preferences, past results (persistent)

Agent at each turn:
  1. Load relevant long-term memory
  2. Combine with current conversation
  3. Decide and act
  4. Update short-term (messages) and optionally long-term memory
```

#### Core Concepts

- Short-term memory: the messages list in a conversation
- Long-term memory: a persistent store (file, database, vector store)
- Memory retrieval: finding relevant past information for the current task
- Memory update: saving new information after tool results or conclusions
- Memory consolidation: summarizing long conversations into key facts
- Forgetting: pruning old or irrelevant memory

#### Syntax & Implementation

```python
import json
from pathlib import Path
from openai import OpenAI

client = OpenAI()

class MemoryStore:
    def __init__(self, path: str = "memory.json"):
        self.path = Path(path)
        self.facts = self._load()

    def _load(self) -> list[dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []

    def save(self, fact: str, category: str = "general"):
        self.facts.append({
            "fact": fact,
            "category": category,
            "timestamp": len(self.facts)
        })
        self.path.write_text(json.dumps(self.facts, indent=2))

    def search(self, query: str) -> list[str]:
        # Simple keyword search; production would use embeddings
        results = []
        for entry in self.facts:
            if any(word in entry["fact"].lower() for word in query.lower().split()):
                results.append(entry["fact"])
        return results[-5:]  # Return last 5 relevant facts

memory = MemoryStore()
memory.save("User prefers concise answers", "preference")
memory.save("User is working on a RAG project", "context")

relevant = memory.search("user preferences")
print(relevant)  # ['User prefers concise answers']
```

#### Real-World Example

A coding assistant that remembers the user's preferred language, framework, and past projects. Each session loads relevant memories, and after the session, key facts are saved for future use.

#### Common Mistakes

- Storing everything without pruning (memory bloat)
- Retrieving irrelevant memories that confuse the LLM
- Not persisting memory (lost on restart)
- Storing sensitive data without access controls

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Agent repeats itself across turns | Memory not loaded into context | Check if memory is prepended to messages | Load relevant memory at session start |
| Agent contradicts past decisions | Memory retrieval returns irrelevant facts | Inspect retrieved facts | Improve retrieval with semantic search |
| Memory grows unbounded | No pruning strategy | Check memory size | Summarize and consolidate old facts |
| Agent forgets within a session | Messages list truncated | Check message count vs. context limit | Summarize older messages when approaching limit |

#### Best Practices

- Distinguish short-term (session) and long-term (persistent) memory
- Use semantic search (embeddings) for retrieval in production
- Consolidate and summarize memory periodically
- Add timestamps and categories to stored facts
- Limit memory retrieved to the most relevant items
- Never store secrets or sensitive data in plain text

#### Hands-On Practice

1. **Basic:** Implement a key-value memory store and retrieve facts.
2. **Guided:** Add memory retrieval to an agent loop.
3. **Independent:** Implement memory consolidation (summarize old facts).
4. **Realistic:** Add semantic search using embeddings for retrieval.
5. **Challenge:** Build a memory system that forgets irrelevant facts automatically.

#### Knowledge Check

- What is the difference between short-term and long-term agent memory?
- How do you prevent memory from growing unbounded?
- Why is simple keyword search insufficient for production memory retrieval?

#### Exit Criteria

- You can implement both session and persistent memory.
- You can retrieve relevant memories for the current context.
- You can prevent memory bloat through consolidation.

#### Next Step → Unit 14.7

---

### Unit 14.7 — Multi-Tool Agent

**What is it?**
A multi-tool agent can select from multiple available tools based on the task, choosing the right tool for each step.

**Why does it matter?**
Real tasks require different capabilities: searching, calculating, writing, reading files, calling APIs. The agent must route to the correct tool based on context.

**Why learn it here?**
Single-tool agents (14.2) showed the basic loop. Multi-tool agents add tool selection logic: the LLM must choose which tool to use from a set of options.

**Prerequisites:** Unit 14.2 (agent loop), Unit 14.1 (tool calling).

#### Mental Model

```text
Available tools: [search, calculate, write_file, read_file, summarize]
    ↓
User task → LLM evaluates which tool is needed
    ↓
LLM picks tool → executes → may pick another tool
    ↓
Agent assembles final answer from tool results
```

#### Core Concepts

- Defining multiple tools with clear, distinct descriptions
- Tool selection: LLM picks the best tool for the current need
- Tool chaining: using output of one tool as input to another
- Reducing tool confusion: clear names and non-overlapping descriptions
- Handling tool unavailability: graceful fallback when no tool fits

#### Syntax & Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

def web_search(query: str) -> str:
    return f"Search results for '{query}': [simulated results]"

def calculator(expression: str) -> str:
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "Error: invalid characters"
    return str(eval(expression))

def read_database(query: str) -> str:
    return f"Database results for '{query}': [simulated data]"

tools = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the internet for current information",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate a mathematical expression",
            "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}, "required": ["expression"]}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_database",
            "description": "Query an internal database for stored records",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
        }
    }
]

tool_map = {"web_search": web_search, "calculator": calculator, "read_database": read_database}

def run_multi_tool_agent(task: str, max_steps: int = 8) -> str:
    messages = [
        {"role": "system", "content": (
            "You have access to web search, a calculator, and a database. "
            "Use the most appropriate tool for each part of the task. "
            "When you have enough information, provide a final answer."
        )},
        {"role": "user", "content": task}
    ]

    for step in range(max_steps):
        response = client.chat.completions.create(
            model="gpt-4o", messages=messages, tools=tools, tool_choice="auto"
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            for tc in msg.tool_calls:
                name = tc.function.name
                args = json.loads(tc.function.arguments)
                result = tool_map[name](**args)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            return msg.content

    return "Max steps reached."

answer = run_multi_tool_agent(
    "What is the population of France multiplied by 3.14?"
)
print(answer)
```

#### Common Mistakes

- Tools with overlapping descriptions cause confusion
- Too many tools overwhelm the LLM (tool bloat)
- Not providing examples of when to use each tool
- Ignoring tool chaining opportunities

#### Best Practices

- Keep tool set focused; fewer tools are better when possible
- Make each tool's purpose singular and distinct
- Add usage examples in tool descriptions when helpful
- Monitor which tools the LLM selects and refine descriptions based on errors
- Group related tools into sub-agents if the set grows large

#### Hands-On Practice

1. **Basic:** Define 3 tools with distinct purposes and verify correct selection.
2. **Guided:** Chain two tools: use output of one as input to another.
3. **Independent:** Build a multi-tool agent that handles a mixed task (search + calculate).
4. **Realistic:** Add a fallback when no tool matches the request.
5. **Challenge:** Benchmark tool selection accuracy across 10 diverse queries.

#### Knowledge Check

- Why do overlapping tool descriptions cause incorrect tool selection?
- How many tools is "too many" for a single agent?
- What strategy do you use when a task requires multiple tools?

#### Exit Criteria

- You can define and route among multiple tools correctly.
- You can chain tool outputs into subsequent tool calls.
- You can diagnose and fix tool selection errors.

#### Next Step → Unit 14.8

---

### Unit 14.8 — Stateful Agent

**What is it?**
A stateful agent maintains persistent state across multiple user turns and sessions, going beyond simple conversation history to track goals, progress, and preferences.

**Why does it matter?**
Real-world agents need to remember what they were doing, track multi-turn tasks, and adapt behavior based on accumulated context. Stateless agents restart from scratch every time.

**Why learn it here?**
Memory (14.6) showed how to store and retrieve facts. Stateful agents combine memory with structured state management: the agent knows what step it is on, what has been completed, and what remains.

**Prerequisites:** Unit 14.6 (memory), Unit 14.7 (multi-tool agent).

#### Mental Model

```text
Agent state = {
    current_goal: str,
    completed_steps: list,
    pending_steps: list,
    context: dict,
    memory: list
}

Each turn:
  1. Load state
  2. Decide next action based on state
  3. Execute
  4. Update state
  5. Persist state
```

#### Core Concepts

- Structured state representation (dict/dataclass)
- State transitions: moving through defined states
- State persistence: saving state to disk or database
- State recovery: resuming after interruption
- Goal tracking: monitoring progress toward completion
- Context windows: managing what state is visible to the LLM

#### Syntax & Implementation

```python
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from openai import OpenAI

client = OpenAI()

@dataclass
class AgentState:
    goal: str = ""
    current_step: int = 0
    completed: list = field(default_factory=list)
    results: dict = field(default_factory=dict)
    status: str = "idle"  # idle, planning, executing, done

class StatefulAgent:
    def __init__(self, state_path: str = "agent_state.json"):
        self.state_path = Path(state_path)
        self.state = self._load_state()

    def _load_state(self) -> AgentState:
        if self.state_path.exists():
            data = json.loads(self.state_path.read_text())
            return AgentState(**data)
        return AgentState()

    def save_state(self):
        self.state_path.write_text(json.dumps(asdict(self.state), indent=2))

    def run(self, user_input: str) -> str:
        self.state.goal = user_input
        self.state.status = "planning"
        self.save_state()

        messages = [
            {"role": "system", "content": (
                f"Goal: {self.state.goal}\n"
                f"Completed: {self.state.completed}\n"
                f"Results: {self.state.results}\n"
                "Decide the next step or provide a final answer."
            )},
            {"role": "user", "content": user_input}
        ]

        response = client.chat.completions.create(
            model="gpt-4o", messages=messages
        )
        answer = response.choices[0].message.content

        self.state.completed.append({"input": user_input, "response": answer[:100]})
        self.state.status = "done"
        self.save_state()

        return answer

agent = StatefulAgent()
print(agent.run("Research Python async patterns"))
print(agent.run("Now summarize what we found"))
print(json.loads(agent.state_path.read_text()))
```

#### Common Mistakes

- State grows unbounded (no pruning)
- State not persisted, lost on restart
- Exposing all state to LLM (context overflow)
- Race conditions in concurrent access

#### Best Practices

- Design state schema explicitly before implementation
- Persist state after every meaningful transition
- Prune state when it exceeds context limits
- Use versioned state for rollback capability
- Serialize state to JSON for inspectability

#### Hands-On Practice

1. **Basic:** Implement a stateful agent that tracks goal and completed steps.
2. **Guided:** Add state persistence and recovery across sessions.
3. **Independent:** Implement state transitions with status tracking.
4. **Realistic:** Add state pruning for long-running agents.
5. **Challenge:** Build state recovery that handles crashes gracefully.

#### Knowledge Check

- How does stateful agent differ from a simple memory store?
- Why must state be persisted after each transition?
- How do you prevent state from growing beyond context limits?

#### Exit Criteria

- You can implement structured state with persistence.
- You can resume an agent from saved state.
- You can manage state lifecycle (create, update, prune, restore).

#### Next Step → Unit 14.9

---

### Unit 14.9 — Multi-Agent Systems

**What is it?**
Multi-agent systems coordinate multiple specialized agents to solve complex tasks that are too large or diverse for a single agent.

**Why does it matter?**
Some tasks require different expertise: one agent researches, another writes, a third reviews. Coordination between agents enables decomposition, specialization, and quality control.

**Why learn it here?**
Single-agent systems (14.2–14.8) showed how one agent can plan, execute, and reflect. Multi-agent systems add coordination, delegation, and communication between agents.

**Prerequisites:** Unit 14.8 (stateful agent), Unit 14.5 (reflection).

#### Mental Model

```text
Orchestrator agent receives task
    ↓
Breaks task into subtasks
    ↓
Assigns subtasks to specialist agents
    ↓
Specialist agents execute independently
    ↓
Orchestrator collects results and synthesizes
```

#### Core Concepts

- Orchestrator agent: coordinates and delegates
- Specialist agents: each handles one type of task
- Communication protocol: how agents share information
- Task decomposition: breaking tasks for delegation
- Result aggregation: combining specialist outputs
- Failure isolation: one agent failing does not crash the system

#### Syntax & Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

class SpecialistAgent:
    def __init__(self, name: str, system_prompt: str):
        self.name = name
        self.system_prompt = system_prompt

    def run(self, task: str) -> str:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": task}
            ]
        )
        return response.choices[0].message.content

class OrchestratorAgent:
    def __init__(self):
        self.specialists = {}

    def add_specialist(self, name: str, agent: SpecialistAgent):
        self.specialists[name] = agent

    def decompose(self, task: str) -> list[dict]:
        specialist_list = ", ".join(self.specialists.keys())
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": (
                    f"Break this task into subtasks. Available specialists: {specialist_list}\n"
                    "Return JSON array with objects: {{\"specialist\": \"name\", \"task\": \"subtask description\"}}"
                )},
                {"role": "user", "content": task}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def run(self, task: str) -> dict:
        plan = self.decompose(task)
        results = {}
        subtasks = plan.get("subtasks", plan) if isinstance(plan, dict) else plan

        for item in subtasks:
            specialist_name = item["specialist"]
            subtask = item["task"]
            if specialist_name in self.specialists:
                result = self.specialists[specialist_name].run(subtask)
                results[specialist_name] = result

        return results

orchestrator = OrchestratorAgent()
orchestrator.add_specialist("researcher", SpecialistAgent(
    "Researcher", "You are a research specialist. Find and summarize information."
))
orchestrator.add_specialist("writer", SpecialistAgent(
    "Writer", "You are a writing specialist. Create clear, well-structured content."
))

results = orchestrator.run("Write a guide on Python async/await")
for name, result in results.items():
    print(f"\n--- {name} ---\n{result[:200]}")
```

#### Common Mistakes

- Too many agents (coordination overhead exceeds benefit)
- Agents have overlapping responsibilities
- No failure handling when an agent fails
- Unclear communication protocol between agents
- Orchestrator does too much instead of delegating

#### Decision Guidance: Single Agent vs. Multi-Agent

| Use Single Agent When | Use Multi-Agent When |
|---|---|
| Task is coherent and focused | Task requires distinct expertise areas |
| One model can handle all parts | Different specialists improve quality |
| Speed is critical | Parallel execution speeds up the process |
| Simplicity is a priority | Complexity justifies coordination overhead |

#### Best Practices

- Start with a single agent; add agents only when needed
- Give each specialist a clear, narrow responsibility
- Isolate failures so one agent's failure does not crash the system
- Log inter-agent communication for debugging
- Use structured message formats between agents

#### Hands-On Practice

1. **Basic:** Build two specialist agents and an orchestrator.
2. **Guided:** Decompose a task and delegate to specialists.
3. **Independent:** Add result aggregation and synthesis.
4. **Realistic:** Add failure handling when one specialist fails.
5. **Challenge:** Implement parallel execution for independent subtasks.

#### Knowledge Check

- When does adding a second agent improve results vs. increase complexity?
- How do you handle a specialist agent that returns poor results?
- What communication format works best between agents?

#### Exit Criteria

- You can decompose tasks and delegate to specialist agents.
- You can aggregate results from multiple agents.
- You can handle agent failures gracefully.

#### Next Step → Unit 14.10

---

### Unit 14.10 — Agent Failure Modes

**What is it?**
A systematic study of the ways agents fail: hallucinated tools, infinite loops, context explosion, stale memory, prompt injection, and unsafe actions.

**Why does it matter?**
Agents fail in ways that differ from traditional software. Understanding these failure modes is critical for building reliable, production-ready agents.

**Why learn it here?**
By this point the learner has built working agents (14.1–14.9). Now they learn what breaks and how to prevent it — essential for production use.

**Prerequisites:** All previous units in Phase 14.

#### Mental Model

Agent failures cluster into categories: reasoning failures (wrong tool, wrong args), control failures (loops, crashes), context failures (overflow, stale data), and security failures (injection, unsafe actions).

#### Core Concepts

- Hallucinated tool calls: LLM invents tools not in the schema
- Invalid tool arguments: malformed or missing required parameters
- Infinite loops: no stop condition or weak termination logic
- Context explosion: conversation grows beyond token limits
- Stale memory: agent acts on outdated information
- Prompt injection: user input manipulates agent behavior
- Unsafe tool requests: agent asks to perform dangerous actions
- Silent failures: tool returns error but agent ignores it

#### Failure Mode Reference

| Failure Mode | Description | Prevention |
|---|---|---|
| Hallucinated tools | LLM calls tools not in schema | Validate tool names against registered tools |
| Invalid arguments | Missing or wrong-type parameters | Schema validation, type checking before execution |
| Infinite loops | Agent never reaches stop condition | Max steps, timeout, done signal |
| Context overflow | Messages exceed token limit | Summarize old messages, sliding window |
| Stale memory | Agent acts on outdated facts | Timestamp memory, refresh before use |
| Prompt injection | User input overrides system instructions | Input sanitization, instruction hierarchy |
| Unsafe actions | Agent requests dangerous operations | Allowlists, approval gates, sandboxing |
| Silent failures | Errors swallowed without feedback | Structured error responses, logging |
| Tool selection errors | Wrong tool for the task | Better descriptions, routing rules |
| Resource exhaustion | Agent uses too many API calls/tokens | Rate limiting, budget caps |

#### Hands-On Practice

1. **Basic:** Reproduce each failure mode in a controlled test.
2. **Guided:** Add prevention for infinite loops (max steps, timeout).
3. **Independent:** Add tool validation to reject hallucinated tool calls.
4. **Realistic:** Inject adversarial user input and verify the agent handles it safely.
5. **Challenge:** Build a monitoring system that detects and logs failure modes in real time.

#### Knowledge Check

- How does prompt injection differ from traditional security attacks?
- Why is "silent failure" more dangerous than an explicit crash?
- What is the most common agent failure mode in production systems?

#### Exit Criteria

- You can identify at least 8 distinct agent failure modes.
- You can implement prevention for each major failure type.
- You can reproduce and mitigate failures in controlled tests.

#### Next Step → Unit 14.11

---

### Unit 14.11 — Agent Security

**What is it?**
Agent security covers protecting agents from prompt injection, tool misuse, data leakage, unauthorized actions, and ensuring safe operation in production.

**Why does it matter?**
Agents take actions. A compromised agent can leak data, execute unauthorized operations, or cause real-world harm. Security is not optional — it is a core engineering requirement.

**Why learn it here?**
Failure modes (14.10) showed what breaks. Security (14.11) shows how to build agents that resist attack and operate within safe boundaries.

**Prerequisites:** Unit 14.10 (failure modes), all prior units.

#### Mental Model

Agent security is defense in depth: multiple layers of protection, because no single measure is sufficient.

```text
Layer 1: Input validation (reject malicious input)
Layer 2: Tool allowlists (only approved tools available)
Layer 3: Argument validation (reject dangerous parameters)
Layer 4: Approval gates (human confirm for risky actions)
Layer 5: Output filtering (prevent data leakage)
Layer 6: Audit logging (trace everything)
```

#### Core Concepts

- Prompt injection defense: system prompt protection, input sanitization
- Tool access control: allowlists, denylists, role-based access
- Argument validation: schema enforcement, type checking, dangerous value detection
- Human-in-the-loop approval for risky actions
- Output filtering: prevent sensitive data from leaking in responses
- Audit logging: record every tool call, argument, and result
- Sandboxing: limit what the agent can access
- Rate limiting: prevent runaway API usage
- Secrets management: never expose API keys or credentials to the LLM

#### Syntax & Implementation

```python
import json
from openai import OpenAI

client = OpenAI()

class SecureAgent:
    def __init__(self):
        self.allowed_tools = {"search", "calculate", "summarize"}
        self.approval_required = {"delete_file", "send_email", "execute_code"}
        self.audit_log = []

    def validate_tool_call(self, tool_name: str, args: dict) -> tuple[bool, str]:
        if tool_name not in self.allowed_tools:
            return False, f"Tool '{tool_name}' not in allowlist"

        dangerous_patterns = ["rm ", "DROP ", "DELETE ", "exec(", "__import__"]
        args_str = json.dumps(args)
        for pattern in dangerous_patterns:
            if pattern.lower() in args_str.lower():
                return False, f"Dangerous pattern detected: {pattern}"

        return True, "approved"

    def needs_approval(self, tool_name: str) -> bool:
        return tool_name in self.approval_required

    def log_action(self, tool_name: str, args: dict, result: str, approved: bool):
        self.audit_log.append({
            "tool": tool_name,
            "args": args,
            "result_preview": result[:100],
            "approved": approved
        })

    def run_with_security(self, task: str, max_steps: int = 5) -> str:
        messages = [
            {"role": "system", "content": (
                "You are a helpful assistant. Use only approved tools. "
                "Never attempt to access files, execute code, or send messages."
            )},
            {"role": "user", "content": task}
        ]

        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": f"Safe {name} tool",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
                }
            } for name in self.allowed_tools
        ]

        for step in range(max_steps):
            response = client.chat.completions.create(
                model="gpt-4o", messages=messages, tools=tools
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                messages.append(msg)
                for tc in msg.tool_calls:
                    name = tc.function.name
                    args = json.loads(tc.function.arguments)

                    valid, reason = self.validate_tool_call(name, args)
                    if not valid:
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": f"Security rejection: {reason}"
                        })
                        self.log_action(name, args, reason, False)
                        continue

                    if self.needs_approval(name):
                        messages.append({
                            "role": "tool", "tool_call_id": tc.id,
                            "content": "Action requires human approval. Please confirm."
                        })
                        self.log_action(name, args, "pending_approval", False)
                        continue

                    result = f"Result of {name}({args})"
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
                    self.log_action(name, args, result, True)
            else:
                return msg.content

        return "Max steps reached."

agent = SecureAgent()
print(agent.run_with_security("Summarize what Python async is"))
print(json.dumps(agent.audit_log, indent=2))
```

#### Best Practices

- Implement least-privilege access: only grant tools the agent actually needs
- Use allowlists for tools and denylists for dangerous arguments
- Require human approval for any action with real-world side effects
- Log everything: every tool call, argument, and result
- Sandboxed execution: run agent tools in isolated environments
- Rate limiting: cap API calls per session and per time window
- Never expose secrets in prompts; use environment variables and secure vaults
- Input sanitization: strip or escape user input before injecting into prompts
- Output filtering: scan responses for sensitive data before returning to users
- Regular security audits of agent configurations and logs

#### Hands-On Practice

1. **Basic:** Implement a tool allowlist and reject unknown tools.
2. **Guided:** Add argument validation for dangerous patterns.
3. **Independent:** Build an approval gate for risky actions.
4. **Realistic:** Add audit logging and review logs for suspicious patterns.
5. **Challenge:** Test prompt injection attacks and verify defenses hold.

#### Knowledge Check

- Why is least-privilege important for agent tools?
- How do you prevent an agent from leaking secrets in its responses?
- What should happen when an agent requests a dangerous action?

#### Exit Criteria

- You can implement defense-in-depth security for agents.
- You can detect and block prompt injection attempts.
- You can build audit logging and approval gates.
- You can explain the security trade-offs of different agent architectures.

#### Next Step → Unit 14.12

---

### Unit 14.12 — Agent Synthesis & Review

**What is it?**
A cumulative integration unit that combines all Phase 14 concepts: tool calling, planning, execution, reflection, memory, multi-tool selection, state, multi-agent coordination, failure prevention, and security.

**Why does it matter?**
Knowing individual components is not enough. The learner must build a complete, production-quality agent system that integrates every concept from this phase.

**Prerequisites:** All previous units in Phase 14.

---

## Mini Project — Research Assistant Agent

**Objective:** Build a multi-tool agent that researches a topic, writes a structured report, reflects on quality, and maintains memory across sessions — with full security controls and failure handling.

**Problem Statement:**
Build an agent system that takes a research question, plans an approach, gathers information from multiple sources, writes a structured report, reviews its own work, and remembers key findings for future queries. The system must be secure, observable, and handle failures gracefully.

**Requirements:**

- Tool calling with at least 3 tools (search, file read/write, calculator)
- Planning: decompose research task into steps
- Execution: carry out each step with error handling
- Reflection: review report quality before finalizing
- Memory: save key findings for future sessions
- Multi-tool selection: route to correct tools
- State: track progress through the research process
- Security: tool allowlists, input validation, audit logging
- Failure handling: max steps, timeout, graceful error messages

**Suggested Architecture:**

```text
User question
    ↓
Orchestrator: plan research steps
    ↓
Search agent: gather information
    ↓
Writer agent: draft report
    ↓
Reviewer agent: critique and revise
    ↓
Memory: save key findings
    ↓
Output: final report + audit log
```

**Milestones:**

1. Define tool schemas and basic agent loop
2. Implement planning and execution
3. Add reflection and quality review
4. Implement memory persistence
5. Add security controls and audit logging
6. Test with 3 different research questions
7. Document failures encountered and mitigations applied

**Expected Output:**

- Agent code (Python)
- 3 completed research reports
- Audit log showing every tool call and result
- Failure report documenting issues encountered and fixes
- Security review checklist
- README explaining setup, usage, and design decisions

**Evaluation Criteria:**

| Criterion | Weight | Description |
|---|---|---|
| Task completion | 25% | Agent produces complete, useful reports |
| Tool usage | 20% | Correct tool selection and error handling |
| Planning quality | 15% | Steps are logical and well-organized |
| Reflection | 10% | Report is reviewed and improved |
| Memory | 10% | Key findings persist across sessions |
| Security | 10% | Allowlists, validation, and audit logging work |
| Documentation | 10% | README is clear and covers all decisions |

**Advanced Extensions:**

- Multi-agent coordination (researcher + writer + reviewer)
- Human-in-the-loop approval for sensitive actions
- Sandboxed code execution
- Monitoring dashboard for agent activity
- A/B testing different planning strategies

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Workflow vs agent | Steps are known and reliable | Tool choice depends on context | Predictability vs flexibility |
| Single agent vs multi-agent | Task is coherent and focused | Task requires distinct expertise | Simplicity vs specialization |
| Short-term vs long-term memory | Within-session context only | Cross-session continuity needed | Speed vs persistence |
| Reflection vs no reflection | Task has verifiable quality criteria | Speed is critical and quality is obvious | Accuracy vs latency |
| Prompt injection defense layers | Low-risk internal tool | High-risk external-facing agent | Security vs usability |

---

## Agent Debugging Playbook

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Agent loops forever | No stop condition or weak state | Inspect trace | Add max steps, explicit done criteria, state guards |
| Tool call fails | Bad schema or missing context | Validate arguments | Add structured schemas and examples |
| Agent uses wrong tool | Poor tool descriptions | Review tool selection trace | Improve descriptions, add routing rules, reduce tool set |
| Agent leaks/uses unsafe data | Weak permissions | Audit inputs/tool access | Least privilege, redaction, human approval |
| Output looks good but task failed | No external verification | Check real tool/result state | Add evaluator and task-success checks |
| Agent hallucinates tool names | Schema not provided or token limit | Check API request payload | Ensure all tools are in schema and context fits |
| Context overflow | Conversation too long | Check message count | Summarize old messages, sliding window |
| Agent contradicts past decisions | Memory not loaded | Check memory retrieval | Load relevant memory at session start |

---

## Phase Review Checklist

- [ ] All units complete.
- [ ] Tool calling implemented with proper schema and application-side execution.
- [ ] Single-tool agent loop built with max steps and stop conditions.
- [ ] Planning decomposes tasks into concrete executable steps.
- [ ] Execution handles errors, validates results, and tracks side effects.
- [ ] Reflection catches issues and bounds revision loops.
- [ ] Memory persists across sessions and retrieves relevant facts.
- [ ] Multi-tool agent routes correctly among distinct tools.
- [ ] Stateful agent persists and recovers state across turns.
- [ ] Multi-agent system decomposes and delegates tasks.
- [ ] At least five failure modes reproduced and mitigated.
- [ ] Security controls implemented: allowlists, validation, approval gates, audit logs.
- [ ] Mini project completed with trace logs and safety controls.
- [ ] Workflow-vs-agent decision explained for at least 3 scenarios.
- [ ] Cumulative review passed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Define tool schemas and build a tool-calling agent from scratch.
2. Implement planning, execution, reflection, and memory in an agent.
3. Build a multi-tool agent that routes correctly among tools.
4. Implement stateful agents with persistence and recovery.
5. Coordinate multiple agents for complex tasks.
6. Identify and mitigate at least 5 agent failure modes.
7. Implement defense-in-depth security for agents.
8. Build an audit trail for every agent action.
9. Explain when to use a workflow instead of an agent.

## Interview / Explain-Back Questions

- What makes a system an agent rather than a chain or pipeline?
- When should you avoid using an agent?
- How do you prevent an agent from looping forever?
- What is the difference between short-term and long-term agent memory?
- How do you secure tool access in an agent system?
- How would you evaluate an agent that uses external tools?
- What is prompt injection and how do you defend against it?
- When does multi-agent coordination improve results vs. increase complexity?
- How do you handle an agent that hallucinates tool names?
- What role does reflection play in agent quality?

## Exit Criteria

Move to Phase 15 only when you can independently build a complete, secure, observable agent system that plans, executes, reflects, remembers, and fails gracefully — and explain every design decision you made.
