# Phase 13 — LangGraph / Stateful Workflows

> **Goal:** Master LangGraph — implement a simple state-machine agent manually first, then use the framework to build stateful, looped, human-approvable workflows.

**Difficulty:** 🟡 Intermediate → 🟠 Advanced  
**Priority:** Essential  
**Prerequisites:** Phase 12 (LLM application patterns), basic agent-loop intuition  
**Mastery target:** Level 5 — independent LangGraph design with state, routing, memory, approval, and failure handling

---

## Why This Phase Exists

Linear chains work until they don't. Real agent workflows branch, loop, require human approval, recover from failures, and must remember context across steps. LangGraph gives you explicit state-machine control over these concerns. Without it, you're stuck writing ad-hoc if/else logic that becomes unmaintainable.

### Phase Mental Model

A graph workflow is a state machine:

```text
State → Node does work → Edge decides next node → State updates → Stop or loop
```

This solves the weakness of linear chains when workflows need loops, conditional routing, human approval, retries, or resumability.

### What This Phase Prepares For

- Phase 14 (Agents) — LangGraph is the primary framework for production agents
- Phase 15 (Evaluation) — tracing and observability require structured workflows
- Production systems — durable state, checkpointing, and human oversight
- Complex RAG pipelines — retrieval loops, reranking, and self-correction

---

## Units

---

### Unit 13.1 — Manual State-Machine Agent

**What is it?**  
Building a state-machine agent from scratch using plain Python dictionaries and functions — no framework.

**Why does it matter?**  
Understanding the raw mechanics before adopting LangGraph prevents magical thinking. You need to see what LangGraph automates to use it wisely.

**Why learn it here?**  
Phase 12 introduced agent loops. This unit forces you to build the loop yourself so the abstraction in later units feels earned, not mysterious.

**Prerequisites:** Phase 12 agent-loop concepts, basic Python functions and dictionaries.

**Mental Model:**

```text
A state machine is a vending machine:
  - It has states (idle, processing, waiting, done)
  - Each state knows which transitions are valid
  - Input triggers a transition
  - The machine never skips a state or guesses
```

**Core Concepts:**

- State as a dictionary with typed fields
- Nodes as pure functions that receive state and return new state
- Edges as transition logic (which node runs next)
- Terminal states (states that stop the machine)
- The main loop that drives transitions

**How It Works:**

1. Define a state dict with all fields the workflow needs.
2. Write node functions: `def node_fn(state) -> state`.
3. Write a transition function: `def router(state) -> str` that returns the next node name.
4. Run a loop: call router, call the returned node, update state, repeat until terminal.

**Syntax & Implementation:**

```python
from typing import Literal

# --- State ---
state = {
    "query": "What is LangGraph?",
    "scratchpad": [],
    "answer": None,
    "step": 0,
    "max_steps": 3,
}

# --- Nodes ---
def retrieve(state: dict) -> dict:
    docs = ["LangGraph is a stateful orchestration framework.",
            "It models workflows as graphs with nodes and edges."]
    state["scratchpad"].append(f"Retrieved {len(docs)} docs")
    state["step"] += 1
    return state

def generate(state: dict) -> dict:
    state["answer"] = "LangGraph lets you build stateful agent workflows as graphs."
    state["scratchpad"].append("Generated answer")
    state["step"] += 1
    return state

# --- Router ---
def router(state: dict) -> Literal["retrieve", "generate", "__end__"]:
    if state["answer"] is not None:
        return "__end__"
    if state["step"] >= state["max_steps"]:
        return "__end__"
    if not state["scratchpad"]:
        return "retrieve"
    return "generate"

# --- Main loop ---
NODES = {"retrieve": retrieve, "generate": generate}

while True:
    next_node = router(state)
    if next_node == "__end__":
        break
    state = NODES[next_node](state)

print(state)
```

**Simple Example:**

A question router that classifies a query as "factual" or "opinion" and routes to different handlers:

```python
def classify(state):
    q = state["query"].lower()
    state["category"] = "factual" if any(w in q for w in ["what", "when", "where", "who"]) else "opinion"
    return state

def handle_factual(state):
    state["answer"] = f"Factual answer about: {state['query']}"
    return state

def handle_opinion(state):
    state["answer"] = f"Opinion about: {state['query']}"
    return state

def router(state):
    if state.get("answer"):
        return "__end__"
    if state.get("category") == "factual":
        return "factual"
    return "opinion"

state = {"query": "What is LangGraph?", "answer": None}
nodes = {"classify": classify, "factual": handle_factual, "opinion": handle_opinion}

while True:
    choice = router(state)
    if choice == "__end__":
        break
    state = nodes[choice](state)

print(state["answer"])
```

**Real-World Example:**

A customer support triage agent: classify incoming tickets, route to the right department, check if escalation is needed, and generate a response or escalate.

**Common Mistakes:**

- Mutating state in-place without returning a new copy (causes hard-to-trace bugs)
- Forgetting a terminal condition (infinite loop)
- Coupling node logic to the router (nodes should be independent)
- Storing too much in state (keep it minimal and typed)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Infinite loop | Router never returns `__end__` | Print router return value each iteration | Add step counter and terminal conditions |
| State is empty after run | Node returns `None` or wrong dict | Print state after each node call | Always `return state` from every node |
| Wrong node runs | Router logic has wrong priority | Log the router decision path | Add debug prints to router, test each branch |
| State fields missing | Typo in key name | `print(state.keys())` | Define state as a TypedDict or dataclass |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Plain Python state machine | Learning, tiny workflows, full control | Complex state, many branches, need persistence |
| LangGraph | Loops, conditional routing, persistence, human-in-the-loop | Simple linear chains |
| Function-calling chain | Linear tool use, no branching needed | Workflows with loops or approval gates |

**Best Practices:**

- Define all state fields upfront with defaults
- Make nodes pure: receive state, return state, no side effects
- Log state transitions for debugging
- Cap iteration count to prevent runaway loops
- Test each node independently before wiring the graph

**Hands-On Practice:**

1. **Basic:** Build the question classifier above. Add a third category.
2. **Guided:** Add a `validate` node that checks the answer is non-empty before terminating.
3. **Independent:** Build a 3-step pipeline: fetch → process → summarize. Add a retry node that re-fetches on failure.
4. **Realistic:** Build a multi-step research agent: query → search → evaluate relevance → refine query or generate answer. Handle the case where no relevant docs are found.
5. **Challenge:** Add state serialization (save/load state to JSON). Prove the workflow can resume from a checkpoint.

**Exit Criteria:**

- You can build a state-machine agent from scratch in plain Python.
- You understand nodes, edges, state, and the main loop.
- You can identify and fix infinite loops and missing-state bugs.

**Next Step:** Replace your manual state and loop with LangGraph's abstraction.

---

### Unit 13.2 — LangGraph State & Nodes

**What is it?**  
LangGraph's `StateGraph` provides a typed state container and node decorator that replaces manual loop logic with a declarative graph definition.

**Why does it matter?**  
Manual state machines work but become verbose. LangGraph gives you state validation, checkpointing, and visualization for free.

**Why learn it here?**  
After building manually in 13.1, you now appreciate what LangGraph automates. You'll adopt it deliberately, not blindly.

**Prerequisites:** Unit 13.1 (manual state machine), Python classes/dataclasses, `pip install langgraph`.

**Mental Model:**

```text
LangGraph StateGraph is a blueprint for a vending machine:
  - You declare all slots (state fields) upfront
  - You install mechanisms (nodes) that do one thing each
  - You wire routes (edges) between mechanisms
  - The framework handles the loop, persistence, and serialization
```

**Core Concepts:**

- `TypedDict` or Pydantic model for state schema
- `StateGraph` as the graph container
- `add_node(name, fn)` to register work functions
- `add_edge(src, dst)` to wire linear transitions
- `compile()` to produce a runnable graph
- `graph.invoke(state)` to execute

**How It Works:**

1. Define a state schema (TypedDict).
2. Create `StateGraph(StateSchema)`.
3. Add nodes with `graph.add_node("name", function)`.
4. Add edges with `graph.add_edge("node_a", "node_b")`.
5. Set entry point with `graph.set_entry_point("first_node")`.
6. Set finish with `graph.set_finish_point("last_node")`.
7. Compile and invoke.

**Syntax & Implementation:**

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    category: str | None
    answer: str | None
    step: int

def classify(state: AgentState) -> AgentState:
    q = state["query"].lower()
    state["category"] = "factual" if "what" in q or "who" in q else "general"
    state["step"] = state.get("step", 0) + 1
    return state

def answer_query(state: AgentState) -> AgentState:
    state["answer"] = f"Answer for {state['category']}: {state['query']}"
    state["step"] += 1
    return state

graph = StateGraph(AgentState)
graph.add_node("classify", classify)
graph.add_node("answer", answer_query)
graph.add_edge("classify", "answer")
graph.set_entry_point("classify")
graph.set_finish_point("answer")

app = graph.compile()
result = app.invoke({"query": "What is LangGraph?", "category": None, "answer": None, "step": 0})
print(result)
```

**Simple Example:**

A two-node workflow: clean text, then classify sentiment.

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class TextState(TypedDict):
    raw_text: str
    cleaned: str
    sentiment: str | None

def clean(state: TextState) -> TextState:
    state["cleaned"] = state["raw_text"].strip().lower()
    return state

def classify(state: TextState) -> TextState:
    state["sentiment"] = "positive" if "great" in state["cleaned"] else "negative"
    return state

g = StateGraph(TextState)
g.add_node("clean", clean)
g.add_node("classify", classify)
g.add_edge("clean", "classify")
g.set_entry_point("clean")
g.set_finish_point("classify")

app = g.compile()
print(app.invoke({"raw_text": "  This is Great!  ", "cleaned": "", "sentiment": None}))
```

**Real-World Example:**

A RAG pipeline: embed query → retrieve documents → rerank → generate answer. Each node does one thing, and the graph wires them linearly.

**Common Mistakes:**

- Using a plain dict instead of TypedDict (loses type safety)
- Forgetting to `compile()` before invoking
- Not returning the state from node functions
- Mutating shared objects without copying

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| `KeyError` on state | TypedDict has wrong field names | Compare state dict keys to TypedDict definition | Match field names exactly |
| Graph won't compile | Entry/finish point not set | Check `set_entry_point` and `set_finish_point` | Set both before compiling |
| Node output ignored | Function returns `None` | Add `return state` to every node | Every node must return the full state dict |
| `AttributeError` on invoke | Graph not compiled | Check `graph.compile()` exists | Always compile before invoke |

**Best Practices:**

- Use TypedDict for all state — never plain dicts in production
- Keep node functions focused: one responsibility each
- Name nodes descriptively ("embed_query", not "step1")
- Use `graph.get_graph().draw_mermaid()` to visualize
- Test node functions independently before adding to graph

**Hands-On Practice:**

1. **Basic:** Recreate the sentiment classifier above. Change the state schema to add a confidence field.
2. **Guided:** Add a third node that formats the output. Wire: clean → classify → format.
3. **Independent:** Build a text summarization pipeline: split → summarize chunks → merge summaries.
4. **Realistic:** Build a document QA pipeline with: load → chunk → embed → retrieve → answer. Use mock functions for embedding/retrieval.
5. **Challenge:** Visualize the graph with `draw_mermaid()`. Screenshot and annotate the flow.

**Exit Criteria:**

- You can define a TypedDict state schema and build a LangGraph workflow.
- You can add nodes, wire edges, set entry/finish points, compile, and invoke.
- You can debug state-related errors using TypedDict validation.

**Next Step:** Add conditional routing so the graph can branch based on state.

---

### Unit 13.3 — Edges & Conditional Routing

**What is it?**  
Conditional edges let the graph make decisions at runtime: based on state, choose which node to visit next.

**Why does it matter?**  
Linear graphs can't handle branching logic. Conditional edges are how LangGraph implements if/else, switch, and dynamic routing.

**Why learn it here?**  
Units 13.1–13.2 built linear graphs. Real workflows branch. This unit adds the control flow that makes graphs useful beyond toy examples.

**Prerequisites:** Units 13.1–13.2 (manual state machine, LangGraph basics).

**Mental Model:**

```text
Conditional edge is a fork in the road:
  - You reach an intersection (a node)
  - You look at a map (state) to decide which road to take
  - Each road leads to a different destination (node)
```

**Core Concepts:**

- `add_conditional_edges(source, router_fn, mapping)` — dynamic routing
- `router_fn(state) -> str` — returns a routing key
- `mapping: dict[str, str]` — maps routing keys to node names
- `START` and `END` constants for entry/exit routing
- Branching and merging paths

**How It Works:**

1. Write a router function that inspects state and returns a string key.
2. Define a mapping from keys to node names.
3. Call `add_conditional_edges(source_node, router_fn, mapping)`.
4. The graph executes the router after the source node, then follows the matching edge.

**Syntax & Implementation:**

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END

class RouterState(TypedDict):
    query: str
    category: str | None
    answer: str | None

def classify(state: RouterState) -> RouterState:
    q = state["query"].lower()
    if any(w in q for w in ["what", "how", "why"]):
        state["category"] = "question"
    elif "debug" in q or "error" in q:
        state["category"] = "debug"
    else:
        state["category"] = "general"
    return state

def answer_question(state: RouterState) -> RouterState:
    state["answer"] = f"Answering: {state['query']}"
    return state

def debug_issue(state: RouterState) -> RouterState:
    state["answer"] = f"Debugging: {state['query']}"
    return state

def handle_general(state: RouterState) -> RouterState:
    state["answer"] = f"General response to: {state['query']}"
    return state

def route_after_classify(state: RouterState) -> str:
    return state["category"]  # "question", "debug", or "general"

graph = StateGraph(RouterState)
graph.add_node("classify", classify)
graph.add_node("answer_question", answer_question)
graph.add_node("debug_issue", debug_issue)
graph.add_node("handle_general", handle_general)

graph.add_conditional_edges(
    "classify",
    route_after_classify,
    {"question": "answer_question", "debug": "debug_issue", "general": "handle_general"}
)
graph.add_edge("answer_question", END)
graph.add_edge("debug_issue", END)
graph.add_edge("handle_general", END)
graph.set_entry_point("classify")

app = graph.compile()
print(app.invoke({"query": "How do I debug LangGraph?", "category": None, "answer": None}))
```

**Simple Example:**

A content router: classify input as "code", "text", or "data" and route to the appropriate handler.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class ContentState(TypedDict):
    content: str
    content_type: str | None
    result: str | None

def detect_type(state: ContentState) -> ContentState:
    c = state["content"]
    if "def " in c or "import " in c:
        state["content_type"] = "code"
    elif "," in c and "\n" in c:
        state["content_type"] = "data"
    else:
        state["content_type"] = "text"
    return state

def process_code(state: ContentState) -> ContentState:
    state["result"] = f"Code review: {len(state['content'].splitlines())} lines"
    return state

def process_text(state: ContentState) -> ContentState:
    state["result"] = f"Text analysis: {len(state['content'].split())} words"
    return state

def process_data(state: ContentState) -> ContentState:
    rows = state["content"].strip().split("\n")
    state["result"] = f"Data: {len(rows)} rows"
    return state

def router(state: ContentState) -> str:
    return state["content_type"]

g = StateGraph(ContentState)
g.add_node("detect", detect_type)
g.add_node("code", process_code)
g.add_node("text", process_text)
g.add_node("data", process_data)
g.add_conditional_edges("detect", router, {"code": "code", "text": "text", "data": "data"})
g.add_edge("code", END)
g.add_edge("text", END)
g.add_edge("data", END)
g.set_entry_point("detect")

app = g.compile()
print(app.invoke({"content": "import os\nprint(os.getcwd())", "content_type": None, "result": None}))
```

**Real-World Example:**

A customer support agent: classify ticket → route to FAQ, billing, or technical support → each handler resolves differently → all paths converge to a summary node.

**Common Mistakes:**

- Router returns a key not in the mapping (runtime error)
- Forgetting to wire edges from branch targets back to END (graph hangs)
- Mutating state in the router (router should be side-effect-free)
- Using conditional edges where a simple edge would work (over-engineering)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| `InvalidStateError` or routing error | Router returns unmapped key | Print router return value | Ensure all router outputs appear in the mapping dict |
| Graph hangs after branch | No edge from branch target to END or next node | Draw graph and check all paths terminate | Add `add_edge(node, END)` for every branch |
| Wrong branch taken | Router logic has wrong conditions | Test router function with each state variant | Fix router conditions, add unit tests for router |
| State not passed to branch | Node doesn't return state | Check node return values | Every node must `return state` |

**Best Practices:**

- Keep router functions pure: no side effects, no state mutation
- Document each branch's expected input conditions
- Test router functions in isolation before wiring
- Use descriptive mapping keys that match the domain
- Always draw/inspect the graph to verify all paths terminate

**Hands-On Practice:**

1. **Basic:** Build the content router above. Add a fourth content type "image" (detect by file extension).
2. **Guided:** Build a classifier that routes "urgent" vs "normal" requests to different response nodes.
3. **Independent:** Build a multi-step classifier: detect type → detect complexity → route to simple or complex handler.
4. **Realistic:** Build a help-desk router: classify ticket → route to department → each department resolves → merge into summary. Ensure all paths reach the summary.
5. **Challenge:** Add a fallback branch that triggers when the classifier confidence is low. The fallback asks for clarification.

**Exit Criteria:**

- You can implement conditional routing with `add_conditional_edges`.
- You can write router functions that inspect state and return routing keys.
- You can verify all graph paths terminate at END.

**Next Step:** Add loops and cycles so nodes can repeat until a condition is met.

---

### Unit 13.4 — Loops & Cycles

**What is it?**  
Loops let a graph revisit nodes until a condition is satisfied — essential for retries, iterative refinement, and convergence.

**Why does it matter?**  
Many real workflows aren't one-pass: agents refine answers, pipelines retry on failure, and search loops continue until relevance is met. Without loops, you'd need to pre-compute an upper bound on iterations.

**Why learn it here?**  
Conditional routing (13.3) showed branching. Loops add the ability to go backward in the graph — a critical step before memory and approval.

**Prerequisites:** Units 13.1–13.3 (state, nodes, conditional edges).

**Mental Model:**

```text
A loop is a feedback cycle:
  Do work → Check result → Not good enough? → Go back and redo → Repeat until satisfied
```

**Core Concepts:**

- Back-edges: wiring a node back to an earlier node
- Iteration counters in state to prevent infinite loops
- Convergence conditions: loop until quality threshold or max steps
- The `while` pattern in manual implementations vs graph back-edges in LangGraph

**How It Works:**

1. Add a `step` or `iteration` counter to state.
2. After a processing node, add a conditional edge that checks the counter or quality.
3. If not converged, route back to the processing node (or a refinement node).
4. If converged or max steps reached, route to END or the next phase.

**Syntax & Implementation:**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RefineState(TypedDict):
    query: str
    draft: str | None
    quality: float
    max_steps: int
    step: int

def generate_draft(state: RefineState) -> RefineState:
    state["step"] += 1
    if state["draft"] is None:
        state["draft"] = f"Initial draft for: {state['query']}"
    else:
        state["draft"] = f"Refined draft v{state['step']} for: {state['query']}"
    state["quality"] = min(0.3 + 0.25 * state["step"], 1.0)
    return state

def should_continue(state: RefineState) -> str:
    if state["quality"] >= 0.8:
        return "done"
    if state["step"] >= state["max_steps"]:
        return "done"
    return "refine"

def finalize(state: RefineState) -> RefineState:
    state["draft"] = f"FINAL: {state['draft']} (quality={state['quality']:.2f})"
    return state

g = StateGraph(RefineState)
g.add_node("generate", generate_draft)
g.add_node("finalize", finalize)
g.set_entry_point("generate")
g.add_conditional_edges("generate", should_continue, {"refine": "generate", "done": "finalize"})
g.add_edge("finalize", END)

app = g.compile()
result = app.invoke({"query": "Explain state machines", "draft": None, "quality": 0.0, "max_steps": 5, "step": 0})
print(result["draft"], f"| quality: {result['quality']}")
```

**Simple Example:**

A number-guessing loop: generate a guess, check if it's close enough, refine until correct.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class GuessState(TypedDict):
    target: int
    guess: int | None
    attempts: int
    max_attempts: int

def make_guess(state: GuessState) -> GuessState:
    state["attempts"] += 1
    if state["guess"] is None:
        state["guess"] = 50
    elif state["guess"] < state["target"]:
        state["guess"] = state["guess"] + 10
    else:
        state["guess"] = state["guess"] - 5
    return state

def check_guess(state: GuessState) -> str:
    if abs(state["guess"] - state["target"]) <= 2:
        return "done"
    if state["attempts"] >= state["max_attempts"]:
        return "done"
    return "retry"

g = StateGraph(GuessState)
g.add_node("guess", make_guess)
g.add_conditional_edges("guess", check_guess, {"retry": "guess", "done": END})
g.set_entry_point("guess")

app = g.compile()
print(app.invoke({"target": 73, "guess": None, "attempts": 0, "max_attempts": 10}))
```

**Real-World Example:**

A RAG self-correction loop: retrieve documents → generate answer → check faithfulness → if not faithful, reformulate query and re-retrieve → repeat up to N times.

**Common Mistakes:**

- No max iteration guard → infinite loop
- State not copied between iterations → stale references
- Loop condition too loose → runs forever; too strict → never converges
- Not logging iteration count → impossible to debug convergence

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Infinite loop | No max iteration or convergence check | Print iteration count each pass | Add `max_steps` to state and check in router |
| Never converges | Quality threshold unreachable | Log quality value each iteration | Adjust threshold or fix quality metric |
| State grows unbounded | Appending to list without clearing | Check list lengths each iteration | Clear or cap accumulators, use step counter |
| Loop exits too early | Convergence check triggers on first pass | Test with known non-converging input | Ensure initial state doesn't satisfy the exit condition |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| Graph back-edge loop | Need state checkpointing and observability | Simple `for` loop suffices |
| Plain `while` loop | Quick scripts, no persistence needed | Need durable state or human intervention |
| Recursion | Tree-structured problems | Deep recursion risks stack overflow |

**Best Practices:**

- Always set a max iteration count
- Log iteration number and key metrics each pass
- Test with both converging and non-converging inputs
- Use graph loops when you need checkpointing or observability
- Keep loop body (node) functions small and testable

**Hands-On Practice:**

1. **Basic:** Build the guesser above. Change the target and max_attempts. Verify it converges.
2. **Guided:** Build a retry loop: call a mock API that fails 60% of the time. Retry until success or max retries.
3. **Independent:** Build a refinement loop: generate a summary, check length, regenerate if too long/short.
4. **Realistic:** Build a self-correcting QA pipeline: answer → check faithfulness → reformulate query → re-answer. Use mock functions.
5. **Challenge:** Add metrics logging to each iteration. Plot convergence over time.

**Exit Criteria:**

- You can implement loops in LangGraph with back-edges and convergence conditions.
- You can prevent infinite loops with iteration guards.
- You can debug convergence issues by inspecting state at each step.

**Next Step:** Add memory so the graph retains context across iterations and invocations.

---

### Unit 13.5 — Memory

**What is it?**  
Memory in LangGraph means persisting state across invocations and accumulating context within a workflow.

**Why does it matter?**  
Agents need to remember past interactions, accumulated results, and intermediate findings. Without memory, every invocation starts from scratch.

**Why learn it here?**  
Loops (13.4) showed iteration within one run. Memory extends this across runs and gives agents context that grows over time.

**Prerequisites:** Units 13.1–13.4 (state, nodes, conditional edges, loops).

**Mental Model:**

```text
Memory is a notebook that persists:
  - Within-run memory: notes accumulated during one workflow execution
  - Cross-run memory: saved state that survives between invocations
  - The graph reads and writes to this notebook at each step
```

**Core Concepts:**

- In-memory state accumulation (lists, counters within one run)
- LangGraph checkpointers for cross-invocation persistence
- `MemorySaver` for development/testing
- `SqliteSaver` or `PostgresSaver` for production
- Thread IDs for isolating conversations
- State snapshots and replay

**How It Works:**

1. Add memory fields to your state (e.g., `history: list`, `context: str`).
2. Nodes append to these fields as they execute.
3. For persistence, add a checkpointer: `memory = MemorySaver(); app = graph.compile(checkpointer=memory)`.
4. Invoke with a thread ID: `app.invoke(state, config={"configurable": {"thread_id": "user-123"}})`.
5. Subsequent invocations with the same thread ID resume from the last checkpoint.

**Syntax & Implementation:**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ChatState(TypedDict):
    messages: list[str]
    response: str | None

def chatbot(state: ChatState) -> ChatState:
    user_msg = state["messages"][-1]
    state["response"] = f"Echo: {user_msg}"
    state["messages"] = state["messages"] + [state["response"]]
    return state

g = StateGraph(ChatState)
g.add_node("chat", chatbot)
g.add_edge(START, "chat")
g.add_edge("chat", END)

memory = MemorySaver()
app = g.compile(checkpointer=memory)

# First invocation
config = {"configurable": {"thread_id": "user-1"}}
app.invoke({"messages": ["Hello!"], "response": None}, config)

# Second invocation — remembers history
result = app.invoke({"messages": ["How are you?"], "response": None}, config)
print(result["messages"])  # ['Hello!', 'Echo: Hello!', 'How are you?', 'Echo: How are you?']
```

**Simple Example:**

A running total that persists across invocations:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class CounterState(TypedDict):
    total: int
    last_addition: int | None

def add_number(state: CounterState) -> CounterState:
    state["total"] = state.get("total", 0) + (state.get("last_addition", 0) or 0)
    return state

g = StateGraph(CounterState)
g.add_node("add", add_number)
g.add_edge(START, "add")
g.add_edge("add", END)

memory = MemorySaver()
app = g.compile(checkpointer=memory)
cfg = {"configurable": {"thread_id": "counter-1"}}

app.invoke({"total": 0, "last_addition": 5}, cfg)
result = app.invoke({"total": 0, "last_addition": 3}, cfg)
print(result["total"])  # 8 — persisted across invocations
```

**Real-World Example:**

A conversational agent that maintains chat history across messages. Each user message is appended to history, and the agent uses prior context to generate responses.

**Common Mistakes:**

- Using the same thread ID for different conversations (state leaks)
- Forgetting to install a checkpointer (state doesn't persist)
- Appending to state without bounds (memory grows forever)
- Not handling state schema evolution across versions

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| State resets between invocations | No checkpointer installed | Check `graph.compile(checkpointer=...)` | Add MemorySaver or production checkpointer |
| Wrong conversation history | Thread ID collision | Check thread_id values | Use unique thread IDs per user/conversation |
| Memory grows without bound | No history cap | Check list length | Add max history length, trim old entries |
| `KeyError` on resume | State schema changed between runs | Compare current schema to saved state | Migrate state or use `.get()` with defaults |

**Best Practices:**

- Use thread IDs to isolate conversations or sessions
- Cap history lists to prevent unbounded growth
- Use `MemorySaver` for development, persistent storage for production
- Test persistence by invoking twice with the same thread ID
- Log thread IDs alongside state for debugging

**Hands-On Practice:**

1. **Basic:** Build the chatbot above. Add a third invocation and verify history grows correctly.
2. **Guided:** Build a counter that adds different numbers across 5 invocations with the same thread.
3. **Independent:** Build a research agent that accumulates findings across iterations. Each iteration adds to a `findings` list.
4. **Realistic:** Build a multi-turn assistant that remembers prior questions. Use different thread IDs for two "users" and verify isolation.
5. **Challenge:** Implement history trimming: keep only the last N messages. Test that older messages are dropped.

**Exit Criteria:**

- You can use LangGraph checkpointers to persist state across invocations.
- You can isolate conversations with thread IDs.
- You can manage memory bounds to prevent unbounded growth.

**Next Step:** Add human approval nodes so agents can pause for review.

---

### Unit 13.6 — Human Approval

**What is it?**  
Human-in-the-loop (HITL) pauses graph execution at a checkpoint, waits for a human to review and approve/reject, then continues.

**Why does it matter?**  
Agents making autonomous decisions can be dangerous. Approval gates ensure humans stay in control for sensitive actions (sending emails, executing trades, modifying data).

**Why learn it here?**  
Memory (13.5) showed persistence. Approval adds intentional interruption — the graph can pause and resume after human input.

**Prerequisites:** Units 13.1–13.5 (state, nodes, edges, loops, memory).

**Mental Model:**

```text
Human approval is a toll booth:
  - The agent drives toward an action
  - It reaches the toll booth (approval node)
  - It stops and shows what it plans to do
  - A human waves it through or redirects it
  - The agent continues based on the human's decision
```

**Core Concepts:**

- `interrupt_before` or `interrupt_after` on a node
- Reviewing the state at the interrupt point
- Resuming with `app.invoke(None, config)` after human input
- Modifying state before resume (approving, rejecting, editing)
- Approval vs rejection vs modification patterns

**How It Works:**

1. Add an approval node to your graph.
2. Compile with `interrupt_before=["approval_node"]` or `interrupt_after`.
3. When the graph reaches the approval node, it pauses and saves a checkpoint.
4. The human inspects state, optionally modifies it, then resumes execution.
5. The approval node reads the human's decision and routes accordingly.

**Syntax & Implementation:**

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class ApprovalState(TypedDict):
    action: str
    approved: bool | None
    result: str | None

def propose_action(state: ApprovalState) -> ApprovalState:
    state["action"] = "Send email to client"
    state["approved"] = None
    return state

def execute_action(state: ApprovalState) -> ApprovalState:
    if state.get("approved"):
        state["result"] = f"Executed: {state['action']}"
    else:
        state["result"] = "Action rejected by human"
    return state

def reject_action(state: ApprovalState) -> ApprovalState:
    state["result"] = "Action cancelled"
    return state

def route_after_approval(state: ApprovalState) -> str:
    if state.get("approved") is True:
        return "execute"
    return "reject"

g = StateGraph(ApprovalState)
g.add_node("propose", propose_action)
g.add_node("approval", lambda s: s)  # passthrough — human reviews here
g.add_node("execute", execute_action)
g.add_node("reject", reject_action)
g.add_edge("propose", "approval")
g.add_conditional_edges("approval", route_after_approval, {"execute": "execute", "reject": "reject"})
g.add_edge("execute", END)
g.add_edge("reject", END)
g.set_entry_point("propose")

memory = MemorySaver()
app = graph.compile(checkpointer=memory, interrupt_before=["approval"])

config = {"configurable": {"thread_id": "approval-1"}}

# Run until interrupt
app.invoke({"action": "", "approved": None, "result": None}, config)

# Human reviews and approves
app.invoke({"approved": True}, config)
```

**Simple Example:**

An agent that proposes a response and pauses for human review before sending:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class DraftState(TypedDict):
    user_message: str
    draft: str | None
    approved: bool | None

def draft_response(state: DraftState) -> DraftState:
    state["draft"] = f"Response to: {state['user_message']}"
    return state

def review_passthrough(state: DraftState) -> DraftState:
    return state

def send_response(state: DraftState) -> DraftState:
    if state.get("approved"):
        return {"draft": state["draft"], "approved": True, "user_message": state["user_message"]}
    return {"draft": None, "approved": False, "user_message": state["user_message"]}

g = StateGraph(DraftState)
g.add_node("draft", draft_response)
g.add_node("review", review_passthrough)
g.add_node("send", send_response)
g.add_edge("draft", "review")
g.add_edge("review", "send")
g.add_edge("send", END)
g.set_entry_point("draft")

memory = MemorySaver()
app = g.compile(checkpointer=memory, interrupt_before=["review"])

config = {"configurable": {"thread_id": "review-1"}}
app.invoke({"user_message": "What's the weather?", "draft": None, "approved": None}, config)

# Human reviews and edits the draft
app.invoke({"draft": "It's sunny today!", "approved": True}, config)
```

**Real-World Example:**

A financial agent that proposes transactions. The approval node shows the transaction details. A compliance officer reviews and approves or flags for revision before execution.

**Common Mistakes:**

- Forgetting to install a checkpointer (interrupts won't work)
- Not handling the "rejected" path (graph hangs)
- Modifying state in a way that breaks the approval node's expectations
- Using interrupt without thread isolation (approval leaks across users)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Graph doesn't pause | No `interrupt_before` or no checkpointer | Check compile args | Add both checkpointer and interrupt config |
| Resume does nothing | Invoking without None first arg | Check resume call signature | Use `app.invoke(None, config)` to resume |
| Wrong state after resume | State overwritten during resume | Log state before and after resume | Only modify fields the human should control |
| Approval not reached | Graph routes around approval node | Draw graph and trace paths | Ensure all paths to sensitive actions go through approval |

**Best Practices:**

- Always provide both approve and reject paths
- Use `interrupt_before` for review of proposed actions
- Log the state snapshot at the interrupt for audit trails
- Test both approval and rejection flows
- Use thread IDs to isolate approval contexts per user

**Hands-On Practice:**

1. **Basic:** Build the approval example above. Test both approve and reject paths.
2. **Guided:** Add a "modify" path where the human edits the draft before approval.
3. **Independent:** Build a multi-step workflow with approval gates at two points: after planning and after drafting.
4. **Realistic:** Build a content publishing pipeline: draft → review → edit → approve → publish. Simulate both approval and rejection.
5. **Challenge:** Add an audit log that records every approval decision with timestamp and reason.

**Exit Criteria:**

- You can implement human-in-the-loop with `interrupt_before` and checkpointer.
- You can handle approve, reject, and modify flows.
- You can isolate approval contexts with thread IDs.

**Next Step:** Add failure handling so the graph recovers from errors gracefully.

---

### Unit 13.7 — Failure Handling

**What is it?**  
Error handling in LangGraph: catching node exceptions, retrying failed steps, routing to fallback paths, and preventing cascading failures.

**Why does it matter?**  
Nodes call APIs, databases, LLMs — all of which fail. Without explicit failure handling, one bad API call crashes the entire workflow.

**Why learn it here?**  
After building state, routing, loops, memory, and approval, you have a complete graph. This unit hardens it against the real world.

**Prerequisites:** Units 13.1–13.6 (full graph construction).

**Mental Model:**

```text
Failure handling is a safety net:
  - Each node is a performer on a high wire
  - The safety net catches them if they fall
  - The net decides: catch and retry, catch and redirect, or catch and stop
```

**Core Concepts:**

- `try/except` within node functions
- Retry logic with exponential backoff
- Fallback nodes and error paths
- State fields for error tracking (`error: str | None`, `retries: int`)
- Routing based on error state
- Failing gracefully to END with error info

**How It Works:**

1. Add error-tracking fields to state (`error`, `retries`).
2. Wrap risky operations in `try/except` inside nodes.
3. On failure, set `state["error"]` and increment `state["retries"]`.
4. Add a conditional edge after risky nodes that checks for errors.
5. Route to retry node (if retries remaining) or fallback/END.

**Syntax & Implementation:**

```python
import time
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class RobustState(TypedDict):
    input_data: str
    result: str | None
    error: str | None
    retries: int
    max_retries: int

def risky_api_call(state: RobustState) -> RobustState:
    import random
    try:
        if random.random() < 0.7:
            raise ConnectionError("API unavailable")
        state["result"] = f"Success: processed {state['input_data']}"
        state["error"] = None
    except Exception as e:
        state["error"] = str(e)
        state["retries"] = state.get("retries", 0) + 1
    return state

def fallback_handler(state: RobustState) -> RobustState:
    state["result"] = f"Fallback: using cached data for {state['input_data']}"
    state["error"] = None
    return state

def route_after_risky(state: RobustState) -> str:
    if state.get("error") is None:
        return "done"
    if state.get("retries", 0) >= state.get("max_retries", 3):
        return "fallback"
    return "retry"

g = StateGraph(RobustState)
g.add_node("call_api", risky_api_call)
g.add_node("fallback", fallback_handler)
g.add_conditional_edges("call_api", route_after_risky, {
    "done": END, "retry": "call_api", "fallback": "fallback"
})
g.add_edge("fallback", END)
g.set_entry_point("call_api")

app = g.compile()
print(app.invoke({"input_data": "user-123", "result": None, "error": None, "retries": 0, "max_retries": 3}))
```

**Simple Example:**

A division-by-zero handler that routes to an error message:

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class MathState(TypedDict):
    a: int
    b: int
    result: str | None

def divide(state: MathState) -> MathState:
    try:
        state["result"] = str(state["a"] / state["b"])
    except ZeroDivisionError:
        state["result"] = "ERROR: division by zero"
    return state

g = StateGraph(MathState)
g.add_node("divide", divide)
g.add_edge(START, "divide")
g.add_edge("divide", END)
app = g.compile()
print(app.invoke({"a": 10, "b": 0, "result": None}))
```

**Real-World Example:**

A multi-source RAG pipeline: try Vector DB first → if timeout, try keyword search → if both fail, return a helpful error message with troubleshooting steps.

**Common Mistakes:**

- Catching `Exception` too broadly (hides bugs)
- Not retrying transient errors (API timeouts, rate limits)
- Retrying permanent errors (bad input, auth failures)
- Not logging errors (impossible to debug in production)
- Error state not reset on retry (error persists in state)

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Graph crashes on API error | No try/except in node | Add exception handling | Wrap risky calls in try/except |
| Retries exhaust immediately | Max retries too low | Check `max_retries` value | Set reasonable retry count |
| Error persists after retry | Error state not cleared on retry | Print state after retry | Reset `error` field at start of retry node |
| Silent failures | Broad except clause | Log exception details | Catch specific exceptions, log them |

**Alternatives:**

| Approach | Use When | Avoid When |
|---|---|---|
| In-node try/except | Simple retry/fallback logic | Complex error orchestration |
| Conditional error routing | Multiple failure modes to handle | Single failure mode |
| External retry middleware | Many nodes need same retry logic | Only one node is risky |
| Dead-letter queue | Failed items need later investigation | Real-time response required |

**Best Practices:**

- Catch specific exceptions, not bare `Exception`
- Log errors with context (input, step, retry count)
- Distinguish transient errors (retry) from permanent errors (fail fast)
- Reset error state at the beginning of retry attempts
- Set max retries to prevent infinite retry loops
- Test with both transient and permanent failure modes

**Hands-On Practice:**

1. **Basic:** Build the division example. Test with b=0 and b=5.
2. **Guided:** Build the API retry example above. Change the failure rate and max_retries. Observe behavior.
3. **Independent:** Build a 3-step pipeline where step 2 can fail. Add retry for step 2 and a fallback that skips step 2.
4. **Realistic:** Build a multi-source retrieval pipeline: try source A → timeout → try source B → empty → try source C → return error if all fail. Log each attempt.
5. **Challenge:** Add exponential backoff: wait 1s, 2s, 4s between retries. Measure total execution time.

**Exit Criteria:**

- You can add try/except and retry logic inside node functions.
- You can route to fallback paths based on error state.
- You can prevent infinite retry loops with max retry limits.
- You can distinguish transient from permanent errors.

**Next Step:** Synthesize everything into a complete LangGraph application.

---

### Unit 13.8 — LangGraph Synthesis & Review

**What is it?**  
A cumulative integration unit that combines state, nodes, edges, conditional routing, loops, memory, human approval, and failure handling into a complete application.

**Why does it matter?**  
Knowing individual features isn't enough. You must build a working system that uses them together and explain your design choices.

**Prerequisites:** Units 13.1–13.7 (all prior units in this phase).

---

## Mini Project: Research Assistant Agent

**Objective:** Build a LangGraph-powered research assistant that takes a question, retrieves information, evaluates quality, refines if needed, and produces a final answer — with human approval and failure handling.

**Problem Statement:** Users need accurate, sourced answers to research questions. A naive RAG pipeline may retrieve irrelevant documents or produce hallucinated answers. This agent self-corrects and asks for human review when confidence is low.

**Requirements:**

- TypedDict state schema with all required fields
- At least 5 nodes: classify, retrieve, evaluate, generate, approve
- Conditional routing based on classification and evaluation scores
- A refinement loop (retrieve → evaluate → regenerate if score < threshold)
- Human approval gate before final output
- Failure handling for retrieval errors with retry logic
- Thread-isolated memory using MemorySaver
- State logging at each step for debugging
- Max iteration guard on the refinement loop

**Concepts Used:**

- State definition (TypedDict)
- Nodes as pure functions
- Conditional edges and routing
- Loops with convergence conditions
- Memory persistence with checkpointer
- Human-in-the-loop with interrupt
- Error handling and retries

**Suggested Architecture:**

```text
START → classify → retrieve → evaluate → [score ≥ 0.7?] → generate → approval → END
                       ↑                    ↓ (score < 0.7)
                       └──── refine_query ─┘
                                        ↓ (retries ≥ 3 or error)
                                     fallback → END
```

**Milestones:**

1. Define state schema with all fields
2. Implement and test each node independently
3. Wire the graph with conditional routing
4. Add the refinement loop with max iteration guard
5. Add error handling to the retrieve node
6. Add human approval before final output
7. Test with MemorySaver and thread isolation
8. Log state at each transition

**Expected Output:**

- Working Python script implementing the full graph
- State trace showing each step's input/output
- Demo with 3 different questions (easy, hard, unanswerable)
- Error handling demo (simulated retrieval failure)
- Approval demo (approve and reject paths)

**Evaluation Criteria:**

- Code runs without errors
- State schema is typed and complete
- All paths terminate at END
- Refinement loop converges or hits max iterations
- Approval gate pauses and resumes correctly
- Errors are caught and handled gracefully
- Memory persists across invocations
- State trace is readable and complete
- README explains architecture and design decisions

**Failure Cases to Test:**

- Retrieval returns empty results
- API timeout during retrieval
- Evaluation score never reaches threshold (loop exhaustion)
- Human rejects the proposed answer
- Invalid input (empty query)
- Same thread ID used for two different questions

**Advanced Extensions:**

- Use a real LLM for generation (OpenAI or Anthropic API)
- Add web search as a retrieval source
- Implement streaming state updates
- Add a feedback loop where the human rates the final answer
- Persist state to SQLite instead of memory

**Deliverables:**

- `research_agent.py` — full implementation
- `state_trace.log` — output showing state at each step
- `README.md` — architecture explanation, setup instructions, design decisions

---

## Knowledge Check

- Why build a state machine manually before using LangGraph?
- What problem do conditional edges solve that simple edges don't?
- How does a loop in LangGraph differ from a Python `for` loop?
- When would you use a checkpointer vs. in-memory state?
- What is the risk of not setting max iterations on a refinement loop?
- How does human approval differ from human-in-the-loop with interrupts?
- Why should node functions be pure (no side effects)?
- What's the difference between a transient and permanent error?

---

## Decision Guidance

| Use Simple Chain When | Use LangGraph When |
|---|---|
| Steps are linear and predictable | The workflow branches, loops, or resumes |
| No human approval is needed | Human approval or checkpoints are needed |
| State is small and temporary | State must be explicit and durable |
| Debugging is easy from logs | You need traceable transitions and control |
| You're prototyping quickly | You're building a production agent |

---

## Phase Review Checklist

- [ ] All 8 units completed
- [ ] Manual state machine built and tested
- [ ] LangGraph StateGraph with TypedDict state
- [ ] Conditional routing implemented and verified
- [ ] Loop with convergence guard working
- [ ] Memory/checkpointer tested across invocations
- [ ] Human approval gate tested (approve + reject)
- [ ] Failure handling with retry and fallback
- [ ] Mini project completed with full graph
- [ ] State trace logged for all transitions
- [ ] All graph paths verified to terminate
- [ ] Cumulative review passed

---

## Mastery Check

Without following a tutorial, you should be able to:

1. Build a state-machine agent from scratch in plain Python
2. Define a TypedDict state and build a LangGraph workflow
3. Implement conditional routing with `add_conditional_edges`
4. Add loops with iteration guards and convergence conditions
5. Use MemorySaver for cross-invocation persistence
6. Implement human-in-the-loop with `interrupt_before`
7. Handle failures with try/except, retry, and fallback
8. Decide when to use LangGraph vs. simpler approaches
9. Debug state-related errors systematically
10. Build a complete multi-node LangGraph application

---

## Interview / Explain-Back Questions

- What is a state machine, and how does LangGraph implement one?
- Explain the difference between a node and an edge in LangGraph.
- Why must node functions return state? What happens if they don't?
- How do conditional edges differ from regular edges?
- When would you use a loop in a LangGraph workflow?
- What is the purpose of a checkpointer in LangGraph?
- How does human-in-the-loop approval work with `interrupt_before`?
- What strategies prevent infinite loops in graph workflows?
- How would you handle a node that calls an unreliable API?
- When should you use LangGraph instead of a simple function chain?
- How do you test a LangGraph workflow?
- What are the trade-offs of using LangGraph over custom state machine code?

---

## Exit Criteria

Move to Phase 14 only when you can independently build a complete LangGraph application that uses state, conditional routing, loops, memory, human approval, and failure handling — and you can explain every design decision.
