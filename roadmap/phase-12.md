# Phase 12 — LangChain / Framework Abstractions

> **Goal:** Master LangChain — build equivalent systems manually first, then use the framework. Understand abstractions over prompts, models, parsers, chains, memory, tools, and RAG.

**Difficulty:** 🟡 Intermediate → 🟠 Advanced  
**Priority:** Essential for LLM application development  
**Prerequisites:** Phase 09 (LLM Foundations), Phase 11 (RAG Fundamentals)  
**Mastery target:** Level 5 — independent decision making for framework vs manual implementation

---

## Why This Phase Exists

Frameworks like LangChain promise speed, but they can also hide control flow, introduce subtle bugs, and make debugging harder. This phase exists because understanding the manual pipeline first means you can diagnose framework failures, choose the right abstraction, and avoid tutorial hell where the learner copies chains without understanding what they do.

The learner needs to build an LLM pipeline from scratch before wrapping it in abstractions. This ensures every LangChain component maps to something the learner already understands — not something they hope works.

### Phase Mental Model

LangChain is a toolbox of abstractions over prompts, models, parsers, retrieval, memory, and tools. It can speed up composition, but it can also hide control flow and make debugging harder.

```text
Manual pipeline (no framework)
    ↓
Understand: prompt → model → parse → tool → result
    ↓
LangChain wraps each step in a reusable abstraction
    ↓
Chains compose steps. Memory adds state. Tools add actions.
    ↓
Decision point: when does the abstraction help vs hurt?
```

### What This Phase Prepares For

- Phase 13: LangGraph / advanced orchestration
- Phase 14: Agents with tool-calling loops
- Phase 15: Evaluation of LLM systems
- Phase 16: Production deployment of LLM apps
- Phase 17: Capstone projects requiring framework choices

---

## Units

### Unit 12.1 — Manual LLM Pipeline (No Framework)

**What is it?**  
Building an LLM-powered application using only raw API calls, string formatting, and Python logic — no LangChain or framework abstractions.

**Why does it matter?**  
If you can build the pipeline manually, you understand every failure point. When a framework breaks, you can fall back to the manual version or diagnose the issue. Most production LLM systems use very little framework code for critical paths.

**Why learn it here?**  
This is the foundation unit for the entire phase. Every subsequent unit maps back to this manual implementation. Without this, the learner will treat LangChain as magic.

**Prerequisites:**  
- Phase 09: LLM API calls, prompt engineering basics  
- Python: functions, dictionaries, string formatting, error handling  
- An OpenAI-compatible API key or local LLM endpoint

**Mental Model:**  
A manual LLM pipeline is a recipe: pick ingredients (data), write instructions (prompt), send to the chef (LLM), parse the dish (response), and serve it (output).

**Core Concepts:**

- API call structure (endpoint, headers, body)
- Prompt construction with string formatting
- System vs user message roles
- Parsing raw LLM responses (JSON, regex, slicing)
- Error handling for API failures
- Temperature and max_tokens basics
- Sequential and branching logic

**How It Works:**

1. Construct a prompt string with user input
2. Send it to the LLM API
3. Receive a raw text response
4. Parse the response into structured data (or handle failure)
5. Feed the result into the next step or return it

**Simple Example:**

```python
import openai
import json

client = openai.OpenAI(api_key="your-key-here")

def extract_sentiment(text: str) -> dict:
    """Manual sentiment extraction — no framework."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Classify sentiment as positive, negative, or neutral. Return JSON: {\"sentiment\": \"...\", \"confidence\": 0.0-1.0}"},
            {"role": "user", "content": text}
        ],
        temperature=0.0,
        response_format={"type": "json_object"}
    )
    raw = response.choices[0].message.content
    return json.loads(raw)

result = extract_sentiment("The product quality is outstanding but shipping was slow.")
print(result)
# {'sentiment': 'neutral', 'confidence': 0.7}
```

**Real-World Example:**

A customer support team builds a ticket classifier. They use raw API calls because: (1) the pipeline is simple — one prompt, one parse, one routing decision; (2) they need full visibility into latency and token costs; (3) they cannot afford framework upgrades breaking a live system.

**Common Mistakes:**

- Not handling API errors (rate limits, timeouts, invalid responses)
- Hardcoding prompts without version control
- Ignoring token count and hitting limits unexpectedly
- Treating LLM output as always valid JSON
- Not testing with edge cases (empty input, adversarial input)

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| API returns error | Invalid key, rate limit, or model name | Check error message and status code | Use correct key, add retry/backoff |
| Response is not valid JSON | Model returned extra text or explanation | Print raw `response.choices[0].message.content` | Improve system prompt, add `"Return only JSON"` |
| Empty or truncated response | `max_tokens` too low | Check token usage in response | Increase `max_tokens` |
| Hallucinated fields | Prompt is ambiguous | Test with multiple inputs | Be explicit about required fields |
| Inconsistent output format | Temperature too high or no schema constraint | Set `temperature=0.0` and use `response_format` | Add JSON schema to system prompt |

**Alternatives:**

| Approach | Use When | Avoid When | Trade-off |
|---|---|---|---|
| Manual (raw API) | Simple pipelines, full control needed | Complex multi-step orchestration | Maximum control, maximum boilerplate |
| LangChain | Quick composition, team familiarity | Debugging-critical paths | Faster development, less visibility |
| LiteLLM | Multi-provider abstraction | Single provider projects | Unified API, extra dependency |

**Best Practices:**

- Always handle API errors with retries and backoff
- Version your prompts in a file, not inline in code
- Log raw responses before parsing for debugging
- Use `temperature=0.0` for deterministic extraction tasks
- Test with adversarial and edge-case inputs
- Track token usage per call for cost monitoring

**Hands-On Practice:**

1. **Basic:** Make a raw API call and print the response. Parse one field from the result.
2. **Guided:** Build a classifier that reads a CSV of reviews and labels each as positive/negative/neutral using raw API calls. Save results to a new CSV.
3. **Independent:** Build a two-step pipeline: (1) extract entities from text, (2) classify each entity's type. No framework allowed.
4. **Realistic:** Handle rate limits with exponential backoff. Log every API call with timestamps, tokens used, and latency. Parse 100 reviews and report accuracy.
5. **Challenge:** Build a manual pipeline that chains 3 LLM calls: summarize → extract keywords → generate a tagline. Compare token cost vs doing it in one prompt.

**Knowledge Check:**

- What are the required fields in an OpenAI chat completion request?
- How do you handle a response that is not valid JSON?
- Why should you set `temperature=0.0` for extraction tasks?
- What happens if you do not handle rate limits?

**Exit Criteria:**

- You can build a multi-step LLM pipeline with raw API calls.
- You can parse and validate LLM output reliably.
- You can handle common API errors gracefully.

**Next Step:** Wrap these manual patterns in LangChain abstractions.

---

### Unit 12.2 — LangChain Models & Prompts

**What is it?**  
LangChain's model wrappers (`ChatOpenAI`) and prompt templates (`ChatPromptTemplate`) that standardize how you format and send messages to LLMs.

**Why does it matter?**  
Manually formatting prompts with f-strings works for simple cases, but production code needs: reusable templates with variables, model configuration in one place, and provider-agnostic interfaces. LangChain provides all three.

**Why learn it here?**  
After building the manual pipeline, you now see what LangChain actually abstracts: prompt formatting and model invocation. This is not magic — it is a dictionary of messages sent to an API call.

**Prerequisites:**  
- Unit 12.1 (manual pipeline)
- Understanding of chat message roles (system, user, assistant)

**Mental Model:**  
A prompt template is a fill-in-the-blank form. A chat model is a wrapper around an API call that returns a message object instead of raw text.

**Core Concepts:**

- `ChatOpenAI` / `ChatAnthropic` / generic `BaseChatModel`
- `ChatPromptTemplate` with `SystemMessagePromptTemplate` and `HumanMessagePromptTemplate`
- Template variables with curly braces
- `format_messages()` to produce a list of message dicts
- `.invoke()` to call the model
- Model parameters: `temperature`, `max_tokens`, `model`

**How It Works:**

1. Create a prompt template with variables
2. Fill in variables with `.invoke({"variable": "value"})`
3. Pass formatted messages to a chat model
4. Receive an `AIMessage` object with `.content`

**Simple Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Model — wraps the API call
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Prompt template — reusable with variables
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role}. Respond in exactly one sentence."),
    ("user", "{input}")
])

# Fill template and call model
messages = prompt.invoke({"role": "teacher", "input": "What is gradient descent?"})
response = llm.invoke(messages)

print(response.content)
# "Gradient descent is an optimization algorithm that iteratively adjusts
#  parameters to minimize a loss function by moving in the direction of
#  steepest descent."
```

**Real-World Example:**

A product team builds a feature request classifier. They define one prompt template stored in a YAML file. The same template is used in a notebook for exploration, in a FastAPI endpoint for production, and in a batch job for weekly analysis. The template variable `{{category_list}}` is filled from a config file, not hardcoded.

**Common Mistakes:**

- Forgetting to use `.invoke()` and trying to call templates directly
- Mixing up message role order (system must come first in most providers)
- Using `{{` instead of `{` for template variables
- Not passing all required template variables
- Ignoring that different providers handle system messages differently

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Template variable error | Missing required variable | Check `.invoke()` args | Pass all template variables |
| Wrong message format | Mixing string and message types | Inspect `format_messages()` output | Use consistent message types |
| API key error | Model wrapper needs its own key | Check env vars or constructor | Pass `api_key` or set `OPENAI_API_KEY` |
| Model not found | Wrong model name | Check provider docs | Use valid model identifier |

**Best Practices:**

- Store prompt templates in YAML/JSON files for version control
- Use `ChatPromptTemplate.from_messages()` for clarity
- Set `temperature` and `model` in the model constructor, not per call
- Test templates with `.format_messages()` before invoking the model
- Log the formatted messages, not just the response

**Hands-On Practice:**

1. **Basic:** Create a prompt template and invoke it with one set of variables.
2. **Guided:** Build a template for three different tasks (classify, summarize, extract). Call each with the same model.
3. **Independent:** Store templates in a YAML file, load them at runtime, and fill variables from user input.
4. **Realistic:** Build a template that takes a document and a list of instructions. Test it with 5 different document-instruction pairs.
5. **Challenge:** Compare the same prompt formatted as (a) a single user message, (b) system + user messages, (c) system + few-shot user messages. Measure response quality differences.

**Knowledge Check:**

- What does `format_messages()` return?
- Why use `ChatPromptTemplate` instead of f-strings?
- How do you pass model parameters like `temperature`?

**Exit Criteria:**

- You can create and use prompt templates with variables.
- You can invoke LangChain chat models and read the response.
- You understand what LangChain abstracts compared to raw API calls.

**Next Step:** Parse LLM outputs into structured data.

---

### Unit 12.3 — Output Parsers

**What is it?**  
LangChain output parsers convert raw LLM text responses into structured Python objects (dicts, lists, Pydantic models).

**Why does it matter?**  
Raw LLM output is a string. Production code needs structured data. Manually parsing JSON with `json.loads()` is fragile. LangChain parsers add schema validation, error handling, and retry logic.

**Why learn it here?**  
After mastering models and prompts, the next bottleneck is getting reliable structured output. Parsers solve this.

**Prerequisites:**  
- Unit 12.2 (models and prompts)
- Python: dictionaries, basic Pydantic or dataclasses
- Understanding of JSON format

**Mental Model:**  
An output parser is a translator that takes the LLM's free-form text and converts it into a typed Python object, with a safety net if the translation fails.

**Core Concepts:**

- `StrOutputParser` — returns raw string
- `JsonOutputParser` — parses JSON from response
- `PydanticOutputParser` — validates against a Pydantic model
- `StructuredOutputParser` — parses from a schema definition
- `output_format_instructions()` — adds parsing instructions to the prompt
- Retry on parse failure

**How It Works:**

1. Define the expected output schema (Pydantic model or JSON schema)
2. Get format instructions from the parser
3. Include instructions in the prompt so the LLM knows the expected format
4. Chain the model output through the parser
5. If parsing fails, retry or raise a clear error

**Simple Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class MovieReview(BaseModel):
    title: str
    rating: float  # 1-10
    summary: str

parser = JsonOutputParser(pydantic_object=MovieReview)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie critic.\n{format_instructions}"),
    ("user", "Review the movie: {movie}")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

chain = prompt | llm | parser
result = chain.invoke({
    "movie": "Inception",
    "format_instructions": parser.get_format_instructions()
})
print(result)
# {'title': 'Inception', 'rating': 9.0, 'summary': 'A mind-bending...'}
```

**Common Mistakes:**

- Not including format instructions in the prompt (the LLM does not know the schema)
- Using `JsonOutputParser` without `response_format={"type": "json_object"}` in the model
- Forgetting that parsers can fail — no error handling
- Using complex Pydantic models that confuse the LLM (keep schemas simple)
- Not testing parsers with malformed LLM output

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Parser returns empty dict | LLM did not output valid JSON | Print raw model response | Add format instructions to prompt |
| Pydantic validation error | LLM output does not match schema | Check which field failed | Simplify schema, add examples |
| Parser works in test but fails in production | Temperature variation or model change | Run multiple times with same input | Lower temperature, add retry logic |
| `Extra content` error | Model returned text before/after JSON | Print full response | Use `response_format` or stricter system prompt |

**Best Practices:**

- Always include format instructions in the prompt
- Use Pydantic models for type safety and validation
- Add retry logic for parse failures in production
- Test parsers with edge cases (empty response, partial JSON, extra text)
- Prefer `response_format={"type": "json_object"}` when using OpenAI models

**Hands-On Practice:**

1. **Basic:** Parse a simple JSON response with `JsonOutputParser`.
2. **Guided:** Define a Pydantic model and parse LLM output into it. Handle one parse failure.
3. **Independent:** Build a pipeline that extracts structured data from 10 unstructured product descriptions.
4. **Realistic:** Add retry logic: if parsing fails, re-prompt the LLM with the raw output and ask it to fix the format.
5. **Challenge:** Compare `JsonOutputParser` vs `PydanticOutputParser` vs manual `json.loads()`. Which handles malformed output best?

**Knowledge Check:**

- Why must format instructions be included in the prompt?
- What is the difference between `JsonOutputParser` and `PydanticOutputParser`?
- How do you handle a parse failure in production?

**Exit Criteria:**

- You can convert LLM text output into typed Python objects.
- You can handle parse failures gracefully.
- You can choose the right parser for a given task.

**Next Step:** Compose multiple steps into a chain.

---

### Unit 12.4 — Chains

**What is it?**  
LangChain chains compose multiple operations (prompt → model → parser → next step) into a single callable pipeline using the pipe (`|`) operator.

**Why does it matter?**  
Real applications need multi-step logic: classify then extract, retrieve then answer, summarize then rewrite. Chains let you wire these steps together without nested callbacks or spaghetti code.

**Why learn it here?**  
After mastering models, prompts, and parsers individually, chains show how to compose them. This is the core LangChain pattern.

**Prerequisites:**  
- Units 12.2–12.3 (models, prompts, parsers)
- Python: function composition, dictionaries

**Mental Model:**  
A chain is a pipeline of steps, like a Unix pipe. Data flows left to right: each step transforms the output and passes it to the next.

**Core Concepts:**

- Pipe operator (`|`) for composing steps
- `RunnableParallel` for concurrent steps
- `RunnableLambda` for custom functions in a chain
- `itemgetter` for extracting fields
- `chain.invoke()` and `chain.batch()`
- `input_schema` and `output_schema` for introspection

**How It Works:**

1. Define individual runnables (prompt, model, parser, custom function)
2. Compose them with `|` into a chain
3. Call `.invoke()` with input data
4. Data flows through each step in order

**Simple Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# Step 1: Summarize
summarize_prompt = ChatPromptTemplate.from_template(
    "Summarize in one sentence: {text}"
)
summarizer = summarize_prompt | llm | StrOutputParser()

# Step 2: Classify the summary
classify_prompt = ChatPromptTemplate.from_template(
    "Classify as positive, negative, or neutral: {summary}"
)
classifier = classify_prompt | llm | StrOutputParser()

# Full chain: summarize then classify
chain = summarizer | (lambda s: {"summary": s}) | classifier

result = chain.invoke({"text": "The new phone has great battery life but a terrible camera."})
print(result)
# "neutral"
```

**Real-World Example:**

A legal tech company builds a contract review pipeline: (1) extract key clauses from a contract, (2) classify each clause by risk level, (3) generate a plain-English summary. Each step is a separate chain composed into one pipeline. The same pipeline works for a Jupyter notebook demo and a production API.

**Common Mistakes:**

- Creating deeply nested chains that are hard to debug
- Not testing each step independently before composing
- Forgetting that chains are lazy — they do nothing until `.invoke()`
- Using `RunnableLambda` for logic that should be a standalone function
- Not inspecting intermediate outputs during development

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Chain returns unexpected type | Wrong parser or missing step | Print output after each step | Add intermediate `.invoke()` calls |
| `RunnableLambda` error | Function signature does not match chain input | Check input/output types | Ensure function accepts and returns dict |
| Chain is slow | Sequential steps that could be parallel | Profile each step | Use `RunnableParallel` for independent steps |
| Intermediate data lost | Wrong key extraction | Inspect `.invoke()` at each step | Use `itemgetter` or lambda to pass correct keys |

**Best Practices:**

- Test each step independently before composing
- Use descriptive names for chain steps
- Log inputs and outputs at each step during development
- Keep chains shallow — prefer 3–5 steps over 10+
- Use `RunnableParallel` for independent steps

**Hands-On Practice:**

1. **Basic:** Build a two-step chain: prompt → model → parser → prompt → model → parser.
2. **Guided:** Build a chain that takes a list of texts, summarizes each, and returns summaries.
3. **Independent:** Build a classify-then-extract chain with a branching path (positive → one extractor, negative → another).
4. **Realistic:** Build a 4-step chain: extract entities → classify each entity → rank by relevance → format as a report.
5. **Challenge:** Build the same pipeline with raw Python functions and with LangChain chains. Compare readability and debugging ease.

**Knowledge Check:**

- What does the `|` operator do in LangChain?
- When should you use `RunnableParallel` instead of sequential steps?
- How do you debug a chain that returns unexpected output?

**Exit Criteria:**

- You can compose multiple steps into a chain.
- You can debug intermediate outputs.
- You can use `RunnableParallel` for concurrent operations.

**Next Step:** Add memory for stateful conversations.

---

### Unit 12.5 — Memory

**What is it?**  
LangChain memory components store conversation history so the LLM can reference previous turns in a multi-turn conversation.

**Why does it matter?**  
LLMs are stateless — each API call is independent. Without memory, the model cannot remember what it said two messages ago. Memory is essential for chatbots, assistants, and any interactive LLM application.

**Why learn it here?**  
After mastering chains (single-turn pipelines), the natural next step is multi-turn conversations. Memory is the bridge from single-shot to interactive.

**Prerequisites:**  
- Unit 12.4 (chains)
- Understanding of chat message history (system, user, assistant roles)

**Mental Model:**  
Memory is a notebook the LLM can read. Each new user message gets appended to the notebook, so the LLM sees the full conversation history — not just the latest question.

**Core Concepts:**

- `ChatMessageHistory` — stores messages in a list
- `ConversationBufferMemory` — full history in context
- `ConversationSummaryMemory` — summarized history (saves tokens)
- `ConversationBufferWindowMemory` — sliding window of last N messages
- `ConversationSummaryBufferMemory` — summarize old, keep recent
- Integrating memory into chains with `RunnableWithMessageHistory`
- Session IDs for multi-user conversations

**How It Works:**

1. Create a message history store (per session)
2. Load history before each LLM call
3. Append new user message and LLM response to history
4. Send full history (or summary) as context to the LLM

**Simple Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("user", "{input}")
])

chain = prompt | llm

# Per-session history store
store = {}

def get_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# Turn 1
config = {"configurable": {"session_id": "user-123"}}
response1 = chain_with_history.invoke({"input": "My name is Alice."}, config)
print(response1.content)
# "Hello, Alice! How can I help you?"

# Turn 2 — model remembers
response2 = chain_with_history.invoke({"input": "What is my name?"}, config)
print(response2.content)
# "Your name is Alice."
```

**Real-World Example:**

A customer support chatbot uses `ConversationSummaryBufferMemory` to keep the last 5 messages in full and summarize older messages. This keeps token costs low while preserving recent context. Each customer gets a unique session ID stored in a Redis-backed message store.

**Common Mistakes:**

- Sending the entire history forever (token cost explodes)
- Not using session IDs (all users share one history)
- Forgetting to append the LLM response to history
- Using memory without `MessagesPlaceholder` in the prompt
- Not handling session expiration in production

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model does not remember previous turns | History not passed to prompt | Check `MessagesPlaceholder` is in prompt | Add `MessagesPlaceholder(variable_name="history")` |
| Token count explodes | Full history without summarization | Check history length | Use `ConversationSummaryBufferMemory` |
| All sessions share history | No session ID or wrong store | Check `get_history` function | Use unique session IDs per user |
| History includes duplicate messages | Message appended twice | Trace the chain | Ensure only one append point |

**Best Practices:**

- Always use session IDs for multi-user systems
- Use summarization for long conversations
- Set a maximum window size for buffer memory
- Test memory with at least 5 turns to verify persistence
- Store message history externally (Redis, database) for production

**Hands-On Practice:**

1. **Basic:** Build a 3-turn conversation with `InMemoryChatMessageHistory`.
2. **Guided:** Build a chatbot that remembers user preferences across 5 turns.
3. **Independent:** Implement `ConversationSummaryBufferMemory` and verify that old messages are summarized.
4. **Realistic:** Build a multi-user chatbot with session IDs. Verify that two sessions do not share history.
5. **Challenge:** Compare token cost of buffer memory vs summary memory for a 20-turn conversation.

**Knowledge Check:**

- Why do LLMs need explicit memory components?
- What is the difference between buffer memory and summary memory?
- How do session IDs prevent cross-user information leakage?

**Exit Criteria:**

- You can build multi-turn conversations with memory.
- You can choose the right memory type for a given use case.
- You can handle session isolation.

**Next Step:** Give LLMs the ability to take actions with tools.

---

### Unit 12.6 — Tools

**What is it?**  
LangChain tools are Python functions decorated with `@tool` that an LLM can call to access external data or perform actions (web search, database queries, calculations, API calls).

**Why does it matter?**  
LLMs cannot access the internet, databases, or real-time data by default. Tools bridge the gap between the LLM's knowledge and the real world. This is the foundation for agents.

**Why learn it here?**  
After mastering chains and memory, tools add the ability to act — not just generate text. This transitions from chat to agency.

**Prerequisites:**  
- Unit 12.4 (chains)
- Unit 12.5 (memory)
- Python: functions, type hints, docstrings

**Mental Model:**  
A tool is a function that the LLM can call. The LLM decides when to call it based on the user's question. The tool executes the function and returns the result to the LLM, which then generates a final answer.

**Core Concepts:**

- `@tool` decorator
- Tool name, description, and argument schema
- `ToolMessage` — the result of a tool call
- `AIMessage.tool_calls` — the LLM's request to call a tool
- `ToolNode` — executes tool calls in a chain
- Bindings: `llm.bind_tools([tool1, tool2])`
- Forced tool calling vs optional tool calling

**How It Works:**

1. Define a tool with `@tool` and a docstring
2. Bind tools to the LLM so it knows what is available
3. The LLM generates a `tool_calls` request in its response
4. Execute the tool and return the result as a `ToolMessage`
5. Feed the result back to the LLM for final answer generation

**Simple Example:**

```python
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm_with_tools = llm.bind_tools([multiply])

# LLM decides to call the tool
response = llm_with_tools.invoke([HumanMessage(content="What is 7 times 6?")])

# Check if tool was called
print(response.tool_calls)
# [{'name': 'multiply', 'args': {'a': 7, 'b': 6}, 'id': '...'}]

# Execute the tool
tool_result = multiply.invoke({"a": 7, "b": 6})
# 42.0

# Feed result back to LLM
final = llm_with_tools.invoke([
    HumanMessage(content="What is 7 times 6?"),
    response,
    ToolMessage(content=str(tool_result), tool_call_id=response.tool_calls[0]["id"])
])
print(final.content)
# "7 times 6 is 42."
```

**Real-World Example:**

A data analyst builds a tool that queries a SQL database. The LLM receives a natural language question, calls the SQL tool with a generated query, receives the query result, and generates a human-readable answer. The tool includes error handling for invalid SQL and timeout protection.

**Common Mistakes:**

- Tool descriptions are vague (the LLM does not know when to use it)
- No error handling inside tools (a failed tool crashes the chain)
- Forgetting to feed tool results back to the LLM
- Making tools that do too many things (keep tools focused)
- Not testing tools independently before binding to an LLM

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| LLM never calls the tool | Tool description is unclear | Test with `llm.bind_tools()` and check response | Improve docstring, add examples in description |
| Tool call fails | Missing required arguments | Check `tool_calls` args | Add argument validation in the tool |
| LLM ignores tool result | Result not fed back properly | Trace message history | Ensure `ToolMessage` with correct `tool_call_id` |
| Tool returns wrong type | Return type mismatch | Check tool signature | Ensure return type is JSON-serializable |

**Best Practices:**

- Write clear, specific tool descriptions — the LLM reads them
- Validate inputs inside tools (the LLM may pass wrong types)
- Always return JSON-serializable results
- Add error handling — return error messages, do not raise exceptions
- Test each tool with edge cases before binding to an LLM
- Keep tools focused on one task (single responsibility)

**Hands-On Practice:**

1. **Basic:** Define a calculator tool and call it manually.
2. **Guided:** Bind the calculator to an LLM and verify it calls the tool correctly.
3. **Independent:** Build 3 tools: calculator, string reverser, and current date getter. Bind all three and test with different questions.
4. **Realistic:** Build a database query tool that takes a natural language question and returns results. Handle invalid queries gracefully.
5. **Challenge:** Build a tool that fetches real-time weather data. Test with ambiguous questions ("Is it warm in Paris?") to verify the LLM calls the tool correctly.

**Knowledge Check:**

- What does the `@tool` decorator do?
- How does the LLM decide when to call a tool?
- What is a `ToolMessage` and why is `tool_call_id` important?
- Why should tools return JSON-serializable data?

**Exit Criteria:**

- You can define and use LangChain tools.
- You can bind tools to an LLM and verify tool calls.
- You can handle tool execution errors.

**Next Step:** Build RAG systems with LangChain.

---

### Unit 12.7 — RAG with LangChain

**What is it?**  
Using LangChain's RAG components (document loaders, text splitters, vector stores, retrievers, and chains) to build a retrieval-augmented generation system.

**Why does it matter?**  
Phase 11 taught RAG concepts manually. LangChain provides ready-made components for document loading, chunking, embedding, retrieval, and chain composition. This unit shows how the manual patterns map to LangChain abstractions.

**Why learn it here?**  
After mastering LangChain's core building blocks (models, prompts, parsers, chains, memory, tools), the learner now applies them to the most common real-world LLM pattern: RAG.

**Prerequisites:**  
- Phase 11 (RAG fundamentals)
- Units 12.1–12.6 (all prior LangChain units)
- Basic understanding of embeddings and vector stores

**Mental Model:**  
LangChain RAG is a factory assembly line: documents go in, get chopped into chunks, embedded into vectors, stored in a vector database, retrieved for each query, injected into a prompt, and fed to an LLM.

**Core Concepts:**

- `DocumentLoader` — loads files, URLs, databases into `Document` objects
- `TextSplitter` — splits documents into chunks
- `Embeddings` — converts text to vectors (OpenAI, HuggingFace)
- `VectorStore` — stores and searches vectors (Chroma, FAISS, Pinecone)
- `Retriever` — searches vector store for relevant chunks
- `RetrievalQA` chain or custom RAG chain
- `create_retrieval_chain()` — high-level RAG builder

**How It Works:**

1. Load documents from a source
2. Split into chunks with overlap
3. Embed chunks and store in a vector store
4. For each query, retrieve top-k relevant chunks
5. Inject chunks into a prompt template
6. Call the LLM with the augmented prompt
7. Return the answer with source attribution

**Simple Example:**

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

# Load
loader = TextLoader("knowledge_base.txt")
docs = loader.load()

# Split
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# Embed and store
vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings())
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG chain
prompt = ChatPromptTemplate.from_template("""
Answer based on the context. If the context does not contain the answer, say so.
Context: {context}
Question: {question}
""")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

answer = rag_chain.invoke("What is the main topic of the document?")
print(answer)
```

**Real-World Example:**

A healthcare company builds a medical Q&A system using LangChain RAG. They load 10,000 medical research papers, chunk them by section (not by paragraph), use a clinical embedding model, and retrieve the top 5 most relevant chunks for each query. The system includes a "sources" field that cites the specific paper and section for each answer.

**Common Mistakes:**

- Chunks too large (irrelevant context) or too small (lost context)
- No overlap between chunks (sentences split mid-thought)
- Retrieving too few or too many chunks
- Not filtering by metadata (e.g., document type, date)
- Using the wrong embedding model for the domain

**Debugging / Troubleshooting:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Answer is irrelevant | Retrieved chunks are irrelevant | Print retrieved docs | Improve chunking, try hybrid search |
| Answer is partial | Chunks do not contain full answer | Check chunk size | Increase `chunk_size` or reduce `k` |
| Answer is hallucinated | Context not relevant or missing | Test with known questions | Add "say I don't know" instruction |
| Retrieval is slow | Vector store too large or wrong index | Benchmark retrieval time | Use approximate nearest neighbor index |
| No answer returned | Empty retrieval results | Check vector store contents | Verify document loading and splitting |

**Best Practices:**

- Chunk by semantic units (paragraphs, sections), not fixed character count
- Use overlap (50–200 characters) to preserve context across chunks
- Include metadata (source, date, section) for filtering
- Use hybrid search (keyword + semantic) when available
- Always test retrieval quality before evaluating answer quality
- Set `temperature=0.0` for factual RAG tasks

**Hands-On Practice:**

1. **Basic:** Load a text file, chunk it, embed it, and retrieve the top 3 chunks for a query.
2. **Guided:** Build a full RAG chain that answers questions from a set of documents.
3. **Independent:** Build RAG over 10+ documents. Add metadata filtering by source.
4. **Realistic:** Compare chunk sizes (256, 512, 1024) and measure answer quality for each.
5. **Challenge:** Build RAG with two different embedding models and compare retrieval quality on 20 test questions.

**Knowledge Check:**

- Why is chunk overlap important?
- How do you evaluate retrieval quality separately from answer quality?
- When would you use metadata filtering?

**Exit Criteria:**

- You can build a full RAG pipeline with LangChain.
- You can choose appropriate chunking and retrieval parameters.
- You can debug retrieval and answer quality issues.

**Next Step:** Compare LangChain with manual implementation.

---

### Unit 12.8 — LangChain vs Manual Implementation

**What is it?**  
A structured comparison of building the same LLM application with raw API calls (manual) and with LangChain, evaluating trade-offs in code clarity, debuggability, dependency cost, and maintainability.

**Why does it matter?**  
The ability to choose between framework and manual implementation is a senior engineering skill. Many projects use LangChain for prototyping but replace critical paths with manual code in production. Understanding both approaches is essential.

**Why learn it here?**  
After building the same system both ways across units 12.1–12.7, the learner is now equipped to make informed decisions about when to use each approach.

**Prerequisites:**  
- All prior units in this phase
- Experience building both manual and LangChain versions of the same pipeline

**Mental Model:**  
LangChain is a power tool. It cuts faster, but it also makes it harder to see exactly what is happening. The manual approach is a hand tool — slower but fully visible.

**Decision Guidance:**

| Use LangChain When... | Prefer Manual When... |
|---|---|
| Prototyping or exploring quickly | The pipeline is simple (1-3 steps) |
| Team already uses the ecosystem | You need minimal dependencies |
| Built-in integrations save real time | Debuggability is critical |
| You can trace and test each abstraction | Framework behavior is harder to reason about |
| The project has standard patterns (RAG, chatbot) | The project has custom control flow |
| You need provider-agnostic code | You use one provider and know its API well |

**Comparison Table:**

| Dimension | Manual (Raw API) | LangChain |
|---|---|---|
| Code volume | More boilerplate | Less boilerplate |
| Debugging | Full visibility | Abstraction hides details |
| Dependencies | `openai` or `httpx` | `langchain` + provider packages |
| Learning curve | API docs only | Framework docs + API docs |
| Flexibility | Unlimited | Constrained by framework design |
| Maintenance | You maintain everything | Framework updates may break code |
| Performance | Direct control | Overhead from abstraction layers |
| Community | Provider-specific | Large LangChain ecosystem |
| Production readiness | Easier to reason about | Requires understanding internals |

**Real-World Example:**

A startup prototypes a customer support bot with LangChain in 2 days. In production, they replace the critical classification step with raw API calls because: (1) LangChain added 200ms latency per call, (2) debugging production failures required reading framework source code, and (3) they needed fine-grained control over retry logic.

**Common Mistakes:**

- Choosing LangChain for everything (over-engineering simple tasks)
- Choosing manual for everything (reinventing wheels for complex patterns)
- Not evaluating both approaches before committing
- Using LangChain without understanding what it abstracts
- Migrating from manual to LangChain mid-project without planning

**Best Practices:**

- Prototype with LangChain, optimize critical paths with manual code
- Keep the LLM call itself simple — do not over-abstract the core
- If you cannot debug a LangChain issue in 30 minutes, switch to manual
- Use LangChain for standard patterns, manual for custom logic
- Document which parts use framework and which are custom

**Hands-On Practice:**

1. **Basic:** Build a sentiment classifier both ways. Count lines of code.
2. **Guided:** Build a RAG system both ways. Compare debugging experience.
3. **Independent:** Build a 3-step pipeline both ways. Measure latency and token cost.
4. **Realistic:** Identify the 20% of a LangChain pipeline that handles 80% of the logic. Replace just that part with manual code.
5. **Challenge:** Write a one-page decision guide for your team on when to use LangChain vs manual code.

**Knowledge Check:**

- When would you use LangChain for prototyping but manual code for production?
- What is the biggest debugging challenge with LangChain?
- How do you evaluate whether LangChain saves real time for a given project?

**Exit Criteria:**

- You can build the same system both ways.
- You can articulate trade-offs clearly.
- You can make informed framework decisions.

**Next Step:** Synthesize everything in a mini project.

---

### Unit 12.9 — LangChain Synthesis & Review

**What is it?**  
A cumulative unit that combines all LangChain concepts into a mini project, followed by a comprehensive review and knowledge check.

**Why does it matter?**  
Knowing individual components is not enough. The learner must build a complete application that uses models, prompts, parsers, chains, memory, tools, and RAG together — and then evaluate whether LangChain was the right choice.

**Prerequisites:**  
- All prior units (12.1–12.8)

---

#### Mini Project: LangChain-Powered Knowledge Assistant

**Objective:**  
Build an interactive knowledge assistant that can answer questions from documents, remember conversation history, and call external tools when needed.

**Problem Statement:**  
A small team wants a chatbot that can answer questions from their internal documentation, remember what the user has asked before, and perform simple calculations when requested.

**Requirements:**

- Load and chunk at least 5 documents
- Build a vector store for retrieval
- Implement a RAG chain that retrieves relevant chunks
- Add conversation memory for multi-turn interaction
- Include at least 2 tools (calculator, web search stub)
- Parse structured output for tool calls
- Handle errors gracefully (retrieval failure, tool failure, parse failure)
- Log all LLM calls with token usage

**Concepts Used:**

- Document loading and chunking (Unit 12.7)
- Embeddings and vector store (Unit 12.7)
- Prompt templates with variables (Unit 12.2)
- Output parsers (Unit 12.3)
- Chains with pipe operator (Unit 12.4)
- Conversation memory (Unit 12.5)
- Tool binding and execution (Unit 12.6)
- Error handling from Unit 12.1

**Suggested Architecture:**

```text
User input
    ↓
[Memory: load history]
    ↓
[Intent classification: question / calculation / general]
    ↓
┌─────────────────┬──────────────────┬─────────────────┐
│ RAG path         │ Tool path         │ Direct path      │
│ Retrieve docs    │ Call calculator   │ Answer from LLM  │
│ Generate answer  │ or web search     │                  │
└─────────────────┴──────────────────┴─────────────────┘
    ↓
[Save to memory]
    ↓
[Return answer with sources]
```

**Milestones:**

1. Document loading and chunking working
2. Vector store populated and retrieval returning relevant chunks
3. Basic RAG chain answering questions
4. Memory added — multi-turn works
5. Tools added — calculator and search stub
6. Error handling for all failure modes
7. Logging and token tracking
8. Final testing and documentation

**Expected Output:**

- Working chatbot application
- README explaining setup and decisions
- Evaluation on 10 test questions (5 retrieval, 2 tool, 3 general)
- Token usage report
- Decision document: LangChain vs manual for this project

**Evaluation Criteria:**

| Criterion | Weight | Target |
|---|---|---|
| Correctness | 30% | Answers are accurate and relevant |
| Architecture | 20% | Clean chain composition, modular design |
| Error handling | 15% | Graceful failure for all components |
| Memory | 15% | Multi-turn context preserved correctly |
| Tools | 10% | Tools called correctly and results used |
| Documentation | 10% | README, decision doc, test results |

**Failure Cases to Test:**

- Empty document collection
- Query with no relevant documents
- Tool call with invalid arguments
- Malformed LLM response
- API rate limit during conversation
- User asks something outside the system's knowledge

**Advanced Extensions:**

- Hybrid search (keyword + semantic)
- Streaming responses
- Web UI with Streamlit or Gradio
- Multi-user sessions with separate histories
- Automated evaluation with test questions
- A/B test two different chunking strategies

---

#### Comprehensive Review

**Decision Guide:**

| Scenario | Recommended Approach | Why |
|---|---|---|
| Quick prototype for stakeholder demo | LangChain | Fast composition, standard patterns |
| Production chatbot with < 100ms latency requirement | Manual | Full control over every millisecond |
| Multi-provider support (OpenAI + Anthropic + local) | LangChain | Provider-agnostic interface |
| Custom RAG with specialized chunking logic | Manual or hybrid | Framework constraints may not fit |
| Team with mixed experience levels | LangChain | Consistent patterns, shared vocabulary |
| Security-critical pipeline | Manual | Full auditability, no hidden behavior |

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| LangChain vs manual API | Standard patterns, prototyping | Simple pipelines, production critical paths | Speed of development vs control/debuggability |
| Buffer memory vs summary memory | Short conversations (< 10 turns) | Long conversations, cost-sensitive | Full history vs token efficiency |
| Pydantic parser vs JSON parser | Complex schemas, type safety | Simple JSON extraction | Validation overhead vs simplicity |
| Sequential chain vs parallel chain | Steps depend on each other | Steps are independent | Simplicity vs speed |
| LangChain vs LangGraph | Linear pipelines, < 5 steps | Complex branching, loops, state machines | Simplicity vs expressiveness |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| LangChain chain returns wrong type | Missing parser or wrong parser | Check chain composition | Add appropriate output parser |
| Framework upgrade breaks code | API change between versions | Check LangChain changelog | Pin versions, test after upgrade |
| Debugging is impossible | Too many abstraction layers | Try manual version of same step | Replace opaque step with manual code |
| Latency is too high | Framework overhead or unnecessary steps | Profile each chain step | Remove unnecessary abstractions |
| Memory grows unbounded | History not summarized or trimmed | Check memory type and window | Use summary or window memory |

---

## Phase Review Checklist

- [ ] All 9 units complete.
- [ ] Manual LLM pipeline built from scratch.
- [ ] LangChain models and prompts used with templates.
- [ ] Output parsers converting LLM text to typed objects.
- [ ] Chains composed with pipe operator.
- [ ] Memory enabling multi-turn conversations.
- [ ] Tools defined, bound, and executed.
- [ ] RAG built with LangChain components.
- [ ] LangChain vs manual comparison completed.
- [ ] Mini project built and documented.
- [ ] Decision guide written for framework choice.

---

## Mastery Check

Without following a tutorial, you should be able to:

1. Build an LLM pipeline with raw API calls.
2. Use LangChain models, prompts, and parsers together.
3. Compose multi-step chains with the pipe operator.
4. Add conversation memory to a LangChain application.
5. Define and use tools with LLM binding.
6. Build a RAG pipeline with LangChain components.
7. Decide when to use LangChain vs manual implementation.
8. Debug common failures in LangChain applications.
9. Build a complete LLM application combining all concepts.

---

## Interview / Explain-Back Questions

- What does LangChain actually do when you call `chain.invoke()`?
- Why should you learn to build LLM pipelines manually before using LangChain?
- What is the difference between `ConversationBufferMemory` and `ConversationSummaryMemory`?
- How does a LangChain tool work under the hood?
- When would you choose raw API calls over LangChain in production?
- What are the risks of using LangChain for a production system?
- How do you debug a LangChain chain that returns unexpected output?
- What is the trade-off between LangChain's convenience and its abstraction cost?
- How does RAG with LangChain differ from RAG built manually?
- When would you use LangGraph instead of LangChain chains?

---

## Exit Criteria

Move to Phase 13 (LangGraph / Advanced Orchestration) only when you can:

- Build an LLM pipeline both manually and with LangChain.
- Compose chains, add memory, use tools, and build RAG with LangChain.
- Articulate when framework abstractions help vs hurt.
- Debug common LangChain failures.
- Complete a mini project that integrates all phase concepts.
- Explain your framework choices to a technical audience.
