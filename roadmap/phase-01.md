# Phase 01 — Python Foundations

> **Goal:** Build solid Python fluency for data work, including NumPy, Pandas, visualization, files, errors, and reusable code.

**Difficulty:** 🟢 Beginner → 🟡 Intermediate  
**Priority:** Essential  
**Prerequisites:** Phase 00  
**Mastery target:** Level 5 — decision making for core Python/data tools

---

## Why This Phase Exists

Machine learning work is still software work. Before models, the learner must be able to write clear Python, manipulate data, inspect errors, and organize small programs. Weak Python creates confusion later because ML errors often look mathematical even when the real problem is data shape, type, indexing, or file handling.

### Phase Mental Model

Python is the control language. NumPy is the math engine. Pandas is the table engine. Matplotlib is the visual inspection tool.

```text
Python basics → data structures → functions → reusable code
       ↓
NumPy arrays → vectorized math
       ↓
Pandas DataFrames → tabular data workflows
       ↓
Visualization + files + errors → practical data programs
```

### What This Phase Prepares For

- math implementation in Phase 02
- statistical exploration in Phase 03
- data cleaning in Phase 04
- model pipelines in Phase 05
- reproducible projects throughout the roadmap

---

## Units

### Unit 01.1 — Python Basics

**What is it?**  
The core syntax and control flow needed to express simple programs.

**Why does it matter?**  
Every data and ML workflow depends on variables, conditions, loops, and types.

**Prerequisites:** Phase 00 environment and notebook workflow.

**Core Concepts:**

- variables and assignment
- numbers, strings, booleans
- comparison and logical operators
- `if` / `elif` / `else`
- `for` and `while` loops
- indentation and code blocks

**Simple Example:**

```python
score = 0.82

if score >= 0.8:
    label = "pass"
else:
    label = "review"

print(label)
```

**Common Mistakes:**

- confusing `=` with `==`
- off-by-one errors in loops
- inconsistent indentation
- treating strings and numbers as interchangeable

**Hands-On Practice:**

1. Basic: write a program that classifies a numeric score.
2. Guided: loop over a list of scores and count passes/fails.
3. Independent: write a small rule-based classifier for simple inputs.
4. Challenge: handle invalid input without crashing.

**Exit Criteria:**

- You can write small Python programs with variables, conditions, and loops.
- You can explain the difference between syntax errors and logic errors.

**Next Step:** Learn data structures so programs can store and transform collections.

---

### Unit 01.2 — Data Structures

**What is it?**  
Python containers for organizing multiple values.

**Why does it matter?**  
Data rarely arrives as one value. You need lists, dictionaries, tuples, and sets to organize records, labels, configuration, and results.

**Mental Model:**

- list → ordered sequence
- tuple → fixed record
- dict → key-value lookup table
- set → unique membership collection

**Core Concepts:**

- indexing and slicing
- mutation vs immutability
- iteration
- comprehensions
- nested structures
- choosing the right structure

**Decision Guidance:**

| Structure | Use When | Avoid When |
|---|---|---|
| `list` | Order matters and items may change | You need fast key lookup |
| `tuple` | Fixed grouped values | You need to modify fields often |
| `dict` | Lookup by name/id/key | You only need ordered numeric operations |
| `set` | Uniqueness and membership checks | You need duplicates or stable ordering |

**Simple Example:**

```python
student = {"name": "Asha", "score": 0.91, "passed": True}
```

**Hands-On Practice:**

1. Basic: create and modify each structure.
2. Guided: convert a list of records into summary counts.
3. Independent: choose the right structures for a small gradebook.
4. Realistic: clean duplicate values while preserving useful metadata.

**Common Mistakes:**

- using a list when a dictionary would be clearer
- mutating a list while iterating over it
- assuming sets preserve duplicates
- building deeply nested structures that are hard to inspect

**Exit Criteria:**

- You can choose the correct data structure for common tasks.
- You can transform nested data without losing meaning.

**Next Step:** Wrap repeated logic into functions.

---

### Unit 01.3 — Functions & Scope

**What is it?**  
Functions package reusable logic with inputs, outputs, and local variables.

**Why does it matter?**  
Reusable functions prevent copy-paste code and make ML/data pipelines testable.

**Mental Model:**  
A function is a small machine: inputs go in, work happens inside, result comes out.

**Core Concepts:**

- `def`
- parameters and arguments
- return values
- default arguments
- local vs global scope
- pure functions vs side effects
- small tests/examples

**Simple Example:**

```python
def accuracy(correct: int, total: int) -> float:
    if total == 0:
        raise ValueError("total must be positive")
    return correct / total
```

**Common Mistakes:**

- printing instead of returning
- relying on global variables
- using mutable default arguments
- making one function do too many things

**Debugging:**

| Symptom | Possible Cause | Fix |
|---|---|---|
| Function returns `None` | Missing `return` | Return the computed value |
| Result changes unexpectedly | Global state or mutation | Pass inputs explicitly and avoid side effects |
| Function is hard to test | Too many responsibilities | Split into smaller functions |

**Hands-On Practice:**

1. Basic: turn repeated code into a function.
2. Guided: write functions for mean, min, max, and standardization.
3. Independent: build a small metrics module.
4. Challenge: add input validation and meaningful errors.

**Exit Criteria:**

- You can write reusable functions with clear inputs and outputs.
- You can explain scope and avoid accidental global-state bugs.

**Next Step:** Learn OOP enough to understand ML library objects.

---

### Unit 01.4 — Object-Oriented Programming

**What is it?**  
Object-oriented programming organizes state and behavior into classes and objects.

**Why does it matter?**  
Many ML tools use objects: estimators, transformers, datasets, dataloaders, models, callbacks.

**Why learn it here?**  
The learner does not need advanced OOP yet, but must understand how library objects store configuration and expose methods.

**Mental Model:**  
A class is a blueprint. An object is one built instance of that blueprint.

**Core Concepts:**

- class and object
- `__init__`
- attributes
- methods
- encapsulation
- inheritance basics
- composition over inheritance

**Simple Example:**

```python
class RunningMean:
    def __init__(self):
        self.total = 0
        self.count = 0

    def update(self, value):
        self.total += value
        self.count += 1

    def value(self):
        return self.total / self.count
```

**Common Mistakes:**

- using classes for everything when functions are simpler
- using inheritance when composition would be clearer
- hiding too much state
- mutating object state without understanding when it changes

**Industry Perspective:**  
In ML projects, classes are useful for reusable pipelines, model wrappers, datasets, and services. Simple scripts and notebooks often need functions more than complex class hierarchies.

**Hands-On Practice:**

1. Basic: create a class with attributes and methods.
2. Guided: build a simple metric tracker.
3. Independent: design a `DataCleaner` object with configurable steps.
4. Challenge: compare a function-based and class-based solution and explain which is clearer.

**Exit Criteria:**

- You can read and use object-based ML APIs.
- You can decide when a class is useful and when a function is enough.

**Next Step:** Use NumPy for fast numerical operations.

---

### Unit 01.5 — NumPy Fundamentals

**What is it?**  
NumPy is Python’s core library for fast numerical arrays and vectorized math.

**Why does it matter?**  
ML data, model parameters, vectors, matrices, and gradients are numeric arrays.

**Mental Model:**  
A NumPy array is a compact grid of same-type numbers designed for fast math.

**Core Concepts:**

- arrays and dtypes
- shape and dimensions
- indexing and slicing
- broadcasting
- vectorized operations
- random numbers
- aggregations

**Simple Example:**

```python
import numpy as np

x = np.array([1, 2, 3])
w = np.array([0.1, 0.2, 0.3])
prediction = np.dot(x, w)
```

**Decision Guidance: List vs NumPy Array**

| Use Python list when... | Use NumPy array when... |
|---|---|
| Data is small or mixed-type | Data is numeric and large |
| You need general container behavior | You need vectorized math |
| Readability matters more than speed | Speed and shape operations matter |

**Common Mistakes:**

- shape mismatch
- unintended broadcasting
- integer division/type issues
- using loops where vectorization is clearer

**Debugging:**

Always inspect:

```python
array.shape
array.dtype
array[:5]
```

**Hands-On Practice:**

1. Basic: create arrays and compute sums/means.
2. Guided: normalize a numeric vector.
3. Independent: implement mean squared error with NumPy.
4. Realistic: debug a shape mismatch in matrix multiplication.
5. Challenge: replace a loop with vectorized code and compare speed/readability.

**Exit Criteria:**

- You can use NumPy arrays for vectorized calculations.
- You can debug shape and dtype problems.

**Next Step:** Use Pandas for labeled tabular data.

---

### Unit 01.6 — Pandas Fundamentals

**What is it?**  
Pandas is a library for working with tabular data using Series and DataFrames.

**Why does it matter?**  
Most beginner and industry ML projects start with messy tables: CSVs, spreadsheets, database exports, logs, or business records.

**Mental Model:**  
A DataFrame is a programmable spreadsheet with named columns and index-aware operations.

**Core Concepts:**

- Series and DataFrames
- loading CSV files
- column selection
- filtering rows
- sorting
- missing values
- `groupby`
- joins/merges
- simple feature creation

**Simple Example:**

```python
import pandas as pd

df = pd.read_csv("data.csv")
summary = df.groupby("category")["price"].mean()
```

**Alternatives:**

| Tool | Use When | Avoid When |
|---|---|---|
| Pandas | Learning, medium-sized data, broad ecosystem | Data is too large for memory |
| Polars | Large/fast tabular workflows, lazy execution | Team/project depends heavily on Pandas ecosystem |
| SQL | Data lives in a database or needs relational querying | You need complex in-memory ML preprocessing |

**Common Mistakes:**

- chained assignment confusion
- not checking missing values
- treating IDs as meaningful numeric features
- using row loops instead of vectorized operations
- merging on wrong keys

**Debugging:**

Inspect data with:

```python
df.head()
df.info()
df.describe()
df.isna().sum()
df.dtypes
```

**Hands-On Practice:**

1. Basic: load a CSV and inspect columns.
2. Guided: filter, group, and summarize rows.
3. Independent: clean a small messy table.
4. Realistic: join two tables and verify row counts.
5. Challenge: create features and explain why they may help a model.

**Exit Criteria:**

- You can load, inspect, filter, group, join, and clean tabular data.
- You can explain when Pandas is the right tool and when another tool may be better.

**Next Step:** Visualize data to see patterns and problems.

---

### Unit 01.7 — Matplotlib & Visualization

**What is it?**  
Visualization turns data into plots that reveal patterns, errors, and relationships.

**Why does it matter?**  
Tables hide many problems. Plots expose outliers, skew, imbalance, trends, and suspicious patterns.

**Core Concepts:**

- line plots
- scatter plots
- bar charts
- histograms
- subplots
- labels and titles
- visual clarity

**Simple Example:**

```python
import matplotlib.pyplot as plt

plt.hist(df["age"], bins=30)
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
```

**Common Mistakes:**

- missing labels/title
- misleading axis scales
- using the wrong chart type
- plotting too much at once
- treating a pretty chart as proof

**Hands-On Practice:**

1. Basic: create line, scatter, bar, and histogram plots.
2. Guided: visualize one numeric and one categorical feature.
3. Independent: choose plots for a small EDA report.
4. Challenge: find one suspicious pattern and explain how you would investigate it.

**Exit Criteria:**

- You can choose simple plots for common data questions.
- You can use plots to support debugging and explanation.

**Next Step:** Learn file I/O and errors so data workflows are robust.

---

### Unit 01.8 — File I/O & Error Handling

**What is it?**  
Reading/writing files and handling failures gracefully.

**Why does it matter?**  
Real projects depend on files: datasets, configs, model artifacts, logs, reports. File and input errors are normal.

**Core Concepts:**

- paths and project-relative paths
- reading/writing text, CSV, JSON
- `pathlib`
- exceptions
- `try` / `except`
- raising meaningful errors
- logging basics

**Simple Example:**

```python
from pathlib import Path
import json

path = Path("config.json")

if path.exists():
    config = json.loads(path.read_text())
else:
    raise FileNotFoundError(f"Missing config: {path}")
```

**Common Mistakes:**

- hardcoding local absolute paths
- catching every exception and hiding the real problem
- assuming files are always clean and present
- overwriting outputs accidentally

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| File not found | Wrong path/working directory | Print `Path.cwd()` | Use project-relative `Path` |
| Unicode error | Encoding mismatch | Check file encoding | Specify encoding such as UTF-8 |
| Silent failure | Broad `except` hides error | Log exception details | Catch specific exceptions |

**Hands-On Practice:**

1. Basic: read and write a text file.
2. Guided: load JSON config and validate required keys.
3. Independent: build a script that processes an input file and writes an output file.
4. Realistic: handle missing, malformed, and empty files.

**Exit Criteria:**

- You can write file-based programs that fail clearly.
- You can avoid path bugs and unsafe exception handling.

**Next Step:** Combine all Python skills into a practical data pipeline.

---

### Unit 01.9 — Python for Data Work Synthesis

**What is it?**  
A cumulative integration unit that combines Python basics, functions, OOP awareness, NumPy, Pandas, visualization, files, and error handling.

**Why does it matter?**  
Knowing isolated syntax is not enough. The learner must build a small data workflow independently.

**Mini Project:** Data Processing Pipeline

**Objective:** Build a small pipeline that loads raw tabular data, validates it, cleans it, computes summaries, creates plots, and writes outputs.

**Requirements:**

- load data from file
- validate required columns
- handle missing or invalid values
- create at least 3 summary statistics
- create at least 2 plots
- save a cleaned output file
- organize repeated logic into functions
- include meaningful errors
- write a short README explaining usage

**Suggested Architecture:**

```text
Input file → validation → cleaning → summary calculations → visualization → output files/report
```

**Expected Output:**

- cleaned CSV
- summary table
- plots
- README
- reusable Python functions or script

**Evaluation Criteria:**

- code runs from a clean environment
- functions are clear and reusable
- errors are helpful
- outputs are reproducible
- plots answer real questions
- README explains decisions and limitations

**Advanced Extensions:**

- command-line arguments
- logging
- simple class wrapper
- unit tests for helper functions
- compare Pandas and Polars for one operation

**Knowledge Check:**

- Why should validation happen before cleaning?
- Which parts belong in functions rather than notebook cells?
- What would break if the input file changed column names?
- How would you make the pipeline easier for another person to run?

**Exit Criteria:**

- You can build a small data-processing workflow without a tutorial.
- You can debug common Python/data/file errors.
- You can explain why you chose each data structure and library.

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Function vs class | Logic is stateless/simple | State/configuration must be preserved | Simplicity vs structure |
| List vs NumPy array | General collection/mixed data | Numeric vectorized computation | Flexibility vs speed/math operations |
| Pandas vs Polars | Learning, ecosystem, medium data | Larger/faster/lazy tabular workloads | Familiarity vs performance |
| Notebook vs script | Exploration and explanation | Reusable repeatable execution | Interactivity vs maintainability |
| Broad exception vs specific exception | Almost never | Expected failure modes are known | Hiding bugs vs clear diagnosis |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Code works in notebook but not script | Hidden notebook state or path issue | Restart kernel/run script from root | Use functions and project-relative paths |
| Pandas column operation fails | Wrong dtype or missing column | Check `df.dtypes` and `df.columns` | Convert types and validate schema |
| NumPy math gives unexpected shape | Broadcasting or dimension mismatch | Print `.shape` before operation | Reshape explicitly and write small tests |
| Program crashes on bad input | No validation/error handling | Try empty/malformed files | Add checks and specific exceptions |
| Code is hard to modify | Too much copy-paste | Search repeated blocks | Extract functions/classes |

---

## Phase Review Checklist

- [ ] All units complete.
- [ ] Python basics practiced with small programs.
- [ ] Data structures chosen intentionally.
- [ ] Functions written with clear inputs and outputs.
- [ ] OOP understood enough to use ML library APIs.
- [ ] NumPy shape/dtype debugging practiced.
- [ ] Pandas EDA and cleaning practiced.
- [ ] Visualizations created with clear labels and purpose.
- [ ] File I/O and error handling practiced.
- [ ] Data processing mini project completed.
- [ ] Cumulative review passed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Write Python to process data independently.
2. Choose appropriate data structures.
3. Organize reusable logic into functions.
4. Read and use class-based APIs.
5. Use NumPy for vectorized computation.
6. Use Pandas for tabular data.
7. Visualize data with Matplotlib.
8. Handle files and errors robustly.
9. Build a small data pipeline from scratch.

## Interview / Explain-Back Questions

- When should you use a list instead of a NumPy array?
- Why is Pandas useful for tabular data?
- What is broadcasting in NumPy, and how can it cause bugs?
- Why should functions return values instead of only printing?
- When is a class useful in ML/data code?
- How do you debug a script that cannot find a file?
- What makes a plot misleading?

## Exit Criteria

Move to Phase 02 only when you can independently build a small, reproducible Python data pipeline and explain your choices, errors, and outputs.
