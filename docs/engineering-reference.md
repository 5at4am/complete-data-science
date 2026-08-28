# Engineering Reference

> **A quick-reference guide to software engineering for AI/ML systems.**

---

## Version Control (Git)

### Basic Commands
```bash
git init                    # initialize repo
git add <file>              # stage changes
git commit -m "message"     # commit
git status                  # check state
git log                     # view history
git branch                  # list branches
git checkout -b <branch>    # create/switch branch
git merge <branch>          # merge branch
```

### Best Practices
- Commit often, with meaningful messages
- One logical change per commit
- Never commit secrets
- Use `.gitignore` for generated files

---

## Python Environments

### Why
- Isolate dependencies per project
- Reproducible builds
- Avoid version conflicts

### uv (recommended)
```bash
uv init
uv add <package>
uv run python script.py
```

### venv
```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
source .venv/bin/activate  # macOS/Linux
```

---

## Project Structure

```
project/
├── src/          # source code
├── tests/        # test suite
├── notebooks/    # exploration
├── data/         # datasets
├── models/       # trained models
├── config/       # configuration
├── docs/         # documentation
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Logging

### Why
- Debug production issues
- Track system behavior
- Monitor performance

### Python logging
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("message")
logger.error("error")
```

---

## Testing

### Why
- Catch bugs early
- Prevent regressions
- Document expected behavior

### Types
- **Unit tests**: test individual functions
- **Integration tests**: test components together
- **End-to-end tests**: test the full system

### pytest
```python
# test_example.py
def test_addition():
    assert add(2, 3) == 5
```

```bash
pytest
```

---

## Type Hints

```python
def process(data: list[float], threshold: float = 0.5) -> dict[str, float]:
    ...
```

- Improves readability
- Enables static checking (mypy)
- Documents intent

---

## Configuration

### Environment Variables
```python
import os
api_key = os.getenv("API_KEY")
```

### .env Files
```
# .env
API_KEY=secret
```
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## API Design (REST)

### Principles
- Resource-based URLs: `/users`, `/predictions`
- HTTP methods: GET, POST, PUT, DELETE
- Status codes: 200, 400, 404, 500
- JSON responses

### FastAPI Example
```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

@app.post("/items")
def create_item(item: Item):
    return {"id": 1, **item.model_dump()}
```

---

## Docker

### Why
- Consistent environments
- Easy deployment
- Isolation

### Dockerfile
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
```

### Commands
```bash
docker build -t myapp .
docker run -p 8000:8000 myapp
```

---

## CI/CD

### Continuous Integration
- Automatically test code on every push
- Catch bugs early

### Continuous Deployment
- Automatically deploy passing code
- Rollback on failure

### GitHub Actions
```yaml
name: CI
on: [push]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest
```

---

## Monitoring & Observability

### For ML Systems
- Model performance drift
- Data drift
- Latency
- Error rates
- Cost

### For LLM Systems
- Prompt versioning
- Model versioning
- Dataset versioning
- Experiment tracking
- Evaluation datasets
- Cost monitoring
- Latency monitoring
- Fallbacks
- Guardrails

---

## Security

### API Keys
- Never hard-code
- Use environment variables
- Rotate regularly

### Authentication/Authorization
- Verify who is calling
- Control what they can do

### Rate Limiting
- Prevent abuse
- Control costs

### Input/Output Validation
- Validate all inputs
- Sanitize outputs

### For GenAI
- Prompt injection defense
- Data leakage prevention
- PII handling
- Tool abuse prevention
- RAG poisoning defense

---

## Experiment Tracking

### Why
- Reproducibility
- Compare experiments
- Track what worked

### Tools
- MLflow
- Weights & Biases
- Simple CSV/JSON logs

### What to Track
- Hyperparameters
- Metrics
- Dataset version
- Code version
- Model artifacts
