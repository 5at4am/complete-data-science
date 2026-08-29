# Phase 16 — Deployment & MLOps

> **Goal:** Master deployment — APIs, Docker, CI/CD, monitoring, and production ML/AI engineering.

**Difficulty:** 🟡 Intermediate → 🔴 Advanced
**Priority:** Essential
**Prerequisites:** Phase 15 (Evaluation, Testing, and Iteration)
**Mastery target:** Level 5 — independent production deployment and incident response

---

## Why This Phase Exists

A model is not finished when it works in a notebook. It is finished when it can be served, monitored, secured, rolled back, and maintained. Most ML projects fail not because the model is wrong but because the system around the model is fragile. This phase turns experiment-stage work into reliable production systems.

### Phase Mental Model

Deployment converts experiments into systems:

```text
Model artifact → API/service → tests → container → CI/CD → monitoring → incident response → improvement
```

Think of it as a relay race. The model passes the baton to the API, which passes to the container, which passes to CI/CD, which passes to monitoring. Each handoff must be clean or the system drops the baton.

### What This Phase Prepares For

- capstone projects (Phase 17) that require deployed, monitored systems
- real-world roles where models serve users and businesses
- reasoning about cost, latency, reliability, and safety at scale
- interview readiness for ML engineering and MLOps roles

---

## Units

---

### Unit 16.1 — API Design (FastAPI)

#### What is it?

An API (Application Programming Interface) is a structured contract that lets software components communicate. A REST API uses HTTP requests and responses to expose functionality. FastAPI is a Python framework that makes building REST APIs fast, type-safe, and self-documenting.

#### Why does it matter?

ML models are useless if nothing can call them. APIs are the bridge between a trained model and any client — a web app, a mobile app, another service, or a batch pipeline. Poor API design creates fragile integrations that break under load, return unclear errors, and frustrate consumers.

#### Why learn it here?

You have built models, evaluated them, and understand their behavior. Now you need to expose that behavior through a reliable interface. Learning API design here gives you the foundation for every deployment step that follows.

#### Prerequisites

- Python functions and classes (Phase 01)
- JSON and HTTP basics (conceptual)
- A trained model artifact ready to serve

#### Mental Model

An API is a restaurant menu. The kitchen (your model) never speaks directly to the customer. The menu (API contract) lists what is available, what inputs are needed, and what outputs to expect. The waiter (API server) takes the order, validates it, brings it to the kitchen, and returns the result.

#### Core Concepts

- HTTP methods: GET, POST, PUT, DELETE
- Endpoints and routes
- Request/response bodies (JSON)
- Status codes (200, 201, 400, 404, 422, 500)
- Path parameters vs query parameters vs request body
- Pydantic models for validation
- Async vs sync endpoints
- Dependency injection
- OpenAPI / Swagger documentation

#### How It Works

FastAPI uses Python type hints and Pydantic to automatically validate requests, serialize responses, and generate API documentation. You define a function, add type annotations, and FastAPI handles parsing, validation, and documentation.

#### Syntax & Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib

app = FastAPI(title="House Price Predictor")

# Load model at startup (not per request)
model = joblib.load("model.pkl")

class PredictionRequest(BaseModel):
    sqft: float
    bedrooms: int
    bathrooms: float

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if request.sqft <= 0:
        raise HTTPException(status_code=400, detail="sqft must be positive")
    if request.bedrooms < 1:
        raise HTTPException(status_code=400, detail="bedrooms must be >= 1")

    features = np.array([[request.sqft, request.bedrooms, request.bathrooms]])
    prediction = model.predict(features)[0]

    return PredictionResponse(
        predicted_price=round(float(prediction), 2),
        model_version="1.0.0"
    )
```

#### Simple Example

```python
# A minimal endpoint
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Hello, ML world"}
```

#### Real-World Example

A recommendation service exposes `/recommend` with user ID and context, returns ranked items with confidence scores, includes rate limiting, request logging, and graceful fallback when the model is unavailable.

#### Common Mistakes

- Loading the model inside the endpoint function (slow on every request)
- Not validating input types and ranges
- Returning raw numpy arrays instead of serializable formats
- Missing a `/health` endpoint for load balancers
- Exposing stack traces to clients in production
- Not using Pydantic for request/response validation

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| 422 Unprocessable Entity | Request body doesn't match Pydantic schema | Check field names and types | Match the schema exactly |
| First request is very slow | Model loads on first call | Trace startup time | Load model at startup with `joblib` |
| Response contains numpy objects | Not converting to Python types | Check serialization | Use `float()`, `.tolist()`, or Pydantic models |
| CORS errors from browser | Missing CORS middleware | Check browser network tab | Add `CORSMiddleware` |
| 500 on valid input | Unhandled exception in predict | Check server logs | Add try/except with meaningful errors |

#### Alternatives

| Framework | Use When | Avoid When |
|---|---|---|
| FastAPI | New Python APIs, type safety, async needed | Legacy codebase on Flask |
| Flask | Simple, well-known, large ecosystem | Need automatic docs, type validation |
| Django REST Framework | Need admin panel, ORM, auth built-in | Lightweight microservice |
| gRPC | High-performance internal service communication | Browser-facing API (needs proxy) |
| BentoML | ML-specific serving with built-in packaging | Need full control over API design |

#### Best Practices

- Load models at startup, not per request
- Use Pydantic for all request/response validation
- Always include a `/health` endpoint
- Version your API (URL or header)
- Return meaningful error messages without leaking internals
- Use async endpoints for I/O-bound work
- Document with OpenAPI (FastAPI does this automatically)
- Set request size limits

#### Hands-On Practice

1. **Basic:** Create a FastAPI app with GET and POST endpoints
2. **Guided:** Add Pydantic validation for a prediction request with ranges
3. **Independent:** Build an API that loads a trained model and serves predictions
4. **Realistic:** Add health checks, error handling, and API documentation
5. **Challenge:** Add rate limiting, request logging, and compare latency with/without async

#### Knowledge Check

- Why should you load the model at startup rather than inside the endpoint?
- What is the difference between 400 and 422 status codes?
- How does Pydantic validation differ from manual if/else checks?
- What happens if your API receives a request with extra unexpected fields?

#### Exit Criteria

- You can build a FastAPI service that validates input, serves predictions, and handles errors
- You can explain status codes, request types, and Pydantic validation
- You can run the API and test it with curl or a client

#### Next Step:** Serve a real ML model with preprocessing and postprocessing.

---

### Unit 16.2 — Model Serving

#### What is it?

Model serving is the process of loading a trained model and making it available for real-time or batch predictions. It includes preprocessing, inference, postprocessing, and response formatting — not just the model call itself.

#### Why does it matter?

A model is only useful if it produces predictions on new data reliably and quickly. Serving is where preprocessing mismatches, latency issues, and deployment bugs appear. Getting this right means your model works the same in production as in your notebook.

#### Why learn it here?

After building an API (16.1), you need to integrate your actual model with proper preprocessing. This unit teaches the full prediction pipeline, not just the model call.

#### Prerequisites

- FastAPI basics (16.1)
- A trained model (from earlier phases)
- Understanding of preprocessing pipelines

#### Mental Model

Model serving is a factory assembly line. Raw materials (input data) arrive, go through preparation stations (validation, preprocessing), reach the main machine (model inference), get finished (postprocessing), and leave as a product (prediction response). Every station must work correctly or the final product is wrong.

#### Core Concepts

- Model loading and artifact management
- Preprocessing consistency (same transforms as training)
- Real-time vs batch vs streaming inference
- Latency and throughput requirements
- Input/output schema validation
- Model versioning
- Fallback strategies
- Resource management (CPU/GPU/memory)

#### How It Works

1. Load the model and preprocessing artifacts at startup
2. Validate incoming request data
3. Apply preprocessing (same transforms as training)
4. Run model inference
5. Apply postprocessing (thresholding, formatting)
6. Return structured response

#### Syntax & Implementation

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
from pathlib import Path

app = FastAPI(title="Churn Predictor")

# Load model AND preprocessing pipeline
MODEL_DIR = Path("artifacts")
model = joblib.load(MODEL_DIR / "model.pkl")
scaler = joblib.load(MODEL_DIR / "scaler.pkl")
feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

class ChurnRequest(BaseModel):
    tenure_months: float
    monthly_charges: float
    total_charges: float
    contract_type: str
    internet_service: str

class ChurnResponse(BaseModel):
    will_churn: bool
    churn_probability: float
    confidence: str

@app.post("/predict", response_model=ChurnResponse)
def predict(request: ChurnRequest):
    # 1. Validate
    if request.tenure_months < 0:
        raise HTTPException(400, "tenure_months must be non-negative")
    if request.contract_type not in ["month-to-month", "one-year", "two-year"]:
        raise HTTPException(400, f"Unknown contract: {request.contract_type}")

    # 2. Preprocess (same as training)
    features = preprocess(request)

    # 3. Inference
    probability = model.predict_proba(features)[0][1]

    # 4. Postprocess
    will_churn = probability >= 0.5
    confidence = "high" if abs(probability - 0.5) > 0.3 else "medium" if abs(probability - 0.5) > 0.1 else "low"

    return ChurnResponse(
        will_churn=will_churn,
        churn_probability=round(float(probability), 4),
        confidence=confidence
    )

def preprocess(req: ChurnRequest) -> np.ndarray:
    numeric = np.array([[req.tenure_months, req.monthly_charges, req.total_charges]])
    scaled = scaler.transform(numeric)
    # Add encoded categorical features
    return scaled  # Simplified for illustration
```

#### Simple Example

```python
import joblib
import numpy as np

# Load and predict
model = joblib.load("model.pkl")
features = np.array([[5.2, 1.0, 0.3]])
prediction = model.predict(features)
print(f"Prediction: {prediction[0]}")
```

#### Real-World Example

A credit scoring system receives loan applications, validates all required fields, runs the same feature engineering pipeline used in training (binning, encoding, scaling), calls the model, applies business rules (auto-approve below threshold, auto-reject above), logs every prediction for audit, and returns a decision with explanation.

#### Common Mistakes

- Using different preprocessing in production than in training
- Not loading preprocessing artifacts alongside the model
- Ignoring input validation and letting bad data through
- Not handling missing or null values the same way as training
- Returning raw model outputs without postprocessing
- Not versioning model artifacts

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Predictions differ between notebook and API | Preprocessing mismatch | Compare transformed features | Package preprocessing as a single pipeline |
| NaN in predictions | Missing values not handled | Check input data for nulls | Add imputation matching training |
| Very slow first prediction | Model not loaded at startup | Time startup vs first request | Load all artifacts in `@app.on_event("startup")` |
| Feature mismatch error | Different feature order/names | Compare training and serving feature lists | Save and load feature names |
| Model file not found | Wrong path or missing artifact | Check `artifacts/` directory | Use absolute paths or env vars |

#### Alternatives

| Approach | Use When | Avoid When |
|---|---|---|
| FastAPI + joblib | Simple model, full control | Need GPU or distributed serving |
| TensorFlow Serving | TF models, batch + real-time | Non-TF models |
| TorchServe | PyTorch models, production features | Non-PT models |
| BentoML | Full ML serving framework | Need minimal dependencies |
| SageMaker/GCP AI Platform | Cloud-managed, auto-scaling | On-premise or cost-sensitive |
| Triton Inference Server | Multi-model, GPU, high throughput | Simple CPU models |

#### Best Practices

- Package the full pipeline (preprocessing + model) as one artifact
- Version model artifacts with metadata (date, metrics, data hash)
- Set up model registry for tracking and rollback
- Monitor prediction latency and throughput
- Log inputs and outputs for debugging and audit
- Implement fallback for model unavailability
- Use batch prediction for non-real-time workloads

#### Hands-On Practice

1. **Basic:** Load a joblib model and make a prediction in a script
2. **Guided:** Wrap a model in a FastAPI endpoint with preprocessing
3. **Independent:** Build a serving pipeline that loads a scaler, encoder, and model
4. **Realistic:** Add logging, error handling, and a health check
5. **Challenge:** Compare real-time serving latency vs batch processing for 1000 records

#### Knowledge Check

- Why must preprocessing be identical in training and serving?
- What is the difference between batch and real-time inference?
- How would you handle a model that is unavailable for 30 seconds?
- What metadata should accompany a deployed model artifact?

#### Exit Criteria

- You can serve a model with consistent preprocessing
- You can explain batch vs real-time trade-offs
- You can debug prediction mismatches between notebook and API

#### Next Step:** Package the entire service in a container for reproducible deployment.

---

### Unit 16.3 — Docker

#### What is it?

Docker is a containerization platform that packages an application and all its dependencies into a single unit (container) that runs the same everywhere — on your laptop, a server, or the cloud.

#### Why does it matter?

"It works on my machine" is the most common deployment failure. Docker eliminates this by freezing the environment: Python version, system libraries, dependencies, and code all ship together. Containers also enable scaling, isolation, and consistent CI/CD.

#### Why learn it here?

You have a working API serving a model. Now you need to ensure it runs identically in any environment. Docker is the industry standard for this.

#### Prerequisites

- API serving a model (16.1, 16.2)
- Basic command line
- Understanding of dependencies and virtual environments

#### Mental Model

A Docker container is a shipping container for software. Just as a physical shipping container holds cargo and can be loaded on any ship, truck, or train, a Docker container holds your code and dependencies and can run on any machine with Docker installed.

```text
Your code + dependencies + runtime → Docker image → Container (running instance)
```

#### Core Concepts

- Image vs container
- Dockerfile syntax (FROM, WORKDIR, COPY, RUN, CMD, EXPOSE)
- Building images
- Running containers
- Volumes for data persistence
- Environment variables
- Multi-stage builds
- Docker Compose for multi-service setups
- Layer caching

#### How It Works

1. Write a `Dockerfile` describing the environment
2. Build an image from the Dockerfile
3. Run a container from the image
4. The container has its own filesystem, network, and process space

#### Syntax & Implementation

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the API
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-api .
docker run -p 8000:8000 -e MODEL_PATH=/app/artifacts/model.pkl ml-api
```

```yaml
# docker-compose.yml for multi-service setup
version: "3.8"
services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/artifacts/model.pkl
    volumes:
      - ./artifacts:/app/artifacts
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
```

#### Simple Example

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install fastapi uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### Real-World Example

A data science team packages their model service with a pinned Python version, specific system libraries for image processing, a health check endpoint, and resource limits. The same image runs on the developer laptop, staging server, and Kubernetes cluster without changes.

#### Common Mistakes

- Using `latest` tag for base images (non-reproducible)
- Copying everything before installing dependencies (breaks layer caching)
- Running as root in the container
- Not using `.dockerignore` (sends entire repo to Docker daemon)
- Hardcoding secrets in the Dockerfile
- Not setting resource limits

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Works locally, fails in container | Missing system dependency or env var | Check `docker logs` | Add `RUN apt-get install` or env vars |
| Large image size | No multi-stage build, unnecessary files | `docker images` | Use slim base, multi-stage build, .dockerignore |
| Container exits immediately | CMD fails or wrong command | `docker logs <container>` | Check command, add ENTRYPOINT if needed |
| Cannot connect to API | Port not exposed or firewall | `docker ps`, curl inside container | Add EXPOSE, check `-p` mapping |
| Slow build | No layer caching | Inspect build output | Order Dockerfile: deps before code |

#### Alternatives

| Tool | Use When | Avoid When |
|---|---|---|
| Docker | Standard containerization, wide support | Need lightweight VMs (use Podman) |
| Podman | Rootless containers, Docker alternative | Team uses Docker Compose heavily |
| Singularity | HPC, scientific computing | Standard web deployment |
| Docker Compose | Multi-service local dev | Single-service apps |
| Kubernetes | Production orchestration at scale | Small projects or single-server |

#### Best Practices

- Use slim base images (`python:3.11-slim`, not `python:3.11`)
- Copy `requirements.txt` first, install, then copy code (layer caching)
- Create a `.dockerignore` file
- Run as non-root user
- Set health checks
- Pin base image versions
- Use multi-stage builds for production
- Never hardcode secrets in images

#### Hands-On Practice

1. **Basic:** Write a Dockerfile for a simple Python script
2. **Guided:** Dockerize the FastAPI model service from 16.2
3. **Independent:** Add health checks, env vars, and volume mounts
4. **Realistic:** Set up Docker Compose with the API and a database
5. **Challenge:** Optimize image size with multi-stage builds and compare

#### Knowledge Check

- What is the difference between an image and a container?
- Why should you copy `requirements.txt` before the rest of the code?
- What is layer caching and how does it speed up builds?
- How do you pass configuration to a container without rebuilding?

#### Exit Criteria

- You can write a Dockerfile and build/run a container
- You can use Docker Compose for multi-service setups
- You can troubleshoot common container issues

#### Next Step:** Automate testing and deployment with CI/CD pipelines.

---

### Unit 16.4 — CI/CD

#### What is it?

CI/CD (Continuous Integration / Continuous Deployment) is the practice of automating testing, building, and deploying code changes. CI ensures every change is tested automatically. CD ensures tested changes reach production safely and quickly.

#### Why does it matter?

Manual deployment is slow, error-prone, and risky. Automated pipelines catch bugs early, enforce quality standards, and make releases predictable. In ML, CI/CD also includes model validation, data checks, and monitoring.

#### Why learn it here?

You have a containerized service. Now you need to automate quality checks and deployments. CI/CD is the bridge between "it works" and "it stays working."

#### Prerequisites

- Git basics
- Docker (16.3)
- Unit testing concepts
- Python project structure

#### Mental Model

CI/CD is an automated quality gate system. Every code change passes through a series of gates (tests, linting, building, deploying). If any gate fails, the change is rejected before it reaches production. This is like a factory assembly line with inspection stations.

```text
Code push → lint → test → build → deploy to staging → manual approve → deploy to production
```

#### Core Concepts

- Git hooks and branching strategies
- GitHub Actions / GitLab CI / Jenkins
- Pipeline stages: lint, test, build, deploy
- Automated testing (unit, integration, E2E)
- Environment promotion (dev → staging → prod)
- Rollback strategies
- Secrets management in CI
- Artifact versioning
- ML-specific CI: data validation, model testing, drift checks

#### How It Works

1. Developer pushes code to a branch
2. CI pipeline triggers automatically
3. Pipeline runs linter, tests, and builds a Docker image
4. If all pass, image is pushed to a registry
5. CD pipeline deploys to staging
6. After approval, deploys to production

#### Syntax & Implementation

```yaml
# .github/workflows/ml-api.yml
name: ML API CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-dev.txt
      - name: Lint
        run: ruff check .
      - name: Type check
        run: mypy src/
      - name: Unit tests
        run: pytest tests/unit -v
      - name: Integration tests
        run: pytest tests/integration -v

  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t ml-api:${{ github.sha }} .
      - name: Run container tests
        run: |
          docker run -d --name test-api -p 8000:8000 ml-api:${{ github.sha }}
          sleep 10
          curl -f http://localhost:8000/health || exit 1
          docker stop test-api

  deploy:
    needs: build
    if: github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to production
        run: echo "Deploy ml-api:${{ github.sha }}"
```

#### Simple Example

```yaml
# Minimal CI pipeline
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

#### Real-World Example

A team's CI pipeline runs linting, type checking, unit tests, and integration tests on every PR. The build step creates a Docker image tagged with the commit SHA. CD deploys to staging automatically, runs smoke tests, and requires manual approval before promoting to production. A separate pipeline monitors model quality weekly.

#### Common Mistakes

- Not testing in the same environment as production
- Skipping integration tests
- Hardcoding secrets in pipeline files
- Not caching dependencies (slow builds)
- Deploying without rollback capability
- Ignoring flaky tests

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Pipeline fails on CI but works locally | Environment difference | Compare Python/OS versions | Use Docker in CI |
| Tests pass but deploy fails | Missing env vars or secrets | Check deployment logs | Add secrets to CI/CD settings |
| Build is very slow | No dependency caching | Check build steps | Add pip cache action |
| Flaky test fails randomly | Race condition or external dependency | Check test timing | Isolate or retry flaky tests |
| Deploy succeeds but app crashes | Missing migration or config | Check container logs | Add pre-deploy checks |

#### Alternatives

| Tool | Use When | Avoid When |
|---|---|---|
| GitHub Actions | GitHub-hosted, free for public repos | Need self-hosted runners |
| GitLab CI | GitLab ecosystem | GitHub ecosystem |
| Jenkins | Full control, self-hosted | Want managed simplicity |
| CircleCI | Fast builds, good caching | Budget constraints |
| GitHub Actions + ArgoCD | Kubernetes deployments | Simple VM deployments |

#### Best Practices

- Run tests in the same environment as production
- Cache dependencies between runs
- Use matrix builds for multi-version testing
- Store secrets in CI/CD settings, not code
- Require PR reviews before merge to main
- Tag images with commit SHA for traceability
- Add rollback capability for every deployment
- Monitor pipeline health (build times, failure rates)

#### Hands-On Practice

1. **Basic:** Create a GitHub Actions workflow that runs pytest
2. **Guided:** Add linting and type checking to the pipeline
3. **Independent:** Build a full CI/CD pipeline with test, build, and deploy stages
4. **Realistic:** Add container testing and health check verification
5. **Challenge:** Add a model validation step that checks prediction quality before deployment

#### Knowledge Check

- What is the difference between CI and CD?
- Why should you test in the same environment as production?
- How does image tagging help with traceability?
- What is a rollback strategy and why is it essential?

#### Exit Criteria

- You can create a CI/CD pipeline that tests, builds, and deploys
- You can explain environment promotion and rollback
- You can debug common pipeline failures

#### Next Step:** Monitor and observe your deployed system.

---

### Unit 16.5 — Monitoring & Observability

#### What is it?

Monitoring is the practice of collecting, analyzing, and acting on data about your running systems. Observability extends this by asking: given what I can see from the outside, can I understand what is happening inside? This includes logs, metrics, and traces.

#### Why does it matter?

A deployed model can silently degrade. Data drifts, latency increases, error rates creep up, costs explode, and users suffer — all without anyone noticing. Monitoring turns invisible problems into visible alerts.

#### Why learn it here?

Your system is deployed and running. Now you need to know if it is healthy, fast, and accurate. Without monitoring, you are flying blind.

#### Prerequisites

- A deployed API service (16.1-16.3)
- Basic understanding of HTTP and logs
- Familiarity with dashboards

#### Mental Model

Monitoring is a car dashboard. The speedometer (latency), fuel gauge (cost), temperature gauge (error rate), and check engine light (drift alert) tell you what is happening inside the engine without opening the hood. Logs are the detailed mechanic's report.

#### Core Concepts

- Three pillars: logs, metrics, traces
- Structured logging (JSON logs)
- Key metrics: latency, throughput, error rate, saturation
- ML-specific metrics: prediction distribution, input drift, model quality proxy
- Alerting rules and thresholds
- Dashboarding (Grafana, Datadog, CloudWatch)
- Health checks and uptime monitoring
- Cost tracking

#### How It Works

1. Instrument your code to emit logs, metrics, and traces
2. Collect them in a centralized system
3. Visualize on dashboards
4. Set up alerts for anomalies
5. Respond to incidents and improve

#### Syntax & Implementation

```python
import time
import logging
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, generate_latest
import numpy as np

app = FastAPI()

# Metrics
REQUEST_COUNT = Counter("api_requests_total", "Total requests", ["endpoint", "status"])
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Request latency", ["endpoint"])

# Structured logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    REQUEST_COUNT.labels(endpoint=request.url.path, status=response.status_code).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)

    logger.info({
        "event": "request",
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round(duration * 1000, 2)
    })
    return response

@app.get("/metrics")
def metrics():
    return generate_latest()
```

#### Simple Example

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Model loaded", extra={"model_version": "1.0", "features": 42})
logger.error("Prediction failed", extra={"error": "missing feature", "input": "..."})
```

#### Real-World Example

A production ML service monitors: request rate, p50/p95/p99 latency, error rate by type, prediction distribution (mean, std, percentiles), input feature distributions, data drift scores (PSI/KS test), model accuracy proxy (user feedback), GPU/CPU utilization, and API costs. Dashboards show trends, and alerts fire when drift exceeds thresholds or latency spikes.

#### Common Mistakes

- Logging PII or secrets in production
- Not monitoring prediction distributions
- Setting alerts too sensitively (alert fatigue)
- Not correlating logs with request IDs
- Monitoring only infrastructure, not ML quality
- Not tracking costs

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| No alerts fire during outage | Missing health check monitoring | Check alert rules | Add uptime monitoring |
| Too many alerts (fatigue) | Thresholds too sensitive | Review alert history | Adjust thresholds, use severity levels |
| Cannot find relevant logs | Unstructured logging | Check log format | Switch to structured JSON logs |
| Latency spike not caught | No percentile monitoring | Check metric collection | Add p95/p99 latency metrics |
| Drift not detected | No input monitoring | Check monitoring coverage | Add feature distribution tracking |

#### Alternatives

| Tool | Use When | Avoid When |
|---|---|---|
| Prometheus + Grafana | Open-source, Kubernetes-native | Need managed service |
| Datadog | Full observability, managed | Cost concerns |
| CloudWatch | AWS ecosystem | Multi-cloud |
| OpenTelemetry | Vendor-neutral instrumentation | Simple single-service setups |
| Evidently AI | ML-specific monitoring | Non-ML systems |

#### Best Practices

- Monitor the three pillars: logs, metrics, traces
- Use structured logging (JSON format)
- Track ML-specific metrics: prediction distribution, drift, accuracy proxy
- Set up alerts with severity levels
- Correlate logs with request IDs
- Monitor costs alongside performance
- Review and update monitoring quarterly
- Test your monitoring (chaos engineering for critical systems)

#### Hands-On Practice

1. **Basic:** Add structured logging to your FastAPI service
2. **Guided:** Add Prometheus metrics for latency, error rate, and throughput
3. **Independent:** Build a Grafana dashboard showing key metrics
4. **Realistic:** Add data drift detection using Evidently AI
5. **Challenge:** Set up alerts for latency spikes, error rate increases, and drift, then trigger them intentionally

#### Knowledge Check

- What are the three pillars of observability?
- Why should you monitor prediction distributions, not just infrastructure?
- How does a request ID help with debugging?
- What is data drift and how would you detect it?

#### Exit Criteria

- You can add structured logging and metrics to a service
- You can build a monitoring dashboard
- You can set up alerts for common failure modes
- You can explain the difference between infrastructure and ML monitoring

#### Next Step:** Apply specialized practices for LLM-based systems.

---

### Unit 16.6 — LLMOps

#### What is it?

LLMOps applies MLOps principles specifically to large language model systems. It covers prompt management, model/dataset versioning, cost and latency monitoring, evaluation tracking, safety guardrails, and the unique challenges of working with generative AI.

#### Why does it matter?

LLM systems differ from traditional ML: they depend on prompts (not just features), costs scale with token usage, outputs are non-deterministic, and safety risks are new (hallucination, prompt injection, harmful content). Standard MLOps is necessary but not sufficient.

#### Why learn it here?

After understanding general monitoring and deployment, you need specialized practices for LLM-specific concerns: prompt versioning, cost tracking, safety monitoring, and evaluation pipelines.

#### Prerequisites

- ML deployment basics (16.1-16.5)
- Understanding of LLMs and prompts (earlier phases)
- Basic understanding of API-based LLM services

#### Mental Model

LLMOps is like managing a fleet of taxis. Each taxi (LLM call) has a fuel cost (tokens), a driver (prompt), passengers (inputs), and destinations (outputs). You need to track cost per ride, passenger satisfaction (quality), driver behavior (safety), and vehicle maintenance (model updates).

#### Core Concepts

- Prompt versioning and management
- Token-level cost tracking and optimization
- Latency monitoring (time to first token, total time)
- Output quality evaluation (human and automated)
- Safety guardrails (content filtering, prompt injection detection)
- Model version management
- Caching and semantic deduplication
- Evaluation datasets and benchmarking
- A/B testing for prompts and models
- Audit logging for compliance

#### How It Works

1. Version all prompts, models, and datasets
2. Track token usage and costs per request/user/model
3. Monitor latency (TTFT, total, throughput)
4. Evaluate output quality periodically
5. Apply safety filters on inputs and outputs
6. Cache repeated queries
7. Compare prompt/model variants with A/B tests

#### Syntax & Implementation

```python
import time
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class LLMRequest:
    prompt_version: str
    model: str
    user_input: str
    max_tokens: int = 1000
    temperature: float = 0.7

@dataclass
class LLMResponse:
    output: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd: float
    cached: bool
    safety_filtered: bool

class LLMTracker:
    def __init__(self, cost_per_1k_input: float, cost_per_1k_output: float):
        self.cost_input = cost_per_1k_input / 1000
        self.cost_output = cost_per_1k_output / 1000
        self.cache = {}

    def make_request(self, req: LLMRequest) -> LLMResponse:
        # Check cache
        cache_key = hashlib.md5(f"{req.prompt_version}:{req.user_input}".encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            cached["cached"] = True
            return LLMResponse(**cached)

        start = time.time()
        # Call LLM API (placeholder)
        output = "Placeholder response"
        input_tokens = len(req.user_input.split()) * 4  # rough estimate
        output_tokens = len(output.split()) * 4
        latency = (time.time() - start) * 1000

        cost = (input_tokens * self.cost_input) + (output_tokens * self.cost_output)

        response = LLMResponse(
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency, 2),
            cost_usd=round(cost, 6),
            cached=False,
            safety_filtered=False
        )

        self.cache[cache_key] = asdict(response)
        return response
```

#### Simple Example

```python
# Token cost calculation
def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = input_tokens * 0.000003   # $3 per 1M input tokens
    output_cost = output_tokens * 0.000015  # $15 per 1M output tokens
    return round(input_cost + output_cost, 6)
```

#### Real-World Example

A customer support chatbot tracks: prompt version v2.3 with model gpt-4o, cost per conversation ($0.02 avg), response quality (human rating 4.2/5), safety incidents (0.1% of requests flagged), latency (p95: 2.1s), and cache hit rate (35%). Weekly reports compare prompt variants and model versions.

#### Common Mistakes

- Not versioning prompts (impossible to reproduce results)
- Ignoring token costs until the bill arrives
- Not monitoring hallucination rates
- Using production models for every request (expensive)
- Not caching repeated queries
- Treating all prompts as equal quality

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Cost suddenly spikes | Prompt change or traffic increase | Check token usage logs | Review prompt changes, add rate limits |
| Response quality dropped | Model version changed or prompt drift | Compare versions | Pin model versions, A/B test prompts |
| Latency increased | Longer prompts or model overload | Check TTFT and token counts | Optimize prompts, add caching |
| Safety incidents increased | New prompt injection vectors | Review flagged requests | Add input validation, update guardrails |
| Cache hit rate low | Poor cache key design | Inspect cache keys | Use semantic hashing instead of exact match |

#### Best Practices

- Version every prompt, model, and dataset
- Track costs per request, user, and feature
- Monitor latency at token level (TTFT, total)
- Evaluate quality with automated metrics and periodic human review
- Apply safety filters on both input and output
- Cache aggressively with semantic deduplication
- Use cheaper models for simple tasks (model routing)
- Maintain evaluation datasets for regression testing
- Log every LLM interaction for audit and improvement

#### Hands-On Practice

1. **Basic:** Build a token cost tracker for API calls
2. **Guided:** Add prompt versioning and caching to an LLM service
3. **Independent:** Build an evaluation pipeline that scores LLM outputs
4. **Realistic:** Add safety guardrails and cost alerts
5. **Challenge:** Implement A/B testing for two prompt variants with statistical significance testing

#### Knowledge Check

- Why is prompt versioning critical for reproducibility?
- How would you detect and prevent prompt injection?
- What is the difference between caching exact matches vs semantic caching?
- How would you set a monthly budget alert for LLM costs?

#### Exit Criteria

- You can version prompts and track costs
- You can implement caching and safety guardrails
- You can build an evaluation pipeline for LLM outputs
- You can explain the unique challenges of LLMOps vs traditional MLOps

#### Next Step:** Secure the entire production system.

---

### Unit 16.7 — Security in Production

#### What is it?

Production security protects your system, data, and users from unauthorized access, data breaches, prompt injection, dependency vulnerabilities, and operational risks. It is not optional — it is a requirement for any system that handles user data or makes automated decisions.

#### Why does it matter?

A single vulnerability can expose user data, incur legal liability, destroy trust, and cost millions. ML systems add unique attack surfaces: prompt injection, data poisoning, model extraction, and adversarial inputs.

#### Why learn it here?

Your system is deployed and monitored. Now you must harden it. Security must be designed in from the start, not bolted on after a breach.

#### Prerequisites

- Deployed API service (16.1-16.3)
- Understanding of HTTP and authentication basics
- Knowledge of environment variables

#### Mental Model

Security is layers of defense. Like a castle with walls, moats, gates, and guards, each layer stops different threats. No single layer is perfect, but together they make attack expensive and difficult.

```text
Input validation → Authentication → Authorization → Rate limiting → Logging → Encryption → Secrets management
```

#### Core Concepts

- Secrets management (environment variables, secret managers)
- Authentication (API keys, JWT, OAuth)
- Authorization (role-based access, permissions)
- Input validation and sanitization
- Rate limiting and throttling
- Prompt injection prevention
- Dependency vulnerability scanning
- Data privacy (encryption at rest and in transit)
- Audit logging
- CORS configuration
- Container security

#### How It Works

1. Store secrets in environment variables or secret managers (never in code)
2. Authenticate every request (API key or token)
3. Authorize based on roles and permissions
4. Validate and sanitize all inputs
5. Rate limit to prevent abuse
6. Log all security-relevant events
7. Encrypt sensitive data
8. Scan dependencies for vulnerabilities

#### Syntax & Implementation

```python
from fastapi import FastAPI, Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import os
import time

app = FastAPI()

API_KEY = os.getenv("API_KEY")
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Rate limiting
request_counts = {}

async def verify_api_key(api_key: str = Security(API_KEY_HEADER)):
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return api_key

def rate_limit(client_id: str, max_requests: int = 100, window: int = 60):
    now = time.time()
    if client_id not in request_counts:
        request_counts[client_id] = []
    # Remove old entries
    request_counts[client_id] = [t for t in request_counts[client_id] if now - t < window]
    if len(request_counts[client_id]) >= max_requests:
        raise HTTPException(429, "Rate limit exceeded")
    request_counts[client_id].append(now)

@app.post("/predict")
def predict(
    data: dict,
    api_key: str = Depends(verify_api_key),
    client_id: str = "default"
):
    rate_limit(client_id)

    # Input validation
    if "text" not in data:
        raise HTTPException(400, "Missing 'text' field")
    if len(data["text"]) > 10000:
        raise HTTPException(400, "Input too long")

    # Sanitize prompt injection
    dangerous_patterns = ["ignore previous", "system prompt", "reveal instructions"]
    for pattern in dangerous_patterns:
        if pattern in data["text"].lower():
            raise HTTPException(400, "Potentially harmful input detected")

    return {"result": "safe prediction"}
```

#### Simple Example

```python
import os

# Never hardcode secrets
API_KEY = os.getenv("API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

if not API_KEY:
    raise ValueError("API_KEY environment variable is required")
```

#### Real-World Example

A production AI service uses API key authentication, role-based access (admin vs user), input validation with length limits, rate limiting per client, prompt injection detection, encrypted data at rest, dependency scanning in CI, audit logging of all predictions, and quarterly security reviews.

#### Common Mistakes

- Hardcoding secrets in source code
- No rate limiting (open to abuse)
- Not validating input length or type
- Exposing detailed error messages to clients
- Not scanning dependencies for CVEs
- Running containers as root
- Storing PII in logs

#### Debugging / Troubleshooting

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| 401 Unauthorized on valid request | Wrong API key header name | Check header name in client | Use correct header (`X-API-Key`) |
| Rate limit triggered too early | Shared client ID across users | Check rate limit key | Use per-user or per-API-key limiting |
| Prompt injection succeeds | Missing input sanitization | Test with injection patterns | Add pattern detection and input validation |
| Secret appears in logs | Unsafe logging of request data | Search log files | Redact sensitive fields before logging |
| Dependency scan fails | Known CVE in package | Check vulnerability report | Update package or find alternative |

#### Alternatives

| Approach | Use When | Avoid When |
|---|---|---|
| API keys | Simple service-to-service auth | Need user-level authorization |
| JWT | Stateless auth with expiry | Need revocation capability |
| OAuth2 | Third-party access, user consent | Internal service only |
| mTLS | Zero-trust, service mesh | Simple deployments |
| AWS Cognito / Auth0 | Managed auth service | Full control needed |

#### Best Practices

- Never hardcode secrets — use env vars or secret managers
- Authenticate every request
- Implement rate limiting per client
- Validate and sanitize all inputs
- Scan dependencies for vulnerabilities regularly
- Run containers as non-root
- Encrypt data at rest and in transit
- Log security events (failed auth, injection attempts)
- Review and update security practices quarterly
- Have an incident response plan

#### Hands-On Practice

1. **Basic:** Move hardcoded secrets to environment variables
2. **Guided:** Add API key authentication to your FastAPI service
3. **Independent:** Implement rate limiting and input validation
4. **Realistic:** Add prompt injection detection and audit logging
5. **Challenge:** Set up dependency scanning and container security checks in CI

#### Knowledge Check

- Why should you never store secrets in source code?
- What is the difference between authentication and authorization?
- How would you detect a prompt injection attack?
- What should you do when a dependency vulnerability is discovered?

#### Exit Criteria

- You can secure an API with authentication and rate limiting
- You can validate inputs and prevent common attacks
- You can manage secrets safely
- You can set up dependency scanning and audit logging

#### Next Step:** Synthesize all deployment skills into a complete project.

---

### Unit 16.8 — Deployment Synthesis & Review

#### What is it?

A cumulative integration unit that combines API design, model serving, Docker, CI/CD, monitoring, LLMOps, and security into a single coherent deployment workflow.

#### Why does it matter?

Knowing isolated deployment skills is not enough. You must build a complete, production-ready system from scratch — making decisions about architecture, trade-offs, and priorities along the way.

#### Why learn it here?

All previous units provide the building blocks. This unit forces you to use them together, revealing gaps in understanding and building confidence for real-world deployment.

#### Prerequisites

- All units 16.1-16.7
- A trained model from a previous phase

#### Mental Model

Deployment is a chain. Every link — API, serving, container, CI/CD, monitoring, security — must hold. This synthesis proves you can build the entire chain, not just individual links.

#### Mini Project — Deployed ML/AI Service

**Objective:** Deploy one previous project as a reliable, production-ready API service with full operational capabilities.

**Problem Statement:** Take a model you built in a previous phase and turn it into a service that a team could actually use: reliable, observable, secure, and maintainable.

**Requirements:**

- FastAPI endpoint with Pydantic validation
- Model loading at startup (not per request)
- Preprocessing pipeline packaged with model
- Health check endpoint (`/health`)
- Dockerfile with health check and non-root user
- Docker Compose setup (optional: add Redis/postgres)
- CI pipeline: lint, test, build
- Structured logging (JSON format)
- Prometheus metrics (latency, error rate, throughput)
- API key authentication
- Rate limiting
- Input validation and length limits
- README with setup, usage, and architecture decisions
- Sample requests (curl or Python client)

**Suggested Architecture:**

```text
Client → API Key check → Rate limiter → Input validation → Preprocessing → Model → Postprocessing → Response
          ↓                                              ↓                  ↓
     Auth logs                                    Request logs          Metrics
          ↓                                              ↓                  ↓
              Structured logs → Centralized logging → Dashboard + Alerts
```

**Milestones:**

1. Working API with model serving (16.1, 16.2)
2. Dockerized with health check (16.3)
3. CI pipeline with tests (16.4)
4. Monitoring and logging (16.5)
5. Security hardening (16.7)

**Expected Output:**

- Working local service with `docker run`
- API documentation (auto-generated by FastAPI)
- Sample request/response examples
- Monitoring dashboard or metrics endpoint
- Security checklist
- README explaining architecture and decisions

**Evaluation Criteria:**

- Reproducible build (works from clean clone)
- Correct predictions matching notebook results
- Safe error messages (no internal details leaked)
- Test coverage for key paths (predict, health, auth)
- Structured logs with request correlation
- Metrics for latency, throughput, and errors
- Documented rollback plan
- Security: secrets not in code, auth enforced, rate limiting active

**Failure Cases to Test:**

- Missing API key → 401
- Invalid input type → 400/422
- Input too large → 400
- Rate limit exceeded → 429
- Model artifact missing → clear error on startup
- Malformed request body → validation error

**Advanced Extensions:**

- A/B testing endpoint for model comparison
- Model version endpoint (swap models without restart)
- Batch prediction endpoint
- Async processing with task queue
- Cost tracking per request
- Prompt injection detection (for LLM services)

**Knowledge Check:**

- Why is the model loaded at startup, not per request?
- How would you rollback to a previous model version if predictions degrade?
- What would break if you changed the preprocessing pipeline after deployment?
- How do you ensure the CI pipeline catches a preprocessing mismatch?
- What metrics would you alert on at 2 AM?

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| FastAPI vs Flask | New Python API needing type safety | Simple legacy service | Auto-docs vs simplicity |
| Docker vs bare metal | Need reproducibility and portability | Single-server, controlled environment | Portability vs simplicity |
| GitHub Actions vs Jenkins | GitHub-hosted, open source | Self-hosted, full control | Managed vs customizable |
| Prometheus vs Datadog | Open source, Kubernetes | Managed, full observability | Cost vs features |
| API key vs JWT | Simple service auth | User-level stateless auth | Simplicity vs flexibility |
| Batch vs real-time inference | Non-time-sensitive bulk work | User-facing low-latency needs | Cost/throughput vs latency |

---

## Production Lifecycle Reference

| Concern | What to Track | When to Check |
|---|---|---|
| Data versioning | Which data produced which model | Every training run |
| Model registry | Versions, metrics, owners, promotion status | Every model change |
| Rollback | Ability to return to known-good version | Before every deployment |
| Drift detection | Whether real inputs differ from training data | Daily/weekly automated checks |
| Retraining triggers | When model should be retrained | When drift or quality drops |
| Cost monitoring | API, compute, storage, latency costs | Weekly review |
| Governance | Limitations, risks, approval requirements | Every major change |

---

## Deployment Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Works locally but fails in container | Missing dependency/env var/path | Check container logs | Pin dependencies, configure env, use relative paths |
| First request is very slow | Model loads per request | Trace startup/request time | Load model once at startup |
| Predictions differ from notebook | Preprocessing mismatch | Compare pipeline outputs | Package preprocessing with model pipeline |
| Service silently degrades | No monitoring or drift checks | Inspect production inputs | Add metrics, alerts, data quality checks |
| Secret leaked in logs | Unsafe logging/config | Search logs/config | Redact secrets, rotate credentials, use secret manager |
| CI pipeline is slow | No caching, full rebuild every time | Check build steps | Add dependency and layer caching |
| Container uses too much memory | Model loaded multiple times | Check memory usage | Load once at startup, share across workers |
| Rollback fails | No versioned artifacts | Check artifact registry | Version all artifacts, test rollback procedures |

---

## Phase Review Checklist

- [ ] All units complete.
- [ ] FastAPI service built with validation and error handling.
- [ ] Model served with consistent preprocessing.
- [ ] Docker container built with health check and non-root user.
- [ ] CI/CD pipeline set up with lint, test, build, and deploy stages.
- [ ] Structured logging and metrics implemented.
- [ ] LLMOps practices applied (if LLM service): prompt versioning, cost tracking, caching.
- [ ] Security hardening: authentication, rate limiting, input validation, secrets management.
- [ ] Deployment mini project completed and documented.
- [ ] Rollback plan documented.
- [ ] Cumulative review passed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Build a FastAPI service that validates input and serves predictions.
2. Package a model with preprocessing as a Docker container.
3. Create a CI/CD pipeline that tests, builds, and deploys automatically.
4. Add structured logging, metrics, and dashboards to a service.
5. Implement API key authentication and rate limiting.
6. Version prompts, track costs, and cache LLM responses (for LLM services).
7. Explain rollback strategies and incident response procedures.
8. Debug container failures, prediction mismatches, and pipeline issues.

## Interview / Explain-Back Questions

- What changes when a model moves from notebook to production?
- How do you prevent preprocessing mismatch between training and serving?
- What should you monitor for an ML service?
- How do you handle model drift?
- How do secrets and rate limits affect AI systems?
- What is the difference between CI and CD, and why do you need both?
- How would you implement a rollback for a model that degrades in production?
- What are the unique security challenges of LLM-based systems?
- How would you reduce LLM API costs without sacrificing quality?
- Explain the three pillars of observability and why each matters.

## Exit Criteria

Move to Phase 17 only when you can independently deploy, monitor, secure, and maintain a production ML/AI service, and explain every architectural decision you made.
