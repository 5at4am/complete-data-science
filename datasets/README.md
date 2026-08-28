# Datasets

> **Every dataset in this curriculum has a purpose.**
> Datasets are chosen to teach specific concepts, not randomly selected.

---

## Dataset Documentation Format

For each dataset, document:

```text
Dataset: [Name]
Source: [Where it comes from]
Problem: [What problem does it solve?]
Target: [What are we predicting?]
Features: [What are the inputs?]
Size: [Number of rows/columns]
Data types: [Numeric, categorical, text, etc.]
Missing values: [Are there missing values?]
Potential leakage: [Any leakage risk?]
Class distribution: [For classification]
Difficulty: [Easy / Medium / Hard]
Why this dataset? [What concept does it teach?]
```

---

## Dataset Progression

The curriculum uses a progression of dataset difficulty:

```
Tiny synthetic data
    ↓
Small clean datasets
    ↓
Messy datasets
    ↓
Real-world datasets
    ↓
Large datasets
    ↓
Domain-specific datasets
```

---

## Datasets Used

<!-- Add datasets as they are introduced -->

### Synthetic Data (Phase 02-03)

- **Purpose:** Learn math and statistics with controlled data.
- **Source:** Generated with NumPy.
- **Why:** Full control over properties (noise, distribution, outliers).

### Scikit-learn Toy Datasets (Phase 04-05)

- **Purpose:** Learn ML with clean, well-understood data.
- **Examples:** `load_iris`, `load_digits`, `load_boston`, `make_classification`.
- **Why:** Small, clean, well-documented — perfect for learning.

### Real-World Datasets (Phase 05+)

- **Purpose:** Apply ML to realistic data.
- **Examples:** Titanic, Iris, Wine, Breast Cancer, Housing.
- **Why:** Introduce real-world messiness (missing values, imbalance).

### Text Datasets (Phase 07+)

- **Purpose:** Learn NLP.
- **Examples:** IMDB reviews, Spam, Newsgroups.
- **Why:** Real text data for classification.

### LLM/RAG Datasets (Phase 11+)

- **Purpose:** Learn RAG and LLM systems.
- **Examples:** Custom documents, Wikipedia, Q&A pairs.
- **Why:** Grounded generation and retrieval evaluation.

---

## Data Storage

- `datasets/raw/` — Original, unmodified data.
- `datasets/processed/` — Cleaned, transformed data.

Large datasets are gitignored. Use `kagglehub` or `opendatasets` to download them.

---

## Downloading Datasets

### Kaggle
```python
import kagglehub
path = kagglehub.dataset_download("dataset-name")
```

Requires `KAGGLE_USERNAME` and `KAGGLE_KEY` environment variables.

### OpenDatasets
```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/...")
```

### Hugging Face
```python
from datasets import load_dataset
dataset = load_dataset("dataset-name")
```

### Built-in (scikit-learn)
```python
from sklearn.datasets import load_iris
data = load_iris()
```
