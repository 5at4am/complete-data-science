# Phase 04 — Data Analysis & Preparation

> **Goal:** Learn to explore, clean, and prepare data for machine learning — including the critical topic of data leakage.

**Difficulty:** 🟡 Intermediate  
**Priority:** Essential  
**Prerequisites:** Phase 01, Phase 03  
**Mastery target:** Level 5 — decision making for data preparation

---

## Why This Phase Exists

Data preparation is where ML projects succeed or fail. The best model cannot fix leaked data, wrong splits, or silent preprocessing errors. This phase teaches the habits that prevent the most common and dangerous ML failures.

### Phase Mental Model

Data preparation is a pipeline with strict ordering:

```text
Raw data → Exploration → Cleaning → Missing values → Outliers
    ↓
Feature engineering → Scaling/Encoding → Split → Preprocessing fit on train only
    ↓
Train set → Validation set → Test set (never touched until final evaluation)
```

**The Golden Rule:** Fit preprocessing on training data only. Apply to validation/test.

### What This Phase Prepares For

- classical ML pipelines in Phase 05
- deep learning data pipelines in Phase 06
- NLP text preprocessing in Phase 07
- RAG document ingestion in Phase 11
- production data pipelines in Phase 16

---

## Units

### Unit 04.1 — Data Exploration & EDA

**What is it?**  
Exploratory Data Analysis (EDA) is the systematic investigation of a dataset to understand its structure, quality, and patterns before modeling.

**Why does it matter?**  
EDA reveals problems (missing values, outliers, leakage, imbalance) and opportunities (features, interactions, simple baselines).

**Prerequisites:** Pandas, Matplotlib, descriptive statistics.

**Mental Model:**  
EDA is detective work. You are looking for clues about what the data can and cannot tell you.

**Core Concepts:**

- data types and schema
- summary statistics per column
- missing value patterns
- distributions and skewness
- correlations and pairwise relationships
- target distribution and class balance
- time-based patterns (if temporal)
- data quality flags

**Implementation:** Full EDA on a dataset.

**Simple Example:**

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data.csv")

# Structure
print(df.info())
print(df.describe(include="all"))

# Target balance
print(df["target"].value_counts(normalize=True))

# Missing
print(df.isnull().sum())

# Correlations
sns.heatmap(df.select_dtypes(include="number").corr(), annot=True)
```

**Common Mistakes:**

- skipping EDA entirely
- only looking at head/tail
- ignoring the target distribution
- not checking for data leakage indicators (future data in features)
- treating EDA as a one-time step instead of iterative

**Hands-On Practice:**

1. Basic: run a standard EDA checklist on a new dataset.
2. Guided: find three data quality issues in a messy dataset.
3. Independent: create an automated EDA report function.
4. Challenge: detect a subtle leakage feature (e.g., ID that encodes target).

**Exit Criteria:**

- You can systematically explore any tabular dataset.
- You can identify data quality issues before modeling.

**Next Step:** Clean the problems you found.

---

### Unit 04.2 — Data Cleaning

**What is it?**  
Data cleaning fixes structural problems: inconsistent formats, duplicates, typos, wrong types, and encoding issues.

**Why does it matter?**  
Garbage in, garbage out. Models learn patterns — including errors.

**Mental Model:**  
Cleaning is standardization. Make the same thing look the same everywhere.

**Core Concepts:**

- type conversion
- string normalization (case, whitespace, encoding)
- duplicate detection and removal
- inconsistent categories (e.g., "NY", "Ny", "New York")
- date/time parsing
- unit standardization

**Implementation:** Clean a messy dataset.

**Simple Example:**

```python
# Standardize categories
df["state"] = df["state"].str.strip().str.upper()
df["state"] = df["state"].replace({"NY": "NY", "NEW YORK": "NY"})

# Parse dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove exact duplicates
df = df.drop_duplicates()
```

**Common Mistakes:**

- cleaning test data differently from train data
- losing information during cleaning (e.g., dropping rows instead of fixing)
- not documenting cleaning decisions
- cleaning before splitting (leakage!)

**Decision Guidance: Drop vs Fix**

| Drop When | Fix When |
|---|---|
| Row is completely empty | Few columns have issues |
| Duplicate is exact | Inconsistency is systematic |
| Fixing would require guessing | Pattern is clear and automatable |

**Hands-On Practice:**

1. Basic: clean a dataset with mixed types and duplicates.
2. Guided: standardize inconsistent categorical values.
3. Independent: write a cleaning pipeline that logs every change.
4. Realistic: clean a dataset where 20% of rows have at least one issue.

**Exit Criteria:**

- You can identify and fix common data quality issues.
- You can write a reproducible cleaning pipeline.

**Next Step:** Handle missing values systematically.

---

### Unit 04.3 — Missing Values

**What is it?**  
Missing values are absent data points. How you handle them affects model performance and validity.

**Why does it matter?**  
Ignoring missingness biases results. Wrong imputation creates fake patterns.

**Mental Model:**  
Missingness is information. The pattern of missingness often relates to the target.

**Core Concepts:**

- MCAR (Missing Completely At Random)
- MAR (Missing At Random)
- MNAR (Missing Not At Random)
- deletion: listwise, pairwise
- imputation: mean/median/mode, model-based, iterative
- missingness indicators
- domain-specific imputation

**Implementation:** Handle missing data.

**Simple Example:**

```python
from sklearn.impute import SimpleImputer, IterativeImputer

# Simple imputation (fit on train only!)
imputer = SimpleImputer(strategy="median")
X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)

# Missingness indicator
X_train["feature_was_missing"] = X_train["feature"].isnull().astype(int)
```

**Decision Guidance: Missing Data Strategy**

| Situation | Strategy |
|---|---|
| MCAR, < 5% missing | Drop rows |
| MAR, numeric | Median imputation + indicator |
| MAR, categorical | Mode imputation + "missing" category |
| MNAR | Model-based imputation + indicator |
| High missingness (> 50%) | Consider dropping feature |

**Common Mistakes:**

- imputing before splitting (leakage!)
- using mean for skewed data
- not adding missingness indicators
- imputing test data with test statistics
- treating "missing" as a single category without analysis

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Model performs worse after imputation | Imputation adds noise | Compare with complete-case analysis | Try indicator + model-based imputation |
| Validation score much lower than train | Imputer fit on full data | Check fit/transform order | Fit on train, transform all |

**Hands-On Practice:**

1. Basic: apply different imputation strategies.
2. Guided: diagnose missingness mechanism (MCAR/MAR/MNAR).
3. Independent: build a pipeline with imputation + indicator.
4. Challenge: handle missingness in a time-series context.

**Exit Criteria:**

- You can diagnose missingness type.
- You can implement leakage-free imputation.

**Next Step:** Detect and handle outliers.

---

### Unit 04.4 — Outliers

**What is it?**  
Outliers are observations that deviate markedly from the rest of the data. They can be errors or genuine extremes.

**Why does it matter?**  
Outliers distort means, variances, correlations, and model fits — especially for linear models and distance-based methods.

**Mental Model:**  
An outlier is a question: is this a measurement error, a rare event, or a different population?

**Core Concepts:**

- detection: IQR, z-score, isolation forest, visualization
- impact assessment
- treatment: cap/winsorize, transform, remove, model robustly
- domain context

**Implementation:** Detect and handle outliers.

**Simple Example:**

```python
# IQR method
Q1 = df["feature"].quantile(0.25)
Q3 = df["feature"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df["feature"] < lower) | (df["feature"] > upper)]

# Winsorization
df["feature_capped"] = df["feature"].clip(lower, upper)
```

**Decision Guidance: Outlier Treatment**

| Situation | Treatment |
|---|---|
| Clear data error | Fix or remove |
| Genuine extreme, linear model | Winsorize or transform (log) |
| Genuine extreme, tree model | Keep (trees are robust) |
| Fraud/anomaly detection | Keep — it's the signal! |

**Common Mistakes:**

- removing outliers blindly
- using z-score on non-normal data
- treating outliers the same for all model types
- not checking if outlier is the target (e.g., fraud)

**Hands-On Practice:**

1. Basic: detect outliers with IQR and z-score.
2. Guided: compare model performance with/without outlier treatment.
3. Independent: build an outlier detection pipeline.
4. Realistic: decide outlier strategy for a fraud dataset.

**Exit Criteria:**

- You can detect outliers using multiple methods.
- You can choose appropriate treatment based on model and context.

**Next Step:** Scale and encode features correctly.

---

### Unit 04.5 — Feature Scaling & Encoding

**What is it?**  
Scaling puts numeric features on comparable scales. Encoding converts categorical features to numbers.

**Why does it matter?**  
Many algorithms (gradient descent, distance-based, regularization) require scaled features. Models cannot read raw categories.

**Mental Model:**  
Scaling = fair comparison. Encoding = translation for the model.

**Core Concepts:**

- standardization (z-score): mean=0, std=1
- normalization (min-max): [0, 1]
- robust scaling: median/IQR
- one-hot encoding
- label encoding (ordinal only)
- target encoding (with CV to avoid leakage)
- frequency encoding
- embedding for high-cardinality

**Implementation:** Scale and encode features.

**Simple Example:**

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

numeric_features = ["age", "income"]
categorical_features = ["city", "education"]

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# Fit on train only!
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
```

**Decision Guidance: Scaling**

| Model | Scaling Needed? | Preferred |
|---|---|---|
| Linear/Logistic Regression | Yes | Standardization |
| SVM | Yes | Standardization |
| KNN | Yes | Standardization |
| Neural Networks | Yes | Standardization |
| Tree-based (RF, XGBoost) | No | Not needed |
| Naive Bayes | No | Not needed |

**Decision Guidance: Encoding**

| Cardinality | Encoding |
|---|---|
| Low (< 10) | One-hot |
| Medium (10–100) | Target encoding (with CV) or frequency |
| High (> 100) | Embedding or hashing |

**Common Mistakes:**

- fitting scaler/encoder on full dataset (leakage!)
- label encoding nominal categories
- one-hot encoding high-cardinality features
- target encoding without CV (leakage!)
- not handling unknown categories in test data

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Validation error on new category | Encoder doesn't handle unknown | Check `handle_unknown` | Use `handle_unknown="ignore"` |
| Features have different scales after scaling | Scaler fit on wrong data | Check fit/transform | Fit on train only |
| Target encoding overfits | No CV in encoding | Check encoding implementation | Use CV target encoding |

**Hands-On Practice:**

1. Basic: apply standardization and one-hot encoding.
2. Guided: implement target encoding with cross-validation.
3. Independent: build a full ColumnTransformer pipeline.
4. Challenge: handle high-cardinality categorical feature.

**Exit Criteria:**

- You can choose and apply appropriate scaling and encoding.
- You can implement leakage-free preprocessing pipelines.

**Next Step:** Split data correctly.

---

### Unit 04.6 — Train/Validation/Test Splits

**What is it?**  
Splitting data into independent sets for training, model selection, and final evaluation.

**Why does it matter?**  
The test set is your final exam. If you peek at it during development, you overestimate performance.

**Mental Model:**  
Train = practice. Validation = coaching feedback. Test = final exam (once!).

**Core Concepts:**

- random split
- stratified split (classification)
- time-based split (temporal data)
- group split (avoid same entity in multiple sets)
- split ratios
- reproducibility (random_state)

**Implementation:** Split data correctly.

**Simple Example:**

```python
from sklearn.model_selection import train_test_split

# Stratified split for classification
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Time series split
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
```

**Common Mistakes:**

- random split for time series data
- not stratifying imbalanced classification
- same group/entity in train and test
- splitting after preprocessing (leakage!)
- using validation set for final reporting

**Decision Guidance: Split Strategy**

| Data Type | Split Method |
|---|---|
| IID tabular | Random stratified |
| Time series | TimeSeriesSplit or expanding window |
| Grouped (users, patients) | GroupKFold |
| Imbalanced classification | Stratified |

**Hands-On Practice:**

1. Basic: perform stratified and time-based splits.
2. Guided: demonstrate leakage from splitting after preprocessing.
3. Independent: implement GroupKFold for user-grouped data.
4. Challenge: design a split for a recommendation system (user-item interactions).

**Exit Criteria:**

- You can choose the correct split strategy for your data.
- You can explain why splitting before preprocessing is critical.

**Next Step:** Understand and prevent data leakage.

---

### Unit 04.7 — Data Leakage (CRITICAL)

**What is it?**  
Data leakage is when information from outside the training dataset is used to create the model, leading to overoptimistic performance estimates.

**Why does it matter?**  
Leakage is the #1 cause of models that work in development but fail in production.

**Mental Model:**  
Leakage is cheating on the exam by seeing the answers beforehand.

**Core Concepts:**

- target leakage: features that contain target information
- train-test leakage: preprocessing fit on full data
- temporal leakage: using future to predict past
- group leakage: same entity in train and test
- feature leakage: derived features that use target

**Implementation:** Demonstrate leakage.

**Simple Example:**

```python
# LEAKAGE EXAMPLE: Scaling before split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # WRONG: uses test data statistics
X_train, X_test = X_scaled[:800], X_scaled[800:]

# CORRECT: Split first, then scale
X_train, X_test = X[:800], X[800:]
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Common Leakage Patterns:**

| Pattern | Example |
|---|---|
| Target in features | "days_since_last_purchase" for churn prediction |
| Future data | Using tomorrow's price to predict today's |
| Preprocessing on full data | Scaling, imputation, encoding fit on all data |
| Group leakage | Same user in train and test |
| Data augmentation leakage | Augmenting before split |

**Debugging:**

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Too-good-to-be-true performance | Leakage | Audit features and pipeline | Remove leaked features, fix pipeline |
| Train ≈ Val but Test << Val | Test leakage or distribution shift | Compare feature distributions | Fix split strategy |
| Feature importance shows ID column | ID encodes target | Check feature engineering | Drop ID columns |

**Hands-On Practice:**

1. Basic: identify leakage in 5 example pipelines.
2. Guided: create a leakage demo and fix it.
3. Independent: audit a previous project for leakage.
4. Challenge: design a leakage-proof pipeline template.

**Exit Criteria:**

- You can identify at least 5 types of leakage.
- You can build a pipeline that is provably leakage-free.

**Next Step:** Synthesize all data preparation skills.

---

### Unit 04.8 — Data Preparation Synthesis & Review

**What is it?**  
A cumulative integration unit building a complete, leakage-free data pipeline.

**Mini Project:** End-to-End Data Pipeline

**Objective:** Build a production-ready data preparation pipeline for a real dataset.

**Requirements:**

- automated EDA report
- cleaning with logged transformations
- missing value handling with indicators
- outlier detection and treatment
- feature scaling and encoding with ColumnTransformer
- correct train/val/test split (stratified or time-based)
- leakage audit checklist
- serialization of fitted preprocessors
- documentation of all decisions

**Suggested Architecture:**

```text
Raw data → EDA → Cleaning → Missing → Outliers → Split → Preprocessing fit on train
                                                              ↓
                                              Save preprocessor → Transform val/test
```

**Expected Output:**

- cleaned and split datasets
- fitted preprocessor object (joblib/pickle)
- EDA report
- leakage audit checklist
- pipeline code

**Evaluation Criteria:**

- zero leakage (verified by audit)
- reproducible from raw data
- handles new categories gracefully
- documented decisions
- preprocessor can be reused in production

**Advanced Extensions:**

- data versioning with DVC
- data quality tests (Great Expectations)
- feature store integration
- automated data drift detection

**Knowledge Check:**

- Why must preprocessing be fit on training data only?
- How do you handle a categorical value in test that wasn't in train?
- What is the difference between MCAR, MAR, and MNAR?
- When would you NOT remove outliers?
- How does target encoding leak data?

**Exit Criteria:**

- You can build a complete, leakage-free data pipeline.
- You can audit any pipeline for leakage.

---

## Required Decision Comparisons

| Comparison | Use First When | Use Second When | Trade-off |
|---|---|---|---|
| Drop rows vs impute | < 5% missing, MCAR | More missing or MAR | Simplicity vs data retention |
| Standardization vs normalization | Outliers present, Gaussian-like | Bounded features, no outliers | Robustness vs bounded range |
| One-hot vs target encoding | Low cardinality | High cardinality | No leakage risk vs compactness |
| Random vs time split | IID data | Temporal data | Simplicity vs realism |
| Simple imputation vs iterative | Few features, simple patterns | Many features, complex patterns | Speed vs accuracy |

---

## Common Failure Cases

| Symptom | Possible Cause | Verify | Fix |
|---|---|---|---|
| Great CV score, terrible production | Leakage or distribution shift | Audit pipeline, check data drift | Fix leakage, add monitoring |
| New category crashes pipeline | Encoder not fitted for unknown | Test with unseen category | `handle_unknown="ignore"` |
| Preprocessing different in prod | Not serialized | Compare train vs prod transforms | Save and load fitted preprocessor |
| Imbalanced classes in val but not train | Non-stratified split | Check class ratios | Use stratified split |
| Feature importance shows leakage | Target-derived feature | Check feature engineering timeline | Remove feature, re-engineer |

---

## Phase Review Checklist

- [ ] All units complete.
- [ ] EDA performed systematically.
- [ ] Data cleaning with logged transformations.
- [ ] Missing values handled with appropriate strategy.
- [ ] Outliers detected and treated contextually.
- [ ] Feature scaling and encoding chosen per model type.
- [ ] Train/val/test split correct for data type.
- [ ] Leakage audit passed.
- [ ] Full pipeline mini project completed.
- [ ] Cumulative review passed.

## Mastery Check

Without following a tutorial, you should be able to:

1. Explore and understand any dataset.
2. Clean messy data with logged transformations.
3. Handle missing values and outliers appropriately.
4. Scale and encode features correctly for the model.
5. Split data without leakage.
6. Recognize data leakage in any pipeline.
7. Build a reusable, production-ready preprocessing pipeline.

## Interview / Explain-Back Questions

- Why is fitting a scaler on the full dataset leakage?
- How do you handle a categorical value in test that wasn't in train?
- What is the difference between MCAR, MAR, and MNAR?
- When would you NOT remove outliers?
- How does target encoding leak data?
- Why use GroupKFold instead of random KFold?

## Exit Criteria

Move to Phase 05 only when you can independently build a complete, leakage-free data pipeline and explain every preprocessing decision.