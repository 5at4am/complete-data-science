# 02. K-Nearest Neighbors (KNN)

> Difficulty: ⭐⭐☆☆☆ | Importance: ⭐⭐⭐⭐☆
> Math Required: ⭐⭐☆☆☆ | Coding Required: ⭐⭐☆☆☆
> GATE Relevance: ⭐⭐⭐⭐☆ | Interview: ⭐⭐⭐⭐☆ | Industry: ⭐⭐⭐☆☆

---

## 01. Algorithm Overview

| Property | Value |
|---|---|
| Algorithm Name | K-Nearest Neighbors (KNN) |
| Category | Supervised Learning |
| Type | Classification (also Regression with KNN-Regression) |
| Parametric / Non-parametric | Non-parametric |
| Generative / Discriminative | Discriminative (instance-based) |
| Main Objective | Classify a new point by majority vote of its K closest training points |
| Input | Feature matrix X (n × d), labels y, integer K |
| Output | Class label (majority vote among K nearest neighbors) |
| Core Idea | "You are the company you keep" — nearby points likely share the same class |
| Typical Use Cases | Recommendation systems, pattern recognition, anomaly detection, missing data imputation |

---

## 02. One-Line Definition

### Beginner Definition
KNN finds the K most similar data points to a new point and lets them vote on the class.

### Technical Definition
KNN is a non-parametric, instance-based (lazy) learner that classifies a query point by assigning it the majority class label among its K nearest neighbors in the training data, as measured by a distance metric.

---

## 03. Intuition

Imagine you walk into a new city and want to know if a neighborhood is "safe" or "dangerous." You don't have a model or rules — you just look at the K closest houses and ask their residents. If most of the K neighbors say "safe," you conclude the neighborhood is safe.

That's exactly KNN: no training phase, no model building. At prediction time, you look at the K closest training examples and let them vote.

The key insight: **similarity in feature space implies similarity in label.** Points that are close together in the feature space likely belong to the same class.

---

## 04. Problem It Solves

**Problem:** Given a labeled dataset, classify new unseen points based on their similarity to known points.

**Example:** You have photos of animals labeled as "cat" or "dog." A new photo comes in. You extract features (ear length, snout length, etc.) and find the K most similar labeled photos. Majority vote decides.

**Why useful:**
- No training phase needed — works immediately.
- Naturally handles multi-class problems.
- Captures complex, non-linear decision boundaries.
- Intuitive and easy to understand.

---

## 05. Where It Fits in Machine Learning

```
Machine Learning
├── Supervised Learning
│   ├── Classification
│   │   ├── Instance-Based (Lazy Learners)
│   │   │   ├── K-Nearest Neighbors  ◄── YOU ARE HERE
│   │   │   └── Radius-Based Neighbors
│   │   ├── Model-Based (Eager Learners)
│   │   │   ├── Logistic Regression
│   │   │   ├── Naive Bayes
│   │   │   ├── SVM
│   │   │   └── Decision Tree / Ensemble Methods
│   │   └── Neural Networks
│   └── Regression
└── Unsupervised Learning
    └── K-Means (uses distance too, but unsupervised)
```

KNN is unique because it's a **lazy learner** — it doesn't build an explicit model during training. All work is deferred to prediction time.

---

## 06. Important Terminology

| Term | Simple Meaning | Technical Meaning |
|---|---|---|
| Lazy Learner | No training phase | The algorithm stores training data and defers all computation to prediction time |
| Instance-Based | Uses raw data points | Classification is based on similarity to individual training instances |
| K | Number of neighbors | A hyperparameter specifying how many nearest neighbors to consider |
| Distance Metric | How "close" is measured | Euclidean, Manhattan, Minkowski, Cosine, etc. |
| Euclidean Distance | Straight-line distance | √(Σ(xᵢ - yᵢ)²) — the L2 norm of the difference vector |
| Manhattan Distance | City-block distance | Σ|xᵢ - yᵢ| — the L1 norm of the difference vector |
| Majority Vote | Most common class | The predicted class is the one that appears most among the K neighbors |
| Weighted Voting | Vote by distance | Closer neighbors get more weight in the vote |
| Curse of Dimensionality | Distance breaks down | In high dimensions, all points become equidistant, reducing KNN effectiveness |

---

## 07. Input and Output

**Input:**
- **Feature matrix X:** n samples × d features (numerical; categorical features must be encoded).
- **Label vector y:** n labels ∈ {0, 1, ..., K-1} for classification.
- **Hyperparameter K:** Number of neighbors (positive integer, usually odd).
- **Distance metric:** Euclidean (default), Manhattan, Minkowski, Cosine, etc.

**Output:**
- **Class label:** Majority vote (or weighted vote) among K nearest neighbors.
- **No learned parameters** — the entire training set IS the model.

---

## 08. Mathematical Foundation

**Core idea:** Classify by proximity. Two points are "similar" if their distance is small.

**Distance metrics:**

**Euclidean Distance (L2 norm):**
```
d(x, y) = √(Σᵢ₌₁ᵈ (xᵢ - yᵢ)²)
```

**Manhattan Distance (L1 norm):**
```
d(x, y) = Σᵢ₌₁ᵈ |xᵢ - yᵢ|
```

**Minkowski Distance (generalization):**
```
d(x, y) = (Σᵢ₌₁ᵈ |xᵢ - yᵢ|ᵖ)^(1/p)
```
where p=1 → Manhattan, p=2 → Euclidean.

**Cosine Similarity:**
```
sim(x, y) = (x · y) / (||x|| · ||y||)
```

**Required math concepts:**
1. Distance/similarity measures
2. Sorting (to find K nearest)
3. Counting (majority vote)

---

## 09. Core Formula

### Formula 1: Euclidean Distance

```
d(x, q) = √(Σᵢ₌₁ᵈ (xᵢ - qᵢ)²)
```

**Meaning:** The straight-line distance between training point x and query point q in d-dimensional space.

**Symbols:**
- x = (x₁, ..., x_d) — a training point
- q = (q₁, ..., q_d) — the query (new) point
- d — number of features

**Intuition:** Measures how far apart two points are. Smaller distance → more similar.

**Example:**
```
x = (3, 4), q = (1, 1)
d = √((3-1)² + (4-1)²) = √(4 + 9) = √13 ≈ 3.606
```

### Formula 2: Majority Vote

```
ŷ = argmax_c Σᵢ∈N_K(q) I(yᵢ = c)
```

**Meaning:** The predicted class is the one that appears most frequently among the K nearest neighbors.

**Symbols:**
- N_K(q) — set of K nearest neighbors of query q
- I(yᵢ = c) — indicator function: 1 if neighbor i has label c, else 0
- c — candidate class label

**Intuition:** Each neighbor "votes" for its class. The class with the most votes wins.

**Example:**
```
K = 5, neighbors' labels = [1, 0, 1, 1, 0]
Class 1 gets 3 votes, Class 0 gets 2 votes.
Predicted class: 1
```

### Formula 3: Weighted Majority Vote

```
ŷ = argmax_c Σᵢ∈N_K(q) wᵢ · I(yᵢ = c)
where wᵢ = 1 / d(xᵢ, q)
```

**Meaning:** Closer neighbors get more voting power (inverse distance weighting).

**Symbols:**
- wᵢ = 1/d(xᵢ, q) — weight inversely proportional to distance
- d(xᵢ, q) — distance from neighbor i to query q

**Intuition:** A very close neighbor's vote should count more than a distant neighbor's vote.

---

## 10. Derivation

KNN does not involve optimization or parameter learning — there is no derivation in the traditional sense.

**However, the choice of K has theoretical backing:**

- As K → 1: The decision boundary becomes very complex (overfitting). Each point forms its own "territory."
- As K → n (all training points): Every point is classified as the majority class of the entire dataset (underfitting).
- Optimal K: Balances bias and variance. Found via cross-validation.

**Distance metric choice rationale:**
- Euclidean distance assumes features are on the same scale → requires normalization.
- Manhattan distance is more robust to outliers (doesn't square large differences).
- Cosine distance is better for high-dimensional sparse data (text).

---

## 11. How the Algorithm Works

```
Training Phase:
  Store all training data (X, y) in memory
  (That's it — no model is built!)

Prediction Phase:
  Given query point q:
       ↓
  1. Compute distance from q to ALL training points
       ↓
  2. Sort distances in ascending order
       ↓
  3. Select K closest training points (neighbors)
       ↓
  4. Count class labels among these K neighbors
       ↓
  5. Predict: majority class (or weighted vote)
```

This is why KNN is called a **lazy learner** — all computation happens at prediction time.

---

## 12. Training Process

**Pre-training:**
- Store the entire training dataset in memory.
- Optionally: build a spatial index (KD-tree, Ball tree) for faster neighbor search.

**During training:**
- Nothing happens. KNN does not learn parameters. The training data IS the model.

**What's learned:**
- Nothing is "learned" in the traditional sense. The model is simply the stored training data.

**Stopping criteria:**
- Not applicable — there's no iterative training process.

**Final model:**
- The stored training set: (X_train, y_train).
- At prediction time, compute distances and vote.

---

## 13. Objective Function / Loss Function

KNN does **not** have an explicit objective function or loss function to optimize during training. It's a non-parametric method.

However, **cross-validation error** serves as the de facto objective for choosing K:
- Try K = 1, 3, 5, 7, ..., up to some limit.
- For each K, compute cross-validation accuracy.
- Choose K with the best cross-validation performance.

**Leave-One-Out Cross-Validation (LOOCV):**
- For each training point, predict its label using the remaining n-1 points.
- LOOCV error = fraction of misclassifications.
- Practical for small datasets.

---

## 14. Optimization

KNN does not use gradient descent or any parameter optimization.

**Optimization happens in the search space:**
- Brute force: compute all n distances → O(n·d) per query.
- KD-tree: organize points in a tree for faster search → O(d·log n) average case.
- Ball tree: better for high dimensions → O(d·log n) average case.

**Choosing K is the "optimization":**
```
For K in [1, 3, 5, 7, 9, ...]:
    accuracy = cross_validate(K)
Select K with best accuracy
```

---

## 15. Complete Numerical Example

**Dataset (2 features, 5 training samples):**

| Sample | x₁ | x₂ | y (Class) |
|--------|-----|-----|-----------|
| A | 2 | 3 | 0 |
| B | 4 | 3 | 0 |
| C | 5 | 5 | 1 |
| D | 7 | 6 | 1 |
| E | 1 | 2 | 0 |

**Query point:** q = (4, 5), K = 3

**Step 1 — Compute Euclidean distances:**

```
d(q, A) = √((4-2)² + (5-3)²) = √(4+4) = √8 = 2.828
d(q, B) = √((4-4)² + (5-3)²) = √(0+4) = √4 = 2.000
d(q, C) = √((4-5)² + (5-5)²) = √(1+0) = √1 = 1.000
d(q, D) = √((4-7)² + (5-6)²) = √(9+1) = √10 = 3.162
d(q, E) = √((4-1)² + (5-2)²) = √(9+9) = √18 = 4.243
```

**Step 2 — Sort by distance:**

| Rank | Point | Distance | Class |
|------|-------|----------|-------|
| 1 | C | 1.000 | 1 |
| 2 | B | 2.000 | 0 |
| 3 | A | 2.828 | 0 |
| 4 | D | 3.162 | 1 |
| 5 | E | 4.243 | 0 |

**Step 3 — Select K=3 nearest:** C, B, A

**Step 4 — Majority vote:**
- Class 1: C (1 vote)
- Class 0: B, A (2 votes)

**Predicted class: 0** (2 out of 3 neighbors are class 0)

**VERIFIED EXAMPLE** — distances and vote hand-computed.

---

## 16. Visual Explanation

### Decision Boundary with K=1 vs K=5

```
K=1 (complex boundary):           K=5 (smooth boundary):

  ○ ○ ○ ○                          ○ ○ ○ ○ ○ ○
  ○ ○ ○●○ ○                        ○ ○ ○○●○ ○ ○
  ○ ○○○ ○○                         ○ ○○○○ ○○○
  ●●●○○○○○                         ●●●●○○○○○
  ●●●●●○○○                         ●●●●●●○○○
  ●●●●●●●○                         ●●●●●●●●○

  ● = Class 0                       ● = Class 0
  ○ = Class 1                       ○ = Class 1

  K=1: very jagged boundary         K=5: smoother boundary
  (high variance, low bias)         (low variance, high bias)
```

### Finding K Nearest Neighbors

```
         q (query)
          ★
        / | \
       /  |  \      ← distances to neighbors
      /   |   \
     A    B    C    ← K=3 nearest neighbors
     ↓    ↓    ↓
    class class class
     0     0     1
         ↓
    Majority: Class 0
```

---

## 17. Algorithm / Pseudocode

```
ALGORITHM: K-Nearest Neighbors (KNN) Classification

1. INPUT: Training data X (n×d), y (n×1), query point q, hyperparameter K
2. TRAINING: Store (X, y) in memory
3. PREDICTION:
   a. FOR each training point x_i:
      i.  Compute distance d(q, x_i)
   b. Sort all distances in ascending order
   c. Select K points with smallest distances
   d. Count class labels among these K points
   e. Predict: ŷ = majority class
4. RETURN ŷ
```

---

## 18. From-Scratch Implementation

```python
import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                               n_informative=2, random_state=42, n_clusters_per_class=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = KNNClassifier(k=5)
    model.fit(X_train, y_train)
    print(f"Train accuracy: {model.score(X_train, y_train):.4f}")
    print(f"Test accuracy:  {model.score(X_test, y_test):.4f}")
```

---

## 19. Code Explanation

```
KNNClassifier class:
  __init__          → stores K (number of neighbors)
  fit               → stores training data in memory (NO computation)
  _euclidean_distance → computes √(Σ(x₁-x₂)²)
  _predict_single   → for one query point:
                      1. compute distance to all training points
                      2. find K smallest distances (argsort + slice)
                      3. get labels of K nearest
                      4. majority vote (Counter.most_common)
  predict           → applies _predict_single to each test point
  score             → computes accuracy
```

---

## 20. Library Implementation

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

param_grid = {'knn__n_neighbors': [3, 5, 7, 9, 11, 15]}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid.fit(X_train, y_train)

print(f"Best K: {grid.best_params_['knn__n_neighbors']}")
print(f"Test accuracy: {grid.score(X_test, y_test):.4f}")
```

**Key parameters:**
- `n_neighbors`: K (default 5).
- `weights`: 'uniform' (equal vote) or 'distance' (inverse distance weighting).
- `metric`: 'euclidean' (default), 'manhattan', 'minkowski', etc.
- `algorithm`: 'auto', 'ball_tree', 'kd_tree', 'brute'.

---

## 21. Hyperparameters

| Hyperparameter | Meaning | Effect | Typical Consideration |
|---|---|---|---|
| K (n_neighbors) | Number of neighbors to vote | Small K → complex boundary (overfit). Large K → smooth boundary (underfit) | Try odd values to avoid ties; cross-validate |
| weights | 'uniform' or 'distance' | 'distance' gives more influence to closer neighbors | 'distance' often better; try both |
| metric | Distance function | Euclidean for general use; Manhattan for sparse/high-dim | Use cosine for text data |
| algorithm | Search algorithm | 'auto' picks best; 'brute' for high dimensions | 'brute' if d > 20 |
| p | Minkowski power | p=1 → Manhattan; p=2 → Euclidean | Tune if using Minkowski |

---

## 22. Parameters vs Hyperparameters

### Parameters (learned)
- **None.** KNN is non-parametric — it has no learned parameters. The training data itself is the model.

### Hyperparameters (chosen)
- **K:** Number of neighbors.
- **weights:** Uniform or distance-weighted voting.
- **metric:** Distance function used.
- **algorithm:** Search method (brute, KD-tree, Ball tree).

---

## 23. Assumptions

| Assumption | What It Means | How to Check | If Violated | Solution |
|---|---|---|---|---|
| Similarity implies same class | Nearby points share labels | Visualize data; check if clusters exist | Model performs randomly | Use a model-based classifier (e.g., SVM, neural network) |
| Features are on similar scales | No feature dominates distance | Check feature ranges | One feature dominates distance | Standardize/normalize features |
| Enough data | Dense coverage of feature space | Check if test points have nearby training points | Poor generalization | Collect more data; reduce dimensionality |
| Low dimensionality | Distance is meaningful | Check d vs n | Curse of dimensionality | Feature selection, PCA |

**Note:** These are soft assumptions. KNN works without them but degrades.

---

## 24. Data Requirements

| Aspect | Requirement |
|---|---|
| Data type | Numerical (encode categorical before use) |
| Missing values | Not handled — must impute or remove |
| Outliers | Sensitive — outliers can dominate nearest neighbors |
| Scaling | **Critical** — must standardize features (different scales distort distances) |
| Feature engineering | Important — remove irrelevant features (they add noise to distance) |
| Dataset size | Needs sufficient data to have nearby neighbors for test points |
| Class imbalance | Can be problematic — majority class may dominate the vote |

---

## 25. Feature Scaling

**Status: REQUIRED**

**Why:** KNN uses distance metrics. If one feature ranges from 0–1000 and another from 0–1, the first feature will completely dominate the distance calculation.

**Example:**
```
Without scaling:
  Feature 1 (age):     25 → 50 (range: 25)
  Feature 2 (income):  30000 → 60000 (range: 30000)
  Income dominates distance completely.

With StandardScaler:
  Both features have mean=0, std=1.
  Both contribute equally to distance.
```

**Methods:**
- **StandardScaler** (z-score): recommended for KNN.
- **MinMaxScaler**: useful when you want bounded distances.

---

## 26. Evaluation Metrics

| Metric | Formula | When to Use |
|---|---|---|
| Accuracy | (TP+TN)/Total | Balanced classes |
| Precision | TP/(TP+FP) | Cost of false positive is high |
| Recall | TP/(TP+FN) | Cost of false negative is high |
| F1-Score | 2PR/(P+R) | Imbalanced classes |
| Confusion Matrix | Table of TP, TN, FP, FN | Understanding error patterns |
| AUC-ROC | Area under ROC curve | Model comparison |

**Training Objective ≠ Evaluation Metric:** KNN has no training objective. Cross-validation accuracy is used to choose K.

---

## 27. Advantages

1. **No training phase:** Instant "training" — just store the data.
2. **Simple and intuitive:** Easy to explain and understand.
3. **Naturally multi-class:** No special handling needed.
4. **Non-linear decision boundaries:** Can capture complex patterns.
5. **Adapts to new data:** Adding new training points immediately updates the model.
6. **No assumptions about data distribution:** Works with any shape of data.

---

## 28. Disadvantages

1. **Slow prediction:** Must compute distances to ALL training points → O(n·d) per query.
2. **High memory usage:** Must store entire training set.
3. **Curse of dimensionality:** Distance becomes meaningless in high dimensions.
4. **Sensitive to irrelevant features:** Noise features dilute meaningful distances.
5. **Sensitive to feature scaling:** Must normalize features.
6. **Sensitive to imbalanced data:** Majority class dominates the vote.
7. **No interpretability:** No model to inspect for feature importance.

---

## 29. When to Use

- ✓ Small-to-medium dataset (< 10K samples).
- ✓ Low-to-moderate dimensionality (< 20 features).
- ✓ Non-linear decision boundaries expected.
- ✓ Quick baseline needed (no training time).
- ✓ Data arrives incrementally (easy to add new points).
- ✓ Interpretability through "nearest examples" is valuable.
- ✓ Recommendation systems ("users like you also liked...").

---

## 30. When NOT to Use

- ✗ Large dataset (> 10K samples) — prediction is too slow.
- ✗ High-dimensional data (> 50 features) — curse of dimensionality.
- ✗ Real-time predictions needed (latency constraints).
- ✗ Interpretability through feature weights is needed.
- ✗ Many irrelevant features without feature selection.
- ✗ Imbalanced classes without resampling.

---

## 31. Real-World Applications

1. **Recommendation Systems**
   - Problem: Recommend products/movies to users
   - Input: User rating vectors
   - Algorithm: KNN (find users with similar taste)
   - Output: Items liked by similar users that the target user hasn't seen

2. **Handwritten Digit Recognition**
   - Problem: Classify digits 0–9
   - Input: Pixel intensities of digit images
   - Algorithm: KNN (find similar digit images)
   - Output: Predicted digit

3. **Missing Data Imputation**
   - Problem: Fill in missing values
   - Input: Dataset with missing features
   - Algorithm: KNN Imputer (use K nearest complete records)
   - Output: Imputed values based on neighbor averages

---

## 32. Failure Cases

1. **Data:** Very few training samples — no nearby neighbors exist for test points.
2. **Mathematical:** Curse of dimensionality — all distances converge, making KNN equivalent to random guessing.
3. **Optimization:** No optimization exists to prevent failure — must be handled by preprocessing.
4. **Generalization:** High K with noisy data — model simply predicts the majority class.
5. **Practical:** Prediction time too slow for real-time applications with large datasets.

---

## 33. Overfitting and Underfitting

**Overfitting (K too small, e.g., K=1):**
- Decision boundary is extremely jagged.
- Training accuracy = 100% (each point is its own neighbor).
- Test accuracy is low.

**Underfitting (K too large, e.g., K=n):**
- Every point is classified as the overall majority class.
- Both train and test accuracy are poor.

**Balanced (optimal K):**
- Smooth decision boundary that captures the data structure.
- Found via cross-validation.

---

## 34. Bias-Variance Perspective

**Small K → Low bias, High variance:**
- Model is very flexible (can learn complex patterns).
- Highly sensitive to individual training points (noisy).

**Large K → High bias, Low variance:**
- Model is very rigid (averages over many points).
- Insensitive to individual training points (smooth).

**Optimal K:** Cross-validation finds the K that balances bias and variance.

---

## 35. Comparison With Similar Algorithms

| Algorithm | Main Idea | Strength | Weakness | Best Use |
|---|---|---|---|---|
| KNN | Distance-based vote | Simple, non-linear, no training | Slow prediction, curse of dim | Small datasets, recommendations |
| Logistic Regression | Linear boundary + sigmoid | Fast, interpretable, probabilistic | Linear boundary only | Baseline, interpretable models |
| Naive Bayes | Conditional independence | Very fast, small data | Independence assumption | Text classification |
| SVM | Maximum margin | Strong in high dimensions | Less interpretable | High-dimensional data |
| Decision Tree | Rule-based splits | Handles non-linearity | Prone to overfitting | When interpretability matters |

---

## 36. Algorithm Selection Guide

```
Small dataset (< 1K samples)?
├── YES
│   ├── Need interpretability? → KNN (show nearest examples)
│   ├── Need probability? → Logistic Regression
│   └── Quick baseline? → KNN
└── NO
    ├── Large dataset (> 10K)?
    │   ├── YES → NOT KNN (too slow). Use SVM, Random Forest, or Neural Networks.
    │   └── NO
    │       ├── High dimensions (> 50)?
    │       │   ├── YES → SVM or feature selection + KNN
    │       │   └── NO → KNN works well
    │       └── Non-linear patterns? → KNN or Decision Tree
```

---

## 37. Common Mistakes

```
❌ Not scaling features before using KNN
   Why wrong: Features with larger ranges dominate the distance metric.
   Correct: Always StandardScaler before KNN.

❌ Using K=1
   Why wrong: Extremely sensitive to noise; overfits badly.
   Correct: Use K=3, 5, or 7; tune via cross-validation.

❌ Using KNN on high-dimensional data without dimensionality reduction
   Why wrong: Curse of dimensionality — all points become equidistant.
   Correct: Apply PCA or feature selection first.

❌ Using accuracy for imbalanced classes
   Why wrong: Majority class dominates both distance and vote.
   Correct: Use weighted voting, resample, or use F1-score.

❌ Using KNN for real-time prediction on large datasets
   Why wrong: Prediction requires computing all n distances — too slow.
   Correct: Use approximate nearest neighbor (ANN) methods or switch to a model-based approach.
```

---

## 38. Interview Questions

### Beginner

**Q1: What is KNN and how does it work?**
A: KNN stores all training data. To classify a new point, it finds the K closest training points (by distance) and lets them vote on the class. The majority class wins.

**Q2: Why is KNN called a "lazy learner"?**
A: Because it does no training. All computation is deferred to prediction time. The "model" is just the stored data.

**Q3: What happens when K=1?**
A: The model overfits — each point forms its own decision region. The boundary is extremely jagged and sensitive to noise.

**Q4: Why must we scale features for KNN?**
A: Distance metrics are sensitive to scale. A feature with range 0–1000 would dominate a feature with range 0–1.

**Q5: What distance metrics can KNN use?**
A: Euclidean (L2), Manhattan (L1), Minkowski (generalized), Cosine, etc.

### Intermediate

**Q6: How do you choose the optimal K?**
A: Use cross-validation. Try K = 1, 3, 5, 7, ... and pick the K that maximizes cross-validation accuracy.

**Q7: What is the curse of dimensionality and how does it affect KNN?**
A: In high dimensions, all points become approximately equidistant. The concept of "nearest" loses meaning, degrading KNN performance.

**Q8: What is weighted KNN?**
A: Instead of equal votes, each neighbor's vote is weighted by 1/distance. Closer neighbors have more influence.

**Q9: What is the time complexity of KNN prediction?**
A: O(n·d) for brute-force search (compute distance to all n training points, each with d features).

**Q10: How does KNN handle multi-class problems?**
A: Naturally — just count votes across all K neighbors for all classes and pick the majority.

### Advanced

**Q11: What data structures improve KNN search speed?**
A: KD-trees (efficient for low dimensions, d < 20), Ball trees (better for higher dimensions), and approximate nearest neighbor (ANN) methods like LSH or HNSW.

**Q12: Why does KNN break down in high dimensions?**
A: The volume of the space grows exponentially with dimension. Points become sparse, and the ratio of distance to the nearest vs farthest neighbor approaches 1.

**Q13: Can KNN be used for regression?**
A: Yes — instead of majority vote, average (or weighted average) the K nearest neighbors' target values.

---

## 39. GATE / Exam Perspective

**Key concepts:**
- KNN is a non-parametric, instance-based, lazy learner.
- Distance metrics: Euclidean (L2), Manhattan (L1), Minkowski (Lp).
- Curse of dimensionality: distance becomes meaningless as d increases.
- Feature scaling is critical for KNN.
- Small K → overfitting (high variance); Large K → underfitting (high bias).
- KNN regression: predict the mean of K nearest neighbors.

**Common traps:**
- Confusing KNN (classification) with K-Means (clustering) — they use distance but are fundamentally different.
- Assuming KNN "learns" parameters — it doesn't.
- Forgetting that KNN is computationally expensive at prediction time.

**Key formulas:**
- Euclidean distance: d = √(Σ(xᵢ - yᵢ)²)
- Manhattan distance: d = Σ|xᵢ - yᵢ|

*(The above are representative concept patterns, not past GATE PYQs.)*

---

## 40. Coding Practice

**Level 1 — Basic:**
Implement Euclidean distance from scratch. Verify: d([0,0], [3,4]) = 5.

**Level 2 — Simple KNN:**
Implement KNN classifier from scratch on a 2D dataset. Visualize decision boundary.

**Level 3 — With sklearn:**
Use KNeighborsClassifier on Iris dataset. Compare K=1, 3, 5, 7, 10.

**Level 4 — Feature scaling:**
Compare KNN performance with and without StandardScaler on a dataset with features of different scales.

**Level 5 — Weighted voting:**
Implement weighted KNN (inverse distance). Compare with uniform voting.

**Level 6 — Dimensionality reduction:**
Apply PCA to reduce dimensions, then use KNN. Observe accuracy vs. number of components.

**Level 7 — Real-world case study:**
Build a movie recommendation system using KNN on the MovieLens dataset.

---

## 41. Practical ML Workflow

```
Problem Definition → "Classify handwritten digits"
Data Collection → "MNIST: 70K images, 28×28 pixels"
EDA → "10 classes, roughly balanced"
Cleaning → "No missing values; normalize pixel values to [0,1]"
Feature Engineering → "Flatten 28×28 to 784 features; apply PCA to reduce to 50"
Split → "60K train, 10K test"
Preprocessing → "StandardScaler on features"
Train → "KNeighborsClassifier(n_neighbors=5, weights='distance')"
Tune → "GridSearchCV: K=[3,5,7,11], weights=['uniform','distance']"
Evaluate → "Accuracy: 97.2%, Confusion matrix shows errors on 4/7, 3/8"
Error Analysis → "Confused similar-looking digits; add edge features"
Deploy → "Serve with FAISS for fast approximate nearest neighbor search"
```

---

## 42. Complexity

| Aspect | Complexity |
|---|---|
| Training time | O(1) — just store data |
| Prediction time (brute) | O(n·d) per query |
| Prediction time (KD-tree) | O(d·log n) average, O(n·d) worst case |
| Space | O(n·d) — store entire training set |
| Scaling with n | Prediction gets linearly slower |
| Scaling with d | Both distance computation and curse of dimensionality |
| Scaling with classes | Negligible — just count labels |

---

## 43. Advanced Concepts

1. **KD-Tree:** Binary tree that partitions space by splitting along feature axes at median values. Efficient for low dimensions but degrades for d > 20.

2. **Ball Tree:** Partitions space using hyperspheres. Better than KD-tree for high dimensions.

3. **Locality-Sensitive Hashing (LSH):** Approximate nearest neighbor method. Trades exactness for speed. Used in production systems.

4. **Condensed KNN:** Reduces the training set by keeping only boundary points. Speeds up prediction.

5. **Adaptive K:** Instead of fixed K, use a different K for each query point based on local data density.

---

## 44. Connections to Other Algorithms

```
         KNN
        / | \
       /  |  \
      /   |   \
K-Means  KNN   Radius
(clustering,  Regression  Neighbors
 unsupervised, (regression)  (fixed
 different    variant     radius
 goal)                    instead of K)
```

- **KNN vs K-Means:** KNN is supervised (uses labels), K-Means is unsupervised (no labels). Both use distance, but for different purposes.
- **KNN vs Decision Tree:** Decision Tree builds an explicit model; KNN stores data. Decision Tree is fast to predict; KNN is slow.
- **KNN Regression → KNN Classification:** Same algorithm, different aggregation (mean vs majority vote).

---

## 45. If You Remember Only 5 Things

1. **KNN is a lazy learner** — no training phase, all work at prediction time.
2. **Always scale features** — distance metrics are meaningless without scaling.
3. **K controls the bias-variance tradeoff** — small K = overfit, large K = underfit.
4. **Curse of dimensionality kills KNN** — distance loses meaning in high dimensions.
5. **Prediction is O(n·d)** — slow for large datasets; consider ANN methods.

---

## 46. Cheat Sheet

| Item | Detail |
|---|---|
| Algorithm | K-Nearest Neighbors |
| Category | Supervised, Instance-Based, Lazy Learner |
| Goal | Classify by majority vote of K closest training points |
| Input | Features X, labels y, integer K, distance metric |
| Output | Majority class label among K neighbors |
| Core Formula | ŷ = argmax_c Σ I(yᵢ=c) for i ∈ K nearest neighbors |
| Loss | None (no training objective) |
| Optimization | None (no parameters to optimize) |
| Parameters | None (non-parametric; training data = model) |
| Hyperparameters | K, weights, metric, algorithm |
| Assumptions | Similarity implies same class, features on same scale |
| Advantages | Simple, no training, non-linear, naturally multi-class |
| Disadvantages | Slow prediction, curse of dimensionality, no interpretability |
| Use When | Small data, non-linear boundaries, quick baseline |
| Avoid When | Large data, high dimensions, real-time prediction |
| Related | K-Means, KD-Tree, Radius Neighbors |
| Key Exam Points | Non-parametric, lazy learner, curse of dimensionality |
| Key Interview Points | Choose K, feature scaling, complexity, curse of dimensionality |

---

## 47. Final Mental Model

```
┌──────────────┐         ┌──────────────┐         ┌──────────┐
│ Training Set │────────→│ Compute all  │────────→│ Sort by  │
│ (X, y)       │  query  │ distances    │         │ distance │
│ stored       │  point  │ to query     │         │          │
└──────────────┘         └──────────────┘         └────┬─────┘
                                                       │
                    ┌──────────────┐         ┌──────────┴─────┐
                    │ Predicted    │←────────│ Pick K nearest │
                    │ class label  │  vote   │ neighbors      │
                    └──────────────┘         └────────────────┘
```

---

## 48. Knowledge Check

### Recall (5)

1. What is the time complexity of KNN prediction (brute force)?
2. What distance metric does KNN use by default?
3. Why is KNN called a "lazy learner"?
4. What is the main hyperparameter in KNN?
5. What happens when K = 1?

### Understanding (5)

6. Why must features be scaled before using KNN?
7. What is the curse of dimensionality?
8. How does weighted voting differ from uniform voting?
9. Why is KNN rarely used in production for large datasets?
10. How does K=1 relate to overfitting?

### Application (5)

11. You have 50K samples and 100 features. Should you use KNN? Why or why not?
12. A KNN model has 100% training accuracy but 60% test accuracy. What's wrong?
13. How would you use KNN for a recommendation system?
14. You notice one feature has range 0–10000 and another has range 0–1. What do you do?
15. How do you handle ties when K is even?

### Mathematical (5)

16. Compute Euclidean distance between (1, 2) and (4, 6).
17. Compute Manhattan distance between (1, 2) and (4, 6).
18. For K=5, neighbors have labels [0, 1, 1, 0, 1]. What's the prediction?
19. For weighted KNN with K=3, distances are [1, 2, 4] and labels are [1, 0, 1]. What's the prediction?
20. If d=100 and n=50, why is KNN likely to perform poorly?

### Interview (5)

21. Compare KNN with logistic regression.
22. How would you speed up KNN for a large dataset?
23. When would you choose KNN over a neural network?
24. What is Leave-One-Out Cross-Validation?
25. Can KNN handle missing data? How?

### Problem Solving (5)

26. Design a spam filter using KNN. What features and preprocessing would you use?
27. Your KNN model is too slow for production. List 3 approaches to speed it up.
28. How would you handle a dataset where 95% of samples belong to one class?
29. You want to use KNN for images. What feature extraction would you do?
30. Compare K=3, K=100, K=1000 on a dataset with 500 samples.

### Answers

**1.** O(n·d) — compute distance to all n training points, each with d features.

**2.** Euclidean distance (L2 norm).

**3.** Because it does no training — all computation is deferred to prediction time.

**4.** K — the number of neighbors.

**5.** Overfitting — the boundary is extremely jagged; each point forms its own region.

**6.** Distance metrics are scale-dependent. Without scaling, features with larger ranges dominate.

**7.** In high dimensions, all points become approximately equidistant, making "nearest neighbor" meaningless.

**8.** Weighted voting gives more influence to closer neighbors (weight = 1/distance); uniform gives equal weight.

**9.** Prediction is O(n·d) per query, which is too slow for large n and real-time constraints.

**10.** K=1 means the model perfectly memorizes training data (zero bias) but is extremely sensitive to noise (high variance).

**11.** No — KNN is too slow at prediction (O(50K × 100) per query). Use a model-based approach or approximate nearest neighbors.

**12.** K is too small (overfitting). Increase K and use cross-validation to find optimal K.

**13.** Find users with similar rating vectors (KNN) and recommend items they liked.

**14.** Standardize both features using StandardScaler before computing distances.

**15.** Use weighted voting (breaks ties naturally) or choose an odd K to avoid ties.

**16.** d = √((4-1)² + (6-2)²) = √(9+16) = √25 = 5.

**17.** d = |4-1| + |6-2| = 3 + 4 = 7.

**18.** Class 1 (3 votes vs 2 votes).

**19.** Weights: w₁=1/1=1, w₂=1/2=0.5, w₃=1/4=0.25. Weighted votes: class 1 = 1+0.25 = 1.25, class 0 = 0.5. Predict: class 1.

**20.** With d=100 >> n=50, all points are approximately equidistant. KNN degrades to random guessing.

**21–30.** Open-ended; review relevant sections for reference.

---

## 49. Final Learning Checklist

- [ ] I can explain KNN in plain English
- [ ] I understand why KNN is called a "lazy learner"
- [ ] I can compute Euclidean and Manhattan distance by hand
- [ ] I know why feature scaling is critical for KNN
- [ ] I can implement KNN from scratch in Python
- [ ] I know how to use sklearn's KNeighborsClassifier
- [ ] I understand the bias-variance tradeoff with K
- [ ] I can explain the curse of dimensionality
- [ ] I know how to choose optimal K via cross-validation
- [ ] I understand weighted vs uniform voting
- [ ] I can compare KNN with model-based classifiers
- [ ] I know the time and space complexity of KNN
- [ ] I understand KD-trees and Ball trees for fast search
- [ ] I can handle class imbalance with KNN
- [ ] I know when to use and when NOT to use KNN
- [ ] I can use KNN for regression (predict mean of K neighbors)
- [ ] I understand how KNN fails in high dimensions
- [ ] I can apply dimensionality reduction before KNN
- [ ] I have completed a project using KNN
- [ ] I can explain KNN to a non-technical person

---

## 50. Quality Control Note

| Criterion | Status | Notes |
|---|---|---|
| Accuracy | ✅ | Distance formulas, complexity, and curse of dimensionality verified |
| Beginner-friendliness | ✅ | Real-life analogy (city neighborhood), step-by-step numerical example |
| Math depth | ✅ | Distance metrics, weighted voting formula, complexity analysis |
| Practical depth | ✅ | sklearn usage, pipeline with scaling, GridSearchCV for K tuning |
| Exam depth | ✅ | Key concepts, common traps, formulas clearly identified |
| Code quality | ✅ | Clean from-scratch implementation with Counter |
| Structure compliance | ✅ | All 50 sections present in order |
