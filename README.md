# Credit Card Fraud Detection

A machine learning project for detecting fraudulent credit card transactions using **Logistic Regression, Decision Trees, Random Forest, and XGBoost**, with a strong focus on handling severe class imbalance and avoiding data leakage.

## 📌 Project Overview

Credit card fraud detection is a highly imbalanced binary classification problem. In the dataset used in this project, fraudulent transactions represent only **0.1727%** of all transactions.

This project explores different approaches for handling this imbalance and compares several machine learning models using metrics that are more meaningful than accuracy, particularly **Precision, Recall, F1-score, ROC-AUC, and PR-AUC**.

The project also demonstrates:

- Exploratory Data Analysis
- Proper train/test splitting
- Robust scaling
- Class weighting
- Random undersampling
- SMOTE oversampling
- Tree-based ensemble models
- Cross-validation
- Randomized hyperparameter search
- Leakage-safe SMOTE pipelines
- Precision-Recall analysis
- Decision threshold tuning

---

## 📊 Dataset

The project uses the **Credit Card Fraud Detection** dataset containing transactions made by European cardholders.

The dataset contains:

- **284,807 transactions**
- **30 input features**
- **1 target variable (`Class`)**
- **492 fraudulent transactions**
- **284,315 genuine transactions**

### Class Distribution

| Class | Transactions | Percentage |
|---|---:|---:|
| Genuine | 284,315 | 99.8273% |
| Fraud | 492 | 0.1727% |

This corresponds to approximately **577 genuine transactions for every fraudulent transaction**.

### Features

The dataset contains:

- `Time` — seconds elapsed between transactions
- `V1`–`V28` — PCA-transformed numerical features
- `Amount` — transaction amount
- `Class` — target variable
  - `0` → Genuine transaction
  - `1` → Fraudulent transaction

The original dataset is available from Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🔍 Exploratory Data Analysis

The notebook performs several EDA steps to understand the dataset and fraud patterns.

### Class Distribution

The extreme class imbalance is visualized to demonstrate why ordinary accuracy can be misleading.

### Transaction Amount

The distribution of transaction amounts is compared between genuine and fraudulent transactions.

### Transaction Time

The distribution of transaction times is examined separately for genuine and fraudulent transactions.

### Feature Correlation

Pearson correlations between the input features and the fraud label are calculated to provide an initial view of feature relevance.

---

## ⚙️ Data Preprocessing

The dataset is first divided into training and testing sets using a **stratified 75/25 split**.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
```

The resulting datasets are:

- Training: **213,605 observations**
- Testing: **71,202 observations**

The fraud rate is preserved in both sets.

### Robust Scaling

`Time` and `Amount` are scaled using `RobustScaler`.

The scaler is:

1. Fit only on the training data
2. Applied to the training data
3. Applied to the test data using the training statistics

This prevents **data leakage** from the test set.

The `V1`–`V28` features are already PCA-transformed and were therefore not additionally scaled.

---

# ⚠️ Why Accuracy Is Not Enough

A model can achieve extremely high accuracy simply by predicting almost every transaction as genuine.

For example, the baseline Logistic Regression achieved:

| Metric | Score |
|---|---:|
| Accuracy | 99.91% |
| Precision | 83.70% |
| Recall | 62.60% |
| F1 | 71.63% |
| ROC-AUC | 96.10% |
| PR-AUC | 71.47% |

Although the accuracy looks excellent, the model missed:

**46 out of 123 fraudulent transactions in the test set.**

For fraud detection, missing a fraudulent transaction can be much more costly than incorrectly flagging a genuine transaction.

Therefore, this project focuses heavily on **Recall, Precision, F1-score and especially PR-AUC**.

---

# ⚖️ Handling Class Imbalance

Three approaches are evaluated with Logistic Regression.

## 1. Class Weighting

The minority class is given a higher penalty during training.

```python
LogisticRegression(
    max_iter=1000,
    class_weight='balanced'
)
```

## 2. Random Undersampling

Majority-class observations are removed from the training set until the classes are balanced.

## 3. SMOTE

**Synthetic Minority Oversampling Technique (SMOTE)** creates synthetic minority-class observations.

Importantly, all imbalance-handling techniques are applied **only to the training data**.

The test set remains untouched and retains its original real-world class distribution.

### Logistic Regression Results

| Model | Accuracy | Precision | Recall | F1 | PR-AUC |
|---|---:|---:|---:|---:|---:|
| Baseline LR | 99.91% | 83.70% | 62.60% | 71.63% | 71.47% |
| LR + SMOTE | 97.61% | 6.07% | 88.62% | 11.37% | 71.08% |
| LR + Class Weight | 97.69% | 6.28% | 88.62% | 11.72% | 70.41% |
| LR + Undersampling | 96.62% | 4.40% | 89.43% | 8.38% | 68.55% |

This demonstrates the fundamental **precision-recall trade-off** associated with handling highly imbalanced data.

---

# 🌳 Tree-Based Models

The project then evaluates three tree-based models:

### Decision Tree

A Decision Tree is trained using:

```python
class_weight='balanced'
```

### Random Forest

A Random Forest is trained using:

```python
n_estimators=200
max_depth=10
class_weight='balanced'
```

### XGBoost

XGBoost uses:

```python
scale_pos_weight
```

where the value is calculated from the negative-to-positive class ratio in the training data.

The calculated imbalance ratio was approximately:

```text
577.9
```

---

# 🔄 Tree Models + SMOTE

The same three tree-based models are also trained on a SMOTE-balanced training set.

This allows a direct comparison between:

- Native imbalance handling
- SMOTE-based imbalance handling

### Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost + Class Weight | 99.95% | 88.39% | 80.49% | 84.26% | 98.03% | **85.57%** |
| XGBoost + SMOTE | 99.88% | 61.08% | 82.93% | 70.34% | 97.84% | 84.17% |
| Random Forest + Class Weight | 99.91% | 71.83% | 82.93% | 76.98% | 98.36% | 80.09% |
| Random Forest + SMOTE | 99.78% | 42.39% | 83.74% | 56.28% | 97.96% | 79.06% |
| Decision Tree + Class Weight | 98.98% | 12.12% | 78.05% | 20.98% | 82.99% | 45.92% |
| Decision Tree + SMOTE | 97.57% | 5.67% | 83.74% | 10.63% | 88.92% | 40.25% |

The initial XGBoost model using `scale_pos_weight` achieved the highest PR-AUC among these models.

---

# 🔧 Hyperparameter Tuning

To improve the XGBoost model, **RandomizedSearchCV** with **Stratified K-Fold Cross-Validation** is used.

The optimization metric is:

```text
Average Precision / PR-AUC
```

rather than accuracy.

This is important because PR-AUC is more informative for highly imbalanced classification problems.

### Cross-Validation

A **3-fold Stratified K-Fold** strategy is used.

```python
StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
```

### Parameters Tuned

The search explores:

- `n_estimators`
- `max_depth`
- `learning_rate`
- `subsample`
- `colsample_bytree`
- SMOTE `k_neighbors`

### Best Parameters

The best configuration found was:

```text
n_estimators       = 350
max_depth          = 6
learning_rate      = 0.2
subsample          = 0.85
colsample_bytree   = 0.85
SMOTE k_neighbors  = 7
```

Best cross-validation PR-AUC:

```text
0.8462
```

---

# 🛡️ Preventing Data Leakage During Cross-Validation

One of the important aspects of the project is that SMOTE is included inside an `imblearn` pipeline:

```python
Pipeline([
    ('smote', SMOTE(...)),
    ('clf', XGBClassifier(...))
])
```

This ensures that SMOTE is applied separately to the training portion of each cross-validation fold.

Applying SMOTE to the complete training dataset **before** cross-validation could allow synthetic information derived from validation samples to influence the training folds and produce overly optimistic validation results.

---

# 🏆 Final Model

The final model is:

**Tuned XGBoost + SMOTE**

with the best parameters obtained through RandomizedSearchCV.

### Test Set Performance

At the default classification threshold of `0.50`:

| Metric | Score |
|---|---:|
| Accuracy | **99.94%** |
| Precision | **80.47%** |
| Recall | **83.74%** |
| F1-score | **82.07%** |
| ROC-AUC | **98.45%** |
| PR-AUC | **85.49%** |

The model detected:

- **103 of 123 fraudulent transactions**
- **20 false negatives**
- **25 false positives**

---

# 📈 Precision-Recall & ROC Analysis

The project evaluates both:

### ROC Curve

Measures the relationship between:

- True Positive Rate
- False Positive Rate

The final model achieved:

```text
ROC-AUC = 0.9845
```

### Precision-Recall Curve

This is particularly important for the highly imbalanced fraud detection problem.

The final model achieved:

```text
PR-AUC = 0.8549
```

---

# 🎯 Decision Threshold Tuning

By default, binary classifiers classify a transaction as fraudulent when:

```text
P(Fraud) >= 0.50
```

However, the optimal threshold does not necessarily have to be `0.50`.

The project evaluates different thresholds and selects the threshold that maximizes the **F1-score**.

### Default Threshold

```text
Threshold = 0.50
F1 = 0.8207
```

### F1-Optimal Threshold

```text
Threshold = 0.9911
F1 = 0.8597
Precision = 0.9694
Recall = 0.7724
```

At this threshold:

- True Positives = **95**
- False Positives = **3**
- False Negatives = **28**
- True Negatives = **71,076**

This illustrates how changing the decision threshold can significantly alter the precision-recall trade-off **without retraining the model**.

In a real fraud-detection system, the threshold should ultimately be selected based on the actual business cost of false positives versus false negatives rather than F1-score alone.

---

# 🔑 Key Learnings

### 1. Accuracy can be misleading

With extreme class imbalance, a model can achieve very high accuracy while still missing a meaningful number of fraud cases.

### 2. PR-AUC is highly useful

For heavily imbalanced binary classification, PR-AUC provides a more informative view of minority-class performance than accuracy.

### 3. Different imbalance techniques behave differently

Class weighting, undersampling and SMOTE can produce substantially different precision-recall trade-offs.

### 4. XGBoost performed strongly

XGBoost achieved the strongest overall performance among the initial tree-based models.

### 5. Cross-validation must be leakage-safe

SMOTE should be performed inside the cross-validation pipeline rather than before cross-validation.

### 6. Threshold tuning matters

The probability threshold can be adjusted depending on whether the application prioritizes:

- Higher fraud detection
- Fewer false alarms
- Better F1-score
- Business cost minimization

---

# 🚀 Possible Future Improvements

The notebook also identifies several possible extensions:

### Anomaly Detection

Explore unsupervised approaches such as:

- Isolation Forest
- One-Class SVM
- Autoencoders

These approaches could be useful when labeled fraud examples are limited.

### Cost-Sensitive Learning

Instead of optimizing F1-score, define an explicit cost matrix:

```text
Cost of missed fraud
vs.
Cost of false alarm
```

and choose the threshold that minimizes the expected financial cost.

### Explainable AI

Use **SHAP** to understand which features contribute to individual fraud predictions and provide transaction-level explanations.

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- XGBoost
- Jupyter Notebook

---

# 📁 Project Structure

```text
credit-card-fraud-detection/
│
├── credit_card_fraud_detection.ipynb
├── creditcard.csv
└── README.md
```

> The dataset itself may need to be downloaded separately from Kaggle due to its size and licensing/distribution requirements.

---

# ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd credit-card-fraud-detection
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost jupyter
```

### 3. Add the dataset

Download `creditcard.csv` and place it in the project directory.

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
credit_card_fraud_detection.ipynb
```

and run the cells sequentially.

---

# 📌 Conclusion

This project demonstrates a complete machine learning workflow for highly imbalanced fraud detection, from exploratory analysis and leakage-safe preprocessing to imbalance handling, model comparison, hyperparameter tuning, and threshold optimization.

The final tuned XGBoost model achieved **99.94% accuracy, 80.47% precision, 83.74% recall, 82.07% F1-score, and 85.49% PR-AUC** at the default threshold.

More importantly, the project highlights that successful fraud detection is not simply about maximizing accuracy. The choice of evaluation metric, imbalance strategy, cross-validation procedure, and classification threshold can have a major impact on the usefulness of the resulting model.
