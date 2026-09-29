# 🛒 Big Mart Sales Prediction

A machine learning regression project for predicting **item-level sales at Big Mart outlets** using product and outlet characteristics.

The project focuses on building a **leakage-free machine learning pipeline**, where the data is split into training, validation, and test sets before model-based preprocessing and model selection are performed.

---

## 📌 Project Overview

The objective of this project is to predict:

```text
Item_Outlet_Sales
```

based on information about the product and the outlet where it is sold.

A major focus of the project is avoiding **data leakage** during preprocessing and model selection.

The workflow compares:

* Linear Regression
* Random Forest Regression
* Different Random Forest hyperparameter configurations

It also uses **model-based imputation** for missing `Item_Weight` values and **group-wise mode imputation** for missing `Outlet_Size` values.

---

## 🎯 Objectives

* Perform exploratory data analysis
* Identify and handle missing values
* Clean inconsistent categorical labels
* Create additional useful features
* Split the data before learned preprocessing
* Impute missing values without data leakage
* Encode categorical variables
* Compare Linear Regression and Random Forest
* Tune Random Forest hyperparameters
* Select the best model using validation data
* Evaluate the final model on an untouched test set

---

# 📊 Dataset

The dataset contains **8,523 observations and 12 original columns**.

### Target Variable

```text
Item_Outlet_Sales
```

This represents the sales of a particular item at a particular outlet.

### Features

| Feature                     | Description                         |
| --------------------------- | ----------------------------------- |
| `Item_Identifier`           | Unique identifier for the product   |
| `Item_Weight`               | Weight of the item                  |
| `Item_Fat_Content`          | Fat-content category of the item    |
| `Item_Visibility`           | Visibility of the item in the store |
| `Item_Type`                 | Type/category of the item           |
| `Item_MRP`                  | Maximum Retail Price of the item    |
| `Outlet_Identifier`         | Unique identifier for the outlet    |
| `Outlet_Establishment_Year` | Year the outlet was established     |
| `Outlet_Size`               | Size of the outlet                  |
| `Outlet_Location_Type`      | Location tier of the outlet         |
| `Outlet_Type`               | Type of outlet                      |
| `Item_Outlet_Sales`         | **Target variable**                 |

---

# 🔎 Missing Values

The original dataset contains missing values in two important columns:

| Column        | Missing Values |
| ------------- | -------------: |
| `Item_Weight` |          1,463 |
| `Outlet_Size` |          2,410 |

The remaining columns do not contain missing values.

Instead of simply replacing these values with global statistics, the project uses approaches designed to avoid information leakage.

---

# 🧹 Data Cleaning

### Item Fat Content

The `Item_Fat_Content` column contains inconsistent representations of the same categories.

The following values are standardized:

```text
low fat → Low Fat
LF      → Low Fat
reg     → Regular
```

The cleaned categories become:

```text
Low Fat
Regular
```

### Item Category

An additional feature is extracted from `Item_Identifier`:

```python
Item_Category = Item_Identifier.str[:2]
```

The first two characters represent broad product categories such as:

* `FD` — Food
* `DR` — Drinks
* `NC` — Non-consumable

---

# 🔀 Train / Validation / Test Split

The dataset is split **before any learned imputation or encoding**.

The split is:

```text
60% Training
20% Validation
20% Testing
```

With 8,523 observations, the resulting datasets contain:

```text
Training:   5,113 rows
Validation: 1,705 rows
Testing:    1,705 rows
```

The split uses:

```python
random_state = 42
```

This is important because all preprocessing models are subsequently fitted only using the training data.

---

# 🛡️ Leakage-Free Preprocessing

One of the main objectives of this project is to ensure that information from the validation and test sets does not influence preprocessing or model selection.

The workflow is:

```text
Raw Data
   │
   ▼
Row-wise Cleaning
   │
   ▼
Train / Validation / Test Split
   │
   ├───────────────┐
   ▼               ▼
Training Data   Validation/Test
   │
   ▼
Fit Imputation Models
   │
   ▼
Transform Validation/Test
   │
   ▼
Train Sales Models
   │
   ▼
Validation Selection
   │
   ▼
Final Test Evaluation
```

The notebook explicitly performs the split before learned imputation and encoding.

---

# 🧩 Missing Value Imputation

## 1. `Item_Weight`

Instead of using mean imputation, the project treats missing item weight as a separate prediction problem.

The following item-related features are used:

```text
Item_Identifier
Item_Fat_Content
Item_Type
Item_Category
Item_MRP
Item_Visibility
```

The target for this sub-model is:

```text
Item_Weight
```

The known-weight training rows are divided into training and validation subsets.

Two types of models are evaluated:

* Linear Regression
* Random Forest Regression

### Best Imputation Model

The best-performing model for predicting `Item_Weight` is:

**Linear Regression**

Validation performance:

```text
RMSE: 1.578479
R²:   0.883153
```

The selected model is then used to predict the missing weights in the training, validation, and test datasets.

---

## 2. `Outlet_Size`

Missing `Outlet_Size` values are filled using the **most common outlet size for the corresponding `Outlet_Type`**.

The mapping is learned using the training data only.

The resulting mappings in the notebook are:

| Outlet Type       | Imputed Size |
| ----------------- | ------------ |
| Grocery Store     | Small        |
| Supermarket Type1 | Small        |
| Supermarket Type2 | Medium       |
| Supermarket Type3 | Medium       |

After imputation, there are no missing values remaining in `Outlet_Size` across the train, validation, and test sets.

---

# 📊 Exploratory Data Analysis

EDA is performed using the **training dataset after imputation**.

The project examines:

* Descriptive statistics
* Numerical feature distributions
* Item weight
* Item visibility
* Item MRP
* Item outlet sales
* Categorical features
* Relationships between features and sales

Histograms with KDE are generated for:

```text
Item_Weight
Item_Visibility
Item_MRP
Item_Outlet_Sales
```

---

# 🤖 Machine Learning Models

Two main regression approaches are evaluated.

## 1. Linear Regression

Linear Regression is used as a baseline regression model.

Categorical features are transformed using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Numerical features are passed through without transformation.

The preprocessing and model are combined using a Scikit-learn `Pipeline`.

---

## 2. Random Forest Regression

Random Forest Regression is used as the primary nonlinear model.

Multiple hyperparameter combinations are evaluated.

The model search includes parameters such as:

```text
n_estimators
max_depth
min_samples_leaf
max_features
```

The Random Forest models are evaluated using the validation dataset.

---

# 🔧 Preprocessing Pipeline

Categorical variables are one-hot encoded using:

```python
OneHotEncoder(handle_unknown='ignore')
```

Numerical variables are passed through directly.

The preprocessing is implemented using:

```text
ColumnTransformer
        ↓
OneHotEncoder
        ↓
Regression Model
```

The complete preprocessing and regression model are wrapped in a Scikit-learn `Pipeline`, ensuring that the transformations are learned only from the training data.

---

# 📏 Evaluation Metrics

The project evaluates regression models using:

### RMSE

Root Mean Squared Error measures the typical magnitude of prediction errors.

```text
RMSE = √Mean Squared Error
```

Lower RMSE indicates better predictive performance.

### R² Score

R² measures the proportion of variance in the target explained by the model.

Higher values indicate better explanatory performance.

The validation models are ranked primarily using **validation RMSE**.

---

# 🧪 Model Selection Strategy

For each candidate model:

1. Fit the model on the training dataset.
2. Predict the validation dataset.
3. Calculate validation RMSE.
4. Calculate validation R².
5. Rank the models according to validation RMSE.
6. Select the model with the lowest validation RMSE.

The test set is **not used during model selection**.

This prevents the test set from influencing the choice of model or hyperparameters.

---

# 📈 Project Workflow

```text
                 Big Mart Dataset
                        │
                        ▼
                 Data Cleaning
                        │
                        ├── Clean Fat Content
                        │
                        └── Create Item Category
                        │
                        ▼
             Train / Validation / Test
                  60% / 20% / 20%
                        │
            ┌───────────┴───────────┐
            ▼                       ▼
      Item Weight              Outlet Size
       Imputation               Imputation
            │                       │
      Linear/RF Model          Group-wise Mode
            │                       │
            └───────────┬───────────┘
                        ▼
                       EDA
                        │
                        ▼
             Feature Preprocessing
                        │
                        ▼
              Model Training
                 │           │
                 ▼           ▼
          Linear Regression  Random Forest
                 │           │
                 └─────┬─────┘
                       ▼
              Validation Ranking
                       │
                       ▼
                 Best Model
                       │
                       ▼
             Final Test Evaluation
```

---

# 📂 Project Structure

```text
Big-Mart-Sales-Prediction/
│
├── Project_12_Big_Mart_Sales_Prediction_fixed.ipynb
├── sale/
│   └── Train.csv
└── README.md
```

The notebook currently expects the dataset at:

```python
sale/Train.csv
```

If your dataset is stored elsewhere, update:

```python
DATA_PATH = 'sale/Train.csv'
```

---

# 🛠️ Technologies Used

### Programming Language

* Python

### Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

### Scikit-learn Components

* `LinearRegression`
* `RandomForestRegressor`
* `train_test_split`
* `ColumnTransformer`
* `Pipeline`
* `OneHotEncoder`
* Regression metrics

---

# 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/big-mart-sales-prediction.git
cd big-mart-sales-prediction
```

### 2. Install dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### 3. Place the dataset

Place the dataset at:

```text
sale/Train.csv
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Open the notebook

```text
Project_12_Big_Mart_Sales_Prediction_fixed.ipynb
```

Run the cells sequentially.

---

# ⭐ Key Takeaways

### Leakage prevention

The most important design choice in the project is splitting the data **before learned preprocessing**.

### Model-based missing-value imputation

`Item_Weight` is treated as a prediction problem rather than simply replacing missing values with a mean.

### Validation-based model selection

Models and hyperparameters are selected using the validation set.

### Untouched test set

The test set is reserved for the final evaluation and is not used during model selection.

### Pipeline-based preprocessing

Encoding and modeling are combined into Scikit-learn pipelines, reducing the risk of preprocessing leakage.

---

## 📌 Project Highlights

```text
Dataset Size       : 8,523 rows
Original Features  : 11 predictors + target
Train Split        : 60%
Validation Split   : 20%
Test Split         : 20%
Random Seed         : 42

Missing Item Weight: 1,463
Missing Outlet Size: 2,410

Best Weight Model  : Linear Regression
Weight RMSE        : 1.5785
Weight R²          : 0.8832

Sales Models       : Linear Regression + Random Forest
Primary Metric     : Validation RMSE
```

All reported values above are taken directly from the uploaded notebook.

---

## 👤 Author

Suryansh Pandey 

If you found this project useful, consider ⭐ starring the repository.
