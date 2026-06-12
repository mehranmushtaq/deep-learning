# ⚡ Power Plant Energy Output Prediction

### Complete ML & DL Comparison Study | Sklearn + PyTorch

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Sklearn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Best R²](https://img.shields.io/badge/Best%20R²-0.9648-brightgreen?style=for-the-badge)
![Models](https://img.shields.io/badge/Models-6-orange?style=for-the-badge)

> *A complete machine learning study comparing 6 different models on the same dataset — from simple Linear Regression to deep ANN to advanced Stacking Ensembles.*

-----

## 📌 Project Overview

This project predicts the **hourly net electrical energy output** of a Combined Cycle Power Plant using environmental sensor readings. What makes this project unique is the **comprehensive model comparison** — every major regression technique is applied, tuned, and evaluated on the same dataset.

Built as a **1st year CSE student**, self-learned from ground up.

-----

## 🏭 Problem Statement

Predict **PE (Net Energy Output in MW)** given:

|Feature|Description                |
|-------|---------------------------|
|AT     |Ambient Temperature (°C)   |
|V      |Exhaust Vacuum (cm Hg)     |
|AP     |Ambient Pressure (millibar)|
|RH     |Relative Humidity (%)      |

-----

## 📊 Dataset

- **File**: `powerplant_data.csv`
- **Samples**: 9,568
- **Features**: 4 input features
- **Target**: PE — Net Energy Output (MW)
- **Missing Values**: None ✅

-----

## 🏆 Model Comparison Results

|#|Model                 |R² Score  |Mean R² (5-Fold CV)|Notes                   |
|-|----------------------|----------|-------------------|------------------------|
|1|Linear Regression     |0.9303    |0.9284             |Baseline                |
|2|ANN (PyTorch)         |0.9342    |—                  |2 hidden layers         |
|3|SVR                   |0.9481    |0.9476             |RBF kernel, GridSearchCV|
|4|Voting Regressor      |0.9589    |0.9580             |LR + RF + SVR           |
|5|Random Forest         |0.9645    |0.9640             |300 trees, GridSearchCV |
|6|**Stacking Regressor**|**0.9648**|**0.9644**         |**LR+RF+SVR → Ridge** 🏆 |

-----

## 📈 Visual Comparison

```
Linear Regression  ████████████████████░░░  R² = 0.9303
ANN (PyTorch)      ████████████████████░░░  R² = 0.9342
SVR                █████████████████████░░  R² = 0.9481
Voting Ensemble    ██████████████████████░  R² = 0.9589
Random Forest      ███████████████████████  R² = 0.9645
Stacking Ensemble  ███████████████████████  R² = 0.9648 🏆
```

-----

## 🧠 Models & Techniques

### 1. Linear Regression (Baseline)

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('LR', LinearRegression())
])
# R² = 0.9303 | MAE = 3.61 | MSE = 20.21
```

### 2. ANN — Artificial Neural Network (PyTorch)

```
Input (4) → Linear(4→6) → ReLU
          → Linear(6→6) → ReLU
          → Linear(6→1)
# Optimizer: Adam | Loss: MSELoss | Epochs: 100
# R² = 0.9342 | Test MSE = 18.82
```

### 3. Support Vector Regressor

```python
Pipeline([
    ('scaler', StandardScaler()),
    ('svr', SVR(kernel='rbf'))
])
# GridSearchCV: C=[0.1,0.5,50,100], epsilon=[0.2,0.4,0.8,1]
# Best R² = 0.9481 | Mean CV R² = 0.9476
```

### 4. Random Forest Regressor

```python
RandomForestRegressor(
    n_estimators=300, max_depth=10,
    min_samples_split=5, min_samples_leaf=2,
    max_features='sqrt'
)
# GridSearchCV: 54 candidates × 5 folds
# Best R² = 0.9645 | Mean CV R² = 0.9640
```

### 5. Voting Regressor (Ensemble)

```python
VotingRegressor(estimators=[
    ('lr', pipe_lr),
    ('rf', best_rf),
    ('svr', best_svr)
], weights=[1, 2, 1])
# R² = 0.9589 | Mean CV R² = 0.9580
```

### 6. Stacking Regressor (Best Model 🏆)

```python
StackingRegressor(
    estimators=[
        ('lr', pipe_lr),
        ('rf', best_rf),
        ('svr', best_svr)
    ],
    final_estimator=Ridge(),
    cv=5
)
# R² = 0.9648 | Mean CV R² = 0.9644
```

-----

## 🔑 Key Learnings

```
✅ Tree models (RF) outperform ANN on small tabular datasets
✅ Stacking > Voting — meta-learner captures model complementarity
✅ GridSearchCV essential for SVR and RF hyperparameter tuning
✅ Pipeline prevents data leakage during cross-validation
✅ Cross-validation gives more reliable estimate than single test split
✅ StandardScaler critical for LR and SVR, optional for RF
✅ ANN needs more data to outshine classical ML on tabular data
```

-----

## 🛠️ Tech Stack

```python
ml_framework  = "Scikit-Learn"
dl_framework  = "PyTorch"
language      = "Python 3.8+"
libraries     = ["pandas", "numpy", "sklearn",
                 "torch", "matplotlib"]
environment   = "Jupyter Notebook"
hardware      = "Apple M4 MPS GPU (ANN training)"
```

-----
## 🚀 How to Run

```bash
# Clone
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning/ann_regression

# Install
pip install torch pandas numpy scikit-learn matplotlib jupyter

# Run ML comparison
jupyter notebook ml_comparison.ipynb

# Run ANN
jupyter notebook ann_regression.ipynb
```

-----

## 💡 Why Stacking Beats Everything

```
Linear Regression  → captures linear patterns
Random Forest      → captures non-linear patterns
SVR                → captures margin-based patterns

Stacking combines all three:
→ Each model learns different aspects
→ Ridge meta-learner finds optimal combination
→ Result: better than any single model
```

-----

## 👨‍💻 Author

**Mehran Mushtaq**

- 🎓 1st Year CSE Student
- 🔥 Self-learned ML & Deep Learning
- 📍 Kashmir, India
- 🐙 [GitHub](https://github.com/mehranmushtaq)
- 💻 [LeetCode](https://leetcode.com/u/mehraan1/)

> *“From Kashmir — built everything from scratch, one commit at a time.”* 🚀

-----

## 🔗 Related Projects

- [ANN Classification — Date Fruit (94.44%)](../ann_classification/)
- [CNN — CIFAR-10 Image Classification (75.32%)](../cnn/)
- [CreditWise — Loan Approval System](https://github.com/mehranmushtaq/creditwise-loan-system)
- [House Price Prediction — XGBoost (R²=0.929)](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch)

-----

⭐ **Star this repo if you find it helpful!**
