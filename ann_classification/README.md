# 🌴 Date Fruit Classification using ANN

**Deep Learning | PyTorch | Multiclass Classification**

![Python](https://img.shields.io/badge/Python-3.x-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-DL-red) ![Accuracy](https://img.shields.io/badge/ANN%20Accuracy-94.44%25-brightgreen) ![Classes](https://img.shields.io/badge/Classes-7-orange)

## Project Overview

This project implements an **Artificial Neural Network (ANN)** from scratch using PyTorch to classify 7 different varieties of Date Fruits based on their physical and morphological features.

The ANN achieves an accuracy of **94.44%** on the test set. To validate this result, the ANN is also benchmarked against two classical machine learning baselines — **Logistic Regression** and **Random Forest** — trained on the same train/test split, confirming that the deep learning approach outperforms traditional methods on this dataset.

## Date Fruit Classes

| Class | Variety |
|-------|---------|
| 1 | BERHI |
| 2 | DEGLET |
| 3 | DOKOL |
| 4 | IRAQI |
| 5 | ROTANA |
| 6 | SAFAVI |
| 7 | SOGAY |

## Dataset

- **Source:** `DateFruit_Dataset.csv`
- **Samples:** 898 rows
- **Features:** 34 input features (morphological measurements)
- **Target:** 7 fruit variety classes
- **Missing Values:** None ✅

**Feature Categories:**
- Geometric features: Area, Perimeter, Major/Minor Axis, Eccentricity
- Shape factors: Solidity, Convex Area, Extent, Aspect Ratio
- Color features: MeanRR, MeanRG, MeanRB (RGB channels)
- Statistical features: StdDev, Skewness, Kurtosis, Entropy per channel
- ALL channel (wavelet) features

### Exploratory Correlation Analysis

A full feature correlation heatmap was generated to inspect relationships between features and the target class.

**Strongest positive correlations with `Class`:**

| Feature | Correlation |
|---|---|
| MAJOR_AXIS | 0.479 |
| SkewRR | 0.441 |
| PERIMETER | 0.420 |
| SkewRG | 0.391 |
| ECCENTRICITY | 0.311 |

**Strongest negative correlations with `Class`:**

| Feature | Correlation |
|---|---|
| ROUNDNESS | -0.488 |
| ALLdaub4RG | -0.444 |
| MeanRG | -0.444 |
| SHAPEFACTOR_2 | -0.426 |
| MeanRR | -0.416 |

## Model Architecture (ANN)

```
Input Layer  →  34 features
     ↓
Hidden Layer 1  →  64 neurons  +  ReLU
     ↓
Hidden Layer 2  →  64 neurons  +  ReLU
     ↓
Output Layer  →  7 neurons (one per class)
```

**Why this architecture?**
- ReLU in hidden layers → avoids vanishing gradient problem
- No activation on output → CrossEntropyLoss handles softmax internally
- 64 neurons → sufficient capacity for 34 input features

### Training Configuration

| Parameter | Value |
|---|---|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Epochs | 100 |
| Batch Size | 32 |
| Train/Test Split | 80% / 20% (stratified) |
| Scaling | StandardScaler |
| Label Encoding | LabelEncoder |

## Baseline Model Comparison

To contextualize the ANN's performance, two classical ML models were trained on the same stratified 80/20 split.

### Logistic Regression

- **Pipeline:** `StandardScaler` → `LogisticRegression`
- **Test Accuracy:** 92.78%
- **5-Fold CV Mean Accuracy:** 91.42% (std: 0.0077)

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (BERHI) | 1.00 | 0.92 | 0.96 | 13 |
| 1 (DEGLET) | 0.83 | 0.75 | 0.79 | 20 |
| 2 (DOKOL) | 0.95 | 0.95 | 0.95 | 41 |
| 3 (IRAQI) | 0.93 | 1.00 | 0.97 | 14 |
| 4 (ROTANA) | 0.94 | 0.97 | 0.96 | 33 |
| 5 (SAFAVI) | 0.95 | 1.00 | 0.98 | 40 |
| 6 (SOGAY) | 0.83 | 0.79 | 0.81 | 19 |

### Random Forest Classifier

- **Tuning:** `GridSearchCV` (5-fold) over `n_estimators`, `max_depth`, `min_samples_leaf`, `min_samples_split`
- **Test Accuracy:** 92.22%

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| 0 (BERHI) | 1.00 | 0.85 | 0.92 | 13 |
| 1 (DEGLET) | 0.83 | 0.75 | 0.79 | 20 |
| 2 (DOKOL) | 0.97 | 0.95 | 0.96 | 41 |
| 3 (IRAQI) | 0.93 | 1.00 | 0.97 | 14 |
| 4 (ROTANA) | 0.97 | 0.97 | 0.97 | 33 |
| 5 (SAFAVI) | 1.00 | 0.95 | 0.97 | 40 |
| 6 (SOGAY) | 0.68 | 0.89 | 0.77 | 19 |

**Top predictive features (Random Forest feature importance):** dominated by geometric measures such as `MAJOR_AXIS`, `PERIMETER`, and `ROUNDNESS`, consistent with the correlation analysis above.

### Model Comparison Summary

| Model | Test Accuracy |
|---|---|
| **ANN (PyTorch)** | **94.44%** |
| Logistic Regression | 92.78% |
| Random Forest | 92.22% |

The ANN outperforms both classical baselines, particularly on the harder-to-separate `DEGLET` and `SOGAY` classes, which all three models struggle with relatively — suggesting some inherent feature overlap between these varieties.

## Results (ANN)

| Metric | Value |
|---|---|
| Total Test Samples | 180 |
| Correct Predictions | 170 |
| Test Accuracy | 94.44% |
| Final Training Loss | ~0.027 |

**Loss Curve Observations:**
- Loss dropped sharply from ~1.69 (epoch 1) to ~0.027 (epoch 100)
- Smooth convergence with no signs of overfitting
- Model generalized well across all 7 classes

## 🛠️ Tech Stack

```
Python          →  Core language
PyTorch         →  Neural network framework
Scikit-learn    →  Preprocessing, baselines, GridSearchCV, metrics
Pandas          →  Data loading and manipulation
NumPy           →  Numerical operations
Matplotlib      →  Plotting (loss curves, feature importance)
Seaborn         →  Correlation heatmaps, confusion matrices
```

## 📁 Project Structure

```
ann_classification/
│
├── ann_classification.ipynb    # Main notebook (ANN + baseline models)
├── DateFruit_Dataset.csv       # Dataset
└── README.md                   # Project documentation
```

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning/ann_classification
```

**2. Install dependencies**
```bash
pip install torch pandas numpy scikit-learn matplotlib seaborn
```

**3. Run the notebook**
```bash
jupyter notebook ann_classification.ipynb
```

## Key Learnings

- Multiclass classification requires CrossEntropyLoss + LabelEncoder
- StandardScaler is preferred over MinMax for ANN training
- `torch.max(outputs, 1)` extracts predicted class index from logits
- `torch.no_grad()` during evaluation saves memory and speeds up inference
- 34 features with proper scaling leads to fast convergence
- Classical baselines (Logistic Regression, Random Forest) are a useful sanity check — the ANN's edge over them (~2 points) justifies the added architectural complexity here
- Cross-validation confirms Logistic Regression's held-out score isn't a fluke of the split

##  Author

**Mehran Mushtaq**
- As 1st Year CSE Student
- ML & Deep Learning

> "Built from scratch. No shortcuts. Just learning."

## 🔗 Related Projects

[ANN Regression — Power Plant Energy Prediction](#)
