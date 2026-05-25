# 🌴 Date Fruit Classification using ANN

### Deep Learning | PyTorch | Multiclass Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![Accuracy](https://img.shields.io/badge/Accuracy-94.44%25-brightgreen?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-7-orange?style=for-the-badge)

-----

## Project Overview

This project implements an **Artificial Neural Network (ANN)** from scratch using **PyTorch** to classify **7 different varieties of Date Fruits** based on their physical and morphological features.

The model achieves an accuracy of **94.44%** on the test set — demonstrating strong generalization on a real-world multiclass classification problem.

-----

## Date Fruit Classes

|Class|Variety|
|-----|-------|
|1    |BERHI  |
|2    |DEGLET |
|3    |DOKOL  |
|4    |IRAQI  |
|5    |ROTANA |
|6    |SAFAVI |
|7    |SOGAY  |

-----

## Dataset

- **Source**: DateFruit_Dataset.csv
- **Samples**: 898 rows
- **Features**: 34 input features (morphological measurements)
- **Target**: 7 fruit variety classes
- **Missing Values**: None ✅

### Feature Categories:

- Geometric features: Area, Perimeter, Major/Minor Axis, Eccentricity
- Shape factors: Solidity, Convex Area, Extent, Aspect Ratio
- Color features: MeanRR, MeanRG, MeanRB (RGB channels)
- Statistical features: StdDev, Skewness, Kurtosis, Entropy per channel
- ALL channel features

-----

##  Model Architecture

```
Input Layer  →  34 features
     ↓
Hidden Layer 1  →  64 neurons  +  ReLU
     ↓
Hidden Layer 2  →  64 neurons  +  ReLU
     ↓
Output Layer  →  7 neurons (one per class)
```

### Why this architecture?

- **ReLU** in hidden layers → avoids vanishing gradient problem
- **No activation on output** → CrossEntropyLoss handles softmax internally
- **64 neurons** → sufficient capacity for 34 input features

-----

## Training Configuration

|Parameter       |Value           |
|----------------|----------------|
|Loss Function   |CrossEntropyLoss|
|Optimizer       |Adam            |
|Epochs          |100             |
|Batch Size      |32              |
|Train/Test Split|80% / 20%       |
|Scaling         |StandardScaler  |
|Label Encoding  |LabelEncoder    |

-----

## Results

|Metric             |Value     |
|-------------------|----------|
|Total Test Samples |180       |
|Correct Predictions|170       |
|**Test Accuracy**  |**94.44%**|
|Final Training Loss|~0.027    |

### Loss Curve Observations:

- Loss dropped sharply from **~1.69** (epoch 1) to **~0.027** (epoch 100)
- Smooth convergence with no signs of overfitting
- Model generalized well across all 7 classes

-----

## 🛠️ Tech Stack

```
Python          →  Core language
PyTorch         →  Neural network framework
Scikit-learn    →  Preprocessing (StandardScaler, LabelEncoder, train_test_split)
Pandas          →  Data loading and manipulation
NumPy           →  Numerical operations
```

-----

## 📁 Project Structure

```
ann_classification/
│
├── ann_classification.ipynb    # Main notebook
├── DateFruit_Dataset.csv       # Dataset
└── README.md                   # Project documentation
```

-----

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning/ann_classification
```

**2. Install dependencies**

```bash
pip install torch pandas numpy scikit-learn
```

**3. Run the notebook**

```bash
jupyter notebook ann_classification.ipynb
```

-----

##  Key Learnings

- **Multiclass classification** requires `CrossEntropyLoss` + `LabelEncoder`
- **StandardScaler** is preferred over MinMax for ANN training
- `torch.max(outputs, 1)` extracts predicted class index from logits
- `torch.no_grad()` during evaluation saves memory and speeds up inference
- **34 features** with proper scaling leads to fast convergence

-----

## 👨‍💻 Author

**Mehran Mushtaq**

- 1st Year CSE Student
- ML & Deep Learning
- 📍 Kashmir, India

> *“Built from scratch. No shortcuts. Just learning.”*

-----

## 🔗 Related Projects

- [ANN Regression — Power Plant Energy Prediction](../ann_regression/)

-----

