# Power Plant Energy Output Prediction using ANN

### Deep Learning | PyTorch | Regression

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?style=for-the-badge&logo=pytorch)
![R2 Score](https://img.shields.io/badge/R²%20Score-93.42%25-brightgreen?style=for-the-badge)
![MSE](https://img.shields.io/badge/Test%20MSE-18.82-orange?style=for-the-badge)

-----

## Project Overview

This project implements an **Artificial Neural Network (ANN)** from scratch using **PyTorch** to predict the **hourly electrical energy output** of a Combined Cycle Power Plant based on environmental sensor readings.

The model achieves an **R² score of 0.934** on the test set — meaning the model explains **93.4% of the variance** in energy output, demonstrating strong predictive performance on a real-world regression problem.

-----

## Problem Statement

Predict the **net hourly electrical energy output (PE)** of a power plant given:

- Ambient Temperature
- Exhaust Vacuum
- Ambient Pressure
- Relative Humidity

-----

## Dataset

- **File**: powerplant_data.csv
- **Samples**: 9,568 rows
- **Features**: 4 input features
- **Target**: PE (Produced Energy in MW)
- **Missing Values**: None ✅

### Features Description:

|Feature|Description                        |
|-------|-----------------------------------|
|AT     |Ambient Temperature (°C)           |
|V      |Exhaust Vacuum (cm Hg)             |
|AP     |Ambient Pressure (millibar)        |
|RH     |Relative Humidity (%)              |
|PE     |Net Energy Output (MW) — **Target**|

-----

## Model Architecture

```
Input Layer  →  4 features (AT, V, AP, RH)
     ↓
Hidden Layer 1  →  6 neurons  +  ReLU
     ↓
Hidden Layer 2  →  6 neurons  +  ReLU
     ↓
Output Layer  →  1 neuron (Linear — continuous output)
```

### Why this architecture?

- **ReLU** in hidden layers → avoids vanishing gradient problem
- **Linear activation on output** → regression needs continuous unbounded output
- **MSELoss** → standard loss for regression problems
- **Small architecture** → dataset has only 4 features, avoids overfitting

-----

## ⚙️ Training Configuration

|Parameter        |Value         |
|-----------------|--------------|
|Loss Function    |MSELoss       |
|Optimizer        |Adam          |
|Epochs           |100           |
|Batch Size       |32            |
|Train/Test Split |80% / 20%     |
|Scaling          |StandardScaler|
|Best Model Saving|✅ torch.save()|

-----

## 📈 Results

|Metric         |Value     |
|---------------|----------|
|Training MSE   |20.48     |
|**Testing MSE**|**18.82** |
|**R² Score**   |**0.9342**|

### Loss Curve Observations:

- Loss dropped sharply from **~205,931** (epoch 1) to near **~20** (epoch 100)
- Training and Validation loss curves stayed **close together** → no overfitting ✅
- Best model saved automatically during training
- Test MSE **lower** than Train MSE → excellent generalization

-----

## 🛠️ Tech Stack

```
Python          →  Core language
PyTorch         →  Neural network framework
Scikit-learn    →  Preprocessing + R² evaluation
Pandas          →  Data loading and manipulation
NumPy           →  Numerical operations
Matplotlib      →  Loss curve visualization
```

-----

## 📁 Project Structure

```
ann_regression/
│
├── ann_regression.ipynb    # Main notebook
├── powerplant_data.csv     # Dataset
├── best_model.pt           # Saved best model weights
└── README.md               # Project documentation
```

-----

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning/ann_regression
```

**2. Install dependencies**

```bash
pip install torch pandas numpy scikit-learn matplotlib
```

**3. Run the notebook**

```bash
jupyter notebook ann_regression.ipynb
```

-----

## 🔑 Key Learnings

- **Regression** requires Linear output activation + MSELoss
- **StandardScaler** normalizes features for faster convergence
- `torch.save(model.state_dict(), "best_model.pt")` saves best model during training
- `torch.no_grad()` during validation prevents unnecessary gradient computation
- **R² score** is the best metric for regression evaluation (not just MSE)
- Training and validation loss **tracking** helps detect overfitting early

-----

## 📉 Predicted vs Actual Values (Sample)

|#|Predicted (MW)|Actual (MW)|
|-|--------------|-----------|
|0|434.96        |433.27     |
|1|436.88        |438.16     |

Very close predictions — model learned the pattern well ✅

-----

## 👨‍💻 Author

**Mehran Mushtaq**

- 1st Year CSE Student
- ML & Deep Learning
- 📍 Kashmir, India

> *“Built from scratch. No shortcuts. Just learning.”*

-----

## 🔗 Related Projects

- [ANN Classification — Date Fruit Variety Classification](../ann_classification/)

-----

