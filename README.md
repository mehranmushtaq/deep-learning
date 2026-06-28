# Deep Learning with PyTorch

### My deep learning journey — one model at a time.

[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)]()

> *Built with PyTorch, trained on real-world datasets. Starting from ANN regression, growing into CNNs, RNNs and beyond.*

---

## 📌 About This Repository

This repository documents my complete **Deep Learning journey from scratch** — every concept learned, every model built, every result achieved. No tutorials copy-pasted. Everything understood, implemented, and tested on real datasets.

Built as a **1st year CSE student**, self-learned from ground up.

---

## 🗂️ Projects

### ⚡ 1. ANN Regression — Power Plant Energy Prediction

📁 [`ann_regression/`](ann_regression)

| Detail        | Value                                       |
| ------------- | ------------------------------------------- |
| Task          | Regression                                  |
| Dataset       | Combined Cycle Power Plant                  |
| Samples       | 9,568                                       |
| Features      | 4 (Temperature, Vacuum, Pressure, Humidity) |
| Target        | Net Energy Output (MW)                      |
| Loss Function | MSELoss                                     |
| Optimizer     | Adam                                        |
| **R² Score**  | **0.9342 (93.42%)** ✅                       |
| Test MSE      | 18.82                                       |

**Highlights:**

- Full PyTorch pipeline from scratch
- StandardScaler preprocessing
- Train/Validation loss tracking across 100 epochs
- Best model saving with `torch.save()`
- Predicted vs Actual values comparison
- **ML Comparison** — ANN benchmarked against classical regressors (Linear Regression, Ridge, Random Forest, etc.) to show where deep learning wins and where it doesn't

---

### 🌴 2. ANN Classification — Date Fruit Variety Classification

📁 [`ann_classification/`](ann_classification)

| Detail            | Value                                                   |
| ----------------- | ------------------------------------------------------- |
| Task              | Multiclass Classification                               |
| Dataset           | Date Fruit Dataset                                      |
| Samples           | 898                                                     |
| Features          | 34 morphological features                               |
| Classes           | 7 (BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY) |
| Loss Function     | CrossEntropyLoss                                        |
| Optimizer         | Adam                                                    |
| **Test Accuracy** | **94.44%** ✅                                            |

**Highlights:**

- LabelEncoder for 7-class target
- `torch.max(outputs, 1)` for class prediction
- `torch.no_grad()` during evaluation
- Loss converged from 1.69 → 0.027 over 100 epochs
- **ML Comparison** — ANN benchmarked against classical classifiers (Logistic Regression, SVM, KNN, Decision Tree, Random Forest) to compare accuracy and training time

---

### 🖼️ 3. CNN — Image Classification

📁 [`cnn/`](cnn)

#### 3a. CIFAR-10

| Detail            | Value                  |
| ----------------- | ---------------------- |
| Task              | Image Classification   |
| Dataset           | CIFAR-10 (torchvision) |
| Training Images   | 50,000                 |
| Test Images       | 10,000                 |
| Image Size        | 32×32×3 (RGB)          |
| Classes           | 10                     |
| Loss Function     | CrossEntropyLoss       |
| Optimizer         | Adam                   |
| **Test Accuracy** | **75.32%** ✅           |

**Architecture:**

```
Input (3×32×32)
→ Conv2d(3→32) + ReLU + MaxPool(2×2)
→ Conv2d(32→64) + ReLU + MaxPool(2×2)
→ Conv2d(64→128) + ReLU + MaxPool(2×2)
→ Flatten → Linear(2048→256) + ReLU
→ Linear(256→10)
```

**Highlights:**

- 3-block CNN architecture from scratch
- Loss dropped from 1.37 → 0.15 over 10 epochs
- `torch.max(outputs, 1)` for class prediction
- Trained on Apple M4 MPS GPU

---

#### 3b. MNIST — Handwritten Digit Recognition

| Detail            | Value                   |
| ----------------- | ----------------------- |
| Task              | Image Classification    |
| Dataset           | MNIST (torchvision)     |
| Training Images   | 60,000                  |
| Test Images       | 10,000                  |
| Image Size        | 28×28×1 (Grayscale)     |
| Classes           | 10 (digits 0–9)         |
| Loss Function     | CrossEntropyLoss        |
| Optimizer         | Adam                    |

**Architecture:**

```
Input (1×28×28)
→ Conv2d(1→32) + ReLU + MaxPool(2×2)
→ Conv2d(32→64) + ReLU + MaxPool(2×2)
→ Flatten → Linear(→128) + ReLU
→ Linear(128→10)
```

**Highlights:**

- 2-block CNN on grayscale images
- Classic entry-level CNN task — great for understanding kernels & feature maps before tackling CIFAR-10
- Trained on Apple M4 MPS GPU

---

## 🧠 Deep Learning Concepts Covered

```
✅ Perceptron & Neural Network fundamentals
✅ Forward & Backward Propagation
✅ Activation Functions (ReLU, Sigmoid, Tanh, Softmax, Linear)
✅ Loss Functions (MSE, MAE, Huber, BCE, Categorical Cross Entropy)
✅ Optimizers (GD, SGD, Mini-batch GD, Momentum, RMSProp, Adam)
✅ Vanishing Gradient Problem & solutions
✅ ReLU and its variants (Leaky ReLU, PReLU, ELU)
✅ Batch / Iteration / Epoch
✅ Weight Updation & Chain Rule in NN
✅ TensorDataset & DataLoader (PyTorch)
✅ Model saving & loading (torch.save / torch.load)
✅ Convolutional Layers (Conv2d, MaxPool2d, Padding, Stride)
✅ Feature Maps & Kernel operations
✅ Flatten layer → Fully Connected
✅ DL vs ML comparison (regression & classification benchmarks)

🔄 In Progress:
⬜ BatchNorm + Dropout (CNN improvement)
⬜ NLP & Text Summarization
⬜ RNN / LSTM
⬜ Transformers
```

---

## 🛠️ Tech Stack

```
framework   = "PyTorch"
language    = "Python 3.8+"
libraries   = ["torch", "torch.nn", "torch.optim",
               "torch.utils.data", "torchvision",
               "sklearn", "pandas", "numpy", "matplotlib"]
environment = "Jupyter Notebook"
hardware    = "Apple M4 MPS GPU"
```

---

## 📊 Results Summary

| Project                        | Type                 | Metric   | Score      |
| ------------------------------ | -------------------- | -------- | ---------- |
| Power Plant Energy (ANN)       | Regression           | R² Score | **0.9342** |
| Date Fruit Classification (ANN)| Multiclass           | Accuracy | **94.44%** |
| CIFAR-10 (CNN)                 | Image Classification | Accuracy | **75.32%** |
| MNIST (CNN)                    | Digit Classification | Accuracy | TBD ✅      |
| ML Comparison — Regression     | Benchmark            | R² Score | Various    |
| ML Comparison — Classification | Benchmark            | Accuracy | Various    |

---

## 🚀 How to Use

**Clone the repository:**

```bash
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning
```

**Install dependencies:**

```bash
pip install torch torchvision pandas numpy scikit-learn matplotlib jupyter
```

**Navigate to any project:**

```bash
cd ann_regression
jupyter notebook ann_regression.ipynb
```

---

## 📁 Repository Structure

```
deep-learning/
│
├── ann_regression/
│   ├── ann_regression.ipynb         # Power Plant Energy Prediction (ANN)
│   ├── ml_comparison.ipynb          # ML vs ANN regression benchmark
│   ├── powerplant_data.csv          # Dataset
│   ├── best_model.pt                # Saved best model
│   └── README.md
│
├── ann_classification/
│   ├── ann_classification.ipynb     # Date Fruit Classification (ANN)
│   ├── ml_comparison.ipynb          # ML vs ANN classification benchmark
│   ├── DateFruit_Dataset.csv        # Dataset
│   └── README.md
│
├── cnn/
│   ├── cnn_for_cifar10.ipynb        # CIFAR-10 Image Classification
│   ├── cnn_for_mnist.ipynb          # MNIST Digit Classification
│   └── README.md
│
└── README.md                        # You are here
```

---

## 👨‍💻 Author

**Mehran Mushtaq**

- 🎓 1st Year CSE Student
- 🔥 Self-learned Deep Learning
- 📍 Kashmir, India
- 🐙 [GitHub](https://github.com/mehranmushtaq)
- 💻 [LeetCode](https://leetcode.com/u/mehraan1/)

---

## 🔗 Related Repositories

- [Machine Learning from Scratch & Sklearn](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch)
- [Exploratory Data Analysis](https://github.com/mehranmushtaq/exploratory-data-analysis)

---

## *"From Kashmir — built everything from scratch, one commit at a time."*

⭐ **Star this repo if you find it helpful!**

