# 🧠 Deep Learning with PyTorch

### My deep learning journey — one model at a time.

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=for-the-badge)

> *Built with PyTorch, trained on real-world datasets. Starting from ANN regression, growing into CNNs, RNNs and beyond.*

-----

## 📌 About This Repository

This repository documents my complete **Deep Learning journey from scratch** — every concept learned, every model built, every result achieved. No tutorials copy-pasted. Everything understood, implemented, and tested on real datasets.

Built as a **1st year CSE student**, self-learned from ground up.

-----

## 🗂️ Projects

### ⚡ 1. ANN Regression — Power Plant Energy Prediction

📁 [`ann_regression/`](./ann_regression/)

|Detail       |Value                                      |
|-------------|-------------------------------------------|
|Task         |Regression                                 |
|Dataset      |Combined Cycle Power Plant                 |
|Samples      |9,568                                      |
|Features     |4 (Temperature, Vacuum, Pressure, Humidity)|
|Target       |Net Energy Output (MW)                     |
|Loss Function|MSELoss                                    |
|Optimizer    |Adam                                       |
|**R² Score** |**0.9342 (93.42%)** ✅                      |
|Test MSE     |18.82                                      |

**Highlights:**

- Full PyTorch pipeline from scratch
- StandardScaler preprocessing
- Train/Validation loss tracking
- Best model saving with `torch.save()`
- Predicted vs Actual values comparison

-----

### 🌴 2. ANN Classification — Date Fruit Variety Classification

📁 [`ann_classification/`](./ann_classification/)

|Detail           |Value                                                 |
|-----------------|------------------------------------------------------|
|Task             |Multiclass Classification                             |
|Dataset          |Date Fruit Dataset                                    |
|Samples          |898                                                   |
|Features         |34 morphological features                             |
|Classes          |7 (BERHI, DEGLET, DOKOL, IRAQI, ROTANA, SAFAVI, SOGAY)|
|Loss Function    |CrossEntropyLoss                                      |
|Optimizer        |Adam                                                  |
|**Test Accuracy**|**94.44%** ✅                                          |

**Highlights:**

- LabelEncoder for 7-class target
- `torch.max(outputs, 1)` for class prediction
- `torch.no_grad()` during evaluation
- Loss converged from 1.69 → 0.027 over 100 epochs

-----

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

🔄 In Progress:
⬜ CNN (Convolutional Neural Networks)
⬜ NLP & Text Summarization
⬜ RNN / LSTM
⬜ Transformers
```

-----

## 🛠️ Tech Stack

```python
framework   = "PyTorch"
language    = "Python 3.8+"
libraries   = ["torch", "torch.nn", "torch.optim",
               "torch.utils.data", "sklearn", 
               "pandas", "numpy", "matplotlib"]
environment = "Jupyter Notebook"
```

-----

## 📊 Results Summary

|Project                  |Type      |Metric  |Score     |
|-------------------------|----------|--------|----------|
|Power Plant Energy       |Regression|R² Score|**0.9342**|
|Date Fruit Classification|Multiclass|Accuracy|**94.44%**|

-----

## 🚀 How to Use

**Clone the repository:**

```bash
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning
```

**Install dependencies:**

```bash
pip install torch pandas numpy scikit-learn matplotlib jupyter
```

**Navigate to any project:**

```bash
cd ann_regression
jupyter notebook ann_regression.ipynb
```

-----

## 📁 Repository Structure

```
deep-learning/
│
├── ann_regression/
│   ├── ann_regression.ipynb
│   ├── ann_regression.py     
│   ├── powerplant_data.csv      
│   ├── best_model.pt          
│   └── README.md                
│
├── ann_classification/
│   ├── ann_classification.ipynb  
│   ├── DateFruit_Dataset.csv     
│   └── README.md             
│
└── README.md                    
```

-----

## 🗺️ Roadmap

```
Phase 1 — ANN ✅
├── ANN Regression        ✅ Done (R²=0.93)
└── ANN Classification    ✅ Done (Acc=94.44%)

Phase 2 — CNN 🔄
├── Image Classification (MNIST)
└── Custom CNN project

Phase 3 — NLP 📅
├── Text Summarization (HuggingFace)
└── Flask deployment

Phase 4 — Advanced 📅
├── RNN / LSTM
└── Transformers
```

-----

## 👨‍💻 Author

**Mehran Mushtaq**

- 🎓 1st Year CSE Student
- 🔥 Self-learned Deep Learning
- 📍 Kashmir, India
- 🐙 [GitHub](https://github.com/mehranmushtaq)
- 💻 [LeetCode](https://leetcode.com/u/mehraan1/)

-----

## 🔗 Related Repositories

- [Machine Learning from Scratch & Sklearn](https://github.com/mehranmushtaq/Machine-Learning-with-scikit-learn-and-from-scratch)
- [Exploratory Data Analysis](https://github.com/mehranmushtaq/exploratory-data-analysis)

-----

  "Kashmir — built everything from scratch, one commit at a time." 


-----

⭐ **Star this repo if you find it helpful!**
