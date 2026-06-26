# 🧠 Deep Learning with CNNs — CIFAR-10 & MNIST

### Deep Learning | PyTorch | Computer Vision

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Projects](https://img.shields.io/badge/Projects-2-purple?style=for-the-badge)

---

## 📁 Projects Overview

| Project | Dataset | Task | Accuracy | Epochs |
|---------|---------|------|----------|--------|
| [CNN for CIFAR-10](#-project-1--cnn-for-cifar-10) | CIFAR-10 | 10-class image classification | **75.32%** | 10 |
| [CNN for MNIST](#-project-2--cnn-for-mnist) | MNIST | Handwritten digit recognition | **99.10%** | 5 |

Both models are built **from scratch** using PyTorch, with no pretrained weights.

---

## CNN for CIFAR-10

### Overview

A CNN that classifies 32×32 RGB images into 10 categories — from airplanes to trucks. Achieves **75.32% accuracy**, well above the 10% random baseline.

### The 10 Classes

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | ✈️ Airplane | 5 | 🐶 Dog |
| 1 | 🚗 Automobile | 6 | 🐸 Frog |
| 2 | 🐦 Bird | 7 | 🐴 Horse |
| 3 | 🐱 Cat | 8 | 🚢 Ship |
| 4 | 🦌 Deer | 9 | 🚛 Truck |

### Dataset

- **Source**: CIFAR-10 (via torchvision)
- **Training samples**: 50,000 images
- **Test samples**: 10,000 images
- **Image size**: 32×32×3 (RGB)

### Preprocessing

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # Scales pixel values from [0,1] to [-1,1]
])
```

### Model Architecture

```
Input (3 × 32 × 32) — RGB image
        ↓
Conv2d(3→32, kernel=3, padding=1) + ReLU + MaxPool(2×2)
        ↓ (32 × 16 × 16)
Conv2d(32→64, kernel=3, padding=1) + ReLU + MaxPool(2×2)
        ↓ (64 × 8 × 8)
Conv2d(64→128, kernel=3, padding=1) + ReLU + MaxPool(2×2)
        ↓ (128 × 4 × 4)
Flatten → 2048
        ↓
Linear(2048 → 256) + ReLU
        ↓
Linear(256 → 10)
        ↓
Output — 10 class scores
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Epochs | 10 |
| Batch Size | 64 |

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **75.32%** ✅ |
| Final Training Loss | 0.1514 |

#### Loss Curve

| Epoch | Loss |
|-------|------|
| 1/10 | 1.3688 |
| 2/10 | 0.9307 |
| 3/10 | 0.7433 |
| 4/10 | 0.6152 |
| 5/10 | 0.5067 |
| 6/10 | 0.4105 |
| 7/10 | 0.3243 |
| 8/10 | 0.2581 |
| 9/10 | 0.1922 |
| 10/10 | **0.1514** |

Loss dropped **89% from epoch 1 to 10** — excellent convergence ✅

### Benchmarks

| Model | Accuracy |
|-------|----------|
| Random Guess | 10% |
| Simple CNN (this project) | **75.32%** ✅ |
| CNN + BatchNorm + Dropout | ~82% |
| ResNet-style | ~90%+ |

### How to Run

```bash
# 1. Navigate to the project folder
cd cnn

# 2. Install dependencies
pip install torch torchvision

# 3. Run the notebook
jupyter notebook cnn_for_cifar10.ipynb
```

---

## CNN for MNIST

### Overview

A CNN that recognizes handwritten digits (0–9) from 28×28 grayscale images. Achieves **99.10% accuracy** in just 5 epochs — near state-of-the-art for a simple CNN.

### Dataset

- **Source**: MNIST (via torchvision)
- **Training samples**: 60,000 images
- **Test samples**: 10,000 images
- **Image size**: 28×28×1 (Grayscale)
- **Classes**: 10 digits (0–9)

### Preprocessing

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    # Grayscale: single channel normalization
])
```

### Model Architecture

```
Input (1 × 28 × 28) — Grayscale image
        ↓
Conv2d(1→32, kernel=3, padding=1) + ReLU + MaxPool(2×2)
        ↓ (32 × 14 × 14)
Conv2d(32→64, kernel=3, padding=1) + ReLU + MaxPool(2×2)
        ↓ (64 × 7 × 7)
Conv2d(64→128, kernel=3, padding=1) + ReLU + MaxPool(2×2)
        ↓ (128 × 3 × 3)
Flatten → 1152  (3 × 3 × 128)
        ↓
Linear(1152 → 256) + ReLU
        ↓
Linear(256 → 10)
        ↓
Output — 10 digit scores
```

> Key difference from CIFAR-10: input channels changed from 3 (RGB) → 1 (grayscale), and flatten size adjusts accordingly to 3×3×128 = 1152.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Epochs | 5 |
| Batch Size | 64 |

### Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **99.10%** ✅ |
| Final Training Loss | 0.0182 |

#### Loss Curve

| Epoch | Loss |
|-------|------|
| 1/5 | 0.1517 |
| 2/5 | 0.0416 |
| 3/5 | 0.0304 |
| 4/5 | 0.0227 |
| 5/5 | **0.0182** |

Loss dropped **88% from epoch 1 to 5** — extremely fast convergence ✅

### How to Run

```bash
# 1. Navigate to the project folder
cd cnn

# 2. Install dependencies
pip install torch torchvision matplotlib numpy

# 3. Run the notebook
jupyter notebook cnn_for_mnist.ipynb
```

---

## 🔑 Key Learnings (Both Projects)

- **Conv2d** extracts spatial features from images using learnable kernels
- **MaxPool2d** reduces spatial dimensions while keeping important features
- **Padding=1** keeps spatial size the same after convolution
- **Flatten** converts 3D feature maps → 1D vector for fully-connected layers
- **CrossEntropyLoss** handles multiclass classification natively
- Grayscale images use **1 input channel**; RGB images use **3**
- CNNs are **translation invariant** — detect features anywhere in an image
- MNIST is simpler (grayscale, clean digits) → fewer epochs needed vs CIFAR-10

---

## 🔮 Next Improvements

```python
# Add BatchNormalization after each Conv layer
nn.BatchNorm2d(32)   # Stabilizes training, faster convergence

# Add Dropout before FC layers
nn.Dropout(0.5)      # Prevents overfitting

# Expected accuracy boost:
# CIFAR-10: 75% → 80-85% 🎯
# MNIST:    99% → 99.5%+ 🎯
```

---

## 📁 Project Structure

```
cnn/
│
├── cnn_for_cifar10.ipynb    # CIFAR-10 notebook
├── cnn_for_mnist.ipynb      # MNIST notebook
├── data/                    # Datasets auto-downloaded here
└── README.md                # This file
```

---

## 👨‍💻 Author

**Mehran Mushtaq**

- 🎓 1st Year CSE Student
- 🔥 Self-learned Deep Learning
- 📍 Kashmir, India
- 🐙 [GitHub](https://github.com/mehranmushtaq)

> *"Kashmir — built everything from scratch, one commit at a time."* 🚀

---

## 🔗 Related Projects

- [ANN Regression — Power Plant Energy Prediction](../ann_regression/)
- [ANN Classification — Date Fruit Variety Classification](../ann_classification/)

---

[![Open CIFAR-10 in nbviewer](https://img.shields.io/badge/Open%20CIFAR--10-nbviewer-orange)](https://nbviewer.org/github/mehranmushtaq/deep-learning/blob/main/cnn/cnn_for_cifar10.ipynb)

⭐ **Star this repo if you found it helpful!**
