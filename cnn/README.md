# 🖼️ CIFAR-10 Image Classification using CNN

### Deep Learning | PyTorch | Computer Vision

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-75.32%25-brightgreen?style=for-the-badge)
![Classes](https://img.shields.io/badge/Classes-10-orange?style=for-the-badge)
![Images](https://img.shields.io/badge/Images-60,000-blue?style=for-the-badge)

-----

## Project Overview

This project implements a **Convolutional Neural Network (CNN)** from scratch using **PyTorch** to classify images from the **CIFAR-10 dataset** into 10 different categories.

The model achieves **75.32% accuracy** on the test set — well above the random baseline of 10% — demonstrating the power of CNNs for image recognition tasks.

-----

## The 10 Classes

|Label|Class       |Label|Class  |
|-----|------------|-----|-------|
|0    |✈️ Airplane  |5    |🐶 Dog  |
|1    |🚗 Automobile|6    |🐸 Frog |
|2    |🐦 Bird      |7    |🐴 Horse|
|3    |🐱 Cat       |8    |🚢 Ship |
|4    |🦌 Deer      |9    |🚛 Truck|

-----

## Dataset

- **Source**: CIFAR-10 (built into torchvision)
- **Training samples**: 50,000 images
- **Test samples**: 10,000 images
- **Image size**: 32×32×3 (RGB)
- **Classes**: 10

### Preprocessing:

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # Scales pixel values from [0,1] to [-1,1]
])
```

-----

## Model Architecture

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

### Architecture Details:

|Layer    |Input   |Output  |Kernel|
|---------|--------|--------|------|
|Conv2d 1 |3×32×32 |32×32×32|3×3   |
|MaxPool 1|32×32×32|32×16×16|2×2   |
|Conv2d 2 |32×16×16|64×16×16|3×3   |
|MaxPool 2|64×16×16|64×8×8  |2×2   |
|Conv2d 3 |64×8×8  |128×8×8 |3×3   |
|MaxPool 3|128×8×8 |128×4×4 |2×2   |
|Flatten  |128×4×4 |2048    |—     |
|Linear 1 |2048    |256     |—     |
|Linear 2 |256     |10      |—     |

-----

## ⚙️ Training Configuration

|Parameter    |Value           |
|-------------|----------------|
|Loss Function|CrossEntropyLoss|
|Optimizer    |Adam            |
|Epochs       |10              |
|Batch Size   |64              |
|Padding      |Same (padding=1)|

-----

## 📈 Results

|Metric             |Value       |
|-------------------|------------|
|**Test Accuracy**  |**75.32%** ✅|
|Final Training Loss|0.1514      |

### Loss Curve:

|Epoch|Loss      |
|-----|----------|
|1/10 |1.3688    |
|2/10 |0.9307    |
|3/10 |0.7433    |
|4/10 |0.6152    |
|5/10 |0.5067    |
|6/10 |0.4105    |
|7/10 |0.3243    |
|8/10 |0.2581    |
|9/10 |0.1922    |
|10/10|**0.1514**|

Loss dropped **89% from epoch 1 to 10** — excellent convergence ✅

-----

## 📊 Accuracy Benchmarks

|Model                    |Accuracy    |
|-------------------------|------------|
|Random Guess             |10%         |
|Simple CNN (this project)|**75.32%** ✅|
|CNN + BatchNorm + Dropout|~82%        |
|ResNet-style             |~90%+       |

-----

## 🚀 How to Run

**1. Clone the repository**

```bash
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning/cnn
```

**2. Install dependencies**

```bash
pip install torch torchvision
```

**3. Run the notebook**

```bash
jupyter notebook cnn_for_cifar10.ipynb
```

> Dataset downloads automatically via torchvision ✅

-----

## 📁 Project Structure

```
cnn_cifar10/
│
├── cnn_for_cifar10.ipynb    # Main notebook
├── data/                    # CIFAR-10 auto-downloaded here
└── README.md                # Project documentation
```

-----

## 🔑 Key Learnings

- **Conv2d** extracts spatial features from images using learnable kernels
- **MaxPool2d** reduces spatial dimensions while keeping important features
- **Padding=1** keeps spatial size same after convolution
- **Flatten** converts 3D feature maps → 1D vector for FC layers
- **CrossEntropyLoss** handles multiclass classification natively
- RGB images have **3 input channels** unlike tabular data
- CNNs are **translation invariant** — detect features anywhere in image

-----

## 🔮 Next Improvements

```python
# Add BatchNormalization after each Conv layer
nn.BatchNorm2d(32)   # Stabilizes training, faster convergence

# Add Dropout before FC layers
nn.Dropout(0.5)      # Prevents overfitting

# Expected accuracy boost → 80-85% 🎯
```

-----

## 👨‍💻 Author

**Mehran Mushtaq**

- 🎓 1st Year CSE Student
- 🔥 Self-learned Deep Learning
- 📍 Kashmir, India
- 🐙 [GitHub](https://github.com/mehranmushtaq)

> *“From Tral, Kashmir — built everything from scratch, one commit at a time.”* 🚀

-----

## 🔗 Related Projects

- [ANN Regression — Power Plant Energy Prediction](../ann_regression/)
- [ANN Classification — Date Fruit Variety Classification](../ann_classification/)

-----
[![Open in nbviewer](https://img.shields.io/badge/Open%20in-nbviewer-orange)](https://nbviewer.org/github/mehranmushtaq/deep-learning/blob/main/cnn/cnn_for_cifar10.ipynb)


⭐ **Star this repo if you found it helpful!**
