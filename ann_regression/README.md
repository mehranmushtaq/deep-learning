# 🔬 Deep Learning Models — PyTorch

> A growing collection of deep learning projects built from scratch using PyTorch.  
> Each project includes a Jupyter Notebook for exploration and a `.py` file for clean implementation.

-----

## Project

|Project                                                                   |Type      |Dataset |R² Score|Status    |
|--------------------------------------------------------------------------|----------|--------|--------|----------|
|[ANN Regression — Power Plant](#ann-regression--power-plant-energy-output)|Regression|UCI CCPP|0.9341  |✅ Complete|

-----

##  ANN Regression — Power Plant Energy Output

###  Problem Statement

Predict the **net hourly electrical energy output (PE)** of a Combined Cycle Power Plant using ambient environmental sensor readings.

This is a classic regression problem that demonstrates how a simple feedforward neural network can outperform traditional ML models on structured/tabular data.

-----

### 📊 Dataset

**Source:** [UCI Machine Learning Repository — CCPP Dataset](https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant)

|Feature|Description                   |Unit    |
|-------|------------------------------|--------|
|AT     |Ambient Temperature           |°C      |
|V      |Exhaust Vacuum                |cm Hg   |
|AP     |Ambient Pressure              |millibar|
|RH     |Relative Humidity             |%       |
|**PE** |**Net Energy Output (Target)**|**MW**  |

- **Total Samples:** 9,568
- **Train / Test Split:** 80% / 20%
- **Null Values:** None

-----

###  Model Architecture

```
Input Layer  →  4 features (AT, V, AP, RH)
     ↓
Hidden Layer 1  →  Linear(4 → 6)  +  ReLU
     ↓
Hidden Layer 2  →  Linear(6 → 6)  +  ReLU
     ↓
Output Layer    →  Linear(6 → 1)  →  PE (predicted)
```

**Why this architecture?**  
For a 4-feature tabular regression task, a lightweight 2-hidden-layer network is sufficient. Deeper networks would overfit on this dataset size. ReLU activations introduce non-linearity to capture complex feature interactions.

-----

### ⚙️Training Configuration

|Parameter         |Value                                    |
|------------------|-----------------------------------------|
|Loss Function     |MSELoss                                  |
|Optimizer         |Adam                                     |
|Learning Rate     |0.001 (default)                          |
|Batch Size        |32                                       |
|Epochs            |100                                      |
|Data Preprocessing|StandardScaler (zero mean, unit variance)|
|Best Model Saving |✅ Checkpoint on lowest validation loss   |

-----

### 📈 Results

|Metric      |Value     |
|------------|----------|
|Training MSE|~20.49    |
|Testing MSE |~19.06    |
|**R² Score**|**0.9341**|


> An R² of **0.934** means the model explains **93.4% of the variance** in power plant energy output — strong performance for a 2-hidden-layer network with no hyperparameter tuning.

**Loss Curve:**

The training and validation loss curves converge cleanly with no signs of overfitting. Validation loss tracking was done correctly on the **held-out test set** using `test_loader`.

-----

### Tech Stack

- Python 3.x
- PyTorch
- Pandas / NumPy
- Scikit-learn (train/test split, StandardScaler, R² score)
- Matplotlib

-----

### How to Run

```bash
# Clone the repo
git clone https://github.com/mehranmushtaq/deep-learning.git
cd deep-learning/ann_regression

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter lab ann_regression.ipynb

# OR run the Python script
python ann_regression.py
```

-----

###  Requirements

```
torch
pandas
numpy
scikit-learn
matplotlib
jupyterlab
```

-----

### 🧩 Key Learnings

- How to build a custom ANN using `nn.Module` in PyTorch
- Importance of feature scaling before training neural networks
- Using `DataLoader` and `TensorDataset` for efficient batch training
- Tracking and comparing training vs. validation loss per epoch
- Saving and loading the best model checkpoint with `torch.save` / `torch.load`
- Evaluating regression models with MSE and R² score

-----

###  Future Improvements

-  Hyperparameter tuning (learning rate, hidden units, layers)
-  Add dropout regularization
-  Experiment with deeper architectures
-  Compare with scikit-learn baselines (Random Forest, XGBoost)
-  Deploy as a simple web API using FastAPI

-----

## 👤 Author

**Mehran Mushtaq**  
📧 mehraan551@gmail.com.com  
🔗 [GitHub](https://github.com/mehranmushtaq)

-----

## 📄 License

This project is open source under the [MIT License](LICENSE).

-----

*⭐ If you found this useful, consider starring the repo!*
