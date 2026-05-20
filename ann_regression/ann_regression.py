import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

class ANN(nn.Module):
    def __init__(self, input_dim):
        super(ANN, self).__init__()
        self.model = nn.Sequential(
            # 1st Hidden Layer
            nn.Linear(input_dim, 6),
            nn.ReLU(),
            
            # 2nd Hidden Layer
            nn.Linear(6, 6),
            nn.ReLU(),
            
            # Output Layer (Linear activation for continuous regression value)
            nn.Linear(6, 1)
        )
        
    def forward(self, x):
        return self.model(x)

def main():
    # --- Configuration Hyperparameters ---
    DATA_PATH = "powerplant_data.csv"
    MODEL_SAVE_PATH = "best_model.pt"
    BATCH_SIZE = 32
    EPOCHS = 100
    RANDOM_STATE = 42

    print("=" * 60)
    print("      STARTING PYTORCH ANN REGRESSION PIPELINE      ")
    print("=" * 60)

  
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Missing dataset! Please place '{DATA_PATH}' in this directory.")

    print("\n[1/5] Loading and preparing dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Features (AT, V, AP, RH) and Target (PE)
    X = df.drop("PE", axis=1)
    y = df["PE"]
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert numpy arrays to PyTorch Float Tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    
    # Create PyTorch DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    input_dim = X_train.shape[1]
    print(f"--> Data split successful: {len(X_train)} train rows, {len(X_test)} test rows.")


    print("\n[2/5] Initializing Neural Network environment...")
    model = ANN(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    
    best_val_loss = float("inf")
    training_losses = []
    validation_losses = []

   
    print(f"\n[3/5] Commencing training loop for {EPOCHS} epochs...\n")
    
    for epoch in range(EPOCHS):
        # Training Pass
        model.train()
        running_train_loss = 0.0
        
        for xb, yb in train_loader:
            optimizer.zero_grad()          # Clear accumulated gradients
            outputs = model(xb)            # Forward propagation
            loss = criterion(outputs, yb)  # Calculate MSE loss
            loss.backward()                # Backpropagation
            optimizer.step()               # Adjust structural weights
            
            running_train_loss += loss.item()
            
        epoch_train_loss = running_train_loss / len(train_loader)
        training_losses.append(epoch_train_loss)
        
        # Validation Evaluation Pass
        model.eval()
        running_val_loss = 0.0
        
        with torch.no_grad():              # Turn off gradient tracking
            for xb, yb in test_loader:
                outputs = model(xb)
                loss = criterion(outputs, yb)
                running_val_loss += loss.item()
                
        epoch_val_loss = running_val_loss / len(test_loader)
        validation_losses.append(epoch_val_loss)
        
        # Validation Model Checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            
        # Log training diagnostics every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch: {epoch+1:3d}/{EPOCHS} | Training Loss: {epoch_train_loss:10.2f} | Validation Loss: {epoch_val_loss:10.2f}")


    print(f"\n[4/5] Training finished. Loading optimized parameter state from '{MODEL_SAVE_PATH}'...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()


    print("\n[5/5] Calculating final generalization matrices...")
    with torch.no_grad():
        train_preds = model(X_train_tensor)
        test_preds = model(X_test_tensor)
        
        final_train_mse = criterion(train_preds, y_train_tensor).item()
        final_test_mse = criterion(test_preds, y_test_tensor).item()
        final_r2 = r2_score(y_test, test_preds.numpy())
        
    print("-" * 60)
    print(f"Final Training Set MSE : {final_train_mse:.4f}")
    print(f"Final Validation Set MSE : {final_test_mse:.4f}")
    print(f"System Model R² Score    : {final_r2:.4f} ({final_r2 * 100:.1f}% Variance Explained)")
    print("-" * 60)

  
    print("\nSample Output Comparison Matrix:")
    predicted_vals = test_preds.numpy().flatten()
    actual_vals = y_test.values
    
    comparison_df = pd.DataFrame({
        "Predicted Energy (PE)": predicted_vals,
        "Actual Energy (PE)": actual_vals,
        "Absolute Error": np.abs(predicted_vals - actual_vals)
    })
    print(comparison_df.head(5).to_string(index=False))
    print("=" * 60)

if __name__ == "__main__":
    main()