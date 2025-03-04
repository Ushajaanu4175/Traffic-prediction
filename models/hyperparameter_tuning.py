import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import json
import pandas as pd
from gnn_lstm_model import TemporalCNN
from preprocess_data import preprocess_data

# ✅ Load and preprocess dataset
df = pd.read_csv("data/traffic_volume_data.csv")
X, y = preprocess_data(df, fit_scalers=True)

# ✅ Split into train & validation sets (80% train, 20% validation)
train_size = int(0.8 * X.shape[0])
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

# ✅ Convert to tensors
X_train, X_val = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_val, dtype=torch.float32)
y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

# ✅ Define evaluation function
def train_and_evaluate(model, optimizer):
    """Train for a few epochs and return validation loss."""
    criterion = nn.MSELoss()
    model.train()
    
    for epoch in range(5):  # 🔥 Train for only 5 epochs to evaluate performance
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
    
    # ✅ Compute validation loss
    model.eval()
    with torch.no_grad():
        val_output = model(X_val)
        val_loss = criterion(val_output, y_val)
    
    return val_loss.item()

# ✅ Define objective function for Optuna
def objective(trial):
    """Optimize hyperparameters using Optuna."""
    
    num_filters = trial.suggest_int("num_filters", 32, 128, step=16)
    kernel_size = trial.suggest_int("kernel_size", 3, 5)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 5e-4, log=True)
    
    # ✅ Initialize model with hyperparameters
    model = TemporalCNN(input_dim=X.shape[1], num_filters=num_filters, kernel_size=kernel_size, dropout=dropout)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)

    # ✅ Train & evaluate model
    return train_and_evaluate(model, optimizer)

# ✅ Run Optuna Optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# ✅ Save best hyperparameters
best_params = study.best_params
with open('models/best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f)

print("✅ Best Hyperparameters Saved:", best_params)
