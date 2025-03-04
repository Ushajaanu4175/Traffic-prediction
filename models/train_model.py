import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from gnn_lstm_model import TemporalCNN
from preprocess_data import preprocess_data
from torch_lr_finder import LRFinder  # ✅ Import Learning Rate Finder

# ✅ Load dataset
df = pd.read_csv('data/traffic_volume_data.csv')

# ✅ Preprocess data
X, y = preprocess_data(df, fit_scalers=True)

# ✅ Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# ✅ Create DataLoader (Fixed batch size)
batch_size =  128
 # Reduced batch size to allow more LR finder iterations
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Training Features Count: {X.shape[1]}")

# ✅ Initialize Model
input_dim = X.shape[1]
model = TemporalCNN(input_dim=input_dim, dropout=0.5)

# ✅ Use CPU (since you don't have a GPU)
device = torch.device("cpu")
model.to(device)

# ✅ RMSprop Optimizer
optimizer = optim.RMSprop(model.parameters(), lr=0.00159, alpha=0.9, eps=1e-8, weight_decay=1e-4)

# ✅ Loss Function (Mean Squared Error)
criterion = nn.MSELoss()

# ✅ Learning Rate Finder
lr_finder = LRFinder(model, optimizer, criterion, device=device)

# ✅ Find Best Learning Rate (Fixed issue with train_loader)
lr_finder.range_test(dataloader, end_lr=0.1, num_iter=100)  # Increased num_iter
lr_finder.plot()  # ✅ Check the plot and update LR
lr_finder.reset()

# ✅ Learning Rate Scheduler (Cosine Annealing)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-6)

# ✅ Early Stopping Variables
best_loss = float('inf')
patience = 10
patience_counter = 0

# ✅ Training Loop
epochs = 100
for epoch in range(epochs):
    model.train()  # Ensure model is in training mode
    epoch_loss = 0
    
    for batch_X, batch_y in dataloader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)  # Ensure data is on CPU

        optimizer.zero_grad()
        
        # ✅ Forward Pass
        output = model(batch_X)
        
        # ✅ Compute Loss
        loss = criterion(output, batch_y)
        loss.backward()
        
        # ✅ Gradient Clipping (Prevents Exploding Gradients)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        
        # ✅ Optimizer Step
        optimizer.step()
        
        # ✅ Track Epoch Loss
        epoch_loss += loss.item()

    # ✅ Compute Average Loss
    avg_loss = epoch_loss / len(dataloader)

    # ✅ Learning Rate Decay
    scheduler.step()

    # ✅ Early Stopping Check
    if avg_loss < best_loss:
        best_loss = avg_loss
        patience_counter = 0
        torch.save(model.state_dict(), "models/best_model.pth")  # Save Best Model
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹ Early stopping at epoch {epoch+1}. Best Loss: {best_loss:.6f}")
            break

    # ✅ Print Training Progress
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.6f}")

print("🎯 Training Complete! Best Model Saved.")
