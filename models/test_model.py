import torch
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from gnn_lstm_model import TemporalCNN  # Ensure model is imported
from preprocess_data import preprocess_data

# ✅ Register the model class before loading
torch.serialization.add_safe_globals([TemporalCNN])

# ✅ Load model
input_dim = 69  # Ensure this matches training input dimension
model = torch.load("models/trained_model.pth", weights_only=False)  # Load full model
model.eval()

# ✅ Load test data
df_test = pd.read_csv("data/traffic_volume_data.csv").sample(40)
X_test, y_test = preprocess_data(df_test, fit_scalers=False)

# ✅ Check for NaN values
print("Any NaNs in X_test?", pd.DataFrame(X_test.numpy()).isna().sum().sum())
print("Any NaNs in y_test?", pd.DataFrame(y_test.numpy()).isna().sum().sum())
print(f"✅ Testing Features Count: {X_test.shape[1]}")  # Ensure it matches training

# ✅ Make Predictions
with torch.no_grad():
    predictions = model(X_test).numpy()
    print("Raw Model Predictions:", predictions)

# ✅ Load target scaler and inverse transform predictions
with open("models/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)

predictions = target_scaler.inverse_transform(predictions)
y_test_inv = target_scaler.inverse_transform(y_test.numpy())
predictions = predictions[:len(y_test_inv)]
print("Predictions:", predictions)

# ✅ Calculate Performance Metrics
mae = mean_absolute_error(y_test_inv, predictions)
rmse = np.sqrt(mean_squared_error(y_test_inv, predictions))
r2 = r2_score(y_test_inv, predictions)

print(f"\n✅ Model Performance Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# ✅ Plot results
plt.figure(figsize=(8, 5))
plt.plot(predictions, label="Predicted Traffic Volume")
plt.plot(y_test_inv, label="Actual Traffic Volume")
plt.legend()
plt.title("Traffic Volume Predictions")
plt.xlabel("Time")
plt.ylabel("Traffic Volume")
plt.show()
