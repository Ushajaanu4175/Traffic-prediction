🚦 Traffic Flow Prediction Using TemporalCNN

1️⃣ Project Overview
Predicts future traffic volume using Temporal Convolutional Network (TCN).
Helps in traffic management, route planning, and congestion control.
2️⃣ Dataset Details
Data Source: data/traffic_volume_data.csv
Includes:
✅ Time-based traffic records (timestamps, days, seasons)
✅ Traffic volume (vehicle count per time window)
✅ Weather conditions (temperature, wind speed, etc.)
✅ Road features (lane count, road type, congestion level)
3️⃣ Data Preprocessing
Handled missing values (drop/fill NaNs).
Scaled and normalized features.
Selected relevant features for better accuracy.
Converted data into PyTorch tensors.
4️⃣ Model Architecture
Temporal Convolutional Network (TCN):
✅ 1D Convolutional Layers – Learn short & long-term dependencies.
✅ Causal & Dilated Convolutions – Ensure predictions depend only on past data.
✅ Residual Connections – Improve gradient flow.
✅ Fully Connected Layers – Output final prediction.
Implemented in temporal_cnn_model.py.
5️⃣ Model Training
Loss Function: MSE Loss
Optimizer: Adam (LR = 0.0015)
Batch Size: 512
Epochs: 100
Metrics: MAE, RMSE, R² Score
6️⃣ Model Testing & Predictions
Used test_model.py to evaluate the trained model.
Steps:
✅ Load trained model (models/trained_model.pth).
✅ Preprocess test data.
✅ Make predictions & inverse transform results.
✅ Compare predictions with actual values.
✅ Plot results (Predicted vs. Actual traffic).
7️⃣ Performance Metrics
MAE: 583.38
RMSE: 713.12
R² Score: 0.84
8️⃣ Summary
✅ Built a TemporalCNN model for traffic prediction.
✅ Preprocessed and cleaned traffic data.
✅ Achieved R² = 0.84, indicating strong predictive power.
✅ Visualized predictions vs. actual traffic trends.
✅ Potential improvements for future enhancements.

🚀 This model can optimize traffic systems, reduce congestion, and improve urban mobility!
