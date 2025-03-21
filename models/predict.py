import pandas as pd
import torch
import numpy as np
import pickle
from preprocess_data import preprocess_data
from temporalcnn import TemporalCNN

# ✅ Load Scalers and Encoders
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("models/encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("models/target_scaler.pkl", "rb") as f:
    target_scaler = pickle.load(f)  # ✅ Fix: Use target scaler for traffic volume

# ✅ Load Best Model
model = TemporalCNN(input_dim=69, dropout=0.5)  # Adjust input_dim if needed
model.load_state_dict(torch.load("models/best_model.pth", map_location=torch.device("cpu"), weights_only=True))
model.eval()

# ✅ Function to Fetch Recent Traffic Data
def get_recent_traffic_data():
    try:
        history_df = pd.read_csv("data/traffic_volume_data.csv")
        last_row = history_df.iloc[-1]  # Get last known record
        return {
            "air_pollution_index": last_row["air_pollution_index"],
            "humidity": last_row["humidity"],
            "wind_speed": last_row["wind_speed"],
            "wind_direction": last_row["wind_direction"],
            "visibility_in_miles": last_row["visibility_in_miles"],
            "dew_point": last_row["dew_point"],
            "last_1_hour_traffic": last_row["traffic_volume"],
            "last_2_hour_traffic": last_row["last_1_hour_traffic"],
            "last_3_hour_traffic": last_row["last_2_hour_traffic"],
            "last_4_hour_traffic": last_row["last_3_hour_traffic"],
            "last_5_hour_traffic": last_row["last_4_hour_traffic"],
            "last_6_hour_traffic": last_row["last_5_hour_traffic"],
        }
    except Exception as e:
        print(f"⚠️ Warning: Could not fetch historical data. Error: {e}")
        return {key: 0 for key in [
            "air_pollution_index", "humidity", "wind_speed", "wind_direction",
            "visibility_in_miles", "dew_point", "last_1_hour_traffic",
            "last_2_hour_traffic", "last_3_hour_traffic", "last_4_hour_traffic",
            "last_5_hour_traffic", "last_6_hour_traffic"
        ]}

# ✅ Fixed Inverse Transform for Traffic Prediction
def inverse_transform_traffic(raw_output):
    """
    Correctly inverse transform the predicted traffic volume using the target scaler.
    """
    raw_output = np.array([[raw_output]])  # Ensure correct shape
    predicted_traffic = target_scaler.inverse_transform(raw_output)[0, 0]  # ✅ Fix: Use correct scaler

    return max(0, round(predicted_traffic))  # Ensure traffic volume is non-negative

# ✅ Traffic Prediction Function
def predict_traffic(date_time, is_holiday, temperature, rain_p_h, snow_p_h, clouds_all, weather_type, weather_description):
    # Auto-fill missing values
    recent_data = get_recent_traffic_data()

    # Create input data
    input_data = pd.DataFrame({
        "date_time": [date_time],
        "is_holiday": [1 if is_holiday.lower() == "yes" else 0],
        "temperature": [temperature],
        "rain_p_h": [rain_p_h],
        "snow_p_h": [snow_p_h],
        "clouds_all": [clouds_all],
        "weather_type": [weather_type],
        "weather_description": [weather_description],
        **recent_data  # Auto-filled values
    })
    input_data["traffic_volume"] = 0  # Dummy value for preprocessing

    # ✅ Preprocess Data
    X_input, _ = preprocess_data(input_data, fit_scalers=False)

    # ✅ Convert to Tensor
    X_tensor = X_input.float().unsqueeze(0)  # Ensure correct shape for model

    # ✅ Predict Traffic Volume
    with torch.no_grad():
        raw_output = model(X_tensor).item()
        # print(f"🧠 Raw Model Output: {raw_output}")

    # ✅ Apply Fixed Inverse Scaling
    predicted_traffic = inverse_transform_traffic(raw_output)

    # ✅ Compute Estimated Congestion Percentage
    max_capacity = 500  # Adjust this based on real data
    congestion_percentage = min((predicted_traffic / max_capacity) * 100, 100)  # Ensure max 100%

    # ✅ Classify Traffic Condition **(Based on Congestion Percentage)**
    if congestion_percentage < 40:
        traffic_condition = "🚦 Low Traffic"
    elif 40 <= congestion_percentage < 70:
        traffic_condition = "🚗 Moderate Traffic"
    else:
        traffic_condition = "🚛 High Traffic"

    # ✅ Print Results
    
    print(f"📊 Traffic Condition: {traffic_condition}")
    print(f"📈 Estimated Congestion: {round(congestion_percentage, 1)}%")

    return predicted_traffic, traffic_condition, round(congestion_percentage, 1)

# ✅ Example Prediction
if __name__ == "__main__":
    predicted_traffic, condition, congestion = predict_traffic(
        "2025-03-07 18:30:00", "No", 22, 0, 0, 40, "Clear", "Clear Sky"
    )
