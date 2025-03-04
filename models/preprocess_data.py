import pandas as pd
import torch
import pickle
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler

def preprocess_data(df, fit_scalers=True):
    """Preprocess the dataset for model training/testing."""
    
    # ✅ Convert date_time to useful features
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['hour'] = df['date_time'].dt.hour
    df['day_of_week'] = df['date_time'].dt.dayofweek
    df['month'] = df['date_time'].dt.month
    df.drop(columns=['date_time'], inplace=True)

    # ✅ One-Hot Encoding for categorical features
    categorical_features = ['weather_type', 'weather_description']
    encoded_df = None
    
    if fit_scalers:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_features = encoder.fit_transform(df[categorical_features])
        with open("models/encoder.pkl", "wb") as f:
            pickle.dump(encoder, f)
    else:
        with open("models/encoder.pkl", "rb") as f:
            encoder = pickle.load(f)
        encoded_features = encoder.transform(df[categorical_features])
    
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out())

    # ✅ Numerical features for scaling
    numerical_features = [
        'air_pollution_index', 'humidity', 'wind_speed', 'wind_direction',
        'visibility_in_miles', 'dew_point', 'temperature', 'rain_p_h', 'snow_p_h',
        'clouds_all', 'hour', 'day_of_week', 'month', 'is_holiday',
        'last_1_hour_traffic', 'last_2_hour_traffic', 'last_3_hour_traffic',
        'last_4_hour_traffic', 'last_5_hour_traffic', 'last_6_hour_traffic'
    ]
    
    if fit_scalers:
        scaler = RobustScaler()
        df[numerical_features] = scaler.fit_transform(df[numerical_features])
        with open("models/scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
    else:
        with open("models/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        df[numerical_features] = scaler.transform(df[numerical_features])

    # ✅ Normalize target variable (traffic_volume)
    if fit_scalers:
        target_scaler = MinMaxScaler(feature_range=(0, 1))  # ✅ Fix: Normalize target correctly
        df['traffic_volume'] = target_scaler.fit_transform(df[['traffic_volume']])
        with open("models/target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)
    else:
        with open("models/target_scaler.pkl", "rb") as f:
            target_scaler = pickle.load(f)
        df['traffic_volume'] = target_scaler.transform(df[['traffic_volume']])
    
    # ✅ Combine features
    X = pd.concat([df[numerical_features], encoded_df], axis=1)
    y = df['traffic_volume']

    # ✅ Fix NaNs in case of missing data
    X.fillna(0, inplace=True)

    # ✅ Convert to PyTorch tensors
    return torch.tensor(X.values, dtype=torch.float32), torch.tensor(y.values, dtype=torch.float32).view(-1, 1)
