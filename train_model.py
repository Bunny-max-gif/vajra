import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load your processed data
# Replace this with actual data loading logic
X = pd.DataFrame({
    'temperature': np.random.normal(25, 5, 1000),
    'relativehumidity': np.random.normal(60, 10, 1000),
    'windspeed': np.random.normal(10, 3, 1000),
    'pm25_lag_1': np.random.normal(50, 20, 1000),
    'pm25_lag_2': np.random.normal(50, 20, 1000),
    'pm25_lag_3': np.random.normal(50, 20, 1000),
    'pm25_lag_7': np.random.normal(50, 20, 1000),
    'pm25_ma_3': np.random.normal(50, 20, 1000),
    'dayofyear': np.random.randint(1, 366, 1000)
})
y = np.random.normal(50, 20, 1000)  # Target variable

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and feature names
joblib.dump({
    'model': model,
    'features': X.columns.tolist()
}, 'model.joblib')