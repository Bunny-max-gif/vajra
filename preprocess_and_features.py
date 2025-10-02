import pandas as pd
import numpy as np

def make_daily_features(pm_df, met_df):
    """
    Create daily features from hourly PM2.5 and meteorology data.
    
    Parameters:
    -----------
    pm_df : pandas.DataFrame
        DataFrame with 'timestamp' and 'pm25' columns
    met_df : pandas.DataFrame
        DataFrame with 'timestamp', 'temperature', 'relativehumidity', 'windspeed' columns
        
    Returns:
    --------
    pandas.DataFrame
        Daily features including lags and moving averages
    """
    # Convert timestamps to datetime
    pm_df['timestamp'] = pd.to_datetime(pm_df['timestamp'])
    met_df['timestamp'] = pd.to_datetime(met_df['timestamp'])
    
    # Resample to daily frequency
    daily_pm = pm_df.set_index('timestamp').resample('D').mean()
    daily_met = met_df.set_index('timestamp').resample('D').mean()
    
    # Merge PM2.5 and meteorology data
    df = pd.merge(daily_pm, daily_met, left_index=True, right_index=True)
    
    # Create lagged features
    df['pm25_lag_1'] = df['pm25'].shift(1)
    df['pm25_lag_2'] = df['pm25'].shift(2)
    df['pm25_lag_3'] = df['pm25'].shift(3)
    df['pm25_lag_7'] = df['pm25'].shift(7)
    
    # Create moving average features
    df['pm25_ma_3'] = df['pm25'].rolling(window=3).mean()
    
    # Add time features
    df['dayofyear'] = df.index.dayofyear
    
    # Drop rows with NaN values
    df = df.dropna()
    
    return df

if __name__ == "__main__":
    # Create sample test data
    dates = pd.date_range(start='2024-01-01', end='2024-01-10', freq='H')
    
    # Sample PM2.5 data
    pm_df = pd.DataFrame({
        'timestamp': dates,
        'pm25': np.random.normal(50, 20, len(dates))
    })
    
    # Sample meteorological data
    met_df = pd.DataFrame({
        'timestamp': dates,
        'temperature': np.random.normal(25, 5, len(dates)),
        'relativehumidity': np.random.normal(60, 10, len(dates)),
        'windspeed': np.random.normal(10, 3, len(dates))
    })
    
    # Test feature creation
    features_df = make_daily_features(pm_df, met_df)
    print("✅ Feature creation successful")
    print(f"Number of features: {len(features_df.columns)}")
    print(f"Number of daily samples: {len(features_df)}")
    print("\nFeature columns:")
    print(features_df.columns.tolist())