import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Compute on-time delivery rate
    df = df.copy()
    df['on_time_rate'] = df['on_time_deliveries'] / df['total_deliveries']
    # Select numeric feature columns for modeling
    feature_cols = ['on_time_rate', 'quality_failures', 'financial_health_score', 'sentiment_score']
    return df[feature_cols]

def scale_features(X: pd.DataFrame, scaler: StandardScaler=None):
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    return X_scaled, scaler
