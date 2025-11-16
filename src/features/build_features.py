# src/features/build_features.py

import pandas as pd
from pathlib import Path


def load_raw(path: str = "data/raw/air_passengers.csv") -> pd.DataFrame:
    """
    Load the raw AirPassengers dataset.
    """
    return pd.read_csv(path, parse_dates=["ds"])


def make_lag_features(df: pd.DataFrame, n_lags: int = 12) -> pd.DataFrame:
    """
    Create lag features for time-series forecasting.
    Example:
        lag_1 = y(t-1)
        lag_2 = y(t-2)
        ...
        lag_n = y(t-n)
    """
    df = df.copy()
    for lag in range(1, n_lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)

    # Remove rows with NaN (first n_lags rows)
    df = df.dropna().reset_index(drop=True)
    return df


def save_processed(df: pd.DataFrame, path: str = "data/processed/air_passengers_lag.csv"):
    """
    Save the processed dataset with lag features.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed dataset saved to: {path}")


if __name__ == "__main__":
    print(">>> Running build_features.py")
    df_raw = load_raw()
    print("Raw shape:", df_raw.shape)
    df_features = make_lag_features(df_raw, n_lags=12)
    print("Features shape:", df_features.shape)
    save_processed(df_features)
