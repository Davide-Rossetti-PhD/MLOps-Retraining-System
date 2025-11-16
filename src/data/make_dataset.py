# src/data/make_dataset.py

import os
from pathlib import Path
import pandas as pd
from statsmodels.datasets import get_rdataset


def load_air_passengers() -> pd.DataFrame:
    """
    Load the 'AirPassengers' time-series dataset and convert it
    into a DataFrame with two columns:
        - ds: datetime index
        - y: observed value
    """
    data = get_rdataset("AirPassengers")
    ts = data.data["value"]

    df = pd.DataFrame({
        "ds": pd.date_range(start="1949-01-01", periods=len(ts), freq="MS"),  # ds --> date stamp
        "y": ts.values,   # number passengers every month
    })

    return df


def save_raw_dataset(df: pd.DataFrame, path: str = "data/raw/air_passengers.csv"):
    """
    Save the dataset as a CSV file inside data/raw/.
    Creates the folder automatically if it does not exist.
    """
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Raw dataset saved to: {path}")


if __name__ == "__main__":
    df = load_air_passengers()
    save_raw_dataset(df)
