# src/models/train_model.py

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn


def load_features(path: str = "data/processed/air_passengers_lag.csv") -> pd.DataFrame:
    """
    Load the processed dataset with lag features.
    """
    return pd.read_csv(path, parse_dates=["ds"])


def train_test_split_time(df: pd.DataFrame, test_size: int = 24):
    """
    Chronological train-test split.
    The last `test_size` rows form the test set.
    """
    df = df.sort_values("ds")

    train_df = df.iloc[:-test_size]
    test_df = df.iloc[-test_size:]

    feature_cols = [c for c in df.columns if c.startswith("lag_")]
    target_col = "y"

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_test = test_df[feature_cols].values
    y_test = test_df[target_col].values

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, n_estimators=200, max_depth=5):
    """
    Train a Random Forest model for time-series forecasting.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model using MAE and RMSE.
    """
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return mae, rmse


def save_model(model, path: str = "models/random_forest_model.joblib"):
    """
    Save the trained model to disk.
    """
    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to: {path}")


if __name__ == "__main__":
    # Load data
    df = load_features()
    X_train, X_test, y_train, y_test = train_test_split_time(df)

    # Create an MLflow experiment
    mlflow.set_experiment("air_passengers_retraining")

    with mlflow.start_run():
        # Set model parameters
        n_estimators = 200
        max_depth = 5

        # Train the model
        model = train_model(X_train, y_train, n_estimators, max_depth)

        # Evaluate
        mae, rmse = evaluate_model(model, X_test, y_test)

        # Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        # Build an input example for model signature
        input_example = X_train[:1]

        # Log model with signature
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=input_example
        )

        print("Training completed.")
        print(f"MAE: {mae:.3f} - RMSE: {rmse:.3f}")
