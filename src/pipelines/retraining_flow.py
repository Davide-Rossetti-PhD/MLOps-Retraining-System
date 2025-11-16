# src/pipelines/retraining_flow.py

import sys
import os

# Make sure the project root is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from prefect import task, flow
import pandas as pd
from scipy.stats import ks_2samp


# Import local modules
from src.data.make_dataset import load_air_passengers, save_raw_dataset
from src.features.build_features import load_raw, make_lag_features, save_processed
from src.models.train_model import (
    load_features,
    train_test_split_time,
    train_model,
    evaluate_model,
    save_model
)


# -------------------------
# TASKS
# -------------------------

@task
def ingest_data():
    print("\n📥 [INGEST] Loading raw AirPassengers dataset...")
    df = load_air_passengers()
    print(f"Raw shape: {df.shape}")
    print(df.head())

    save_raw_dataset(df)
    print("Raw dataset saved to: data/raw/air_passengers.csv")
    
    return "data/raw/air_passengers.csv"


@task
def generate_features(raw_path: str):
    print("\n🧱 [FEATURES] Creating lag features (12 months)...")
    
    df_raw = load_raw(raw_path)
    print(f"Loaded raw data: {df_raw.shape}")

    df_feat = make_lag_features(df_raw, n_lags=12)
    print(f"Processed data shape: {df_feat.shape}")
    print(df_feat.head())

    save_processed(df_feat)
    print("Processed dataset saved to: data/processed/air_passengers_lag.csv")
    
    return "data/processed/air_passengers_lag.csv"


@task
def train_and_evaluate(processed_path: str):
    import mlflow
    import mlflow.sklearn

    print("\n🤖 [TRAIN] Training model with MLflow tracking...")

    df = load_features(processed_path)
    X_train, X_test, y_train, y_test = train_test_split_time(df)

    mlflow.set_experiment("air_passengers_retraining")

    with mlflow.start_run():

        model = train_model(X_train, y_train)
        mae, rmse = evaluate_model(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)

        print(f"Model MAE: {mae:.3f}, RMSE: {rmse:.3f}")

        input_example = X_train[:1]

        mlflow.sklearn.log_model(
            model,
            "model",
            input_example=input_example
        )

        save_model(model)

    return mae, rmse



@task
def check_drift(mae: float, threshold: float = 20.0):
    print("\n🔍 [DRIFT] Checking drift threshold...")
    drift = mae > threshold
    print(f"Threshold: {threshold} | Current MAE: {mae:.3f}")
    print(f"Drift detected: {drift}")
    return drift


@task
def ks_drift_test(processed_path: str, reference_path: str = "data/processed/air_passengers_lag.csv", alpha=0.05):
    print("\n🔬 [DRIFT] Running KS drift test...")

    df_new = pd.read_csv(processed_path)
    df_ref = pd.read_csv(reference_path)

    drift_results = {}
    drift_flag = False

    feature_cols = [c for c in df_new.columns if c.startswith("lag_")]

    for col in feature_cols:
        stat, pvalue = ks_2samp(df_new[col], df_ref[col])
        drift_results[col] = {"statistic": stat, "pvalue": pvalue}

        if pvalue < alpha:
            drift_flag = True

    print("KS-test results:")
    for k, v in drift_results.items():
        print(f"{k}: stat={v['statistic']:.3f}, pvalue={v['pvalue']:.3f}")

    print(f"⚠️ Drift detected: {drift_flag}")
    return drift_flag


# -------------------------
# MAIN FLOW
# -------------------------

@flow
def retraining_flow():
    print("\n🚀 Starting full retraining pipeline...\n")

    raw_path = ingest_data()
    processed_path = generate_features(raw_path)
    mae, rmse = train_and_evaluate(processed_path)
    drift = ks_drift_test(processed_path)


    # FINAL SUMMARY
    print("\n================ RETRAINING SUMMARY ================")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"Drift detected: {drift}")
    print("====================================================\n")

    return mae, rmse, drift


# Manual run
if __name__ == "__main__":
    retraining_flow()
