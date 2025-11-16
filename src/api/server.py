from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np

app = FastAPI(title="AirPassengers Forecasting API")

# Load model
try:
    model = joblib.load("models/random_forest_model.joblib")
    print("Model loaded successfully.")
except:
    raise RuntimeError("Model file not found. Train the model first.")


class InputData(BaseModel):
    lags: list[float] = Field(..., description="List of 12 lag values")


@app.get("/")
def home():
    return {"message": "AirPassengers Forecasting API is running!"}


@app.post("/predict")
def predict(data: InputData):
    if len(data.lags) != 12:
        raise HTTPException(
            status_code=400,
            detail=f"You must provide exactly 12 lag values. Received {len(data.lags)}."
        )

    X = np.array(data.lags).reshape(1, -1)
    y_pred = model.predict(X)

    return {
        "prediction": float(y_pred[0]),
        "input_lags": data.lags
    }
