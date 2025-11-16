# 🛠️ MLOps Retraining System with Prefect, MLflow, FastAPI & Docker

This repository contains a complete, production-ready **MLOps pipeline** for automated model retraining, drift detection, experiment tracking, and API-based model serving.  
It is designed to replicate the architecture used in modern machine-learning teams.

---

# 📌 Project Overview

This system performs:

### ✅ Automated Retraining Pipeline (Prefect)
- Loads raw time-series data (AirPassengers)
- Builds 12-lag features
- Trains a RandomForest model for forecasting
- Performs drift detection using the KS-test
- Logs all metrics, parameters and artifacts to MLflow
- Saves the latest model to `models/`

### ✅ Model Serving (FastAPI)
- Exposes a `/predict` endpoint
- Validates input lags with Pydantic
- Loads the latest trained model automatically

### ✅ Full Experiment Tracking (MLflow)
- Every retraining run is logged
- Metrics, parameters and models are versioned
- UI available via Docker (port 5500)

### ✅ Complete Containerization (Docker)
- `api` container → FastAPI serving
- `prefect` container → retraining pipeline
- `mlflow` container → experiment tracking
- Shared volumes for: `models/`, `data/`, `mlruns/`



---

# 🚀 How to Run the System

## 1️⃣ Clone the project

```bash
git clone https://github.com/YOUR_USERNAME/MLOps-Retraining-Prefect.git
cd MLOps-Retraining-Prefect
```

## 2️⃣ Build & Run with Docker Compose

```bash
docker compose build
docker compose up
```

## 3️⃣ Access the Services
| Service       | URL                                                      | Description           |
| ------------- | -------------------------------------------------------- | --------------------- |
| **FastAPI**   | [http://localhost:8000/docs](http://localhost:8000/docs) | Interactive API docs  |
| **MLflow UI** | [http://localhost:5500](http://localhost:5500)           | Metrics, runs, models |

---

## 🗂️ Project Structure

```
MLOps-Retraining-Prefect/
│
├── docker/
│   ├── api/
│   │   └── Dockerfile
│   ├── prefect/
│   │   └── Dockerfile
│   └── mlflow/
│       └── Dockerfile
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── pipeline/
│   └── api/
│
├── models/
├── data/
├── mlruns/
├── notebooks/
├── requirements.txt
├── docker-compose.yml
└── README.md
```

---

# 📊 Key Technologies

| Category            | Tools                       |
| ------------------- | --------------------------- |
| Orchestration       | Prefect                     |
| Experiment Tracking | MLflow                      |
| Model Serving       | FastAPI, Uvicorn            |
| Containerization    | Docker, Docker Compose      |
| Machine Learning    | Scikit-Learn, Pandas, NumPy |
| Drift Detection     | SciPy (KS-test)             |


# 🛣️ Future Improvements

- Replace RandomForest with a PyTorch forecasting model
- Add CI/CD for automated deployment
- Add inference logging and monitoring
- Connect to real external data sources
- Create a batch prediction pipeline

