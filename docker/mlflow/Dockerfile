FROM python:3.11-slim

WORKDIR /mlflow

RUN pip install mlflow

EXPOSE 5000

CMD ["mlflow", "ui", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "/mlflow/mlruns"]
