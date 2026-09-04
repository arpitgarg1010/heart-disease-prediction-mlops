# Heart Disease Prediction Platform

An end-to-end Machine Learning Operations (MLOps) project for predicting heart disease using the UCI Heart Disease dataset.

The project demonstrates the complete machine learning lifecycle:

- Data acquisition
- Exploratory data analysis
- Data preprocessing
- Feature engineering
- Model development and evaluation
- Experiment tracking with MLflow
- REST API development with FastAPI
- Containerization with Docker
- Docker Compose
- Kubernetes deployment
- Prometheus monitoring
- Automated testing
- CI/CD with GitHub Actions

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- FastAPI
- Uvicorn
- MLflow
- Docker
- Docker Compose
- Kubernetes
- Prometheus
- Postman
- Newman
- GitHub Actions

## Dataset

UCI Heart Disease Dataset.

The dataset contains clinical attributes used to train and evaluate machine learning models for heart disease prediction.

## Project Workflow

Data Acquisition
↓
EDA
↓
Data Preprocessing
↓
Feature Engineering
↓
Model Training
↓
Model Evaluation
↓
MLflow Experiment Tracking
↓
FastAPI
↓
Docker
↓
Kubernetes
↓
Prometheus Monitoring
↓
CI/CD

## Project Structure

```text
Heart_Disease_MLOps/
├── api/
│   ├── app.py
│   └── schemas.py
├── data/
├── models/
├── notebooks/
├── reports/
├── screenshots/
├── tests/
│   └── postman/
├── deployment/
│   ├── deployment.yaml
│   └── service.yaml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md