#Heart Disease Prediction Platform

An end to end machine learning Operations (MLOps) project that predicts the likelihood of disease using clinical data using the UCI Heart Disease Dataset.

This project demonstrates the complete machine learning lifecycle:

- Data Acquisition
- Exploratory Data analysis
- Feature Engineering
- Model Engineering
- Experiment Tracking
- API development
- Containerization
- CI/CD
- Deployment
- Monitoring


----

## Tech Stack
- Python
- Scikit-learn
- FastAPI
- MLflow
- Docker
- Postman
- Newman

## Dataset
UCI Heart Disease Dataset

## Project Workflow
Data Acquisition
↓
EDA
↓
Preprocessing
↓
Model Training
↓
MLflow Tracking
↓
FastAPI
↓
Docker
↓
Deployment

## Running the Project

python -m venv .venv
pip install -r requirements.txt

uvicorn api.app:app --reload

## Docker

docker build -t heart-disease-api .

docker run -p 8000:8000 heart-disease-api

## API

GET /

POST /predict

## Testing

Postman Collection

Newman Report

## Folder Structure