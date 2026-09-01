# Assignment 01 — Original Submission Evidence

## Frozen Submission

- Submission tag: `assignment-01-submission`
- Commit: `11582954c29a478c36f56fc6e9a2cd5752634c21`
- Date: 12 July 2026

## Original Submission vs Assignment Requirements

| Requirement | Evidence in Submitted Commit | Status |
|---|---|---|
| Data acquisition | `data/raw/heart.csv` | Completed |
| EDA | `notebooks/01_EDA.ipynb` | Completed |
| Data preprocessing | `notebooks/02_Data_Preprocessing.ipynb` | Completed |
| Two classifiers | `notebooks/03_Model_Development_and_Evaluation.ipynb` | Completed |
| Accuracy, Precision, Recall, F1, ROC-AUC | Model evaluation notebook | Completed |
| 5-fold cross-validation | Model evaluation notebook | Completed |
| Random Forest hyperparameter tuning | GridSearchCV | Completed |
| ROC curves and confusion matrices | `artifacts/` | Completed |
| MLflow experiment tracking | `notebooks/04_Experiment_Tracking_MLflow.ipynb` | Completed |
| MLflow database/runs | `notebooks/mlflow.db`, `notebooks/mlruns/` | Completed |
| Saved model | `models/best_model.pkl` | Completed |
| FastAPI prediction API | `api/app.py`, `api/schemas.py` | Completed |
| API testing | Postman collection + Newman report | Completed |
| Docker containerization | `Dockerfile` + Docker Swagger evidence | Completed |
| Automated unit tests | No pytest test suite | Not completed |
| CI/CD | No GitHub Actions/Jenkins workflow | Not completed |
| Production deployment | No cloud/Kubernetes deployment | Not completed |
| Monitoring | No Prometheus/Grafana/basic monitoring implementation | Not completed |
| Professional project report | No final project report in submitted repository | Not completed |

## Metrics Evidence

### Logistic Regression

- Accuracy: 0.8689
- Precision: 0.8125
- Recall: 0.9286
- F1: 0.8667
- ROC-AUC: 0.9513
- 5-fold CV Accuracy: 0.8217

### Random Forest

- Best parameters: `max_depth=5`, `n_estimators=200`
- Accuracy: 0.9016
- Precision: 0.8667
- Recall: 0.9286
- F1: 0.8966
- ROC-AUC: 0.9567
- 5-fold CV Accuracy: 0.8215

Both models also have confusion-matrix and ROC-curve artifacts.

## API Testing Evidence

The submitted Newman report documents:

- 4 total API requests
- 0 failed tests
- API assertions
- Request execution results

## Re-evaluation Note

The evaluator feedback stated:

> "No report, metrics, CI/CD, and cloud deployment."

The original submission clearly lacked the report, CI/CD and production/cloud deployment components.

However, model evaluation metrics and MLflow-tracked metrics were present in the exact frozen submission commit and can be independently verified from the submitted notebooks and artifacts.