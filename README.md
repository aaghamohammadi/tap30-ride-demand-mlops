![GitHub License](https://img.shields.io/github/license/aaghamohammadi/tap30-ride-demand-mlops)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)
![code style: black](https://img.shields.io/badge/code%20style-black-black)
![Python Version](https://img.shields.io/badge/python-3.10-blue?logo=python&logoColor=white)
![Docker](https://img.shields.io/badge/docker-ready-blue?logo=docker&logoColor=white)
![Kubernetes](https://img.shields.io/badge/kubernetes-deployed-blue?logo=kubernetes&logoColor=white)
![FastAPI](https://img.shields.io/badge/api-FastAPI-009688?logo=fastapi&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-blue?logo=mlflow&logoColor=white)

# Tapsi Ride Demand Prediction

Tapsi is a ridesharing company located in Iran, recording approximately 200 million travels in 2024.

The company shares a portion of their data for taxi demand in Tehran, which is divided into grid cells (rows and columns). This MLOps project aims to predict taxi demand at different times for specific areas in the city.
## ML Pipeline Components

- **Data Ingestion**: Connects to Cloud Object Storage, retrieves data, and stores it locally.
- **Data Processing**: Prepares and transforms the data to be ready for model training.
- **Model Training**: Trains a Random Forest model (using sklearn) on the processed data.

## Tools and Technologies

- **API**: FastAPI is used to create a web API for accessing predictions.
- **Experiment Tracking**: MLFlow is used for tracking experiments and model performance.
- **Containerization**: Docker is used to containerize the project.
- **CI/CD**: Github Actions automates building and pushing Docker images to Docker Hub.
- **Deployment**: Kubernetes is used as the deployment platform.

## Testing the API

After deploying the model on Kubernetes, you can test the API with the following curl command:

```bash
curl -X 'POST' \
  'http://localhost/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "hour_of_day": 18,
  "day": 3,
  "row": 4,
  "col": 7
}'
```
