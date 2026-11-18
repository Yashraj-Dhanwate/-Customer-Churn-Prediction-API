# Customer Churn Prediction API

## Project Overview
This project provides a real-time Machine Learning inference REST API to predict customer churn using FastAPI and a trained Machine Learning model.

## Technologies Used
- Python 3.12
- FastAPI
- Scikit-learn
- Pandas
- NumPy
- Joblib
- Uvicorn
- Docker
- Pytest

## Project Structure
```text
Task6_New_Project/
│
├── main.py
├── best_churn_model.joblib
├── requirements.txt
├── Dockerfile
├── README.md
└── tests/
    └── test_api.py
```

## Installation and Setup

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

4. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the API

```bash
python -m uvicorn main:app --reload
```

API URL: http://127.0.0.1:8000

Interactive API documentation: http://127.0.0.1:8000/docs

## Prediction Endpoint

**Method:** POST

**Endpoint:** `/predict`

Example request:

```json
{
  "tenure_months": 12,
  "support_tickets": 2,
  "monthly_spend_inr": 1500,
  "last_login_days": 5,
  "plan_type": "Basic"
}
```

The API returns a churn prediction and probability information.

## Run Unit Tests

```bash
python -m pytest -v
```

## Docker Instructions

Build the Docker image:

```bash
docker build -t customer-churn-api .
```

Run the container:

```bash
docker run -p 8000:8000 customer-churn-api
```

## Features
- Real-time customer churn prediction
- REST API using FastAPI
- Input validation
- Prediction probability output
- Automated unit testing
- Docker support

## Author
Yashraj Dhanwate