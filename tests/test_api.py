from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_valid_input():
    payload = {
        "tenure_months": 12,
        "support_tickets": 2,
        "monthly_spend_inr": 1500,
        "last_login_days": 5,
        "plan_type": "Basic"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "churn_probability" in response.json()


def test_predict_invalid_input():
    payload = {
        "tenure_months": "abc",
        "support_tickets": 2,
        "monthly_spend_inr": 1500,
        "last_login_days": 5,
        "plan_type": "Basic"
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422