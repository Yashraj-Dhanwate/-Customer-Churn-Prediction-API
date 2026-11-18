from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_churn_model.joblib"

# The Task 4 artifact is a scikit-learn Pipeline containing preprocessing
# (imputation, scaling and categorical encoding) plus the trained estimator.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Customer Churn Prediction API",
    description="REST API for the Rab Tech Academy customer churn model.",
    version="1.0.0",
)

class ChurnRequest(BaseModel):
    tenure_months: float = Field(..., ge=0, description="Customer tenure in months")
    support_tickets: float = Field(..., ge=0, description="Number of support tickets")
    monthly_spend_inr: float = Field(..., ge=0, description="Monthly spend in INR")
    last_login_days: float = Field(..., ge=0, description="Days since last login")
    plan_type: str = Field(..., min_length=1, description="Customer plan category")

@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(request: ChurnRequest):
    try:
        input_df = pd.DataFrame([request.model_dump()])
        prediction = int(model.predict(input_df)[0])

        if not hasattr(model, "predict_proba"):
            raise HTTPException(
                status_code=500,
                detail="The loaded model does not support probability predictions."
            )

        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)
        probability_by_class = {
            str(int(label)): float(prob)
            for label, prob in zip(classes, probabilities)
        }

        churn_probability = probability_by_class.get("1")
        if churn_probability is None:
            raise HTTPException(
                status_code=500,
                detail="Could not identify the churn class (expected class label 1)."
            )

        return {
            "prediction": prediction,
            "predicted_class": "Churn" if prediction == 1 else "Not Churn",
            "churn_probability": round(churn_probability, 6),
            "not_churn_probability": round(1.0 - churn_probability, 6),
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc
