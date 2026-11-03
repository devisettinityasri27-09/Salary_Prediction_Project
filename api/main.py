
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os

# --------------------------------
# 1. Initialize FastAPI
# --------------------------------
app = FastAPI(
    title="Salary Prediction API",
    description="MLOps Salary Prediction REST API",
    version="1.0.0"
)

# --------------------------------
# 2. Load model and preprocessor
# --------------------------------
MODEL_PATH = "models/lab7_random_forest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

model = None
preprocessor = None


@app.on_event("startup")
def load_model():
    global model, preprocessor

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    if not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    print("[SUCCESS] Model loaded!")
    print("[SUCCESS] Preprocessor loaded!")


# --------------------------------
# 3. Define input schema
# --------------------------------
class EmployeeData(BaseModel):
    Age: int = Field(..., ge=18, le=100)
    Gender: str
    Education_Level: str
    Job_Title: str
    Years_of_Experience: float = Field(..., ge=0, le=80)


# --------------------------------
# 4. Root endpoint
# --------------------------------
@app.get("/")
def home():
    return {
        "message": "Welcome to Salary Prediction API",
        "status": "running",
        "docs": "/docs"
    }


# --------------------------------
# 5. Health check endpoint
# --------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None
    }


# --------------------------------
# 6. Salary prediction endpoint
# --------------------------------
@app.post("/predict")
def predict_salary(employee: EmployeeData):

    if model is None or preprocessor is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded"
        )

    try:
        input_data = pd.DataFrame([{
            "Age": employee.Age,
            "Gender": employee.Gender,
            "Education Level": employee.Education_Level,
            "Job Title": employee.Job_Title,
            "Years of Experience": employee.Years_of_Experience
        }])

        processed_data = preprocessor.transform(input_data)

        prediction = model.predict(processed_data)[0]

        return {
            "predicted_salary": round(float(prediction), 2),
            "currency": "INR",
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
