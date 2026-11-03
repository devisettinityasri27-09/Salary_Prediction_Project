
import numpy as np
import pandas as pd
import joblib
import json
import os
from datetime import datetime

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

def evaluate_model():

    print("\n[INFO] Starting Model Evaluation...")

    X_test = np.load("data/processed/X_test_final.npy")
    y_test = np.load("data/processed/y_test.npy")

    model_path = "models/lab7_random_forest.pkl"

    if not os.path.exists(model_path):
        print("[ERROR] Trained model not found!")
        return False

    model = joblib.load(model_path)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    print("\n====================================")
    print("       MODEL EVALUATION REPORT")
    print("====================================")
    print(f"MAE  : ₹{mae:,.2f}")
    print(f"MSE  : {mse:,.2f}")
    print(f"RMSE : ₹{rmse:,.2f}")
    print(f"R²   : {r2:.4f}")
    print("====================================")

    os.makedirs("reports", exist_ok=True)

    report = {
        "model": "RandomForestRegressor",
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2_Score": float(r2),
        "evaluation_status": "SUCCESS",
        "timestamp": datetime.now().isoformat()
    }

    with open(
        "reports/lab7_model_evaluation.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=4)

    # Save actual vs predicted values
    comparison = pd.DataFrame({
        "Actual_Salary": y_test,
        "Predicted_Salary": predictions,
        "Absolute_Error": np.abs(y_test - predictions)
    })

    comparison.to_csv(
        "outputs/lab7_actual_vs_predicted.csv",
        index=False
    )

    print("\n[SUCCESS] Evaluation completed!")
    print("[SUCCESS] Report saved to reports/lab7_model_evaluation.json")
    print("[SUCCESS] Predictions saved to outputs/lab7_actual_vs_predicted.csv")

    return True


if __name__ == "__main__":
    success = evaluate_model()

    if not success:
        raise SystemExit(1)
