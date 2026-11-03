
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_NAME = "Salary_Prediction_Production_Model"


def train_and_register():

    print("\n[INFO] Starting Model Training and Registry...")

    X_train = np.load("data/processed/X_train_final.npy")
    X_test = np.load("data/processed/X_test_final.npy")
    y_train = np.load("data/processed/y_train.npy")
    y_test = np.load("data/processed/y_test.npy")

    mlflow.set_experiment("Salary_Prediction")

    with mlflow.start_run(run_name="Lab7_RandomForest_Registry"):

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )

        print("[INFO] Training Random Forest...")
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, predictions)

        print("\n[INFO] Model Evaluation")
        print(f"MAE  : ₹{mae:,.2f}")
        print(f"MSE  : {mse:,.2f}")
        print(f"RMSE : ₹{rmse:,.2f}")
        print(f"R²   : {r2:.4f}")

        mlflow.log_params({
            "model": "RandomForestRegressor",
            "n_estimators": 200,
            "max_depth": 15,
            "random_state": 42
        })

        mlflow.log_metrics({
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "r2_score": r2
        })

        if os.path.exists("models/preprocessor.pkl"):
            mlflow.log_artifact(
                "models/preprocessor.pkl",
                artifact_path="preprocessing"
            )

        print("\n[INFO] Registering model in MLflow...")

        mlflow.sklearn.log_model(
            sk_model=model,
            name="random_forest_model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        joblib.dump(
            model,
            "models/lab7_random_forest.pkl"
        )

        print("\n[SUCCESS] Model training completed!")
        print("[SUCCESS] Model registered in MLflow!")
        print(f"[INFO] Model name: {MODEL_NAME}")

        return r2


if __name__ == "__main__":
    train_and_register()
