import os
import numpy as np
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_and_register_model():

    print("[INFO] --- Starting MLflow Run with Model Registry ---")

    # --------------------------------------------------
    # 1. SET MLFLOW EXPERIMENT
    # --------------------------------------------------

    mlflow.set_experiment("Salary_Prediction")

    # --------------------------------------------------
    # 2. LOAD PROCESSED DATA
    # --------------------------------------------------

    try:

        X_train = np.load(
            "data/processed/X_train_final.npy"
        )

        X_test = np.load(
            "data/processed/X_test_final.npy"
        )

        y_train = np.load(
            "data/processed/y_train.npy"
        )

        y_test = np.load(
            "data/processed/y_test.npy"
        )

    except FileNotFoundError:

        print(
            "[ERROR] Processed data not found. "
            "Please run the Lab 5 pipeline first."
        )

        return

    print(
        f"[INFO] Training data shape: {X_train.shape}"
    )

    print(
        f"[INFO] Testing data shape: {X_test.shape}"
    )

    # --------------------------------------------------
    # 3. MODEL PARAMETERS
    # --------------------------------------------------

    params = {
        "n_estimators": 200,
        "max_depth": 15,
        "random_state": 42
    }

    # --------------------------------------------------
    # 4. START MLFLOW RUN
    # --------------------------------------------------

    with mlflow.start_run(
        run_name="RandomForest_Registry"
    ):

        # --------------------------------------------------
        # 5. LOG PARAMETERS
        # --------------------------------------------------

        mlflow.log_params(params)

        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "problem_type",
            "Regression"
        )

        # --------------------------------------------------
        # 6. TRAIN RANDOM FOREST
        # --------------------------------------------------

        print(
            "[INFO] Training RandomForestRegressor..."
        )

        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )

        rf_model.fit(
            X_train,
            y_train
        )

        print(
            "[SUCCESS] Model training completed!"
        )

        # --------------------------------------------------
        # 7. PREDICTION
        # --------------------------------------------------

        print(
            "[INFO] Evaluating model..."
        )

        y_pred = rf_model.predict(
            X_test
        )

        # --------------------------------------------------
        # 8. CALCULATE METRICS
        # --------------------------------------------------

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            y_pred
        )

        # --------------------------------------------------
        # 9. LOG METRICS TO MLFLOW
        # --------------------------------------------------

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }

        mlflow.log_metrics(
            metrics
        )

        # --------------------------------------------------
        # 10. DISPLAY RESULTS
        # --------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "MODEL PERFORMANCE"
        )

        print(
            "=========================================="
        )

        print(
            f"MAE  = ₹{mae:,.2f}"
        )

        print(
            f"MSE  = {mse:,.2f}"
        )

        print(
            f"RMSE = ₹{rmse:,.2f}"
        )

        print(
            f"R²   = {r2:.4f}"
        )

        # --------------------------------------------------
        # 11. ATTACH PREPROCESSING PIPELINE
        # --------------------------------------------------

        if os.path.exists(
            "models/preprocessor.pkl"
        ):

            print(
                "[INFO] Attaching preprocessing pipeline..."
            )

            mlflow.log_artifact(
                "models/preprocessor.pkl",
                artifact_path="preprocessing_pipeline"
            )

            print(
                "[SUCCESS] Preprocessing pipeline attached!"
            )

        else:

            print(
                "[WARNING] models/preprocessor.pkl not found."
            )

        # --------------------------------------------------
        # 12. REGISTER MODEL
        # --------------------------------------------------

        print(
            "[INFO] Pushing model to MLflow Registry..."
        )

        mlflow.sklearn.log_model(

            sk_model=rf_model,

            name="random_forest_model",

            registered_model_name=(
                "Salary_Prediction_Production_Model"
            ),

            # Required for Random Forest with
            # current MLflow/skops validation
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        # --------------------------------------------------
        # 13. SUCCESS MESSAGE
        # --------------------------------------------------

        print(
            "\n=========================================="
        )

        print(
            "MODEL REGISTRATION SUCCESSFUL"
        )

        print(
            "=========================================="
        )

        print(
            f"[SUCCESS] MAE  = ₹{mae:,.2f}"
        )

        print(
            f"[SUCCESS] RMSE = ₹{rmse:,.2f}"
        )

        print(
            f"[SUCCESS] R²   = {r2:.4f}"
        )

        print(
            "[SUCCESS] Model successfully registered under name:"
        )

        print(
            "Salary_Prediction_Production_Model"
        )

        print(
            "=========================================="
        )


if __name__ == "__main__":

    train_and_register_model()