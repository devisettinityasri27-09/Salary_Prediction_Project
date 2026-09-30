import os
import joblib
import numpy as np
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def train_and_track(
    run_name="RandomForest_Baseline",
    params=None
):

    # ==========================================
    # MODEL PARAMETERS
    # ==========================================

    if params is None:
        params = {
            "n_estimators": 200,
            "random_state": 42
        }

    print("\n==========================================")
    print(f"Starting MLflow Run: {run_name}")
    print("==========================================")

    # ==========================================
    # LOAD PROCESSED DATA
    # ==========================================

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

    print("Training data shape:", X_train.shape)
    print("Testing data shape :", X_test.shape)

    # ==========================================
    # CREATE / SELECT MLFLOW EXPERIMENT
    # ==========================================

    mlflow.set_experiment(
        "Salary_Prediction"
    )

    # ==========================================
    # START MLFLOW RUN
    # ==========================================

    with mlflow.start_run(
        run_name=run_name
    ):

        # ======================================
        # LOG PARAMETERS
        # ======================================

        mlflow.log_params(
            params
        )

        mlflow.log_param(
            "model_family",
            "RandomForestRegressor"
        )

        mlflow.log_param(
            "problem_type",
            "Regression"
        )

        # ======================================
        # CREATE MODEL
        # ======================================

        model = RandomForestRegressor(
            **params
        )

        # ======================================
        # TRAIN MODEL
        # ======================================

        print("\nTraining Random Forest Regressor...")

        model.fit(
            X_train,
            y_train
        )

        print(
            "Model training completed!"
        )

        # ======================================
        # PREDICTIONS
        # ======================================

        y_pred = model.predict(
            X_test
        )

        # ======================================
        # CALCULATE METRICS
        # ======================================

        mae = mean_absolute_error(
            y_test,
            y_pred
        )

        mse = mean_squared_error(
            y_test,
            y_pred
        )

        rmse = np.sqrt(
            mse
        )

        r2 = r2_score(
            y_test,
            y_pred
        )

        metrics = {
            "mae": mae,
            "mse": mse,
            "rmse": rmse,
            "r2_score": r2
        }

        # ======================================
        # LOG METRICS TO MLFLOW
        # ======================================

        mlflow.log_metrics(
            metrics
        )

        print("\n==========================================")
        print("MODEL PERFORMANCE")
        print("==========================================")

        print(
            f"MAE  : ₹{mae:,.2f}"
        )

        print(
            f"MSE  : {mse:,.2f}"
        )

        print(
            f"RMSE : ₹{rmse:,.2f}"
        )

        print(
            f"R²   : {r2:.4f}"
        )

        # ======================================
        # CREATE ARTIFACT DIRECTORY
        # ======================================

        os.makedirs(
            "artifacts",
            exist_ok=True
        )

        # ======================================
        # ACTUAL VS PREDICTED PLOT
        # ======================================

        print(
            "\nCreating actual vs predicted plot..."
        )

        fig_actual, ax_actual = plt.subplots(
            figsize=(7, 6)
        )

        ax_actual.scatter(
            y_test,
            y_pred,
            alpha=0.7
        )

        # Perfect prediction line
        min_value = min(
            y_test.min(),
            y_pred.min()
        )

        max_value = max(
            y_test.max(),
            y_pred.max()
        )

        ax_actual.plot(
            [min_value, max_value],
            [min_value, max_value],
            linestyle="--"
        )

        ax_actual.set_xlabel(
            "Actual Salary"
        )

        ax_actual.set_ylabel(
            "Predicted Salary"
        )

        ax_actual.set_title(
            f"Actual vs Predicted Salary - {run_name}"
        )

        actual_path = (
            "artifacts/actual_vs_predicted.png"
        )

        fig_actual.savefig(
            actual_path,
            bbox_inches="tight"
        )

        plt.close(
            fig_actual
        )

        # Log actual vs predicted plot
        mlflow.log_artifact(
            actual_path,
            artifact_path="plots"
        )

        print(
            "Actual vs predicted plot logged successfully!"
        )

        # ======================================
        # RESIDUAL PLOT
        # ======================================

        print(
            "Creating residual plot..."
        )

        residuals = (
            y_test - y_pred
        )

        fig_residual, ax_residual = plt.subplots(
            figsize=(7, 6)
        )

        ax_residual.scatter(
            y_pred,
            residuals,
            alpha=0.7
        )

        ax_residual.axhline(
            y=0,
            linestyle="--"
        )

        ax_residual.set_xlabel(
            "Predicted Salary"
        )

        ax_residual.set_ylabel(
            "Residual"
        )

        ax_residual.set_title(
            f"Residual Plot - {run_name}"
        )

        residual_path = (
            "artifacts/residual_plot.png"
        )

        fig_residual.savefig(
            residual_path,
            bbox_inches="tight"
        )

        plt.close(
            fig_residual
        )

        # Log residual plot
        mlflow.log_artifact(
            residual_path,
            artifact_path="plots"
        )

        print(
            "Residual plot logged successfully!"
        )

        # ======================================
        # LOG DATASET METADATA
        # ======================================

        metadata_path = (
            "data/processed/dataset_metadata.json"
        )

        if os.path.exists(
            metadata_path
        ):

            mlflow.log_artifact(
                metadata_path,
                artifact_path="metadata"
            )

            print(
                "Dataset metadata logged successfully!"
            )

        # ======================================
        # LOG MODEL TO MLFLOW
        # ======================================

        print(
            "\nLogging model to MLflow..."
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=[
                "sklearn.tree._tree.Tree"
            ]
        )

        print(
            "Model logged successfully!"
        )

        # ======================================
        # SAVE LOCAL MODEL BACKUP
        # ======================================

        os.makedirs(
            "models",
            exist_ok=True
        )

        joblib.dump(
            model,
            "models/random_forest_mlflow_model.pkl"
        )

        print(
            "Local model saved to:"
        )

        print(
            "models/random_forest_mlflow_model.pkl"
        )

        # ======================================
        # SUCCESS
        # ======================================

        print("\n==========================================")
        print(
            "MLFLOW RUN COMPLETED SUCCESSFULLY!"
        )
        print("==========================================")

        print(
            f"Run Name : {run_name}"
        )

        print(
            f"R² Score : {r2:.4f}"
        )

        print(
            f"RMSE     : ₹{rmse:,.2f}"
        )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    train_and_track(
        run_name="RandomForest"
    )