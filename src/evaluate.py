import os
import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(model, name, X_test, y_test):

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate regression metrics
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

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "predictions": y_pred
    }


def evaluate_all_models():

    # ==========================================
    # LOAD TEST DATA
    # ==========================================

    X_test = np.load(
        "data/processed/X_test_final.npy"
    )

    y_test = np.load(
        "data/processed/y_test.npy"
    )

    print("Test data loaded successfully!")
    print("Test data shape:", X_test.shape)

    # ==========================================
    # LOAD TRAINED MODELS
    # ==========================================

    model_files = {
        "Linear Regression":
            "models/linear_regression_baseline.pkl",

        "Ridge Regression":
            "models/ridge_regression_baseline.pkl",

        "Lasso Regression":
            "models/lasso_regression_baseline.pkl",

        "Elastic Net":
            "models/elastic_net_baseline.pkl",

        "Decision Tree":
            "models/decision_tree_baseline.pkl",

        "Random Forest":
            "models/random_forest_baseline.pkl",

        "Gradient Boosting":
            "models/gradient_boosting_baseline.pkl",

        "Extra Trees":
            "models/extra_trees_baseline.pkl"
    }

    results = {}

    # ==========================================
    # EVALUATE EACH MODEL
    # ==========================================

    print("\n==========================================")
    print("MODEL EVALUATION")
    print("==========================================\n")

    for model_name, model_path in model_files.items():

        model = joblib.load(
            model_path
        )

        result = evaluate_model(
            model,
            model_name,
            X_test,
            y_test
        )

        results[model_name] = result

        print(f"--- {model_name} ---")
        print(f"MAE  : ₹{result['MAE']:,.2f}")
        print(f"MSE  : {result['MSE']:,.2f}")
        print(f"RMSE : ₹{result['RMSE']:,.2f}")
        print(f"R²   : {result['R2']:.4f}")
        print()

    # ==========================================
    # CREATE RESULTS DATAFRAME
    # ==========================================

    results_for_csv = {}

    for name, result in results.items():

        results_for_csv[name] = {
            "MAE": result["MAE"],
            "MSE": result["MSE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        }

    results_df = pd.DataFrame(
        results_for_csv
    ).T

    results_df = results_df.sort_values(
        by="R2",
        ascending=False
    )

    # ==========================================
    # SELECT BEST BASELINE MODEL
    # ==========================================

    best_model_name = results_df[
        "R2"
    ].idxmax()

    best_r2 = results_df.loc[
        best_model_name,
        "R2"
    ]

    best_model = joblib.load(
        model_files[best_model_name]
    )

    print("==========================================")
    print(
        f"Best Baseline Model: {best_model_name}"
    )
    print(
        f"R² Score: {best_r2:.4f}"
    )
    print("==========================================")

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    results_df.to_csv(
        "outputs/baseline_model_results.csv"
    )

    print(
        "\nBaseline results saved successfully!"
    )

    # ==========================================
    # ERROR ANALYSIS
    # ==========================================

    predictions = best_model.predict(
        X_test
    )

    errors = y_test - predictions
    absolute_errors = np.abs(errors)

    errors_df = pd.DataFrame({
        "Actual_Salary": y_test,
        "Predicted_Salary": predictions,
        "Error": errors,
        "Absolute_Error": absolute_errors
    })

    errors_df.to_csv(
        "outputs/salary_prediction_errors.csv",
        index=False
    )

    print(
        "Salary prediction errors saved successfully!"
    )

    print(
        "\nMean Absolute Error: "
        f"₹{absolute_errors.mean():,.2f}"
    )

    print(
        "Minimum Error: "
        f"₹{absolute_errors.min():,.2f}"
    )

    print(
        "Maximum Error: "
        f"₹{absolute_errors.max():,.2f}"
    )


if __name__ == "__main__":
    evaluate_all_models()