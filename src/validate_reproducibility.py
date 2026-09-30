import os
import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def validate_reproducibility():

    print("\n==========================================")
    print("REPRODUCIBILITY VALIDATION")
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

    print("Data loaded successfully!")
    print("Training shape:", X_train.shape)
    print("Testing shape :", X_test.shape)

    # ==========================================
    # FIXED MODEL PARAMETERS
    # ==========================================

    params = {
        "n_estimators": 200,
        "random_state": 42
    }

    # ==========================================
    # TRAIN MODEL - RUN 1
    # ==========================================

    print("\nTraining model - Run 1...")

    model_1 = RandomForestRegressor(
        **params
    )

    model_1.fit(
        X_train,
        y_train
    )

    predictions_1 = model_1.predict(
        X_test
    )

    r2_1 = r2_score(
        y_test,
        predictions_1
    )

    print(
        f"Run 1 R² Score: {r2_1:.4f}"
    )

    # ==========================================
    # TRAIN MODEL - RUN 2
    # ==========================================

    print("\nTraining model - Run 2...")

    model_2 = RandomForestRegressor(
        **params
    )

    model_2.fit(
        X_train,
        y_train
    )

    predictions_2 = model_2.predict(
        X_test
    )

    r2_2 = r2_score(
        y_test,
        predictions_2
    )

    print(
        f"Run 2 R² Score: {r2_2:.4f}"
    )

    # ==========================================
    # COMPARE RESULTS
    # ==========================================

    print("\n==========================================")
    print("REPRODUCIBILITY RESULTS")
    print("==========================================")

    print(
        f"Run 1 R² Score: {r2_1:.4f}"
    )

    print(
        f"Run 2 R² Score: {r2_2:.4f}"
    )

    # Check predictions
    predictions_same = np.array_equal(
        predictions_1,
        predictions_2
    )

    r2_same = (
        r2_1 == r2_2
    )

    print(
        f"Predictions identical: {predictions_same}"
    )

    print(
        f"R² identical: {r2_same}"
    )

    # ==========================================
    # FINAL VALIDATION
    # ==========================================

    if predictions_same and r2_same:

        print("\n==========================================")
        print("REPRODUCIBILITY VALIDATION PASSED!")
        print("==========================================")

        print(
            "The same data, parameters and random seed "
            "produced identical results."
        )

    else:

        print("\n==========================================")
        print("REPRODUCIBILITY VALIDATION FAILED!")
        print("==========================================")

        print(
            "The two runs produced different results."
        )

        raise RuntimeError(
            "Model reproducibility validation failed."
        )

    # ==========================================
    # SAVE VALIDATION RESULT
    # ==========================================

    os.makedirs(
        "outputs",
        exist_ok=True
    )

    result = {
        "run_1_r2": r2_1,
        "run_2_r2": r2_2,
        "predictions_identical": predictions_same,
        "r2_identical": r2_same,
        "reproducibility_passed": (
            predictions_same and r2_same
        )
    }

    joblib.dump(
        result,
        "outputs/reproducibility_result.pkl"
    )

    print(
        "\nValidation result saved to:"
    )

    print(
        "outputs/reproducibility_result.pkl"
    )


if __name__ == "__main__":

    validate_reproducibility()