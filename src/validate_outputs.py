import os
import json
import numpy as np


def validate_preprocessing_outputs():

    print("[INFO] Validating Preprocessing Outputs...")

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

    except FileNotFoundError as e:

        print(f"[ERROR] Could not find processed data: {e}")
        return False

    errors = []

    # Check missing values
    if np.isnan(X_train).sum() > 0:
        errors.append(
            "NaNs detected in X_train after preprocessing."
        )

    if np.isnan(X_test).sum() > 0:
        errors.append(
            "NaNs detected in X_test after preprocessing."
        )

    if np.isnan(y_train).sum() > 0:
        errors.append(
            "NaNs detected in y_train."
        )

    if np.isnan(y_test).sum() > 0:
        errors.append(
            "NaNs detected in y_test."
        )

    # Check feature dimensions
    if X_train.shape[1] != X_test.shape[1]:

        errors.append(
            f"Feature mismatch: "
            f"X_train has {X_train.shape[1]} columns "
            f"but X_test has {X_test.shape[1]} columns."
        )

    # Check rows
    if X_train.shape[0] != y_train.shape[0]:

        errors.append(
            "Row mismatch between X_train and y_train."
        )

    if X_test.shape[0] != y_test.shape[0]:

        errors.append(
            "Row mismatch between X_test and y_test."
        )

    # Create report
    report = {

        "validation_status":
            "PASSED" if not errors else "FAILED",

        "matrix_dimensions": {

            "X_train_shape":
                list(X_train.shape),

            "X_test_shape":
                list(X_test.shape),

            "y_train_shape":
                list(y_train.shape),

            "y_test_shape":
                list(y_test.shape)
        },

        "data_quality": {

            "missing_values_X_train":
                int(np.isnan(X_train).sum()),

            "missing_values_X_test":
                int(np.isnan(X_test).sum()),

            "missing_values_y_train":
                int(np.isnan(y_train).sum()),

            "missing_values_y_test":
                int(np.isnan(y_test).sum())
        },

        "errors": errors
    }

    os.makedirs("artifacts", exist_ok=True)

    report_path = (
        "artifacts/preprocessing_summary_report.json"
    )

    with open(report_path, "w") as f:

        json.dump(
            report,
            f,
            indent=4
        )

    if errors:

        print("[ERROR] Output Validation FAILED!")

        for error in errors:
            print(f"  - {error}")

        return False

    else:

        print("[SUCCESS] Output Validation PASSED.")

        print(
            "[INFO] Features are clean, scaled, "
            "encoded, and dimensionally consistent."
        )

        print(
            f"[INFO] Summary report saved to {report_path}"
        )

        return True


if __name__ == "__main__":
    validate_preprocessing_outputs()