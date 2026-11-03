
import numpy as np
import json
import os
from datetime import datetime

PROCESSED_PATH = "data/processed"
REPORT_PATH = "artifacts/preprocessing_quality_report.json"


def validate_outputs():

    print("\n[INFO] Validating Preprocessing Outputs...")

    errors = []

    files = {
        "X_train": f"{PROCESSED_PATH}/X_train_final.npy",
        "X_test": f"{PROCESSED_PATH}/X_test_final.npy",
        "y_train": f"{PROCESSED_PATH}/y_train.npy",
        "y_test": f"{PROCESSED_PATH}/y_test.npy"
    }

    # Check files exist
    for name, path in files.items():
        if not os.path.exists(path):
            errors.append(f"Missing file: {path}")

    if errors:
        print("[ERROR] Required preprocessing files missing!")
        for error in errors:
            print(error)
        return False

    # Load arrays
    X_train = np.load(files["X_train"])
    X_test = np.load(files["X_test"])
    y_train = np.load(files["y_train"])
    y_test = np.load(files["y_test"])

    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] X_test shape : {X_test.shape}")
    print(f"[INFO] y_train shape: {y_train.shape}")
    print(f"[INFO] y_test shape : {y_test.shape}")

    # Validate dimensions
    if X_train.ndim != 2 or X_test.ndim != 2:
        errors.append("Feature arrays must be 2-dimensional")

    if y_train.ndim != 1 or y_test.ndim != 1:
        errors.append("Target arrays must be 1-dimensional")

    # Validate sample consistency
    if X_train.shape[0] != y_train.shape[0]:
        errors.append("Training feature and target rows mismatch")

    if X_test.shape[0] != y_test.shape[0]:
        errors.append("Testing feature and target rows mismatch")

    # Validate feature consistency
    if X_train.shape[1] != X_test.shape[1]:
        errors.append("Training and testing feature dimensions mismatch")

    # Check NaN and infinity
    if not np.isfinite(X_train).all():
        errors.append("Invalid values found in X_train")

    if not np.isfinite(X_test).all():
        errors.append("Invalid values found in X_test")

    if not np.isfinite(y_train).all():
        errors.append("Invalid values found in y_train")

    if not np.isfinite(y_test).all():
        errors.append("Invalid values found in y_test")

    os.makedirs("artifacts", exist_ok=True)

    report = {
        "stage": "Lab 7 - Preprocessing Quality Validation",
        "status": "FAILED" if errors else "PASSED",
        "training_shape": list(X_train.shape),
        "testing_shape": list(X_test.shape),
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("\n[ERROR] Preprocessing validation FAILED!")
        for error in errors:
            print(f" - {error}")
        return False

    print("\n[SUCCESS] Preprocessing Output Validation PASSED!")
    print("[SUCCESS] Data is clean and dimensionally consistent.")
    print(f"[SUCCESS] Report saved to {REPORT_PATH}")

    return True


if __name__ == "__main__":
    success = validate_outputs()

    if not success:
        raise SystemExit(1)
