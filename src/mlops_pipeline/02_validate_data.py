
import pandas as pd
import json
import os
from datetime import datetime

DATA_PATH = "data/raw/Salary _Data.csv"
REPORT_PATH = "artifacts/data_validation_report.json"

REQUIRED_COLUMNS = [
    "Age",
    "Gender",
    "Education Level",
    "Job Title",
    "Years of Experience",
    "Salary"
]


def validate_data():
    print("\n[INFO] Starting Data Validation...")

    df = pd.read_csv(DATA_PATH)
    errors = []

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        errors.append(f"Missing columns: {missing_columns}")

    # Check empty dataset
    if df.empty:
        errors.append("Dataset is empty!")

    # Check negative values
    for col in ["Age", "Years of Experience", "Salary"]:
        if col in df.columns:
            numeric_data = pd.to_numeric(
                df[col], errors="coerce"
            )

            if (numeric_data.dropna() < 0).any():
                errors.append(
                    f"Negative values found in {col}"
                )

    # Check missing values
    missing_values = int(df.isnull().sum().sum())

    print(f"[INFO] Total records: {len(df)}")
    print(f"[INFO] Missing values: {missing_values}")

    os.makedirs("artifacts", exist_ok=True)

    report = {
        "stage": "Lab 7 - Data Validation",
        "status": "FAILED" if errors else "PASSED",
        "total_records": int(len(df)),
        "missing_values": missing_values,
        "errors": errors,
        "timestamp": datetime.now().isoformat()
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    if errors:
        print("\n[ERROR] Data Validation FAILED!")

        for error in errors:
            print(f" - {error}")

        return False

    print("\n[SUCCESS] Data Validation PASSED!")
    print(f"[SUCCESS] Report saved to {REPORT_PATH}")

    return True


if __name__ == "__main__":
    success = validate_data()

    if not success:
        raise SystemExit(1)
