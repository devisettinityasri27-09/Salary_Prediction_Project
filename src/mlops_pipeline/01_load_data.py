
import pandas as pd
import json
import os
from datetime import datetime

DATA_PATH = "data/raw/Salary _Data.csv"
REPORT_PATH = "artifacts/data_load_report.json"


def load_data():
    print("\n[INFO] Starting Salary Prediction Data Loading...")

    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found: {DATA_PATH}")
        return False

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        print("[ERROR] Dataset is empty!")
        return False

    print("[SUCCESS] Dataset loaded successfully!")
    print(f"[INFO] Dataset shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Missing values: {df.isnull().sum().sum()}")

    os.makedirs("artifacts", exist_ok=True)

    report = {
        "stage": "Lab 7 - Data Loading",
        "status": "SUCCESS",
        "dataset_path": DATA_PATH,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": int(df.isnull().sum().sum()),
        "timestamp": datetime.now().isoformat()
    }

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    print(f"[SUCCESS] Data loading report saved: {REPORT_PATH}")

    return True


if __name__ == "__main__":
    success = load_data()

    if not success:
        raise SystemExit(1)
