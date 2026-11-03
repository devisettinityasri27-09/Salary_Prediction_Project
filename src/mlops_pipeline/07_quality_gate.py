
import os
import json

print("\n[INFO] Starting Model Quality Gate...\n")

# --------------------------------
# 1. Load evaluation report
# --------------------------------
REPORT_PATH = "reports/lab7_model_evaluation.json"

if not os.path.exists(REPORT_PATH):
    raise FileNotFoundError(
        "Model evaluation report not found! Run evaluation first."
    )

with open(REPORT_PATH, "r") as file:
    metrics = json.load(file)

# --------------------------------
# 2. Define quality thresholds
# --------------------------------
MIN_R2 = 0.70
MAX_MAE = 15000
MAX_RMSE = 20000

# --------------------------------
# 3. Extract metrics (correct keys)
# --------------------------------
r2_score = metrics.get("R2_Score")
mae = metrics.get("MAE")
rmse = metrics.get("RMSE")

print("[INFO] Model Performance:")
print(f"R2 Score : {r2_score:.4f}")
print(f"MAE     : ₹{mae:,.2f}")
print(f"RMSE    : ₹{rmse:,.2f}")

# --------------------------------
# 4. Quality checks
# --------------------------------
checks = {
    "R2 Score Check": r2_score is not None and r2_score >= MIN_R2,
    "MAE Check": mae is not None and mae <= MAX_MAE,
    "RMSE Check": rmse is not None and rmse <= MAX_RMSE
}

print("\n[INFO] Running Quality Checks...\n")

for check, passed in checks.items():
    status = "PASSED" if passed else "FAILED"
    print(f"{check}: {status}")

# --------------------------------
# 5. Save quality report
# --------------------------------
quality_status = all(checks.values())

os.makedirs("artifacts", exist_ok=True)

quality_report = {
    "quality_status": "PASSED" if quality_status else "FAILED",
    "thresholds": {
        "min_r2": MIN_R2,
        "max_mae": MAX_MAE,
        "max_rmse": MAX_RMSE
    },
    "metrics": metrics,
    "checks": checks
}

with open("artifacts/quality_gate_report.json", "w") as file:
    json.dump(quality_report, file, indent=4)

# --------------------------------
# 6. Final decision
# --------------------------------
if quality_status:
    print("\n[SUCCESS] MODEL PASSED THE QUALITY GATE!")
    print("[SUCCESS] Report saved to artifacts/quality_gate_report.json")
else:
    print("\n[ERROR] MODEL FAILED THE QUALITY GATE!")
    print("[ERROR] Report saved to artifacts/quality_gate_report.json")
    raise SystemExit(1)
