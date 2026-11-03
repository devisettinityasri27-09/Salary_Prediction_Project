
import os
import joblib

print("\n========================================")
print("    LAB 9: AUTOMATED RECOVERY")
print("========================================\n")

MODEL_PATH = "models/lab7_random_forest.pkl"
BACKUP_MODEL_PATH = "models/production_model.pkl"

# --------------------------------
# Check model and recover
# --------------------------------
if os.path.exists(MODEL_PATH):
    print("[SUCCESS] Primary model is available!")

elif os.path.exists(BACKUP_MODEL_PATH):
    print("[WARNING] Primary model missing!")
    print("[INFO] Recovering from backup model...")

    model = joblib.load(BACKUP_MODEL_PATH)

    print("[SUCCESS] Model recovered from backup!")

else:
    print("[ERROR] No model or backup available!")
    raise FileNotFoundError(
        "Recovery failed: No trained model found."
    )

# --------------------------------
# Check preprocessor
# --------------------------------
PREPROCESSOR_PATH = "models/preprocessor.pkl"

if os.path.exists(PREPROCESSOR_PATH):
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    print("[SUCCESS] Preprocessor loaded successfully!")
else:
    raise FileNotFoundError(
        "Recovery failed: Preprocessor not found."
    )

print("\n========================================")
print("[SUCCESS] RECOVERY PROCESS COMPLETED!")
print("========================================")
