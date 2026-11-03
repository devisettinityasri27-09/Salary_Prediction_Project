
import os
import joblib
import numpy as np

print("\n========================================")
print("    LAB 9: FAILURE INJECTION TESTING")
print("========================================\n")

# --------------------------------
# Test 1: Missing model file
# --------------------------------
print("[TEST 1] Testing missing model file...")

missing_model_path = "models/non_existing_model.pkl"

try:
    joblib.load(missing_model_path)

except FileNotFoundError:
    print("[SUCCESS] Missing model failure detected!")

except Exception as e:
    print(f"[INFO] Error detected: {e}")

# --------------------------------
# Test 2: Corrupted input data
# --------------------------------
print("\n[TEST 2] Testing corrupted input data...")

try:
    corrupted_data = np.array([[np.nan, np.inf, -np.inf]])

    if not np.isfinite(corrupted_data).all():
        raise ValueError("Input contains NaN or infinite values!")

except ValueError as e:
    print(f"[SUCCESS] Corrupted input detected: {e}")

# --------------------------------
# Test 3: Invalid prediction input
# --------------------------------
print("\n[TEST 3] Testing invalid prediction input...")

try:
    invalid_input = np.array([["invalid", "data", None]])

    if invalid_input.dtype == object:
        raise TypeError("Invalid data type detected!")

except TypeError as e:
    print(f"[SUCCESS] Invalid prediction input detected: {e}")

# --------------------------------
# Test 4: Model file existence check
# --------------------------------
print("\n[TEST 4] Testing model availability...")

model_path = "models/lab7_random_forest.pkl"

if os.path.exists(model_path):
    print("[SUCCESS] Trained model is available!")
else:
    print("[ERROR] Model file missing!")

# --------------------------------
# Final report
# --------------------------------
print("\n========================================")
print("[SUCCESS] FAILURE INJECTION TESTS DONE!")
print("========================================")
