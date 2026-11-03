
import os
import pandas as pd
import joblib

print("\n[INFO] Starting Salary Prediction Inference...\n")

# --------------------------------
# 1. Load trained model
# --------------------------------
MODEL_PATH = "models/lab7_random_forest.pkl"
PREPROCESSOR_PATH = "models/preprocessor.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(f"Preprocessor not found: {PREPROCESSOR_PATH}")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

print("[SUCCESS] Model loaded!")
print("[SUCCESS] Preprocessor loaded!")

# --------------------------------
# 2. Input employee details
# --------------------------------
employee_data = {
    "Age": 30,
    "Gender": "Female",
    "Education Level": "Bachelor's",
    "Job Title": "Data Scientist",
    "Years of Experience": 5
}

input_df = pd.DataFrame([employee_data])

print("\n[INFO] Employee Details:")
print(input_df.to_string(index=False))

# --------------------------------
# 3. Preprocess input
# --------------------------------
processed_input = preprocessor.transform(input_df)

print("\n[SUCCESS] Input preprocessing completed!")

# --------------------------------
# 4. Predict salary
# --------------------------------
predicted_salary = model.predict(processed_input)[0]

# --------------------------------
# 5. Display result
# --------------------------------
print("\n====================================")
print("       SALARY PREDICTION")
print("====================================")
print(f"Predicted Salary: ₹{predicted_salary:,.2f}")
print("====================================")

print("\n[SUCCESS] Inference completed!")
