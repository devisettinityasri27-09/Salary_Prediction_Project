import os
import pandas as pd


def validate_salary_data():

    print("[INFO] Validating Salary Prediction Dataset...")

    data_path = "data/raw/Salary _Data.csv"

    if not os.path.exists(data_path):

        print(
            f"[ERROR] Target file not found at: {data_path}"
        )

        return False

    df = pd.read_csv(data_path)

    required_columns = [
        "Age",
        "Gender",
        "Education Level",
        "Job Title",
        "Years of Experience",
        "Salary"
    ]

    errors = []

    # Check required columns
    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:

        errors.append(
            f"Missing columns: {missing_columns}"
        )

    # Check numeric columns
    numeric_columns = [
        "Age",
        "Years of Experience",
        "Salary"
    ]

    for column in numeric_columns:

        if column in df.columns:

            original = df[column]

            # Convert to numeric
            converted = pd.to_numeric(
                original,
                errors="coerce"
            )

            # Find values that are neither numeric nor missing
            invalid_values = original[
                original.notna() &
                converted.isna()
            ]

            if len(invalid_values) > 0:

                errors.append(
                    f"{column} contains invalid "
                    f"non-numeric values: "
                    f"{invalid_values.tolist()}"
                )

    # Check negative values
    if "Age" in df.columns:

        age = pd.to_numeric(
            df["Age"],
            errors="coerce"
        )

        if (age.dropna() < 0).any():

            errors.append(
                "Age contains negative values."
            )

    if "Years of Experience" in df.columns:

        experience = pd.to_numeric(
            df["Years of Experience"],
            errors="coerce"
        )

        if (experience.dropna() < 0).any():

            errors.append(
                "Years of Experience contains negative values."
            )

    if "Salary" in df.columns:

        salary = pd.to_numeric(
            df["Salary"],
            errors="coerce"
        )

        if (salary.dropna() < 0).any():

            errors.append(
                "Salary contains negative values."
            )

    # Report
    if errors:

        print("[ERROR] Data Validation FAILED.")

        for error in errors:
            print(f"  - {error}")

        os.makedirs("artifacts", exist_ok=True)

        with open(
            "artifacts/data_validation_errors.txt",
            "w"
        ) as f:

            for error in errors:
                f.write(error + "\n")

        return False

    print("[SUCCESS] Data Validation PASSED.")

    print(
        f"[INFO] Dataset contains {len(df)} records."
    )

    print(
        f"[INFO] Missing values found: "
        f"{df.isnull().sum().sum()}"
    )

    return True


if __name__ == "__main__":
    validate_salary_data()