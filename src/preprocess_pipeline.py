import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def run_preprocessing():

    print("[INFO] Starting Sklearn Pipeline Preprocessing...")

    # 1. Load raw data
    data_path = "data/raw/Salary _Data.csv"
    df = pd.read_csv(data_path)

    print("[INFO] Dataset loaded successfully!")
    print(f"[INFO] Original shape: {df.shape}")

    # 2. Basic cleaning
    df = df.dropna(how="all")
    df = df.drop_duplicates()

    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df["Years of Experience"] = pd.to_numeric(
        df["Years of Experience"],
        errors="coerce"
    )
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")

    categorical_columns = [
        "Gender",
        "Education Level",
        "Job Title"
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna(df[column].mode()[0])

    # 3. Separate features and target
    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    # 4. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # 5. Identify columns
    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    num_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # 6. Numerical pipeline
    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # 7. Categorical pipeline
    cat_pipeline = Pipeline(
        steps=[
            (
                "ohe",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # 8. Combined preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )

    # 9. Transform data
    print("[INFO] Fitting preprocessing pipeline...")

    X_train_final = preprocessor.fit_transform(X_train)

    print("[INFO] Transforming test data...")

    X_test_final = preprocessor.transform(X_test)

    # 10. Save processed data
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    np.save(
        "data/processed/X_train_final.npy",
        X_train_final
    )

    np.save(
        "data/processed/X_test_final.npy",
        X_test_final
    )

    np.save(
        "data/processed/y_train.npy",
        y_train.to_numpy()
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.to_numpy()
    )

    # 11. Save preprocessing pipeline
    joblib.dump(
        preprocessor,
        "models/preprocessor.pkl"
    )

    # 12. Save metadata
    metadata = {
        "dataset_name": "Salary Prediction Dataset",
        "train_shape": list(X_train_final.shape),
        "test_shape": list(X_test_final.shape),
        "numerical_features": num_cols,
        "categorical_features": cat_cols,
        "target": "Salary"
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("[SUCCESS] Preprocessing completed successfully!")
    print(
        f"[INFO] Training shape: {X_train_final.shape}"
    )
    print(
        f"[INFO] Testing shape: {X_test_final.shape}"
    )
    print(
        "[SUCCESS] Pipeline saved as models/preprocessor.pkl"
    )


if __name__ == "__main__":
    run_preprocessing()