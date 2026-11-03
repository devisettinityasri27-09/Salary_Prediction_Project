
import pandas as pd
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_PATH = "data/raw/Salary _Data.csv"
PROCESSED_PATH = "data/processed"


def preprocess_data():

    print("\n[INFO] Starting Salary Prediction Preprocessing...")

    df = pd.read_csv(DATA_PATH)

    # Remove completely empty rows
    df.dropna(how="all", inplace=True)

    # Remove duplicate records
    df.drop_duplicates(inplace=True)

    # Separate features and target
    X = df.drop("Salary", axis=1)
    y = pd.to_numeric(df["Salary"], errors="coerce")

    # Remove rows where target is missing
    valid_indices = y.notna()
    X = X.loc[valid_indices]
    y = y.loc[valid_indices]

    # Identify columns
    numeric_cols = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    # Numeric preprocessing
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("[INFO] Fitting preprocessing pipeline...")

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Create output folder
    os.makedirs(PROCESSED_PATH, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save processed data
    np.save(
        f"{PROCESSED_PATH}/X_train_final.npy",
        X_train_processed
    )
    np.save(
        f"{PROCESSED_PATH}/X_test_final.npy",
        X_test_processed
    )
    np.save(
        f"{PROCESSED_PATH}/y_train.npy",
        y_train.to_numpy()
    )
    np.save(
        f"{PROCESSED_PATH}/y_test.npy",
        y_test.to_numpy()
    )

    # Save preprocessing pipeline
    import joblib
    joblib.dump(preprocessor, "models/preprocessor.pkl")

    # Save metadata
    metadata = {
        "dataset": "Salary Prediction Dataset",
        "original_shape": list(df.shape),
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "features_after_encoding": int(X_train_processed.shape[1]),
        "numeric_features": numeric_cols,
        "categorical_features": categorical_cols,
        "target": "Salary"
    }

    with open(
        f"{PROCESSED_PATH}/dataset_metadata.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=4)

    print("\n[SUCCESS] Preprocessing completed!")
    print(f"[INFO] Training shape: {X_train_processed.shape}")
    print(f"[INFO] Testing shape: {X_test_processed.shape}")
    print("[SUCCESS] Processed files and preprocessor saved!")

    return True


if __name__ == "__main__":
    preprocess_data()
