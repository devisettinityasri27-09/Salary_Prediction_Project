import os
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def preprocess_data():

    # Load raw dataset
    data_path = "data/raw/Salary _Data.csv"
    df = pd.read_csv(data_path)

    print("Dataset loaded successfully!")
    print("Original shape:", df.shape)

    # Remove completely empty rows
    df = df.dropna(how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Convert numerical columns
    df["Age"] = pd.to_numeric(
        df["Age"],
        errors="coerce"
    )

    df["Years of Experience"] = pd.to_numeric(
        df["Years of Experience"],
        errors="coerce"
    )

    df["Salary"] = pd.to_numeric(
        df["Salary"],
        errors="coerce"
    )

    # Fill numerical missing values
    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    df["Years of Experience"] = df[
        "Years of Experience"
    ].fillna(
        df["Years of Experience"].median()
    )

    df["Salary"] = df["Salary"].fillna(
        df["Salary"].median()
    )

    # Fill categorical missing values
    categorical_columns = [
        "Gender",
        "Education Level",
        "Job Title"
    ]

    for column in categorical_columns:
        df[column] = df[column].fillna(
            df[column].mode()[0]
        )

    # Separate features and target
    X = df.drop("Salary", axis=1)
    y = df["Salary"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # Identify column types
    cat_cols = X_train.select_dtypes(
        include=["object", "category"]
    ).columns

    num_cols = X_train.select_dtypes(
        include=["int64", "float64"]
    ).columns

    # Scale numerical features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train[num_cols]
    )

    X_test_scaled = scaler.transform(
        X_test[num_cols]
    )

    # Encode categorical features
    ohe = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    X_train_encoded = ohe.fit_transform(
        X_train[cat_cols]
    )

    X_test_encoded = ohe.transform(
        X_test[cat_cols]
    )

    # Combine numerical and categorical features
    X_train_final = np.hstack(
        (
            X_train_scaled,
            X_train_encoded
        )
    )

    X_test_final = np.hstack(
        (
            X_test_scaled,
            X_test_encoded
        )
    )

    # Create directories if required
    os.makedirs(
        "data/processed",
        exist_ok=True
    )

    os.makedirs(
        "models",
        exist_ok=True
    )

    # Save processed arrays
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
        y_train.values
    )

    np.save(
        "data/processed/y_test.npy",
        y_test.values
    )

    # Save preprocessing objects
    joblib.dump(
        scaler,
        "models/scaler.pkl"
    )

    joblib.dump(
        ohe,
        "models/ohe.pkl"
    )

    # Save metadata
    metadata = {
        "dataset_name": "Salary Prediction Dataset",
        "train_shape": X_train_final.shape,
        "test_shape": X_test_final.shape,
        "numerical_features": list(num_cols),
        "categorical_features": list(cat_cols),
        "target": "Salary"
    }

    with open(
        "data/processed/dataset_metadata.json",
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=4
        )

    print("Preprocessing completed successfully!")
    print("Training shape:", X_train_final.shape)
    print("Testing shape:", X_test_final.shape)


if __name__ == "__main__":
    preprocess_data()