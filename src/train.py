import os
import joblib
import numpy as np

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor
)


def train_models():

    # ==========================================
    # LOAD PROCESSED TRAINING DATA
    # ==========================================

    X_train = np.load(
        "data/processed/X_train_final.npy"
    )

    y_train = np.load(
        "data/processed/y_train.npy"
    )

    print("Processed training data loaded successfully!")
    print("Training data shape:", X_train.shape)
    print("Training target shape:", y_train.shape)

    # ==========================================
    # DEFINE BASELINE MODELS
    # ==========================================

    models = {

        "Linear Regression": LinearRegression(),

        "Ridge Regression": Ridge(
            alpha=1.0
        ),

        "Lasso Regression": Lasso(
            alpha=0.01,
            max_iter=10000
        ),

        "Elastic Net": ElasticNet(
            alpha=0.01,
            l1_ratio=0.5,
            max_iter=10000
        ),

        "Decision Tree": DecisionTreeRegressor(
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),

        "Gradient Boosting": GradientBoostingRegressor(
            random_state=42
        ),

        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            random_state=42
        )
    }

    # ==========================================
    # TRAIN MODELS
    # ==========================================

    trained_models = {}

    print("\n==========================================")
    print("MODEL TRAINING")
    print("==========================================")

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        model.fit(
            X_train,
            y_train
        )

        trained_models[model_name] = model

        print(
            f"{model_name} trained successfully!"
        )

    # ==========================================
    # CREATE MODELS DIRECTORY
    # ==========================================

    os.makedirs(
        "models",
        exist_ok=True
    )

    # ==========================================
    # SAVE TRAINED MODELS
    # ==========================================

    print("\n==========================================")
    print("SAVING MODELS")
    print("==========================================")

    for model_name, model in trained_models.items():

        file_name = (
            model_name
            .replace(" ", "_")
            .lower()
        )

        model_path = (
            f"models/{file_name}_baseline.pkl"
        )

        joblib.dump(
            model,
            model_path
        )

        print(
            f"{model_name} saved to: {model_path}"
        )

    print(
        "\nAll baseline models trained "
        "and saved successfully!"
    )


# ==========================================
# RUN SCRIPT
# ==========================================

if __name__ == "__main__":
    train_models()