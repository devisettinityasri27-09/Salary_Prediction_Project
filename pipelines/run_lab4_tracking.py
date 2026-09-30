import subprocess
import sys


def run_command(command):

    print("\n==========================================")
    print(f"Running: {command}")
    print("==========================================")

    result = subprocess.run(
        command,
        shell=True
    )

    if result.returncode != 0:
        print(f"\nCommand failed: {command}")
        sys.exit(result.returncode)


def main():

    print("\n==========================================")
    print("SALARY PREDICTION - LAB 4 MLFLOW")
    print("==========================================")

    # ==========================================
    # RUN MLFLOW TRAINING AND TRACKING
    # ==========================================

    run_command(
        "python src/train_mlflow.py"
    )

    # ==========================================
    # COMPLETED
    # ==========================================

    print("\n==========================================")
    print("LAB 4 MLFLOW PIPELINE COMPLETED!")
    print("==========================================")

    print("\nMLflow experiment:")
    print("Salary_Prediction")

    print("\nTracked items:")
    print("- Parameters")
    print("- MAE")
    print("- MSE")
    print("- RMSE")
    print("- R²")
    print("- Actual vs Predicted plot")
    print("- Residual plot")
    print("- Dataset metadata")
    print("- Random Forest model")


if __name__ == "__main__":
    main()