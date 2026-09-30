import subprocess
import sys


def run_step(step_name, command):

    print("\n")
    print("=" * 60)
    print(f"LAB 5 - {step_name}")
    print("=" * 60)

    result = subprocess.run(
        command,
        shell=True
    )

    if result.returncode != 0:

        print("\n==========================================")
        print(f"FAILED: {step_name}")
        print("==========================================")

        sys.exit(result.returncode)

    print(f"\n{step_name} completed successfully!")


def main():

    print("\n==========================================")
    print("SALARY PREDICTION - LAB 5")
    print("COMPLETE ML PIPELINE")
    print("==========================================")

    # ==========================================
    # STEP 1: PREPROCESSING
    # ==========================================

    run_step(
        "DATA PREPROCESSING",
        "python src/preprocess.py"
    )

    # ==========================================
    # STEP 2: MODEL TRAINING
    # ==========================================

    run_step(
        "MODEL TRAINING",
        "python src/train.py"
    )

    # ==========================================
    # STEP 3: MODEL EVALUATION
    # ==========================================

    run_step(
        "MODEL EVALUATION",
        "python src/evaluate.py"
    )

    # ==========================================
    # STEP 4: REPRODUCIBILITY
    # ==========================================

    run_step(
        "REPRODUCIBILITY VALIDATION",
        "python src/validate_reproducibility.py"
    )

    # ==========================================
    # STEP 5: MLFLOW TRACKING
    # ==========================================

    run_step(
        "MLFLOW EXPERIMENT TRACKING",
        "python src/train_mlflow.py"
    )

    # ==========================================
    # COMPLETED
    # ==========================================

    print("\n")
    print("=" * 60)
    print("LAB 5 COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    print("\nComplete pipeline executed:")
    print("1. Data preprocessing       [DONE]")
    print("2. Model training           [DONE]")
    print("3. Model evaluation         [DONE]")
    print("4. Reproducibility          [DONE]")
    print("5. MLflow tracking          [DONE]")

    print("\nGenerated artifacts:")
    print("- Processed datasets")
    print("- Trained baseline models")
    print("- Evaluation results")
    print("- Prediction error analysis")
    print("- Reproducibility result")
    print("- MLflow experiment")
    print("- MLflow model")
    print("- Actual vs predicted plot")
    print("- Residual plot")


if __name__ == "__main__":
    main()