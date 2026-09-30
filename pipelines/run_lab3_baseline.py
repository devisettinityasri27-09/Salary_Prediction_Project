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
    print("SALARY PREDICTION - LAB 3 BASELINE")
    print("==========================================")

    # ==========================================
    # STEP 1: PREPROCESS DATA
    # ==========================================

    run_command(
        "python src/preprocess.py"
    )

    # ==========================================
    # STEP 2: TRAIN BASELINE MODELS
    # ==========================================

    run_command(
        "python src/train.py"
    )

    # ==========================================
    # STEP 3: EVALUATE MODELS
    # ==========================================

    run_command(
        "python src/evaluate.py"
    )

    # ==========================================
    # STEP 4: VALIDATE REPRODUCIBILITY
    # ==========================================

    run_command(
        "python src/validate_reproducibility.py"
    )

    # ==========================================
    # COMPLETED
    # ==========================================

    print("\n==========================================")
    print("LAB 3 BASELINE PIPELINE COMPLETED!")
    print("==========================================")

    print("\nGenerated outputs include:")
    print("- Processed dataset")
    print("- Baseline models")
    print("- Evaluation results")
    print("- Prediction error analysis")
    print("- Reproducibility result")


if __name__ == "__main__":
    main()