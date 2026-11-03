
import subprocess
import sys
import os

def run_script(script_path):
    print("\n" + "=" * 60)
    print(f"EXECUTING: {script_path}")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, script_path],
        check=False
    )

    if result.returncode != 0:
        print(f"\nERROR: {script_path} failed!")
        sys.exit(result.returncode)

    print(f"\nSUCCESS: {script_path} completed!")


def main():
    print("\nSALARY PREDICTION - COMPLETE ML PIPELINE")
    print("=" * 60)

    scripts = [
        "src/validate_data.py",
        "src/preprocess_pipeline.py",
        "src/validate_outputs.py",
        "src/train.py",
        "src/evaluate.py",
        "src/validate_reproducibility.py",
        "src/train_mlflow.py",
        "src/train_registry.py",
        "src/automate_lifecycle.py",
        "src/generate_registry_report.py"
    ]

    for script in scripts:
        if not os.path.exists(script):
            print(f"\nERROR: File not found - {script}")
            sys.exit(1)

        run_script(script)

    print("\n" + "=" * 60)
    print("COMPLETE ML PIPELINE EXECUTED SUCCESSFULLY!")
    print("Salary Prediction MLOps Pipeline - Lab 7")
    print("=" * 60)


if __name__ == "__main__":
    main()
