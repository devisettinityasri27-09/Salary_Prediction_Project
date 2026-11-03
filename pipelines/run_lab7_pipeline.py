
import subprocess
import sys

print("\n========================================")
print("     LAB 7: COMPLETE MLOPS PIPELINE")
print("========================================\n")

scripts = [
    "src/mlops_pipeline/01_load_data.py",
    "src/mlops_pipeline/02_validate_data.py",
    "src/mlops_pipeline/03_preprocessing.py",
    "src/mlops_pipeline/04_validate_outputs.py",
    "src/mlops_pipeline/05_train_registry.py",
    "src/mlops_pipeline/06_model_evaluation.py",
    "src/mlops_pipeline/08_automate_lifecycle.py",
    "src/mlops_pipeline/09_inference.py"
]

for script in scripts:
    print(f"\n[RUNNING] {script}")
    result = subprocess.run(
        [sys.executable, script],
        check=False
    )

    if result.returncode != 0:
        print(f"\n[FAILED] Pipeline stopped at {script}")
        sys.exit(result.returncode)

print("\n========================================")
print("[SUCCESS] LAB 7 PIPELINE COMPLETED!")
print("========================================")
