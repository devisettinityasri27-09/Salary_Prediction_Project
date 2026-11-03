
import subprocess
import sys

print("\n========================================")
print("     LAB 9: FAILURE & RECOVERY PIPELINE")
print("========================================\n")

scripts = [
    "src/mlops_pipeline/10_failure_injection.py",
    "src/mlops_pipeline/11_recovery.py"
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
print("[SUCCESS] LAB 9 PIPELINE COMPLETED!")
print("========================================")
