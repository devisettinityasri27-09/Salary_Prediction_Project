
import subprocess
import sys

print("\n[INFO] Starting Model Lifecycle Automation...\n")

result = subprocess.run(
    [sys.executable, "src/automate_lifecycle.py"],
    check=False
)

if result.returncode == 0:
    print("\n[SUCCESS] Model lifecycle automation completed!")
else:
    print("\n[ERROR] Model lifecycle automation failed!")
    sys.exit(result.returncode)
