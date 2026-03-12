import sys
import subprocess
import os

os.environ["PYTHONPATH"] = "."

commands = [
    [r"venv\Scripts\python", r"models\priority_classifier.py"],
    [r"venv\Scripts\python", r"models\eta_regressor.py"],
    [r"venv\Scripts\python", r"models\officer_router.py"],
    [r"venv\Scripts\python", r"models\similarity_search.py"]
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ.copy())
print("All models trained successfully!")
