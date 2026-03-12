import subprocess
import time
import requests
import json
import sys
import os

sys.stdout.reconfigure(encoding='utf-8')

commands = [
    [r"venv\Scripts\python", r"data\generate_data.py"],
    [r"venv\Scripts\python", r"models\priority_classifier.py"],
    [r"venv\Scripts\python", r"models\eta_regressor.py"],
    [r"venv\Scripts\python", r"models\officer_router.py"],
    [r"venv\Scripts\python", r"models\similarity_search.py"]
]

os.environ["PYTHONPATH"] = "."

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, env=os.environ.copy())

print("\nStarting Uvicorn Server to test similarity search...")
proc = subprocess.Popen([r"venv\Scripts\python", "-m", "uvicorn", "api.main:app", "--port", "8002"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

server_started = False
for _ in range(60):
    try:
        r = requests.get("http://127.0.0.1:8002/health")
        if r.status_code == 200 and r.json().get("models_loaded"):
            server_started = True
            break
    except Exception:
        pass
    time.sleep(1)

if not server_started:
    print("Server failed to start!")
    proc.terminate()
    sys.exit(1)

try:
    print('\nSending POST /complaint/text {"text": "No water supply for 3 days"}')
    r = requests.post("http://127.0.0.1:8002/complaint/text", json={"text": "No water supply for 3 days"})
    data = r.json()
    print("Similar Complaints:")
    print(json.dumps(data["similar_complaints"], indent=2, ensure_ascii=False))
except Exception as e:
    print(e)
finally:
    proc.terminate()
