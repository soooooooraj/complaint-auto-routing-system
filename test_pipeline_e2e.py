import subprocess
import time
import requests
import json
import sys

# Ensure utf-8 output matching
sys.stdout.reconfigure(encoding='utf-8')

print("Starting Uvicorn Server...")
proc = subprocess.Popen([r"venv\Scripts\python", "-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"],
                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

# Wait for server to load all ML models into memory
server_started = False
for _ in range(60):
    try:
        r = requests.get("http://127.0.0.1:8000/health")
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

print("Server is up and models are loaded.\n")

tests = [
    {"lang": "English", "text": "Sewage overflow on main road causing health hazard"},
    {"lang": "Hindi", "text": "सड़क पर बड़ा गड्ढा है"},
    {"lang": "Malayalam", "text": "റോഡിൽ വലിയ കുഴി ഉണ്ട്"},
    {"lang": "Tamil", "text": "பாலைவன சாலையில் பெரிய குழி உள்ளது"}
]

try:
    for t in tests:
        print(f"--- Testing {t['lang']} ---")
        print(f"Input: {t['text']}")
        r = requests.post("http://127.0.0.1:8000/complaint/text", json={"text": t['text']})
        if r.status_code == 200:
            data = r.json()
            print(f"Detected Language: {data['detected_language']}")
            print(f"Priority: {data['priority']} (Confidence: {data['priority_confidence']:.2f})")
            print(f"ETA Days: {data['eta_days']}")
            print(f"Assigned Dept: {data['assigned_officers'][0]['department']} (Officer: {data['assigned_officers'][0]['name']})\n")
        else:
            print(f"Error {r.status_code}: {r.text}\n")
except Exception as e:
    print(e)
finally:
    proc.terminate()
    stdout, _ = proc.communicate()
    print("===== SERVER LOG EXCERPT =====")
    lines = stdout.decode("utf-8", errors="ignore").split('\n')
    for line in lines[:15]:
        print(line.strip())
    print("...")
