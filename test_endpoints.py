import requests
import json
import urllib.request
import os

print("--- TEXT ENDPOINT TEST ---")
r_text = requests.post("http://127.0.0.1:8000/complaint/text", json={"text": "Sewage overflow causing health hazard"})
res_text = r_text.json()
print(f"Priority: {res_text['priority']}")
print(f"ETA Days: {res_text['eta_days']}")
print(f"Transcription/Clean Text: {res_text['complaint_text']}")
print("\n")

print("--- AUDIO ENDPOINT TEST ---")
# Download a short sample speech audio
sample_url = "https://www2.cs.uic.edu/~i101/SoundFiles/preamble10.wav"
audio_path = "test_audio.wav"
try:
    urllib.request.urlretrieve(sample_url, audio_path)
    with open(audio_path, "rb") as f:
        r_audio = requests.post("http://127.0.0.1:8000/complaint/audio", files={"file": f})
    
    res_audio = r_audio.json()
    print(f"Priority: {res_audio['priority']}")
    print(f"ETA Days: {res_audio['eta_days']}")
    print(f"Transcription: {res_audio['complaint_text']}")
except Exception as e:
    print(f"Audio Test Failed: {e}")
finally:
    if os.path.exists(audio_path):
        os.remove(audio_path)
