import subprocess
import requests
import json
import os
import imageio_ffmpeg

print("--- TEXT ENDPOINT TEST ---")
r_text = requests.post("http://127.0.0.1:8000/complaint/text", json={"text": "Sewage overflow causing health hazard"})
res_text = r_text.json()
print("Priority:", res_text['priority'])
print("ETA Days:", res_text['eta_days'])
print("Clean Text:", res_text['complaint_text'])
print("\n")

print("--- AUDIO ENDPOINT TEST ---")
audio_path = "synthetic_test.wav"
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# Create a dummy 3-second audio file
subprocess.run([ffmpeg_exe, "-f", "lavfi", "-i", "sine=frequency=1000:duration=3", "-ac", "1", "-ar", "16000", audio_path, "-y"], capture_output=True)

try:
    with open(audio_path, "rb") as f:
        r_audio = requests.post("http://127.0.0.1:8000/complaint/audio", files={"file": f})
    
    res_audio = r_audio.json()
    print("Priority:", res_audio['priority'])
    print("ETA Days:", res_audio['eta_days'])
    print("Transcription:", res_audio['complaint_text'])
except Exception as e:
    print("Audio Test Failed:", e)
finally:
    if os.path.exists(audio_path):
        os.remove(audio_path)
