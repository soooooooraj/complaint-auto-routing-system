import subprocess
import requests
import json
import os
import imageio_ffmpeg

print("--- OGG AUDIO ENDPOINT TEST ---")
audio_path = "whatsapp_test.ogg"
ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

# Create a dummy 3-second OGG audio file
print("Generating dummy test.ogg...")
subprocess.run([ffmpeg_exe, "-f", "lavfi", "-i", "sine=frequency=1000:duration=3", "-ac", "1", "-ar", "16000", "-c:a", "libvorbis", audio_path, "-y"], capture_output=True)

try:
    with open(audio_path, "rb") as f:
        print("Sending to API...")
        r_audio = requests.post("http://127.0.0.1:8000/complaint/audio", files={"file": ("whatsapp_test.ogg", f, "audio/ogg")})
    
    print(f"Status Code: {r_audio.status_code}")
    
    try:
        res_audio = r_audio.json()
        print(json.dumps(res_audio, indent=2))
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        print(f"Raw Response: {r_audio.text}")
        
except Exception as e:
    print("Audio Test Failed:", e)
finally:
    if os.path.exists(audio_path):
        os.remove(audio_path)
