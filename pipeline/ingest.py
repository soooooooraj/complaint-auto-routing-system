import os
import tempfile
import logging
import whisper
import ffmpeg
import imageio_ffmpeg
import os
import shutil
from pipeline.translate import detect_and_translate

ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
ffmpeg_dir = os.path.dirname(ffmpeg_exe)
ffmpeg_alias = os.path.join(ffmpeg_dir, "ffmpeg.exe")

if not os.path.exists(ffmpeg_alias):
    shutil.copyfile(ffmpeg_exe, ffmpeg_alias)

os.environ["PATH"] += os.pathsep + ffmpeg_dir

logger = logging.getLogger(__name__)

# Lazy load whisper model
_whisper_model = None

def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        logger.info("Loading Whisper base model (this might take a moment on first run)...")
        _whisper_model = whisper.load_model("base")
    return _whisper_model

def process_text(text: str) -> dict:
    """
    Takes raw text input.
    Runs detect_and_translate from pipeline/translate.py
    Returns clean English text + language.
    """
    result = detect_and_translate(text)
    return {
        "clean_text": result["translated"],
        "detected_language": result["detected_language"]
    }

def process_audio(file_path: str) -> dict:
    import subprocess
    temp_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)
    
    try:
        ffmpeg_cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-i", file_path,
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            temp_wav_path,
            "-y"
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        model = get_whisper_model()
        result = model.transcribe(temp_wav_path)
        transcription = result["text"].strip()
        
        return process_text(transcription)
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def process_video(file_path: str) -> dict:
    import subprocess
    temp_fd, temp_wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(temp_fd)
    
    try:
        ffmpeg_cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-i", file_path,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            temp_wav_path,
            "-y"
        ]
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return process_audio(temp_wav_path)
    finally:
        if os.path.exists(temp_wav_path):
            os.remove(temp_wav_path)

def process_input(input_type: str, data: str) -> dict:
    """
    input_type is one of: "text", "audio", "video"
    Returns: {"clean_text": ..., "detected_language": ..., "input_type": input_type}
    """
    if input_type == "text":
        result = process_text(data)
    elif input_type == "audio":
        result = process_audio(data)
    elif input_type == "video":
        result = process_video(data)
    else:
        raise ValueError(f"Unknown input_type: {input_type}. Must be 'text', 'audio', or 'video'")
        
    return {
        "clean_text": result["clean_text"],
        "detected_language": result["detected_language"],
        "input_type": input_type
    }

if __name__ == "__main__":
    import sys
    import json
    
    sys.stdout.reconfigure(encoding='utf-8')
    
    print("Testing process_input with text...")
    test_result = process_input("text", "सड़क पर बड़ा गड्ढा है")
    print(json.dumps(test_result, indent=2, ensure_ascii=False))
