import logging
from langdetect import detect
from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)

def detect_and_translate(text):
    """
    Detects language of the input text.
    If English ('en'), returns as is.
    If Hindi ('hi'), Malayalam ('ml'), or Marathi ('mr') (or others), translates to English.
    """
    try:
        lang = detect(text)
    except Exception as e:
        logger.warning(f"Could not detect language for text: '{text[:30]}...'. Defaulting to 'en'. Error: {e}")
        lang = "en"
        
    if lang == "en":
        return {
            "original": text,
            "translated": text,
            "detected_language": lang
        }
    
    try:
        translator = GoogleTranslator(source='auto', target='en')
        translated_text = translator.translate(text)
    except Exception as e:
        logger.error(f"Translation failed for text: '{text[:30]}...'. Error: {e}")
        translated_text = text # Fallback to original text on failure
        
    return {
        "original": text,
        "translated": translated_text,
        "detected_language": lang
    }

def batch_translate(complaints_list):
    """
    Takes a list of complaint texts and runs detect_and_translate on each.
    """
    results = []
    for text in complaints_list:
        results.append(detect_and_translate(text))
    return results

if __name__ == "__main__":
    import sys
    import json
    
    sys.stdout.reconfigure(encoding='utf-8')
    
    test_inputs = [
        "The road has a large pothole", # English
        "सड़क पर बड़ा गड्ढा है",          # Hindi
        "റോഡിൽ വലിയ കുഴി ഉണ്ട്",           # Malayalam
        "रस्त्यावर मोठा खड्डा आहे"           # Marathi
    ]
    
    results = batch_translate(test_inputs)
    print(json.dumps(results, indent=2, ensure_ascii=False))
