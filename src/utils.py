"""
Utility functions for Multilingual Translation and Offline Text-to-Speech.
"""
import os
import tempfile
import logging
import requests
from typing import Optional
from deep_translator import GoogleTranslator, MyMemoryTranslator

logger = logging.getLogger(__name__)

SUPPORTED_LANGUAGES = {
    "Telugu": {"code": "te", "mymemory": "te-IN"},
    "Hindi": {"code": "hi", "mymemory": "hi-IN"},
    "Tamil": {"code": "ta", "mymemory": "ta-IN"},
    "Kannada": {"code": "kn", "mymemory": "kn-IN"},
    "Malayalam": {"code": "ml", "mymemory": "ml-IN"},
}

def translate_caption(text: str, target_language: str) -> str:
    """
    Translates an English caption into one of the supported Indian languages
    (Telugu, Hindi, Tamil, Kannada, Malayalam) using deep-translator with
    multi-source fallback.
    
    Args:
        text (str): English caption to translate.
        target_language (str): Target language name (e.g. 'Telugu', 'Hindi').
        
    Returns:
        str: Translated caption text.
    """
    if not text or not text.strip():
        return ""
        
    lang_info = SUPPORTED_LANGUAGES.get(target_language)
    if not lang_info:
        raise ValueError(f"Language '{target_language}' is not supported. Supported: {list(SUPPORTED_LANGUAGES.keys())}")
        
    code = lang_info["code"]
    mymemory_code = lang_info["mymemory"]
    clean_text = text.strip()
    
    # 1. Attempt using deep-translator's GoogleTranslator
    try:
        translated = GoogleTranslator(source="en", target=code).translate(clean_text)
        if translated and translated.strip():
            return translated.strip()
    except Exception as e:
        logger.warning(f"GoogleTranslator in deep-translator failed: {e}. Trying MyMemoryTranslator...")
        
    # 2. Attempt using deep-translator's MyMemoryTranslator
    try:
        translated = MyMemoryTranslator(source="en-GB", target=mymemory_code).translate(clean_text)
        if translated and translated.strip():
            return translated.strip()
    except Exception as e:
        logger.warning(f"MyMemoryTranslator failed: {e}. Trying Google Translate API endpoint...")
        
    # 3. Direct Google Translate API endpoint fallback
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "en", "tl": code, "dt": "t", "q": clean_text}
        resp = requests.get(url, params=params, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return data[0][0][0].strip()
    except Exception as e:
        logger.error(f"Fallback translation failed: {e}")
        raise RuntimeError(f"Translation failed: Could not connect to translation services. Please check your network connection.")

    raise RuntimeError("Translation service unavailable. Please try again.")

def generate_speech(text: str) -> bytes:
    """
    Synthesizes speech from English caption text using pyttsx3 offline.
    Handles Windows COM threading safely and returns the WAV audio bytes.
    
    Args:
        text (str): English caption to convert to speech.
        
    Returns:
        bytes: Raw WAV audio bytes suitable for Streamlit st.audio().
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech for empty text.")
        
    # Handle Windows COM threading in Streamlit background threads
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception as e:
        logger.debug(f"pythoncom.CoInitialize() skipped or failed: {e}")

    temp_path = None
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", 160) # Natural speaking pace
        
        # Create a temporary WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_path = fp.name
            
        engine.save_to_file(text.strip(), temp_path)
        engine.runAndWait()
        
        # Read the generated audio bytes
        with open(temp_path, "rb") as f:
            audio_bytes = f.read()
            
        return audio_bytes
    except Exception as e:
        logger.error(f"Text-to-speech generation failed: {e}")
        raise RuntimeError(f"Text-to-Speech failed: {e}")
    finally:
        # Clean up temporary file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass
        # Uninitialize COM for this thread
        try:
            import pythoncom
            pythoncom.CoUninitialize()
        except Exception:
            pass
