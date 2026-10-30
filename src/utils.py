"""
Utility functions for Multilingual Translation and Cross-Platform Text-to-Speech.
"""
import os
import sys
import platform
import tempfile
import logging
import subprocess
import shutil
from typing import Optional
import requests
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
    Synthesizes speech from English caption text into WAV audio.
    Supports both Windows and Linux (Streamlit Community Cloud):
    - Windows: Uses Microsoft SAPI5 via pyttsx3 with thread-safe COM handling.
    - Linux: Uses eSpeak / eSpeak-ng via pyttsx3 with a direct system eSpeak CLI fallback.
    
    Args:
        text (str): English caption to convert to speech.
        
    Returns:
        bytes: Raw WAV audio bytes suitable for Streamlit st.audio().
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech for empty text.")
        
    clean_text = text.strip()
    current_os = platform.system()
    temp_path = None
    
    # ----------------------------------------------------
    # WINDOWS: Microsoft SAPI5 via pyttsx3
    # ----------------------------------------------------
    if current_os == "Windows":
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except Exception as e:
            logger.debug(f"pythoncom.CoInitialize() skipped or failed: {e}")

        try:
            import pyttsx3
            try:
                engine = pyttsx3.init("sapi5")
            except Exception:
                engine = pyttsx3.init()
                
            engine.setProperty("rate", 160)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                temp_path = fp.name
                
            engine.save_to_file(clean_text, temp_path)
            engine.runAndWait()
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
                return audio_bytes
            else:
                raise RuntimeError("Audio file was not generated or is empty.")
        except Exception as e:
            logger.error(f"Windows Text-to-Speech failed: {e}")
            raise RuntimeError(f"Text-to-Speech failed on Windows (SAPI5): {e}")
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
            try:
                import pythoncom
                pythoncom.CoUninitialize()
            except Exception:
                pass

    # ----------------------------------------------------
    # LINUX (Streamlit Community Cloud) & Other Platforms: eSpeak / eSpeak-ng
    # ----------------------------------------------------
    else:
        # Attempt 1: pyttsx3 using espeak driver
        try:
            import pyttsx3
            engine = pyttsx3.init("espeak")
            engine.setProperty("rate", 150)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                temp_path = fp.name
                
            engine.save_to_file(clean_text, temp_path)
            engine.runAndWait()
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                with open(temp_path, "rb") as f:
                    audio_bytes = f.read()
                try:
                    os.remove(temp_path)
                except Exception:
                    pass
                return audio_bytes
        except Exception as espeak_py_err:
            logger.warning(
                f"pyttsx3 with espeak driver encountered an issue: {espeak_py_err}. "
                f"Falling back to direct eSpeak system command..."
            )
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception:
                    pass

        # Attempt 2: Direct eSpeak / eSpeak-ng system CLI command
        espeak_bin = shutil.which("espeak-ng") or shutil.which("espeak")
        if espeak_bin:
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
                    temp_path = fp.name
                    
                cmd = [espeak_bin, "-w", temp_path, "-s", "150", clean_text]
                res = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10,
                    text=True
                )
                
                if res.returncode == 0 and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
                    with open(temp_path, "rb") as f:
                        audio_bytes = f.read()
                    return audio_bytes
                else:
                    logger.error(f"eSpeak CLI failed with returncode {res.returncode}: {res.stderr}")
            except Exception as cli_err:
                logger.error(f"Direct eSpeak command execution failed: {cli_err}")
            finally:
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

        # If all Linux TTS attempts failed, raise a clear actionable error message
        raise RuntimeError(
            "eSpeak speech engine is not available on this server. "
            "Please ensure 'packages.txt' is committed to the repository root with 'espeak', "
            "'espeak-ng', and 'libespeak1' so Streamlit Community Cloud installs the speech engine."
        )
