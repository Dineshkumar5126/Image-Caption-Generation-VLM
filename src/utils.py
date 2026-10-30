"""
Utility functions for Multilingual Translation and Neural Text-to-Speech.
"""
import asyncio
import concurrent.futures
import logging
import time
from typing import Optional
import edge_tts
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

# High-quality Microsoft Neural Voice (Indian English) used consistently on Windows and Linux
TTS_VOICE = "en-IN-NeerjaNeural"


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


async def _synthesize_edge_tts(text: str) -> bytes:
    """
    Internal coroutine to stream MP3 audio chunks from edge-tts.
    """
    communicate = edge_tts.Communicate(text, TTS_VOICE, connect_timeout=20)
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


def generate_speech(text: str) -> bytes:
    """
    Synthesizes speech from English caption text into MP3 audio using edge-tts
    with Microsoft Neural voice 'en-IN-NeerjaNeural'.
    
    Ensures identical, natural-sounding voice output across both Windows localhost
    and Linux Streamlit Community Cloud. Execution is safely isolated in a dedicated
    thread with its own event loop to guarantee compatibility with Streamlit's runtime.

    Args:
        text (str): English caption to convert to speech.
        
    Returns:
        bytes: Raw MP3 audio bytes suitable for Streamlit st.audio().
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech for empty text.")
        
    clean_text = text.strip()

    def _run_in_isolated_thread() -> bytes:
        last_error = None
        for attempt in range(2):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_data = loop.run_until_complete(_synthesize_edge_tts(clean_text))
                if audio_data:
                    return audio_data
            except Exception as e:
                last_error = e
                logger.warning(f"edge-tts synthesis attempt {attempt + 1} failed: {e}")
                if attempt == 0:
                    time.sleep(1)
                    continue
            finally:
                loop.close()

        if last_error:
            raise RuntimeError(f"Text-to-Speech synthesis failed: {last_error}")
        raise RuntimeError("Text-to-Speech synthesis produced empty audio.")

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_in_isolated_thread)
        return future.result(timeout=45)
