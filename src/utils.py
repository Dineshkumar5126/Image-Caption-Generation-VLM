"""
Utility functions for Multilingual Translation and Hybrid Text-to-Speech
(Primary: Online Microsoft Neural Edge-TTS; Fallback: Offline Windows SAPI5).
"""
import asyncio
import concurrent.futures
import logging
import os
import platform
import socket
import tempfile
import time
from typing import NamedTuple, Optional
import aiohttp
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

# Online Neural Voice (Primary)
ONLINE_TTS_VOICE = "en-IN-NeerjaNeural"


class TTSOutput(NamedTuple):
    """Container for speech synthesis results."""
    audio_bytes: bytes
    format: str       # 'audio/mp3' or 'audio/wav'
    voice_name: str   # Exact name of voice used
    is_online: bool   # True if online neural voice, False if offline fallback


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
        raise RuntimeError("Translation failed: Could not connect to translation services. Please check your network connection.")

    raise RuntimeError("Translation service unavailable. Please try again.")


async def _synthesize_edge_tts(text: str) -> bytes:
    """
    Streams MP3 audio chunks from edge-tts using the exact neural voice en-IN-NeerjaNeural.
    Forces IPv4 TCPConnector to avoid IPv6 getaddrinfo delays or connection timeouts.
    """
    connector = aiohttp.TCPConnector(family=socket.AF_INET)
    communicate = edge_tts.Communicate(
        text,
        ONLINE_TTS_VOICE,
        connector=connector,
        connect_timeout=15
    )
    audio_chunks = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            audio_chunks.append(chunk["data"])
    return b"".join(audio_chunks)


def _synthesize_offline_sapi5(text: str) -> tuple[bytes, str, str]:
    """
    Synthesizes speech locally using Windows SAPI5 via pyttsx3.
    Inspects installed system voices and selects a female voice (e.g. Microsoft Zira)
    if available; otherwise uses the default installed system voice.
    
    Returns:
        tuple[bytes, str, str]: (wav_audio_bytes, 'audio/wav', voice_name)
    """
    if platform.system() != "Windows":
        raise RuntimeError("Offline SAPI5 speech synthesis is only supported on Windows.")

    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception as e:
        logger.debug(f"pythoncom.CoInitialize() skipped or failed: {e}")

    temp_path = None
    try:
        import pyttsx3
        try:
            engine = pyttsx3.init("sapi5")
        except Exception:
            engine = pyttsx3.init()

        engine.setProperty("rate", 160)

        # Inspect installed voices and prefer female voice
        voices = engine.getProperty("voices")
        selected_voice = None
        for v in voices:
            v_gender = str(getattr(v, "gender", "")).lower()
            v_name = getattr(v, "name", "").lower()
            if "female" in v_gender or "zira" in v_name or "female" in v_name:
                selected_voice = v
                break

        if selected_voice:
            engine.setProperty("voice", selected_voice.id)
            voice_name = getattr(selected_voice, "name", "Microsoft Zira Desktop")
        elif voices:
            engine.setProperty("voice", voices[0].id)
            voice_name = getattr(voices[0], "name", "System Default Voice")
        else:
            voice_name = "Default SAPI5 Voice"

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as fp:
            temp_path = fp.name

        engine.save_to_file(text, temp_path)
        engine.runAndWait()

        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            with open(temp_path, "rb") as f:
                wav_bytes = f.read()
            return wav_bytes, "audio/wav", voice_name
        else:
            raise RuntimeError("Offline speech generation produced an empty audio file.")
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


def generate_speech(text: str) -> TTSOutput:
    """
    Synthesizes speech from English caption text.
    
    1. Primary (Online): Uses edge-tts with Microsoft Neural voice 'en-IN-NeerjaNeural'.
       Produces MP3 audio bytes.
    2. Fallback (Offline): If online TTS fails (no internet, timeout, or service error),
       automatically falls back to Windows SAPI5 via pyttsx3, inspecting installed voices
       and selecting a female voice (e.g. Microsoft Zira Desktop) or default system voice.
       Produces WAV audio bytes.
       
    Args:
        text (str): English caption to convert to speech.
        
    Returns:
        TTSOutput: NamedTuple containing (audio_bytes, format, voice_name, is_online).
    """
    if not text or not text.strip():
        raise ValueError("Cannot synthesize speech for empty text.")
        
    clean_text = text.strip()
    online_error = None

    # ---------------------------------------------------------
    # 1. PRIMARY: Online Neural TTS (edge-tts: en-IN-NeerjaNeural)
    # ---------------------------------------------------------
    def _run_online_in_thread() -> bytes:
        last_err = None
        for attempt in range(2):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                audio_data = loop.run_until_complete(_synthesize_edge_tts(clean_text))
                if audio_data and len(audio_data) > 0:
                    return audio_data
            except Exception as e:
                last_err = e
                logger.warning(f"Online edge-tts attempt {attempt + 1} failed: {e}")
                time.sleep(0.5)
            finally:
                loop.close()
        raise RuntimeError(f"Edge-TTS failed: {last_err}")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_online_in_thread)
            mp3_bytes = future.result(timeout=25)
            if mp3_bytes and len(mp3_bytes) > 0:
                logger.info(f"Generated speech using online neural voice: {ONLINE_TTS_VOICE}")
                return TTSOutput(
                    audio_bytes=mp3_bytes,
                    format="audio/mp3",
                    voice_name=ONLINE_TTS_VOICE,
                    is_online=True
                )
    except Exception as e:
        online_error = e
        logger.warning(f"Online neural TTS unavailable ({e}). Initiating offline fallback...")

    # ---------------------------------------------------------
    # 2. FALLBACK: Offline TTS (Windows SAPI5 / pyttsx3)
    # ---------------------------------------------------------
    if platform.system() == "Windows":
        try:
            wav_bytes, fmt, voice_name = _synthesize_offline_sapi5(clean_text)
            logger.info(f"Generated speech using offline Windows SAPI5 voice: {voice_name}")
            return TTSOutput(
                audio_bytes=wav_bytes,
                format=fmt,
                voice_name=voice_name,
                is_online=False
            )
        except Exception as offline_error:
            logger.error(f"Offline speech synthesis also failed: {offline_error}")
            raise RuntimeError(
                f"Text-to-Speech failed. Online error: {online_error}. "
                f"Offline error: {offline_error}"
            )

    # On non-Windows platforms (e.g. Linux Streamlit Cloud), if online fails and offline SAPI5 is not applicable
    raise RuntimeError(
        f"Online neural speech synthesis failed: {online_error}. "
        "Offline fallback is not available on this server platform."
    )
