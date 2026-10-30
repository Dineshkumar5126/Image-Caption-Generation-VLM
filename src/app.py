"""
Streamlit application script.
"""
import streamlit as st
from PIL import Image
import sys
import os

# Ensure the project root directory is in sys.path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import generate_caption
from src.utils import translate_caption, generate_speech, SUPPORTED_LANGUAGES

def main():
    favicon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "favicon.png")
    favicon = Image.open(favicon_path) if os.path.exists(favicon_path) else None

    st.set_page_config(
        page_title="Image Caption Generator",
        page_icon=favicon,
        layout="wide",
        initial_sidebar_state="collapsed"
    )

    # Initialize session state variables for captions, audio, and translations
    if "caption" not in st.session_state:
        st.session_state["caption"] = None
    if "translated_caption" not in st.session_state:
        st.session_state["translated_caption"] = None
    if "translated_lang" not in st.session_state:
        st.session_state["translated_lang"] = None
    if "tts_audio" not in st.session_state:
        st.session_state["tts_audio"] = None
    if "current_image_name" not in st.session_state:
        st.session_state["current_image_name"] = None

    # Custom styling
    st.markdown("""
        <style>
        /* 1. Hide Streamlit Header, Toolbar, Deploy Button, MainMenu, and Decoration */
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        div[data-testid="stToolbar"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        .stDeployButton {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        #MainMenu {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        footer {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        div[data-testid="stDecoration"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        div[data-testid="stStatusWidget"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
        }

        /* 2. Theme & Layout Sizing */
        .stApp {
            background-color: #0b1120;
            color: #f1f5f9;
        }

        .block-container {
            width: 78% !important;
            max-width: 1250px !important;
            min-width: 320px !important;
            padding-top: 2rem !important;
            padding-bottom: 3rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        @media (max-width: 1024px) {
            .block-container {
                width: 90% !important;
                padding-left: 1.5rem !important;
                padding-right: 1.5rem !important;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                width: 95% !important;
                padding-left: 1rem !important;
                padding-right: 1rem !important;
            }
        }

        /* 3. Typography & Header */
        .main-header {
            text-align: center;
            margin-bottom: 2rem;
        }

        .main-header h1 {
            font-size: 2.3rem;
            font-weight: 800;
            color: #f8fafc;
            margin-bottom: 0.4rem;
            letter-spacing: -0.02em;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #94a3b8;
            margin-bottom: 0;
            font-weight: 400;
        }

        /* 4. Section Card Containers */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background-color: #111827;
            border: 1px solid #1f2937;
            border-radius: 12px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.5rem;
        }

        .card-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.35rem;
        }

        .card-desc {
            font-size: 0.92rem;
            color: #94a3b8;
            margin-bottom: 1rem;
        }

        .control-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #cbd5e1;
            margin-bottom: 0.35rem;
        }

        /* 5. Caption & Result Boxes */
        .caption-box {
            background: linear-gradient(135deg, #1e3a8a 0%, #172554 100%);
            color: #ffffff;
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            font-size: 1.15rem;
            font-weight: 500;
            margin-top: 0.5rem;
            margin-bottom: 1rem;
            border-left: 5px solid #3b82f6;
            line-height: 1.6;
        }

        .translated-box {
            background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
            color: #ffffff;
            padding: 1.25rem 1.5rem;
            border-radius: 10px;
            font-size: 1.15rem;
            font-weight: 500;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
            border-left: 5px solid #10b981;
            line-height: 1.6;
        }

        .translated-label {
            font-size: 0.9rem;
            font-weight: 600;
            color: #6ee7b7;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 1rem;
        }

        /* 6. Form Controls & Buttons */
        div[data-testid="stFileUploader"] {
            background-color: #0d1526;
            border: 1px dashed #334155;
            border-radius: 10px;
            padding: 0.75rem;
        }

        button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
            border: 1px solid #3b82f6 !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            border-radius: 8px !important;
            box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.25) !important;
        }

        button[kind="primary"]:hover {
            background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
            border-color: #60a5fa !important;
        }

        button[kind="secondary"] {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            color: #f1f5f9 !important;
            font-weight: 500 !important;
            border-radius: 8px !important;
        }

        button[kind="secondary"]:hover {
            background-color: #334155 !important;
            border-color: #475569 !important;
            color: #ffffff !important;
        }

        div[data-testid="stRadio"] > div {
            gap: 1.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # A. HEADER
    st.markdown("""
        <div class="main-header">
            <h1>Image Caption Generator</h1>
            <div class="subtitle">Generate meaningful descriptions from images using Vision-Language Models.</div>
        </div>
    """, unsafe_allow_html=True)

    # B. IMAGE UPLOAD SECTION
    with st.container(border=True):
        st.markdown("""
            <div class="card-title">Image Upload</div>
            <div class="card-desc">Upload a standard image file (JPG, JPEG, or PNG) to generate a descriptive caption.</div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )

        image = None
        if uploaded_file is not None:
            try:
                # Reset state if a different image is uploaded
                if st.session_state.get("current_image_name") != uploaded_file.name:
                    st.session_state["current_image_name"] = uploaded_file.name
                    st.session_state["caption"] = None
                    st.session_state["translated_caption"] = None
                    st.session_state["translated_lang"] = None
                    st.session_state["tts_audio"] = None

                image = Image.open(uploaded_file).convert("RGB")

                # Display preview with balanced proportions
                col_img_space1, col_img_center, col_img_space2 = st.columns([1, 4, 1])
                with col_img_center:
                    st.image(image, use_container_width=True)

            except Exception as e:
                st.error(f"An error occurred while processing the image: {e}")

    # C. CAPTION GENERATION SECTION (active when image is uploaded)
    if image is not None:
        with st.container(border=True):
            st.markdown("""
                <div class="card-title">Caption Generation</div>
                <div class="card-desc">Select the desired description detail level and generate the caption.</div>
            """, unsafe_allow_html=True)

            col_mode, col_btn = st.columns([3, 1], vertical_alignment="bottom")
            with col_mode:
                st.markdown('<div class="control-label">Detail Level</div>', unsafe_allow_html=True)
                mode = st.radio(
                    "Detail Level",
                    ("Short", "Medium", "Detailed"),
                    index=1,
                    horizontal=True,
                    label_visibility="collapsed"
                )

            with col_btn:
                generate_btn = st.button("Generate Caption", type="primary", use_container_width=True)

            if generate_btn:
                with st.spinner("Loading BLIP model..."):
                    from src.model import get_model_and_processor
                    get_model_and_processor()

                with st.spinner(f"Generating {mode.lower()} caption..."):
                    caption = generate_caption(image, mode=mode)

                    if caption.startswith("Error"):
                        st.error(caption)
                        st.session_state["caption"] = None
                    else:
                        st.session_state["caption"] = caption.capitalize()
                        st.session_state["translated_caption"] = None
                        st.session_state["translated_lang"] = None
                        st.session_state["tts_audio"] = None

        # D. GENERATED CAPTION SECTION
        if st.session_state.get("caption"):
            with st.container(border=True):
                st.markdown("""
                    <div class="card-title">Generated Caption</div>
                """, unsafe_allow_html=True)

                st.markdown(f'<div class="caption-box">{st.session_state["caption"]}</div>', unsafe_allow_html=True)

                # Offline English TTS controls
                col_tts_btn, col_tts_player = st.columns([1, 3], vertical_alignment="center")
                with col_tts_btn:
                    tts_btn = st.button("Listen (English)", use_container_width=True)

                if tts_btn:
                    with st.spinner("Synthesizing speech..."):
                        try:
                            audio_bytes = generate_speech(st.session_state["caption"])
                            st.session_state["tts_audio"] = audio_bytes
                        except Exception as e:
                            st.error(f"Text-to-Speech Error: {e}")

                with col_tts_player:
                    if st.session_state.get("tts_audio"):
                        st.audio(st.session_state["tts_audio"], format="audio/wav")

            # E. MULTILINGUAL TRANSLATION SECTION
            with st.container(border=True):
                st.markdown("""
                    <div class="card-title">Multilingual Translation</div>
                    <div class="card-desc">Translate the English caption into Telugu, Hindi, Tamil, Kannada, or Malayalam.</div>
                """, unsafe_allow_html=True)

                col_lang, col_tr_btn = st.columns([3, 1], vertical_alignment="bottom")
                with col_lang:
                    st.markdown('<div class="control-label">Target Language</div>', unsafe_allow_html=True)
                    selected_lang = st.selectbox(
                        "Select Language",
                        options=list(SUPPORTED_LANGUAGES.keys()),
                        key="selected_lang_dropdown",
                        label_visibility="collapsed"
                    )

                with col_tr_btn:
                    translate_btn = st.button("Translate", use_container_width=True)

                if translate_btn:
                    with st.spinner(f"Translating to {selected_lang}..."):
                        try:
                            translated_text = translate_caption(st.session_state["caption"], selected_lang)
                            st.session_state["translated_caption"] = translated_text
                            st.session_state["translated_lang"] = selected_lang
                        except Exception as e:
                            st.error(f"Translation Error: {e}")

                if st.session_state.get("translated_caption"):
                    st.markdown(f'<div class="translated-label">Translated Caption ({st.session_state["translated_lang"]}):</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="translated-box">{st.session_state["translated_caption"]}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
