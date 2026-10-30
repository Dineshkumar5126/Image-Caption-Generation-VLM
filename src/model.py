"""
Model loading and inference script.
"""
import os
import zipfile
import torch
import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Hugging Face Repository for cloud deployment and fallback
HF_MODEL_ID = "Dinesh516/blip-image-captioning-flickr8k"

@st.cache_resource(show_spinner=False)
def get_model_and_processor():
    """
    Loads the BLIP model and processor:
    1. If the local 'models/trained_model.zip' exists, extracts it (if not already extracted)
       and loads the local fine-tuned model weights.
    2. If the local ZIP is unavailable (e.g. deployed on Streamlit Community Cloud),
       downloads and loads the fine-tuned model and processor directly from the Hugging Face
       Model Hub repository ('Dinesh516/blip-image-captioning-flickr8k').
       
    Cached via Streamlit to persist across reruns without reloading.
    Returns the processor, model, and device.
    """
    try:
        logger.info("Initializing BLIP model and processor...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using compute device: {device}")
        
        # Check for local fine-tuned model zip
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        zip_path = os.path.join(base_dir, "models", "trained_model.zip")
        extracted_dir = os.path.join(base_dir, "models", "trained_model_extracted")
        
        if os.path.exists(zip_path):
            logger.info(f"Local model archive found at: {zip_path}")
            if not os.path.exists(extracted_dir):
                logger.info(f"Extracting {zip_path} to {extracted_dir}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extracted_dir)
            
            # Google Colab zips usually preserve the full path (e.g. /content/models/...)
            nested_dir = os.path.join(extracted_dir, "content", "models", "blip_flickr8k_500")
            if os.path.exists(nested_dir):
                model_source = nested_dir
            else:
                model_source = extracted_dir
                
            logger.info(f"Loading local fine-tuned model from: {model_source}")
            processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            model = BlipForConditionalGeneration.from_pretrained(model_source)
        else:
            # Cloud Fallback: Load fine-tuned model and processor from Hugging Face repository
            logger.info(
                f"Local trained_model.zip not found. Loading fine-tuned model and processor "
                f"from Hugging Face repository: {HF_MODEL_ID}"
            )
            try:
                processor = BlipProcessor.from_pretrained(HF_MODEL_ID)
            except Exception as proc_err:
                logger.error(f"Failed to load processor from {HF_MODEL_ID}: {proc_err}")
                raise RuntimeError(
                    f"Failed to load processor from Hugging Face repository '{HF_MODEL_ID}': {proc_err}"
                )
                
            try:
                model = BlipForConditionalGeneration.from_pretrained(HF_MODEL_ID)
            except Exception as model_err:
                logger.error(f"Failed to load model weights from {HF_MODEL_ID}: {model_err}")
                raise RuntimeError(
                    f"Failed to load fine-tuned model from Hugging Face repository '{HF_MODEL_ID}': {model_err}"
                )
        
        model.to(device)
        logger.info("Fine-tuned BLIP model and processor loaded successfully.")
        return processor, model, device
        
    except Exception as e:
        logger.error(f"Failed to load the model: {e}")
        raise RuntimeError(f"Model loading failed: {e}")

def cleanup_caption(caption: str) -> str:
    """
    Carefully removes common irrelevant stock-photo or watermark phrases 
    sometimes hallucinated by the model without deleting meaningful details.
    """
    irrelevant_phrases = [
        "stock photo", "stock image", "watermark", "getty images", 
        "alamy", "shutterstock", "royalty free", "image id", "image code"
    ]
    
    caption_lower = caption.lower()
    for phrase in irrelevant_phrases:
        if phrase in caption_lower:
            # Replace the exact matches case-insensitively using basic string manipulation
            caption = caption.replace(phrase, "")
            caption = caption.replace(phrase.capitalize(), "")
            caption = caption.replace(phrase.title(), "")
            
    # Remove any extra whitespace left behind
    return " ".join(caption.split())

def generate_caption(image: Image.Image, mode: str = "Medium") -> str:
    """
    Generates a caption for the given image based on the selected mode.
    
    Args:
        image (PIL.Image.Image): The input image.
        mode (str): Caption mode ('Short', 'Medium', 'Detailed').
        
    Returns:
        str: The generated caption or an error message.
    """
    # Basic error handling for invalid input
    if image is None or not isinstance(image, Image.Image):
        return "Error: Invalid image input. Please provide a valid PIL Image."
        
    try:
        processor, model, device = get_model_and_processor()
        
        # Define generation parameters based on the mode.
        mode = mode.capitalize()
        
        if mode == "Short":
            # Concise: Low beams, negative length penalty to encourage stopping early
            gen_kwargs = {
                "max_new_tokens": 12, 
                "min_new_tokens": 3,
                "num_beams": 3,
                "length_penalty": 0.5 
            }
        elif mode == "Detailed":
            # Descriptive: High beams and length penalty to force the model to explore longer, 
            # descriptive paths. Repetition penalty prevents the model from looping.
            gen_kwargs = {
                "max_new_tokens": 75, 
                "min_new_tokens": 18, 
                "num_beams": 7,           
                "length_penalty": 2.5,    
                "repetition_penalty": 1.2 
            }
        else:  # Default to Medium
            # Balanced settings for a standard sentence
            gen_kwargs = {
                "max_new_tokens": 30, 
                "min_new_tokens": 10,
                "num_beams": 5,
                "length_penalty": 1.0
            }
            
        # Preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)
            
        # Generate the caption tokens
        output_tokens = model.generate(**inputs, **gen_kwargs)
        
        # Decode tokens to text
        caption = processor.decode(output_tokens[0], skip_special_tokens=True)
        
        # Run cleanup to strip away stock photo artifacts
        caption = cleanup_caption(caption)
        
        return caption.strip()
        
    except Exception as e:
        error_msg = f"Error during caption generation: {str(e)}"
        logger.error(error_msg)
        return error_msg
