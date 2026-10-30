"""
Script to download a small subset of the Flickr8k dataset for testing.
Uses Hugging Face 'datasets' in streaming mode to prevent downloading the full archive.
"""
import os
import logging
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def download_subset():
    # 1. Setup paths matching our existing project structure
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, "data", "Images")
    captions_file = os.path.join(base_dir, "data", "captions.txt")
    
    os.makedirs(images_dir, exist_ok=True)
    
    # 2. Check for the Hugging Face datasets library
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("The 'datasets' library is required to run this script.")
        logger.error("Please install it using: pip install datasets")
        return

    # We need a subset of 500 unique images for Review 2 demonstration
    target_image_count = 500
    downloaded_images = set()
    captions_data = []
    
    logger.info(f"Connecting to Hugging Face dataset 'intro/flickr8k' in streaming mode...")
    logger.info(f"Targeting {target_image_count} unique images...")
    
    try:
        # 3. Use streaming=True so it downloads lazily (only what we iterate over)
        # This prevents the 1GB zip file from being downloaded to disk.
        dataset = load_dataset("intro/flickr8k", split="train", streaming=True)
    except Exception as e:
        logger.error(f"Failed to connect to the dataset stream. Network error or dataset unavailable: {e}")
        return
        
    try:
        for item in dataset:
            # HF datasets can have varying column names based on the repo.
            # Typical image column is 'image', typical text is 'caption' or 'text'.
            img = item.get("image")
            
            # Extract filename or fallback to generating a clean one
            file_name = item.get("file_name") or item.get("filename") or item.get("image_id")
            if not file_name:
                file_name = f"image_{len(downloaded_images)}.jpg"
            else:
                file_name = str(file_name)
                if not file_name.endswith(".jpg"):
                    file_name = f"{file_name}.jpg"
            
            # Extract caption(s)
            caption = item.get("caption") or item.get("text") or item.get("captions")
            if not caption and "caption_0" in item:
                caption = [item.get(f"caption_{i}") for i in range(5) if item.get(f"caption_{i}")]
            
            if img is None or not caption:
                continue
                
            # If we haven't hit our 140 unique image quota yet, save the new image
            if file_name not in downloaded_images:
                if len(downloaded_images) >= target_image_count:
                    # We have reached our goal of unique images. 
                    # We can safely stop streaming.
                    break
                    
                # 4. Save the image locally
                img_path = os.path.join(images_dir, file_name)
                try:
                    # Ensure image is in RGB format before saving to avoid issues with Grayscale/RGBA
                    img.convert("RGB").save(img_path)
                    downloaded_images.add(file_name)
                except Exception as e:
                    logger.error(f"Failed to save image {file_name} locally: {e}")
                    continue
            
            # 5. Format the annotations to be completely compatible with dataset.py / prepare_dataset.py
            # Our parser supports the tab-delimited format natively: image_name \t caption
            if isinstance(caption, list):
                # If the dataset provides all 5 captions as a list in one row
                for cap in caption:
                    clean_cap = str(cap).strip().replace("\n", " ").replace("\t", " ")
                    captions_data.append(f"{file_name}\t{clean_cap}")
            else:
                # If the dataset provides 1 caption per row
                clean_cap = str(caption).strip().replace("\n", " ").replace("\t", " ")
                captions_data.append(f"{file_name}\t{clean_cap}")
                
    except Exception as e:
        logger.error(f"Stream interrupted or parsing failed: {e}")
        
    if len(downloaded_images) == 0:
        logger.error("No images were successfully downloaded. Please check your network connection.")
        return
        
    # 6. Save annotations strictly into the data/ directory
    try:
        with open(captions_file, 'w', encoding='utf-8') as f:
            for line in captions_data:
                f.write(f"{line}\n")
    except Exception as e:
        logger.error(f"Failed to write captions to {captions_file}: {e}")
        return
        
    logger.info("\n--- Download Summary ---")
    logger.info(f"Successfully downloaded {len(downloaded_images)} unique images.")
    logger.info(f"Successfully saved {len(captions_data)} captions.")
    logger.info(f"Images directory: {images_dir}")
    logger.info(f"Captions file:    {captions_file}")
    
    logger.info("\nNext Step: Run 'python training/prepare_dataset.py' to distribute these 500 images into train/val/test splits.")

if __name__ == "__main__":
    download_subset()
