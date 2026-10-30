import os
import sys
import urllib.request
from PIL import Image

# Add the project root to sys.path to allow importing from src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.model import generate_caption

def main():
    print("Testing BLIP model implementation...")
    
    # URL of a sample image (using a public demo image from Salesforce)
    image_url = "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    image_path = "sample_test_image.jpg"
    
    if not os.path.exists(image_path):
        print(f"Downloading sample image from {image_url}...")
        try:
            urllib.request.urlretrieve(image_url, image_path)
        except Exception as e:
            print(f"Error downloading image: {e}")
            sys.exit(1)
            
    print("Loading image...")
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image: {e}")
        sys.exit(1)
    
    print("\n--- Generating Captions ---")
    modes = ["Short", "Medium", "Detailed"]
    
    for mode in modes:
        print(f"\nMode: {mode}")
        caption = generate_caption(image, mode=mode)
        print(f"Caption: {caption}")

if __name__ == "__main__":
    main()
