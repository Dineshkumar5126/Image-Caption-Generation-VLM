"""
Dataset preparation script for the Flickr8k dataset.
This script reads the raw dataset, verifies its integrity, and splits it into 
reproducible Training, Validation, and Testing sets by IMAGE (not by caption).
"""
import os
import random
import logging

# Configure simple, readable logging for the terminal
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def prepare_dataset(images_dir, captions_file, output_splits_dir, train_pct=0.8, val_pct=0.1, random_seed=42):
    """
    Reads the dataset annotations, verifies that referenced images exist, 
    splits the dataset by image, and saves the filenames for each split into text files.
    """
    # 2. Verify that the necessary directories and files exist
    if not os.path.exists(images_dir):
        logger.error(f"Image directory not found: {images_dir}")
        return
    if not os.path.exists(captions_file):
        logger.error(f"Captions file not found: {captions_file}")
        return

    # Create the output directory for splits if it doesn't exist
    os.makedirs(output_splits_dir, exist_ok=True)
    
    # Dictionaries and tracking variables
    # We map img_name -> list of captions to ensure all captions for one image stay together
    image_to_captions = {}
    missing_images = set()
    invalid_annotations = 0

    logger.info("Reading and validating annotations...")
    
    # 1. Read using the same flexible format supported by our dataset.py
    with open(captions_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("image,caption"): 
            continue # Skip empties and headers
            
        # Parse delimiter natively (handles both TXT and CSV formats)
        if "\t" in line:
            parts = line.split('\t', 1)
            img_name = parts[0].split('#')[0].strip()
            caption = parts[1].strip() if len(parts) > 1 else ""
        elif "," in line:
            parts = line.split(',', 1)
            img_name = parts[0].strip()
            caption = parts[1].strip() if len(parts) > 1 else ""
        else:
            invalid_annotations += 1
            continue
            
        if not img_name or not caption:
            invalid_annotations += 1
            continue

        # 3. Check that the image actually exists on the hard drive
        img_path = os.path.join(images_dir, img_name)
        if not os.path.exists(img_path):
            missing_images.add(img_name)
            continue
            
        if img_name not in image_to_captions:
            image_to_captions[img_name] = []
        image_to_captions[img_name].append(caption)

    # 9. Report data integrity issues clearly
    if invalid_annotations > 0:
        logger.warning(f"Skipped {invalid_annotations} invalid annotation lines due to formatting.")
    if len(missing_images) > 0:
        logger.warning(f"Found {len(missing_images)} images referenced in annotations but missing from the image folder.")

    valid_images = list(image_to_captions.keys())
    total_images = len(valid_images)
    
    if total_images == 0:
        logger.error("No valid images found to split. Please check your dataset paths and ensure Flickr8k is downloaded.")
        return

    logger.info(f"Total valid images successfully paired: {total_images}")

    # 4, 5 & 6. Create reproducible splits strictly by IMAGE using a fixed random seed
    random.seed(random_seed)
    random.shuffle(valid_images)

    train_end = int(total_images * train_pct)
    val_end = train_end + int(total_images * val_pct)

    train_images = valid_images[:train_end]
    val_images = valid_images[train_end:val_end]
    test_images = valid_images[val_end:]

    # 7. Save the split information (the filenames) to the output directory
    splits = {
        "train_images.txt": train_images,
        "val_images.txt": val_images,
        "test_images.txt": test_images
    }

    for filename, image_list in splits.items():
        filepath = os.path.join(output_splits_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            for img in image_list:
                f.write(f"{img}\n")

    # 8. Print useful statistics
    logger.info("\n--- Dataset Split Statistics ---")
    for name, img_list in splits.items():
        # Calculate how many total captions exist in this split
        total_caps = sum(len(image_to_captions[img]) for img in img_list)
        logger.info(f"{name}: {len(img_list)} images, {total_caps} total captions")
        
    logger.info(f"\nPreparation complete! Split files saved to: {output_splits_dir}")

if __name__ == "__main__":
    # Resolve the absolute path to the project root, no matter where the script is executed from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # We assume the user will place the Flickr8k dataset inside the 'data' folder
    prepare_dataset(
        images_dir=os.path.join(base_dir, "data", "Images"),
        captions_file=os.path.join(base_dir, "data", "captions.txt"),
        output_splits_dir=os.path.join(base_dir, "data", "splits"),
        train_pct=0.8,  # 80% Training
        val_pct=0.1,    # 10% Validation (Leaving 10% for Testing)
        random_seed=42  # Fixed seed ensures the exact same split if run multiple times
    )
