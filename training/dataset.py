"""
Dataset loading component for the Flickr8k dataset.
"""
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

# Set up logging for dataset operations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Flickr8kDataset(Dataset):
    """
    PyTorch Dataset for loading the Flickr8k images and captions for BLIP fine-tuning.
    """
    def __init__(self, images_dir, captions_file, processor, split_image_ids=None):
        """
        Args:
            images_dir (str): Path to the directory containing Flickr8k images.
            captions_file (str): Path to the annotations file (e.g., captions.txt or Flickr8k.token.txt).
            processor: The Hugging Face BLIP processor used to prepare data for the model.
            split_image_ids (set or list, optional): A collection of image filenames to include in this split 
                                                     (used for separating train/val/test). If None, loads all.
        """
        self.images_dir = images_dir
        self.processor = processor
        self.samples = []
        
        # Check if annotations file exists
        if not os.path.exists(captions_file):
            logger.warning(f"Annotations file not found: {captions_file}. The dataset will be empty.")
            return
            
        # Parse the annotations file
        # Note: We support both the Kaggle CSV format (image,caption) and original txt format (image#0 \t caption)
        with open(captions_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            
            # Skip empty lines and CSV headers
            if not line or line.startswith("image,caption"): 
                continue 
                
            # Parse the delimiter
            if "\t" in line:
                # Original format: 1000268201_693b08cb0e.jpg#0   A child in a pink dress...
                parts = line.split('\t', 1)
                img_name = parts[0].split('#')[0].strip()
                caption = parts[1].strip() if len(parts) > 1 else ""
            elif "," in line:
                # CSV format: 1000268201_693b08cb0e.jpg,A child in a pink dress...
                parts = line.split(',', 1)
                img_name = parts[0].strip()
                caption = parts[1].strip() if len(parts) > 1 else ""
            else:
                continue # Invalid format
                
            # Handle invalid entries gracefully
            if not img_name or not caption:
                continue
                
            # If we are strictly loading a specific split (e.g., validation set), skip images not in the split
            if split_image_ids is not None and img_name not in split_image_ids:
                continue
                
            # Verify the image file actually exists on disk before adding it to our dataset
            img_path = os.path.join(self.images_dir, img_name)
            if os.path.exists(img_path):
                # Since Flickr8k has ~5 captions per image, this loop will naturally add 
                # 5 separate (image_path, caption) dictionaries to our samples list.
                self.samples.append({"image_path": img_path, "caption": caption})
                
        logger.info(f"Successfully loaded {len(self.samples)} valid (image, caption) pairs.")

    def __len__(self):
        """Returns the total number of valid caption samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Retrieves and preprocesses the image and caption at the given index.
        Returns a dictionary of PyTorch tensors ready for the BLIP model.
        """
        sample = self.samples[idx]
        image_path = sample["image_path"]
        caption = sample["caption"]
        
        try:
            # 1. Load the image
            image = Image.open(image_path).convert("RGB")
            
            # 2. Use the BLIP processor to prepare the image and text for the model.
            # - return_tensors="pt": Returns PyTorch tensors natively.
            # - padding="max_length" & truncation=True: Ensures all text tensors are exactly the same size,
            #   which is absolutely required for batching during training.
            encoding = self.processor(
                images=image, 
                text=caption, 
                return_tensors="pt",
                padding="max_length", 
                truncation=True,
                max_length=50 
            )
            
            # 3. The processor adds a batch dimension by default (e.g., shape [1, 3, 224, 224]).
            # We must remove it here using squeeze(0) because the PyTorch DataLoader will automatically 
            # stack these items into its own batches later.
            encoding = {k: v.squeeze(0) for k, v in encoding.items()}
            
            return encoding
            
        except Exception as e:
            # If an image is corrupt or processing fails, log it and gracefully skip to the next item
            # to prevent the entire training loop from crashing midway.
            logger.error(f"Error processing {image_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))
