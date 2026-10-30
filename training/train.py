"""
Training script for fine-tuning the pretrained BLIP model on the Flickr8k dataset.
Designed to be simple, readable, and ready for execution on a GPU environment (like Google Colab).
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

# Import our custom dataset component
from dataset import Flickr8kDataset

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_split_ids(split_filepath):
    """Utility function to load image filenames for a specific split (train/val/test)."""
    if not os.path.exists(split_filepath):
        return None
    with open(split_filepath, 'r', encoding='utf-8') as f:
        # Return a set for O(1) fast lookups
        return set([line.strip() for line in f.readlines() if line.strip()])

def train_model(
    images_dir,
    captions_file,
    splits_dir,
    output_dir,
    batch_size=4,
    learning_rate=5e-5,
    num_epochs=3
):
    # 1. Device configuration (Crucial for Colab compatibility)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using hardware device: {device.type.upper()}")
    
    # 2. Check for dataset existence before starting anything
    if not os.path.exists(images_dir) or not os.path.exists(captions_file):
        logger.error("Dataset files missing. Please ensure Flickr8k is downloaded and extracted.")
        return
        
    train_split_path = os.path.join(splits_dir, "train_images.txt")
    val_split_path = os.path.join(splits_dir, "val_images.txt")
    
    if not os.path.exists(train_split_path) or not os.path.exists(val_split_path):
        logger.error(f"Split files not found in {splits_dir}. Run prepare_dataset.py first.")
        return

    train_ids = load_split_ids(train_split_path)
    val_ids = load_split_ids(val_split_path)
    logger.info(f"Loaded {len(train_ids)} training images and {len(val_ids)} validation images.")

    # 3. Load Pretrained Model & Processor
    model_name = "Salesforce/blip-image-captioning-base"
    logger.info(f"Loading pretrained model: {model_name}...")
    try:
        processor = BlipProcessor.from_pretrained(model_name)
        model = BlipForConditionalGeneration.from_pretrained(model_name)
        model.to(device)
    except Exception as e:
        logger.error(f"Failed to load Hugging Face model/processor: {e}")
        return

    # 4. Prepare Datasets & DataLoaders
    train_dataset = Flickr8kDataset(images_dir, captions_file, processor, split_image_ids=train_ids)
    val_dataset = Flickr8kDataset(images_dir, captions_file, processor, split_image_ids=val_ids)

    if len(train_dataset) == 0:
        logger.error("Training dataset is empty. Cannot start training.")
        return

    # We shuffle training data to prevent the model from learning the order of images
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 5. Optimizer
    # AdamW is the standard modern optimizer for transformer models
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    os.makedirs(output_dir, exist_ok=True)
    
    # 6. The Training Loop
    logger.info("Starting training loop...")
    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch + 1} / {num_epochs} ---")
        
        # --- TRAINING PHASE ---
        model.train() # Set model to training mode
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to GPU (if available)
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            # In causal language modeling, the labels are just the input_ids.
            # However, we MUST replace the padding tokens with -100 so the
            # PyTorch CrossEntropyLoss function ignores them during loss calculation.
            labels = input_ids.clone()
            labels[labels == processor.tokenizer.pad_token_id] = -100

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad() # Clear gradients for the next step
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- VALIDATION PHASE ---
        model.eval() # Set model to evaluation mode (disables dropout, etc.)
        total_val_loss = 0.0
        
        with torch.no_grad(): # Don't calculate gradients during validation (saves memory)
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                labels = input_ids.clone()
                labels[labels == processor.tokenizer.pad_token_id] = -100

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_val_loss += outputs.loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        logger.info(f"Epoch {epoch + 1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 7. Save Checkpoint
        # We save uniquely named checkpoints per epoch alongside the processor
        epoch_checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch + 1}")
        os.makedirs(epoch_checkpoint_dir, exist_ok=True)
        model.save_pretrained(epoch_checkpoint_dir)
        processor.save_pretrained(epoch_checkpoint_dir)
        logger.info(f"Saved checkpoint to {epoch_checkpoint_dir}")

    # 8. Save Final Model
    final_output_dir = os.path.join(output_dir, "final_finetuned_model")
    os.makedirs(final_output_dir, exist_ok=True)
    model.save_pretrained(final_output_dir)
    processor.save_pretrained(final_output_dir)
    logger.info(f"Training Complete! Final model saved to {final_output_dir}")


if __name__ == "__main__":
    # Ensure this script is resilient to execution location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Configuration parameters
    IMAGES_DIR = os.path.join(base_dir, "data", "Images")
    CAPTIONS_FILE = os.path.join(base_dir, "data", "captions.txt")
    SPLITS_DIR = os.path.join(base_dir, "data", "splits")
    
    # We will save our NEW fine-tuned weights here, perfectly preserving the original Hugging Face cache
    OUTPUT_MODEL_DIR = os.path.join(base_dir, "models", "blip-flickr8k-finetuned")
    
    # Launch training
    train_model(
        images_dir=IMAGES_DIR,
        captions_file=CAPTIONS_FILE,
        splits_dir=SPLITS_DIR,
        output_dir=OUTPUT_MODEL_DIR,
        batch_size=4,          # Lower batch size since BLIP is large; adjustable based on GPU VRAM
        learning_rate=5e-5,    # Standard LR for Transformer fine-tuning
        num_epochs=3           # 3 Epochs is a solid baseline to prevent overfitting
    )
