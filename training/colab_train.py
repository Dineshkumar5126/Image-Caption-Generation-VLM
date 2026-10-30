"""
Google Colab tailored training script for the Image Caption Generation project.
This script ensures GPU usage, avoids checkpoint bloat, and runs evaluation right after training.
"""
import os
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForConditionalGeneration

# Import existing dataset and evaluation logic
from dataset import Flickr8kDataset
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def load_split_ids(split_filepath):
    """Utility function to load image filenames for a specific split."""
    if not os.path.exists(split_filepath):
        return None
    with open(split_filepath, 'r', encoding='utf-8') as f:
        return set([line.strip() for line in f.readlines() if line.strip()])

def main():
    # 1. Path Configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    images_dir = os.path.join(base_dir, "data", "Images")
    captions_file = os.path.join(base_dir, "data", "captions.txt")
    splits_dir = os.path.join(base_dir, "data", "splits")
    output_model_dir = os.path.join(base_dir, "models", "blip_flickr8k_500")
    
    # 2. Hyperparameters
    batch_size = 4
    learning_rate = 5e-5
    num_epochs = 3
    
    # 3. Hardware check
    logger.info("Checking hardware...")
    if not torch.cuda.is_available():
        logger.error("CUDA is NOT available! For Colab, please ensure you are connected to a GPU runtime.")
        logger.error("Go to Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU")
        return
        
    device = torch.device("cuda")
    logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
    
    # 4. Load Data Splits
    train_split_path = os.path.join(splits_dir, "train_images.txt")
    val_split_path = os.path.join(splits_dir, "val_images.txt")
    
    train_ids = load_split_ids(train_split_path)
    val_ids = load_split_ids(val_split_path)
    
    logger.info(f"Loaded {len(train_ids) if train_ids else 0} training images.")
    logger.info(f"Loaded {len(val_ids) if val_ids else 0} validation images.")
    
    # 5. Load Model
    model_name = "Salesforce/blip-image-captioning-base"
    logger.info(f"Loading pretrained model: {model_name}...")
    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForConditionalGeneration.from_pretrained(model_name)
    model.to(device)
    
    # 6. Initialize Datasets & DataLoaders
    # Added num_workers=2 and pin_memory=True for faster batch fetching on Colab
    train_dataset = Flickr8kDataset(images_dir, captions_file, processor, split_image_ids=train_ids)
    val_dataset = Flickr8kDataset(images_dir, captions_file, processor, split_image_ids=val_ids)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    
    # 7. Optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # 8. Training Loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        logger.info(f"--- Epoch {epoch + 1} / {num_epochs} ---")
        model.train()
        total_train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
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
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
                
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation Phase ---
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
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
                
        avg_val_loss = total_val_loss / len(val_loader)
        logger.info(f"Epoch {epoch + 1} Summary -> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 9. Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            logger.info(f"New best validation loss ({best_val_loss:.4f})! Saving model...")
            os.makedirs(output_model_dir, exist_ok=True)
            model.save_pretrained(output_model_dir)
            processor.save_pretrained(output_model_dir)

    logger.info("Training complete! Model is saved to " + output_model_dir)
    
    # 10. Evaluation Phase (using existing logic)
    logger.info("\n=== Starting Evaluation on Test Set ===")
    results_file = os.path.join(output_model_dir, "evaluation_results.json")
    
    evaluate_model(
        images_dir=images_dir,
        captions_file=captions_file,
        splits_dir=splits_dir,
        model_dir=output_model_dir,
        output_results_file=results_file,
        num_examples=5
    )

if __name__ == "__main__":
    main()
