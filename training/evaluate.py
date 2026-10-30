"""
Evaluation script for the fine-tuned BLIP model on the Flickr8k dataset.
This script calculates captioning-specific metrics (BLEU) against the test split.
"""
import os
import json
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Attempt to import NLTK for BLEU scoring
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    logger.warning("NLTK library not found. BLEU score calculation will be skipped.")
    logger.warning("To enable BLEU scores, install NLTK: pip install nltk")

def evaluate_model(images_dir, captions_file, splits_dir, model_dir, output_results_file, num_examples=5):
    """
    Evaluates the fine-tuned model against the test split.
    """
    # 1. Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using hardware device: {device.type.upper()}")
    
    # 2. Verify all required files and directories exist
    test_split_path = os.path.join(splits_dir, "test_images.txt")
    if not os.path.exists(test_split_path):
        logger.error(f"Test split file not found at {test_split_path}. Please run prepare_dataset.py first.")
        return
        
    if not os.path.exists(model_dir):
        logger.error(f"Fine-tuned model directory not found at {model_dir}.")
        logger.error("Please run train.py to fine-tune the model before evaluating.")
        return
        
    if not os.path.exists(captions_file):
        logger.error(f"Captions file not found at {captions_file}.")
        return

    # 3. Load Test Image Filenames
    with open(test_split_path, 'r', encoding='utf-8') as f:
        test_images = [line.strip() for line in f.readlines() if line.strip()]
        
    if len(test_images) == 0:
        logger.error("Test split is empty.")
        return
        
    logger.info(f"Loaded {len(test_images)} images from the test split.")

    # 4. Map test images to their reference captions
    # BLEU score requires comparing 1 generated caption against ALL human references (typically 5 per image)
    references = {img: [] for img in test_images}
    
    with open(captions_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("image,caption"): 
                continue
                
            # Parse delimiter (supports both TXT and CSV formats)
            if "\t" in line:
                parts = line.split('\t', 1)
                img_name = parts[0].split('#')[0].strip()
                caption = parts[1].strip() if len(parts) > 1 else ""
            elif "," in line:
                parts = line.split(',', 1)
                img_name = parts[0].strip()
                caption = parts[1].strip() if len(parts) > 1 else ""
            else:
                continue
                
            # Only save references for images that are strictly in our test set
            if img_name in references and caption:
                references[img_name].append(caption)

    # 5. Load Fine-Tuned Model & Processor
    logger.info(f"Loading fine-tuned model from {model_dir}...")
    try:
        base_model_id = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(base_model_id)
        model = BlipForConditionalGeneration.from_pretrained(model_dir)
        model.to(device)
        model.eval() # Set to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load fine-tuned model: {e}")
        return

    # 6. Evaluation Loop
    logger.info("Starting evaluation on test set. This may take a while...")
    
    results = []
    total_bleu1 = 0.0
    total_bleu4 = 0.0
    evaluated_count = 0
    
    # Use NLTK smoothing to prevent BLEU scores from bottoming out to 0 on short sentences
    smoother = SmoothingFunction().method1 if NLTK_AVAILABLE else None

    # Evaluate a subset or the full test set
    # For a college project, evaluating all 1,000+ test images can take time on a CPU,
    # but we will loop through all valid ones.
    for img_name in test_images:
        img_path = os.path.join(images_dir, img_name)
        
        if not os.path.exists(img_path):
            continue
            
        ref_captions = references.get(img_name, [])
        if not ref_captions:
            continue
            
        try:
            image = Image.open(img_path).convert("RGB")
            
            # Preprocess and generate caption
            inputs = processor(images=image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                # We use beam search for higher quality validation generation
                output_ids = model.generate(**inputs, max_new_tokens=30, num_beams=5)
                
            generated_caption = processor.decode(output_ids[0], skip_special_tokens=True).strip()
            
            # Note: Caption evaluation uses NLP metrics (BLEU, ROUGE, METEOR).
            # It does NOT use "Classification Accuracy" (which is for predicting discrete categories like dog/cat).
            bleu1, bleu4 = 0.0, 0.0
            if NLTK_AVAILABLE:
                # NLTK BLEU expects lists of tokens
                ref_tokens = [ref.lower().split() for ref in ref_captions]
                gen_tokens = generated_caption.lower().split()
                
                # BLEU-1: Unigram precision (did it get individual words right?)
                bleu1 = sentence_bleu(ref_tokens, gen_tokens, weights=(1, 0, 0, 0), smoothing_function=smoother)
                # BLEU-4: 4-gram precision (did it get full phrases right?)
                bleu4 = sentence_bleu(ref_tokens, gen_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)
                
                total_bleu1 += bleu1
                total_bleu4 += bleu4

            results.append({
                "image": img_name,
                "generated": generated_caption,
                "references": ref_captions,
                "bleu_1": round(bleu1, 4) if NLTK_AVAILABLE else None,
                "bleu_4": round(bleu4, 4) if NLTK_AVAILABLE else None
            })
            
            evaluated_count += 1
            
            # Print progress for every 100 images
            if evaluated_count % 100 == 0:
                logger.info(f"Evaluated {evaluated_count} images...")
                
        except Exception as e:
            logger.error(f"Error evaluating image {img_name}: {e}")

    # 7. Aggregate and Save Results
    if evaluated_count == 0:
        logger.error("No images were successfully evaluated.")
        return
        
    avg_bleu1 = total_bleu1 / evaluated_count if NLTK_AVAILABLE else None
    avg_bleu4 = total_bleu4 / evaluated_count if NLTK_AVAILABLE else None

    # Print a few visual examples to the console
    logger.info("\n=== EVALUATION EXAMPLES ===")
    for i in range(min(num_examples, len(results))):
        ex = results[i]
        logger.info(f"Image: {ex['image']}")
        logger.info(f"Generated: {ex['generated']}")
        logger.info(f"Reference 1: {ex['references'][0]}")
        if NLTK_AVAILABLE:
            logger.info(f"BLEU-4 Score: {ex['bleu_4']:.4f}")
        logger.info("-" * 40)

    # Print Final NLP Metrics explicitly distinguishing them from Classification Accuracy
    logger.info("\n=== FINAL TEST SET METRICS ===")
    logger.info("Note: Captioning tasks evaluate linguistic overlap, NOT classification accuracy.")
    if NLTK_AVAILABLE:
        logger.info(f"Average BLEU-1 (Word Match):   {avg_bleu1:.4f}")
        logger.info(f"Average BLEU-4 (Phrase Match): {avg_bleu4:.4f}")
    else:
        logger.info("BLEU metrics unavailable (NLTK not installed).")

    # Save to a readable JSON file
    os.makedirs(os.path.dirname(output_results_file), exist_ok=True)
    
    report = {
        "metrics": {
            "average_bleu_1": avg_bleu1,
            "average_bleu_4": avg_bleu4,
            "total_evaluated": evaluated_count
        },
        "examples": results
    }
    
    with open(output_results_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=4)
        
    logger.info(f"\nDetailed evaluation report saved to: {output_results_file}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    IMAGES_DIR = os.path.join(base_dir, "data", "Images")
    CAPTIONS_FILE = os.path.join(base_dir, "data", "captions.txt")
    SPLITS_DIR = os.path.join(base_dir, "data", "splits")
    
    # Dynamically find the fine-tuned model directory
    extracted_dir = os.path.join(base_dir, "models", "trained_model_extracted")
    nested_dir = os.path.join(extracted_dir, "content", "models", "blip_flickr8k_500")
    
    if os.path.exists(nested_dir):
        MODEL_DIR = nested_dir
    elif os.path.exists(extracted_dir):
        MODEL_DIR = extracted_dir
    else:
        MODEL_DIR = "Salesforce/blip-image-captioning-base"
    
    # Save results to a dedicated evaluation folder
    RESULTS_FILE = os.path.join(base_dir, "models", "evaluation_results.json")
    
    evaluate_model(
        images_dir=IMAGES_DIR,
        captions_file=CAPTIONS_FILE,
        splits_dir=SPLITS_DIR,
        model_dir=MODEL_DIR,
        output_results_file=RESULTS_FILE,
        num_examples=5
    )
