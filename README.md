# Image Caption Generation using Vision-Language Models

## Project Objective
The objective of this college deep learning project is to build and fine-tune a Vision-Language Model (VLM) capable of generating contextually meaningful captions for given images.

## Problem Statement
Automatically describing the visual content of an image is a fundamental challenge in artificial intelligence that connects computer vision and natural language processing. This project tackles the challenge by leveraging a modern, pretrained Vision-Language Model to translate pixel data into human-readable text.

## Technologies Used
- **Python 3**
- **PyTorch**: Deep learning framework for training and inference.
- **Hugging Face Transformers**: For the BLIP model and processor architecture.
- **Streamlit**: For building the interactive front-end web application.

## VLM / BLIP Approach
This project utilizes the **BLIP (Bootstrapping Language-Image Pre-training)** architecture by Salesforce. 
The model pipeline consists of:
1. **Vision Encoder:** Extracts visual features from the input image.
2. **Language Decoder:** Uses the visual features as context to auto-regressively generate a text sequence (the caption).

## Dataset Used
The project uses a subset of the **Flickr8k Dataset**:
- **Total Images:** 500 images (Sub-sampled for rapid prototyping)
- **Captions:** 5 human-written captions per image (2,500 total captions)
- **Splits:** 400 Train / 50 Validation / 50 Test

## Training & Fine-Tuning Approach
- **Base Model:** `Salesforce/blip-image-captioning-base`
- **Optimizer:** AdamW (`learning_rate=5e-5`)
- **Epochs:** 3
- **Batch Size:** 4
- **Environment:** Designed for Google Colab (T4 GPU) using standard PyTorch `DataLoader` with `num_workers=2` and `pin_memory=True` for optimized I/O.
- Only the best validation model is saved.

## Features Currently Implemented
- ✅ Complete PyTorch custom `Dataset` and `DataLoader` for Flickr8k.
- ✅ Google Colab optimized training loop with best-model checkpointing.
- ✅ BLEU-1 and BLEU-4 score evaluation on the test set (50 images with 250 reference captions).
  - **BLEU-1 Score:** 0.5796
  - **BLEU-4 Score:** 0.1759
- ✅ Interactive Streamlit web application.
- ✅ Three selectable caption modes (Short, Medium, Detailed) controlling model generation parameters (beam search, length penalty).
- ✅ Dynamic fallback mechanism: Automatically uses the local fine-tuned model if available, else falls back to the pretrained Hugging Face base model.

## Project Structure
```text
├── data/
│   ├── Images/                 # 500 subset images
│   ├── splits/                 # Train/Val/Test text files
│   └── captions.txt            # Tab-delimited annotations
├── models/
│   ├── blip_flickr8k_500/      # Fine-tuned output (if trained locally)
│   └── trained_model.zip       # Colab export zip (auto-extracts at runtime)
├── src/
│   ├── app.py                  # Streamlit frontend
│   └── model.py                # Model loading and inference logic
├── training/
│   ├── colab_train.py          # Optimized Colab training script
│   ├── dataset.py              # PyTorch Dataset class
│   ├── download_subset.py      # HuggingFace streaming dataset downloader
│   ├── evaluate.py             # BLEU score evaluation
│   └── prepare_dataset.py      # Split generation
├── README.md
└── requirements.txt
```

## Installation Requirements
```bash
pip install -r requirements.txt
pip install nltk datasets  # Required for evaluation and downloading
```

## How to Run the Streamlit Application
From the root of the project, run:
```bash
streamlit run src/app.py
```

## How the Trained Model is Loaded
The application (`src/model.py`) is designed to prioritize your fine-tuned model. 
1. Upon startup, it checks for `models/trained_model.zip`.
2. If found, it extracts it to `models/trained_model_extracted` and automatically detects the nested Colab folder structure.
3. It loads the **fine-tuned weights** from the extracted directory, while securely loading the **Processor Configuration** from the base Hugging Face model to avoid tokenization errors.
4. If no ZIP is found, it falls back entirely to the base `Salesforce/blip-image-captioning-base` model.

## Example Usage
1. Open the Streamlit web app.
2. Upload a standard JPG or PNG image.
3. Select "Short", "Medium", or "Detailed".
4. Click "Generate Caption" to trigger the pipeline and view the resulting text.

## Limitations
- The fine-tuned model was trained on a very small subset (400 images) for demonstration purposes, meaning it may struggle with out-of-domain images compared to the massive base model.
- Inference is currently performed on CPU by default if a local CUDA-capable GPU is not detected, which can take several seconds per image.

## Future Scope
- Expand fine-tuning to the full Flickr8k or MS-COCO dataset.
- Implement Mixed Precision (FP16) in the training loop for faster execution.
- Add support for exporting the final model to ONNX or TensorRT for accelerated inference on CPUs.
