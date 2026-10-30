# Hugging Face Model Upload & Hosting Guide

This guide explains how to upload the fine-tuned BLIP model weights (`models/trained_model.zip` or the extracted weights in `models/trained_model_extracted/`) to the **Hugging Face Model Hub**.

Hosting your fine-tuned weights on Hugging Face allows your Streamlit Cloud deployment to download the trained weights on demand without exceeding GitHub's 100 MB file limit.

---

## 1. Prerequisites

1. Create a free account at [Hugging Face](https://huggingface.co/join) (if you do not have one).
2. Generate an Access Token with **Write** permission:
   - Go to **Settings** &rarr; **Access Tokens** &rarr; **New token** (`Role: Write`).
   - Copy your token (e.g., `hf_...`).
3. Install `huggingface_hub` on your machine:
   ```bash
   pip install huggingface_hub
   ```

---

## 2. Option A: Upload Extracted Model Folder via Python (Recommended)

This is the fastest and most reliable method. It uploads the actual model files (`model.safetensors`, `config.json`, `tokenizer.json`, etc.) directly to your new Hugging Face repository.

### Step 1: Create a Python Upload Script
Run the following Python script from the root of your project:

```python
from huggingface_hub import HfApi, create_repo
import os

# 1. Your Hugging Face Credentials & Desired Repo Name
HF_TOKEN = "your_hf_token_here"  # Replace with your Hugging Face write token
REPO_NAME = "blip-image-captioning-flickr8k"  # Your repository name

api = HfApi(token=HF_TOKEN)
user = api.whoami()["name"]
repo_id = f"{user}/{REPO_NAME}"

print(f"Target repository: {repo_id}")

# 2. Create the repository on Hugging Face Model Hub (if it doesn't already exist)
create_repo(repo_id=repo_id, repo_type="model", token=HF_TOKEN, exist_ok=True)
print("Repository created or verified successfully.")

# 3. Path to your extracted fine-tuned model
local_model_dir = os.path.join(
    "models", "trained_model_extracted", "content", "models", "blip_flickr8k_500"
)

# Verify the path exists
if not os.path.exists(local_model_dir):
    raise FileNotFoundError(f"Model folder not found at {local_model_dir}. Please extract trained_model.zip first.")

print(f"Uploading files from {local_model_dir}...")

# 4. Upload all model files to Hugging Face
api.upload_folder(
    folder_path=local_model_dir,
    repo_id=repo_id,
    repo_type="model",
    token=HF_TOKEN,
    commit_message="Initial commit of fine-tuned BLIP model weights"
)

print(f"\nModel successfully uploaded! View your model at:")
print(f"https://huggingface.co/{repo_id}")
```

---

## 3. Option B: Upload via Hugging Face CLI

Alternatively, you can use the Hugging Face Command Line Interface:

1. Log in from your terminal:
   ```bash
   huggingface-cli login
   ```
   (Paste your write token when prompted)

2. Create the repository:
   ```bash
   huggingface-cli repo create blip-image-captioning-flickr8k --type model
   ```

3. Upload the model folder:
   ```bash
   huggingface-cli upload username/blip-image-captioning-flickr8k ./models/trained_model_extracted/content/models/blip_flickr8k_500 . --repo-type model
   ```

---

## 4. Verification

After uploading, visit `https://huggingface.co/<your-username>/blip-image-captioning-flickr8k` to verify the presence of:
- `model.safetensors` (~854 MB)
- `config.json`
- `generation_config.json`
- `processor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`

---

## 5. Integrating with `src/model.py` for Cloud Deployment

Once uploaded, update `src/model.py` so that when `models/trained_model.zip` is not found locally, it automatically downloads your fine-tuned weights from Hugging Face instead of defaulting to the base Salesforce model:

```python
# In src/model.py:
base_model_id = "Salesforce/blip-image-captioning-base"
hf_fine_tuned_repo = "your-username/blip-image-captioning-flickr8k" # Your HF repo

if os.path.exists(zip_path):
    # Use local extracted zip
    ...
else:
    # Use your fine-tuned model hosted on Hugging Face!
    model_id = hf_fine_tuned_repo
```
Both the processor and the weights will download directly and securely via Hugging Face's global CDN during the first cloud run and be cached automatically by Streamlit.

