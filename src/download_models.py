import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

MODELS_TO_DOWNLOAD = [
    {"name": "BART", "path": "facebook/bart-large-cnn", "type": "seq2seq"},
    {"name": "PEGASUS", "path": "google/pegasus-multi_news", "type": "seq2seq"},
    {"name": "PRIMERA", "path": "allenai/PRIMERA", "type": "seq2seq"},
    {"name": "Gemma-3-4B", "path": "google/gemma-3-4b-it", "type": "causal"}
]

def download_models():
    print("--- Starting Model Download ---")
    print("This script will download all models to the local Hugging Face cache.")
    print("Once completed, running the main script will not require re-downloading.\n")

    for model_cfg in MODELS_TO_DOWNLOAD:
        name = model_cfg["name"]
        path = model_cfg["path"]
        m_type = model_cfg["type"]
        
        print(f"[{name}] Downloading {path}...")
        try:
            # Download Tokenizer
            print(f"  - Downloading Tokenizer...")
            AutoTokenizer.from_pretrained(path)
            
            # Download Model
            print(f"  - Downloading Model...")
            if m_type == "seq2seq":
                AutoModelForSeq2SeqLM.from_pretrained(path)
            elif m_type == "causal":
                # For Gemma, we might need specific dtype or device_map, but for caching, standard load is fine.
                # We use torch_dtype="auto" to avoid loading full float32 if not needed, though download is same.
                AutoModelForCausalLM.from_pretrained(path, torch_dtype="auto")
            
            print(f"[{name}] Successfully downloaded and cached.\n")
            
        except Exception as e:
            print(f"[{name}] FAILED to download: {e}\n")

    print("--- All downloads finished ---")

if __name__ == "__main__":
    download_models()
