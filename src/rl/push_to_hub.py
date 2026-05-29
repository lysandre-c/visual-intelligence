#!/usr/bin/env python3
"""Standalone script to push final LLaVA DPO weights to Hugging Face Hub.

This script bypasses pipeline errors and version-specific push_to_hub keyword argument
issues by directly using huggingface_hub's `upload_folder` utility.
"""

import logging
from pathlib import Path
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

def main():
    repo_id = "Matisse6410/LlaVa-1.5-SDPO"
    folder_path = Path("results/rl_alignment/final")
    
    if not folder_path.exists():
        logger.error(f"Folder {folder_path} does not exist. Ensure training completed successfully.")
        return
        
    logger.info(f"Pushing final weights from '{folder_path}' to HF Hub repo: {repo_id}")
    # Add your Hugging Face write token here if not set in environment (e.g. hf_token = "hf_...")
    # If left as a placeholder or empty, it will fall back to the HF_TOKEN environment variable.
    hf_token = "<put token here>"

    try:
        if hf_token and hf_token != "YOUR_HF_WRITE_TOKEN_HERE" and hf_token.strip() != "":
            api = HfApi(token=hf_token)
        else:
            api = HfApi()
        
        # This will upload the entire folder (adapter_config.json, adapter_model.safetensors,
        # multi_modal_projector.pt, and README.md) in a single robust API call.
        api.upload_folder(
            folder_path=str(folder_path),
            repo_id=repo_id,
            repo_type="model",
        )
        logger.info(f"Successfully pushed all weights and configs to Hub at {repo_id}!")
    except Exception as e:
        logger.error(
            f"Failed to push to Hub.\n"
            f"Please ensure the HF_TOKEN environment variable is set and has write permissions to {repo_id}.\n"
            f"Error details: {e}"
        )

if __name__ == "__main__":
    main()
