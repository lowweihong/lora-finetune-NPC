#!/usr/bin/env python3
"""
Script to upload your fine-tuned model to Hugging Face Hub.
This allows you to deploy your app without uploading large model files to GitHub.
Supports both quantized and original models.
"""

from huggingface_hub import HfApi, login
import os
from pathlib import Path

def get_folder_size(path):
    """Get total size of folder in GB"""
    total = 0
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            if os.path.exists(filepath):
                total += os.path.getsize(filepath)
    return total / (1024 ** 3)  # Convert to GB

def upload_model():
    """Upload model to Hugging Face Hub"""
    
    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Error: HF_TOKEN environment variable not set")
        print("Get your token from: https://huggingface.co/settings/tokens")
        print("Then run: export HF_TOKEN='your_token_here'")
        return
    
    # Login
    try:
        login(token=hf_token, add_to_git_credential=False)
        print("✓ Logged in to Hugging Face")
    except Exception as e:
        print(f"Error logging in: {e}")
        return
    
    # Get model info
    print("\n" + "=" * 60)
    print("Model Upload to Hugging Face Hub")
    print("=" * 60)
    
    # Get username
    api = HfApi()
    user_info = api.whoami(token=hf_token)
    username = user_info["name"]
    print(f"Username: {username}")
    
    # Check for available models
    quantized_path = "./npc_finetuned_quantized"
    original_path = "./npc_finetuned_bertscore-eval-noeval"
    
    available_models = []
    if os.path.exists(quantized_path):
        size = get_folder_size(quantized_path)
        available_models.append(("Quantized", quantized_path, size))
    if os.path.exists(original_path):
        size = get_folder_size(original_path)
        available_models.append(("Original", original_path, size))
    
    if not available_models:
        print(f"\nError: No model found!")
        print(f"Checked: {quantized_path}, {original_path}")
        return
    
    # Let user choose which model to upload
    print("\nAvailable models:")
    for i, (name, path, size) in enumerate(available_models, 1):
        print(f"  {i}. {name} model ({size:.2f} GB) - {path}")
    
    if len(available_models) == 1:
        model_name, model_path, model_size = available_models[0]
        print(f"\nUsing: {model_name} model")
    else:
        choice = input(f"\nWhich model to upload? (1-{len(available_models)}, default=1): ").strip() or "1"
        try:
            idx = int(choice) - 1
            model_name, model_path, model_size = available_models[idx]
        except (ValueError, IndexError):
            print("Invalid choice, using first model")
            model_name, model_path, model_size = available_models[0]
    
    print(f"\nSelected: {model_name} model ({model_size:.2f} GB)")
    
    # Repository ID
    default_repo = "npc-finetuned-quantized" if "quantized" in model_path.lower() else "npc-finetuned-bertscore"
    repo_id = input(f"\nEnter repository name (default='{default_repo}'): ").strip()
    if not repo_id:
        repo_id = default_repo
    
    full_repo_id = f"{username}/{repo_id}"
    
    # Privacy
    print("\nMake model public or private?")
    print("1. Public (anyone can use, no token needed)")
    print("2. Private (requires HF_TOKEN)")
    choice = input("Choice (1 or 2, default=1): ").strip() or "1"
    private = choice == "2"
    
    print(f"\n" + "=" * 60)
    print("Upload Summary")
    print("=" * 60)
    print(f"Repository: {full_repo_id}")
    print(f"Model: {model_name} ({model_size:.2f} GB)")
    print(f"Source: {model_path}")
    print(f"Privacy: {'Private' if private else 'Public'}")
    print(f"\nNote: Uploading {model_size:.2f} GB may take 30-60+ minutes depending on your connection.")
    print("=" * 60)
    
    confirm = input("\nProceed with upload? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Upload cancelled")
        return
    
    # Create repository first (with privacy setting)
    print("\n" + "=" * 60)
    print("Creating repository...")
    print("=" * 60)
    try:
        api.create_repo(
            repo_id=full_repo_id,
            repo_type="model",
            token=hf_token,
            private=private,
            exist_ok=True  # Don't error if repo already exists
        )
        print(f"✓ Repository created/verified: {full_repo_id}")
    except Exception as e:
        print(f"Note: {e}")
        print("(Repository may already exist, continuing with upload...)")
    
    # Upload with progress indication
    print("\n" + "=" * 60)
    print("Uploading model...")
    print("=" * 60)
    print("This may take a while for large models. Please be patient.")
    print("You can monitor progress in the Hugging Face web interface.")
    print("=" * 60 + "\n")
    
    try:
        api.upload_folder(
            folder_path=model_path,
            repo_id=full_repo_id,
            repo_type="model",
            token=hf_token,
            commit_message=f"Upload {model_name} model ({model_size:.2f} GB)"
        )
        print("\n" + "=" * 60)
        print("✓ Upload successful!")
        print("=" * 60)
        print(f"\nModel available at: https://huggingface.co/{full_repo_id}")
        print(f"\nTo use in deployment, set environment variable:")
        print(f"  HF_MODEL_ID={full_repo_id}")
        if private:
            print(f"  HF_TOKEN={hf_token}")
        print("\nThe app will automatically load from Hugging Face Hub when deployed!")
    except Exception as e:
        print(f"\nError uploading: {e}")
        import traceback
        traceback.print_exc()
        print("\nTip: For very large uploads, you might want to:")
        print("1. Use Git LFS instead (see DEPLOYMENT_ALTERNATIVES.md)")
        print("2. Try uploading during off-peak hours")
        print("3. Check your internet connection stability")

if __name__ == "__main__":
    upload_model()

