# upload_private.py
from huggingface_hub import HfApi
import os

api = HfApi()
repo_id = "Akshaya1303/surgical-weights-private"  # PRIVATE REPO
model_dir = "models"

print("Uploading to PRIVATE repo...")

for file in os.listdir(model_dir):
    if file.endswith(".pth"):
        local_path = os.path.join(model_dir, file)
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"Uploading {file} ({size_mb:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=file,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"Uploaded: {file}")

print("ALL MODELS UPLOADED PRIVATELY!")
print("View at: https://huggingface.co/Akshaya/surgical-weights-private")