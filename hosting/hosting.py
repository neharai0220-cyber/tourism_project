from huggingface_hub import HfApi
import os

# Read token from environment (provided by GitHub Actions secret HF_TOKEN)
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable is not set.")

api = HfApi(token=hf_token)

api.upload_folder(
    folder_path="tourism_project/deployment",  # local folder to upload
    repo_id="NehaRai22/Wellness_Tourism_Prediction",  # target Space repo
    repo_type="space",
    path_in_repo=""  # optional subfolder in the repo
)
