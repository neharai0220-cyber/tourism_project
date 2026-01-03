
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
import os

# Read token from environment
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable is not set.")

# Initialize API client
api = HfApi(token=hf_token)

repo_id = "NehaRai22/Wellness_Tourism_Prediction"
repo_type = "dataset"
local_data_path = "tourism_project/data/tourism.csv"
remote_data_filename = "tourism.csv"

# Check or create dataset repo
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

# Upload file if present
if not os.path.exists(local_data_path):
    print(f"Error: Local file '{local_data_path}' not found. Please upload 'tourism.csv' to 'tourism_project/data'.")
else:
    print(f"Uploading '{local_data_path}' to '{repo_id}' as '{remote_data_filename}'...")
    api.upload_file(
        path_or_fileobj=local_data_path,
        path_in_repo=remote_data_filename,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=f"Upload {remote_data_filename}"
    )
    print(f"File '{remote_data_filename}' uploaded successfully to '{repo_id}'.")
