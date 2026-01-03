# for data manipulation

# for data manipulation
import pandas as pd
import sklearn
import os
import re
from sklearn.model_selection import train_test_split
from huggingface_hub import HfApi

# Read token from environment
hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise RuntimeError("HF_TOKEN environment variable is not set.")

api = HfApi(token=hf_token)

DATASET_PATH = "hf://datasets/NehaRai22/Wellness_Tourism_Prediction/tourism.csv"

df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")
print(f"Initial df shape: {df.shape}")

# Drop 'Unnamed: 0' or similar if it exists
unnamed_cols = [col for col in df.columns if re.match(r'Unnamed: \d+', col)]
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)
    print(f"Dropped unnamed columns: {unnamed_cols}")
    print(f"df shape after dropping unnamed columns: {df.shape}")

target_col = 'ProdTaken'
X = df.drop(columns=[target_col, 'CustomerID'])
y = df[target_col]

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Xtrain shape after split: {Xtrain.shape}")
print(f"ytrain shape after split: {ytrain.shape}")

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="NehaRai22/tourism_project",
        repo_type="dataset",
        commit_message=f"Update split data files for {file_path} after dropping Unnamed columns"
    )
    print(f"Uploaded {file_path} to Hugging Face.")
