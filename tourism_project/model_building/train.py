# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
# for model serialization
import joblib
import os
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()

dataset_repo_id = "NehaRai22/Wellness_Tourism_Prediction"

# Download files locally before reading with pandas to avoid hf:// path issues
# Force download to ensure latest versions are fetched
Xtrain_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="Xtrain.csv", repo_type="dataset", force_download=True)
Xtest_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="Xtest.csv", repo_type="dataset", force_download=True)
ytrain_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="ytrain.csv", repo_type="dataset", force_download=True)
ytest_path_local = hf_hub_download(repo_id=dataset_repo_id, filename="ytest.csv", repo_type="dataset", force_download=True)

Xtrain = pd.read_csv(Xtrain_path_local)
Xtest = pd.read_csv(Xtest_path_local)
ytrain = pd.read_csv(ytrain_path_local).squeeze()
ytest = pd.read_csv(ytest_path_local).squeeze()

print(f"Shape of Xtrain loaded in train.py: {Xtrain.shape}")
print(f"Shape of ytrain loaded in train.py: {ytrain.shape}")

# Define numeric and categorical features (these should be consistent with prep.py)
numeric_features = [
        "Age",
        "NumberOfPersonVisiting",
        "NumberOfTrips",
        "NumberOfChildrenVisiting",
        "NumberOfFollowups",
        "DurationOfPitch",
        "PitchSatisfactionScore",
        "MonthlyIncome",
  ]

categorical_features = [
        
