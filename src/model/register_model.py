# register model

import json
import mlflow
import logging
import os
import dagshub
import sys
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")
from dotenv import load_dotenv

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

try:
    from src.logger import logging
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    from src.logger import logging

# -------------------- 🔐 Secure Environment Setup --------------------
load_dotenv()  # Load environment variables from .env

# Read credentials from environment
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

# Validate all required environment variables
missing_vars = []
for var_name, var in {
    "MLFLOW_TRACKING_USERNAME": mlflow_username,
    "MLFLOW_TRACKING_URI": mlflow_uri,
    "DAGSHUB_REPO_OWNER": dagshub_owner,
    "DAGSHUB_REPO_NAME": dagshub_repo,
    "DAGSHUB_TOKEN": dagshub_token
}.items():
    if not var:
        missing_vars.append(var_name)

if missing_vars:
    raise EnvironmentError(f"❌ Missing required environment variables: {', '.join(missing_vars)}")

# Patch MLflow to use Bearer token for DagsHub
from mlflow.utils.rest_utils import http_request_kwargs
http_request_kwargs["headers"] = {"Authorization": f"Bearer {dagshub_token}"}

# Set MLflow URI
mlflow.set_tracking_uri(mlflow_uri)

# Optional: Initialize DagsHub tracking
try:
    os.environ["DAGSHUB_TOKEN"] = dagshub_token
    dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
    logging.info("DagsHub initialized with token authentication")
except Exception as e:
    logging.warning("Failed to initialize DagsHub: %s. Continuing with MLflow only.", e)

# ---------------------------------------------------------------------

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_version = mlflow.register_model(model_uri, model_name)

        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error('Error during model registration: %s', e)
        raise

def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        register_model(model_name, model_info)
    except Exception as e:
        logging.error('Failed to complete the model registration process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
