import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import mlflow
import mlflow.sklearn
import dagshub
import os
import sys
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
load_dotenv()  # Load local .env (ignored in CI, but helpful locally)

# Read credentials from environment
mlflow_username = os.getenv("MLFLOW_TRACKING_USERNAME")
mlflow_password = os.getenv("MLFLOW_TRACKING_PASSWORD")
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")

# Validate all required environment variables
missing_vars = []
for var_name, var in {
    "MLFLOW_TRACKING_USERNAME": mlflow_username,
    "MLFLOW_TRACKING_PASSWORD": mlflow_password,
    "MLFLOW_TRACKING_URI": mlflow_uri,
    "DAGSHUB_REPO_OWNER": dagshub_owner,
    "DAGSHUB_REPO_NAME": dagshub_repo
}.items():
    if not var:
        missing_vars.append(var_name)

if missing_vars:
    raise EnvironmentError(f"❌ Missing required environment variables: {', '.join(missing_vars)}")

# Set credentials for MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
mlflow.set_tracking_uri(mlflow_uri)

# Optional: Initialize DagsHub for MLflow tracking
dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)

# ---------------------------------------------------------------------

def load_model(file_path: str):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
        logging.info('Model loaded from %s', file_path)
        return model
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    try:
        y_pred = clf.predict(X_test)
        y_pred_proba = clf.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    try:
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')

            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            save_metrics(metrics, 'reports/metrics.json')

            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)

            mlflow.sklearn.log_model(clf, "model")
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()
