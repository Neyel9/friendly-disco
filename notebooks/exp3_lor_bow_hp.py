import os
import re
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & DAGsHub integration
# Load environment variables from .env file
load_dotenv()

# -------------------- 🔐 Secure Environment Setup --------------------
# Read credentials from environment
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

# Validate required environment variables
missing_vars = []
for var_name, var in {
    "MLFLOW_TRACKING_URI": mlflow_uri,
    "DAGSHUB_REPO_OWNER": dagshub_owner,
    "DAGSHUB_REPO_NAME": dagshub_repo,
    "DAGSHUB_TOKEN": dagshub_token
}.items():
    if not var:
        missing_vars.append(var_name)

if missing_vars:
    print(f"⚠️  Warning: Missing environment variables: {', '.join(missing_vars)}")
    print("🔧 Notebook will run in local mode without MLflow integration")

# Set up MLflow authentication for DagsHub (if token is available)
if dagshub_token:
    # DagsHub uses token-based authentication where the token serves as both username and password
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# Set MLflow URI (if available)
if mlflow_uri:
    mlflow.set_tracking_uri(mlflow_uri)

# Optional: Initialize DagsHub tracking (if credentials are available)
if dagshub_owner and dagshub_repo and dagshub_token:
    try:
        os.environ["DAGSHUB_TOKEN"] = dagshub_token
        dagshub.init(repo_owner=dagshub_owner, repo_name=dagshub_repo, mlflow=True)
        print("✅ DagsHub initialized with token authentication")
    except Exception as e:
        print(f"⚠️  Warning: Failed to initialize DagsHub: {e}. Continuing without DagsHub integration.")
else:
    print("ℹ️  DagsHub credentials not found. Running in local mode.")
mlflow.set_experiment("LoR Hyperparameter Tuning")


# ==========================
# Text Preprocessing Functions
# ==========================
def preprocess_text(text):
    """Applies multiple text preprocessing steps."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)  # Remove punctuation
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = " ".join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])  # Lemmatization & stopwords removal
    
    return text.strip()


# ==========================
# Load & Prepare Data
# ==========================
def load_and_prepare_data(filepath):
    """Loads, preprocesses, and vectorizes the dataset."""
    df = pd.read_csv(filepath)
    
    # Apply text preprocessing
    df["review"] = df["review"].astype(str).apply(preprocess_text)
    
    # Filter for binary classification
    df = df[df["sentiment"].isin(["positive", "negative"])]
    df["sentiment"] = df["sentiment"].map({"negative": 0, "positive": 1})
    
    # Convert text data to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["review"])
    y = df["sentiment"]
    
    return train_test_split(X, y, test_size=0.2, random_state=42), vectorizer


# ==========================
# Train & Log Model
# ==========================
def train_and_log_model(X_train, X_test, y_train, y_test, vectorizer):
    """Trains a Logistic Regression model with GridSearch and logs results to MLflow."""
    
    param_grid = {
        "C": [0.1, 1, 10],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"]
    }
    
    with mlflow.start_run():
        grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring="f1", n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Log all hyperparameter tuning runs
        for params, mean_score, std_score in zip(grid_search.cv_results_["params"], 
                                                 grid_search.cv_results_["mean_test_score"], 
                                                 grid_search.cv_results_["std_test_score"]):
            with mlflow.start_run(run_name=f"LR with params: {params}", nested=True):
                model = LogisticRegression(**params)
                model.fit(X_train, y_train)
                
                y_pred = model.predict(X_test)
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "precision": precision_score(y_test, y_pred),
                    "recall": recall_score(y_test, y_pred),
                    "f1_score": f1_score(y_test, y_pred),
                    "mean_cv_score": mean_score,
                    "std_cv_score": std_score
                }
                
                # Log parameters & metrics
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                
                print(f"Params: {params} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_score']:.4f}")

        # Log the best model
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        best_f1 = grid_search.best_score_

        mlflow.log_params(best_params)
        mlflow.log_metric("best_f1_score", best_f1)
        mlflow.sklearn.log_model(best_model, "model")
        
        print(f"\nBest Params: {best_params} | Best F1 Score: {best_f1:.4f}")


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    (X_train, X_test, y_train, y_test), vectorizer = load_and_prepare_data("notebooks/data.csv")
    train_and_log_model(X_train, X_test, y_train, y_test, vectorizer)
