from flask import Flask, render_template, request
import mlflow
from mlflow.exceptions import MlflowException
import pickle
import os
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import re
import dagshub
from dotenv import load_dotenv
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(df):
    """Remove sentences with less than 3 words."""
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)

    return text

# Below code block is for production use
# -------------------------------------------------------------------------------------
# -------------------- 🔐 Secure Environment Setup --------------------
load_dotenv()  # Load environment variables from .env

# Read credentials from environment
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

# Validate required environment variables (only check if not in test mode)
if not os.getenv("TESTING"):
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
        print("🔧 Flask app will run in local mode without MLflow integration")

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

# # -------------------------------------------------------------------------------------



# Initialize Flask app
app = Flask(__name__)

# from prometheus_client import CollectorRegistry

# Create a custom registry
registry = CollectorRegistry()

# Define your custom metrics using this registry
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# ------------------------------------------------------------------------------------------
# Model and vectorizer setup
model_name = "my_model"

def get_latest_model_version(model_name, desired_stage="Staging"):
    """Fetch the latest model version from a given stage (e.g. 'Staging' or 'Production')."""
    client = mlflow.MlflowClient()
    # Fetch all versions for the model
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise Exception(f"No versions found for model '{model_name}'")

    # Filter by desired stage
    filtered = [v for v in versions if v.current_stage == desired_stage]
    if not filtered:
        raise Exception(f"No model version found in stage '{desired_stage}' for model '{model_name}'")
    
    # Sort based on version number (converted to int)
    latest = max(filtered, key=lambda v: int(v.version))
    return latest.version

try:
    # Change desired_stage below if needed (e.g., "Production")
    model_version = get_latest_model_version(model_name, desired_stage="Staging")
    model_uri = f"models:/{model_name}/{model_version}"
    print(f"✅ Fetching model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

try:
    vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
except Exception as e:
    print(f"❌ Failed to load vectorizer: {e}")
    vectorizer = None
# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()

    try:
        # Check if model and vectorizer are available
        if model is None:
            error_msg = "❌ Model not available. Please check MLflow connection and ensure a model is registered in 'Staging' stage."
            print(error_msg)
            return render_template("index.html", result=None, error=error_msg)

        if vectorizer is None:
            error_msg = "❌ Vectorizer not available. Please check if vectorizer.pkl exists."
            print(error_msg)
            return render_template("index.html", result=None, error=error_msg)

        text = request.form["text"]

        # Validate input
        if not text or text.strip() == "":
            error_msg = "⚠️ Please enter some text to analyze."
            return render_template("index.html", result=None, error=error_msg)

        # Clean text
        text = normalize_text(text)

        # Convert to features
        features = vectorizer.transform([text])
        features_df = pd.DataFrame(features.toarray(), columns=[str(i) for i in range(features.shape[1])])

        # Predict
        result = model.predict(features_df)
        prediction = result[0]

        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()

        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)

        return render_template("index.html", result=prediction, error=None)

    except Exception as e:
        error_msg = f"❌ Prediction failed: {str(e)}"
        print(error_msg)
        return render_template("index.html", result=None, error=error_msg)

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint to diagnose system status."""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": vectorizer is not None,
        "mlflow_uri": os.getenv("MLFLOW_TRACKING_URI", "Not set"),
        "dagshub_token": "Set" if os.getenv("DAGSHUB_TOKEN") else "Not set",
        "dagshub_owner": os.getenv("DAGSHUB_REPO_OWNER", "Not set"),
        "dagshub_repo": os.getenv("DAGSHUB_REPO_NAME", "Not set")
    }

    if not model:
        status["model_error"] = "Model failed to load from MLflow"
    if not vectorizer:
        status["vectorizer_error"] = "Vectorizer failed to load from local file"

    return status, 200

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose only custom Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker
