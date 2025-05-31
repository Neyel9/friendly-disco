# promote model

import os
import logging
import mlflow
import dagshub
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------- 🔐 Secure Environment Setup --------------------
load_dotenv()  # Load environment variables from .env

def promote_model():
    # Read credentials from environment
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
    dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
    dagshub_token = os.getenv("DAGSHUB_TOKEN")

    # Validate all required environment variables
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
        raise EnvironmentError(f"❌ Missing required environment variables: {', '.join(missing_vars)}")

    # Set up MLflow authentication for DagsHub
    # DagsHub uses token-based authentication where the token serves as both username and password
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

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

    try:
        client = mlflow.MlflowClient()
        model_name = "my_model"

        logging.info(f"🔍 Checking for models in staging for '{model_name}'...")

        # Get the latest version in staging
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
        if not staging_versions:
            logging.error(f"❌ No model versions found in 'Staging' stage for model '{model_name}'")
            raise ValueError(f"No model versions in staging for {model_name}")

        latest_version_staging = staging_versions[0].version
        logging.info(f"✅ Found model version {latest_version_staging} in staging")

        # Archive the current production model
        logging.info("🔄 Checking for existing production models to archive...")
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])

        if prod_versions:
            for version in prod_versions:
                logging.info(f"📦 Archiving production model version {version.version}")
                client.transition_model_version_stage(
                    name=model_name,
                    version=version.version,
                    stage="Archived"
                )
                logging.info(f"✅ Model version {version.version} archived successfully")
        else:
            logging.info("ℹ️  No existing production models to archive")

        # Promote the new model to production
        logging.info(f"🚀 Promoting model version {latest_version_staging} to Production...")
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_staging,
            stage="Production"
        )

        logging.info(f"🎉 SUCCESS: Model version {latest_version_staging} promoted to Production!")
        print(f"✅ Model version {latest_version_staging} promoted to Production")

    except Exception as e:
        logging.error(f"❌ Failed to promote model: {e}")
        raise

if __name__ == "__main__":
    promote_model()
