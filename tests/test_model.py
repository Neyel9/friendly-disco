# load test + signature test + performance test

import unittest
import mlflow
import os
import pandas as pd
import logging
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # -------------------- 🔐 Secure Environment Setup --------------------
        load_dotenv()  # Load environment variables from .env

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
            # Load the new model from MLflow model registry
            cls.new_model_name = "my_model"
            logging.info(f"🔍 Looking for latest model version for '{cls.new_model_name}'...")

            cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
            if not cls.new_model_version:
                raise ValueError(f"No model versions found for '{cls.new_model_name}' in Staging")

            cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
            logging.info(f"📦 Loading model from URI: {cls.new_model_uri}")

            cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)
            logging.info(f"✅ Model loaded successfully: version {cls.new_model_version}")

            # Load the vectorizer
            logging.info("📊 Loading vectorizer...")
            cls.vectorizer = pickle.load(open('models/vectorizer.pkl', 'rb'))
            logging.info("✅ Vectorizer loaded successfully")

            # Load holdout test data
            logging.info("📈 Loading holdout test data...")
            cls.holdout_data = pd.read_csv('data/processed/test_bow.csv')
            logging.info(f"✅ Test data loaded: {cls.holdout_data.shape[0]} samples, {cls.holdout_data.shape[1]} features")

        except Exception as e:
            logging.error(f"❌ Failed to set up test environment: {e}")
            raise

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        """Test that the model was loaded successfully from MLflow registry."""
        logging.info("🧪 Testing model loading...")
        self.assertIsNotNone(self.new_model, "Model should be loaded and not None")
        logging.info("✅ Model loading test passed")

    def test_model_signature(self):
        """Test that the model has the correct input/output signature."""
        logging.info("🧪 Testing model signature...")

        # Create a dummy input for the model based on expected input shape
        input_text = "hi how are you"
        input_data = self.vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=[str(i) for i in range(input_data.shape[1])])

        logging.info(f"📊 Input shape: {input_df.shape}")

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)
        logging.info(f"🎯 Prediction shape: {prediction.shape}")

        # Verify the input shape
        expected_features = len(self.vectorizer.get_feature_names_out())
        self.assertEqual(input_df.shape[1], expected_features,
                        f"Input should have {expected_features} features, got {input_df.shape[1]}")

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0],
                        "Prediction length should match input batch size")
        self.assertEqual(len(prediction.shape), 1,
                        "Prediction should be 1-dimensional for binary classification")

        logging.info("✅ Model signature test passed")

    def test_model_performance(self):
        """Test that the model meets minimum performance thresholds."""
        logging.info("🧪 Testing model performance...")

        # Extract features and labels from holdout test data
        X_holdout = self.holdout_data.iloc[:,0:-1]
        y_holdout = self.holdout_data.iloc[:,-1]

        logging.info(f"📊 Holdout data: {X_holdout.shape[0]} samples, {X_holdout.shape[1]} features")

        # Predict using the new model
        logging.info("🎯 Making predictions on holdout data...")
        y_pred_new = self.new_model.predict(X_holdout)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Log the actual performance metrics
        logging.info(f"📈 Model Performance Metrics:")
        logging.info(f"   • Accuracy:  {accuracy_new:.4f}")
        logging.info(f"   • Precision: {precision_new:.4f}")
        logging.info(f"   • Recall:    {recall_new:.4f}")
        logging.info(f"   • F1 Score:  {f1_new:.4f}")

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        logging.info(f"🎯 Performance Thresholds:")
        logging.info(f"   • Min Accuracy:  {expected_accuracy}")
        logging.info(f"   • Min Precision: {expected_precision}")
        logging.info(f"   • Min Recall:    {expected_recall}")
        logging.info(f"   • Min F1 Score:  {expected_f1}")

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy,
                               f'Accuracy {accuracy_new:.4f} should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision,
                               f'Precision {precision_new:.4f} should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall,
                               f'Recall {recall_new:.4f} should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1,
                               f'F1 score {f1_new:.4f} should be at least {expected_f1}')

        logging.info("✅ Model performance test passed - all metrics meet thresholds!")

if __name__ == "__main__":
    unittest.main()
