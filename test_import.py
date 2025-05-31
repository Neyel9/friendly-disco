#!/usr/bin/env python3
"""
Simple test script to check if the MLflow import issue is resolved
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

print("Testing imports...")

try:
    import mlflow
    print("✅ MLflow imported successfully")
except ImportError as e:
    print(f"❌ Failed to import MLflow: {e}")
    sys.exit(1)

try:
    import mlflow.sklearn
    print("✅ MLflow sklearn imported successfully")
except ImportError as e:
    print(f"❌ Failed to import mlflow.sklearn: {e}")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    print("✅ dotenv imported successfully")
except ImportError as e:
    print(f"❌ Failed to import dotenv: {e}")
    sys.exit(1)

# Test the specific import that was causing issues
print("\nTesting the problematic import that was removed...")
try:
    # This should NOT work in newer MLflow versions
    from mlflow.utils.rest_utils import http_request_kwargs
    print("⚠️  WARNING: http_request_kwargs import still works - this might be an older MLflow version")
except ImportError:
    print("✅ http_request_kwargs import correctly fails (as expected in newer MLflow)")

print("\nTesting basic MLflow functionality...")
try:
    # Test basic MLflow operations
    mlflow.set_tracking_uri("file:///tmp/test_mlflow")
    print("✅ MLflow tracking URI set successfully")
except Exception as e:
    print(f"❌ Failed to set MLflow tracking URI: {e}")

print("\nAll import tests completed!")
