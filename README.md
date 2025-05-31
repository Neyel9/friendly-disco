# Friendly Disco

An end-to-end sentiment analysis project demonstrating best practices in machine learning development, deployment, and operations.

![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)

## 📌 Project Overview

**Friendly Disco** is a comprehensive sentiment analysis project that integrates the full machine learning lifecycle, from data ingestion to model deployment. It leverages tools like DVC for data versioning, MLflow for experiment tracking, Prometheus & Grafana for monitoring, and Flask for serving the model. The project also includes S3 integration for data storage and is designed to be deployed on Amazon EKS (Elastic Kubernetes Service) for scalable production deployment.

## 🚀 Features

* **Data Versioning**: Utilize DVC to manage and version datasets efficiently.
* **Experiment Tracking**: Employ MLflow to track experiments, parameters, and metrics.
* **S3 Integration**: Fetch datasets directly from Amazon S3 buckets for scalable data handling.
* **Monitoring**: Use Prometheus and Grafana to monitor model and application metrics in real time.
* **Model Deployment**: Deploy trained models using a Flask API.
* **EKS Deployment**: Seamless deployment to Kubernetes using Amazon EKS.
* **CI/CD Integration**: Implement continuous integration and deployment pipelines.
* **Modular Architecture**: Maintain a clean and modular codebase for scalability.

## 🗂️ Project Structure

```
friendly-disco/
├── .dvc/                   # DVC configurations
├── .github/workflows/      # GitHub Actions workflows
├── docs/                   # Project documentation
├── flask_app/              # Flask application for model serving
├── mlruns/                 # MLflow tracking data
├── notebooks/              # Jupyter notebooks for exploration
├── references/             # Reference materials
├── reports/                # Generated reports and logs
├── scripts/                # Utility scripts
├── src/                    # Source code for data processing and modeling
├── tests/                  # Unit and integration tests
├── .gitignore              # Git ignore file
├── Dockerfile              # Docker configuration
├── LICENSE                 # Project license
├── Makefile                # Makefile for automation
├── README.md               # Project README
├── dvc.yaml                # DVC pipeline configuration
├── params.yaml             # Parameters for the pipeline
├── requirements.txt        # Python dependencies
└── setup.py                # Package setup
```

## 🛠️ Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/Sarthaksina/friendly-disco.git
   cd friendly-disco
   ```

2. **Create and activate a virtual environment**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Install DVC**:

   ```bash
   pip install dvc
   ```

5. **Set up environment variables**:

   Create a `.env` file in the root directory and add the following:

   ```env
   MLFLOW_TRACKING_URI=your_tracking_uri_here
   DAGSHUB_TOKEN=your_dagshub_token
   DAGSHUB_REPO_OWNER=Sarthaksina
   DAGSHUB_REPO_NAME=friendly-disco
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   S3_BUCKET_NAME=your_s3_bucket_name
   ```

5. **Initialize DVC and pull data**:

   ```bash
   dvc init
   dvc pull
   ```

**Troubleshooting DagsHub Authentication:**

If you are encountering authentication errors with DagsHub, please try the following:

*   **Verify your DagsHub token:** Make sure that the `DAGSHUB_TOKEN` environment variable is set correctly in your `.env` file with your DagsHub access token.
*   **Check your DagsHub token validity:** Ensure that your DagsHub access token is valid and has the necessary permissions. You can regenerate your token in your DagsHub settings under "Tokens".
*   **Verify repository details:** Ensure that `DAGSHUB_REPO_OWNER` and `DAGSHUB_REPO_NAME` match your actual DagsHub repository.
*   **Check your network connection:** Make sure that you have a stable network connection and that you are able to connect to DagsHub.
*   **Restart your terminal:** Sometimes, restarting your terminal can help to resolve authentication issues.

**Note:** This project now uses token-based authentication with DagsHub. The old username/password authentication method has been replaced with the more secure `DAGSHUB_TOKEN` approach.

## 📈 Usage

1. **Run the ML pipeline**:

   ```bash
   python src/train_pipeline.py
   ```

2. **Serve the model with Flask**:

   ```bash
   python flask_app/app.py
   ```

   The API will be available at `http://localhost:5000`.

3. **Monitor using Prometheus and Grafana**:

   * Prometheus metrics are exposed at `/metrics` endpoint.
   * Use Grafana dashboards to visualize application and model metrics.

4. **Track experiments**:

   Run `mlflow ui` to launch the tracking interface and inspect your experiments.

## 🔪 Testing

Run the test suite using:

```bash
pytest tests/
```

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## 📬 Contact

For any inquiries or feedback, please contact [Sarthaksina](mailto:sarthaksina@example.com).
