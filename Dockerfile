FROM python:3.10-slim

WORKDIR /app

# Copy Flask app files
COPY flask_app/ /app/

# Copy models (create directory if it doesn't exist)
RUN mkdir -p /app/models
COPY models/vectorizer.pkl /app/models/vectorizer.pkl

# Install Python dependencies
RUN pip install -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader stopwords wordnet

EXPOSE 5000

# For testing - use Python directly
CMD ["python", "app.py"]

# For production - use Gunicorn (commented out for now)
# CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "--workers", "1", "app:app"]