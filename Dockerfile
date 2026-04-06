FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Environment variables (overridden at runtime)
ENV API_BASE_URL="https://api-inference.huggingface.co/v1"
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
ENV HF_TOKEN=""

# Run FastAPI server
CMD ["python", "app.py"]
