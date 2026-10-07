FROM python:3.10-slim

LABEL maintainer="Esvanth Mohankumar <x24311073@student.ncirl.ie>"
LABEL project="QuantumShield IDS"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Create necessary directories
RUN mkdir -p data/processed/cicids2017 data/processed/ciciot2023 \
             models/classical models/quantum \
             results/shap_plots

# Expose Flask port
EXPOSE 5000

# Default: run the API
CMD ["python", "deployment/api.py"]
