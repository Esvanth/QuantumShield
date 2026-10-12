# QuantumShield - API container.
#
# Serves the weighted soft-voting ensemble with the validation-fitted weights
# and threshold, so the deployed system matches the evaluated one.

FROM python:3.11-slim

# curl is here for the healthcheck, nothing else is needed at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so a code change does not re-download PyTorch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    DATASET=cicids2017 \
    PORT=5000

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:5000/health || exit 1

# Flask's own server, since gunicorn is not in requirements.txt. Fine for a
# demonstration deployment; a production system would use a WSGI server.
CMD ["python", "-m", "deployment.api"]