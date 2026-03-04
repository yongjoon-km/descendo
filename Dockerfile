# Dockerfile
FROM python:3.13-slim

# Set the working directory inside the container
WORKDIR /app

ENV PYTHONPATH=/app
# Install dependencies first (this caches the layer so builds are faster)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the rest of your application code into the container
COPY engine/ ./engine/
COPY models/ ./models/
COPY task/ ./task/
COPY main.py .
