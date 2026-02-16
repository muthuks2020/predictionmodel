# ODLD Recommendation Engine - SageMaker Container
# Python 3.11

FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH="/opt/program:${PATH}"

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/program/requirements.txt
RUN pip install --no-cache-dir -r /opt/program/requirements.txt

# Copy source code
COPY config/ /opt/program/config/
COPY src/ /opt/program/src/
COPY endpoints/ /opt/program/endpoints/

# SageMaker expects the training script at specific location
COPY endpoints/training/train.py /opt/program/train
COPY endpoints/inference/inference.py /opt/program/serve

# Make entry points executable
RUN chmod +x /opt/program/train /opt/program/serve

WORKDIR /opt/program

# SageMaker training entry point
ENTRYPOINT ["python"]
