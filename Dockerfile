# Start from the official Python image
FROM python:3.10-slim AS build

# Set the Current Working Directory inside the container
WORKDIR /app

# Update and upgrade system packages
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y --no-install-recommends \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Upgrade pip, setuptools, and wheel within the virtual environment
RUN /opt/venv/bin/pip install --no-cache-dir -U pip setuptools wheel

# Activate the virtual environment
SHELL ["/bin/bash", "-c"]
RUN source /opt/venv/bin/activate

# Copy the Python requirements file
COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install --no-cache-dir -r requirements.txt

# Start a new stage from a fresh Python slim image
FROM python:3.10-slim

# Set the Current Working Directory inside the container
WORKDIR /app

# Install necessary dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Copy the Python application code
COPY . .

# Copy dependencies from the previous stage
COPY --from=build /opt/venv /opt/venv

# Add the virtual environment's binary directory to PATH
ENV PATH="/opt/venv/bin:${PATH}"

# Command to run the Python application
CMD ["python", "main.py"]
