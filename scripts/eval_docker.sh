#!/bin/bash

# Script to build HumanEval Docker sandbox image with sudo
set -e  # Exit on any error

IMAGE_NAME="humaneval-sandbox"
DOCKERFILE_CONTENT='FROM python:3.9-slim

# Install required packages
RUN pip install numpy pandas

# Create non-root user for security
RUN useradd -m -u 1000 codeeval
USER codeeval
WORKDIR /home/codeeval

# Set resource limits
LABEL description="HumanEval code execution sandbox"
'

echo "Building HumanEval Docker sandbox image..."

# Check if image already exists
if sudo docker images --format "table {{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}:latest$"; then
    echo "Image ${IMAGE_NAME} already exists. Rebuilding..."
    sudo docker rmi ${IMAGE_NAME} || true
fi

# Create temporary directory for build context
BUILD_DIR=$(mktemp -d)
trap "rm -rf ${BUILD_DIR}" EXIT

# Write Dockerfile to temporary location
echo "${DOCKERFILE_CONTENT}" > "${BUILD_DIR}/Dockerfile"

echo "Building Docker image from ${BUILD_DIR}..."

# Build the image with sudo
sudo docker build \
    -t ${IMAGE_NAME} \
    --rm \
    "${BUILD_DIR}"

echo "Docker image '${IMAGE_NAME}' built successfully!"

# Optional: Show image info
echo -e "\nImage details:"
sudo docker images ${IMAGE_NAME}