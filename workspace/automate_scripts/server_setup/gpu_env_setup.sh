#!/bin/bash

set -eu

# Help function
help()
{
   echo "Usage: $0 [-h]"
   echo "Installs and configures NVIDIA container runtime (Ubuntu 22.04+ compatible)."
   echo ""
   echo "Options:"
   echo "  -h    show help message and exit"
   exit 1
}

while getopts ":h" opt; do
  case ${opt} in
    h )
      help
      ;;
    \? )
      echo "Invalid option: $OPTARG" 1>&2
      exit 1
      ;;
  esac
done

# Check if NVIDIA drivers are installed
echo "Checking NVIDIA drivers..."
if nvidia-smi | grep -q 'NVIDIA-SMI' || nvidia-smi | grep -q 'Driver Version'; then
  echo "GPU drivers are set up."
else
  echo "GPU env failed. Please install NVIDIA GPU drivers first. Run 'nvidia-smi' to verify installation."
  exit 1
fi

# Remove conflicting old config
echo "Cleaning up old NVIDIA APT sources..."
sudo rm -f /etc/apt/sources.list.d/nvidia*.list
sudo rm -f /etc/apt/trusted.gpg.d/nvidia*.gpg
sudo rm -f /etc/apt/keyrings/nvidia-archive-keyring.gpg
sudo rm -f /etc/apt/keyrings/nvidia-container-toolkit.gpg

# Set up nvidia-container-runtime
echo "Setting up NVIDIA container runtime for Ubuntu 22.04..."
distribution="ubuntu22.04"

# Create keyring directory if not exist
sudo mkdir -p /etc/apt/keyrings

# Download and install GPG key
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor | \
  sudo tee /etc/apt/keyrings/nvidia-container-toolkit.gpg > /dev/null

# Add repository with signed-by keyring
curl -s -L https://nvidia.github.io/libnvidia-container/${distribution}/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install container toolkit
echo "Installing nvidia-container-toolkit..."
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Restart Docker to apply
echo "Restarting Docker..."
sudo systemctl restart docker

# Check runtime hook path
echo "Checking if nvidia-container-runtime-hook is accessible..."
if ! which nvidia-container-runtime-hook > /dev/null 2>&1; then
    echo "Warning: nvidia-container-runtime-hook not found in PATH."
else
    echo "nvidia-container-runtime-hook is available."
fi

# Test with Docker
echo "Testing Docker GPU access..."
if sudo docker run -it --rm --gpus all ubuntu nvidia-smi; then
    echo "NVIDIA container runtime set up successfully."
else
    echo "An error occurred while running Docker with GPU. Please check Docker and GPU configuration."
    exit 1
fi
