#!/bin/bash

set -eux

# Help function
help()
{
   echo "Usage: $0 [-h]"
   echo "Installs and configures NVIDIA container runtime."
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
  echo "Please install NVIDIA GPU drivers."
  exit 1
fi

# Set up nvidia-container-runtime
echo "Setting up nvidia-container-runtime..."
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo service docker restart

# Install nvidia-container-runtime
echo "Installing nvidia-container-runtime..."
sudo apt-get install nvidia-container-runtime -y

# Ensure nvidia-container-runtime-hook is accessible
echo "Ensuring the nvidia-container-runtime-hook is accessible from $PATH..."
which nvidia-container-runtime-hook

# Expose GPUs for use
echo "Exposing GPUs for use..."
sudo docker run -it --rm --gpus all ubuntu nvidia-smi
