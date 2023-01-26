#!/bin/sh
set -eux

#echo "Ekfz_swarm_2" | sudo -S su
#If you decided to use the 'echo <password> | sudo -S' option, to avoid exposing the password on the command history, start the command with a SPACE character. Of course, the best option is to pipe the password from a secure file.
echo "Check gpu drivers"
if nvidia-smi | grep -q 'NVIDIA-SMI' || 'Driver Version' ; then   echo "gpu driver set up"; fi

echo "Set install nvidia-container-runtime env"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

echo "install nvidia-container-runtime"
sudo apt-get install nvidia-container-runtime -y

echo "Ensure the nvidia-container-runtime-hook is accessible from $PATH"
which nvidia-container-runtime-hook

echo Expose GPUs for use
sudo docker run -it --rm --gpus all ubuntu nvidia-smi