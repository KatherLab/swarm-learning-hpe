#!/bin/sh
set -eux

help() {
  echo "Usage: $0 [-h]"
  echo "  -h    display help"
}

while getopts ":h" opt; do
  case ${opt} in
    h )
      help
      exit 0
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      help
      exit 1
      ;;
    : )
      echo "Option -$OPTARG requires an argument" 1>&2
      help
      exit 1
      ;;
  esac
done

echo "Installing required packages"
sudo apt-get update
sudo apt-get install -y docker.io curl git openssh-server openssh-client
# Update package manager and install pip
sudo apt install python3-pip -y
# Install gdown package for downloading from Google Drive
pip install -U --no-cache-dir gdown --pre

echo "Creating folder for Swarm Learning"
DIR="/opt/hpe/swarm-learning-hpe"
if [ -d "$DIR" ]; then
  cd "$DIR"
  git pull
  git checkout dev_radiology #TODO: change to master after merge
else
  sudo mkdir -p /opt/hpe
  sudo git clone https://github.com/KatherLab/swarm-learning-hpe.git "$DIR"
  git pull
  git checkout dev_radiology #TODO: change to master after merge
fi

echo "Setting permissions for Swarm Learning folder"
sudo chmod 777 -R "$DIR"

echo setup vpn tunnel for swarm learning
sudo sh ./workspace/automate_scripts/server_setup/setup_vpntunnel.sh

echo "Installation completed successfully!"
