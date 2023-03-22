#!/bin/sh
set -eu

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


sudo apt-get update
#sudo apt-get purge container.io -y
echo "Installing docker environment"
sudo apt-get remove docker -y
sudo apt-get remove docker.io -y
sudo apt-get remove containerd -y
sudo apt-get remove runc -y
sudo apt-get update
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg \
    lsb-release
sudo mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo docker run hello-world

echo "Installing required packages"
sudo apt-get install -y curl git openssh-server openssh-client
sudo service docker start
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
  #git checkout dev_radiology #TODO: change to master after merge
fi

#echo "Setting permissions for Swarm Learning folder"
#sudo chmod 777 -R "$DIR"

# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "Pre-requisites installed successfully."