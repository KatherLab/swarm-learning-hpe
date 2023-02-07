#!/bin/sh
set -eux

#sudo snap install docker
sudo snap install curl

echo Install Docker
curl -fsSL get.docker.com | sudo sh

echo Test Docker installation:
sudo groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

echo Create Folder / Working directory for Swarm Learning:
DIR="/opt/hpe/swarm-learning-hpe"
if [ -d "$DIR" ]; then
  # Take action if $DIR exists. #
  cd "$DIR"
  git pull
  git checkout dev_automation
else
  sudo mkdir /opt/hpe
  cd /opt/hpe
  sudo git clone https://github.com/KatherLab/swarm-learning-hpe.git
fi

sudo chmod 777 -R "$DIR"