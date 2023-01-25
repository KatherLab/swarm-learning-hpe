#!/bin/sh
set -eux

echo Install Docker
curl -fsSL get.docker.com | sudo sh

echo Test Docker installation:
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

echo Create Folder / Working directory for Swarm Learning:
sudo mkdir /opt/hpe
cd /opt/hpe
sudo git clone https://github.com/KatherLab/swarm-learning-hpe.git