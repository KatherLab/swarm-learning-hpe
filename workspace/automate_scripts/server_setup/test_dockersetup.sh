#!/bin/bash

set -eu

echo "Testing Docker installation"
groupadd -f docker
sudo usermod -aG docker $USER
newgrp docker
docker run hello-world

if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi

echo "Docker installation successful."