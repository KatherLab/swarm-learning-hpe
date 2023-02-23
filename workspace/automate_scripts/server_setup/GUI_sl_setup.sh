#!/bin/sh
set -eux

INSTALLER_PATH="./license_and_softwares/HPE_SWARM_LEARNING_INSTALLER_LINUX_Q2V41-11036"

# Check if the installer has already been installed
if dpkg -l | grep -q "hpe-swarm-learning"; then
    echo "HPE Swarm Learning is already installed."
    exit 0
fi

# Make the installer executable and run it with sudo
sudo chmod +x "$INSTALLER_PATH"
sudo "$INSTALLER_PATH"
