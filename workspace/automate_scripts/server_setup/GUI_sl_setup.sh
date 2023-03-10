#!/bin/sh
set -eu

INSTALLER_PATH="./license_and_softwares/HPE_SWARM_LEARNING_INSTALLER_LINUX_Q2V41-11036"

# Check if the installer has already been installed
if dpkg -l | grep -q "hpe-swarm-learning"; then
    echo "HPE Swarm Learning is already installed."
    exit 0
fi

# Make the installer executable and run it with sudo
sudo chmod +x "$INSTALLER_PATH"
sudo "$INSTALLER_PATH"
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "HPE Swarm Learning installed successfully."