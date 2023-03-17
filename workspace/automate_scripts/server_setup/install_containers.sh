#!/bin/sh
set -eu

# Define the directory of the script and the parent directory
script_dir=$(realpath $(dirname "${0}"))
workspace_dir="$script_dir"/../

# Help function
help() {
  echo "Usage: sh script.sh [OPTIONS]"
  echo "This script downloads and sets up the HPE Swarm Learning environment."
  echo ""
  echo "Options:"
  echo "-h, --help        show help information"
  exit 0
}

# Parse command line options
while [ $# -gt 0 ]; do
  case "$1" in
    -h|--help)
      help
      ;;
    *)
      echo "Unknown option: $1"
      help
      ;;
  esac
  shift
done

# Login to HPE hub
echo "Login to HPE hub, please ask TUD maintainer for the username and password since Github doesn't allow to store the credentials in the code."´
# Prompt the user for the email and password
read -p "Please enter your email: " user_email
read -s -p "Please enter your password: " user_password
echo
hub_url="hub.myenterpriselicense.hpe.com"
# Use the provided email and password for the Docker login command
sudo docker login -u "$user_email" -p "$user_password" "$hub_url"

# Pull images
echo "Download Swarm Network (SN) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sn:1.2.0
echo "Download Swarm Learning (SL) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/sl:1.2.0
echo "Download Swarm Learning Command Interface (SWCI) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/swci:1.2.0
echo "Download Swarm Operator (SWOP) Node"
sudo docker pull hub.myenterpriselicense.hpe.com/hpe_eval/swarm-learning/swop:1.2.0

# Extract files
sudo tar -xf $script_dir/license_and_softwares/HPE_SWARM_LEARNING_DOCS_EXAMPLES_SCRIPTS_Q2V41-11033.tar.gz -C /opt/hpe/swarm-learning-hpe/
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "HPE Swarm Learning installed successfully."