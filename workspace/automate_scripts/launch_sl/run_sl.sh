#!/bin/sh

set -eu

# Get the directory containing this script
script_dir=$(cd "$(dirname "$0")" && pwd)
# Define a help function
show_help() {
  echo "Usage: $(basename "$0") [-w WORKSPACE]"
  echo ""
  echo "Launch a swarm learning container"
  echo ""
  echo "Options:"
  echo "  -w WORKSPACE    The name of the workspace directory to use (default: none)"
  echo "  -r              Remove any stopped containers"
  echo "  -h              Show this help message"
  echo ""
  exit 0
}

# Set default values for options
workspace=""
remove_stopped_containers=false

# Process command line options
while getopts "rw:h" opt; do
  case ${opt} in
    r ) remove_stopped_containers=true ;; # Remove any stopped containers
    w ) workspace=${OPTARG} ;;
    h ) show_help ;;
    \? ) show_help ;;
    : ) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done
# Remove any stopped containers if specified
if [ "$remove_stopped_containers" = true ]; then
  docker rm $(docker ps --filter status=exited -q)
fi
ip_addr=$(ip addr show eno1 | awk '/inet / {print $2}' | cut -d'/' -f1)

if [ -z "$ip_addr" ]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi
echo $workspace
# Set the Docker image based on the workspace
if [ "$workspace" = "marugoto_mri" ]; then
  ml_image="user-env-marugoto-swop"
elif [ "$workspace" = "odelia-breast-mri" ]; then
  ml_image="user-env-pyt1.13-swop"
else
  echo "Error: invalid workspace specified."
  exit 1
fi

# Launch the swarm learning container with the specified options
"$script_dir"/../../swarm_learning_scripts/run-sl \
  --name=sl2 \
  --host-ip="$ip_addr" \
  --sn-ip="$ip_addr"\
  --sn-api-port=30304 \
  --sl-fs-port=16000 \
  --key=/opt/hpe/swarm-learning-hpe/cert/sl-TUD-key.pem \
  --cert=/opt/hpe/swarm-learning-hpe/cert/sl-TUD-cert.pem \
  --capath=/opt/hpe/swarm-learning-hpe/cert/ca/capath \
  --ml-it \
  --ml-image="$ml_image" \
  --ml-name=ml2 \
  --ml-w=/tmp/test \
  --ml-entrypoint=python3 \
  --ml-cmd=model/main.py \
  --ml-v=workspace/"$workspace"/model:/tmp/test/model \
  --ml-v=workspace/"$workspace"/user/data-and-scratch/data:/platform/data \
  --ml-e MODEL_DIR=model \
  --ml-e MAX_EPOCHS=5 \
  --ml-e MIN_PEERS=1 \
  --ml-e https_proxy= \
  --ml-user 0:0 \
  --apls-ip="$ip_addr"


