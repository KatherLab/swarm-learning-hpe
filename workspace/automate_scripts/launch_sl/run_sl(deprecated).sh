#!/bin/bash

set -euo pipefail

# Get the directory containing this script
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Remove any stopped containers
docker rm $(docker ps --filter status=exited -q) || true

# Define a help function
show_help() {
  echo "Usage: $(basename "$0") [-w WORKSPACE]"
  echo ""
  echo "Launch a swarm learning container"
  echo ""
  echo "Options:"
  echo "  -w WORKSPACE    The name of the workspace directory to use (default: none)"
  echo "  -h              Show this help message"
  echo ""
  exit 0
}

# Process command line options
while getopts ":w:h" opt; do
  case ${opt} in
    w ) workspace=${OPTARG} ;;
    h ) show_help ;;
    \? ) show_help ;;
    : ) echo "Option -$OPTARG requires an argument." >&2; exit 1 ;;
  esac
done

# Launch the swarm learning container with the specified options
"$script_dir"/../../swarm_learning_scripts/run-sl \
  --name=sl1 \
  --host-ip=192.168.33.102 \
  --sn-ip=192.168.33.102 \
  --sn-api-port=30304 \
  --sl-fs-port=16000 \
  --key=/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/sl-192.168.33.102-key.pem \
  --cert=/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/sl-192.168.33.102-cert.pem \
  --capath=workspace/"$workspace"/cert/ca/capath \
  --ml-it \
  --ml-image=swarm \
  --ml-name=ml1 \
  --ml-w=/tmp/test \
  --ml-entrypoint=python3 \
  --ml-cmd=model/main.py \
  --ml-v=workspace/"$workspace"/model:/tmp/test/model \
  --ml-e MODEL_DIR=model \
  --ml-e MAX_EPOCHS=5 \
  --ml-e MIN_PEERS=2 \
  --ml-e https_proxy= \
  --apls-ip=192.168.33.102