#!/bin/sh
set -eux

# Print usage information
usage() {
    echo "Usage: $0 [-w workspace]" >&2
    echo "  -h             display this help and exit" >&2
    echo "  -w workspace   optionally specify the workspace to use" >&2
    exit 1
}

# Initialize default values
workspace=""
sl_cli_lib="sl-cli-lib" # Default

# Parse command-line options
while getopts ":hw:" opt; do
    case $opt in
        h)
            usage
            ;;
        w)
            workspace=$OPTARG
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            usage
            ;;
        :)
            echo "Option -$OPTARG requires an argument." >&2
            usage
            ;;
    esac
done

# Set sl_cli_lib based on workspace if provided
case $workspace in
    swag-latent-diffusion)
        sl_cli_lib="sl-cli-lib-latent-diffusion"
        ;;
    score-based-model)
        sl_cli_lib="sl-cli-lib-sbm"
        ;;
    *)
        sl_cli_lib="sl-cli-lib" # Default
        ;;
esac

# Volume setup
sudo docker volume rm -f "$sl_cli_lib" || true
sudo docker volume create "$sl_cli_lib"
sudo docker container create --name helper -v "$sl_cli_lib":/data hello-world

# Copy swarmlearning wheel
sudo docker cp -L /opt/hpe/swarm-learning-hpe/sllib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data

# Copy requirements.txt only if workspace is given
if [ -n "$workspace" ]; then
    req_path="/opt/hpe/swarm-learning-hpe/workspace/${workspace}/requirements.txt"
    if [ -f "$req_path" ]; then
        sudo docker cp -L "$req_path" helper:/data
    else
        echo "Warning: requirements.txt not found for workspace '$workspace'"
    fi
fi

# Clean up helper container
sudo docker rm helper

# Docker network setup
host_network="host-net"
if sudo docker network list | grep -q "$host_network"; then
    sudo docker network rm "$host_network"
fi

sudo docker network create "$host_network"

echo "SL CLI library volume created successfully."
