#!/bin/sh
set -eux
# Print usage information
usage() {
    echo "Usage: $0 [-w workspace]" >&2
    echo "  -h             display this help and exit" >&2
    echo "  -w workspace   specify the workspace to use" >&2
    exit 1
}

# Initialize default values
workspace=""
sl_cli_lib="sl-cli-lib" # Default value

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

# Check and set sl_cli_lib based on workspace argument
case $workspace in
    swag-latent-diffusion)
        sl_cli_lib="sl-cli-lib-latent-diffusion"
        ;;
    score-based-model)
        sl_cli_lib="sl-cli-lib-sbm"
        ;;
    *)
        sl_cli_lib="sl-cli-lib" # Default case
        ;;
esac

# Ensure a workspace is provided
if [ -z "$workspace" ]; then
    echo "Workspace not specified." >&2
    usage
fi

# Rest of the script where $sl_cli_lib is now set based on the workspace argument
sudo docker volume rm -f "$sl_cli_lib" || true
sudo docker volume create "$sl_cli_lib"
sudo docker container create --name helper -v "$sl_cli_lib":/data hello-world
sudo docker cp -L /opt/hpe/swarm-learning-hpe/sllib/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data

# Adjust the docker cp command to use the specified workspace for requirements.txt
if [ -n "$workspace" ]; then
    sudo docker cp -L "/opt/hpe/swarm-learning-hpe/workspace/${workspace}/requirements.txt" helper:/data
fi

sudo docker rm helper

# Docker network setup and error handling remain unchanged
host_network="host-net"
if sudo docker network list | grep -q "$host_network"; then
    sudo docker network rm "$host_network"
fi
sudo docker network create "$host_network" #--subnet="$ip_addr/"24
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "SL CLI library volume created successfully."
