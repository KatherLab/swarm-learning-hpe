#!/bin/sh
set -eu
# Print usage information
usage() {
    echo "Usage: $0 -w WORKSPACE" >&2
    echo "  -w WORKSPACE   path to the Swarm Learning workspace directory" >&2
    echo "  -h             display this help and exit" >&2
    exit 1
}

# Parse command-line options
while getopts ":w:h" opt; do
    case $opt in
        w)
            workspace=$OPTARG
            ;;
        h)
            usage
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

# Check that the workspace option is set
if [ -z "$workspace" ]; then
    echo "Error: WORKSPACE is a required option." >&2
    usage
fi

# Set up the SL CLI library volume
sl_cli_lib="sl-cli-lib"
if sudo docker volume list | grep -q "$sl_cli_lib"; then
    sudo docker volume rm -f "$sl_cli_lib"
fi
sudo docker volume create "$sl_cli_lib"
sudo docker container create --name helper -v "$sl_cli_lib":/data hello-world
sudo docker cp -L "${0%/*}"/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data
sudo docker rm helper

# Create a Docker network for the host
ip_addr=$(ip addr show tun0 | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')
host_network="host-$ip_addr-net"
if sudo docker network list | grep -q "$host_network"; then
    sudo docker network rm "$host_network"
fi
sudo docker network create "$host_network" #--subnet="$ip_addr/"24
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "SL CLI library volume created successfully."