#!/bin/bash
set -eu

# Define script name and directory
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help()
{
    echo "Usage: $script_name [-h]"
    echo ""
    echo "Options:"
    echo "  -h, --help   Show help message and exit"
    echo ""
    exit 1
}

# Process command line options
while [ $# -gt 0 ]
do
    key="$1"
    case $key in
        -h|--help)
            help
            shift
            ;;
        *)
            echo "Unknown option: $1"
            help
            shift
            ;;
    esac
done

# Remove cert folder if it exists
[ -d "cert" ] && sudo rm -r "cert"

# Check if lib folder exists, if exists then remove the folder
[ -d "lib" ] && sudo rm -r "lib"

# Clean up images
image_names=("hub.myenterpriselicense.hpe.com/hpe/swarm-learning/*" "user-env-pyt1.13-swop" "user-env-marugoto-swop")
for image in "${image_names[@]}"
do
    containers=$(sudo docker ps -a -q -f ancestor="$image")
    [ ! -z "$containers" ] && sudo docker rm $containers
    sudo docker rmi -f $image || true
done

# Cleanup volume
sudo docker volume rm sl-cli-lib || true

echo "Old SL cleaned up successfully."
