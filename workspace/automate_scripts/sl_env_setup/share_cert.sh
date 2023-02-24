#!/bin/sh
set -eux

# Obtain IP address of current machine
ip_addr=$(ip addr show tun0 | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

# Obtain script name and directory
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help() {
    echo ""
    echo "Usage: $script_name -t <target_host> -w <workspace>"
    echo "Options:"
    echo "  -t : The target host to copy the certificate from."
    echo "  -w : The workspace to copy the certificate to."
    echo ""
    exit 1
}

# Process command options
while getopts "t:w:h?" opt; do
    case "$opt" in
        t ) target_host="$OPTARG" ;;
        w ) workspace="$OPTARG" ;;
        h ) help ;;
        ? ) help ;;
    esac
done

# Check if required parameters are provided
if [ -z "$target_host" ] || [ -z "$workspace" ]; then
    echo "Some or all of the parameters are empty"
    help
fi

# Copy the certificate from target_host to current machine's workspace
# Modify the following line to match the path to the certificate on the target host
# Modify the user name to match the user name on the target host, for example change swarm here to root
sudo scp swarm@"$target_host":/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/ca/capath/ca-"$target_host"-cert.pem /opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/ca/capath
sudo scp /opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/ca/capath/ca-"$ip_addr"-cert.pem swarm@"$target_host":/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/ca/capath/
