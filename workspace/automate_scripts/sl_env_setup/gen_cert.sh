#!/bin/sh
set -eu

# Help function
help()
{
   echo ""
   echo "Usage: $0 -w <workspace>"
   echo ""
   echo "Generate SSL certificates for HPE Swarm Learning nodes."
   echo ""
   echo "Options:"
   echo "-i <host_index>  Required. The host index for SSL certificates that will be generated."
   echo "-h               Display this help message."
   exit 1
}

# Process command options
while getopts "i:h" opt
do
   case "$opt" in
      i ) host_index="$OPTARG" ;;
      h ) help ;;
      * ) help ;;
   esac
done

# Checks
if [ -z "$host_index" ]
then
   echo "Error: The '-i' option is required."
   help
fi

# Get IP address
ip_addr=$(ip addr show eno1 | awk '/inet / {print $2}' | cut -d'/' -f1)

if [[ -z "$ip_addr" ]]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi

# Generate SSL certificates
script_dir=$(realpath $(dirname "${0}"))

sudo "$script_dir"/../../swarm_learning_scripts/gen-cert -i "$host_index"
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "SSL certificates generated successfully."
