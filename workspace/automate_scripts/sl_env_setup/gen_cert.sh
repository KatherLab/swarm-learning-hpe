#!/bin/sh
set -eux

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
ip_addr=$(ip addr show tun0 | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

# Generate SSL certificates
script_dir=$(realpath $(dirname "${0}"))

sudo "$script_dir"/../../swarm_learning_scripts/gen-cert -i "$host_index"
