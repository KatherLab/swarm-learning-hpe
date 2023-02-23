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
   echo "-w <workspace>   Required. The directory where SSL certificates will be generated."
   echo "-h               Display this help message."
   exit 1
}

# Process command options
while getopts "w:h" opt
do
   case "$opt" in
      w ) workspace="$OPTARG" ;;
      h ) help ;;
      * ) help ;;
   esac
done

# Checks
if [ -z "$workspace" ]
then
   echo "Error: The '-w' option is required."
   help
fi

# Get IP address
ip_addr=$(ip addr show | awk '/inet 10\./{print $2}' | cut -d'/' -f1)

# Generate SSL certificates
script_dir=$(realpath $(dirname "${0}"))
sudo "$script_dir"/../../swarm_learning_scripts/gen-cert -e "$workspace" -i "$ip_addr"
