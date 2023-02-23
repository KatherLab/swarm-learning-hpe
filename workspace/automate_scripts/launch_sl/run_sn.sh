#!/bin/sh
set -eux

# Default values
workspace=""
host=""
sentinel=""
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help()
{
   echo ""
   echo "Usage: sh ${script_name} -w <workspace> -i <host> -s <sentinel_ip>"
   echo ""
   echo "Options:"
   echo "-w    The workspace name"
   echo "-i    The host IP address"
   echo "-s    The sentinel IP address"
   echo "-h    Show help"
   echo ""
   exit 1
}

# Process command options
while getopts "w:s:h" opt
do
   case "$opt" in
      w ) workspace="$OPTARG" ;;
      s ) sentinel="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Check required options are set
if [ -z "$workspace" ] || [ -z "$sentinel" ]
then
   echo "Error: missing required options"
   help
fi

ip_addr=$(ip addr show | awk '/inet 10\./{print $2}' | cut -d'/' -f1)

if [ -z "$ip_addr" ]
then
   echo "Error: invalid host IP address"
   exit 1
fi

if [ $ip_addr = $sentinel ]
then
   echo "This host is a sentinel node and will be used for initiating the cluster"
   sn_command="--sentinel"
else
   echo "This host is not a sentinel node and will not be used for initiating the cluster, only as a swarm network node"
   sn_command="--sentinel-ip=$sentinel"
fi

sudo $script_dir/../../swarm_learning_scripts/run-sn \
     -it --rm \
     --name=sn"$ip_addr" \
     --network=host-"$ip_addr"-net \
     --host-ip="$ip_addr" \
     "$sn_command" \
     --sn-p2p-port=30303 \
     --sn-api-port=30304 \
     --key=workspace/"$workspace"/cert/sn-"$ip_addr"-key.pem \
     --cert=workspace/"$workspace"/cert/sn-"$ip_addr"-cert.pem \
     --capath=workspace/"$workspace"/cert/ca/capath \
     --apls-ip="$sentinel" \
     --apls-port=5000
