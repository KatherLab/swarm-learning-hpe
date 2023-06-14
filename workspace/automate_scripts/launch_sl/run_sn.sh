#!/bin/sh
set -eu

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
   echo "Usage: sh ${script_name} -i <host> -s <sentinel_ip> -d <host_index>"
   echo ""
   echo "Options:"
   echo "-i    The host IP address"
   echo "-s    The sentinel IP address"
   echo "-d    The host index, chose from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich] for your site"
   echo "-l    The IP adress of the License Server"
   echo "-h    Show help"
   echo ""
   exit 1
}
# Process command options
while getopts "s:d:l:h" opt
do
   case "$opt" in
      s ) sentinel="$OPTARG" ;;
      d ) host_index="$OPTARG" ;;
      l ) license="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Check required options are set
if [ -z "$sentinel" ] || [ -z "$host_index" ]
then
   echo "Error: missing required options"
   help
fi

ip_addr=$(ip addr show tun0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

if [[ -z "$ip_addr" ]]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi

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
     --name=sn_node \
     --network=host-net \
     --host-ip="$ip_addr" \
     "$sn_command" \
     --sn-p2p-port=30303 \
     --sn-api-port=30304 \
     --key=cert/sn-"$host_index"-key.pem \
     --cert=cert/sn-"$host_index"-cert.pem \
     --capath=cert/ca/capath \
     --apls-ip="$license" \
     --apls-port=5000
