#!/bin/sh
#set -eux

# Default values
workspace="swag_latent_diffusion"
host=""
sentinel="192.168.33.100"
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help()
{
   echo ""
   echo "Usage: sh ${script_name} -i <host> -s <sentinel_ip> -d <host_index>"
   echo ""
   echo "Options:"
   echo "-s    The sentinel IP address, $sentinel by default"
   echo "-d    The host index, chose from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich] for your site"
   echo "-h    Show help"
   echo ""
   exit 1
}
# Process command options
while getopts "s:d:h" opt
do
   case "$opt" in
      s ) sentinel="$OPTARG" ;;
      d ) host_index="$OPTARG" ;;
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

ip_addr=$(ip addr show eno1 | awk '/inet / {print $2}' | cut -d'/' -f1)

if [ -z "$ip_addr" ]; then
    echo "Error: eno1 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
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
     -d --rm \
     --name=sn_node \
     --network=host-net \
     --host-ip="$ip_addr" \
     "$sn_command" \
     --sn-p2p-port=30303 \
     --sn-api-port=30304 \
     --key=cert/sn-"$host_index"-key.pem \
     --cert=cert/sn-"$host_index"-cert.pem \
     --capath=cert/ca/capath \
     --apls-ip="$sentinel" \
     -e SWARM_LOG_LEVEL=DEBUG \


echo "SN node started, waiting for the network to be ready"
echo "Use 'cklog --sn' to follow the logs of the SN node"
echo "Use 'stophpe --sn' to stop the SN node, or 'stophpe --all' to stop all running nodes"
