#!/bin/sh
#set -eux

# Default values
workspace="odelia-breast-mri"
sentinel="172.24.4.67"

# Get the IP address of the current machine
ip_addr=$(ip addr show tun0 | awk '/inet / {print $2}' | cut -d'/' -f1)

if [ -z "$ip_addr" ]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi

# Get the name of this script, the directory it is in, and the current timestamp
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))
time_stamp=$(date +%Y%m%d_%H%M%S)

# Help function
help() {
  echo ""
  echo "Usage: sh $script_name -w <workspace> -s <sentinel>"
  echo ""
  echo "Options:"
  echo "-w <workspace>   The name of the workspace directory to use, $workspace by default for Odelia project."
  echo "-s <sentinel>    The IP address of the machine acting as the swarm sentinel, $sentinel by default."
  echo "-d <host_index>  Chose from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich] for your site"
  echo "-h               Show this help message."
  echo ""
  exit 1
}

# Process command line options
while getopts "w:s:d:h" opt; do
  case "${opt}" in
    w)
      workspace="${OPTARG}"
      ;;
    s)
      sentinel="${OPTARG}"
      ;;
    d)
      host_index="${OPTARG}"
      ;;
    h)
      help
      ;;
    *)
      help
      ;;
  esac
done

# Check that the required options are set
if [ -z "$workspace" ] || [ -z "$sentinel" ] || [ -z "$host_index" ]; then
  echo "Error: The -w and -s options are required."
  help
fi

# Run the SWOP container
sudo $script_dir/../../swarm_learning_scripts/run-swop --rm -d\
  --name=swop"$ip_addr" \
  --network=host-net \
  --sn-ip="$sentinel" \
  --sn-api-port=30304 \
  --usr-dir=workspace/"$workspace"/swop \
  --profile-file-name=swop_profile_"$ip_addr".yaml \
  --key=cert/swop-"$host_index"-key.pem \
  --cert=cert/swop-"$host_index"-cert.pem \
  --capath=cert/ca/capath \
  -e http_proxy= -e https_proxy= \
  --apls-ip="$sentinel" \
  -e SWOP_KEEP_CONTAINERS=True \
  -e SWARM_LOG_LEVEL=DEBUG \
  -e SL_DEVMODE_KEY=REVWTU9ERS0yMDI0LTAzLTE4 \

echo "SWOP container started"
echo "Use 'cklog --swop' to follow the logs of the SWOP node"
echo "Use 'stophpe --swop' to stop the SWOP node, or 'stophpe --all' to stop all running nodes"
