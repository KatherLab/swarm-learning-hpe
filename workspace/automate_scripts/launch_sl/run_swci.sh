#!/bin/sh
#set -eux

# Default values
workspace="score-based-model"
sentinel="100.125.38.128"

# Print usage message
usage() {
  echo "Usage: $0 [-w <workspace>] [-s <sentinel IP>] [-h]"
  echo "Starts the SWCI container on the specified sentinel node."
  echo ""
  echo "Options:"
  echo "  -w <workspace>   Workspace directory, $workspace by default"
  echo "  -s <sentinel IP> IP address of the sentinel node, $sentinel by default"
  echo "  -d <host_index>  Chose from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich] for your site"
  echo "  -h               Show this help message"
  exit 1
}


# Parse command line arguments
while getopts ":w:s:d:h" opt; do
  case $opt in
    w)
      workspace="$OPTARG"
      ;;
    s)
      sentinel="$OPTARG"
      ;;
    d)
      host_index="$OPTARG"
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

# Verify that sentinel IP is set
if [ -z "$sentinel" ] || [ -z "$host_index" ]; then
  echo "Error: Sentinel IP address must be specified using -i, host_index is also required" >&2
  usage
fi

# Check if this host is the sentinel
ip_addr=$(ip addr show tailscale0 | awk '/inet / {print $2}' | cut -d'/' -f1)

if [ -z "$ip_addr" ]; then
    echo "Error: tailscale0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi
#if [ "$ip_addr" != "$sentinel" ]; then
  #echo "Error: This host is not the sentinel node" >&2
  #exit 1
#fi

# Set script variables
script_name=$(basename "$0")
script_dir=$(realpath "$(dirname "$0")")
time_stamp=$(date +%Y%m%d_%H%M%S)

# Print configuration info
echo "Starting SWCI container..."
echo "  Workspace: $workspace"
echo "  Sentinel IP: $sentinel"
echo "  Timestamp: $time_stamp"

# Update configuration files
cp "workspace/$workspace/swci/taskdefs/swarm_task_pre.yaml" "workspace/$workspace/swci/taskdefs/swarm_task.yaml"
cp "workspace/$workspace/swci/taskdefs/user_env_build_task_pre.yaml" "workspace/$workspace/swci/taskdefs/user_env_build_task.yaml"
cp "workspace/$workspace/swci/taskdefs/swarm_task_local_compare_pre.yaml" "workspace/$workspace/swci/taskdefs/swarm_task_local_compare.yaml"
cp "workspace/$workspace/swci/swci-init_pre" "workspace/$workspace/swci/swci-init"
sed -i "s+<TIME_STAMP>+$time_stamp+g" "workspace/$workspace/swci/taskdefs/swarm_task.yaml" "workspace/$workspace/swci/taskdefs/user_env_build_task.yaml" "workspace/$workspace/swci/taskdefs/swarm_task_local_compare.yaml" "workspace/$workspace/swci/swci-init"

# Start the SWCI container
sudo "$script_dir/../../swarm_learning_scripts/run-swci" \
  -d --rm --name="swci-$ip_addr" \
  --network="host-net" \
  --usr-dir="workspace/$workspace/swci" \
  --init-script-name="swci-init" \
  --key="cert/swci-$host_index-key.pem" \
  --cert="cert/swci-$host_index-cert.pem" \
  --capath="cert/ca/capath" \
  -e "http_proxy=" -e "https_proxy=" --apls-ip="$sentinel" \
  -e "SWCI_RUN_TASK_MAX_WAIT_TIME=5000" \
  -e "SWCI_GENERIC_TASK_MAX_WAIT_TIME=5000" \
  -e SWARM_LOG_LEVEL=DEBUG \
   -e SL_DEVMODE_KEY=REVWTU9ERS0yMDI0LTAzLTI4 \


echo "SWCI container started successfully"
echo "Use 'cklog --swci' to follow the logs of the SWCI node"
echo "Use 'stophpe --swci' to stop the SWCI node, or 'stophpe --all' to stop all running nodes"