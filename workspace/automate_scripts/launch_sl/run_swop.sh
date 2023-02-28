#!/bin/sh

set -eux

# Get the IP address of the current machine
ip_addr=$(ip addr show tun0 | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

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
  echo "-w <workspace>   The name of the workspace directory to use."
  echo "-s <sentinel>    The IP address of the machine acting as the swarm sentinel."
  echo "-h               Show this help message."
  echo ""
  exit 1
}

# Process command line options
while getopts "w:s:h" opt; do
  case "${opt}" in
    w)
      workspace="${OPTARG}"
      ;;
    s)
      sentinel="${OPTARG}"
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
if [ -z "$workspace" ] || [ -z "$sentinel" ]; then
  echo "Error: The -w and -s options are required."
  help
fi

# Run the SWOP container
sudo $script_dir/../../swarm_learning_scripts/run-swop -it --rm \
  --name=swop"$ip_addr" \
  --network=host-"$ip_addr"-net \
  --usr-dir=workspace/"$workspace"/swop \
  --profile-file-name=swop_profile_"$ip_addr".yaml \
  --key=workspace/"$workspace"/cert/swop-"$ip_addr"-key.pem \
  --cert=workspace/"$workspace"/cert/swop-"$ip_addr"-cert.pem \
  --capath=workspace/"$workspace"/cert/ca/capath \
  -e http_proxy= -e https_proxy= \
  --apls-ip="$sentinel" \
  --apls-port=5000 \
  -e SWOP_KEEP_CONTAINERS=True
