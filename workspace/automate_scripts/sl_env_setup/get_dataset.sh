#!/bin/sh
set -eux
# Print usage information
usage() {
  echo "Usage: sh get_dataset.sh -w WORKSPACE_NAME -s SENTINEL_IP [-h]"
  echo "Fetches the dataset from the sentinel node and unzips it into the workspace directory."
  echo ""
  echo "Options:"
  echo "  -w WORKSPACE_NAME    The name of the workspace."
  echo "  -s SENTINEL_IP       The IP address of the sentinel node."
  echo "  -h                   Show this help message."
  exit 1
}

# Parse command-line options
while getopts ":w:s:h" opt; do
    case $opt in
        w)
            workspace=$OPTARG
            ;;
        h)
            usage
            ;;
        s)
            sentinal_ip=$OPTARG
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

# Check that required options were provided
if [[ -z "$workspace" || -z "$sentinal_ip" ]]; then
  echo "Missing required options."
  usage
fi

# Fetch and unzip the dataset
scp -r swarm@$sentinal_ip:/mnt/sda1/swarm-learning/radiology-dataset/odelia_dataset_unilateral_256x256x32.zip workspace/$workspace/user/data-and-scratch/data
unzip workspace/$workspace/user/data-and-scratch/data/odelia_dataset_unilateral_256x256x32.zip -d workspace/$workspace/user/data-and-scratch/data
