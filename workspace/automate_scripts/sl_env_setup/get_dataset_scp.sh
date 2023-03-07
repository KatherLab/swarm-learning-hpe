#!/bin/sh
set -eu
# Print usage information
usage() {
  echo "Usage: sh get_dataset_scp.sh -w WORKSPACE_NAME -s SENTINEL_IP [-h]"
  echo "Fetches the dataset from the sentinel node and unzips it into the workspace directory."
  echo ""
  echo "Options:"
  echo "  -w WORKSPACE_NAME    The name of the workspace."
  echo "  -s SENTINEL_IP       The IP address of the sentinel node."
  echo "  -h                   Show this help message."
  exit 1
}

# Parse command-line options
while getopts ":s:h" opt; do
    case $opt in
        h)
            usage
            ;;
        s)
            sentinel_ip=$OPTARG
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
if [[ -z "$sentinel_ip" ]]; then
  echo "Missing sentinel node ."
  usage
fi

# Fetch and unzip the dataset
sudo mkdir -p workspace/odelia-breast-mri/user/data-and-scratch/data
sudo mkdir -p workspace/marugoto_mri/user/data-and-scratch/data
sudo scp swarm@$sentinel_ip:/mnt/sda1/swarm-learning/radiology-dataset/odelia_dataset_only_sub.zip workspace/odelia-breast-mri/user/data-and-scratch/data
sudo scp swarm@$sentinel_ip:/mnt/sda1/swarm-learning/radiology-dataset/features_odelia_sub_imagenet.zip workspace/marugoto_mri/user/data-and-scratch/data
sudo scp -r swarm@$sentinel_ip:/mnt/sda1/swarm-learning/radiology-dataset/tables/ workspace/odelia-breast-mri/user/data-and-scratch/data
unzip workspace/odelia-breast-mri/user/data-and-scratch/data/odelia_dataset_only_sub.zip -d workspace/odelia-breast-mri/user/data-and-scratch/data
unzip workspace/marugoto_mri/user/data-and-scratch/data/features_odelia_sub_imagenet.zip -d workspace/marugoto_mri/user/data-and-scratch/data
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "Dataset fetched successfully."