#!/bin/sh
set -eu
# Print usage information
usage() {
  echo "Usage: sh get_dataset_gdown.sh -w WORKSPACE_NAME -s SENTINEL_IP [-h]"
  echo "Fetches the dataset from the google drive and unzips it into the workspace directory. Will be faster comparing to scp."
  echo ""
  echo "Options:"
  echo "  -h                   Show this help message."
  exit 1
}

# Parse command-line options
while getopts ":h" opt; do
    case $opt in
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



# Fetch and unzip the dataset
sudo mkdir -p workspace/odelia-breast-mri/user/data-and-scratch/data
sudo mkdir -p workspace/marugoto_mri/user/data-and-scratch/data
#sudo scp swarm@$sentinel_ip:/mnt/sda1/swarm-learning/radiology-dataset/odelia_dataset_only_sub.zip workspace/odelia-breast-mri/user/data-and-scratch/data
#sudo scp swarm@$sentinel_ip:/mnt/sda1/swarm-learning/radiology-dataset/features_odelia_sub_imagenet.zip workspace/marugoto_mri/user/data-and-scratch/data
#sudo scp -r swarm@$sentinel_ip:/mnt/sda1/swarm-learning/radiology-dataset/tables/ workspace/odelia-breast-mri/user/data-and-scratch/data
gdown --fuzzy https://drive.google.com/drive/folders/1clbK91sQv8bYtGAhC9AmZAbgVo473yrz?usp=share_link -O workspace/odelia-breast-mri/user/data-and-scratch/data --folder
gdown --fuzzy https://drive.google.com/file/d/1igy2emjH4HmLf_KNbv-4aLv5_8GeFoPq/view?usp=share_link -O workspace/marugoto_mri/user/data-and-scratch/data
gdown --fuzzy https://drive.google.com/file/d/1HKjqbALJgZCqLeNJ-_xKk0lEDNcJV_5Z/view?usp=share_link -O workspace/odelia-breast-mri/user/data-and-scratch/data


unzip workspace/odelia-breast-mri/user/data-and-scratch/data/odelia_dataset_only_sub.zip -d workspace/odelia-breast-mri/user/data-and-scratch/data
unzip workspace/marugoto_mri/user/data-and-scratch/data/features_odelia_sub_imagenet.zip -d workspace/marugoto_mri/user/data-and-scratch/data
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "Dataset fetched successfully."