#!/bin/sh
set -eux

# Function to print help text
print_help() {
   echo "Usage: $0 -s <sentinel_host> -w <workspace> -n <num_peers> -e <num_epochs>"
   echo "Options:"
   echo "  -s: Hostname or IP address of the Sentinel node"
   echo "  -w: Name of the workspace directory"
   echo "  -n: Number of minimum peers"
   echo "  -e: Number of maximum epochs"
   echo "  -h: Print this help text"
}

# Parse command line options
while getopts ":s:w:n:e:h" opt; do
   case $opt in
      s) SENTINEL_HOST="$OPTARG" ;;
      w) WORKSPACE="$OPTARG" ;;
      n) NUM_PEERS="$OPTARG" ;;
      e) NUM_EPOCHS="$OPTARG" ;;
      h) print_help; exit 0 ;;
      \?) echo "Invalid option -$OPTARG" >&2; print_help; exit 1 ;;
      :) echo "Option -$OPTARG requires an argument." >&2; print_help; exit 1 ;;
   esac
done

# Check required options
if [ -z "$SENTINEL_HOST" ] || [ -z "$WORKSPACE" ] || [ -z "$NUM_PEERS" ] || [ -z "$NUM_EPOCHS" ]
then
   echo "Error: Missing required option(s)." >&2
   print_help
   exit 1
fi

# Get the IP address of this node
IP_ADDR=$(ip addr show | awk '/inet 10\./{print $2}' | cut -d'/' -f1)

# Set up directories
SCRIPT_NAME=$(basename "${0}")
SCRIPT_DIR=$(realpath $(dirname "${0}"))
WORKSPACE_DIR="/opt/hpe/swarm-learning-hpe/"
cd $WORKSPACE_DIR

cp workspace/"$WORKSPACE"/swop/swop_profile.yaml workspace/"$WORKSPACE"/swop/swop_profile_"$IP_ADDR".yaml
cp workspace/"$WORKSPACE"/swci/swci-init_ori workspace/"$WORKSPACE"/swci/swci-init_pre
cp workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_ori.yaml workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_pre.yaml
cp workspace/"$WORKSPACE"/swci/taskdefs/user_env_build_task_ori.yaml workspace/"$WORKSPACE"/swci/taskdefs/user_env_build_task_pre.yaml
cp workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_local_compare_ori.yaml workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_local_compare_pre.yaml


# Loop over the files to update
for file in workspace/"$WORKSPACE"/swop/swop_profile_"$IP_ADDR".yaml workspace/"$WORKSPACE"/swci/swci-init_pre workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_pre.yaml workspace/"$WORKSPACE"/swci/taskdefs/user_env_build_task_pre.yaml workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_local_compare_pre.yaml
do
   # Replace placeholders with values
   sed -i "s+<CURRENT-PATH>+$(pwd)+g" "$file"
   sed -i "s+<SN-IPADDRESS>+$SENTINEL_HOST+g" "$file"
   sed -i "s+<HOST-IPADDRESS>+$IP_ADDR+g" "$file"
   sed -i "s+<MODULE-NAME>+$WORKSPACE+g" "$file"
   sed -i "s+<NUM-MIN_PEERS>+$NUM_PEERS+g" "$file"
   sed -i "s+<NUM-MAX_EPOCHS>+$NUM_EPOCHS+g" "$file"
done