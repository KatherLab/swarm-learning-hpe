#!/bin/sh
set -e

# Function to print help text
print_help() {
   echo "Usage: $0 -s <sentinel_host> -w <workspace> -n <num_peers> -e <num_epochs>"
   echo "Options:"
   echo "  -s: Hostname or IP address of the Sentinel node"
   echo "  -w: Name of the workspace directory"
   echo "  -n: Number of minimum peers"
   echo "  -e: Number of maximum epochs"
   echo "  -d: Hose index"
   echo "  -h: Print this help text"
}

# Parse command line options
while getopts ":s:w:d:n:e:h" opt; do
   case $opt in
      s) SENTINEL_HOST="$OPTARG" ;;
      w) WORKSPACE="$OPTARG" ;;
      n) NUM_PEERS="$OPTARG" ;;
      e) NUM_EPOCHS="$OPTARG" ;;
      d) HOST_INDEX="$OPTARG" ;;
      h) print_help; exit 0 ;;
      \?) echo "Invalid option -$OPTARG" >&2; print_help; exit 1 ;;
      :) echo "Option -$OPTARG requires an argument." >&2; print_help; exit 1 ;;
   esac
done

# Check required options
if [ -z "$SENTINEL_HOST" ] || [ -z "$WORKSPACE" ] || [ -z "$NUM_PEERS" ] || [ -z "$HOST_INDEX" ];
then
   echo "Error: Missing required option(s)." >&2
   print_help
   exit 1
fi

# Get the IP address of this node
IP_ADDR=$(ip addr show tun0 2>/dev/null | grep 'inet ' | awk '{print $2}' | cut -f1 -d'/')

if [ -z "$IP_ADDR" ]; then
    echo "Error: tun0 interface not found. Please connect to the VPN first. Use script setup_vpntunnel.sh"
    exit 1
fi

# Set up directories
WORKSPACE_DIR="/opt/hpe/swarm-learning-hpe/"
cd $WORKSPACE_DIR

sudo cp workspace/"$WORKSPACE"/swop/swop_profile.yaml workspace/"$WORKSPACE"/swop/swop_profile_"$IP_ADDR".yaml
sudo cp workspace/"$WORKSPACE"/swci/swci-init_ori workspace/"$WORKSPACE"/swci/swci-init_pre
sudo cp workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_ori.yaml workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_pre.yaml
sudo cp workspace/"$WORKSPACE"/swci/taskdefs/user_env_build_task_ori.yaml workspace/"$WORKSPACE"/swci/taskdefs/user_env_build_task_pre.yaml
sudo cp workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_local_compare_ori.yaml workspace/"$WORKSPACE"/swci/taskdefs/swarm_task_local_compare_pre.yaml


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
   sed -i "s+<HOST-INDEX>+$HOST_INDEX+g" "$file"
done
# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "Replacement completed successfully."