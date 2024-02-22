#!/bin/sh
# set -eu
host_index=""

# Help function
help() {
   echo ""
   echo "Generate alias for swarm learning scripts"
   echo "-d    The host index, choose from [TUD, Ribera, VHIO, Radboud, UKA, Utrecht, Mitera, Cambridge, Zurich] for your site"
   echo "-h    Display this help message."
   exit 1
}

# Process command options
while getopts "d:h" opt; do
   case "$opt" in
      h ) help ;;
      d ) host_index="$OPTARG" ;;
      * ) help ;;
   esac
done

# Check required options are set
if [ -z "$host_index" ]; then
   echo "Error: missing host index"
   help
fi

# Define aliases
alias runsn="sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_sn.sh -d $host_index"
alias runswop="sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_swop.sh -d $host_index"
alias runswci="sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_swci.sh -d $host_index"
alias runsl="sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_sl.sh -d $host_index"
alias cklog="sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/check_latest_log.sh -d $host_index"
alias stophpe="sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/stopswarm.sh -d $host_index"

# Inform user about manual sourcing
echo "Aliases generated successfully. Please source your shell configuration file (e.g., ~/.bashrc or ~/.zshrc) to apply changes."

# Commented out the automatic sourcing and error check
# . ~/.bashrc
# if [ $? -ne 0 ]; then
#     echo "An error occurred while running the script. Please check the output above for more details."
#     exit 1
# fi
