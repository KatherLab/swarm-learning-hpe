#!/bin/bash
#set -eu

# Help function
help()
{
   echo ""
   echo "Generate alias for swarm learning scripts"
   echo ""
   echo "-h               Display this help message."
   exit 1
}

# Process command options
while getopts "i:h" opt
do
   case "$opt" in
      h ) help ;;
      * ) help ;;
   esac
done

alias runsn='sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_sn.sh'
alias runswop='sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_swop.sh'
alias runswci='sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_swci.sh'
alias runsl='sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/run_sl.sh'
alias cklog='sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/check_latest_log.sh'
alias stophpe='sudo sh /opt/hpe/swarm-learning-hpe/workspace/automate_scripts/launch_sl/stopswarm.sh'

# if bash then run source ~/.bashrc
. ~/.bashrc
# if zsh then run source ~/.zshrc
#source ~/.zshrc


# If an error occurs, print an error message and exit
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script. Please check the output above for more details."
    exit 1
fi
echo "alias generated successfully."
