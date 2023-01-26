#!/bin/sh
set -eux

ip_addr=$(hostname -I | awk '{print $1}')

# Help function
help()
{
   echo ""
   echo "Ask jeff how to use the damn script"
   echo ""
   exit 1
}

# Process command options
while getopts "w:h?" opt
do
   case "$opt" in
      w ) workspace="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Checks
if [ -z "$workspace" ]
then
   echo "Some or all of the parameters are empty";
   help
fi


cp -r workspace/swarm_learning_scripts/gen-cert workspace/"$workspace"/
sudo ./workspace/"$workspace"/gen-cert -e "$workspace" -i "$ip_addr"