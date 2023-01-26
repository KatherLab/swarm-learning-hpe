#!/bin/sh
set -eux

ip_addr=$(hostname -I | awk '{print $1}')
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Help function
help()
{
   echo ""
   echo "Ask jeff how to use the damn script"
   echo ""
   exit 1
}

# Process command options
while getopts "w:i:h?" opt
do
   case "$opt" in
      w ) workspace="$OPTARG" ;;
      i ) host="$OPTARG" ;;
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