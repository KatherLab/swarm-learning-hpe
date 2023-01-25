#!/bin/sh
set -eux

ip_addr=$(hostname -I | awk '{print $1}')
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))

# Process command options
while getopts "e:i:" opt
do
   case "$opt" in
      e ) workspace="$OPTARG" ;;
      i ) host="$OPTARG" ;;
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