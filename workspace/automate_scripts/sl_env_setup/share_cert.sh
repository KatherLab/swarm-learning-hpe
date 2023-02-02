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
while getopts "t:w:h?" opt
do
   case "$opt" in
      t ) target_host="$OPTARG" ;;
      w ) workspace="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

# Checks
if [ -z "$target_host" ] || [ -z "$workspace" ]
then
   echo "Some or all of the parameters are empty";
   help
fi
sudo scp swarm@"$target_host":/opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/ca/capath/ca-"$target_host"-cert.pem /opt/hpe/swarm-learning-hpe/workspace/"$workspace"/cert/ca/capath