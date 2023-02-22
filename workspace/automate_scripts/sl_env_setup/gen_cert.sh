#!/bin/sh
set -eux

ip_addr=$(ip addr show | awk '/inet 10\./{print $2}' | cut -d'/' -f1)

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

#get current directory
script_dir=$(realpath $(dirname "${0}"))
echo $script_dir
sudo $script_dir/../../swarm_learning_scripts/gen-cert -e "$workspace" -i "$ip_addr"