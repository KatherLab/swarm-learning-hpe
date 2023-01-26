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
while getopts "w:i:s:h?" opt
do
   case "$opt" in
      w ) workspace="$OPTARG" ;;
      i ) host="$OPTARG" ;;
      s ) sentinal="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

if [ $ip_addr = $sentinal ]
then
   echo "This host a sentinal node and will be used for initiating the cluster"
   sudo ../swarm_learning_scripts/run-swci -it --rm --name=swci"$ip_addr" \
  --network=host-"$ip_addr"-net --usr-dir=workspace/"$workspace"/swci \
  --init-script-name=swci-init --key=workspace/"$workspace"/cert/swci-"$ip_addr"-key.pem \
  --cert=workspace/"$workspace"/cert/swci-"$ip_addr"-cert.pem \
  --capath=workspace/"$workspace"/cert/ca/capath \
-e http_proxy= -e https_proxy= --apls-ip="$sentinal" --apls-port 5000 --apls-ip="$sentinal" --apls-port 5000 -e SWOP_KEEP_CONTAINERS=True
else
   echo "This host is not a sentinal node and will not be used for initiating the cluster, only as swarm network node"
   exit 1
fi
