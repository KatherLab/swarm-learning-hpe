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

# Checks
if [ $ip_addr = $sentinal ]
then
   echo "This host a sentinal node and will be used for initiating the cluster"
   sn_command="--sentinel"
else
   echo "This host is not a sentinal node and will not be used for initiating the cluster, only as swarm network node"
   sn_command="--sentinel-ip=$sentinal"
fi

sudo ../swarm_learning_scripts/run-sn -it --rm --name=sn"$ip_addr" --network=host-"$ip_addr"-net --host-ip="$ip_addr" --sentinel --sn-p2p-port=30303 --sn-api-port=30304 --key=workspace/"$workspace"/cert/sn-"$ip_addr"-key.pem --cert=workspace/"$workspace"/cert/sn-"$ip_addr"-cert.pem --capath=workspace/"$workspace"/cert/ca/capath --apls-ip="$ip_addr" --apls-port 5000

#sudo ./scripts/bin/run-sn -it --rm --name=sn1 --network=host-1-net --host-ip=192.168.33.102 "$sn_command" --sn-p2p-port=30303 --sn-api-port=30304 --key=workspace/katherlab/cert/sn-1-key.pem --cert=workspace/katherlab/cert/sn-1-cert.pem --capath=workspace/katherlab/cert/ca/capath --apls-ip=192.168.33.102 --apls-port 5000
#sudo ./scripts/bin/run-sn -it --rm --name=sn2 --network=host-2-net --host-ip=192.168.33.103 "$sn_command" --sn-p2p-port=30303 --sn-api-port=30304 --key=workspace/katherlab/cert/sn-2-key.pem --cert=workspace/katherlab/cert/sn-2-cert.pem --capath=workspace/katherlab/cert/ca/capath --apls-ip=192.168.33.102 --apls-port 5000