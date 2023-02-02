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

sudo $script_dir/../../swarm_learning_scripts/run-swop -it --rm --name=swop"$ip_addr" \
--network=host-"$ip_addr"-net --usr-dir=workspace/"$workspace"/swop \
--profile-file-name=swop_profile_"$ip_addr".yaml \
--key=workspace/"$workspace"/cert/swop-"$ip_addr"-key.pem \
--cert=workspace/"$workspace"/cert/swop-"$ip_addr"-cert.pem \
--capath=workspace/"$workspace"/cert/ca/capath \
-e http_proxy= -e https_proxy= --apls-ip="$sentinal" --apls-port 5000 -e SWOP_KEEP_CONTAINERS=True