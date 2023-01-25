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
while getopts "e:i:h?" opt
do
   case "$opt" in
      e ) workspace="$OPTARG" ;;
      i ) host="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done

sudo docker volume rm sl-cli-lib
sudo docker volume create sl-cli-lib
sudo docker container create --name helper -v sl-cli-lib:/data hello-world
sudo docker cp -L ./swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data
sudo docker rm helper

sudo docker network create host-"$ip_addr"-net