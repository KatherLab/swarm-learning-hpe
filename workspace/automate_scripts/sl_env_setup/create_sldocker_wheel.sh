#!/bin/sh
set -eux
script_dir=$(realpath $(dirname "${0}"))
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
while getopts "h?" opt
do
   case "$opt" in
      h ) help ;;
      ? ) help ;;
   esac
done

#if sudo docker volume list | grep -q 'sl-cli-lib'; then sudo docker volume rm sl-cli-lib; fi
sudo docker volume create sl-cli-lib
sudo docker container create --name helper -v sl-cli-lib:/data hello-world
sudo docker cp -L $script_dir/swarmlearning-client-py3-none-manylinux_2_24_x86_64.whl helper:/data
sudo docker rm helper

#if sudo docker network list | grep -q host-"$ip_addr"-net; then sudo docker network rm host-"$ip_addr"-net; fi
sudo docker network create host-"$ip_addr"-net