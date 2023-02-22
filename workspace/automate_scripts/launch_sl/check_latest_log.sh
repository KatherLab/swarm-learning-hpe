#!/bin/sh
set -eux

ip_addr=$(ip addr show | awk '/inet 10\./{print $2}' | cut -d'/' -f1)
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

# TODO: incompleted
sudo docker ps -a # If the container has already been stopped
sudo docker logs <container> # list logs of ml node container