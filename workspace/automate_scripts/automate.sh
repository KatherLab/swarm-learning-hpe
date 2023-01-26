#!/bin/sh
set -eux

ip_addr=$(hostname -I | awk '{print $1}')
script_name=$(basename "${0}")
script_dir=$(realpath $(dirname "${0}"))
cd "$script_dir"/..

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
while getopts "w:i:s:n:e:h?" opt
do
   case "$opt" in
      w ) workspace_name="$OPTARG" ;;
      i ) host_ip="$OPTARG" ;;
      s ) sentinal_ip="$OPTARG" ;;
      n ) num_peers="$OPTARG" ;;
      e ) num_epochs="$OPTARG" ;;
      h ) help ;;
      ? ) help ;;
   esac
done


if [[ $* == *--server_setup* ]]; then
  sh ./automate_scripts/server_setup/gpu_env_setup.sh
  sh ./automate_scripts/server_setup/test_open_exposed_ports.sh
  sh ./automate_scripts/server_setup/prerequisites.sh
  sh ./automate_scripts/server_setup/server_setup.sh
  sh ./automate_scripts/server_setup/install_containers.sh
fi

if [[ $* == *--sl_env_setup* ]]; then

  # Checks
  if [ -z "$host_ip" ] || [ -z "$workspace_name" ]
    then
       echo "Some or all of the parameters are empty";
       help
    fi
  sh ./automate_scripts/sl_env_setup/gen_cert.sh -w "$workspace_name"

  # Checks
  if [ $ip_addr = $sentinal_ip ]
  then
     echo "This host a sentinal node and will be used for initiating the cluster"
     sn_command="--sentinel"
     for value in $host_ip
        do
         echo sharing certificate with ip address $value
         sh ./automate_scripts/sl_env_setup/share_cert.sh -t "$value" -w "$workspace_name"
        done
  else
     echo "This host is not a sentinal node and will not be used for initiating the cluster, only as swarm network node"
     sn_command="--sentinel-ip=$sentinal_ip"
     sh ./automate_scripts/sl_env_setup/share_cert.sh -t "$sentinal_ip" -w "$workspace_name"

  fi
  sh ./automate_scripts/sl_env_setup/replacement.sh -w "$workspace_name" -s "$sentinal_ip" -n "$num_peers" -e "$num_epochs"
  sh ./automate_scripts/sl_env_setup/license_server_fix.sh
fi
